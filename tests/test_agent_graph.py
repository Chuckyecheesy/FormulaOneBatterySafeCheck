"""Agent layer tests (spec/06-test-cases.md, AG-1 to AG-3, plus graph routing)."""

import logging

import pytest

from formulatech import agents
from formulatech.agents import build_race_graph, explain, run_race_assessment, template_message
from formulatech.config import load_thresholds
from formulatech.ml.train import FEATURES

OVERCHARGED = {"voltage_v": 5.0, "current_a": -1.2, "temperature_c": 25.0, "duration_min": 60, "soc_percent": 50.0}  # S1-1
HOT = {"voltage_v": 3.9, "current_a": 0.5, "temperature_c": 30.0, "duration_min": 1, "soc_percent": 50.0}  # S2-4
CLEAN = {"voltage_v": 3.9, "current_a": 0.5, "temperature_c": 25.0, "duration_min": 60, "soc_percent": 50.0}


class FakeChatModel:
    def __init__(self, text="safe explanation", exc=None):
        self.text, self.exc, self.calls = text, exc, 0

    def invoke(self, prompt):
        self.calls += 1
        if self.exc:
            raise self.exc

        class Response:
            content = self.text

        return Response()


def _run(inputs, llm):
    return run_race_assessment(inputs, graph=build_race_graph(model_factory=lambda: llm))


@pytest.fixture
def validated_model(monkeypatch):
    """The deployed model (gates checked at training time); the prediction is set per test."""
    monkeypatch.setattr(agents, "get_model_version", lambda: "test")

    def set_prediction(value):
        monkeypatch.setattr(agents, "predict_efficiency", lambda *a: value)

    return set_prediction


def test_overcharge_short_circuit():
    llm = FakeChatModel()
    state = _run(OVERCHARGED, llm)

    assert state["verdict"] == "DO_NOT_PROCEED"
    assert state["failures"][0]["code"] == "OVERCHARGED"
    assert [r["stage"] for r in state["stage_results"]] == [1]
    assert llm.calls == 1


def test_thermal_lists_every_failure_and_skips_prediction():
    llm = FakeChatModel()
    state = _run(HOT, llm)

    assert state["verdict"] == "DO_NOT_PROCEED"
    assert [f["code"] for f in state["failures"]] == ["HIGH_DTDT", "HIGH_D2TDT2", "HIGH_TCHEM"]
    assert {r["stage"] for r in state["stage_results"]} == {1, 2}
    assert llm.calls == 3  # R1 passed, so it is not explained


def test_ag1_llm_cannot_override_verdict(caplog):
    with caplog.at_level(logging.WARNING, logger="formulatech.agents"):
        state = _run(OVERCHARGED, FakeChatModel(text="PROCEED. The battery is safe to proceed."))

    assert state["verdict"] == "DO_NOT_PROCEED"
    assert state["summary"].startswith("DO NOT PROCEED")
    assert "contradicts computed verdict" in caplog.text


@pytest.mark.parametrize("llm_factory", [
    lambda: FakeChatModel(exc=TimeoutError("ollama timed out")),
    lambda: (_ for _ in ()).throw(RuntimeError("ollama offline")),
])
def test_ag2_fallback_uses_template_messages(llm_factory):
    state = run_race_assessment(OVERCHARGED, graph=build_race_graph(model_factory=llm_factory))

    expected = template_message(state["failures"][0], OVERCHARGED, load_thresholds())
    assert state["verdict"] == "DO_NOT_PROCEED"
    assert state["explanations"] == [expected]


def test_ag3_every_failure_has_recorded_values_and_thresholds():
    state = _run(HOT, FakeChatModel())

    for failure in state["failures"]:
        assert failure["recorded"] and failure["thresholds"]
        assert failure["recorded"].keys() == failure["thresholds"].keys()


@pytest.mark.parametrize("t0", [None, 20.0])  # None: the value in thresholds.yaml
def test_prompt_explains_tchem_as_rise_above_baseline(t0):
    seen = {}

    class CapturingModel:
        def invoke(self, prompt):
            seen["prompt"] = prompt
            class Response:
                content = "ok"
            return Response()

    result = {
        "rule": "R4",
        "code": "HIGH_TCHEM",
        "passed": False,
        "recorded": {"T_chem": 1.0},
        "recorded_kind": "calculated",
        "thresholds": {"T_chem": "> 0.3"},
        "reason": "Heat exposure to the surrounding environment is too high",
    }
    cfg = load_thresholds()
    if t0 is not None:
        cfg = {**cfg, "thermal": {**cfg["thermal"], "initial_temp_c": t0}}
    baseline = f'{cfg["thermal"]["initial_temp_c"]:g}'
    fallback = template_message(result, {"temperature_c": 26.0, "duration_min": 60}, cfg)

    explain(CapturingModel(), result, fallback, cfg)

    assert f"temperature rise above the {baseline}°C baseline" in seen["prompt"]
    assert f"T_t - {baseline}°C" in seen["prompt"]
    if t0 is not None:  # the baseline comes from cfg, not a literal
        assert "25°C" not in seen["prompt"]


def test_summary_states_failed_check_reasons():
    state = _run(HOT, FakeChatModel())

    assert "dT/dt: 0.08333 °C/s" in state["summary"]
    assert "Fire hazard threshold: > 0.03 °C/s." in state["summary"]
    assert "max(0" not in state["summary"] and "calculated from" not in state["summary"]
    assert "T_chem: 5.00 °C" in state["summary"]


def test_s3_1_can_proceed(validated_model):
    validated_model(98.2)
    llm = FakeChatModel()
    state = _run(CLEAN, llm)

    assert state["verdict"] == "CAN_PROCEED"
    assert state["failures"] == []
    assert llm.calls == 0  # nothing failed, nothing to explain


def test_s3_2_low_efficiency(validated_model):
    validated_model(65.0)
    state = _run(CLEAN, FakeChatModel())

    assert state["verdict"] == "DO_NOT_PROCEED"
    (failure,) = state["failures"]
    assert failure["code"] == "LOW_EFFICIENCY"
    assert failure["recorded"] == {"efficiency_pct": 65.0}
    assert failure["thresholds"] == {"efficiency_pct": "< 70"}


def test_s3_3_exactly_at_threshold_passes(validated_model):
    validated_model(70.0)
    assert _run(CLEAN, FakeChatModel())["verdict"] == "CAN_PROCEED"


@pytest.mark.parametrize(
    "predicted, verdict",
    [
        (69.99, "DO_NOT_PROCEED"),
        (69.999999999, "DO_NOT_PROCEED"),  # below 70 after rounding to 9 decimals
        (69.9999999996, "CAN_PROCEED"),  # rounds to 70.0, which passes (strict <)
        (70.000000001, "CAN_PROCEED"),
        (0.0, "DO_NOT_PROCEED"),
    ],
)
def test_s3_efficiency_boundary(validated_model, predicted, verdict):
    validated_model(predicted)
    state = _run(CLEAN, FakeChatModel())

    assert state["verdict"] == verdict
    if verdict == "DO_NOT_PROCEED":
        (failure,) = state["failures"]
        assert failure["code"] == "LOW_EFFICIENCY"
        assert state["comment"] == agents.COMMENT_FIRE_RISK_IN_RACE


def test_s3_4_runtime_ignores_model_gates(monkeypatch, tmp_path):
    """Model gates are checked at training time, not at runtime: a failed-gate flag does not block."""
    meta = tmp_path / "meta.json"
    meta.write_text('{"version": "v1", "gates": {"gates_passed": false}}')
    monkeypatch.setattr(agents, "META_PATH", meta)
    monkeypatch.setattr(agents, "predict_efficiency", lambda *a: 98.0)
    state = _run(CLEAN, FakeChatModel())

    assert state["verdict"] == "CAN_PROCEED"
    assert state["stage_results"][-1]["model_version"] == "v1"


def test_prediction_error_gives_no_verdict(monkeypatch):
    def boom(*a):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(agents, "predict_efficiency", boom)
    with pytest.raises(agents.PredictionUnavailable):
        _run(CLEAN, FakeChatModel())


@pytest.mark.parametrize("prediction", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_prediction_gives_no_verdict(validated_model, prediction):
    validated_model(prediction)
    with pytest.raises(agents.PredictionUnavailable):
        _run(CLEAN, FakeChatModel())


@pytest.mark.parametrize("contents", [None, "{not json", "[]", '{"version": 7}'])
def test_unreadable_metadata_only_loses_the_version(monkeypatch, tmp_path, contents):
    meta = tmp_path / "meta.json"
    if contents is not None:
        meta.write_text(contents)
    monkeypatch.setattr(agents, "META_PATH", meta)
    monkeypatch.setattr(agents, "predict_efficiency", lambda *a: 65.0)
    state = _run(CLEAN, FakeChatModel())

    assert state["verdict"] == "DO_NOT_PROCEED"
    (failure,) = state["failures"]
    assert failure["code"] == "LOW_EFFICIENCY"
    assert failure["model_version"] is None


def test_predict_efficiency_passes_soc_to_model(monkeypatch, tmp_path):
    seen = {}

    class FakeModel:
        def load_model(self, path):
            pass

        def predict(self, X):
            seen["X"] = X
            return [98.0]

    model_file = tmp_path / "model.json"
    model_file.write_text("{}")
    monkeypatch.setattr(agents, "MODEL_PATH", model_file)
    monkeypatch.setattr(agents, "XGBRegressor", FakeModel)

    assert agents.predict_efficiency(3.9, 0.5, 25.0, 60, soc_percent=42.0) == 98.0
    X = seen["X"]
    assert list(X.columns) == FEATURES
    assert X.loc[0, "SOC (%)"] == 42.0
    assert not X.isna().any().any()


def test_missing_soc_gives_no_verdict(validated_model):
    validated_model(98.0)
    inputs = {k: v for k, v in CLEAN.items() if k != "soc_percent"}
    with pytest.raises(agents.PredictionUnavailable):
        _run(inputs, FakeChatModel())


@pytest.mark.parametrize("inputs", [OVERCHARGED, HOT])  # stage 1, stage 2
def test_comment_fire_risk_now(inputs):
    assert _run(inputs, FakeChatModel())["comment"] == agents.COMMENT_FIRE_RISK_NOW


def test_comment_fire_risk_during_race(validated_model):
    validated_model(65.0)
    assert _run(CLEAN, FakeChatModel())["comment"] == agents.COMMENT_FIRE_RISK_IN_RACE


def test_no_comment_on_can_proceed(validated_model):
    validated_model(98.0)
    state = _run(CLEAN, FakeChatModel())
    assert state["verdict"] == "CAN_PROCEED"
    assert state["comment"] == ""


def test_comment_wording():
    assert agents.COMMENT_FIRE_RISK_NOW == "Your battery is at risk of fire hazard if you start the race now."
    assert agents.COMMENT_FIRE_RISK_IN_RACE == "Your battery is at risk of fire hazard during the middle of the race."


# ---- @tool wrappers (make_check_tools) ----


@pytest.fixture
def check_tools():
    return agents.make_check_tools(load_thresholds())


def test_check_tools_names_and_args(check_tools):
    assert {name: list(t.args) for name, t in check_tools.items()} == {
        "check_overcharge_tool": ["voltage_v", "current_a"],
        "check_thermal_tool": ["temperature_c", "duration_min"],
        "check_efficiency_tool": ["voltage_v", "current_a", "temperature_c", "duration_min", "soc_percent"],
    }


def test_check_tools_match_deterministic_rules(check_tools, validated_model):
    from dataclasses import asdict

    from formulatech.rules import check_overcharge, check_thermal

    cfg = load_thresholds()
    validated_model(65.0)
    assert check_tools["check_overcharge_tool"].invoke({"voltage_v": 5.0, "current_a": -1.2}) == [
        asdict(check_overcharge(5.0, -1.2, cfg))]
    assert check_tools["check_thermal_tool"].invoke({"temperature_c": 30.0, "duration_min": 1}) == [
        asdict(r) for r in check_thermal(30.0, 1, cfg)]
    assert check_tools["check_efficiency_tool"].invoke(CLEAN) == [agents.evaluate_prediction(CLEAN, cfg)]


def test_check_tool_docstrings_have_no_threshold_literals(check_tools):
    cfg = load_thresholds()
    limits = [cfg["overcharge"]["voltage_max_v"], *cfg["thermal"].values(), cfg["efficiency"]["min_percent"]]
    for t in check_tools.values():
        for value in limits:
            if isinstance(value, (int, float)) and value not in (0, 25):  # 0/25 can appear as ordinary words/units
                assert f"{value:g}" not in t.description, (t.name, value)


def test_check_tools_use_the_given_cfg():
    cfg = load_thresholds()
    strict = {**cfg, "thermal": {**cfg["thermal"], "dT_dt_max_c_per_s": 0.0}}
    tools = agents.make_check_tools(strict)
    (dtdt, *_) = tools["check_thermal_tool"].invoke({"temperature_c": 25.5, "duration_min": 60})
    assert dtdt["code"] == "HIGH_DTDT" and not dtdt["passed"]
