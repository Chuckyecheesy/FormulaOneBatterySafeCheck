"""Agent layer tests (spec/06-test-cases.md, AG-1 to AG-3, plus graph routing)."""

import logging

import pytest

from formulatech import agents
from formulatech.agents import build_race_graph, run_race_assessment, template_message
from formulatech.config import load_thresholds

OVERCHARGED = {"voltage_v": 5.0, "current_a": -1.2, "temperature_c": 25.0, "duration_min": 60}  # S1-1
HOT = {"voltage_v": 3.9, "current_a": 0.5, "temperature_c": 30.0, "duration_min": 1}  # S2-4
CLEAN = {"voltage_v": 3.9, "current_a": 0.5, "temperature_c": 25.0, "duration_min": 60}


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
    """Model gates passed; the prediction is set per test."""
    status = {"version": "test", "gates_passed": True, "tolerance_accuracy": 95.0, "r2": 0.95,
              "r2_train": 0.99, "cv_mean": 0.95, "cv_std": 0.01, "fit_status": "good"}
    monkeypatch.setattr(agents, "get_model_status", lambda: status)

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


def test_summary_states_failed_check_reasons():
    state = _run(HOT, FakeChatModel())

    assert "dT/dt: 0.08333 °C/s" in state["summary"]
    assert "Hazard threshold: > 0.03 °C/s." in state["summary"]
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


def test_s3_4_model_not_validated(monkeypatch):
    monkeypatch.setattr(agents, "get_model_status", lambda: {
        "version": "test", "gates_passed": False, "tolerance_accuracy": 24.0, "r2": 0.76,
        "r2_train": 0.84, "fit_status": "good"})
    state = _run(CLEAN, FakeChatModel())

    assert state["verdict"] == "DO_NOT_PROCEED"
    (failure,) = state["failures"]
    assert failure["code"] == "MODEL_NOT_VALIDATED"
    assert failure["recorded"]["r2"] == 0.76


def test_prediction_error_fails_closed(validated_model, monkeypatch):
    def boom(*a):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(agents, "predict_efficiency", boom)
    state = _run(CLEAN, FakeChatModel())

    assert state["verdict"] == "DO_NOT_PROCEED"
    assert state["failures"][0]["code"] == "MODEL_NOT_VALIDATED"
