"""FR-7 audit log: one strict-JSON line per check."""

import json
import math

import pytest

from formulatech.agents import build_race_graph, run_race_assessment
from formulatech.audit import write_audit_record
from formulatech.config import load_thresholds


def _strict_load(line):
    """json.loads that rejects the non-standard Infinity / NaN tokens, like strict parsers do."""
    def reject(token):
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(line, parse_constant=reject)


def test_non_finite_floats_are_written_as_strict_json(audit_log):
    write_audit_record({"calculated": {"dT_dt": math.inf, "neg": -math.inf, "nan": math.nan, "ok": 1.5},
                        "nested": [[math.inf], (math.nan,)]})
    record = _strict_load(audit_log.read_text())
    assert record == {"calculated": {"dT_dt": "Infinity", "neg": "-Infinity", "nan": "NaN", "ok": 1.5},
                      "nested": [["Infinity"], ["NaN"]]}


@pytest.mark.parametrize("minutes", [5e-324, 1e-160])  # dT/dt or d²T/dt² overflow to inf
def test_extreme_reading_audit_line_is_strict_json(audit_log, minutes):
    inputs = {"voltage_v": 3.9, "current_a": 0.5, "temperature_c": 30.0, "duration_min": minutes, "soc_percent": 50.0}
    state = run_race_assessment(inputs, graph=build_race_graph(model_factory=lambda: None))

    (line,) = audit_log.read_text().splitlines()
    record = _strict_load(line)
    assert record["verdict"] == state["verdict"] == "DO_NOT_PROCEED"
    assert "Infinity" in record["calculated"].values()
    assert record["calculated"]["T_chem"] == 5.0


CLEAN = {"voltage_v": 3.9, "current_a": 0.5, "temperature_c": 25.0, "duration_min": 60, "soc_percent": 50.0}


def _records(audit_log):
    return [_strict_load(line) for line in audit_log.read_text().splitlines()]


def test_every_check_is_audited_with_fr7_fields(audit_log, monkeypatch):
    from datetime import datetime

    from formulatech import agents

    monkeypatch.setattr(agents, "predict_efficiency", lambda *a: 98.0)
    monkeypatch.setattr(agents, "get_model_version", lambda: "v-test")
    graph = build_race_graph(model_factory=lambda: None)
    run_race_assessment(CLEAN, graph=graph)
    run_race_assessment({**CLEAN, "voltage_v": 5.0, "current_a": -1.2}, graph=graph)

    clean, overcharged = _records(audit_log)
    assert datetime.fromisoformat(clean["timestamp"]).utcoffset().total_seconds() == 0
    assert clean["check_id"] != overcharged["check_id"]
    assert clean["outcome"] == "verdict" and clean["error"] is None
    assert clean["inputs"] == CLEAN and clean["t_seconds"] == 3600
    assert clean["calculated"] == {"dT_dt": 0.0, "d2T_dt2": 0.0, "T_chem": 0.0, "efficiency_pct": 98.0}
    assert [(r["stage"], r["code"], r["passed"]) for r in clean["stage_results"]] == [
        (1, "OVERCHARGED", True), (2, "HIGH_DTDT", True), (2, "HIGH_D2TDT2", True), (2, "HIGH_TCHEM", True),
        (3, "LOW_EFFICIENCY", True)]
    dtdt_max = load_thresholds()["thermal"]["dT_dt_max_c_per_s"]
    assert clean["stage_results"][1]["thresholds"] == {"dT_dt": f"> {dtdt_max:g}"}
    assert clean["model_version"] == "v-test" and clean["verdict"] == "CAN_PROCEED"

    assert overcharged["verdict"] == "DO_NOT_PROCEED" and overcharged["failed_codes"] == ["OVERCHARGED"]
    assert [r["stage"] for r in overcharged["stage_results"]] == [1]


def test_check_without_verdict_is_still_audited(audit_log, monkeypatch):
    from formulatech import agents

    def boom(*a):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(agents, "predict_efficiency", boom)
    with pytest.raises(agents.PredictionUnavailable):
        run_race_assessment(CLEAN, graph=build_race_graph(model_factory=lambda: None))

    (record,) = _records(audit_log)
    assert record["outcome"] == "error" and record["verdict"] is None
    assert record["error"] == "PredictionUnavailable: The efficiency model could not produce a prediction"
    assert [r["stage"] for r in record["stage_results"]] == [1, 2, 2, 2]  # stages 1-2 up to the error


def test_audit_write_failure_does_not_block_the_verdict(monkeypatch, tmp_path, caplog):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setenv("FORMULATECH_AUDIT_LOG", str(blocker / "audit.jsonl"))
    state = run_race_assessment({**CLEAN, "voltage_v": 5.0, "current_a": -1.2},
                                graph=build_race_graph(model_factory=lambda: None))

    assert state["verdict"] == "DO_NOT_PROCEED"
    assert "Could not write the audit record" in caplog.text


def test_api_check_writes_one_audit_line(audit_log):
    from fastapi.testclient import TestClient

    from formulatech.api import create_app

    client = TestClient(create_app(graph=build_race_graph(model_factory=lambda: None)))
    client.post("/api/check", json={**CLEAN, "duration_min": 0})  # invalid input: no check runs
    client.post("/api/check", json={**CLEAN, "voltage_v": 5.0, "current_a": -1.2})

    (record,) = _records(audit_log)
    assert record["failed_codes"] == ["OVERCHARGED"]
