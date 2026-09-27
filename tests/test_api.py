"""HTTP backend tests for the frontend (spec/01-requirements.md §1.1, spec/05-ui.md)."""

import pytest
from fastapi.testclient import TestClient

from formulatech import agents
from formulatech.agents import build_race_graph
from formulatech.api import create_app, validate_inputs
from formulatech.config import load_thresholds

CLEAN = {"voltage_v": 3.9, "current_a": 0.5, "temperature_c": 25.0, "duration_min": 60, "soc_percent": 50.0}


class FakeChatModel:
    def invoke(self, prompt):
        class Response:
            content = "explanation"

        return Response()


@pytest.fixture
def client():
    return TestClient(create_app(graph=build_race_graph(model_factory=FakeChatModel)))


@pytest.fixture
def validated_model(monkeypatch):
    status = {"version": "test", "gates_passed": True, "tolerance_accuracy": 95.0, "r2": 0.95,
              "r2_train": 0.99, "cv_mean": 0.95, "cv_std": 0.01}
    monkeypatch.setattr(agents, "get_model_status", lambda: status)

    def set_prediction(value):
        monkeypatch.setattr(agents, "predict_efficiency", lambda *a: value)

    return set_prediction


@pytest.mark.parametrize("field, value, message", [
    ("voltage_v", "", "Battery voltage is required."),
    ("current_a", "abc", "Battery current must be a number."),
    ("temperature_c", "nan", "Battery temperature must be a finite number."),
    ("duration_min", 0, "Charging duration must be greater than 0 minutes."),
    ("duration_min", -5, "Charging duration must be greater than 0 minutes."),
    ("soc_percent", 101, "State of charge must be between 0 and 100."),
    ("temperature_c", 150, "Battery temperature must be between -40 and 100."),
])
def test_validation_errors(field, value, message):
    inputs, errors = validate_inputs({**CLEAN, field: value}, load_thresholds())
    assert errors == {field: message}
    assert field not in inputs


def test_negative_current_allowed():
    inputs, errors = validate_inputs({**CLEAN, "current_a": -1.2}, load_thresholds())
    assert errors == {} and inputs["current_a"] == -1.2


def test_invalid_input_gives_no_verdict(client):
    response = client.post("/api/check", json={**CLEAN, "duration_min": 0})
    assert response.status_code == 422
    body = response.json()
    assert "verdict" not in body
    assert body["errors"]["duration_min"] == "Charging duration must be greater than 0 minutes."


def test_config_serves_input_limits(client):
    body = client.get("/api/config").json()
    assert body["input_limits"] == load_thresholds()["input_limits"]
    assert body["seconds_per_minute"] == 60


def test_overcharged_card(client):
    body = client.post("/api/check", json={**CLEAN, "voltage_v": 5.0, "current_a": -1.2}).json()

    assert body["verdict"] == "DO_NOT_PROCEED"
    assert [s["status"] for s in body["stages"]] == ["failed", "skipped", "skipped"]
    (card,) = body["failures"]
    assert card["stage_name"] == "Overcharge"
    assert card["reason"] == "Battery already overcharged"
    assert card["recorded"] == ["battery voltage 5.00 V", "battery current -1.20 A"]
    assert card["thresholds"] == ["battery voltage > 4.2 V AND battery current < 0 A"]
    assert "R1" not in str(card) and "OVERCHARGED" not in str(card)  # internal IDs are not shown
    assert body["comment"] == agents.COMMENT_FIRE_RISK_NOW


def test_single_thermal_failure_stops(client, monkeypatch):
    """Issue #3: one thermal failure is enough to stop; the prediction must not run."""
    monkeypatch.setattr(agents, "evaluate_prediction", lambda *a: pytest.fail("stage 3 must not run"))
    body = client.post("/api/check", json={**CLEAN, "temperature_c": 25.31}).json()

    assert body["verdict"] == "DO_NOT_PROCEED"
    assert [s["status"] for s in body["stages"]] == ["passed", "failed", "skipped"]
    (card,) = body["failures"]
    assert card["recorded"] == ["T_chem = 0.31 °C"]
    assert card["thresholds"] == ["T_chem > 0.3 °C"]


def test_every_thermal_failure_listed(client):
    body = client.post("/api/check", json={**CLEAN, "temperature_c": 30.0, "duration_min": 1}).json()

    assert [c["recorded"][0] for c in body["failures"]] == [
        "dT/dt = 0.08333 °C/s", "d²T/dt² = 0.001389 °C/s²", "T_chem = 5.00 °C"]
    assert {c["stage"] for c in body["failures"]} == {2}


def test_can_proceed(client, validated_model):
    validated_model(98.2)
    body = client.post("/api/check", json=CLEAN).json()

    assert body["verdict"] == "CAN_PROCEED"
    assert body["failures"] == []
    assert body["predicted_efficiency_pct"] == 98.2
    assert body["comment"] == ""
    assert [s["status"] for s in body["stages"]] == ["passed", "passed", "passed"]


def test_low_efficiency_card(client, validated_model):
    validated_model(65.0)
    body = client.post("/api/check", json=CLEAN).json()

    (card,) = body["failures"]
    assert card["threshold_label"] == "Required to pass"
    assert card["recorded"] == ["predicted efficiency = 65.00 % (XGBoost model vtest)"]
    assert card["thresholds"] == ["predicted efficiency ≥ 70 %"]
    assert body["comment"] == agents.COMMENT_FIRE_RISK_IN_RACE


def test_frontend_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Race Readiness Check" in response.text
