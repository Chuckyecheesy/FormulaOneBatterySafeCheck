"""HTTP backend tests for the frontend (spec/01-requirements.md §1.1, spec/05-ui.md)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from formulatech import agents
from formulatech.agents import build_race_graph
from formulatech.api import create_app, validate_inputs
from formulatech.config import load_thresholds

# Backend response for a 65 % prediction, also served to the browser by the Playwright test.
LOW_EFFICIENCY_FIXTURE = Path(__file__).resolve().parents[1] / "e2e" / "fixtures" / "low-efficiency-response.json"
UI_FIELDS = ["verdict", "stages", "failures", "predicted_efficiency_pct", "comment", "t_seconds"]

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
    monkeypatch.setattr(agents, "get_model_version", lambda: "test")

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
    ("voltage_v", "inf", "Battery voltage must be a finite number."),
    ("current_a", float("-inf"), "Battery current must be a finite number."),
    ("soc_percent", True, "State of charge is required."),
    ("voltage_v", None, "Battery voltage is required."),
    ("voltage_v", [1], "Battery voltage must be a number."),
    ("voltage_v", 10**400, "Battery voltage must be a finite number."),  # int too big for a float
    # just outside each limit
    ("voltage_v", 10.0001, "Battery voltage must be between 0 and 10."),
    ("voltage_v", -0.0001, "Battery voltage must be between 0 and 10."),
    ("current_a", 500.01, "Battery current must be between -500 and 500."),
    ("current_a", -500.01, "Battery current must be between -500 and 500."),
    ("temperature_c", -40.01, "Battery temperature must be between -40 and 100."),
    ("duration_min", 1440.01, "Charging duration must be between 0 and 1440."),
    ("soc_percent", -0.001, "State of charge must be between 0 and 100."),
])
def test_validation_errors(field, value, message):
    inputs, errors = validate_inputs({**CLEAN, field: value}, load_thresholds())
    assert errors == {field: message}
    assert field not in inputs


@pytest.mark.parametrize("field, value", [
    ("voltage_v", 0), ("voltage_v", 10),
    ("current_a", -500), ("current_a", 500),
    ("temperature_c", -40), ("temperature_c", 100),
    ("duration_min", 1e-9), ("duration_min", 1440),
    ("soc_percent", 0), ("soc_percent", 100),
    ("voltage_v", "4.1"),  # numeric string
])
def test_values_at_limits_accepted(field, value):
    inputs, errors = validate_inputs({**CLEAN, field: value}, load_thresholds())
    assert errors == {}
    assert inputs[field] == float(value)


def test_missing_field_is_required():
    raw = {k: v for k, v in CLEAN.items() if k != "soc_percent"}
    assert validate_inputs(raw, load_thresholds())[1] == {"soc_percent": "State of charge is required."}


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


def test_low_efficiency_does_not_proceed(client, validated_model):
    validated_model(65.0)
    response = client.post("/api/check", json=CLEAN)
    body = response.json()

    assert response.status_code == 200
    assert body["verdict"] == "DO_NOT_PROCEED"
    assert [s["status"] for s in body["stages"]] == ["passed", "passed", "failed"]
    assert body["predicted_efficiency_pct"] == 65.0
    (card,) = body["failures"]
    assert card["stage"] == 3 and card["stage_name"] == "Prediction"
    assert card["reason"] == "Predicted efficiency is too low, so the car cannot proceed into the race"
    assert card["recorded_kind"] == "calculated"


def test_low_efficiency_matches_e2e_fixture(client, validated_model):
    """The Playwright test serves this fixture to the page, so it must match the real backend."""
    validated_model(65.0)
    body = client.post("/api/check", json=CLEAN).json()
    assert {k: body[k] for k in UI_FIELDS} == json.loads(LOW_EFFICIENCY_FIXTURE.read_text())


def test_prediction_unavailable_returns_error_not_verdict(client, monkeypatch):
    def boom(*a):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(agents, "predict_efficiency", boom)
    response = client.post("/api/check", json=CLEAN)

    assert response.status_code == 503
    assert response.json() == {"error": "The efficiency model could not produce a prediction"}


def test_frontend_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Race Readiness Check" in response.text


@pytest.mark.parametrize("minutes", [1e-200, 5e-324])  # t² underflows to 0 for these
def test_tiny_duration_does_not_crash(client, minutes):
    hot = client.post("/api/check", json={**CLEAN, "duration_min": minutes, "temperature_c": 30.0})
    assert hot.status_code == 200
    assert hot.json()["verdict"] == "DO_NOT_PROCEED"
    assert [s["status"] for s in hot.json()["stages"]] == ["passed", "failed", "skipped"]
    at_baseline = client.post("/api/check", json={**CLEAN, "duration_min": minutes, "temperature_c": 25.0})
    assert at_baseline.status_code == 200
    assert at_baseline.json()["stages"][1]["status"] == "passed"  # no temperature rise, so no thermal failure


def test_huge_integer_is_a_field_error(client):
    response = client.post("/api/check", content='{"voltage_v": 1' + "0" * 400 + ', "current_a": 0.5, '
                           '"temperature_c": 25, "duration_min": 60, "soc_percent": 50}',
                           headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json() == {"errors": {"voltage_v": "Battery voltage must be a finite number."}}


@pytest.mark.parametrize("minutes", [1e-200, 5e-324])
def test_tiny_duration_never_renders_raw_inf(client, minutes):
    """dT/dt overflows to inf for tiny t; the card must word it, not print the float (see _rate_unit)."""
    body = client.post("/api/check", json={**CLEAN, "duration_min": minutes, "temperature_c": 30.0}).json()
    recorded = [line for card in body["failures"] for line in card["recorded"]]
    # d²T/dt² overflows for both durations; dT/dt only for the smaller one, so assert the invariant.
    assert "d²T/dt² = above the measurable range" in recorded
    assert not any("inf" in line or "nan" in line for line in recorded)
    assert "inf" not in body["summary"] and "nan" not in body["summary"]
