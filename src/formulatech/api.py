"""HTTP backend for the Race Readiness Check frontend (spec/05-ui.md).

Run with:  uv run uvicorn formulatech.api:app --reload
then open http://127.0.0.1:8000/

The verdict comes from the LangGraph pipeline in `formulatech.agents`; this module only
validates input (spec/01-requirements.md §1.1) and shapes the result for the UI.
Every threshold shown on screen is read from spec/thresholds.yaml.
"""

from __future__ import annotations

import math
import os
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from formulatech.agents import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    _fmt,
    _rate,
    _temp,
    build_race_graph,
    run_race_assessment,
)
from formulatech.config import REPO_ROOT, load_thresholds
from formulatech.rules import SECONDS_PER_MINUTE

FRONTEND_DIR = REPO_ROOT / "frontend"

# Form field -> (input_limits key, label shown in errors).
FIELDS = {
    "voltage_v": ("voltage_v", "Battery voltage"),
    "current_a": ("current_a", "Battery current"),
    "temperature_c": ("temperature_c", "Battery temperature"),
    "duration_min": ("duration_min", "Charging duration"),
    "soc_percent": ("soc_percent", "State of charge"),
}

STAGE_NAMES = {1: "Overcharge", 2: "Thermal", 3: "Prediction"}


# ---------------------------------------------------------------------------
# Validation (spec/01-requirements.md §1.1)
# ---------------------------------------------------------------------------


def validate_inputs(raw: dict[str, Any], cfg: dict) -> tuple[dict[str, float], dict[str, str]]:
    """Return (inputs, errors). Invalid input never produces a verdict (IN-4)."""
    limits = cfg["input_limits"]
    inputs: dict[str, float] = {}
    errors: dict[str, str] = {}
    for name, (limit_key, label) in FIELDS.items():
        value = raw.get(name)
        # IN-1: required, finite number (bools are not numbers here)
        if value is None or value == "" or isinstance(value, bool):
            errors[name] = f"{label} is required."
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors[name] = f"{label} must be a number."
            continue
        if not math.isfinite(number):
            errors[name] = f"{label} must be a finite number."
            continue

        lim = limits[limit_key]
        if name == "duration_min" and number <= lim["min"]:  # IN-2
            errors[name] = "Charging duration must be greater than 0 minutes."
        elif number < lim["min"] or number > lim["max"]:  # IN-2a, IN-3
            errors[name] = f"{label} must be between {_fmt(lim['min'])} and {_fmt(lim['max'])}."
        else:
            inputs[name] = number
    return inputs, errors


# ---------------------------------------------------------------------------
# Result cards (spec/05-ui.md §3). Rule IDs and reason codes are not displayed.
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} %"


def failure_card(failure: dict[str, Any], stage: int, cfg: dict) -> dict[str, Any]:
    """One card per failed rule: reason, recorded value(s), threshold(s) and stage."""
    rec = failure["recorded"]
    th, oc, gates = cfg["thermal"], cfg["overcharge"], cfg["model_gates"]
    threshold_label = "Fire hazard threshold"
    match failure["code"]:
        case "OVERCHARGED":
            reason = "Battery already overcharged"
            recorded = [f"battery voltage {rec['V']:.2f} V", f"battery current {rec['I']:.2f} A"]
            thresholds = [
                f"battery voltage > {_fmt(oc['voltage_max_v'])} V AND "
                f"battery current < {_fmt(oc['current_max_a'])} A"
            ]
        case "HIGH_DTDT":
            reason = failure["reason"]
            recorded = [f"dT/dt = {_rate(rec['dT_dt'])} °C/s"]
            thresholds = [f"dT/dt > {_fmt(th['dT_dt_max_c_per_s'])} °C/s"]
        case "HIGH_D2TDT2":
            reason = failure["reason"]
            recorded = [f"d²T/dt² = {_rate(rec['d2T_dt2'])} °C/s²"]
            thresholds = [f"d²T/dt² > {_fmt(th['d2T_dt2_max_c_per_s2'])} °C/s²"]
        case "HIGH_TCHEM":
            reason = failure["reason"]
            recorded = [f"T_chem = {_temp(rec['T_chem'])} °C"]
            thresholds = [f"T_chem > {_fmt(th['t_chem_max_c'])} °C"]
        case "LOW_EFFICIENCY":
            reason = failure["reason"]
            recorded = [
                f"predicted efficiency = {_pct(rec['efficiency_pct'])} "
                f"(XGBoost model v{failure.get('model_version')})"
            ]
            threshold_label = "Required to pass"
            thresholds = [f"predicted efficiency ≥ {_fmt(cfg['efficiency']['min_percent'])} %"]
        case "MODEL_NOT_VALIDATED":
            reason = f"{failure['reason']}, so the race cannot be cleared"
            recorded = [
                f"tolerance accuracy = {_pct(rec.get('tolerance_accuracy'))}",
                f"R² = {_rate(rec.get('r2'))}",
                f"CV R² mean = {_rate(rec.get('cv_r2_mean'))}, std = {_rate(rec.get('cv_r2_std'))}",
            ]
            threshold_label = "Required to pass"
            thresholds = [
                f"tolerance accuracy > {_fmt(gates['tolerance_accuracy_min_percent'])} %",
                f"R² > {_fmt(gates['r2_min'])}",
                f"CV R² mean > {_fmt(gates['r2_min'])} and std ≤ {_fmt(gates['cv_r2_std_max'])}",
            ]
        case _:
            reason, recorded, thresholds = failure["reason"], [], []
    return {
        "stage": stage,
        "stage_name": STAGE_NAMES.get(stage, ""),
        "reason": reason,
        "recorded_kind": failure["recorded_kind"],
        "recorded": recorded,
        "threshold_label": threshold_label,
        "thresholds": thresholds,
    }


def stage_progress(stage_results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Status of each stage: passed, failed, or skipped (a stage after a failure never runs)."""
    progress = []
    for stage, name in STAGE_NAMES.items():
        ran = [r for r in stage_results if r["stage"] == stage]
        status = "skipped" if not ran else ("passed" if all(r["passed"] for r in ran) else "failed")
        progress.append({"stage": stage, "name": name, "status": status})
    return progress


def build_response(state: dict[str, Any], cfg: dict) -> dict[str, Any]:
    """Shape the graph state for the UI.

    The UI assumes the model has already been validated, so the internal fail-closed
    MODEL_NOT_VALIDATED record is not surfaced as a normal user-facing card.
    """
    results = state["stage_results"]
    failed = [r for r in results if not r["passed"] and r["code"] != "MODEL_NOT_VALIDATED"]
    efficiency = next((r["recorded"]["efficiency_pct"] for r in results if r["code"] == "LOW_EFFICIENCY"), None)
    return {
        "verdict": state["verdict"],
        "stages": stage_progress(results),
        "failures": [failure_card(r, r["stage"], cfg) for r in failed],
        "predicted_efficiency_pct": efficiency,
        "comment": state["comment"],
        "summary": state["summary"],
        "explanations": state["explanations"],
        "t_seconds": state["inputs"]["duration_min"] * SECONDS_PER_MINUTE,
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_app(*, graph=None, cfg: dict | None = None) -> FastAPI:
    cfg = cfg if cfg is not None else load_thresholds()
    app = FastAPI(title="FormulaTech Race Readiness Check")
    state: dict[str, Any] = {"graph": graph}

    def get_graph():
        if state["graph"] is None:  # built on first use so the app starts without Ollama
            state["graph"] = build_race_graph(
                model_name=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
                base_url=os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL),
                cfg=cfg,
            )
        return state["graph"]

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        """Input limits for client-side validation (the server validates again)."""
        return {"input_limits": cfg["input_limits"], "seconds_per_minute": SECONDS_PER_MINUTE}

    @app.post("/api/check")
    def check(raw: dict[str, Any]):
        inputs, errors = validate_inputs(raw, cfg)
        if errors:
            return JSONResponse(status_code=422, content={"errors": errors})
        final = run_race_assessment(inputs, graph=get_graph())
        return build_response(final, cfg)

    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
