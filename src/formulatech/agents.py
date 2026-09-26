"""LangGraph-based AI agent layer for the race readiness checker.

This module implements the four-agent flow described in spec/04-agents.md while
keeping the final verdict deterministic and local to the machine. The LLM is used
only for explanations; it never decides whether a check passes or fails.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, TypedDict

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from xgboost import XGBRegressor

from formulatech.config import REPO_ROOT, load_thresholds
from formulatech.ml.train import FEATURES
from formulatech.rules import check_overcharge, check_thermal

MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "efficiency_model.json"
META_PATH = MODEL_DIR / "efficiency_model_meta.json"


class RaceState(TypedDict):
    inputs: dict[str, float]
    stage_results: list[dict[str, Any]]
    explanations: list[str]
    verdict: str | None
    failures: list[dict[str, Any]]
    summary: str


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


def _as_serializable(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        return value.__dict__
    return value


def _template_fallback(stage: str, result: dict[str, Any]) -> str:
    code = result.get("code", "CHECK")
    return (
        f"Template explanation for {stage}: the deterministic safety check reported "
        f"{code}. The local model explanation was unavailable, so the system used the "
        "fallback message while keeping the computed verdict unchanged."
    )


def _safe_model_factory(model_name: str = DEFAULT_OLLAMA_MODEL, base_url: str = DEFAULT_OLLAMA_URL):
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:  # pragma: no cover - handled by runtime fallback
        raise RuntimeError("langchain-ollama is not installed") from exc
    return ChatOllama(model=model_name, base_url=base_url, temperature=0.0)


def _explain_with_llm(
    model: Any,
    stage: str,
    result: dict[str, Any],
    *,
    model_factory: Callable[..., Any] | None = None,
) -> str:
    if model is None and model_factory is not None:
        try:
            model = model_factory()
        except Exception:
            return _template_fallback(stage, result)

    if model is None:
        return _template_fallback(stage, result)

    prompt = (
        "You are an assistant that explains a battery safety check to a driver. "
        "Do not change the verdict or say that the model decided the pass/fail. "
        "Describe only the result in plain language and include the recorded numeric value(s) "
        "and threshold(s) from the JSON below. Do not mention rule IDs or reason codes.\n\n"
        f"RESULT_JSON={json.dumps(result, sort_keys=True, default=str)}"
    )

    try:
        response = model.invoke(prompt)
        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)
        return str(content).strip() or _template_fallback(stage, result)
    except Exception:
        return _template_fallback(stage, result)


def get_model_status() -> dict[str, Any]:
    """Read the saved metadata from the trained model, if present."""
    if not META_PATH.exists():
        return {"version": None, "gates_passed": False, "tolerance_accuracy": None, "r2": None, "cv_mean": None, "cv_std": None, "fit_status": "unknown"}

    with META_PATH.open() as fh:
        meta = json.load(fh)

    metrics = meta.get("metrics", {})
    gates = meta.get("gates", {})
    return {
        "version": meta.get("version"),
        "gates_passed": bool(gates.get("gates_passed", False)),
        "tolerance_accuracy": metrics.get("tolerance_accuracy"),
        "r2": metrics.get("r2_test"),
        "r2_train": metrics.get("r2_train"),
        "cv_mean": metrics.get("cv_r2_mean"),
        "cv_std": metrics.get("cv_r2_std"),
        "fit_status": metrics.get("fit_status", "unknown"),
    }


def predict_efficiency(voltage_v: float, current_a: float, temperature_c: float, duration_min: float) -> float:
    """Load the saved XGBoost model and predict battery efficiency."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))
    X = pd.DataFrame(
        [{
            "Voltage (V)": voltage_v,
            "Current (A)": current_a,
            "Battery Temp (°C)": temperature_c,
            "Charging Duration (min)": duration_min,
        }],
        columns=FEATURES,
    )
    pred = model.predict(X)
    return float(pred[0])


def _result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "__dict__"):
        payload = result.__dict__.copy()
        payload["passed"] = getattr(result, "passed")
        return payload
    return {"value": result}


def compute_verdict(stage_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic final verdict: only the tool result decides pass/fail."""
    failures = [res for res in stage_results if not res.get("passed", True)]
    verdict = "DO_NOT_PROCEED" if failures else "CAN_PROCEED"

    summary = (
        "Battery is safe to proceed into the race."
        if verdict == "CAN_PROCEED"
        else "Battery does not meet the pass criteria and must not proceed into the race."
    )

    return {
        "verdict": verdict,
        "failures": [
            {k: v for k, v in res.items() if k not in {"stage", "passed"}}
            for res in failures
        ],
        "summary": summary,
    }


def _append_result(state: RaceState, stage: int, result: Any) -> None:
    payload = _result_to_dict(result)
    payload["stage"] = stage
    state["stage_results"].append(payload)


def _node_overcharge(state: RaceState, *, model_factory: Callable[..., Any] | None = None) -> RaceState:
    cfg = load_thresholds()
    result = check_overcharge(state["inputs"]["voltage_v"], state["inputs"]["current_a"], cfg)
    _append_result(state, 1, result)

    explanation = _explain_with_llm(None, "overcharge", _result_to_dict(result), model_factory=model_factory)
    state["explanations"].append(explanation)
    return state


def _node_thermal(state: RaceState, *, model_factory: Callable[..., Any] | None = None) -> RaceState:
    cfg = load_thresholds()
    results = check_thermal(state["inputs"]["temperature_c"], state["inputs"]["duration_min"], cfg)
    for item in results:
        _append_result(state, 2, item)

    for item in results:
        explanation = _explain_with_llm(None, "thermal", _result_to_dict(item), model_factory=model_factory)
        state["explanations"].append(explanation)
    return state


def _node_prediction(state: RaceState, *, model_factory: Callable[..., Any] | None = None) -> RaceState:
    cfg = load_thresholds()
    threshold = cfg["efficiency"]["min_percent"]

    status = get_model_status()
    if not status["gates_passed"]:
        result = {
            "rule": "R6",
            "code": "MODEL_NOT_VALIDATED",
            "passed": False,
            "recorded": {"gates_passed": status["gates_passed"]},
            "recorded_kind": "model",
            "thresholds": {"gates_passed": "true"},
            "unit": {"gates_passed": "bool"},
            "reason": "Model does not pass the required validation gates.",
        }
        _append_result(state, 3, result)
        explanation = _explain_with_llm(None, "prediction", result, model_factory=model_factory)
        state["explanations"].append(explanation)
        return state

    try:
        efficiency = predict_efficiency(
            state["inputs"]["voltage_v"],
            state["inputs"]["current_a"],
            state["inputs"]["temperature_c"],
            state["inputs"]["duration_min"],
        )
    except Exception:
        efficiency = float("nan")
        result = {
            "rule": "R5",
            "code": "LOW_EFFICIENCY",
            "passed": False,
            "recorded": {"efficiency_pct": efficiency},
            "recorded_kind": "calculated",
            "thresholds": {"efficiency_pct": f"< {threshold}"},
            "unit": {"efficiency_pct": "%"},
            "reason": "Prediction could not be generated and the model is treated as invalid.",
        }
        _append_result(state, 3, result)
        explanation = _explain_with_llm(None, "prediction", result, model_factory=model_factory)
        state["explanations"].append(explanation)
        return state

    passed = efficiency >= threshold
    result = {
        "rule": "R5",
        "code": "LOW_EFFICIENCY" if not passed else "EFFICIENCY_OK",
        "passed": passed,
        "recorded": {"efficiency_pct": efficiency},
        "recorded_kind": "calculated",
        "thresholds": {"efficiency_pct": f"< {threshold}"},
        "unit": {"efficiency_pct": "%"},
        "reason": "Prediction is below the minimum efficiency threshold." if not passed else "Prediction is above the threshold.",
    }
    _append_result(state, 3, result)
    explanation = _explain_with_llm(None, "prediction", result, model_factory=model_factory)
    state["explanations"].append(explanation)
    return state


def _node_decision(state: RaceState) -> RaceState:
    verdict = compute_verdict(state["stage_results"])
    state["verdict"] = verdict["verdict"]
    state["failures"] = verdict["failures"]
    state["summary"] = verdict["summary"]
    return state


def _route_after_overcharge(state: RaceState) -> str:
    return "decision" if any(not item.get("passed", True) for item in state["stage_results"]) else "thermal"


def _route_after_thermal(state: RaceState) -> str:
    return "decision" if any(not item.get("passed", True) for item in state["stage_results"]) else "prediction"


def _route_after_prediction(state: RaceState) -> str:
    return "decision"


def build_race_graph(
    *,
    model_name: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    model_factory: Callable[..., Any] | None = None,
):
    """Build the four-node LangGraph state machine from the spec."""
    graph = StateGraph(RaceState)

    if model_factory is None:
        def model_factory():
            return _safe_model_factory(model_name=model_name, base_url=base_url)

    graph.add_node("overcharge", lambda s: _node_overcharge(s, model_factory=model_factory))
    graph.add_node("thermal", lambda s: _node_thermal(s, model_factory=model_factory))
    graph.add_node("prediction", lambda s: _node_prediction(s, model_factory=model_factory))
    graph.add_node("decision", _node_decision)

    graph.add_edge(START, "overcharge")
    graph.add_conditional_edges("overcharge", _route_after_overcharge, {"decision": "decision", "thermal": "thermal"})
    graph.add_conditional_edges("thermal", _route_after_thermal, {"decision": "decision", "prediction": "prediction"})
    graph.add_conditional_edges("prediction", _route_after_prediction, {"decision": "decision"})
    graph.add_edge("decision", END)
    return graph.compile()


def run_race_assessment(inputs: dict[str, float], *, graph=None) -> RaceState:
    """Execute the graph for a single battery reading."""
    if graph is None:
        graph = build_race_graph()

    state: RaceState = {
        "inputs": inputs,
        "stage_results": [],
        "explanations": [],
        "verdict": None,
        "failures": [],
        "summary": "",
    }
    return graph.invoke(state)
