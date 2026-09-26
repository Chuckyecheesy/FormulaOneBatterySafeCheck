"""LangGraph-based AI agent layer for the race readiness checker (spec/04-agents.md).

Code decides, agents explain: every node runs its deterministic check in code, then
asks a local Ollama model to explain the result. The LLM text only goes into
`explanations`; routing, `verdict`, `failures` and `summary` come from `stage_results`.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict
from typing import Any, Callable, TypedDict

import pandas as pd
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from xgboost import XGBRegressor

from formulatech.config import MODELS_DIR, load_thresholds
from formulatech.ml.train import FEATURES
from formulatech.rules import COMPARE_DECIMALS, check_overcharge, check_thermal

logger = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "efficiency_model.json"
META_PATH = MODELS_DIR / "efficiency_model_meta.json"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
LLM_TIMEOUT_S = 8  # NFR-2: fall back to the template after this
LLM_NUM_PREDICT = 150  # NFR-1: keep explanations short


class RaceState(TypedDict):
    inputs: dict[str, float]  # voltage_v, current_a, temperature_c, duration_min, soc_percent
    stage_results: list[dict[str, Any]]  # check result records, appended by each node
    explanations: list[str]  # LLM text only; never read back into the verdict
    verdict: str | None  # set only by the decision node, from compute_verdict
    failures: list[dict[str, Any]]
    summary: str


# ---------------------------------------------------------------------------
# Template messages (spec/05-ui.md §3.1). Thresholds are filled in from cfg.
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:g}"


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"  # 4 significant figures


def _temp(value: float) -> str:
    return f"{value:.2f}"  # 2 decimal places


def template_message(result: dict[str, Any], inputs: dict[str, float], cfg: dict) -> str:
    """Plain-language message for a failed check, used as the fallback and in the summary."""
    rec, th, gates = result["recorded"], cfg["thermal"], cfg["model_gates"]
    t0 = _fmt(th["initial_temp_c"])
    t_c = _temp(inputs["temperature_c"])
    match result["code"]:
        case "OVERCHARGED":
            oc = cfg["overcharge"]
            return (
                f"Battery already overcharged. Battery voltage: {_fmt(rec['V'])} V, "
                f"battery current: {_fmt(rec['I'])} A. Battery voltage hazard threshold: "
                f"> {_fmt(oc['voltage_max_v'])} V. Battery current hazard threshold: "
                f"< {_fmt(oc['current_max_a'])} A. (Unsafe when both are true.)"
            )
        case "HIGH_DTDT":
            t_s = _fmt(inputs["duration_min"] * 60)
            return (
                f"Thermal stress from battery charging is at high risk. dT/dt: {_rate(rec['dT_dt'])} °C/s "
                f"(calculated from T = {t_c} °C, T0 = {t0} °C, t = {t_s} s). "
                f"Hazard threshold: > {_fmt(th['dT_dt_max_c_per_s'])} °C/s."
            )
        case "HIGH_D2TDT2":
            return (
                f"Heat is increasing at a very fast rate while the battery charges. "
                f"d²T/dt²: {_rate(rec['d2T_dt2'])} °C/s². "
                f"Hazard threshold: > {_fmt(th['d2T_dt2_max_c_per_s2'])} °C/s²."
            )
        case "HIGH_TCHEM":
            return (
                f"Heat exposure to the surrounding environment is too high. "
                f"T_chem: {_temp(rec['T_chem'])} °C (= max(0, {t_c} − {t0})). "
                f"Hazard threshold: > {_fmt(th['t_chem_max_c'])} °C."
            )
        case "LOW_EFFICIENCY":
            return (
                f"Predicted efficiency is too low, so the car cannot proceed into the race. "
                f"Predicted efficiency: {_temp(rec['efficiency_pct'])} % "
                f"(XGBoost model v{result.get('model_version')}). "
                f"Required to pass: ≥ {_fmt(cfg['efficiency']['min_percent'])} %."
            )
        case "MODEL_NOT_VALIDATED":
            return (
                f"Efficiency model not validated, so the race cannot be cleared. "
                f"Tolerance accuracy: {_rate(rec['tolerance_accuracy'])} % "
                f"(needs > {_fmt(gates['tolerance_accuracy_min_percent'])} %), "
                f"R²: {_rate(rec['r2'])} (needs > {_fmt(gates['r2_min'])})."
            )
    return f"{result['reason']}."


# ---------------------------------------------------------------------------
# LLM explanation with fallback
# ---------------------------------------------------------------------------


def _default_model_factory(model_name: str, base_url: str) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=0,
        num_predict=LLM_NUM_PREDICT,
        client_kwargs={"timeout": LLM_TIMEOUT_S},
    )


def explain(llm: Any, result: dict[str, Any], fallback: str) -> str:
    """Ask the LLM to explain a computed result. Fall back to the template on any error (NFR-2)."""
    if llm is None:
        return fallback
    payload = {k: v for k, v in result.items() if k not in {"rule", "code", "stage"}}
    prompt = (
        "Explain this battery safety check result to a race engineer in one or two sentences. "
        "Do not change the result. Include the recorded value(s) and threshold(s). "
        "Do not mention rule IDs or reason codes.\n"
        f"RESULT_JSON={json.dumps(payload, default=str)}"
    )
    try:
        response = llm.invoke(prompt)
        text = str(getattr(response, "content", response)).strip()
    except Exception:  # Ollama down, error, or timeout
        logger.warning("LLM explanation failed; using template message", exc_info=True)
        return fallback
    return text or fallback


_PROCEED_CLAIM = re.compile(r"\b(can|may|safe to|able to|cleared to)\s+proceed\b|^\s*proceed\b", re.I)


def _claims_proceed(text: str) -> bool:
    return bool(_PROCEED_CLAIM.search(text)) and not re.search(r"\b(not|cannot|can't)\s+proceed\b", text, re.I)


# ---------------------------------------------------------------------------
# Deterministic stage 3 tools
# ---------------------------------------------------------------------------


def get_model_status() -> dict[str, Any]:
    """Read the saved metadata from the trained model. Missing metadata means not validated."""
    if not META_PATH.exists():
        return {"version": None, "gates_passed": False, "tolerance_accuracy": None, "r2": None,
                "r2_train": None, "cv_mean": None, "cv_std": None}

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
    }


def predict_efficiency(
    voltage_v: float, current_a: float, temperature_c: float, duration_min: float, soc_percent: float
) -> float:
    """Load the saved XGBoost model and predict battery efficiency (duration in minutes, SOC in %)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))
    X = pd.DataFrame(
        [{
            "SOC (%)": soc_percent,
            "Voltage (V)": voltage_v,
            "Current (A)": current_a,
            "Battery Temp (°C)": temperature_c,
            "Charging Duration (min)": duration_min,
        }],
        columns=FEATURES,
    )
    return float(model.predict(X)[0])


def _model_not_validated(status: dict[str, Any], cfg: dict, reason: str) -> dict[str, Any]:
    """R6 record. Stage 3 fails closed (FR-4)."""
    g = cfg["model_gates"]
    return {
        "rule": "R6",
        "code": "MODEL_NOT_VALIDATED",
        "passed": False,
        "recorded": {
            "tolerance_accuracy": status.get("tolerance_accuracy"),
            "r2": status.get("r2"),
            "cv_r2_mean": status.get("cv_mean"),
            "cv_r2_std": status.get("cv_std"),
        },
        "recorded_kind": "calculated",
        "thresholds": {
            "tolerance_accuracy": f"> {_fmt(g['tolerance_accuracy_min_percent'])}",
            "r2": f"> {_fmt(g['r2_min'])}",
            "cv_r2_mean": f"> {_fmt(g['r2_min'])}",
            "cv_r2_std": f"≤ {_fmt(g['cv_r2_std_max'])}",
        },
        "unit": {"tolerance_accuracy": "%", "r2": "", "cv_r2_mean": "", "cv_r2_std": ""},
        "reason": reason,
    }


def evaluate_prediction(inputs: dict[str, float], cfg: dict) -> dict[str, Any]:
    """R5/R6 in code: R6 if the model is not validated or cannot predict, else R5 with strict `<`."""
    status = get_model_status()
    if not status["gates_passed"]:
        return _model_not_validated(status, cfg, "Efficiency model not validated")
    try:
        raw = predict_efficiency(inputs["voltage_v"], inputs["current_a"], inputs["temperature_c"],
                                 inputs["duration_min"], inputs["soc_percent"])
    except Exception:
        logger.exception("Efficiency prediction failed")
        return _model_not_validated(status, cfg, "Efficiency model could not produce a prediction")
    if not math.isfinite(raw):
        return _model_not_validated(status, cfg, "Efficiency model could not produce a prediction")

    eff = round(raw, COMPARE_DECIMALS)
    min_pct = cfg["efficiency"]["min_percent"]
    return {
        "rule": "R5",
        "code": "LOW_EFFICIENCY",
        "passed": not eff < min_pct,
        "recorded": {"efficiency_pct": eff},
        "recorded_kind": "calculated",
        "thresholds": {"efficiency_pct": f"< {_fmt(min_pct)}"},
        "unit": {"efficiency_pct": "%"},
        "reason": "Predicted efficiency is too low, so the car cannot proceed into the race",
        "model_version": status["version"],
    }


# ---------------------------------------------------------------------------
# Verdict (deterministic)
# ---------------------------------------------------------------------------


def compute_verdict(stage_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Final verdict: CAN_PROCEED only if every check that ran passed (FR-5)."""
    failures = [r for r in stage_results if not r["passed"]]
    return {
        "verdict": "DO_NOT_PROCEED" if failures else "CAN_PROCEED",
        "failures": [{k: v for k, v in r.items() if k not in {"stage", "passed"}} for r in failures],
    }


def _any_failed(results: list[dict[str, Any]]) -> bool:
    return any(not r["passed"] for r in results)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_race_graph(
    *,
    model_name: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    model_factory: Callable[[], Any] | None = None,
    cfg: dict | None = None,
):
    """Build the four-node StateGraph: Overcharge → Thermal → Prediction → Race Decision."""
    cfg = cfg if cfg is not None else load_thresholds()
    factory = model_factory or (lambda: _default_model_factory(model_name, base_url))
    try:
        llm = factory()
    except Exception:
        logger.warning("Could not create the LLM; all explanations will use templates", exc_info=True)
        llm = None

    def explained(state: RaceState, results: list[dict[str, Any]], stage: int) -> dict:
        """Partial update: append results; explain only the failures (nothing to explain on a pass)."""
        results = [{**r, "stage": stage} for r in results]
        new = [explain(llm, r, template_message(r, state["inputs"], cfg)) for r in results if not r["passed"]]
        return {"stage_results": state["stage_results"] + results, "explanations": state["explanations"] + new}

    # Agent 1: Overcharge (stage 1)
    def overcharge_node(state: RaceState) -> dict:
        i = state["inputs"]
        return explained(state, [asdict(check_overcharge(i["voltage_v"], i["current_a"], cfg))], 1)

    # Agent 2: Thermal (stage 2); all three rules always run (FR-2)
    def thermal_node(state: RaceState) -> dict:
        i = state["inputs"]
        return explained(state, [asdict(r) for r in check_thermal(i["temperature_c"], i["duration_min"], cfg)], 2)

    # Agent 3: Prediction (stage 3)
    def prediction_node(state: RaceState) -> dict:
        return explained(state, [evaluate_prediction(state["inputs"], cfg)], 3)

    # Agent 4: Race Decision; no LLM for verdict, failures or summary
    def decision_node(state: RaceState) -> dict:
        result = compute_verdict(state["stage_results"])
        if result["verdict"] == "CAN_PROCEED":
            summary = "CAN PROCEED into the race."
        else:  # state every failed check reason
            summary = "DO NOT PROCEED. " + " ".join(
                template_message(f, state["inputs"], cfg) for f in result["failures"]
            )
            for text in state["explanations"]:
                if _claims_proceed(text):
                    logger.warning("LLM explanation contradicts computed verdict %s: %r", result["verdict"], text)
        return {**result, "summary": summary}

    # Routing reads stage_results, never LLM output
    def route_after_overcharge(state: RaceState) -> str:
        return "decision" if _any_failed(state["stage_results"]) else "thermal"

    def route_after_thermal(state: RaceState) -> str:
        return "decision" if _any_failed(state["stage_results"]) else "prediction"

    graph = StateGraph(RaceState)
    graph.add_node("overcharge", overcharge_node)
    graph.add_node("thermal", thermal_node)
    graph.add_node("prediction", prediction_node)
    graph.add_node("decision", decision_node)
    graph.add_edge(START, "overcharge")
    graph.add_conditional_edges("overcharge", route_after_overcharge, {"thermal": "thermal", "decision": "decision"})
    graph.add_conditional_edges("thermal", route_after_thermal, {"prediction": "prediction", "decision": "decision"})
    graph.add_edge("prediction", "decision")
    graph.add_edge("decision", END)
    return graph.compile()


def run_race_assessment(inputs: dict[str, float], *, graph=None) -> RaceState:
    """Execute the graph for a single, already-validated battery reading."""
    if graph is None:
        graph = build_race_graph()
    return graph.invoke({
        "inputs": inputs,
        "stage_results": [],
        "explanations": [],
        "verdict": None,
        "failures": [],
        "summary": "",
    })
