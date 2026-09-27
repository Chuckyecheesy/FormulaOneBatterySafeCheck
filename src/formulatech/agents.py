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
from langchain_core.tools import BaseTool, tool
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
    comment: str  # fixed risk comment for the failed stage (spec/05-ui.md §3.2); empty on CAN_PROCEED


# Fixed comments, by failed stage (spec/05-ui.md §3.2). Stages 1–2: fire hazard now; stage 3 (R5 only): during the race.
COMMENT_FIRE_RISK_NOW = "Your battery is at risk of fire hazard if you start the race now."
COMMENT_FIRE_RISK_IN_RACE = "Your battery is at risk of fire hazard during the middle of the race."


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
    rec, th = result["recorded"], cfg["thermal"]
    match result["code"]:
        case "OVERCHARGED":
            oc = cfg["overcharge"]
            return (
                f"Battery already overcharged. Battery voltage: {_fmt(rec['V'])} V, "
                f"battery current: {_fmt(rec['I'])} A. Battery voltage fire hazard threshold: "
                f"> {_fmt(oc['voltage_max_v'])} V. Battery current fire hazard threshold: "
                f"< {_fmt(oc['current_max_a'])} A. (Unsafe when both are true.)"
            )
        case "HIGH_DTDT":
            return (
                f"Thermal stress from battery charging is at high risk. dT/dt: {_rate(rec['dT_dt'])} °C/s. "
                f"Fire hazard threshold: > {_fmt(th['dT_dt_max_c_per_s'])} °C/s."
            )
        case "HIGH_D2TDT2":
            return (
                f"Heat is increasing at a very fast rate while the battery charges. "
                f"d²T/dt²: {_rate(rec['d2T_dt2'])} °C/s². "
                f"Fire hazard threshold: > {_fmt(th['d2T_dt2_max_c_per_s2'])} °C/s²."
            )
        case "HIGH_TCHEM":
            return (
                f"Heat exposure to the surrounding environment is too high. "
                f"T_chem: {_temp(rec['T_chem'])} °C. "
                f"Fire hazard threshold: > {_fmt(th['t_chem_max_c'])} °C."
            )
        case "LOW_EFFICIENCY":
            return (
                f"Predicted efficiency is too low, so the car cannot proceed into the race. "
                f"Predicted efficiency: {_temp(rec['efficiency_pct'])} % "
                f"(XGBoost model v{result.get('model_version')}). "
                f"Required to pass: ≥ {_fmt(cfg['efficiency']['min_percent'])} %."
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


def explain(llm: Any, result: dict[str, Any], fallback: str, cfg: dict) -> str:
    """Ask the LLM to explain a computed result. Fall back to the template on any error (NFR-2).

    The thermal baseline in the prompt is read from cfg, never hard-coded (spec/04-agents.md §3.4).
    """
    if llm is None:
        return fallback
    payload = {k: v for k, v in result.items() if k not in {"rule", "code", "stage"}}
    t0 = _fmt(cfg["thermal"]["initial_temp_c"])
    prompt = (
        "Explain this battery safety check result to a race engineer in one or two sentences. "
        "Do not change the result. Include the recorded value(s) and threshold(s). "
        "Do not mention rule IDs or reason codes. "
        f"For T_chem, treat it as the temperature rise above the {t0}°C baseline, not the absolute battery temperature. "
        f"In other words, T_chem = max(0, T_t - {t0}°C).\n"
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
# The model's acceptance gates (spec/03-ml-model.md §5) are checked when it is trained
# (`python -m formulatech.ml.train`), not here: the deployed model is assumed to have passed them.


class PredictionUnavailable(RuntimeError):
    """Stage 3 could not produce a prediction. There is no verdict: never a CAN PROCEED."""


def get_model_version() -> str | None:
    """Version of the deployed model, from its saved metadata. Display only; None if unreadable."""
    try:
        with META_PATH.open() as fh:
            version = json.load(fh).get("version")
    except Exception:
        return None
    return version if isinstance(version, str) else None


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


def evaluate_prediction(inputs: dict[str, float], cfg: dict) -> dict[str, Any]:
    """R5 in code, strict `<`. Raises PredictionUnavailable if the model cannot predict."""
    try:
        raw = predict_efficiency(inputs["voltage_v"], inputs["current_a"], inputs["temperature_c"],
                                 inputs["duration_min"], inputs["soc_percent"])
    except Exception as exc:
        raise PredictionUnavailable("The efficiency model could not produce a prediction") from exc
    if not math.isfinite(raw):
        raise PredictionUnavailable("The efficiency model could not produce a prediction")

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
        "model_version": get_model_version(),
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
# @tool wrappers around the deterministic checks
# ---------------------------------------------------------------------------


def make_check_tools(cfg: dict) -> dict[str, BaseTool]:
    """Wrap each stage's deterministic check as a @tool bound to `cfg`.

    The graph nodes invoke these in code (spec/04-agents.md §3.3); the LLM never calls them
    in the pipeline. The same tools can be given to a read-only Q&A agent. Docstrings hold no
    threshold values: the returned records carry the thresholds from thresholds.yaml.
    """

    @tool
    def check_overcharge_tool(voltage_v: float, current_a: float) -> list[dict]:
        """Run the overcharge check on a battery voltage in V and battery current in A.
        Returns one check result record with the recorded values, thresholds and pass/fail."""
        return [asdict(check_overcharge(voltage_v, current_a, cfg))]

    @tool
    def check_thermal_tool(temperature_c: float, duration_min: float) -> list[dict]:
        """Run the thermal checks (dT/dt, d²T/dt², T_chem) on a battery temperature in °C
        measured after charging for duration_min minutes. Returns one record per check."""
        return [asdict(r) for r in check_thermal(temperature_c, duration_min, cfg)]

    @tool
    def check_efficiency_tool(
        voltage_v: float, current_a: float, temperature_c: float, duration_min: float, soc_percent: float
    ) -> list[dict]:
        """Predict efficiency (%) for the reading with the deployed XGBoost model and compare it
        with the minimum. Returns one check result record."""
        inputs = {"voltage_v": voltage_v, "current_a": current_a, "temperature_c": temperature_c,
                  "duration_min": duration_min, "soc_percent": soc_percent}
        return [evaluate_prediction(inputs, cfg)]

    return {t.name: t for t in (check_overcharge_tool, check_thermal_tool, check_efficiency_tool)}


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
    tools = make_check_tools(cfg)

    def explained(state: RaceState, results: list[dict[str, Any]], stage: int) -> dict:
        """Partial update: append results; explain only the failures (nothing to explain on a pass)."""
        results = [{**r, "stage": stage} for r in results]
        new = [explain(llm, r, template_message(r, state["inputs"], cfg), cfg) for r in results if not r["passed"]]
        return {"stage_results": state["stage_results"] + results, "explanations": state["explanations"] + new}

    def run_tool(name: str, inputs: dict[str, float]) -> list[dict[str, Any]]:
        """Invoke a check tool in code with its arguments taken from the inputs (never from the LLM)."""
        check = tools[name]
        return check.invoke({k: inputs[k] for k in check.args})

    # Agent 1: Overcharge (stage 1)
    def overcharge_node(state: RaceState) -> dict:
        return explained(state, run_tool("check_overcharge_tool", state["inputs"]), 1)

    # Agent 2: Thermal (stage 2); all three rules always run (FR-2)
    def thermal_node(state: RaceState) -> dict:
        return explained(state, run_tool("check_thermal_tool", state["inputs"]), 2)

    # Agent 3: Prediction (stage 3)
    def prediction_node(state: RaceState) -> dict:
        try:
            results = run_tool("check_efficiency_tool", state["inputs"])
        except PredictionUnavailable:
            raise
        except Exception as exc:  # missing or invalid input: no prediction, so no verdict
            raise PredictionUnavailable("The efficiency model could not produce a prediction") from exc
        return explained(state, results, 3)

    # Agent 4: Race Decision; no LLM for verdict, failures, summary or comment
    def decision_node(state: RaceState) -> dict:
        result = compute_verdict(state["stage_results"])
        if result["verdict"] == "CAN_PROCEED":
            summary = "CAN PROCEED into the race."
            comment = ""
        else:  # state every failed check reason
            failed_codes = {r["code"] for r in result["failures"]}
            if "LOW_EFFICIENCY" in failed_codes:
                comment = COMMENT_FIRE_RISK_IN_RACE
            else:
                comment = COMMENT_FIRE_RISK_NOW
            summary = "DO NOT PROCEED. " + " ".join(
                template_message(f, state["inputs"], cfg) for f in result["failures"]
            )
            for text in state["explanations"]:
                if _claims_proceed(text):
                    logger.warning("LLM explanation contradicts computed verdict %s: %r", result["verdict"], text)
        return {**result, "summary": summary, "comment": comment}

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
        "comment": "",
    })
