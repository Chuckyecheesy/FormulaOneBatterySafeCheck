---
name: langgraph-agents
description: Build or change the FormulaTech AI agent layer with LangGraph and a local Ollama model (langchain-ollama ChatOllama). Covers model choice, installation, the StateGraph node pattern, @tool, create_react_agent, and graph state. Use when the user asks to add, change or debug agents, LangGraph graphs, Ollama/ChatOllama calls, tool calling, or src/formulatech/agents.py.
---

# SKILL: LangGraph Agents with Local Ollama

## Skill Objective
Write agent code that follows [spec/04-agents.md](../../../spec/04-agents.md): **code decides, agents explain.** Deterministic Python functions make every threshold comparison. The LLM only writes explanations of results it is given. It never decides a pass/fail and never writes `verdict` or `failures`.

## Activation Triggers
- The user asks to add, change or debug an agent, a LangGraph graph, or an Ollama call
- The user edits [src/formulatech/agents.py](../../../src/formulatech/agents.py)
- The user mentions `StateGraph`, `ChatOllama`, `@tool`, `create_react_agent`, or tool calling

## Hard Rules (from CLAUDE.md, never break)
1. **Local only.** Use Ollama through `langchain-ollama`. Never use a hosted LLM API (OpenAI, Anthropic, and so on).
2. **One `StateGraph`, one node per agent:** Overcharge → Thermal → Prediction → Race Decision, with `add_conditional_edges` for the two short-circuits.
3. **The node calls the tool in code, not the LLM.** In the check pipeline the LLM must not choose whether to run a check.
4. **Routing and verdict read `stage_results`, never LLM text.** The verdict comes from `compute_verdict`.
5. **No threshold literals in code or prompts.** Read them from `spec/thresholds.yaml` via `load_thresholds()` and pass them into the prompt as data.
6. **Always have a fallback.** If Ollama is down, errors, or takes longer than 8 s, use the template message from [spec/05-ui.md](../../../spec/05-ui.md) §3.1. The graph still finishes with the computed verdict.

## Prerequisites & Model Choice
Run an Ollama version and an open-weight model that **natively supports tool calling** (function calling). Models without tool calling return plain text instead of properly formatted tool requests, and a tool-calling agent loop (Template B) breaks.

- **Recommended models:** `llama3.1:8b` (the project default), `llama3.2:3b` (faster), or `qwen2.5:7b`.
- The check pipeline (Template A) only asks the model for explanations, so it works with any model. Tool calling matters only for Template B.

```bash
ollama pull llama3.1:8b      # or: ollama pull llama3.2
ollama serve                  # default http://localhost:11434
ollama list                   # confirm the model is present
```

## Step-by-Step Implementation

### 1. Install dependencies
The project uses `uv`, and these are already listed in `pyproject.toml`:

```bash
uv add langgraph langchain-ollama langchain-core
# without uv: pip install langgraph langchain-ollama langchain-core
```

### 2. Initialize Ollama
Always create the model with `ChatOllama(...)`. Use `temperature=0` for repeatable text, a short `num_predict` to stay within NFR-1 (≤ 10 s per check), and an 8 s client timeout for the NFR-2 fallback.

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0,
    num_predict=150,
    client_kwargs={"timeout": 8},
)
```

### 3. Template A: the check pipeline (use this for stages 1–4)
Each node runs its deterministic tool in code, appends the result record, then asks the LLM to explain it. The LLM output goes only into `explanations`. The decision node builds `verdict`, `failures` and `summary` from `stage_results` alone, and the summary states every failed check reason. `get_model_status` and `predict_efficiency` are the deterministic stage 3 tools in `src/formulatech/agents.py`.

```python
import json
from dataclasses import asdict
from typing import Any, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from formulatech.agents import get_model_status, predict_efficiency
from formulatech.config import load_thresholds
from formulatech.rules import COMPARE_DECIMALS, check_overcharge, check_thermal

llm = ChatOllama(model="llama3.1:8b", temperature=0, num_predict=150, client_kwargs={"timeout": 8})


class CheckState(TypedDict):
    inputs: dict[str, float]             # voltage_v, current_a, temperature_c, duration_min
    stage_results: list[dict[str, Any]]  # check result records, appended by each node
    explanations: list[str]              # LLM text only; never read back into the verdict
    verdict: str | None                  # set only by the decision node, from compute_verdict
    failures: list[dict[str, Any]]
    summary: str


def template_message(result: dict) -> str:
    """Fallback text (spec/05-ui.md §3.1), built from the record so no threshold is hard-coded."""
    parts = [f"{result['reason']}."]
    for name, value in result["recorded"].items():
        unit = result["unit"].get(name, "")
        parts.append(f"{name}: {value} {unit} (hazard threshold: {result['thresholds'][name]} {unit}).")
    return " ".join(parts)


def explain(result: dict) -> str:
    """Ask the LLM to explain a computed result. Fall back to the template on any error (NFR-2)."""
    prompt = (
        "Explain this battery safety check result to a race engineer in one or two sentences. "
        "Do not change the result. Include the recorded value(s) and threshold(s). "
        "Do not mention rule IDs or reason codes.\n"
        f"RESULT_JSON={json.dumps(result, default=str)}"
    )
    try:
        text = llm.invoke(prompt).content.strip()
    except Exception:  # Ollama down, error, or 8 s timeout
        return template_message(result)
    return text or template_message(result)


def any_failed(results: list[dict]) -> bool:
    return any(not r["passed"] for r in results)


# ---- Agent 1: Overcharge (stage 1) ----
def overcharge_node(state: CheckState) -> dict:
    cfg = load_thresholds()
    i = state["inputs"]
    result = asdict(check_overcharge(i["voltage_v"], i["current_a"], cfg))   # 1. code decides
    update = {"stage_results": state["stage_results"] + [result]}
    if not result["passed"]:                                                # skip LLM when nothing to explain
        update["explanations"] = state["explanations"] + [explain(result)]  # 2. LLM only explains
    return update


# ---- Agent 2: Thermal (stage 2) — all three rules always run (FR-2) ----
def thermal_node(state: CheckState) -> dict:
    cfg = load_thresholds()
    i = state["inputs"]
    results = [asdict(r) for r in check_thermal(i["temperature_c"], i["duration_min"], cfg)]
    return {
        "stage_results": state["stage_results"] + results,
        "explanations": state["explanations"] + [explain(r) for r in results if not r["passed"]],
    }


# ---- Agent 3: Prediction (stage 3) — fails closed with R6 ----
def model_not_validated(status: dict, cfg: dict, reason: str) -> dict:
    g = cfg["model_gates"]
    gap = None
    if status.get("r2_train") is not None and status.get("r2") is not None:
        gap = round(status["r2_train"] - status["r2"], COMPARE_DECIMALS)
    return {
        "rule": "R6", "code": "MODEL_NOT_VALIDATED", "passed": False,
        "recorded": {"tolerance_accuracy": status.get("tolerance_accuracy"), "r2": status.get("r2"), "r2_gap": gap},
        "recorded_kind": "calculated",
        "thresholds": {
            "tolerance_accuracy": f"> {g['tolerance_accuracy_min_percent']:g}",
            "r2": f"> {g['r2_min']:g}",
            "r2_gap": f">= {g['fit_r2_gap_min']:g} and < {g['fit_r2_gap_max']:g}",
        },
        "unit": {"tolerance_accuracy": "%", "r2": "", "r2_gap": ""},
        "reason": reason,
    }


def prediction_node(state: CheckState) -> dict:
    cfg = load_thresholds()
    i = state["inputs"]
    status = get_model_status()
    if not status["gates_passed"]:
        result = model_not_validated(status, cfg, "Efficiency model not validated, so the race cannot be cleared")
    else:
        try:
            eff = round(predict_efficiency(i["voltage_v"], i["current_a"], i["temperature_c"], i["duration_min"]),
                        COMPARE_DECIMALS)
        except Exception:
            result = model_not_validated(status, cfg, "Efficiency model could not produce a prediction")
        else:
            min_pct = cfg["efficiency"]["min_percent"]
            result = {
                "rule": "R5", "code": "LOW_EFFICIENCY", "passed": not eff < min_pct,   # strict <
                "recorded": {"efficiency_pct": eff}, "recorded_kind": "calculated",
                "thresholds": {"efficiency_pct": f"< {min_pct:g}"}, "unit": {"efficiency_pct": "%"},
                "reason": "Predicted efficiency is too low, so the car cannot proceed into the race",
            }
    update = {"stage_results": state["stage_results"] + [result]}
    if not result["passed"]:
        update["explanations"] = state["explanations"] + [explain(result)]
    return update


# ---- Agent 4: Race Decision — deterministic, no LLM for verdict/failures ----
def compute_verdict(stage_results: list[dict]) -> dict:
    failures = [r for r in stage_results if not r["passed"]]
    return {"verdict": "DO_NOT_PROCEED" if failures else "CAN_PROCEED", "failures": failures}


def decision_node(state: CheckState) -> dict:
    result = compute_verdict(state["stage_results"])
    if result["verdict"] == "CAN_PROCEED":
        summary = "CAN PROCEED into the race."
    else:  # state every failed check reason (CLAUDE.md rule logic)
        summary = "DO NOT PROCEED. " + " ".join(template_message(f) for f in result["failures"])
    return {**result, "summary": summary}


# ---- Routing reads stage_results, never LLM output ----
def route_after_overcharge(state: CheckState) -> str:
    return "decision" if any_failed(state["stage_results"]) else "thermal"


def route_after_thermal(state: CheckState) -> str:
    return "decision" if any_failed(state["stage_results"]) else "prediction"


graph = StateGraph(CheckState)
graph.add_node("overcharge", overcharge_node)
graph.add_node("thermal", thermal_node)
graph.add_node("prediction", prediction_node)
graph.add_node("decision", decision_node)
graph.add_edge(START, "overcharge")
graph.add_conditional_edges("overcharge", route_after_overcharge, {"thermal": "thermal", "decision": "decision"})
graph.add_conditional_edges("thermal", route_after_thermal, {"prediction": "prediction", "decision": "decision"})
graph.add_edge("prediction", "decision")
graph.add_edge("decision", END)
app = graph.compile()

final_state = app.invoke({
    "inputs": {"voltage_v": 5.0, "current_a": -1.2, "temperature_c": 25.0, "duration_min": 60},
    "stage_results": [], "explanations": [], "verdict": None, "failures": [], "summary": "",
})
print(final_state["verdict"], "-", final_state["summary"])
```

Nodes return **partial updates** (a dict with only the keys they change). LangGraph merges them into the state.

### 4. Template B: a tool-calling agent (only for follow-up Q&A, never for the verdict)
Use this only for an extra, optional agent. For example, an engineer asks "why did this fail?" after the verdict is shown. The tools must be **read-only wrappers around deterministic code**. The answer is display text and must never change `verdict` or `failures`.

```python
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from formulatech.config import load_thresholds
from formulatech.rules import thermal_quantities


@tool
def get_thermal_quantities(temperature_c: float, duration_min: float) -> dict:
    """Calculate dT/dt (°C/s), d²T/dt² (°C/s²) and T_chem (°C) for a battery temperature
    in °C measured after charging for duration_min minutes. Use this whenever the user asks
    about heating rate, heating acceleration or chemical heat exposure."""
    return thermal_quantities(temperature_c, duration_min, load_thresholds())


@tool
def get_thresholds() -> dict:
    """Return the hazard thresholds from thresholds.yaml. Use this before stating any limit."""
    cfg = load_thresholds()
    return {"overcharge": cfg["overcharge"], "thermal": cfg["thermal"], "efficiency": cfg["efficiency"]}


llm = ChatOllama(model="llama3.1:8b", temperature=0, client_kwargs={"timeout": 8})
qa_agent = create_react_agent(llm, tools=[get_thermal_quantities, get_thresholds])

result = qa_agent.invoke({"messages": [("user", "What is dT/dt at 40 °C after 5 minutes?")]})
print(result["messages"][-1].content)
```

On LangGraph ≥ 1.0, `create_react_agent` still works but raises `LangGraphDeprecatedSinceV10`. Its replacement is `from langchain.agents import create_agent`, which needs the `langchain` package.

## Key Architectural Concepts

- **`@tool` decorator:** turns a normal Python function into a schema (name, typed arguments, description) that LangGraph passes directly to Ollama. The **triple-quoted docstring** is the instruction the LLM reads to decide *when and why* to pick this tool, so write it for the model: say what the tool returns, the units, and when to use it. The type hints become the argument schema.
- **`create_react_agent`** (from `langgraph.prebuilt`): replaces building the graph by hand for a standard tool loop. It sets up a `StateGraph` with an **Agent node** (the model decides what to do next) and a **Tool node** (`ToolNode`, which runs the chosen function), joined by a conditional edge. The loop is: if the model's last message has tool calls, go to the Tool node, then back to the Agent node; otherwise stop. **Use it only for Template B.** In the check pipeline the model would be choosing whether a safety check runs, which breaks Hard Rule 3. In LangGraph 1.x it still works but gives a `LangGraphDeprecatedSinceV10` warning. The replacement is `from langchain.agents import create_agent`, which needs the separate `langchain` package (not installed here). The warning is safe to ignore until LangGraph 2.0.
- **State management:** in `create_react_agent`, LangGraph keeps a `messages` list in the conversation state. When the model decides it needs a tool, it appends an `AIMessage` with `tool_calls`. The Tool node runs the function and appends the output as a `ToolMessage`. The whole sequence is sent back to the LLM, which writes the final reply. In Template A the state is the explicit `CheckState` TypedDict instead: each node returns the keys it updates, and routing functions read those keys.
- **`ChatOllama(...)`:** the only way to create the LLM in this repo. It needs a running `ollama serve` and a model that has been pulled.

## Verification Checklist
Before finishing any agent change:
- [ ] `grep -rnE "openai|anthropic" src/` finds no hosted LLM clients
- [ ] No threshold number appears in a prompt string or in agent code
- [ ] Test AG-1: with the LLM mocked to say "PROCEED" on S1-1 inputs, the verdict is still `DO_NOT_PROCEED`
- [ ] Test AG-2: with Ollama stopped or timing out, the graph still returns the `compute_verdict` result using template messages
- [ ] Test AG-3: every failure includes the recorded value(s) and threshold(s)
- [ ] `uv run pytest` passes

## Prohibited Behaviors
- **NEVER** let an LLM output set `verdict`, `failures`, or a routing decision.
- **NEVER** use `create_react_agent`, or any tool-calling loop, to run the stage 1–3 checks.
- **NEVER** call a hosted LLM API.
- **NEVER** hard-code thresholds in prompts or tool docstrings. Fetch them at runtime.
