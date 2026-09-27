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
3. **Nodes run their check in code; the LLM only explains** (spec/04-agents.md §3.3). Each check is a `@tool` from `make_check_tools(cfg)`, but in the pipeline the node invokes it with arguments taken from `state["inputs"]`. There is no `create_react_agent` in the pipeline, and the LLM never calls a pipeline tool or picks its arguments.
4. **Routing and verdict read `stage_results`, never LLM text.** The verdict comes from `compute_verdict`.
5. **No threshold literals in code or prompts.** Read them from `spec/thresholds.yaml` via `load_thresholds()` and pass them into the prompt as data.
6. **Always have a fallback.** If Ollama is down, errors, or takes longer than 8 s, use the template message from [spec/05-ui.md](../../../spec/05-ui.md) §3.1. The graph still finishes with the computed verdict.

## Prerequisites & Model Choice
The check pipeline (Template A) only asks the model to write text, so any chat model works. Only the optional Q&A agent (Template B) needs a model that **natively supports tool calling** (function calling). Without it, the model returns plain text instead of tool requests and the loop breaks.

- **Recommended models:** `llama3.2` (the project default, 3B), `llama3.1:8b` (larger, slower), or `qwen2.5:7b`.
- The verdict never depends on the model: if Ollama is down or replies badly, Template A uses the template message.

```bash
ollama pull llama3.2         # or: ollama pull llama3.1:8b
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
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0,
    num_predict=150,
    client_kwargs={"timeout": 8},
)
```

### 3. Template A: the check pipeline (use this for stages 1–4)
This is the pattern in [src/formulatech/agents.py](../../../src/formulatech/agents.py) and [spec/04-agents.md](../../../spec/04-agents.md) §3.3. One `StateGraph` with one node per agent. Each stage's deterministic check is wrapped as a `@tool` by `make_check_tools(cfg)`, but **the node invokes the tool in code**, with the arguments taken from `state["inputs"]`. There is no agent loop and the LLM never calls a pipeline tool. Every node follows two steps:

1. **Run the tool in code** and append its records (tagged with `stage`) to `stage_results`.
2. **Explain only the failures.** `explain(llm, result, fallback)` makes one `llm.invoke(prompt)` call per failed record, with the record as JSON in the prompt. On any error, timeout or empty reply it returns the §3.1 `template_message`. A stage that passed makes no LLM call.

The Race Decision node has no LLM: `verdict` and `failures` come from `compute_verdict`, `summary` from `template_message` (every failed check reason), and `comment` is the fixed §3.2 sentence chosen by the failed stage. If an explanation claims the car can proceed on a DO NOT PROCEED verdict, the node logs the mismatch and the computed verdict stands.

```python
from dataclasses import asdict
from typing import Any, TypedDict

from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph

from formulatech.rules import check_overcharge, check_thermal


class RaceState(TypedDict):
    inputs: dict[str, float]             # voltage_v, current_a, temperature_c, duration_min, soc_percent
    stage_results: list[dict[str, Any]]  # check result records, appended by each node
    explanations: list[str]              # LLM text only; never read back into the verdict
    verdict: str | None                  # set only by the decision node, from compute_verdict
    failures: list[dict[str, Any]]
    summary: str
    comment: str                         # fixed risk comment (05-ui.md §3.2); empty on CAN_PROCEED


def make_check_tools(cfg: dict) -> dict[str, BaseTool]:
    """Wrap each stage's check as a @tool bound to cfg. Nodes invoke these in code."""

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

    # check_efficiency_tool wraps evaluate_prediction (R5; raises PredictionUnavailable if it cannot predict)
    return {t.name: t for t in (check_overcharge_tool, check_thermal_tool)}


def build_race_graph(*, model_factory=None, cfg=None):
    cfg = cfg if cfg is not None else load_thresholds()
    try:
        llm = (model_factory or default_ollama)()
    except Exception:
        llm = None                                   # every explanation uses the template
    tools = make_check_tools(cfg)

    def run_tool(name: str, inputs: dict[str, float]) -> list[dict[str, Any]]:
        check = tools[name]                          # arguments come from state, never from the LLM
        return check.invoke({k: inputs[k] for k in check.args})

    def explained(state: RaceState, results: list[dict[str, Any]], stage: int) -> dict:
        results = [{**r, "stage": stage} for r in results]
        new = [explain(llm, r, template_message(r, state["inputs"], cfg)) for r in results if not r["passed"]]
        return {"stage_results": state["stage_results"] + results, "explanations": state["explanations"] + new}

    def overcharge_node(state: RaceState) -> dict:
        return explained(state, run_tool("check_overcharge_tool", state["inputs"]), 1)

    def thermal_node(state: RaceState) -> dict:      # all three rules always run (FR-2)
        return explained(state, run_tool("check_thermal_tool", state["inputs"]), 2)

    def prediction_node(state: RaceState) -> dict:   # cannot predict -> PredictionUnavailable (no verdict)
        ...

    def decision_node(state: RaceState) -> dict:     # no LLM: compute_verdict + template_message + comment
        ...

    # Routing reads stage_results, never LLM output
    def route_after_overcharge(state: RaceState) -> str:
        return "decision" if any_failed(state["stage_results"]) else "thermal"

    def route_after_thermal(state: RaceState) -> str:
        return "decision" if any_failed(state["stage_results"]) else "prediction"

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
```

Nodes return **partial updates** (a dict with only the keys they change). LangGraph merges them into the state. See `agents.py` for the full `explain`, `prediction_node` and `decision_node`. Run it with `run_race_assessment(inputs)`. At most one LLM call per failed record, so NFR-1 holds with `num_predict` about 150.

### 4. Template B: a follow-up Q&A agent (never for the verdict)
Use this for an extra, optional agent outside the check pipeline. For example, an engineer asks "why did this fail?" after the verdict is shown. The tools must be **read-only wrappers around deterministic code**. The answer is display text and must never change `verdict` or `failures`.

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


llm = ChatOllama(model="llama3.2", temperature=0, client_kwargs={"timeout": 8})
qa_agent = create_react_agent(llm, tools=[get_thermal_quantities, get_thresholds])

result = qa_agent.invoke({"messages": [("user", "What is dT/dt at 40 °C after 5 minutes?")]})
print(result["messages"][-1].content)
```

On LangGraph ≥ 1.0, `create_react_agent` still works but raises `LangGraphDeprecatedSinceV10`. Its replacement is `from langchain.agents import create_agent`, which needs the `langchain` package.

## Key Architectural Concepts

- **`@tool` decorator:** turns a normal Python function into a schema (name, typed arguments, description) that LangGraph passes directly to Ollama. The **triple-quoted docstring** is the instruction the LLM reads to decide *when and why* to pick this tool, so write it for the model: say what the tool returns, the units, and when to use it. The type hints become the argument schema.
- **`create_react_agent`** (from `langgraph.prebuilt`): replaces building the graph by hand for a standard tool loop. It sets up a `StateGraph` with an **Agent node** (the model decides what to do next) and a **Tool node** (`ToolNode`, which runs the chosen function), joined by a conditional edge. The loop is: if the model's last message has tool calls, go to the Tool node, then back to the Agent node; otherwise stop. It is used only in Template B, where the agent can pick freely among read-only tools. It is never used in the check pipeline. In LangGraph 1.x it still works but gives a `LangGraphDeprecatedSinceV10` warning. The replacement is `from langchain.agents import create_agent`, which needs the separate `langchain` package (not installed here). The warning is safe to ignore until LangGraph 2.0.
- **Invoking a tool in code:** `check_thermal_tool.invoke({"temperature_c": 30.0, "duration_min": 1})` validates the arguments against the schema and returns the function's Python result unchanged. `tool.args` lists the argument names, which is how `run_tool` picks them out of `state["inputs"]`.
- **State management:** in `create_react_agent`, LangGraph keeps a `messages` list in the conversation state. When the model decides it needs a tool, it appends an `AIMessage` with `tool_calls`. The Tool node runs the function and appends the output as a `ToolMessage`. The whole sequence is sent back to the LLM, which writes the final reply. This applies to Template B only. The check pipeline has no `messages`; its state is the explicit `RaceState` TypedDict, each node returns the keys it updates, and routing functions read those keys.
- **`ChatOllama(...)`:** the only way to create the LLM in this repo. It needs a running `ollama serve` and a model that has been pulled.

## Verification Checklist
Before finishing any agent change:
- [ ] `grep -rnE "openai|anthropic" src/` finds no hosted LLM clients
- [ ] No threshold number appears in a prompt string or in agent code
- [ ] Test AG-1: with the LLM mocked to say "PROCEED" on S1-1 inputs, the verdict is still `DO_NOT_PROCEED`
- [ ] Test AG-2: with Ollama stopped or timing out, the graph still returns the `compute_verdict` result using template messages
- [ ] Test AG-3: every failure includes the recorded value(s) and threshold(s)
- [ ] A stage that passes makes no LLM call; each failed record makes at most one
- [ ] `uv run pytest` passes

## Prohibited Behaviors
- **NEVER** let an LLM output set `verdict`, `failures`, or a routing decision.
- **NEVER** let the LLM call a pipeline check tool or fill in its arguments, or read a check result from LLM text.
- **NEVER** wrap a pipeline stage in `create_react_agent`; the node invokes its tool in code (spec/04-agents.md §3.3).
- **NEVER** call a hosted LLM API.
- **NEVER** hard-code thresholds in prompts or tool docstrings. Fetch them at runtime.
