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
3. **Each stage is a `create_react_agent` that calls its check through `@tool`, but the LLM cannot change the result.** Stage tools take **no arguments** (they read the inputs from graph state), the node reads the result from the tool's **artifact** (never from LLM text), and if the LLM skips the tool or Ollama fails, the node runs the check in code anyway.
4. **Routing and verdict read `stage_results`, never LLM text.** The verdict comes from `compute_verdict`.
5. **No threshold literals in code or prompts.** Read them from `spec/thresholds.yaml` via `load_thresholds()` and pass them into the prompt as data.
6. **Always have a fallback.** If Ollama is down, errors, or takes longer than 8 s, use the template message from [spec/05-ui.md](../../../spec/05-ui.md) §3.1. The graph still finishes with the computed verdict.

## Prerequisites & Model Choice
Run an Ollama version and an open-weight model that **natively supports tool calling** (function calling). Models without tool calling return plain text instead of properly formatted tool requests, and a tool-calling agent loop (Template B) breaks.

- **Recommended models:** `llama3.2` (the project default, 3B), `llama3.1:8b` (larger, slower), or `qwen2.5:7b`.
- Both templates use tool calling. If a model skips the tool, Template A still runs the check in code and uses the template message, so the verdict is never affected, only the explanation.

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
One `StateGraph` with one node per agent. Each stage node (1–3) builds a `create_react_agent` with a single `@tool` that wraps the stage's deterministic check. The agent calls the tool and explains the result. Three safeguards keep the decision in code:

- **No tool arguments.** The tool reads `state["inputs"]` through a closure, so the LLM cannot pass made-up values.
- **Results come from the artifact.** `response_format="content_and_artifact"` gives the LLM the JSON as `content` and gives the node the raw records as `ToolMessage.artifact`. Only the artifact goes into `stage_results`.
- **The check always runs.** If the LLM never calls the tool, Ollama is down, or the loop hits `recursion_limit`, the node calls the tool itself and uses the §3.1 template message.

The Race Decision node has no agent: `verdict`, `failures` and `summary` come from `compute_verdict` and `template_message`, and the summary states every failed check reason. `compute_verdict`, `evaluate_prediction` and `template_message` are the deterministic helpers in `src/formulatech/agents.py`.

```python
import json
import warnings
from dataclasses import asdict
from typing import Any, Callable, TypedDict

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from formulatech.agents import compute_verdict, evaluate_prediction, template_message
from formulatech.config import load_thresholds
from formulatech.rules import check_overcharge, check_thermal

warnings.filterwarnings("ignore", message=".*create_react_agent.*")  # deprecated in LangGraph 1.x, still supported

cfg = load_thresholds()
llm = ChatOllama(model="llama3.2", temperature=0, num_predict=150, client_kwargs={"timeout": 8})

STAGE_PROMPT = (
    "You are one stage of a pre-race battery safety check. Call your check tool exactly once. "
    "Then explain the tool result to a race engineer in one or two sentences. Do not change the "
    "result. Include the recorded value(s) and threshold(s). Do not mention rule IDs or reason codes."
)


class CheckState(TypedDict):
    inputs: dict[str, float]             # voltage_v, current_a, temperature_c, duration_min
    stage_results: list[dict[str, Any]]  # tool artifacts, appended by each node
    explanations: list[str]              # LLM text only; never read back into the verdict
    verdict: str | None                  # set only by the decision node, from compute_verdict
    failures: list[dict[str, Any]]
    summary: str


def stage_tool(name: str, description: str, check: Callable[[], list[dict]]):
    """Wrap a deterministic check as a no-argument @tool, so the LLM cannot change the inputs.

    content_and_artifact: the LLM sees the JSON content; the node reads the artifact.
    """
    @tool(name, description=description, response_format="content_and_artifact")
    def run() -> tuple[str, list[dict]]:
        results = check()
        return json.dumps(results, default=str), results
    return run


def run_stage_agent(state: CheckState, check_tool, task: str) -> dict:
    """One create_react_agent per stage: the agent calls the tool and explains the result."""
    results, text = None, ""
    try:
        agent = create_react_agent(llm, tools=[check_tool], prompt=STAGE_PROMPT)
        out = agent.invoke({"messages": [("user", task)]}, {"recursion_limit": 6})
        tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage) and m.name == check_tool.name]
        if tool_msgs:
            results = tool_msgs[-1].artifact            # the tool's own output, not LLM text
            text = str(out["messages"][-1].content).strip()
    except Exception:                                  # Ollama down, error, timeout, recursion limit
        pass
    if results is None:                                # LLM skipped the tool: run the check in code anyway
        results = check_tool.invoke({"type": "tool_call", "id": "fallback", "name": check_tool.name,
                                     "args": {}}).artifact
        text = ""
    failed = [r for r in results if not r["passed"]]
    if failed:  # nothing to explain on a pass; template fallback if the LLM gave no text (NFR-2)
        text = text or " ".join(template_message(r, state["inputs"], cfg) for r in failed)
    update = {"stage_results": state["stage_results"] + results}
    if failed:
        update["explanations"] = state["explanations"] + [text]
    return update


# ---- Agent 1: Overcharge (stage 1) ----
def overcharge_node(state: CheckState) -> dict:
    i = state["inputs"]
    check_tool = stage_tool(
        "check_overcharge",
        "Run the overcharge check on the recorded battery voltage and current. Takes no arguments.",
        lambda: [asdict(check_overcharge(i["voltage_v"], i["current_a"], cfg))],
    )
    return run_stage_agent(state, check_tool, "Check whether the battery is already overcharged.")


# ---- Agent 2: Thermal (stage 2) — the tool always runs all three rules (FR-2) ----
def thermal_node(state: CheckState) -> dict:
    i = state["inputs"]
    check_tool = stage_tool(
        "check_thermal",
        "Run the thermal checks (dT/dt, d²T/dt², T_chem) on the recorded battery temperature and "
        "charging duration. Takes no arguments.",
        lambda: [asdict(r) for r in check_thermal(i["temperature_c"], i["duration_min"], cfg)],
    )
    return run_stage_agent(state, check_tool, "Check whether the battery has a thermal hazard right now.")


# ---- Agent 3: Prediction (stage 3) — R6 if the model is not validated, else R5 ----
def prediction_node(state: CheckState) -> dict:
    check_tool = stage_tool(
        "check_efficiency",
        "Check the efficiency model's quality gates, then predict efficiency for the recorded "
        "reading and compare it with the minimum. Takes no arguments.",
        lambda: [evaluate_prediction(state["inputs"], cfg)],
    )
    return run_stage_agent(state, check_tool, "Check the predicted efficiency for this race.")


# ---- Agent 4: Race Decision — deterministic, no LLM for verdict/failures/summary ----
def decision_node(state: CheckState) -> dict:
    result = compute_verdict(state["stage_results"])
    if result["verdict"] == "CAN_PROCEED":
        summary = "CAN PROCEED into the race."
    else:  # state every failed check reason
        summary = "DO NOT PROCEED. " + " ".join(
            template_message(f, state["inputs"], cfg) for f in result["failures"])
    return {**result, "summary": summary}


# ---- Routing reads stage_results, never LLM output ----
def any_failed(results: list[dict]) -> bool:
    return any(not r["passed"] for r in results)


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

Nodes return **partial updates** (a dict with only the keys they change). LangGraph merges them into the state. Each stage makes at least two LLM calls (tool call, then explanation), so keep `num_predict` short to stay within NFR-1.

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
- **`create_react_agent`** (from `langgraph.prebuilt`): replaces building the graph by hand for a standard tool loop. It sets up a `StateGraph` with an **Agent node** (the model decides what to do next) and a **Tool node** (`ToolNode`, which runs the chosen function), joined by a conditional edge. The loop is: if the model's last message has tool calls, go to the Tool node, then back to the Agent node; otherwise stop. In Template A each stage node wraps one `create_react_agent` with one no-argument tool, and the node, not the agent, writes `stage_results` from the tool artifact. In Template B the agent can pick freely among read-only tools. In LangGraph 1.x it still works but gives a `LangGraphDeprecatedSinceV10` warning. The replacement is `from langchain.agents import create_agent`, which needs the separate `langchain` package (not installed here). The warning is safe to ignore until LangGraph 2.0.
- **`response_format="content_and_artifact"`:** the tool returns `(content, artifact)`. The LLM only sees `content`; the `ToolMessage.artifact` keeps the exact Python result for code to read. This is how a stage node gets the check result without trusting LLM text.
- **State management:** in `create_react_agent`, LangGraph keeps a `messages` list in the conversation state. When the model decides it needs a tool, it appends an `AIMessage` with `tool_calls`. The Tool node runs the function and appends the output as a `ToolMessage`. The whole sequence is sent back to the LLM, which writes the final reply. In Template A each stage agent's `messages` stay inside its node; the outer graph state is the explicit `CheckState` TypedDict: each node returns the keys it updates, and routing functions read those keys.
- **`ChatOllama(...)`:** the only way to create the LLM in this repo. It needs a running `ollama serve` and a model that has been pulled.

## Verification Checklist
Before finishing any agent change:
- [ ] `grep -rnE "openai|anthropic" src/` finds no hosted LLM clients
- [ ] No threshold number appears in a prompt string or in agent code
- [ ] Test AG-1: with the LLM mocked to say "PROCEED" on S1-1 inputs, the verdict is still `DO_NOT_PROCEED`
- [ ] Test AG-2: with Ollama stopped or timing out, the graph still returns the `compute_verdict` result using template messages
- [ ] Test AG-3: every failure includes the recorded value(s) and threshold(s)
- [ ] With the LLM mocked to never call the tool, every stage still runs its check and the verdict is unchanged
- [ ] `uv run pytest` passes

## Prohibited Behaviors
- **NEVER** let an LLM output set `verdict`, `failures`, or a routing decision.
- **NEVER** give a stage tool arguments the LLM fills in, or read a check result from LLM text instead of the tool artifact.
- **NEVER** let a stage end without its check result: if the agent didn't call the tool, run it in code.
- **NEVER** call a hosted LLM API.
- **NEVER** hard-code thresholds in prompts or tool docstrings. Fetch them at runtime.
