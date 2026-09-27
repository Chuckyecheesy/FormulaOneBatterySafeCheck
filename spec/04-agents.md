# 04 — AI Agents

## 1. Principle: code decides, agents explain

Every threshold comparison runs in **deterministic code** (the tools below). Agents call those tools, read the results, and write the human-readable explanation. **No agent can turn a FAIL into a PASS.** The final verdict is the logical AND of the stage results computed in code (FR-8). If an agent's summary disagrees with the computed verdict, the computed verdict is shown and the mismatch is logged.

## 2. Agents

The four agents are nodes in a **LangGraph** `StateGraph`. Each node runs its deterministic tool in code, then asks a local **Ollama** model to explain the result. Conditional edges make the stops (short-circuits) explicit:

```
 START
   │
   ▼
 Agent 1: Overcharge ──R1 fail──────────────────────┐
   │ pass                                           │
   ▼                                                │
 Agent 2: Thermal ─────any of R2/R3/R4 fail─────────┤
   │ all pass                                       │
   ▼                                                ▼
 Agent 3: Prediction ─────────────────────► Agent 4: Race Decision ──► END
```

### Agent 1 — Overcharge Agent
- **Job:** check whether the battery is already overcharged (voltage and current).
- **Tool:** `check_overcharge(voltage_v, current_a) → {passed, rule: "R1", recorded: {V, I}, thresholds: {V: ">4.2", I: "<0"}}`
- **Output:** the tool result plus a one-sentence explanation.
- **Short-circuit:** if `passed = false`, the graph skips Agents 2 and 3 and goes straight to Agent 4.

### Agent 2 — Thermal Risk Agent
- **Job:** check for a very high risk of temperature increase (dT/dt, d²T/dt², T_chem).
- **Tool:** `check_thermal(temp_c, duration_min) → {passed, t_seconds, dT_dt, d2T_dt2, t_chem, failures: [R2|R3|R4 ...]}`
- **Short-circuit:** if any rule fails, the graph skips Agent 3 and goes straight to Agent 4.
- **Output:** the tool result plus an explanation of each failure, based on the recorded (calculated) value and its threshold value.

### Agent 3 — Prediction Agent
- **Job:** check the efficiency prediction result.
- **Tools:**
  - `get_model_status() → {version, gates_passed, tolerance_accuracy, r2, r2_train, cv_mean, cv_std}`
  - `predict_efficiency(voltage_v, current_a, temp_c, duration_min, soc_percent) → {efficiency_pct}`
- **Logic (in tool code):** if `gates_passed = false`, return R6 without predicting. Otherwise predict and apply R5.
- **Output:** the prediction, the model quality summary, and an explanation.

### Agent 4 — Race Decision Agent
- **Job:** decide whether to proceed into the race, and write the final message.
- **Input:** the stage results collected in the graph state by Agents 1–3.
- **Tool:** `compute_verdict(stage_results) → {verdict, failures}`, which is deterministic.
- **Output schema:**
  ```json
  {
    "verdict": "CAN_PROCEED" | "DO_NOT_PROCEED",
    "failures": [
      { "rule": "R1", "code": "OVERCHARGED", "reason": "...", "recorded": {"battery_voltage_v": 5.0, "battery_current_a": -1.2},
        "thresholds": {"battery_voltage_v": "> 4.2", "battery_current_a": "< 0"} }
    ],
    "calculated": { "t_seconds": 3600, "dT_dt": 0.0, "d2T_dt2": 0.0, "t_chem": 0.0, "efficiency_pct": 98.1 },
    "summary": "Plain-language explanation for the driver/engineer"
  }
  ```
- `verdict` and `failures` are copied from `compute_verdict`, not written by the model.
- `comment` is a fixed sentence chosen in code by the failed stage (05-ui.md §3.2), not written by the model. It is empty on CAN PROCEED and when the efficiency model is not validated (R6).

## 3. Implementation notes

### 3.1 Stack

- **Orchestration:** LangGraph (`langgraph`). One `StateGraph` with the four nodes above and `add_conditional_edges` for the two short-circuits.
- **LLM:** Ollama, running locally (`ollama serve`, default `http://localhost:11434`), called through `langchain-ollama` (`ChatOllama`).
- **Model:** default `llama3.2` (the 3B model). Configurable in settings (for example `llama3.1:8b` or `qwen2.5:7b`). The model is used only to write explanations, so it doesn't need to be reliable at tool calling.
- Everything runs on the local machine. No battery data leaves it.

### 3.2 Graph state

```python
class CheckState(TypedDict):
    inputs: dict              # V, I, T_t, duration_min, t_seconds, soc_percent
    stage_results: list[dict] # check result records (02-safety-rules.md), appended by each node
    explanations: list[str]   # one per agent, written by the LLM
    verdict: str | None       # set only by Agent 4, from compute_verdict
    failures: list[dict]
    summary: str
    comment: str              # fixed risk comment for the failed stage (05-ui.md §3.2), set by Agent 4
```

### 3.3 Node pattern

Every agent node follows the same two steps:

1. **Call the tool in code** (for example `check_thermal(...)`) and append its result to `stage_results`. The LLM never makes this call and never sees raw inputs without the tool result.
2. **Ask Ollama to explain** the tool result, based on the recorded value and its threshold. The prompt contains the tool result as JSON, with thresholds filled in from `thresholds.yaml`. The LLM's text goes into `explanations` only. It is never read back into `verdict` or `failures`. The prompt tells the model not to mention rule IDs or reason codes, since the explanation is returned by the API in plain language. The web UI does not display it (see [05-ui.md](05-ui.md) §4).

Routing functions for the conditional edges read `stage_results`, not LLM output.

### 3.4 Latency and fallback

- Up to four local LLM calls per check. To stay within NFR-1 (≤ 10 s), keep explanations short (`num_predict` about 150 tokens) and skip the LLM call for a stage that passed with nothing to explain.
- **Fallback (NFR-2):** if Ollama is not running, errors, or a call times out (8 s), the node uses the template message from [05-ui.md](05-ui.md) §3.1 instead. The graph still completes and returns the `compute_verdict` result.
- Prompts must not contain threshold literals. Thresholds are passed in from `thresholds.yaml` at runtime so that prompts and code stay in sync.
