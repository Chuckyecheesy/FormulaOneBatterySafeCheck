# 07 — Decisions & Open Questions

## Q3 — How d²T/dt² is calculated (RESOLVED)
**Decided:** `d²T/dt² = ΔT / t²`, where `ΔT = T_t − T0` and `t` is the charging duration in seconds.

## Q4 — Tolerance δ (RESOLVED)
**Decided:** δ = **0.1 percentage points**. Efficiency only varies by about 2.4 pp in the data (std 0.54 pp), so this is a tight band. Combined with the > 80 % gate, the model must predict most test rows within ±0.1 pp.

## Q9 — Tech stack (LOW)
Agents: **decided**, LangGraph with a local Ollama model (see [04-agents.md](04-agents.md) §3). Rest of the stack is a proposal: Python (FastAPI + xgboost + scikit-learn for metrics) backend and a lightweight web frontend. Note that `xgboost` and `scikit-learn` are not currently installed in this environment.
