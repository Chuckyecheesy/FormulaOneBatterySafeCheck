# Specification: Pre-Race Battery Safety Check

## Purpose

Before a race, the team enters a battery reading. The system decides whether the car can **proceed into the race** or **must not proceed**. When the answer is no, it explains why, giving the threshold and the recorded value for every failed check.

It answers two questions:

1. **Is the battery dangerous right now?** Checks whether it is already overcharged, and whether it has a fire risk from heat build-up.
2. **Will the battery be dangerous during the race?** An XGBoost model predicts charging efficiency. Low efficiency means a likely fire during the race if the car starts now.

## Decision pipeline

```
 User input: V, I, T_t, duration (min)
        │
        ▼
 Convert duration → t (seconds)
        │
        ▼
 Stage 1: Overcharge check ── I < 0 A AND V > 4.2 V ──► DO NOT PROCEED (stop)
        │ pass
        ▼
 Stage 2: Thermal checks (evaluate all three)
   • dT/dt    > 0.03  °C/s
   • d²T/dt²  > 0.0005 °C/s²      ── any true ──► DO NOT PROCEED (list every failure)
   • T_chem   > 0.3   °C
        │ all pass
        ▼
 Stage 3: XGBoost efficiency prediction
   (model must meet the acceptance gates: tolerance accuracy, R², good fit)
   • efficiency < 70 %            ──────────────► DO NOT PROCEED
        │ pass
        ▼
 CAN PROCEED INTO RACE
```

## Files

| File | Contents |
|------|----------|
| [01-requirements.md](01-requirements.md) | Inputs, validation, outputs, functional and non-functional requirements |
| [02-safety-rules.md](02-safety-rules.md) | Stage 1 (overcharge) and Stage 2 (thermal) rules, formulas, reason codes |
| [03-ml-model.md](03-ml-model.md) | Stage 3: the XGBoost model, training data, and quality gates |
| [04-agents.md](04-agents.md) | The four AI agents and how they hand off to each other |
| [05-ui.md](05-ui.md) | Input form, result screens, and failure message templates |
| [06-test-cases.md](06-test-cases.md) | Test cases for every stage, the model, and the agents |
| [07-open-questions.md](07-open-questions.md) | Decisions on d²T/dt², tolerance δ, and the tech stack |
| [thresholds.yaml](thresholds.yaml) | Every threshold, read by code at runtime |

Thresholds and rule-logic for agents are listed in [../CLAUDE.md](../CLAUDE.md) for reference. Code reads them from [thresholds.yaml](thresholds.yaml).
