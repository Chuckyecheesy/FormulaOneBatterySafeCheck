# 01 — Requirements

## 1. Inputs

The operator enters exactly four values:

| # | Field | Symbol | Unit (entered) | Example | Internal unit | Conversion |
|---|-------|--------|----------------|---------|---------------|------------|
| 1 | Battery voltage | `V` | V | `0`, `4.1`, `5` | V | none |
| 2 | Battery current | `I` | A | `0`, `0.5`, `-1.2` | A | none (negative allowed) |
| 3 | Battery temperature | `T_t` | °C | `25`, `33.4` | °C | none |
| 4 | Charging duration | `t` | **minutes** | `60` | **seconds** | `t = minutes × 60` |

- `T_t` is the battery temperature entered by the user. It is the temperature measured after the battery has been charging for time `t`.
- `t` (in seconds) is used by every rate calculation in [02-safety-rules.md](02-safety-rules.md).
- The efficiency model in [03-ml-model.md](03-ml-model.md) takes duration in **minutes**, because the training data is stored in minutes.

### 1.1 Validation (applied before any rule runs)

- **IN-1** All four fields are required and must be finite numbers. Empty, non-numeric, `NaN` and infinite values are rejected. Decimals are allowed, and negative values are allowed for every field except duration.
- **IN-2** Charging duration must be **>= 0**. `t = 0` makes dT/dt undefined, so the form rejects it with the message *"Charging duration must be greater than 0 minutes."*
- **IN-3** Values outside the physical sanity limits in `thresholds.yaml → input_limits` are rejected with a field-level error. These limits catch typos. They are not hazard thresholds.
- **IN-4** Invalid input never produces a verdict. The UI shows the field errors and no result.

## 2. Outputs

- **OUT-1** A verdict: `CAN PROCEED` or `DO NOT PROCEED`.
- **OUT-2** For `DO NOT PROCEED`: one entry per failed rule, each containing:
  - a plain-language reason (rule IDs and reason codes are internal and not displayed)
  - the recorded value(s), marked as input or calculated, with units
  - the hazard threshold(s), including the comparison operator

## 3. Functional requirements

- **FR-1** Stage 1 (overcharge) runs first. If it fails, stop: Stages 2 and 3 do not run, and the verdict is `DO NOT PROCEED`.
- **FR-2** Stage 2 (thermal) evaluates **all three** thermal rules, even after one fails, so that every failure is reported together.
- **FR-3** Stage 3 (efficiency prediction) runs only if Stages 1 and 2 pass.
- **FR-4** If the model has not passed its quality gates (see [03-ml-model.md](03-ml-model.md) §5), Stage 3 fails **closed**: the verdict is `DO NOT PROCEED`, with rule R6.
- **FR-5** `CAN PROCEED` is returned only when every stage passes.
- **FR-6** All thresholds are loaded from `thresholds.yaml`. No threshold appears as a literal in code or prompts.
- **FR-7** Every check is logged with its inputs, calculated values, stage results, model version, and timestamp, so that each decision can be audited.
- **FR-8** The final verdict is computed **deterministically in code**. AI agents explain and summarise it, but they cannot change it (see [04-agents.md](04-agents.md)).

## 4. Non-functional requirements

- **NFR-1** End-to-end response in ≤ 2 s without agent narration and ≤ 10 s with it.
- **NFR-2** If the agent layer is unavailable (network or API error), the system still returns the deterministic verdict and a template-generated reason.
- **NFR-3** Identical inputs and model version always produce the identical verdict.
