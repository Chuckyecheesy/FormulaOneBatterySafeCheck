# FormulaTech — Pre-Race Battery Safety Check

The app takes a battery reading (voltage, current, temperature, charging duration, state of charge) and decides whether the car can **proceed into the race** or **must not proceed**. When the answer is no, it gives every reason, with the threshold and the recorded value for each. Full specification: [spec/README.md](spec/README.md).

## Hazard thresholds (single source of truth)

Code must read these values from [spec/thresholds.yaml](spec/thresholds.yaml), which mirrors this table. Never hard-code them anywhere else. If you change a value, update this table, `thresholds.yaml`, and the spec in the same change.

| ID | Rule / reason code | Quantity | Unit | Fails when | Threshold |
|----|-------|----------|------|------------|-----------|
| `OVERCHARGE` | R1 `OVERCHARGED` (stage 1) | Battery current `I` and battery voltage `V` | A, V | `I < 0` **and** `V > 4.2` (both must be true) | `I < 0 A` and `V > 4.2 V` |
| `T0` | Thermal baseline | Battery temperature at t = 0 | °C | — (constant) | `25` |
| `DTDT_MAX` | R2 `HIGH_DTDT` (stage 2) | `dT/dt` | °C/s | `dT/dt > 0.03` | `0.03` |
| `D2TDT2_MAX` | R3 `HIGH_D2TDT2` (stage 2) | `d²T/dt² = ΔT / t²` | °C/s² | `d²T/dt² > 0.0005` | `0.0005` |
| `TCHEM_MAX` | R4 `HIGH_TCHEM` (stage 2) | `T_chem = max(0, T_t − T0)` | °C | `T_chem > 0.3` | `0.3` |
| `EFFICIENCY_MIN` | R5 `LOW_EFFICIENCY` (stage 3) | XGBoost predicted efficiency | % | `efficiency < 70` | `70` |

## Model acceptance thresholds (XGBoost efficiency model)

A trained model may only be used for stage 3 if it passes all three accuracy gates: tolerance accuracy and R² on the held-out test set, and 5-fold CV R² on the train split:

| ID | Metric | Passes when | Threshold |
|----|--------|-------------|-----------|
| `TOL_DELTA` | Tolerance band δ for tolerance accuracy | — | `0.1` percentage points |
| `TOL_ACC_MIN` | Tolerance accuracy (δ) | `> 80 %` | `80` |
| `R2_MIN` | R² | `> 0.90` | `0.90` |
| `CV_R2_STD_MAX` | CV R²: 5-fold CV R² std (CV mean R² must also be > 0.90) | `≤ 0.03` | `0.03` |

Train R², the train − test R² gap and the early-stopping best iteration are reported for information only; they are not gates. There is no `fit_r2_gap_min` or `fit_r2_gap_max` gate, and no `good_fit` pass/fail label.

**Gate logic depends only on R² test, CV R² and tolerance accuracy:**

```
gates_passed = (tolerance_accuracy > 80)
           and (R²_test > 0.90)
           and (CV_R²_mean > 0.90 and CV_R²_std ≤ 0.03)
```

No other metric can pass or fail the model.

If any gate fails, stage 3 returns R6 `MODEL_NOT_VALIDATED` and the verdict is DO NOT PROCEED.

## Rule logic

The checks answer two questions:

- **Fire hazard now (stages 1–2):** `OVERCHARGE`, `DTDT_MAX`, `D2TDT2_MAX` and `TCHEM_MAX` check whether the battery is a fire hazard at the moment of the reading.
- **Fire hazard during the race (stage 3):** `EFFICIENCY_MIN` checks the predicted efficiency to decide whether the battery is likely to become a fire hazard during the race if the car starts now.

1. **Check `OVERCHARGE` first.** If `I < 0 A` and `V > 4.2 V` → **DO NOT PROCEED**. Stop here and state failed check reason.
2. **If overcharge is clean (pass), check `DTDT_MAX`, `D2TDT2_MAX` and `TCHEM_MAX`.** Evaluate all three. If any fails → **DO NOT PROCEED**, listing every failed check. Stop here.
3. **If all three are clean, check the prediction against `EFFICIENCY_MIN`.** (The model must first pass its acceptance thresholds above; otherwise → **DO NOT PROCEED**.) The model predicts efficiency from the user's input from the UI. If `efficiency < EFFICIENCY_MIN` → **DO NOT PROCEED**. Stop here and state failed check reason.
4. **Otherwise → CAN PROCEED** into the race.

## Rules for working in this repo

- Comparisons are strict (`>` and `<`), exactly as written above. A value equal to a threshold passes. Round calculated values to 9 decimal places before comparing, to avoid floating-point errors at the boundary.
- Charging duration is entered in **minutes** and converted to **seconds** (`t = minutes × 60`) before any rate calculation.
- The pass/fail decision is made by deterministic code. AI agents call that code and explain its result. An LLM must never decide a threshold comparison on its own.
- AI agents are built with **LangGraph** (one `StateGraph`, one node per agent) and a local **Ollama** model via `langchain-ollama`. Do not use a hosted LLM API. See [spec/04-agents.md](spec/04-agents.md).
- Dataset: `ev_battery_charging_data.csv` (1000 rows). Target column: `Efficiency (%)`.

# Claude Instructions

- When asked to explain code, always provide a detailed, line-by-line breakdown.
- Break down nested loops, conditions, and callbacks explicitly.
