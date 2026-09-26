# 02 — Safety Rules (Stages 1 & 2)

These rules tell you whether the battery is dangerous **right now**. All thresholds come from `thresholds.yaml`. The values shown here are the defaults.

## Rule and reason-code index

| Rule | Reason code | Stage | Reason text |
|------|-------------|-------|-------------|
| R1 | `OVERCHARGED` | 1 | Battery is already overcharged |
| R2 | `HIGH_DTDT` | 2 | Thermal stress from battery charging is at high risk |
| R3 | `HIGH_D2TDT2` | 2 | Heat is increasing at a very fast rate while the battery charges |
| R4 | `HIGH_TCHEM` | 2 | Heat exposure to the surrounding environment is too high |
| R5 | `LOW_EFFICIENCY` | 3 | Predicted efficiency is too low, so the car cannot proceed into the race (see [03-ml-model.md](03-ml-model.md)) |
| R6 | `MODEL_NOT_VALIDATED` | 3 | Efficiency model not validated (see [03-ml-model.md](03-ml-model.md)) |

## Stage 1 — Overcharge check

| Rule | Condition for FAIL |
|------|--------------------|
| **R1** `OVERCHARGED` | `I < 0 A` **AND** `V > 4.2 V` |

- Both conditions must be true for the rule to fail.
- On failure: **STOP**. Stages 2 and 3 do not run, and the verdict is `DO NOT PROCEED`.

| Example | V | I | Result |
|---------|---|---|--------|
| Overcharged | 5.0 | −1.2 | FAIL |
| High voltage, positive current | 5.0 | 0.5 | pass (moves to Stage 2) |
| Negative current, normal voltage | 3.9 | −1.2 | pass |
| Exactly at threshold | 4.2 | −1.2 | pass (strict `>`) |

## Stage 2 — Thermal hazard checks ("fire hazard now")

### 2.1 Variables and derived quantities

- `T_t`: the battery temperature entered by the user (°C), measured after charging for time `t`
- `T0 = 25 °C`: the theoretical battery temperature at `t = 0`
- `t`: the charging duration entered by the user, converted to **seconds** (`minutes × 60`). Must be > 0.
- `ΔT = T_t − T0`

The derived quantities are:

| Quantity | Formula | Unit | Meaning |
|----------|---------|------|---------|
| Heating rate | `dT/dt = ΔT / t` | °C/s | Thermal stress from charging |
| Heating acceleration | `d²T/dt² = ΔT / t²` | °C/s² | How fast the heating rate itself is rising (very fast charging) |
| Chemical heat exposure | `T_chem = max(0, T_t − T0)` | °C | Heat exposure above the initial state |

A negative `ΔT` (battery cooler than 25 °C) gives negative rates and `T_chem = 0`, so all Stage 2 rules pass.

### 2.2 Rules

| Rule | Condition for FAIL | Reason text |
|------|--------------------|-------------|
| **R2** `HIGH_DTDT` | `dT/dt > 0.03 °C/s` | Thermal stress from battery charging is at high risk |
| **R3** `HIGH_D2TDT2` | `d²T/dt² > 0.0005 °C/s²` | Heat is increasing at a very fast rate while the battery charges |
| **R4** `HIGH_TCHEM` | `T_chem > 0.3 °C` | Heat exposure to the surrounding environment is too high |

- All three rules are evaluated (FR-2). If **any** one fails, the verdict is `DO NOT PROCEED`, and every failed rule is listed.
- Comparisons are strict (`>`). A value exactly equal to the threshold passes.
- **Floating-point handling:** before comparison, round each calculated value to 9 decimal places. Without this, `25.3 − 25` evaluates to `0.30000000000000071` and would wrongly fail R4 at the boundary. Rounding for display is separate: 4 significant figures for rates, 2 decimal places for temperatures.

### 2.3 Worked example

Input: `T_t = 40 °C`, duration `5 min`, so `t = 300 s`.

| Check | Calculated | Fails when | Result |
|-------|-----------|------------|--------|
| dT/dt | (40 − 25) / 300 = **0.05 °C/s** | > 0.03 | FAIL (R2) |
| d²T/dt² | 0.05 / 300 = **0.000167 °C/s²** | > 0.0005 | pass |
| T_chem | max(0, 15) = **15 °C** | > 0.3 | FAIL (R4) |

Verdict: **DO NOT PROCEED**, with two reasons listed (R2 `HIGH_DTDT`, R4 `HIGH_TCHEM`).

## Check result record

Every rule (R1–R6) returns the same record. The UI and the agents both use it:

```
{ rule, code, passed, recorded: {name: value, ...}, recorded_kind: "input" | "calculated",
  thresholds: {name: "<op> <value>", ...}, unit, reason }
```

Message templates for each rule are in [05-ui.md](05-ui.md) §3.1.

## Evaluation order (pseudo-code)

```python
def evaluate(V, I, T_t, minutes, cfg, model):
    t = minutes * 60
    failures = []

    # Stage 1
    if I < cfg.overcharge.current_max_a and V > cfg.overcharge.voltage_max_v:
        return Verdict.DO_NOT_PROCEED, [R1(V, I)]          # short-circuit

    # Stage 2 — evaluate all
    dT = T_t - cfg.thermal.initial_temp_c
    rate = round(dT / t, 9)
    accel = round(dT / t**2, 9)            # d²T/dt² = ΔT / t²
    tchem = round(max(0.0, dT), 9)
    if rate  > cfg.thermal.dT_dt_max_c_per_s:     failures.append(R2(rate))
    if accel > cfg.thermal.d2T_dt2_max_c_per_s2:  failures.append(R3(accel))
    if tchem > cfg.thermal.t_chem_max_c:          failures.append(R4(tchem))
    if failures:
        return Verdict.DO_NOT_PROCEED, failures

    # Stage 3 — see 03-ml-model.md
    if not model.gates_passed:
        return Verdict.DO_NOT_PROCEED, [R6(model.metrics)]
    eff = model.predict(V, I, T_t, minutes)
    if eff < cfg.efficiency.min_percent:
        return Verdict.DO_NOT_PROCEED, [R5(eff)]

    return Verdict.CAN_PROCEED, []
```
