# 06 — Test Cases

`t = minutes × 60`, `ΔT = T − 25`, `dT/dt = ΔT/t`, `d²T/dt² = ΔT/t²`, `T_chem = max(0, ΔT)`. Calculated values are rounded to 9 decimal places before comparison.

## Stage 1

| ID | V | I | T | min | Expected | Notes |
|----|---|---|---|-----|----------|-------|
| S1-1 | 5.0 | -1.2 | 25 | 60 | DO NOT PROCEED — R1 | The example from the brief. Stages 2 and 3 not run |
| S1-2 | 4.21 | -0.01 | 25 | 60 | DO NOT PROCEED — R1 | Just past both limits |
| S1-3 | 4.2 | -1.0 | 25 | 60 | R1 passes | V is exactly 4.2, and the rule is strict `>` |
| S1-4 | 5.0 | 0.0 | 25 | 60 | R1 passes | I is exactly 0, and the rule is strict `<` |
| S1-5 | 5.0 | 0.5 | 25 | 60 | R1 passes | High V with positive current is not overcharge under R1 |
| S1-6 | 3.8 | -2.0 | 25 | 60 | R1 passes | Negative current at low V |

## Stage 2

| ID | T | min | t (s) | dT/dt | d²T/dt² | T_chem | Expected |
|----|---|-----|-------|-------|---------|--------|----------|
| S2-1 | 25.0 | 60 | 3600 | 0 | 0 | 0 | all pass → go to Stage 3 |
| S2-2 | 25.3 | 60 | 3600 | 8.3e-5 | 2.3e-8 | 0.30 | pass (T_chem is exactly 0.3 after rounding to 9 dp; strict `>`) |
| S2-3 | 25.31 | 60 | 3600 | 8.6e-5 | 2.4e-8 | 0.31 | **R4 fail** |
| S2-4 | 30.0 | 1 | 60 | 0.0833 | 1.39e-3 | 5.0 | **R2 + R3 + R4 fail** (all three listed) |
| S2-5 | 20.0 | 60 | 3600 | −0.0014 | −3.9e-7 | 0 | all pass (battery cooler than T0) |
| S2-6 | 25.2 | 0.1 | 6 | 0.0333 | 0.00556 | 0.2 | **R2 + R3 fail**, R4 passes |

## Stage 3

| ID | Setup | Expected |
|----|-------|----------|
| S3-1 | Model gates passed; prediction 98.2 % | CAN PROCEED |
| S3-2 | Model gates passed; prediction 65 % (mocked) | DO NOT PROCEED — R5, recorded 65 %, threshold < 70 % |
| S3-3 | Model gates passed; prediction exactly 70.0 % | CAN PROCEED (strict `<`) |
| S3-4 | The model cannot predict (missing model file, or a NaN/∞ prediction) | Error, no verdict (never CAN PROCEED). Model gates are not checked at runtime |

## Input validation

| ID | Input | Expected |
|----|-------|----------|
| IV-1 | duration = 0 | Field error; no verdict |
| IV-2 | duration = -5 | Field error |
| IV-3 | voltage = "abc" | Field error |
| IV-4 | temperature empty | Submit disabled |

## Model quality (training pipeline)

| ID | Check |
|----|-------|
| MQ-1 | `Degradation Rate (%)` is not in the feature list |
| MQ-2 | Tolerance accuracy is computed exactly per the formula (unit test with a hand-computed fixture) |
| MQ-3 | R² matches `sklearn.metrics.r2_score` on a fixture |
| MQ-4 | The train − test R² gap is reported but never fails a gate (e.g. gap 0.002 with every other gate passing → `gates_passed = true`) |
| MQ-4a | Best iteration is reported but never fails a gate (e.g. best iteration `n_estimators − 1` with the three accuracy gates passing → `gates_passed = true`) |
| MQ-5 | The test split is not touched during tuning (assert on indices) |

## Agent layer

| ID | Check |
|----|-------|
| AG-1 | With the LLM mocked to "say PROCEED" on S1-1 inputs, the displayed verdict is still DO NOT PROCEED and the mismatch is logged |
| AG-2 | With an LLM timeout, the template-based result is returned within NFR-1 |
| AG-3 | Every failure card includes the recorded value(s) and threshold(s) |
