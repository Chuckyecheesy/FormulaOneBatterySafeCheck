# 03 — Efficiency Prediction Model (Stage 3)

## 1. Purpose

Predict charging **efficiency (%)** from the operator's inputs. A prediction below **70%** means the battery is likely to cause a fire during the race if it starts now, so the verdict is `DO NOT PROCEED`.

| Rule | Condition for FAIL | Reason text |
|------|--------------------|-------------|
| **R5** `LOW_EFFICIENCY` | `efficiency_pred < 70 %` | Predicted efficiency is too low, so the car cannot proceed into the race |
| **R6** `MODEL_NOT_VALIDATED` | model failed any quality gate in §5 | Efficiency model not validated, so the race cannot be cleared |

Message templates are in [05-ui.md](05-ui.md) §3.1.

## 2. Dataset

- Source: `ev_battery_charging_data.csv` (1,000 rows).
- Observed ranges:

  | Column | Min | Max |
  |--------|-----|-----|
  | Voltage (V) | 3.50 | 4.20 |
  | Current (A) | 10.0 | 99.8 |
  | Battery Temp (°C) | 20.0 | 40.0 |
  | Charging Duration (min) | 20.6 | 119.9 |
  | SOC (%) | 10.4 | 100.0 |
  | Efficiency (%) | 96.79 | 99.18 (mean 98.00, std 0.54) |

- Target: `Efficiency (%)`.
- **Features:** `SOC (%)`, `Voltage (V)`, `Current (A)`, `Battery Temp (°C)`, `Charging Duration (min)`, in this order. These are the five values the operator enters, so all are available at inference time.
- **Excluded columns:**
  - `Degradation Rate (%)` has a correlation of **−1.00** with efficiency. It is a direct leak of the target and must never be a feature.
  - `Optimal Charging Duration Class`, `Ambient Temp`, `Charging Mode`, `Battery Type`, `Charging Cycles`, `EV Model` are not entered by the operator, so they can't be used unless the UI adds them.
- Units: the model is trained on duration in **minutes**, as stored in the CSV. The inference path must pass minutes, not the converted seconds.

## 3. Data split

- **80% train / 20% test.** The test set is held out and used once per model version.
- Early stopping needs a validation set. It is taken from **inside** the 80% training split (for example, the held-out fold during 5-fold CV), so the 20% test set stays untouched.
- `random_state` fixed and recorded with the model artifact.
- The test set is never used for tuning.

## 4. Training

- Library: `xgboost` (`XGBRegressor`), `objective="reg:squarederror"`.
- Starting hyperparameters: `n_estimators=10000` with `early_stopping_rounds=100` on a validation slice of the training split, `max_depth=3–5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight≥5`, `reg_lambda=1`.
- Tuning: grid or random search scored by 5-fold CV R² on the train split only.
- Artifact saved with: model file, feature list and order, training data hash, hyperparameters, all metrics from §5, and the training timestamp.

## 5. Quality gates (all must pass before the model is used for decisions)

Metrics are computed on the **held-out test set** unless stated otherwise.

### 5.1 Tolerance accuracy

$$\text{Tolerance Accuracy}_{\delta} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left( |y_i - \hat{y}_i| \le \delta \right) \times 100$$

- **Gate:** Tolerance Accuracy > **80%**
- `δ` is in percentage points of efficiency. `δ = 0.1 pp`.

### 5.2 Coefficient of determination

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

- **Gate:** R² > **0.90** (the model explains more than 90% of the variance).

### 5.3 Fit checks (overfitting and underfitting)

Both checks below must pass:

| Check | Gate |
|-------|------|
| Cross-validation stability | 5-fold CV on the train split: mean R² > 0.90 **and** std ≤ 0.03 |
| Early stopping | Best iteration < `n_estimators` (validation loss actually plateaued) |

Underfitting is ruled out by the R² gate (§5.2) and the CV mean: a model that explains more than 90% of held-out variance is not underfitting. Overfitting is ruled out by the same R² gate, CV stability, and early stopping.

Train R² and the gap `R²_train − R²_test` are saved and reported **for information only**. There is no `fit_r2_gap_min` or `fit_r2_gap_max` gate and no `good_fit` label. A small gap simply means the model generalises well; with test R² > 0.90 the gap is expected to stay below 0.10.

A learning-curve plot (train vs validation RMSE by boosting round, with validation taken from the training split) is saved with each artifact for manual review.

### 5.4 Gate outcome

- All gates pass → `gates_passed = true`, and the model may be deployed.
- Any gate fails → `gates_passed = false`. The model can still be stored for analysis, but at runtime Stage 3 returns **R6** (fail closed, FR-4).

## 6. Inference

- Input order must match the stored feature list.
- Stage 3 returns the check result record from [02-safety-rules.md](02-safety-rules.md), plus `model_metrics`.

## 7. Known limitation — read before building

On the current dataset, efficiency ranges **only from 96.8% to 99.2%**, and no row is below 70%. XGBoost is tree-based and cannot predict values outside the target range it was trained on, so **R5 can never fire with this dataset**. The original four features explained only about 76% of the variance (linear baseline); adding `SOC (%)` raises the baseline to R² ≈ 1.0, which is why SOC is an operator input.
