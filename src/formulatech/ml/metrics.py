"""Model quality metrics and gates (spec/03-ml-model.md §5)."""

import numpy as np

# Calculated values are rounded before threshold comparison (CLAUDE.md).
COMPARE_DECIMALS = 9


def tolerance_accuracy(y_true, y_pred, delta: float) -> float:
    """Percentage of predictions within ±delta of the true value (§5.1)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    within = np.round(np.abs(y_true - y_pred), COMPARE_DECIMALS) <= delta
    return float(within.mean() * 100)


def r2(y_true, y_pred) -> float:
    """R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)² (§5.2)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def classify_fit(r2_train: float, r2_test: float, gap_min: float, gap_max: float) -> str:
    """Classify the train/test R² gap (§5.3): good fit is gap_min ≤ gap < gap_max."""
    gap = round(r2_train - r2_test, COMPARE_DECIMALS)
    if gap < gap_min:
        return "underfit"
    if gap >= gap_max:
        return "overfit"
    return "good"


def evaluate_gates(metrics: dict, gates_cfg: dict) -> dict:
    """Apply every quality gate. Returns {gate_name: passed} plus an overall `gates_passed`."""
    rd = lambda x: round(x, COMPARE_DECIMALS)  # noqa: E731
    gates = {
        "tolerance_accuracy": rd(metrics["tolerance_accuracy"]) > gates_cfg["tolerance_accuracy_min_percent"],
        "r2": rd(metrics["r2_test"]) > gates_cfg["r2_min"],
        "good_fit": metrics["fit_status"] == "good",
        "cv_stability": (
            rd(metrics["cv_r2_mean"]) > gates_cfg["r2_min"]
            and rd(metrics["cv_r2_std"]) <= gates_cfg["cv_r2_std_max"]
        ),
        "early_stopping": metrics["best_iteration"] + 1 < metrics["n_estimators"],
    }
    gates["gates_passed"] = all(gates.values())
    return gates
