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


def evaluate_gates(metrics: dict, gates_cfg: dict) -> dict:
    """Apply the three accuracy gates: tolerance accuracy, test R², and CV R² (mean and std).
    Returns {gate_name: passed} plus an overall `gates_passed`.

    `r2_train`, `r2_gap` and `best_iteration` are information only (§5.3) and are not read here.
    """
    rd = lambda x: round(x, COMPARE_DECIMALS)  # noqa: E731
    gates = {
        "tolerance_accuracy": rd(metrics["tolerance_accuracy"]) > gates_cfg["tolerance_accuracy_min_percent"],
        "r2": rd(metrics["r2_test"]) > gates_cfg["r2_min"],
        "cv_stability": (
            rd(metrics["cv_r2_mean"]) > gates_cfg["r2_min"]
            and rd(metrics["cv_r2_std"]) <= gates_cfg["cv_r2_std_max"]
        ),
    }
    gates["gates_passed"] = all(gates.values())
    return gates
