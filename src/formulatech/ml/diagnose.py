"""Overfitting / underfitting diagnostics for the saved efficiency model (spec/03-ml-model.md §5.3).

Information only: nothing here passes or fails the model. Run after training with:
    uv run python -m formulatech.ml.diagnose
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import KFold, learning_curve, validation_curve  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from formulatech.config import MODELS_DIR, load_thresholds  # noqa: E402
from formulatech.ml.metrics import r2  # noqa: E402
from formulatech.ml.train import META_FILE, RANDOM_STATE, fit_with_early_stopping, load_data, split_data  # noqa: E402

TUNED_KEYS = ["learning_rate", "max_depth", "min_child_weight", "reg_lambda", "reg_alpha"]
TRAIN_SIZES = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10]

SIZE_PLOT = "diagnose_training_size.png"
DEPTH_PLOT = "diagnose_max_depth.png"
REPORT_FILE = "diagnose_report.json"


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


def _plot(x, train, val, xlabel: str, title: str, path, delta: float | None = None, chosen=None) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(x, train, "o-", label="train")
    plt.plot(x, val, "o-", label="validation (CV)")
    if delta is not None:
        plt.axhline(delta, ls=":", c="red", label=f"δ = {delta} pp")
    if chosen is not None:
        plt.axvline(chosen, ls="--", c="grey", label="chosen")
    plt.xlabel(xlabel)
    plt.ylabel("RMSE (efficiency pp)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def diagnose(out_dir=MODELS_DIR) -> dict:
    meta = json.loads((out_dir / META_FILE).read_text())
    hp = {k: v for k, v in meta["hyperparameters"].items() if k != "early_stopping_rounds"}
    gates_cfg = load_thresholds()["model_gates"]
    delta = gates_cfg["tolerance_delta_pp"]

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 1: train vs validation vs test error, with the same early-stopping split as train.py
    model, (X_fit, X_val, y_fit, y_val) = fit_with_early_stopping(X_train, y_train, {k: hp[k] for k in TUNED_KEYS})
    errors = {
        name: {"rmse": rmse(ys, model.predict(xs)), "r2": r2(ys, model.predict(xs))}
        for name, xs, ys in [("train", X_fit, y_fit), ("validation", X_val, y_val), ("test", X_test, y_test)]
    }
    errors["test_to_train_rmse_ratio"] = errors["test"]["rmse"] / errors["train"]["rmse"]

    # Step 2: simple baselines on the test set
    linear = LinearRegression().fit(X_train, y_train)
    baselines = {
        "mean": {"rmse": rmse(y_test, np.full(len(y_test), y_train.mean()))},
        "linear": {
            "rmse": rmse(y_test, linear.predict(X_test)),
            "r2": r2(y_test, linear.predict(X_test)),
            "intercept": float(linear.intercept_),
            "coefficients": dict(zip(X.columns, map(float, linear.coef_))),
        },
    }

    # Steps 4–5: 5-fold CV curves on the train split, at the saved model's number of rounds
    fixed = XGBRegressor(**{**hp, "n_estimators": model.best_iteration + 1}, random_state=RANDOM_STATE)
    cv = KFold(gates_cfg["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
    scoring = "neg_root_mean_squared_error"

    sizes, tr, va = learning_curve(fixed, X_train, y_train, train_sizes=TRAIN_SIZES, cv=cv, scoring=scoring, n_jobs=-1)
    by_size = [{"rows": int(n), "train_rmse": float(-a.mean()), "validation_rmse": float(-b.mean())}
               for n, a, b in zip(sizes, tr, va)]
    _plot(sizes, -tr.mean(1), -va.mean(1), "Training rows", "Error by training-set size",
          out_dir / SIZE_PLOT, delta=delta)

    tr, va = validation_curve(fixed, X_train, y_train, param_name="max_depth", param_range=DEPTHS, cv=cv,
                              scoring=scoring, n_jobs=-1)
    by_depth = [{"max_depth": d, "train_rmse": float(-a.mean()), "validation_rmse": float(-b.mean())}
                for d, a, b in zip(DEPTHS, tr, va)]
    _plot(DEPTHS, -tr.mean(1), -va.mean(1), "max_depth", "Error by tree depth",
          out_dir / DEPTH_PLOT, chosen=hp["max_depth"])

    report = {
        "model_version": meta["version"],
        "hyperparameters": hp,
        "step1_errors": errors,
        "step2_baselines": baselines,
        # Step 3 is the boosting-round curve that train.py already saves as learning_curve.png
        "step3_best_iteration": {"best_iteration": model.best_iteration, "n_estimators": hp["n_estimators"]},
        "step4_training_size": by_size,
        "step5_max_depth": by_depth,
        "step6_cv_r2": {k: meta["metrics"][k] for k in ["cv_r2_folds", "cv_r2_mean", "cv_r2_std"]},
    }
    (out_dir / REPORT_FILE).write_text(json.dumps(report, indent=2))
    return report


def print_report(report: dict) -> None:
    e, b = report["step1_errors"], report["step2_baselines"]
    print("Overfitting / underfitting diagnostics (information only, not gates)\n")
    print("1. Error (RMSE, pp):  "
          f"train {e['train']['rmse']:.4f}, validation {e['validation']['rmse']:.4f}, test {e['test']['rmse']:.4f}"
          f"  → test/train ratio {e['test_to_train_rmse_ratio']:.2f}")
    print(f"2. Baselines (test RMSE, pp):  predict mean {b['mean']['rmse']:.4f}, linear {b['linear']['rmse']:.2e}"
          f" (R² {b['linear']['r2']:.4f})")
    s = report["step3_best_iteration"]
    print(f"3. Best iteration {s['best_iteration']} of {s['n_estimators']} (see learning_curve.png)")
    print("4. Training size:  " + ", ".join(
        f"{r['rows']} rows {r['train_rmse']:.3f}/{r['validation_rmse']:.3f}" for r in report["step4_training_size"]))
    print("5. max_depth:      " + ", ".join(
        f"{r['max_depth']}: {r['train_rmse']:.3f}/{r['validation_rmse']:.3f}" for r in report["step5_max_depth"]))
    best = min(report["step5_max_depth"], key=lambda r: r["validation_rmse"])
    print(f"   lowest validation error at max_depth={best['max_depth']}"
          f" (chosen: {report['hyperparameters']['max_depth']})")
    c = report["step6_cv_r2"]
    print(f"6. CV R²: {c['cv_r2_mean']:.4f} ± {c['cv_r2_std']:.4f}")
    print(f"\nPlots: {SIZE_PLOT}, {DEPTH_PLOT}; report: {REPORT_FILE} (in models/)")


if __name__ == "__main__":
    print_report(diagnose())
