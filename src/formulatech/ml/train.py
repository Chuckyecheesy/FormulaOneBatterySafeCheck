"""Train the XGBoost efficiency model (spec/03-ml-model.md §2–§5).

Run with:  uv run python -m formulatech.ml.train
"""

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold, train_test_split  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from formulatech.config import DATA_PATH, MODELS_DIR, load_thresholds  # noqa: E402
from formulatech.ml.metrics import evaluate_gates, r2, tolerance_accuracy  # noqa: E402

# §2: the four values the operator enters. Order is fixed and saved with the model.
FEATURES = [
    "SOC (%)",
    "Voltage (V)",
    "Current (A)",
    "Battery Temp (°C)",
    "Charging Duration (min)",
]
TARGET = "Efficiency (%)"
# Leaks the target (correlation −1.00). Must never be a feature.
LEAKY_COLUMNS = ["Degradation Rate (%)"]

RANDOM_STATE = 42
TEST_SIZE = 0.20  # §3: 80% train / 20% test
VALIDATION_SIZE = 0.15  # early-stopping slice, taken from inside the training data
EARLY_STOPPING_ROUNDS = 50

# §4 starting hyperparameters.
BASE_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 5000,  # headroom so validation loss can plateau before the last round
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 10.0,
    "reg_alpha": 1.0,
}
# §4 tuning grid, scored by 5-fold CV R² on the train split only.
PARAM_GRID = {
    "learning_rate": [0.03, 0.05, 0.075, 0.1],
    "max_depth": [2,3,4],
    "min_child_weight": [1,3,5],
    "reg_lambda": [10],
    "reg_alpha": [1.0]
}

MODEL_FILE = "efficiency_model.json"
META_FILE = "efficiency_model_meta.json"
CURVE_FILE = "learning_curve.png"
# A model that fails any gate is saved here, never over the deployed model in out_dir (§5).
REJECTED_DIR = "rejected"


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    assert not set(FEATURES) & set(LEAKY_COLUMNS), "leaky column in feature list"
    return df[FEATURES], df[TARGET]


def split_data(X: pd.DataFrame, y: pd.Series):
    """§3: hold out 20% as the test set. It is used once, after tuning."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def fit_with_early_stopping(X, y, params: dict) -> tuple[XGBRegressor, tuple]:
    """Fit on part of (X, y), early-stopping on a validation slice of the rest.

    Returns the model and the (X_fit, X_val, y_fit, y_val) split it used.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
    )
    model_params = {**BASE_PARAMS, **params}
    model = XGBRegressor(
        **model_params,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric="rmse",
        random_state=RANDOM_STATE,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_fit, y_fit), (X_val, y_val)], verbose=False)
    return model, (X_fit, X_val, y_fit, y_val)


def cross_validate(X_train, y_train, params: dict, folds: int) -> list[float]:
    """5-fold CV R² on the train split. Each fold early-stops on its own inner slice,
    so the held-out fold is only used for scoring."""
    scores = []
    for fit_idx, score_idx in KFold(folds, shuffle=True, random_state=RANDOM_STATE).split(X_train):
        model, _ = fit_with_early_stopping(X_train.iloc[fit_idx], y_train.iloc[fit_idx], params)
        pred = model.predict(X_train.iloc[score_idx])
        scores.append(r2(y_train.iloc[score_idx], pred))
    return scores


def tune(X_train, y_train, folds: int) -> tuple[dict, list[float], list[dict]]:
    """Grid search over PARAM_GRID. Returns the best params, their CV scores, and every trial."""
    trials = []
    for values in itertools.product(*PARAM_GRID.values()):
        params = dict(zip(PARAM_GRID.keys(), values))
        scores = cross_validate(X_train, y_train, params, folds)
        trials.append({"params": params, "cv_r2_mean": float(np.mean(scores)), "cv_r2_std": float(np.std(scores))})
        print(f"  {params}  CV R² = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    best = max(trials, key=lambda t: t["cv_r2_mean"])
    best_scores = cross_validate(X_train, y_train, best["params"], folds)
    return best["params"], best_scores, trials


def save_learning_curve(model: XGBRegressor, path: Path) -> None:
    """§5.3: train vs validation RMSE by boosting round, for manual review."""
    history = model.evals_result()
    plt.figure(figsize=(7, 4))
    plt.plot(history["validation_0"]["rmse"], label="train")
    plt.plot(history["validation_1"]["rmse"], label="validation")
    plt.axvline(model.best_iteration, color="grey", linestyle="--", label="best iteration")
    plt.xlabel("Boosting round")
    plt.ylabel("RMSE (efficiency %)")
    plt.title("XGBoost efficiency model — learning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_file(path: Path, write) -> None:
    """Write to a temporary file next to `path`, then swap it in, so `path` is never half-written.

    The temp name keeps the real extension last: XGBoost picks the save format from it.
    """
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    write(tmp)
    tmp.replace(path)


def train(data_path: Path = DATA_PATH, out_dir: Path = MODELS_DIR) -> dict:
    """Run the full pipeline and save the artifact. Returns the metadata written to disk.

    Only a model that passes every gate is saved to `out_dir`, where the app loads it. A failing
    model goes to `out_dir / REJECTED_DIR` for analysis, and the deployed model is left untouched.
    """
    gates_cfg = load_thresholds()["model_gates"]
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)}")

    print("Tuning (5-fold CV on train split):")
    best_params, cv_scores, trials = tune(X_train, y_train, gates_cfg["cv_folds"])

    model, (X_fit, _, y_fit, _) = fit_with_early_stopping(X_train, y_train, best_params)

    # The test set is used here, once, after all tuning is finished.
    pred_test = model.predict(X_test)
    r2_train = r2(y_fit, model.predict(X_fit))
    r2_test = r2(y_test, pred_test)
    metrics = {
        "tolerance_accuracy": tolerance_accuracy(y_test, pred_test, gates_cfg["tolerance_delta_pp"]),
        "r2_train": r2_train,  # information only (§5.3)
        "r2_test": r2_test,
        "r2_gap": r2_train - r2_test,  # information only (§5.3)
        "cv_r2_mean": float(np.mean(cv_scores)),
        "cv_r2_std": float(np.std(cv_scores)),
        "cv_r2_folds": cv_scores,
        "best_iteration": int(model.best_iteration),
        "n_estimators": BASE_PARAMS["n_estimators"],
    }
    gates = evaluate_gates(metrics, gates_cfg)

    dest = out_dir if gates["gates_passed"] else out_dir / REJECTED_DIR
    dest.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "target": TARGET,
        "data_file": data_path.name,
        "data_sha256": file_sha256(data_path),
        "random_state": RANDOM_STATE,
        "split": {"train": len(X_train), "test": len(X_test), "validation_fraction_of_train": VALIDATION_SIZE},
        "hyperparameters": {**BASE_PARAMS, **best_params, "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
        "tuning_trials": trials,
        "metrics": metrics,
        "gates": gates,
        "gates_passed": gates["gates_passed"],
        "deployed": gates["gates_passed"],
        "saved_to": str(dest),
    }
    # Model first, then metadata: the app reads the version from the metadata.
    _replace_file(dest / MODEL_FILE, lambda tmp: model.save_model(tmp))
    _replace_file(dest / META_FILE, lambda tmp: tmp.write_text(json.dumps(meta, indent=2)))
    save_learning_curve(model, dest / CURVE_FILE)
    return meta


def print_report(meta: dict) -> None:
    m, g = meta["metrics"], meta["gates"]
    cfg = load_thresholds()["model_gates"]
    mark = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    print("\nQuality gates (test set):")
    print(f"  Tolerance accuracy (δ={cfg['tolerance_delta_pp']}): {m['tolerance_accuracy']:.1f} %"
          f"  (needs > {cfg['tolerance_accuracy_min_percent']})  {mark(g['tolerance_accuracy'])}")
    print(f"  R² test: {m['r2_test']:.4f}  (needs > {cfg['r2_min']})  {mark(g['r2'])}")
    print(f"  CV R²: {m['cv_r2_mean']:.4f} ± {m['cv_r2_std']:.4f}"
          f"  (needs mean > {cfg['r2_min']}, std ≤ {cfg['cv_r2_std_max']})  {mark(g['cv_stability'])}")
    print("\nInformation only (not gates):")
    print(f"  R² train: {m['r2_train']:.4f}, train − test R² gap: {m['r2_gap']:.4f}")
    print(f"  Early stopping: best iteration {m['best_iteration']} of {m['n_estimators']}")
    print(f"\ngates_passed = {meta['gates_passed']}")
    if meta["gates_passed"]:
        print(f"Model deployed to {meta['saved_to']}.")
    else:
        print(f"Model REJECTED: it failed the acceptance gates (spec/03-ml-model.md §5). Saved to "
              f"{meta['saved_to']} for analysis; the deployed model was not changed.")


if __name__ == "__main__":
    meta = train()
    print_report(meta)
    raise SystemExit(0 if meta["gates_passed"] else 1)
