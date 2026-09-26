"""Model quality tests (spec/06-test-cases.md, MQ-1 to MQ-5)."""

import numpy as np
import pytest
from sklearn.metrics import r2_score

from formulatech.ml.metrics import evaluate_gates, r2, tolerance_accuracy
from formulatech.ml.train import FEATURES, LEAKY_COLUMNS, fit_with_early_stopping, load_data, split_data


def test_mq1_degradation_rate_not_a_feature():
    assert not set(FEATURES) & set(LEAKY_COLUMNS)
    X, _ = load_data()
    assert "Degradation Rate (%)" not in X.columns
    assert "SOC (%)" in FEATURES


def test_mq2_tolerance_accuracy_hand_computed():
    y_true = [98.0, 98.0, 98.0, 98.0]
    y_pred = [98.05, 98.1, 98.2, 97.0]  # errors 0.05, 0.1 (boundary, counts), 0.2, 1.0
    assert tolerance_accuracy(y_true, y_pred, delta=0.1) == pytest.approx(50.0)


def test_mq3_r2_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.normal(98, 0.5, 50)
    y_pred = y_true + rng.normal(0, 0.1, 50)
    assert r2(y_true, y_pred) == pytest.approx(r2_score(y_true, y_pred))


GATES_CFG = {"tolerance_accuracy_min_percent": 80.0, "r2_min": 0.90, "cv_r2_std_max": 0.03, "early_stopping_margin": 100}
PASSING_METRICS = {"tolerance_accuracy": 100.0, "r2_train": 0.9997, "r2_test": 0.9974, "r2_gap": 0.0023,
                   "cv_r2_mean": 0.9976, "cv_r2_std": 0.0004, "best_iteration": 8000, "n_estimators": 10000}


def test_mq4_r2_gap_is_information_only():
    gates = evaluate_gates(PASSING_METRICS, GATES_CFG)
    assert "good_fit" not in gates
    assert gates["gates_passed"] is True


def test_mq4a_early_stopping_fails_on_last_round():
    gates = evaluate_gates({**PASSING_METRICS, "best_iteration": 9999, "n_estimators": 10000}, GATES_CFG)
    assert gates["early_stopping"] is False
    assert gates["gates_passed"] is False


def test_mq5_split_is_80_20_and_disjoint():
    X, y = load_data()
    X_train, X_test, _, _ = split_data(X, y)
    assert len(X_train) == 800 and len(X_test) == 200
    assert set(X_train.index).isdisjoint(X_test.index)


def test_fit_with_early_stopping_allows_tuned_learning_rate():
    X, y = load_data()
    model, _ = fit_with_early_stopping(
        X.iloc[:100],
        y.iloc[:100],
        {"learning_rate": 0.03, "max_depth": 2, "min_child_weight": 5},
    )
    assert model is not None
    assert len(model.predict(X.iloc[:10])) == 10
