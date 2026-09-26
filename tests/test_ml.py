"""Model quality tests (spec/06-test-cases.md, MQ-1 to MQ-5)."""

import numpy as np
import pytest
from sklearn.metrics import r2_score

from formulatech.ml.metrics import classify_fit, r2, tolerance_accuracy
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


@pytest.mark.parametrize(
    "r2_train, r2_test, expected",
    [
        (0.99, 0.80, "overfit"),  # MQ-4:  gap 0.19
        (0.97, 0.95, "underfit"),  # MQ-4a: gap 0.02
        (0.99, 0.93, "good"),  # MQ-4b: gap 0.06
        (0.95, 0.90, "good"),  # MQ-4c: gap exactly 0.05
        (1.00, 0.90, "overfit"),  # MQ-4c: gap exactly 0.1
    ],
)
def test_mq4_classify_fit(r2_train, r2_test, expected):
    assert classify_fit(r2_train, r2_test, gap_min=0.05, gap_max=0.1) == expected


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
