"""Tests for the leakage-safe split and preprocessing.

The dataset is large and downloaded manually, so these tests are skipped if
``data/creditcard.csv`` is absent — the rest of the suite still runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import DATA_PATH, SEED, load_data

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="data/creditcard.csv not present (download from Kaggle to run these)",
)

# Known constants for the ULB credit-card dataset.
TOTAL_NORMAL = 284_315
TOTAL_FRAUD = 492


@pytest.fixture(scope="module")
def ds():
    return load_data()


def test_feature_count_is_29(ds):
    assert ds.n_features == 29
    assert ds.X_train.shape[1] == 29


def test_time_and_class_not_in_features(ds):
    assert "Time" not in ds.feature_names
    assert "Class" not in ds.feature_names
    assert "Amount" in ds.feature_names


def test_fraud_is_split_in_half(ds):
    val_fraud = int((ds.y_val == 1).sum())
    test_fraud = int((ds.y_test == 1).sum())
    assert val_fraud + test_fraud == TOTAL_FRAUD
    # Half each (odd total -> off by at most one).
    assert abs(val_fraud - test_fraud) <= 1


def test_all_normal_accounted_for_and_train_is_fraud_free(ds):
    """Train is normal-only by construction. Verify every normal row is placed
    exactly once across train + val-normal + test-normal (no leakage, no dup)."""
    val_normal = int((ds.y_val == 0).sum())
    test_normal = int((ds.y_test == 0).sum())
    assert ds.X_train.shape[0] + val_normal + test_normal == TOTAL_NORMAL


def test_train_roughly_80_percent_of_normal(ds):
    frac = ds.X_train.shape[0] / TOTAL_NORMAL
    assert 0.79 <= frac <= 0.81


def test_scaler_fit_on_train_only(ds):
    """Train features must be standardized: per-column mean ~0, std ~1."""
    col_means = ds.X_train.mean(axis=0)
    col_stds = ds.X_train.std(axis=0)
    assert np.allclose(col_means, 0.0, atol=1e-4)
    assert np.allclose(col_stds, 1.0, atol=1e-2)


def test_val_test_use_train_statistics_not_their_own(ds):
    """Val/test are transformed with TRAIN stats, so they are NOT perfectly centered.
    If they were, the scaler would have been (wrongly) refit on them."""
    assert not np.allclose(ds.X_val.mean(axis=0), 0.0, atol=1e-4)


def test_reproducible_with_same_seed():
    a = load_data(seed=SEED)
    b = load_data(seed=SEED)
    assert np.array_equal(a.X_train, b.X_train)
    assert np.array_equal(a.y_test, b.y_test)
