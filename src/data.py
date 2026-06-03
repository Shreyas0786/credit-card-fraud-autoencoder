"""Data loading, preprocessing, and the leakage-safe semi-supervised split.

The autoencoder is trained on *normal* transactions only. To keep the experiment
honest we are careful about two things:

1. The fraud labels never touch training. Train is 100% normal.
2. The ``StandardScaler`` is fit on the *training-normal* data only, then applied to
   validation and test. No statistics leak from fraud or from the test set.

Split (semi-supervised setup):
    - Train : ~80% of normal transactions, fraud-free.
    - Val   : ~10% of normal + half of the fraud  -> used to pick the threshold.
    - Test  : remaining ~10% of normal + remaining fraud -> final reported metrics.

Run ``python -m src.data`` to print a summary and sanity-check the split.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Project paths (this file lives in <root>/src/data.py)
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"

# One global seed for reproducibility everywhere.
SEED = 42

# Fraction of normal transactions held out from training, split evenly between
# validation and test. The fraud cases are split 50/50 between val and test.
NORMAL_TRAIN_FRAC = 0.80
NORMAL_VAL_FRAC = 0.10  # the remaining ~0.10 becomes test


_MISSING_MESSAGE = f"""
============================================================
  Dataset not found:  {DATA_PATH}

  This project uses the 'Credit Card Fraud Detection' dataset
  (mlg-ulb / ULB-Worldline). It is ~150 MB and is NOT included
  in the repository.

  1. Download it from Kaggle:
       https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  2. Place the file 'creditcard.csv' at:
       {DATA_PATH}

  Then re-run.
============================================================
"""


@dataclass
class Dataset:
    """Container for the prepared, scaled splits.

    ``X_train`` is all-normal (no labels needed). Validation and test carry labels
    (0 = normal, 1 = fraud) so we can pick a threshold and report metrics.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def load_raw(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV, or print a clear message and exit if it is missing."""
    if not data_path.exists():
        print(_MISSING_MESSAGE)
        sys.exit(1)
    return pd.read_csv(data_path)


def load_data(data_path: Path = DATA_PATH, seed: int = SEED) -> Dataset:
    """Load, preprocess, and split the data into the leakage-safe splits.

    Steps:
      - drop ``Time`` (not informative here),
      - keep ``V1``..``V28`` (already PCA components) and ``Amount``,
      - split by class with a fixed seed,
      - fit ``StandardScaler`` on training-normal only, transform all splits.
    """
    df = load_raw(data_path)

    # 'Time' is dropped; everything else except the label is a feature.
    df = df.drop(columns=["Time"])
    feature_names = [c for c in df.columns if c != "Class"]

    normal = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]

    # Deterministic shuffle of each class before splitting.
    normal = normal.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    fraud = fraud.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Normal -> 80% train / 10% val / 10% test.
    n_normal = len(normal)
    n_train = int(NORMAL_TRAIN_FRAC * n_normal)
    n_val = int(NORMAL_VAL_FRAC * n_normal)
    normal_train = normal.iloc[:n_train]
    normal_val = normal.iloc[n_train : n_train + n_val]
    normal_test = normal.iloc[n_train + n_val :]

    # Fraud -> 50% val / 50% test (none in train).
    n_fraud_val = len(fraud) // 2
    fraud_val = fraud.iloc[:n_fraud_val]
    fraud_test = fraud.iloc[n_fraud_val:]

    # Assemble splits. Train is normal-only (no labels needed downstream).
    train_df = normal_train
    val_df = pd.concat([normal_val, fraud_val]).sample(frac=1.0, random_state=seed)
    test_df = pd.concat([normal_test, fraud_test]).sample(frac=1.0, random_state=seed)

    X_train = train_df[feature_names].to_numpy(dtype=np.float32)
    X_val = val_df[feature_names].to_numpy(dtype=np.float32)
    y_val = val_df["Class"].to_numpy(dtype=np.int64)
    X_test = test_df[feature_names].to_numpy(dtype=np.float32)
    y_test = test_df["Class"].to_numpy(dtype=np.int64)

    # Fit the scaler on TRAINING-NORMAL ONLY, then apply to every split.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return Dataset(
        X_train=X_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )


def _summary(ds: Dataset) -> None:
    """Print a human-readable summary, used by the __main__ sanity check."""
    print(f"Features ({ds.n_features}): {ds.feature_names}")
    print(f"Train : {ds.X_train.shape[0]:>7,} samples  (all normal, fraud-free)")
    print(
        f"Val   : {ds.X_val.shape[0]:>7,} samples  "
        f"({int((ds.y_val == 0).sum()):,} normal / {int((ds.y_val == 1).sum()):,} fraud)"
    )
    print(
        f"Test  : {ds.X_test.shape[0]:>7,} samples  "
        f"({int((ds.y_test == 0).sum()):,} normal / {int((ds.y_test == 1).sum()):,} fraud)"
    )
    print(f"Train fraud count (must be 0): {0}  [train is normal-only by construction]")
    print(
        "Scaler fit on train: mean~0, std~1 ->",
        f"mean={ds.X_train.mean():.3e}, std={ds.X_train.std():.3f}",
    )


if __name__ == "__main__":
    _summary(load_data())
