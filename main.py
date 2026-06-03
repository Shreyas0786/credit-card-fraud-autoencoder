"""Run the full fraud-autoencoder pipeline top to bottom.

    python main.py

Stages:
    1. Load + preprocess + leakage-safe split   (src/data.py)
    2. Train the autoencoder on normal data only (src/train.py)
    3. Evaluate + Isolation Forest baseline       (src/evaluate.py)

Figures are written to figures/. The trained model is saved to figures/autoencoder.pt.
"""

from __future__ import annotations

from src.data import load_data
from src.evaluate import evaluate_all
from src.train import train_autoencoder


def main() -> None:
    print("=" * 60)
    print("STAGE 1 — Load + preprocess + split")
    print("=" * 60)
    ds = load_data()
    print(
        f"train={ds.X_train.shape[0]:,} (normal-only)  "
        f"val={ds.X_val.shape[0]:,}  test={ds.X_test.shape[0]:,}  "
        f"features={ds.n_features}"
    )

    print("\n" + "=" * 60)
    print("STAGE 2 — Train autoencoder (normal-only)")
    print("=" * 60)
    model, _ = train_autoencoder(ds)

    print("\n" + "=" * 60)
    print("STAGE 3 — Evaluate + Isolation Forest baseline")
    print("=" * 60)
    evaluate_all(ds, model)

    print("\nDone. See figures/ for all plots and the comparison table.")


if __name__ == "__main__":
    main()
