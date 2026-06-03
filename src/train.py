"""Training loop for the autoencoder.

Trains on normal transactions only (``Dataset.X_train``) with Adam + MSE. Per epoch
we record two numbers:

  - train loss : mean reconstruction MSE over the training (normal) batches,
  - val loss   : mean reconstruction MSE over the *normal* validation samples.

We deliberately track validation loss on normal samples only. The autoencoder is
supposed to reconstruct normal data well; fraud is expected to reconstruct poorly, so
mixing it into the "val loss" would make the curve meaningless as an overfitting signal.

Outputs:
  - ``figures/loss_curve.png`` — train vs val loss per epoch,
  - ``figures/autoencoder.pt``  — trained model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import SEED, Dataset
from .model import Autoencoder

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
WEIGHTS_PATH = FIGURES / "autoencoder.pt"
LOSS_CURVE_PATH = FIGURES / "loss_curve.png"


@dataclass
class History:
    """Per-epoch loss record."""

    train_loss: list[float]
    val_loss: list[float]


def set_seed(seed: int = SEED) -> None:
    """Seed Python/NumPy/torch RNGs for reproducible training."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_autoencoder(
    ds: Dataset,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = SEED,
    save: bool = True,
    verbose: bool = True,
) -> tuple[Autoencoder, History]:
    """Train the autoencoder on normal data and return (model, history)."""
    set_seed(seed)

    model = Autoencoder(n_features=ds.n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train = torch.from_numpy(ds.X_train)
    loader = DataLoader(
        TensorDataset(X_train),
        batch_size=batch_size,
        shuffle=True,
    )

    # Validation loss is measured on NORMAL val samples only (see module docstring).
    X_val_normal = torch.from_numpy(ds.X_val[ds.y_val == 0])

    history = History(train_loss=[], val_loss=[])

    for epoch in range(1, epochs + 1):
        model.train()
        running, n_seen = 0.0, 0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch)
            n_seen += len(batch)
        train_loss = running / n_seen

        model.eval()
        with torch.no_grad():
            val_recon = model(X_val_normal)
            val_loss = criterion(val_recon, X_val_normal).item()

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        if verbose:
            print(
                f"epoch {epoch:>3}/{epochs}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
            )

    if save:
        FIGURES.mkdir(exist_ok=True)
        torch.save(model.state_dict(), WEIGHTS_PATH)
        _plot_loss_curve(history)
        if verbose:
            print(f"\nSaved weights -> {WEIGHTS_PATH}")
            print(f"Saved loss curve -> {LOSS_CURVE_PATH}")

    return model, history


def _plot_loss_curve(history: History) -> None:
    """Save the train vs val loss curve to figures/loss_curve.png."""
    epochs = range(1, len(history.train_loss) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history.train_loss, label="train (normal)")
    plt.plot(epochs, history.val_loss, label="val (normal)")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction MSE")
    plt.title("Autoencoder training — reconstruction loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    plt.savefig(LOSS_CURVE_PATH, dpi=150)
    plt.close()


if __name__ == "__main__":
    from .data import load_data

    train_autoencoder(load_data())
