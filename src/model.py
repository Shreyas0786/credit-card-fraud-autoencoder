"""The undercomplete autoencoder.

A small fully-connected autoencoder that squeezes the ~29-dim input through a
7-dim bottleneck and reconstructs it:

    29 -> 20 -> 14 -> 7  (bottleneck)  -> 14 -> 20 -> 29

ReLU on the hidden layers, linear output (the inputs are standardized, so they are
not bounded to any range). Trained with MSE reconstruction loss on normal data only.
The intuition: the network learns to reconstruct *normal* transactions well; fraud,
never seen in training, reconstructs poorly and so produces a high error.
"""

from __future__ import annotations

import torch
from torch import nn


class Autoencoder(nn.Module):
    """Undercomplete fully-connected autoencoder.

    Args:
        n_features: input/output dimensionality (number of features, ~29).
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 20),
            nn.ReLU(),
            nn.Linear(20, 14),
            nn.ReLU(),
            nn.Linear(14, 7),  # bottleneck
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(7, 14),
            nn.ReLU(),
            nn.Linear(14, 20),
            nn.ReLU(),
            nn.Linear(20, n_features),  # linear output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def reconstruction_error(model: Autoencoder, x: torch.Tensor) -> torch.Tensor:
    """Per-sample reconstruction MSE — this is the anomaly score.

    Returns a 1-D tensor of length ``len(x)``: the mean squared error between each
    input row and its reconstruction.
    """
    model.eval()
    with torch.no_grad():
        recon = model(x)
        return ((recon - x) ** 2).mean(dim=1)
