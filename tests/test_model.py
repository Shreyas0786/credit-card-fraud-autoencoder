"""Tests for the autoencoder architecture and the anomaly score.

No dataset needed — these use small synthetic tensors.
"""

from __future__ import annotations

import torch

from src.model import Autoencoder, reconstruction_error


def test_output_shape_matches_input():
    model = Autoencoder(n_features=29)
    x = torch.randn(16, 29)
    out = model(x)
    assert out.shape == x.shape


def test_bottleneck_is_seven_dim():
    """The encoder must compress to a 7-dim bottleneck (undercomplete)."""
    model = Autoencoder(n_features=29)
    x = torch.randn(4, 29)
    encoded = model.encoder(x)
    assert encoded.shape == (4, 7)


def test_reconstruction_error_is_per_sample_and_nonnegative():
    model = Autoencoder(n_features=29)
    x = torch.randn(10, 29)
    err = reconstruction_error(model, x)
    assert err.shape == (10,)
    assert torch.all(err >= 0)


def test_architecture_respects_feature_count():
    """Input/output dimensionality follows n_features."""
    model = Autoencoder(n_features=12)
    x = torch.randn(3, 12)
    assert model(x).shape == (3, 12)
