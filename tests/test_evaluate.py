"""Tests for threshold selection, metrics, and that plots produce files.

All synthetic — no dataset or training needed.
"""

from __future__ import annotations

import numpy as np

import src.evaluate as ev
from src.evaluate import best_f1_threshold, compute_metrics


def test_threshold_separates_clean_data():
    """With perfectly separable scores, the chosen threshold sits in the gap."""
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 5.0, 6.0, 7.0])
    thr = best_f1_threshold(y, scores)
    assert 0.3 < thr <= 5.0


def test_threshold_chosen_from_validation_scores_range():
    """The threshold must be one of the validation score values (no test peeking)."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)])
    thr = best_f1_threshold(y, scores)
    assert scores.min() <= thr <= scores.max()


def test_metrics_perfect_separation():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.1, 9.0, 9.5])
    m = compute_metrics("toy", y, scores, threshold=1.0)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.pr_auc == 1.0
    assert m.roc_auc == 1.0


def test_metrics_all_missed():
    """Threshold so high nothing is flagged -> recall 0."""
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.0, 0.1, 2.0, 2.5])
    m = compute_metrics("toy", y, scores, threshold=100.0)
    assert m.recall == 0.0


def _synthetic_scores():
    rng = np.random.default_rng(1)
    y = np.array([0] * 200 + [1] * 30)
    scores = np.concatenate([rng.normal(0.3, 0.1, 200), rng.normal(2.0, 0.5, 30)])
    return y, scores


def test_plots_write_files(tmp_path, monkeypatch):
    """Each plotting function should create its PNG."""
    monkeypatch.setattr(ev, "FIGURES", tmp_path)
    y, scores = _synthetic_scores()
    thr = 1.0

    ev.plot_error_distribution(y, scores, thr)
    ev.plot_confusion_matrix(y, scores, thr)
    ev.plot_roc_pr(y, scores)
    rows = [
        compute_metrics("Autoencoder", y, scores, thr),
        compute_metrics("Isolation Forest", y, scores, thr),
    ]
    ev.plot_comparison_table(rows)

    for fname in [
        "error_distribution.png",
        "confusion_matrix.png",
        "roc_pr_curves.png",
        "comparison_table.png",
    ]:
        assert (tmp_path / fname).exists(), f"plot not written: {fname}"


def test_markdown_table_has_all_rows():
    y, scores = _synthetic_scores()
    rows = [
        compute_metrics("Autoencoder", y, scores, 1.0),
        compute_metrics("Isolation Forest", y, scores, 1.0),
    ]
    md = ev.markdown_table(rows)
    assert "Autoencoder" in md
    assert "Isolation Forest" in md
    assert "PR-AUC" in md
