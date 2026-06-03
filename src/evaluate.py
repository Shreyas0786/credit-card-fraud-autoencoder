"""Evaluation, threshold selection, plots, and the Isolation Forest baseline.

The anomaly score is the per-sample reconstruction MSE. The decision threshold is
chosen on the **validation** set (the value that maximizes F1) and then applied,
unchanged, to the **test** set for the final reported metrics. The test set is never
used to tune anything.

Because fraud is ~0.17% of the data, **accuracy is meaningless** — a model that calls
everything "normal" scores 99.8%. We report precision / recall / F1 at the chosen
threshold plus ROC-AUC and PR-AUC, with **PR-AUC as the headline metric**.

Figures produced:
  - figures/error_distribution.png  — reconstruction error, normal vs fraud (the money plot)
  - figures/confusion_matrix.png    — confusion matrix at the chosen threshold
  - figures/roc_pr_curves.png       — ROC and PR curves
  - figures/comparison_table.png    — autoencoder vs Isolation Forest
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .data import SEED, Dataset
from .model import Autoencoder, reconstruction_error

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


@dataclass
class Metrics:
    """Metrics for one model on the test set, at a val-chosen threshold."""

    name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float  # average precision — the headline metric

    def as_row(self) -> list[str]:
        return [
            self.name,
            f"{self.precision:.3f}",
            f"{self.recall:.3f}",
            f"{self.f1:.3f}",
            f"{self.pr_auc:.3f}",
        ]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def ae_scores(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    """Per-sample reconstruction MSE for the autoencoder (higher = more anomalous)."""
    return reconstruction_error(model, torch.from_numpy(X)).numpy()


def iforest_scores(ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Fit Isolation Forest on the same normal-only training data; score val and test.

    sklearn's ``score_samples`` returns higher values for *more normal* points, so we
    negate it to get an anomaly score where higher = more anomalous (matching the AE).
    """
    clf = IsolationForest(n_estimators=100, random_state=SEED, n_jobs=-1)
    clf.fit(ds.X_train)  # train-normal only — same setup as the AE
    val = -clf.score_samples(ds.X_val)
    test = -clf.score_samples(ds.X_test)
    return val, test


# --------------------------------------------------------------------------- #
# Threshold + metrics
# --------------------------------------------------------------------------- #
def best_f1_threshold(y_val: np.ndarray, scores_val: np.ndarray) -> float:
    """Pick the threshold on validation that maximizes F1.

    Uses the precision/recall operating points from ``precision_recall_curve`` and
    returns the threshold with the highest F1. This is the defensible, no-test-leakage
    way to set the cutoff.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, scores_val)
    # precision/recall have length len(thresholds)+1; align by dropping the last point.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def compute_metrics(
    name: str, y: np.ndarray, scores: np.ndarray, threshold: float
) -> Metrics:
    """Compute precision/recall/F1 at ``threshold`` plus threshold-free ROC/PR-AUC."""
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Metrics(
        name=name,
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc_score(y, scores),
        pr_auc=average_precision_score(y, scores),
    )


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_error_distribution(
    y_test: np.ndarray, scores_test: np.ndarray, threshold: float
) -> None:
    """The money plot: reconstruction-error distribution, normal vs fraud overlaid."""
    normal = scores_test[y_test == 0]
    fraud = scores_test[y_test == 1]

    plt.figure(figsize=(7.5, 4.5))
    bins = np.linspace(0, np.percentile(scores_test, 99.5), 80)
    plt.hist(normal, bins=bins, alpha=0.6, label="normal", density=True)
    plt.hist(fraud, bins=bins, alpha=0.6, label="fraud", density=True)
    plt.axvline(
        threshold, color="red", linestyle="--", label=f"threshold = {threshold:.3f}"
    )
    plt.xlabel("Reconstruction error (MSE)")
    plt.ylabel("Density")
    plt.title("Reconstruction error — normal vs fraud (test set)")
    plt.legend()
    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    plt.savefig(FIGURES / "error_distribution.png", dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_test: np.ndarray, scores_test: np.ndarray, threshold: float
) -> None:
    """Confusion matrix at the chosen threshold."""
    y_pred = (scores_test >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["normal", "fraud"])
    ax.set_yticks([0, 1], labels=["normal", "fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (autoencoder, test)")
    # Annotate each cell with its count.
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    plt.savefig(FIGURES / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_roc_pr(y_test: np.ndarray, scores_test: np.ndarray) -> None:
    """ROC and PR curves side by side for the autoencoder."""
    fpr, tpr, _ = roc_curve(y_test, scores_test)
    roc_auc = roc_auc_score(y_test, scores_test)
    precision, recall, _ = precision_recall_curve(y_test, scores_test)
    pr_auc = average_precision_score(y_test, scores_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC curve")
    ax1.legend()

    ax2.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall curve")
    ax2.legend()

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    plt.savefig(FIGURES / "roc_pr_curves.png", dpi=150)
    plt.close()


def plot_comparison_table(rows: list[Metrics]) -> None:
    """Render the autoencoder-vs-baseline comparison as a saved table image."""
    header = ["Model", "Precision", "Recall", "F1", "PR-AUC"]
    cell_text = [m.as_row() for m in rows]

    fig, ax = plt.subplots(figsize=(7.5, 1.4 + 0.4 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    ax.set_title("Autoencoder vs Isolation Forest (test set)", pad=12)
    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    plt.savefig(FIGURES / "comparison_table.png", dpi=150, bbox_inches="tight")
    plt.close()


def markdown_table(rows: list[Metrics]) -> str:
    """Return a Markdown comparison table (for pasting into the README)."""
    lines = [
        "| Model | Precision | Recall | F1 | PR-AUC |",
        "|-------|-----------|--------|----|--------|",
    ]
    for m in rows:
        lines.append(
            f"| {m.name} | {m.precision:.3f} | {m.recall:.3f} | "
            f"{m.f1:.3f} | {m.pr_auc:.3f} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate_all(ds: Dataset, model: Autoencoder, verbose: bool = True) -> list[Metrics]:
    """Run the full evaluation: AE + Isolation Forest, all figures, comparison table."""
    # --- Autoencoder ---
    ae_val = ae_scores(model, ds.X_val)
    ae_test = ae_scores(model, ds.X_test)
    ae_thr = best_f1_threshold(ds.y_val, ae_val)
    ae_metrics = compute_metrics("Autoencoder", ds.y_test, ae_test, ae_thr)

    # --- Isolation Forest baseline (same split) ---
    if_val, if_test = iforest_scores(ds)
    if_thr = best_f1_threshold(ds.y_val, if_val)
    if_metrics = compute_metrics("Isolation Forest", ds.y_test, if_test, if_thr)

    rows = [ae_metrics, if_metrics]

    # --- Figures (autoencoder-focused, plus the comparison table) ---
    plot_error_distribution(ds.y_test, ae_test, ae_thr)
    plot_confusion_matrix(ds.y_test, ae_test, ae_thr)
    plot_roc_pr(ds.y_test, ae_test)
    plot_comparison_table(rows)

    if verbose:
        print(f"\nChosen threshold (val, max-F1): {ae_thr:.4f}")
        print(
            f"Autoencoder  ROC-AUC={ae_metrics.roc_auc:.3f}  "
            f"PR-AUC={ae_metrics.pr_auc:.3f}"
        )
        print("\n" + markdown_table(rows) + "\n")
        print("Saved figures -> figures/{error_distribution,confusion_matrix,"
              "roc_pr_curves,comparison_table}.png")

    return rows
