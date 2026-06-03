"""Streamlit demo for the fraud autoencoder.

A small interactive app for the recorded walkthrough: sample a transaction (random,
or specifically a fraud / normal one), see its reconstruction error against the chosen
threshold, and the verdict — FLAGGED or NORMAL.

Run from the project root:

    streamlit run app/demo_app.py

Requires a trained model at ``figures/autoencoder.pt`` (run ``python main.py`` first)
and the dataset at ``data/creditcard.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

# Make the project root importable so ``import src...`` works under streamlit.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_data  # noqa: E402
from src.evaluate import ae_scores, best_f1_threshold  # noqa: E402
from src.model import Autoencoder  # noqa: E402
from src.train import WEIGHTS_PATH  # noqa: E402


@st.cache_resource
def load_everything():
    """Load data, the trained model, and compute the val-chosen threshold once."""
    ds = load_data()
    model = Autoencoder(n_features=ds.n_features)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    # Threshold is chosen on validation (max F1) — same as in evaluation.
    val_scores = ae_scores(model, ds.X_val)
    threshold = best_f1_threshold(ds.y_val, val_scores)

    # Pre-compute test scores so sampling is instant.
    test_scores = ae_scores(model, ds.X_test)
    return ds, model, threshold, test_scores


def pick_index(y_test: np.ndarray, mode: str) -> int:
    """Pick a random test-set row index matching the requested class."""
    if mode == "Random fraud":
        pool = np.where(y_test == 1)[0]
    elif mode == "Random normal":
        pool = np.where(y_test == 0)[0]
    else:  # "Random (any)"
        pool = np.arange(len(y_test))
    return int(np.random.choice(pool))


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Fraud Autoencoder Demo",
    page_icon=":material/credit_card:",
    layout="wide",
)

st.subheader(":material/credit_card: Credit-Card Fraud Detection — Autoencoder")
# Center the page title (st.subheader renders as an <h3>).
st.markdown("<style>h3 { text-align: center; }</style>", unsafe_allow_html=True)
st.markdown(
    "The autoencoder was trained on **normal transactions only**. A transaction's "
    "**reconstruction error** is its anomaly score — fraud reconstructs poorly, so a "
    "high error gets **FLAGGED**."
)

ds, model, threshold, test_scores = load_everything()

st.sidebar.header("Pick a transaction")
mode = st.sidebar.radio(
    "Sample from the test set:",
    ["Random fraud", "Random normal", "Random (any)"],
)
if (
    st.sidebar.button("Sample a transaction", icon=":material/shuffle:")
    or "idx" not in st.session_state
):
    st.session_state.idx = pick_index(ds.y_test, mode)

idx = st.session_state.idx
score = float(test_scores[idx])
actual = int(ds.y_test[idx])
flagged = score >= threshold

# --- Verdict ---
col1, col2, col3 = st.columns(3)
col1.metric("Reconstruction error", f"{score:.3f}")
col2.metric("Threshold", f"{threshold:.3f}")
verdict_md = (
    ":red[:material/warning: **FLAGGED**]"
    if flagged
    else ":green[:material/check_circle: **NORMAL**]"
)
col3.markdown("Model verdict")
col3.markdown(f"## {verdict_md}")

# Error vs threshold as a simple progress-style bar (capped at 2x threshold for display).
ratio = min(score / (2 * threshold), 1.0)
st.progress(ratio, text=f"error / (2× threshold) = {score:.2f} / {2 * threshold:.2f}")

# --- Was it right? (we know the true label for the demo) ---
truth = "FRAUD" if actual == 1 else "NORMAL"
correct = (actual == 1) == flagged
if correct:
    st.success(
        f"Ground truth: **{truth}** — model was **correct**",
        icon=":material/check_circle:",
    )
else:
    st.error(
        f"Ground truth: **{truth}** — model was **wrong**",
        icon=":material/cancel:",
    )

# --- Per-feature reconstruction (which features the model could not reproduce) ---
with st.expander("See the transaction's feature values and reconstruction"):
    x = torch.from_numpy(ds.X_test[idx : idx + 1])
    with torch.no_grad():
        recon = model(x).numpy().ravel()
    original = ds.X_test[idx]
    per_feat_err = (original - recon) ** 2

    import pandas as pd

    table = pd.DataFrame(
        {
            "feature": ds.feature_names,
            "value (scaled)": original,
            "reconstructed": recon,
            "squared error": per_feat_err,
        }
    ).sort_values("squared error", ascending=False)
    st.caption("Sorted by squared error — the features the model reconstructed worst.")
    st.dataframe(table, use_container_width=True, hide_index=True)

st.caption(
    f"Test-set index {idx}. Threshold chosen on validation (max F1). "
    "Values are standardized (mean 0, std 1)."
)
