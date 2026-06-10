# Credit-Card Fraud Detection with a Deep Autoencoder

> MRP — *Artificial Intelligence*, Sofia University (Dr. Petiushko)

Unsupervised / semi-supervised fraud detection on credit-card transactions.
An undercomplete autoencoder is trained on **normal transactions only**; at test time a
transaction's **reconstruction error** is used as an anomaly score — fraud reconstructs
poorly because the model never learned it. An **Isolation Forest** is included as a
non-deep baseline to keep the comparison honest.

---

## How it works (plain English)

Fraud is rare (~1 in 600 here) and labels are scarce, so instead of teaching a model what
fraud looks like, we teach it what **normal** looks like — extremely well. An autoencoder
compresses each transaction to a 7-number summary and rebuilds it. Trained on normal
transactions only, it rebuilds normal ones accurately (low error) but fumbles fraud it
never saw (high error). That **reconstruction error is the fraud alarm score**: above the
threshold → **FLAGGED**.

---

## Results (autoencoder vs. baseline)

| Model | Precision | Recall | F1 | PR-AUC |
|-------|-----------|--------|----|--------|
| Autoencoder | 0.750 | 0.683 | 0.715 | **0.718** |
| Isolation Forest | 0.376 | 0.451 | 0.410 | 0.378 |

> Autoencoder ROC-AUC = 0.935. Threshold (max-F1 on validation) = 4.65.

> Because fraud is ~0.17% of all transactions, **accuracy is meaningless** here —
> **PR-AUC** is the headline metric.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

### Dataset (manual download — required)

The dataset is **not** included (it is ~150 MB and gitignored).

1. Download **"Credit Card Fraud Detection"** by `mlg-ulb` from Kaggle:
   <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
2. Place the file at: **`data/creditcard.csv`**

The code checks for this file and prints a clear message if it is missing.

---

## How to run

Run the full pipeline (load → split → train → evaluate → baseline → save figures):

```bash
python main.py
```

Runs in ~90s on CPU. Figures are written to `figures/`; the trained model is saved to
`figures/autoencoder.pt`.

**Method details:**
- **Split (leakage-safe):** train = ~80% of normal only; validation = ~10% normal + half the
  fraud (used to pick the threshold); test = remaining ~10% normal + remaining fraud (final
  metrics only). `StandardScaler` is fit on **training-normal only**. Seed = 42.
- **Model:** undercomplete autoencoder `29 → 20 → 14 → 7 → 14 → 20 → 29`, ReLU hidden /
  linear output, MSE loss, Adam, 40 epochs, batch 256.
- **Threshold:** the value maximizing F1 on validation, then frozen for test.

### Demo app (for the recorded walkthrough)

```bash
streamlit run app/demo_app.py
```

Pick or sample a transaction, see its reconstruction error against the chosen threshold,
and the verdict (**FLAGGED** / **NORMAL**).

### Tests

```bash
pytest
```

23 tests: structural (files/config), data split & no-leakage, model shapes, threshold &
metrics, and that all plots are produced.

---

## Links

- **Slide deck (PDF):** included in this repository
- **Demo video:** <https://youtu.be/MtDNSEQ_yB8>
- **Code:** this repository

---

## Project layout

```
.
├── main.py              # runs the full pipeline top-to-bottom
├── src/
│   ├── data.py          # load + preprocess + leakage-safe split
│   ├── model.py         # autoencoder definition
│   ├── train.py         # training loop
│   └── evaluate.py      # metrics + plots + baseline
├── app/demo_app.py      # Streamlit demo
├── tests/               # structural tests
├── figures/             # generated plots
├── slides/              # final PDF slide deck
└── data/                # creditcard.csv goes here (gitignored)
```

## Figures (in `figures/`)

| File | What it shows |
|------|----------------|
| `loss_curve.png` | Train vs validation reconstruction loss per epoch |
| `error_distribution.png` | Reconstruction error, normal vs fraud (the key separation plot) |
| `confusion_matrix.png` | Confusion matrix at the chosen threshold |
| `roc_pr_curves.png` | ROC and Precision-Recall curves |
| `comparison_table.png` | Autoencoder vs Isolation Forest |

## References

1. Dal Pozzolo, Caelen, Johnson, Bontempi (2015). *Calibrating Probability with
   Undersampling for Unbalanced Classification.* IEEE SSCI/CIDM. — source of the dataset.
2. Liu, Ting, Zhou (2008). *Isolation Forest.* IEEE ICDM, 413–422. — the baseline.
3. Sakurada & Yairi (2014). *Anomaly Detection Using Autoencoders with Nonlinear
   Dimensionality Reduction.* MLSDA Workshop (ACM). — closest method to this work.
4. An & Cho (2015). *Variational Autoencoder based Anomaly Detection using Reconstruction
   Probability.* SNU technical report.
5. Hawkins, He, Williams, Baxter (2002). *Outlier Detection Using Replicator Neural
   Networks.* DaWaK.

Dataset: *Credit Card Fraud Detection* (mlg-ulb / ULB-Worldline),
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>.
