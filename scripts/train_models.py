#!/usr/bin/env python3
"""Retrain and persist all modeling artifacts used by the Streamlit app.

Usage (from repo root):
    python scripts/train_models.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from content.modelisation import MODEL_CHOICES, _compute_metrics, _predict, _train_and_save


def main():
    for choice in MODEL_CHOICES:
        print(f"Training {choice}…")
        artifact, _X_tr, X_te, _y_tr, y_te, _X_tr_sc, X_te_sc, cutoff = _train_and_save(choice)
        y_pred = _predict(artifact, X_te_sc, X_te)
        metrics = _compute_metrics(y_te, y_pred)
        print(
            f"  cutoff={cutoff.date()}  "
            f"R²={metrics['R²']:.4f}  "
            f"RMSE=${metrics['RMSE ($)']:,.0f}  "
            f"MAE=${metrics['MAE ($)']:,.0f}"
        )
    print("Done. Artifacts written to models/.")


if __name__ == "__main__":
    main()
