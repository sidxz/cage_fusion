#!/usr/bin/env python
"""
scripts/evaluate_openadmet.py
==============================
Evaluate a fine-tuned CAGEFusion model on the official OpenADMET test set
and compare MA-RAE against the leaderboard.

Produces:
  - Console report with per-endpoint metrics + delta vs leaderboard #1
  - /data-1/cage-fusion-admet/submissions/evaluation_report.csv
  - /data-1/cage-fusion-admet/submissions/submission.csv  (original units)

Usage
-----
    # Evaluate the best fine-tuned model:
    python scripts/evaluate_openadmet.py

    # Evaluate a specific checkpoint directory:
    python scripts/evaluate_openadmet.py --checkpoint /data-1/cage-fusion-admet/checkpoints

    # Also print per-endpoint scatter (requires matplotlib):
    python scripts/evaluate_openadmet.py --plot
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cage_fusion import AutoCageFusion
from cage_fusion.data import CageFusionDataModule
from cage_fusion.utils.device_utils import move_bmg_to_device
from benchmarks.openadmet.data_loader import load_openadmet, OPENADMET_LABEL_COLS
from benchmarks.openadmet.preprocessing import forward_transform, inverse_transform
from benchmarks.openadmet.marae import compute_marae, print_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate_openadmet")

# ── Directories ───────────────────────────────────────────────────────────────

ROOT           = "/data-1/cage-fusion-admet"
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
FEATURE_DIR    = os.path.join(ROOT, "features")
SUBMISSION_DIR = os.path.join(ROOT, "submissions")

MODEL_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"
LABEL_COLS       = OPENADMET_LABEL_COLS

LEADERBOARD_TOP1 = 0.5113   # pebble, MA-RAE on full test set


def parse_args():
    p = argparse.ArgumentParser(description="OpenADMET evaluation + submission")
    p.add_argument("--checkpoint",   type=str, default=CHECKPOINT_DIR,
                   help="Path to the fine-tuned checkpoint directory.")
    p.add_argument("--batch-size",   type=int, default=64)
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--plot",         action="store_true",
                   help="Save per-endpoint scatter plots.")
    p.add_argument("--no-leaderboard", action="store_true",
                   help="Suppress leaderboard comparison line.")
    return p.parse_args()


def run_inference(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run model on a DataLoader; return (preds, labels, smiles)."""
    all_preds, all_labels, all_smiles = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            bmg, token_embs, attn_mask, aux_feats, labels, input_ids, smiles_batch, _, _ = batch
            bmg        = move_bmg_to_device(bmg, device)
            token_embs = token_embs.to(device)
            attn_mask  = attn_mask.to(device)
            aux_feats  = aux_feats.to(device)
            input_ids  = input_ids.to(device)
            labels     = labels.to(device)

            out = model(
                bmg=bmg,
                sequence_embeddings=token_embs,
                attn_mask=attn_mask,
                aux_feats=aux_feats,
                input_ids_batch=input_ids,
                smiles_batch=smiles_batch,
            )
            all_preds.append(out.logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_smiles.extend(smiles_batch)

    return (
        np.vstack(all_preds),
        np.vstack(all_labels),
        all_smiles,
    )


def save_scatter_plots(y_true, y_pred, label_names, output_dir):
    """Save one scatter plot per endpoint."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    for i, name in enumerate(label_names):
        mask = ~np.isnan(y_true[:, i])
        yt, yp = y_true[mask, i], y_pred[mask, i]
        if len(yt) < 2:
            continue
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(yt, yp, alpha=0.4, s=12)
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("True (log scale)")
        ax.set_ylabel("Predicted (log scale)")
        ax.set_title(name)
        plt.tight_layout()
        safe = name.replace("/", "_").replace("-", "_")
        plt.savefig(os.path.join(output_dir, f"{safe}.png"), dpi=120)
        plt.close()


def main():
    args = parse_args()
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── 1. Load official test set ─────────────────────────────────────────────
    _, test_df_raw = load_openadmet(
        cache_dir=os.path.join(ROOT, "datasets")
    )
    # Save molecule names before transform
    mol_names = test_df_raw.get("Molecule_Name", pd.Series(range(len(test_df_raw)))).tolist()

    # Forward-transform labels for metric computation
    test_df_transformed = forward_transform(test_df_raw.copy(), cols=LABEL_COLS)

    # ── 2. Build inference data module ────────────────────────────────────────
    # Load the scaler that was fitted on the training data during fine-tuning
    import joblib
    scaler_path = os.path.join(args.checkpoint, "aux_features_scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.isfile(scaler_path) else None
    if scaler is None:
        logger.warning("No aux_features_scaler.pkl found — aux features will be re-scaled.")

    dm = CageFusionDataModule.for_inference(
        csv_path=_df_to_temp_csv(test_df_transformed),
        label_cols=LABEL_COLS,
        model_checkpoint=MODEL_CHECKPOINT,
        scaler=scaler,
        cache_dir=os.path.join(FEATURE_DIR, "eval"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── 3. Load model ─────────────────────────────────────────────────────────
    logger.info("Loading model from %s", args.checkpoint)
    model = AutoCageFusion.from_pretrained(args.checkpoint).to(device)
    logger.info("Model type: %s", type(model).__name__)

    # ── 4. Run inference ──────────────────────────────────────────────────────
    preds_log, labels_log, smiles = run_inference(model, dm.test_loader, device)

    # ── 5. Compute MA-RAE (log scale) ─────────────────────────────────────────
    results = compute_marae(labels_log, preds_log, label_names=LABEL_COLS)
    print("\n" + "=" * 60)
    print("  OpenADMET ExpansionRx — Evaluation Report")
    print("=" * 60)
    print_report(results, leaderboard_top=None if args.no_leaderboard else LEADERBOARD_TOP1)
    print("=" * 60 + "\n")

    # ── 6. Save evaluation report CSV ─────────────────────────────────────────
    report_rows = []
    for name, m in results["per_endpoint"].items():
        report_rows.append({"endpoint": name, **m})
    report_df = pd.DataFrame(report_rows)
    report_df.loc[len(report_df)] = {
        "endpoint": "MA-RAE", "n": "", "mae": "", "rae": results["ma_rae"],
        "r2": "", "spearman": "", "kendall": "",
    }
    report_path = os.path.join(SUBMISSION_DIR, "evaluation_report.csv")
    report_df.to_csv(report_path, index=False)
    logger.info("Evaluation report saved to %s", report_path)

    # ── 7. Generate submission CSV (original units) ───────────────────────────
    preds_orig = inverse_transform(preds_log, cols=LABEL_COLS)
    sub_df = pd.DataFrame({"Molecule_Name": mol_names})
    for i, col in enumerate(LABEL_COLS):
        sub_df[col] = preds_orig[:, i]
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    logger.info("Submission CSV saved to %s", sub_path)

    # ── 8. Optional scatter plots ─────────────────────────────────────────────
    if args.plot:
        plot_dir = os.path.join(SUBMISSION_DIR, "scatter_plots")
        save_scatter_plots(labels_log, preds_log, LABEL_COLS, plot_dir)
        logger.info("Scatter plots saved to %s", plot_dir)


def _df_to_temp_csv(df: pd.DataFrame) -> str:
    """Write a DataFrame to a temp CSV and return the path."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False,
        dir=os.path.join(ROOT, "datasets"),
    )
    df.to_csv(tmp.name, index=False)
    return tmp.name


if __name__ == "__main__":
    main()
