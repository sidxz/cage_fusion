#!/usr/bin/env python
"""
scripts/pretrain_admet_regression.py
=====================================
Stage 2 — Broad ADMET regression pretraining on TDC + MoleculeNet datasets.

Run AFTER Stage-1 classification pretraining.  Warm-starts from the
classification backbone (recommended) via --init-from-backbone.

Trains CAGEFusionForRegression on ~28k molecules across ~13 endpoints using
masked MSE (NaN targets are ignored per-task).  The pretrained backbone is
saved to /data-1/cage-fusion-pretrain/regression/checkpoints/ and can be
loaded with strict=False for any downstream fine-tuning task.

Usage
-----
    # Recommended: warm-start from Stage-1 classification backbone
    uv run python scripts/pretrain_admet_regression.py \\
        --init-from-backbone /data-1/cage-fusion-pretrain/classification/checkpoints/backbone.bin

    # From scratch (not recommended):
    uv run python scripts/pretrain_admet_regression.py

    # Skip dataset download if already cached:
    uv run python scripts/pretrain_admet_regression.py --skip-download \\
        --init-from-backbone /data-1/cage-fusion-pretrain/classification/checkpoints/backbone.bin

Output (all under /data-1/cage-fusion-pretrain/regression/)
------------------------------------------------------------
    checkpoints/
        best_model.pt          ← best val RMSE checkpoint (full .pt)
        pytorch_model.bin      ← HF-format weights (backbone + head)
        backbone.bin           ← encoder-only weights for cross-task loading
        config.json
        training_args.json
    features/
        train_cage_fusion.h5
        val_cage_fusion.h5
    logs/
        training_history.csv
        *.png  (loss/rmse/mae/r2 curves)
"""

from __future__ import annotations

import argparse
import logging
import os

import torch

from cage_fusion import AutoCageFusion, CageFusionConfig
from cage_fusion.data import CageFusionDataModule
from cage_fusion.training import Trainer, TrainingArguments
from cage_fusion.benchmarks.openadmet.data_loader import (
    build_pretrain_dataset,
    get_pretrain_label_cols,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pretrain_admet")


# ── Directories ───────────────────────────────────────────────────────────────

ROOT          = "/data-1/cage-fusion-pretrain/regression"
DATASET_DIR   = os.path.join(ROOT, "datasets")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
FEATURE_DIR   = os.path.join(ROOT, "features")
LOG_DIR       = os.path.join(ROOT, "logs")
MERGED_CSV    = os.path.join(DATASET_DIR, "pretrain_merged.csv")

MODEL_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"

# ── Config — FIXED: hidden_size=128 is the canonical backbone width ───────────

def build_config(num_labels: int, label_names: list[str]) -> CageFusionConfig:
    return CageFusionConfig(
        num_labels=num_labels,
        model_task="regression",
        label_names=label_names,
        model_checkpoint=MODEL_CHECKPOINT,
        # ── backbone (DO NOT CHANGE after pretraining) ──
        hidden_size=128,
        graph_dim=300,
        embedding_dim=384,
        aux_feature_dim=217,
        attn_mode="cross",
        num_heads=8,
        co_attention_layers=2,
        cross_attn_dropout=0.15,
        proj_dropout=0.10,
        fusion_dropout_1=0.3,
        fusion_dropout_2=0.2,
        use_fg_prompt=True,
        use_co_attention=True,
        use_aux_features=True,
        norm_type="layer",
    )


def parse_args():
    p = argparse.ArgumentParser(description="CAGEFusion broad ADMET pretraining")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--val-split",     type=float, default=0.15)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--bf16",          action="store_true",
                   help="Enable BF16 autocast (recommended on A6000 Ada / Ampere+).")
    p.add_argument("--skip-download", action="store_true",
                   help="Use cached pretrain_merged.csv if it exists.")
    p.add_argument("--skip-featurize", action="store_true",
                   help="Skip ChemBERTa featurisation if HDF5 caches already exist.")
    p.add_argument("--init-from-backbone", type=str, default=None,
                   help="Path to backbone.bin from Stage-1 classification to initialise the encoder.")
    p.add_argument("--push-to-hub",   action="store_true",
                   help="Push best checkpoint to cage-fusion/cage-fusion-pretrained.")
    p.add_argument("--hf-token",      type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()

    for d in [DATASET_DIR, CHECKPOINT_DIR, FEATURE_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load / build dataset ───────────────────────────────────────────────
    if args.skip_download and os.path.isfile(MERGED_CSV):
        import pandas as pd
        logger.info("Loading cached dataset from %s", MERGED_CSV)
        merged_df = pd.read_csv(MERGED_CSV)
    else:
        logger.info("Fetching TDC + MoleculeNet datasets…")
        merged_df = build_pretrain_dataset(
            tdc_cache=DATASET_DIR,
            moleculenet_cache=DATASET_DIR,
            output_csv=MERGED_CSV,
        )

    label_cols = get_pretrain_label_cols(merged_df)
    logger.info("Endpoints (%d): %s", len(label_cols), label_cols)
    logger.info("Dataset: %d molecules", len(merged_df))

    # ── 2. Build data module ──────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        merged_df, test_size=args.val_split, random_state=args.seed
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    logger.info("Train: %d  Val: %d", len(train_df), len(val_df))

    dm = CageFusionDataModule.from_dataframes(
        train_df=train_df,
        val_df=val_df,
        label_cols=label_cols,
        model_checkpoint=MODEL_CHECKPOINT,
        cache_dir=FEATURE_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        skip_featurize=args.skip_featurize,
    )

    # ── 3. Build model ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    config = build_config(len(label_cols), label_cols)
    if args.init_from_backbone:
        model = AutoCageFusion.from_backbone(args.init_from_backbone, config, device=device)
    else:
        model = AutoCageFusion.from_config(config).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %s total / %s trainable", f"{total:,}", f"{trainable:,}")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    train_args = TrainingArguments(
        output_dir=LOG_DIR,
        checkpoints_dir=CHECKPOINT_DIR,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        warmup_fraction=0.05,
        num_workers=args.num_workers,
        seed=args.seed,
        primary_metric="rmse",
        primary_metric_direction="min",
        bf16=args.bf16,
    )
    train_args.save(CHECKPOINT_DIR)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_loader=dm.train_loader,
        val_loader=dm.val_loader,
        device=device,
    )
    history = trainer.train()

    best_rmse = min(history["val_rmse"])
    logger.info("Pretraining complete.  Best val RMSE: %.4f", best_rmse)

    # ── 5. Save backbone ──────────────────────────────────────────────────────
    model.save_backbone(CHECKPOINT_DIR)
    logger.info("Backbone saved to %s/backbone.bin", CHECKPOINT_DIR)

    # ── 6. Save scaler ────────────────────────────────────────────────────────
    dm.save_scaler(CHECKPOINT_DIR)
    logger.info("Scaler saved to %s", CHECKPOINT_DIR)

    # ── 7. Push to HuggingFace ────────────────────────────────────────────────
    if args.push_to_hub:
        from cage_fusion import CageFusionPipeline
        url = CageFusionPipeline.push_to_hub(
            CHECKPOINT_DIR,
            repo_id="cage-fusion/cage-fusion-pretrained",
            model="best",
            token=args.hf_token,
            commit_message=(
                f"Broad ADMET pretraining: {len(label_cols)} endpoints, "
                f"{len(merged_df):,} molecules, TDC+MoleculeNet"
            ),
        )
        logger.info("Pushed to HuggingFace: %s", url)


if __name__ == "__main__":
    main()
