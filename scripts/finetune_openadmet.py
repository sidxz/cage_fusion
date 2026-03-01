#!/usr/bin/env python
"""
scripts/finetune_openadmet.py
==============================
Stage 2 — Fine-tune CAGEFusion on the OpenADMET ExpansionRx benchmark.

Two-phase training:
  Phase A (head warmup)  — backbone frozen, only the 9-label regression head trains.
  Phase B (full finetune)— backbone unfrozen, full model trained with MA-RAE as
                           the checkpoint selection metric.

Prerequisite: run scripts/pretrain_admet.py first to produce the pretrained
backbone at /data-1/cage-fusion-pretrain/checkpoints/.

Usage
-----
    # Fine-tune from pretrained backbone (recommended):
    python scripts/finetune_openadmet.py

    # Fine-tune from scratch (no pretraining):
    python scripts/finetune_openadmet.py --from-scratch

    # Custom options:
    python scripts/finetune_openadmet.py \\
        --epochs-A 5 --epochs-B 80 \\
        --lr-A 1e-3 --lr-B 3e-4 \\
        --seed 42

Output (all under /data-1/cage-fusion-admet/)
----------------------------------------------
    checkpoints/
        best_model.pt          ← best val MA-RAE checkpoint
        pytorch_model.bin      ← HF-format weights
        backbone.bin
        config.json
        training_args.json
    features/
        train_cage_fusion.h5
        val_cage_fusion.h5
        test_cage_fusion.h5
    logs/
        training_history.csv
        *.png
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
from sklearn.model_selection import train_test_split

from cage_fusion import AutoCageFusion, CageFusionConfig
from cage_fusion.data import CageFusionDataModule
from cage_fusion.training import Trainer, TrainingArguments
from cage_fusion.benchmarks.openadmet.data_loader import load_openadmet, OPENADMET_LABEL_COLS
from cage_fusion.benchmarks.openadmet.preprocessing import forward_transform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finetune_openadmet")

# ── Directories ───────────────────────────────────────────────────────────────

PRETRAIN_CKPT  = "/data-1/cage-fusion-pretrain/regression/checkpoints"
ROOT           = "/data-1/cage-fusion-admet"
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
FEATURE_DIR    = os.path.join(ROOT, "features")
LOG_DIR        = os.path.join(ROOT, "logs")

MODEL_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"
LABEL_COLS       = OPENADMET_LABEL_COLS


def build_finetune_config() -> CageFusionConfig:
    """Build a fine-tuning config that matches the pretrained backbone width."""
    return CageFusionConfig(
        num_labels=len(LABEL_COLS),
        model_task="regression",
        label_names=LABEL_COLS,
        model_checkpoint=MODEL_CHECKPOINT,
        # ── must match pretrained backbone exactly ──
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
    p = argparse.ArgumentParser(description="CAGEFusion OpenADMET fine-tuning")
    p.add_argument("--epochs-A",      type=int,   default=5,
                   help="Head-warmup epochs (backbone frozen).")
    p.add_argument("--epochs-B",      type=int,   default=35,
                   help="Full fine-tuning epochs (backbone unfrozen).")
    p.add_argument("--lr-A",          type=float, default=1e-3,
                   help="Learning rate for Phase A.")
    p.add_argument("--lr-B",          type=float, default=3e-4,
                   help="Learning rate for Phase B.")
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--val-split",     type=float, default=0.15,
                   help="Fraction of training data to hold out for validation.")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--bf16",          action="store_true",
                   help="Enable BF16 autocast (recommended on A6000 Ada / Ampere+).")
    p.add_argument("--from-scratch",  action="store_true",
                   help="Skip backbone loading; train from random initialisation.")
    p.add_argument("--pretrain-ckpt", type=str,   default=PRETRAIN_CKPT,
                   help="Path to pretrained checkpoint directory.")
    return p.parse_args()


def main():
    args = parse_args()

    for d in [CHECKPOINT_DIR, FEATURE_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── 1. Load official training DataFrame only ──────────────────────────────
    # The test set is never loaded here — it is handled exclusively by
    # evaluate_openadmet.py after training is complete.
    train_raw, _ = load_openadmet(
        cache_dir=os.path.join(ROOT, "datasets")
    )

    # ── 2. Log-transform label columns (forward transform) ────────────────────
    train_raw = forward_transform(train_raw, cols=LABEL_COLS)

    # ── 3. Carve validation set from training data ────────────────────────────
    # We hold out a small fraction for early stopping / checkpoint selection.
    # The official test set is never touched during training.
    train_df, val_df = train_test_split(
        train_raw, test_size=args.val_split, random_state=args.seed
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    logger.info(
        "Split: %d train / %d val (official test set withheld)",
        len(train_df), len(val_df),
    )

    # ── 4. Build data module ──────────────────────────────────────────────────
    dm = CageFusionDataModule.from_dataframes(
        train_df=train_df,
        val_df=val_df,
        label_cols=LABEL_COLS,
        model_checkpoint=MODEL_CHECKPOINT,
        cache_dir=FEATURE_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── 5. Build model ────────────────────────────────────────────────────────
    config = build_finetune_config()

    if args.from_scratch:
        logger.info("Training from scratch (no pretrained backbone).")
        model = AutoCageFusion.from_config(config).to(device)
    else:
        pretrain_ckpt = args.pretrain_ckpt
        if not os.path.isdir(pretrain_ckpt):
            logger.warning(
                "Pretrained checkpoint not found at '%s'. "
                "Run scripts/pretrain_admet_regression.py first, or use --from-scratch.",
                pretrain_ckpt,
            )
            sys.exit(1)
        logger.info("Loading pretrained backbone from %s", pretrain_ckpt)
        model = AutoCageFusion.from_backbone(pretrain_ckpt, config, device=device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %s total / %s trainable", f"{total:,}", f"{trainable:,}")

    # ── 6. Phase A — head-only warmup ────────────────────────────────────────
    if not args.from_scratch and args.epochs_A > 0:
        logger.info("=== Phase A: head warmup (%d epochs, backbone frozen) ===",
                    args.epochs_A)
        model.freeze_backbone()
        trainable_A = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Phase A trainable params: %s", f"{trainable_A:,}")

        ckpt_A = os.path.join(CHECKPOINT_DIR, "phaseA")
        os.makedirs(ckpt_A, exist_ok=True)

        args_A = TrainingArguments(
            output_dir=os.path.join(LOG_DIR, "phaseA"),
            checkpoints_dir=ckpt_A,
            num_epochs=args.epochs_A,
            batch_size=args.batch_size,
            learning_rate=args.lr_A,
            weight_decay=args.weight_decay,
            max_grad_norm=1.0,
            warmup_fraction=0.0,
            num_workers=args.num_workers,
            seed=args.seed,
            primary_metric="rmse",
            primary_metric_direction="min",
            bf16=args.bf16,
        )
        Trainer(
            model=model,
            args=args_A,
            train_loader=dm.train_loader,
            val_loader=dm.val_loader,
            device=device,
        ).train()

        model.unfreeze_backbone()
        logger.info("Phase A complete. Backbone unfrozen for Phase B.")

    # ── 7. Phase B — full fine-tuning ─────────────────────────────────────────
    logger.info("=== Phase B: full fine-tuning (%d epochs) ===", args.epochs_B)

    args_B = TrainingArguments(
        output_dir=LOG_DIR,
        checkpoints_dir=CHECKPOINT_DIR,
        num_epochs=args.epochs_B,
        batch_size=args.batch_size,
        learning_rate=args.lr_B,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        warmup_fraction=0.05,
        num_workers=args.num_workers,
        seed=args.seed,
        primary_metric="marae",           # leaderboard metric
        primary_metric_direction="min",
        bf16=args.bf16,
    )
    args_B.save(CHECKPOINT_DIR)

    history = Trainer(
        model=model,
        args=args_B,
        train_loader=dm.train_loader,
        val_loader=dm.val_loader,
        device=device,
    ).train()

    best_marae = min(history["val_marae"])
    logger.info("Fine-tuning complete.  Best val MA-RAE: %.4f", best_marae)

    # ── 8. Save backbone + scaler ─────────────────────────────────────────────
    model.save_backbone(CHECKPOINT_DIR)
    dm.save_scaler(CHECKPOINT_DIR)
    logger.info("Backbone and scaler saved to %s", CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
