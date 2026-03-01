#!/usr/bin/env python
"""
scripts/pretrain_admet_classification.py
=========================================
Stage 1 — Broad ADMET classification pretraining on TDC datasets.

Trains CAGEFusionForMultiLabelClassification on ~90k molecules across 17
binary endpoints using masked BCE (NaN targets ignored per-task).

Either script can be run first or used to continue from the other's checkpoint
via --init-from-backbone.  The typical order is classification → regression,
but both scripts are symmetric and interchangeable.

Endpoints covered (17):
  Absorption:  HIA, Pgp_Inhibitor, Bioavailability, BBB
  CYP inh.:    CYP1A2, CYP2C9, CYP2C19, CYP2D6, CYP3A4
  CYP subs.:   CYP2C9_Substrate, CYP2D6_Substrate, CYP3A4_Substrate
  Toxicity:    AMES_Mutagenicity, DILI, Skin_Sensitizer, Carcinogen, ClinTox

Usage
-----
    # From scratch (downloads TDC data):
    uv run python scripts/pretrain_admet_classification.py

    # Warm-start from regression backbone:
    uv run python scripts/pretrain_admet_classification.py \\
        --init-from-backbone /data-1/cage-fusion-pretrain/regression/checkpoints/backbone.bin

    # Subsequent runs (cached data):
    uv run python scripts/pretrain_admet_classification.py --skip-download

Output (all under /data-1/cage-fusion-pretrain/classification/)
---------------------------------------------------------------
    checkpoints/
        best_model.pt          ← best val AUC checkpoint (full .pt)
        pytorch_model.bin      ← HF-format weights
        backbone.bin           ← encoder-only weights for cross-task loading
        config.json
        training_args.json
    features/
        train_cage_fusion.h5
        val_cage_fusion.h5
    logs/
        training_history.csv
        *.png  (loss/auc/mcc/pr curves)
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
    build_pretrain_classification_dataset,
    get_pretrain_label_cols,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pretrain_admet_classification")


# ── Checkpoint load report ────────────────────────────────────────────────────

_SUBMODULE_LABELS = {
    "graph_encoder":  "Graph encoder (D-MPNN)",
    "graph_proj":     "Graph projection",
    "embedding_proj": "Sequence projection (ChemBERTa)",
    "both_proj":      "Self-attn both projection",
    "co_attn_layers": "Co-attention layers",
    "aux_mlp":        "Auxiliary feature MLP",
    "fg_prompter":    "Functional group prompt",
    "fusion":         "Fusion MLP",
    "scale_graph":    "Scale params (graph/attn/aux/fg)",
    "scale_attn":     "(merged above)",
    "scale_aux":      "(merged above)",
    "alpha":          "(merged above)",
}
_SCALE_GROUP = {"scale_graph", "scale_attn", "scale_aux", "alpha"}


def _print_checkpoint_report(
    state: dict, missing: list, unexpected: list, model,
    shape_skipped: list | None = None,
) -> None:
    """Print a rich table showing which components were loaded from checkpoint."""
    from rich import box as rich_box
    from rich.console import Console
    from rich.table import Table

    loaded: dict[str, int] = {}
    scale_params = 0
    for k, v in state.items():
        parts = k.split(".")
        sub = parts[1] if len(parts) > 1 else parts[0]
        if sub in _SCALE_GROUP:
            scale_params += v.numel()
        else:
            loaded.setdefault(sub, 0)
            loaded[sub] += v.numel()
    if scale_params:
        loaded["scale_graph"] = scale_params

    fresh: dict[str, int] = {}
    model_sd = model.state_dict()
    for k in list(missing) + list(shape_skipped or []):
        sub = k.split(".")[0]
        fresh.setdefault(sub, 0)
        if k in model_sd:
            fresh[sub] += model_sd[k].numel()

    total_loaded = sum(loaded.values())
    total_fresh  = sum(fresh.values())

    console = Console()
    table = Table(
        title="[bold cyan]Pretrained Encoder Weights — Load Report[/bold cyan]",
        box=rich_box.ROUNDED,
        border_style="cyan",
        header_style="bold",
        show_header=True,
    )
    table.add_column("Component",  min_width=36)
    table.add_column("Source",     justify="center", min_width=22)
    table.add_column("Parameters", justify="right",  min_width=12)

    for sub, n in sorted(loaded.items()):
        if _SUBMODULE_LABELS.get(sub) == "(merged above)":
            continue
        label = _SUBMODULE_LABELS.get(sub, sub)
        table.add_row(label, "[green]checkpoint[/green]", f"{n:,}")

    for sub, n in sorted(fresh.items()):
        table.add_row(sub, "[yellow]fresh init[/yellow]", f"{n:,}")

    for k in unexpected:
        table.add_row(k, "[red]unexpected / skipped[/red]", "—")

    if shape_skipped:
        table.add_row(
            f"[dim]{len(shape_skipped)} shape-mismatched keys[/dim]",
            "[dim]re-initialized[/dim]", "—",
        )

    table.add_section()
    table.add_row("[bold]Total from checkpoint[/bold]", "", f"[green]{total_loaded:,}[/green]")
    table.add_row("[bold]Total fresh init[/bold]",      "", f"[yellow]{total_fresh:,}[/yellow]")

    console.print()
    console.print(table)
    console.print()


# ── Directories ───────────────────────────────────────────────────────────────

ROOT           = "/data-1/cage-fusion-pretrain/classification"
DATASET_DIR    = os.path.join(ROOT, "datasets")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
FEATURE_DIR    = os.path.join(ROOT, "features")
LOG_DIR        = os.path.join(ROOT, "logs")
MERGED_CSV     = os.path.join(DATASET_DIR, "pretrain_classification_merged.csv")

MODEL_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"


# ── Config — must match Stage-1a backbone width ───────────────────────────────

def build_config(num_labels: int, label_names: list[str]) -> CageFusionConfig:
    return CageFusionConfig(
        num_labels=num_labels,
        model_task="classification",
        label_names=label_names,
        model_checkpoint=MODEL_CHECKPOINT,
        # ── backbone (fixed width — must match across all pretraining stages) ──
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
    p = argparse.ArgumentParser(description="CAGEFusion Stage-1: ADMET classification pretraining")
    p.add_argument("--epochs",              type=int,   default=50)
    p.add_argument("--batch-size",          type=int,   default=64)
    p.add_argument("--lr",                  type=float, default=3e-4)
    p.add_argument("--weight-decay",        type=float, default=1e-4)
    p.add_argument("--val-split",           type=float, default=0.15)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--num-workers",         type=int,   default=4)
    p.add_argument("--bf16",                action="store_true",
                   help="Enable BF16 autocast (Ampere+ GPUs).")
    p.add_argument("--skip-download",       action="store_true",
                   help="Use cached pretrain_classification_merged.csv if it exists.")
    p.add_argument("--skip-featurize",      action="store_true",
                   help="Skip ChemBERTa featurisation if HDF5 caches already exist.")
    p.add_argument("--init-from-backbone",  type=str, default=None,
                   help="Path to backbone.bin from a prior pretraining run to warm-start the encoder.")
    p.add_argument("--push-to-hub",         action="store_true",
                   help="Push best checkpoint to HuggingFace Hub.")
    p.add_argument("--hf-token",            type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()

    for d in [DATASET_DIR, CHECKPOINT_DIR, FEATURE_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── 1. Load / build dataset ───────────────────────────────────────────────
    if args.skip_download and os.path.isfile(MERGED_CSV):
        import pandas as pd
        logger.info("Loading cached classification dataset from %s", MERGED_CSV)
        merged_df = pd.read_csv(MERGED_CSV)
    else:
        logger.info("Fetching TDC classification datasets…")
        merged_df = build_pretrain_classification_dataset(
            tdc_cache=DATASET_DIR,
            output_csv=MERGED_CSV,
        )

    label_cols = get_pretrain_label_cols(merged_df)
    logger.info("Endpoints (%d): %s", len(label_cols), label_cols)
    logger.info("Dataset: %d molecules", len(merged_df))

    # ── 2. Train / val split ──────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        merged_df, test_size=args.val_split, random_state=args.seed
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    logger.info("Train: %d  Val: %d", len(train_df), len(val_df))

    # ── 3. Build data module ──────────────────────────────────────────────────
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

    # ── 4. Build model ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    config = build_config(len(label_cols), label_cols)
    model  = AutoCageFusion.from_config(config).to(device)

    # Optionally warm-start encoder from a prior pretraining checkpoint
    if args.init_from_backbone:
        if os.path.isfile(args.init_from_backbone):
            state = torch.load(args.init_from_backbone, map_location="cpu")
            # Filter shape-mismatched keys (e.g. graph encoder's task-specific
            # predictor head, which is sized by num_labels in the source task)
            model_sd = model.state_dict()
            shape_skipped = [
                k for k, v in state.items()
                if k in model_sd and v.shape != model_sd[k].shape
            ]
            compatible = {
                k: v for k, v in state.items()
                if k not in shape_skipped
            }
            missing, unexpected = model.load_state_dict(compatible, strict=False)
            _print_checkpoint_report(compatible, missing, unexpected, model,
                                     shape_skipped=shape_skipped)
        else:
            logger.warning("--init-from-backbone path not found: %s", args.init_from_backbone)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %s total / %s trainable", f"{total:,}", f"{trainable:,}")

    # ── 5. Train ──────────────────────────────────────────────────────────────
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
        primary_metric="auc",
        primary_metric_direction="max",
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

    best_auc = max(history["val_auc"])
    logger.info("Classification pretraining complete.  Best val AUC: %.4f", best_auc)

    # ── 6. Save backbone ──────────────────────────────────────────────────────
    model.save_backbone(CHECKPOINT_DIR)
    logger.info("Backbone saved to %s/backbone.bin", CHECKPOINT_DIR)

    # ── 7. Save scaler ────────────────────────────────────────────────────────
    dm.save_scaler(CHECKPOINT_DIR)

    # ── 8. Push to HuggingFace ────────────────────────────────────────────────
    if args.push_to_hub:
        from cage_fusion import CageFusionPipeline
        url = CageFusionPipeline.push_to_hub(
            CHECKPOINT_DIR,
            repo_id="cage-fusion/cage-fusion-pretrained-classification",
            model="best",
            token=args.hf_token,
            commit_message=(
                f"Stage-1b classification pretraining: {len(label_cols)} endpoints, "
                f"{len(merged_df):,} molecules, TDC"
            ),
        )
        logger.info("Pushed to HuggingFace: %s", url)


if __name__ == "__main__":
    main()
