#!/usr/bin/env python3
"""
MoleculeNet benchmark for CAGE-Fusion — single-phase training
Matches paths, variables, and loader setup from the phased script, but trains in one pass.
"""

import os
import sys
import glob
import json
import shutil
import random
import argparse
import traceback
from functools import partial
from collections import Counter

import numpy as np
import pandas as pd
import torch
import joblib
import deepchem as dc
from deepchem.feat import RawFeaturizer

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from rich.console import Console
from rich.table import Table
from rich.traceback import install

# ========= Project path setup (match training/phased script) =========
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../cage_fusion")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ========= Local imports =========
from cage_fusion.configs import get_default_config
from cage_fusion.featurizers import featurize_and_save_streaming
from cage_fusion.models import CAGEFusionModel
from cage_fusion.engine.training import train_model
from cage_fusion.engine.evaluation import evaluate_model
from cage_fusion.engine.dataset import CageFusionStreamingDataset
from cage_fusion.engine.data_utils import collate_fn_for_cage_fusion
from cage_fusion.engine.utils import move_bmg_to_device, compute_pos_weight_from_h5
from cage_fusion.engine.logging import plot_confusion_matrix
from cage_fusion.utils.model_utils import load_partial_weights
from cage_fusion.utils.logging_utils import logger

# ========= Console / rich setup =========
install()
console = Console()

# ========= Run + directory config (parity with phased script) =========
USE_CO_ATTENTION = True
ATTN_MODE = "cross"  # 'cross' | 'self_tokens' | 'self_graph' | 'self_both'
USE_AUX_FEATURES = True
USE_FG_PROMPT = True
EMBEDDING_MODEL = "bert"
CO_ATTENTION_LAYERS = 1

DEFAULT_DATASET = "bace_classification"
DEFAULT_SEED = 54
DEFAULT_FORCE_RERUN = False
DEFAULT_RERUN_TRAIN = False
DEFAULT_SPLITTER = "scaffold"

# Single-phase knobs
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 0.005
DEFAULT_NUM_EPOCHS = 35
DEFAULT_WARMUP_FRACTION = 0.1  # fraction of total steps

DIRNAME = (
    f"Benchmark-{DEFAULT_DATASET}-{CO_ATTENTION_LAYERS}-{EMBEDDING_MODEL}-v4"
)
DATA_ROOT = "data"
DATA_DIR = os.path.join(DATA_ROOT, "molnet")
CACHE_ROOT = os.path.join("cache", EMBEDDING_MODEL, DIRNAME)
FEATURES_ROOT = os.path.join(DATA_ROOT, "features", EMBEDDING_MODEL, DIRNAME)
CHECKPOINTS_ROOT = os.path.join("checkpoints", EMBEDDING_MODEL, DIRNAME)
OUTPUT_ROOT = os.path.join("output", EMBEDDING_MODEL, DIRNAME)

MODEL_BEST = "best_model.pt"
MODEL_LATEST = "latest_checkpoint.pt"
USE_PRETRAINED_WEIGHTS = False
MIN_TOKEN_FREQ = 10  # kept for optional token prior utilities


# ========= Utilities =========
def set_seed(seed_value=DEFAULT_SEED):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed_value}.")


def _worker_init(_):
    # prevent thread oversubscription per worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    # for read-only HDF5 access across processes, this can reduce stalls
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def compute_token_prior(tokenizer, texts, min_freq=MIN_TOKEN_FREQ):
    token_freq = Counter(tok for txt in texts for tok in tokenizer.tokenize(txt))
    prior = np.zeros(tokenizer.vocab_size, dtype=np.float32)
    for tok, count in token_freq.items():
        adj_count = max(count, min_freq)
        importance = 1.0 / np.sqrt(adj_count)
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid != tokenizer.unk_token_id:
            prior[tid] = importance
    if np.max(prior) > 0:
        prior /= np.max(prior)
    return torch.tensor(prior)


def load_moleculenet_dataset(dataset_name, data_dir, seed, splitter):
    console.rule(
        f"[bold yellow]Loading MoleculeNet: {dataset_name} (split: {splitter})"
    )
    try:
        loader_fn = getattr(dc.molnet, f"load_{dataset_name}")
        tasks, datasets, _ = loader_fn(
            featurizer=RawFeaturizer(),
            splitter=splitter,
            reload=True,
            data_dir=data_dir,
            seed=seed,
        )
        train_ds, val_ds, test_ds = datasets

        def ds_to_df(ds, task_list):
            data = {"SMILES": ds.ids}
            for i, task in enumerate(task_list):
                data[task] = ds.y[:, i]
            return pd.DataFrame(data)

        df_train = ds_to_df(train_ds, tasks)
        df_val = ds_to_df(val_ds, tasks)
        df_test = ds_to_df(test_ds, tasks)

        logger.info(f"Tasks: {tasks}")
        logger.info(f"Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
        return df_train, df_val, df_test, tasks
    except AttributeError:
        logger.error(f"Dataset loader 'load_{dataset_name}' not found in DeepChem.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        traceback.print_exc()
        sys.exit(1)


def run_final_evaluation(
    checkpoint_path, title, test_loader, device, cache_dir, output_dir
):
    """
    Load model from checkpoint and evaluate on test set (tables + optional confusion matrices).
    """
    console.rule(f"[bold green]Final Evaluation on Test Set ({title})")
    if not os.path.exists(checkpoint_path):
        logger.error(f"{title} checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    best_thresholds = checkpoint.get(
        "best_thresholds", [0.5] * config.get("num_tasks", 1)
    )
    logger.info(f"Best thresholds from checkpoint: {best_thresholds}")

    model = CAGEFusionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    logger.info(f"Loaded {title} model from epoch {checkpoint['epoch']}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    criterion = torch.nn.BCEWithLogitsLoss()

    (test_loss, test_mcc, test_auc, test_pr, _, per_task_metrics, _, _, _) = (
        evaluate_model(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_tasks=config["num_tasks"],
            label_names=config["tasks"],
            use_precomputed_thresholds=best_thresholds,
            cache_dir=os.path.join(cache_dir, f"test_eval_{title.lower()}"),
            plot_attn=True,
            tokenizer_obj=tokenizer,
        )
    )

    console.rule(f"[bold magenta]Final Test Results ({title})")
    if per_task_metrics:
        task_table = Table(title=f"CAGE-Fusion Test Performance ({title})")
        task_table.add_column("Task", style="cyan")
        task_table.add_column("ROC-AUC", style="magenta")
        task_table.add_column("PR-AUC", style="green")
        task_table.add_column("MCC", style="yellow")
        for i, (mcc, auc, pr) in enumerate(per_task_metrics):
            task_table.add_row(
                f"{config['tasks'][i]}", f"{auc:.4f}", f"{pr:.4f}", f"{mcc:.4f}"
            )
        task_table.add_row(
            "[bold]Macro-Avg[/bold]",
            f"[bold]{test_auc:.4f}[/bold]",
            f"[bold]{test_pr:.4f}[/bold]",
            f"[bold]{test_mcc:.4f}[/bold]",
        )
        console.print(task_table)

    # Confusion matrices (optional but keeps parity with phased script)
    logger.info("Generating confusion matrices...")
    all_labels, all_probs = [], []
    for batch in test_loader:
        (
            bmg,
            sequence_embeddings,
            attn_mask,
            aux_feats,
            labels,
            input_ids_batch,
            smiles_batch,
            original_indices_batch,
            ids_list,
        ) = batch
        bmg = move_bmg_to_device(bmg, device)
        sequence_embeddings = sequence_embeddings.to(device)
        attn_mask = attn_mask.to(device)
        aux_feats = aux_feats.to(device)
        input_ids_batch = input_ids_batch.to(device)
        with torch.no_grad():
            logits, *_ = model(
                bmg=bmg,
                sequence_embeddings=sequence_embeddings,
                attn_mask=attn_mask,
                aux_feats=aux_feats,
                input_ids_batch=input_ids_batch,
                smiles_batch=smiles_batch,
                return_attn=False,
            )
            probs = torch.sigmoid(logits)
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    preds = (all_probs >= np.array(best_thresholds)[None, :]).astype(int)

    cm_dir = os.path.join(
        output_dir, f"confusion_matrices-{title.lower().replace(' ', '_')}"
    )
    os.makedirs(cm_dir, exist_ok=True)
    for i, task_name in enumerate(config["tasks"]):
        y_true = all_labels[:, i]
        y_pred = preds[:, i]
        save_path = os.path.join(cm_dir, f"cm_{task_name}.png")
        plot_confusion_matrix(
            y_true, y_pred, title=f"Confusion Matrix - {task_name}", save_path=save_path
        )
    logger.info(f"Saved confusion matrices to {cm_dir}")


# ========= Main pipeline =========
def run_benchmark(dataset_name, seed, force_rerun, rerun_train, splitter):
    # Build run-scoped paths (match phased script layout)
    run_id = f"{dataset_name}_seed{seed}"
    base_cache_dir = os.path.join(CACHE_ROOT, run_id)
    features_dir = os.path.join(FEATURES_ROOT, run_id)
    checkpoints_dir = os.path.join(CHECKPOINTS_ROOT, run_id)
    output_dir = os.path.join(OUTPUT_ROOT, run_id)
    data_dir = os.path.join(DATA_DIR, dataset_name)

    # Base config (then overlay run specifics to mirror phased script)
    config = get_default_config()
    config.update(
        dict(
            use_co_attention=USE_CO_ATTENTION,
            attn_mode=ATTN_MODE,
            use_aux_features=USE_AUX_FEATURES,
            use_fg_prompt=USE_FG_PROMPT,
            co_attention_layers=CO_ATTENTION_LAYERS,
            learning_rate=DEFAULT_LR,
            base_cache_dir=base_cache_dir,
            features_dir=features_dir,
            checkpoints_dir=checkpoints_dir,
            output_dir=output_dir,
            data_dir=data_dir,
            batch_size=DEFAULT_BATCH_SIZE,
            num_epochs=DEFAULT_NUM_EPOCHS,
            warmup_fraction=DEFAULT_WARMUP_FRACTION,
        )
    )

    console.rule(
        f"[bold cyan]MoleculeNet Benchmark (Single-Phase): {dataset_name} | Seed={seed} | Splitter={splitter} | "
        f"ForceRerun={force_rerun} | RerunTrain={rerun_train}"
    )
    set_seed(seed)

    # Clear paths if requested
    if force_rerun:
        for d in [base_cache_dir, features_dir, checkpoints_dir, output_dir]:
            if os.path.exists(d):
                logger.warning(f"Force rerun enabled. Deleting {d}")
                shutil.rmtree(d)

    if rerun_train:
        # Require features & scaler to exist
        required = [
            os.path.join(features_dir, "train_cage_fusion.h5"),
            os.path.join(features_dir, "val_cage_fusion.h5"),
            os.path.join(features_dir, "test_cage_fusion.h5"),
            os.path.join(checkpoints_dir, "aux_features_scaler.pkl"),
        ]
        missing = [f for f in required if not os.path.exists(f)]
        if missing:
            logger.error(f"Cannot rerun training; missing: {missing}")
            logger.error("Run without --rerun-train or with --force-rerun first.")
            sys.exit(1)
        if os.path.exists(base_cache_dir):
            logger.warning(f"Rerun training: deleting cache {base_cache_dir}")
            shutil.rmtree(base_cache_dir)
        if os.path.exists(checkpoints_dir):
            logger.warning(f"Rerun training: deleting {checkpoints_dir}/*.pt")
            for pt_file in glob.glob(os.path.join(checkpoints_dir, "*.pt")):
                os.remove(pt_file)

    # Create dirs
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(base_cache_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load MoleculeNet
    df_train, df_val, df_test, tasks = load_moleculenet_dataset(
        dataset_name=dataset_name, data_dir=data_dir, seed=seed, splitter=splitter
    )
    config["num_tasks"] = len(tasks)
    config["tasks"] = tasks

    # Featurization (streaming, aligned with phased/training script)
    console.rule("[bold yellow]Featurization and Setup")
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    embedding_model = AutoModel.from_pretrained(config["model_checkpoint"]).eval()

    # Precompute expected HDF5 paths and scaler path
    h5_paths = {
        "train": os.path.join(features_dir, "train_cage_fusion.h5"),
        "val": os.path.join(features_dir, "val_cage_fusion.h5"),
        "test": os.path.join(features_dir, "test_cage_fusion.h5"),
    }
    scaler_path = os.path.join(checkpoints_dir, "aux_features_scaler.pkl")

    if rerun_train:
        logger.info(
            "Rerun training: reusing existing features and scaler; skipping featurization."
        )
        scaler = joblib.load(scaler_path)
    else:
        scaler = None
        for split, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
            fit_scaler = split == "train"
            df = df.copy().reset_index().rename(columns={"index": "original_index"})
            h5, returned_scaler, _n = featurize_and_save_streaming(
                df=df,
                name=split,
                label_cols=tasks,
                cache_dir=features_dir,
                tokenizer=tokenizer,
                model=embedding_model,
                fit_scaler=fit_scaler,
                scaler=out_scaler if (out_scaler := scaler) is not None else None,
                batch_size=500,
            )
            h5_paths[split] = h5
            if split == "train":
                scaler = returned_scaler
                joblib.dump(scaler, scaler_path)
                logger.info(f"Scaler saved to {scaler_path}")

    # Datasets + loaders (mirror phased script)
    g = torch.Generator().manual_seed(seed)
    collate_with_pad = partial(
        collate_fn_for_cage_fusion, pad_token_id=tokenizer.pad_token_id
    )

    num_workers = 2
    common_loader_kwargs = dict(
        collate_fn=collate_with_pad,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        multiprocessing_context="spawn",
        worker_init_fn=_worker_init,
    )
    common_dataset_kwargs = dict(
        tokenizer_pad_id=tokenizer.pad_token_id,
        prefer_normalized_aux=True,
        return_ids=True,
        total_num_workers=num_workers,  # +1 main
        graph_cache="auto",
        single_worker_graph_cache=True,
        emb_cache_store_dtype=np.float32,
        return_emb_dtype=torch.float32,
    )

    train_loader = torch.utils.data.DataLoader(
        CageFusionStreamingDataset(h5_paths["train"], **common_dataset_kwargs),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=g,
        **common_loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        CageFusionStreamingDataset(h5_paths["val"], **common_dataset_kwargs),
        batch_size=config["batch_size"],
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        CageFusionStreamingDataset(h5_paths["test"], **common_dataset_kwargs),
        batch_size=config["batch_size"],
        shuffle=False,
        **common_loader_kwargs,
    )

    # Persist run config for reproducibility
    with open(os.path.join(checkpoints_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Configuration saved.")

    # Model + single-phase training
    device = torch.device(config["device"])
    model = CAGEFusionModel(config).to(device)
    if USE_PRETRAINED_WEIGHTS:
        pretrain_weights_path = os.path.join(
            CHECKPOINTS_ROOT, "pretrained", "pretrained_model.pt"
        )
        logger.info(f"Loading pretrained weights from {pretrain_weights_path}")
        load_partial_weights(model, pretrain_weights_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    pos_weight = compute_pos_weight_from_h5(h5_path=h5_paths["train"]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.get("warmup_fraction", 0.1)),
        num_training_steps=total_steps,
    )

    console.rule("[bold yellow]Starting Training (single-phase)")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        config=config,
        label_names=tasks,
        tokenizer_obj=tokenizer,
    )

    best_model_path = os.path.join(checkpoints_dir, MODEL_BEST)
    latest_model_path = os.path.join(checkpoints_dir, MODEL_LATEST)

    # Free memory before eval
    del model
    torch.cuda.empty_cache()

    run_final_evaluation(
        checkpoint_path=latest_model_path,
        title="Latest Model",
        test_loader=test_loader,
        device=device,
        cache_dir=base_cache_dir,
        output_dir=output_dir,
    )
    run_final_evaluation(
        checkpoint_path=best_model_path,
        title="Best Model",
        test_loader=test_loader,
        device=device,
        cache_dir=base_cache_dir,
        output_dir=output_dir,
    )

    console.rule("[bold green]✨ Benchmark Complete!")


# ========= CLI =========
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MoleculeNet benchmark for CAGE-Fusion (single-phase)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="MoleculeNet dataset name (e.g., bace_classification)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        default=DEFAULT_FORCE_RERUN,
        help="Delete cache/features/checkpoints/output and start fresh",
    )
    parser.add_argument(
        "--rerun-train",
        action="store_true",
        default=DEFAULT_RERUN_TRAIN,
        help="Rerun training only (requires existing features + scaler)",
    )
    parser.add_argument(
        "--splitter",
        type=str,
        default=DEFAULT_SPLITTER,
        choices=["scaffold", "random", "stratified"],
        help="Dataset split method",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        dataset_name=args.dataset,
        seed=args.seed,
        force_rerun=args.force_rerun,
        rerun_train=args.rerun_train,
        splitter=args.splitter,
    )
