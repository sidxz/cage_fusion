#!/usr/bin/env python3

"""
Script to run MoleculeNet benchmark for CAGE-Fusion model.
"""

import os
import sys
import torch
import joblib
import glob
import numpy as np
import pandas as pd
import shutil
import traceback
import argparse
import random
from collections import Counter
from rich.table import Table
from rich.console import Console

# ======== Configurable Paths and Variables ========
DATA_ROOT = "data"  # Top-level data directory
DATA_DIR = os.path.join(DATA_ROOT, "molnet")
CACHE_ROOT = os.path.join(DATA_ROOT, "cache")
FEATURES_ROOT = os.path.join(DATA_ROOT, "features")
CHECKPOINTS_ROOT = "checkpoints"
OUTPUT_ROOT = "output"
DEFAULT_DATASET = "bace_classification"
DEFAULT_FORCE_RERUN = False
DEFAULT_RERUN_TRAIN = False
DEFAULT_SPLITTER = "scaffold"

DEFAULT_SEED = 54
DEFAULT_BATCH_SIZE = 32

DEFAULT_NUM_EPOCHS = 30
DEFAULT_LR = 5e-4
USE_PRETRAINED_WEIGHTS = False  # Whether to use pretrained weights


MODEL_BEST = "best_model.pt"
MODEL_LATEST = "latest_checkpoint.pt"
MIN_TOKEN_FREQ = 10

# ======== Project Path Setup ========
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ======== Third-party & Local Imports ========
import deepchem as dc
from deepchem.feat import RawFeaturizer
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from rich.console import Console
from rich.table import Table
from rich.traceback import install

from cage_fusion.configs import get_default_config
from cage_fusion.featurizers import featurize_and_save_streaming
from cage_fusion.models import CAGEFusionModel
from cage_fusion.engine.training import train_model
from cage_fusion.engine.evaluation import evaluate_model
from cage_fusion.engine.dataset import CageFusionStreamingDataset
from cage_fusion.engine.data_utils import collate_fn_for_cage_fusion
from cage_fusion.engine.utils import compute_pos_weight_from_h5
from cage_fusion.utils.logging_utils import logger
from cage_fusion.utils.model_utils import load_partial_weights


# ======== Console Setup ========
install()
console = Console()


def set_seed(seed_value=DEFAULT_SEED):
    """Set seed for reproducibility across libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed_value} for reproducibility.")


def compute_token_prior(tokenizer, texts, min_freq=MIN_TOKEN_FREQ):
    """
    Computes a token importance prior based on inverse frequency.
    """
    logger.info("Computing token importance prior...")
    token_freq = Counter(tok for txt in texts for tok in tokenizer.tokenize(txt))
    prior = np.zeros(tokenizer.vocab_size, dtype=np.float32)
    for tok, count in token_freq.items():
        adj_count = max(count, min_freq)
        importance = 1.0 / np.sqrt(adj_count)
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid != tokenizer.unk_token_id:
            prior[tid] = importance
    prior /= np.max(prior)
    logger.info("Token importance prior computed.")
    return torch.tensor(prior)


def load_moleculenet_dataset(dataset_name, data_dir, seed, splitter):
    """
    Loads a MoleculeNet dataset using DeepChem.
    Returns: train_df, val_df, test_df, tasks
    """
    console.rule(
        f"[bold yellow]Loading MoleculeNet Dataset: {dataset_name} (split: {splitter})"
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

        logger.info(f"Loaded dataset: {dataset_name}, Tasks: {tasks}")
        logger.info(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

        return df_train, df_val, df_test, tasks

    except AttributeError:
        logger.error(f"Dataset loader 'load_{dataset_name}' not found in DeepChem.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        traceback.print_exc()
        sys.exit(1)


from rich.table import Table
from rich.console import Console


def run_final_evaluation(checkpoint_path, title, test_loader, device, cache_dir):
    """
    Load model from checkpoint and evaluate on test set.
    """
    console = Console()
    console.rule(f"[bold green]Final Evaluation on Test Set ({title})")
    if not os.path.exists(checkpoint_path):
        logger.error(f"{title} checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    best_thresholds = checkpoint.get("best_thresholds")
    logger.info(f"Best thresholds: {best_thresholds}")

    model = CAGEFusionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    logger.info(f"Loaded {title} model from epoch {checkpoint['epoch']}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    criterion = torch.nn.BCEWithLogitsLoss()
    (
        test_loss,
        test_mcc,
        test_auc,
        test_pr,
        _,  # test_preds
        per_task_metrics,
        _,  # attn
        _,  # barplot_paths
        _,  # topk_tables
    ) = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_tasks=config["num_tasks"],
        label_names=config["tasks"],
        use_precomputed_thresholds=best_thresholds,
        cache_dir=os.path.join(cache_dir, f"test_eval_{title.lower()}"),
        plot_attn=True,  # Always plot for final eval
        tokenizer_obj=tokenizer,
    )

    console.rule(f"[bold magenta]Final Test Set Results ({title})")
    console.print(
        f"  Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test MCC: {test_mcc:.4f}"
    )
    if per_task_metrics:
        task_table = Table(
            title=f"Per-Task Test Metrics ({title})",
            header_style="bold green",
            show_footer=False,
        )
        task_table.add_column("Task", style="cyan")
        task_table.add_column("ROC-AUC", style="magenta", justify="right")
        task_table.add_column("MCC", style="yellow", justify="right")
        task_table.add_column("PR-AUC", style="green", justify="right")

        for i, (mcc, auc, pr) in enumerate(per_task_metrics):
            task = (
                config["tasks"][i]
                if "tasks" in config and i < len(config["tasks"])
                else f"Task {i}"
            )
            task_table.add_row(task, f"{auc:.3f}", f"{mcc:.3f}", f"{pr:.3f}")

        # Macro averages row (add separator for clarity)
        task_table.add_section()
        task_table.add_row(
            "[bold]Macro-Avg[/bold]",
            f"[bold]{test_auc:.4f}[/bold]",
            f"[bold]{test_mcc:.4f}[/bold]",
            f"[bold]{test_pr:.4f}[/bold]",
        )
        console.print(task_table)


def run_benchmark(dataset_name, seed, force_rerun, rerun_train, splitter):
    """
    Main pipeline for running the CAGE-Fusion benchmark.
    """
    # Directory for this run
    config = get_default_config()

    run_id = f"{dataset_name}_seed{seed}"

    config["base_cache_dir"] = os.path.join(CACHE_ROOT, run_id)
    config["features_dir"] = os.path.join(FEATURES_ROOT, run_id)
    config["checkpoints_dir"] = os.path.join(CHECKPOINTS_ROOT, run_id)
    config["data_dir"] = os.path.join(DATA_DIR, dataset_name)
    config["output_dir"] = os.path.join(OUTPUT_ROOT, run_id)

    console.rule(
        f"[bold cyan]MoleculeNet Benchmark: {dataset_name} (Seed: {seed}), Force Rerun: {force_rerun}, Splitter: {splitter}"
    )
    set_seed(seed)

    # Optionally clear cache and checkpoints
    if force_rerun:
        if os.path.exists(config["base_cache_dir"]):
            logger.warning(
                f"Force rerun enabled. Deleting cache: {config['base_cache_dir']}"
            )
            shutil.rmtree(config["base_cache_dir"])
        if os.path.exists(config["features_dir"]):
            logger.warning(
                f"Force rerun enabled. Deleting features: {config['features_dir']}"
            )
            shutil.rmtree(config["features_dir"])
        if os.path.exists(config["checkpoints_dir"]):
            logger.warning(
                f"Force rerun enabled. Deleting checkpoints: {config['checkpoints_dir']}"
            )
            shutil.rmtree(config["checkpoints_dir"])

    if rerun_train:
        # check if features dir exists, if not, error out
        if not os.path.exists(config["features_dir"]):
            logger.error(
                f"Cannot rerun training without existing features. Please run with --force-rerun to regenerate features."
            )
            sys.exit(1)
        if os.path.exists(config["base_cache_dir"]):
            logger.warning(
                f"Force rerun enabled. Deleting cache: {config['base_cache_dir']}"
            )
            shutil.rmtree(config["base_cache_dir"])
        if os.path.exists(config["checkpoints_dir"]):
            logger.warning(
                f"Rerun training enabled. Deleting checkpoints: {config['checkpoints_dir']}"
            )
            for pt_file in glob.glob(os.path.join(config["checkpoints_dir"], "*.pt")):
                os.remove(pt_file)

    # Create fresh dirs if needed
    os.makedirs(config["data_dir"], exist_ok=True)
    os.makedirs(config["base_cache_dir"], exist_ok=True)
    os.makedirs(config["features_dir"], exist_ok=True)
    os.makedirs(config["checkpoints_dir"], exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)

    # Data loading
    df_train, df_val, df_test, tasks = load_moleculenet_dataset(
        dataset_name=dataset_name,
        data_dir=config["data_dir"],
        seed=seed,
        splitter=splitter,
    )

    config["num_tasks"] = len(tasks)
    config["tasks"] = tasks

    config["batch_size"] = DEFAULT_BATCH_SIZE
    config["num_epochs"] = DEFAULT_NUM_EPOCHS
    config["learning_rate"] = DEFAULT_LR

    console.rule("[bold yellow]Featurization and Setup")
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    embedding_model = AutoModel.from_pretrained(config["model_checkpoint"]).eval()

    h5_paths, glob_paths = {}, {}
    scaler = None
    for split, df_original in [("train", df_train), ("val", df_val), ("test", df_test)]:
        df = (
            df_original.copy().reset_index().rename(columns={"index": "original_index"})
        )

        h5, glob_p, scaler_obj, num_featurized_samples = featurize_and_save_streaming(
            df=df,
            name=split,
            label_cols=tasks,
            cache_dir=config["features_dir"],
            tokenizer=tokenizer,
            model=embedding_model,
            fit_scaler=(split == "train"),
            scaler=scaler,
        )
        if split == "train":
            scaler = scaler_obj
            # Save the scaler for later use
            scaler_path = os.path.join(
                config["checkpoints_dir"], "aux_features_scaler.pkl"
            )
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")

        h5_paths[split], glob_paths[split] = h5, glob_p

    g = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        CageFusionStreamingDataset(
            h5_paths["train"],
            os.path.join(config["features_dir"], f"train_graph_feats.pkl"),
            tokenizer.pad_token_id,
        ),
        batch_size=config["batch_size"],
        collate_fn=collate_fn_for_cage_fusion,
        shuffle=True,
        generator=g,
    )
    val_loader = torch.utils.data.DataLoader(
        CageFusionStreamingDataset(
            h5_paths["val"],
            os.path.join(config["features_dir"], f"val_graph_feats.pkl"),
            tokenizer.pad_token_id,
        ),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn_for_cage_fusion,
    )
    test_loader = torch.utils.data.DataLoader(
        CageFusionStreamingDataset(
            h5_paths["test"],
            os.path.join(config["features_dir"], f"test_graph_feats.pkl"),
            tokenizer.pad_token_id,
        ),
        batch_size=config["batch_size"],
        collate_fn=collate_fn_for_cage_fusion,
        shuffle=False,
    )
    
    # Save config json to checkpoints directory
    config_path = os.path.join(config["checkpoints_dir"], "config.json")
    with open(config_path, "w") as f:
        import json
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {config_path}")

    device = torch.device(config["device"])
    model = CAGEFusionModel(config).to(device)
    if USE_PRETRAINED_WEIGHTS:
        pretrain_weights_path = os.path.join(CHECKPOINTS_ROOT, "pretrained", "pretrained_model.pt")
        logger.info(f"\033[1;34mLoading pretrained weights from {pretrain_weights_path}\033[0m")
        load_partial_weights(model, pretrain_weights_path)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    pos_weight = compute_pos_weight_from_h5(h5_path=h5_paths["train"]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            len(train_loader) * config["num_epochs"] * config["warmup_fraction"]
        ),
        num_training_steps=len(train_loader) * config["num_epochs"],
    )

    console.rule("[bold yellow]Starting Training")
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        device,
        config,
        tasks,
        tokenizer,
    )

    best_model_path = os.path.join(config["checkpoints_dir"], MODEL_BEST)
    latest_model_path = os.path.join(config["checkpoints_dir"], MODEL_LATEST)

    # Clear memory
    del model
    torch.cuda.empty_cache()

    run_final_evaluation(
        checkpoint_path=latest_model_path,
        title="Latest Model",
        test_loader=test_loader,
        device=device,
        cache_dir=config["base_cache_dir"],
    )

    run_final_evaluation(
        checkpoint_path=best_model_path,
        title="Best Model",
        test_loader=test_loader,
        device=device,
        cache_dir=config["base_cache_dir"],
    )

    console.rule("[bold green]✨ Benchmark Complete!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MoleculeNet benchmark for CAGE-Fusion model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Name of the MoleculeNet dataset to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        default=DEFAULT_FORCE_RERUN,
        help="Force rerunning by deleting cache and checkpoints.",
    )
    parser.add_argument(
        "--rerun-train",
        action="store_true",
        default=DEFAULT_RERUN_TRAIN,
        help="Rerun training only, skipping featurization.",
    )
    parser.add_argument(
        "--splitter",
        type=str,
        default=DEFAULT_SPLITTER,
        choices=["scaffold", "random", "stratified"],
        help="Dataset splitting method.",
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
