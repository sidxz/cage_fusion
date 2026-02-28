#!/usr/bin/env python3
"""
Hyperparameter optimization script for CAGE-Fusion using Optuna.

This script leverages a shared feature cache to avoid redundant computations
across trials. Each trial runs with a distinct set of hyperparameters and
stores its outputs in a unique directory.
"""

import os
import sys
import shutil
import joblib
import argparse
import traceback

import torch
import optuna
import pandas as pd
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from rich.console import Console
from rich.table import Table

# ======== Project Path Setup ========
# Ensure the project root is in the system path to locate cage_fusion modules.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ======== Local Imports ========
from cage_fusion.configuration import CageFusionConfig
from cage_fusion.modeling import CAGEFusionForMultiLabelClassification
from cage_fusion.training import Trainer, TrainingArguments
from cage_fusion.data import CageFusionStreamingDataset, collate_cage_fusion
from cage_fusion.utils import compute_pos_weight_from_h5
from cage_fusion.featurization import featurize_and_save_streaming
from cage_fusion.utils.logging import logger

# Assuming run_benchmark is in the same directory or a discoverable path.
from .run_benchmark import (
    load_moleculenet_dataset,
    set_seed,
    CACHE_ROOT,
    FEATURES_ROOT,
    CHECKPOINTS_ROOT,
    DATA_DIR,
    OUTPUT_ROOT,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_WARMUP_FRAC,
)

# ======== Console Setup ========
console = Console()


def ensure_dir(directory_path: str):
    """Creates a directory if it does not already exist."""
    os.makedirs(directory_path, exist_ok=True)


def prepare_features_once(args: argparse.Namespace, features_dir: str):
    """
    Handles the featurization of the dataset.

    This is a potentially time-consuming step that is run only once per
    dataset/seed/splitter configuration. It creates a feature cache that
    all Optuna trials will use.
    """
    # Check if featurization is already complete.
    required_files = [
        os.path.join(features_dir, f"{split}_cage_fusion.h5")
        for split in ["train", "val", "test"]
    ]
    if all(os.path.exists(f) for f in required_files) and not args.force_featurize:
        logger.info(f"Using cached features from: {features_dir}")
        return

    logger.info(f"Starting one-time featurization for dataset: {args.dataset}")
    if args.force_featurize:
        logger.warning(
            f"Force re-featurization enabled. Deleting existing features in {features_dir}"
        )
        if os.path.exists(features_dir):
            shutil.rmtree(features_dir)

    ensure_dir(features_dir)

    # Load raw dataset
    set_seed(args.seed)
    data_dir = os.path.join(DATA_DIR, args.dataset)
    df_train, df_val, df_test, tasks = load_moleculenet_dataset(
        dataset_name=args.dataset,
        data_dir=data_dir,
        seed=args.seed,
        splitter=args.splitter,
    )

    # Setup models for embedding
    config = get_default_config()
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    embedding_model = AutoModel.from_pretrained(config["model_checkpoint"]).eval()

    # Process each split
    scaler = None
    for split, df_original in [("train", df_train), ("val", df_val), ("test", df_test)]:
        logger.info(f"Featurizing '{split}' split...")
        df = (
            df_original.copy().reset_index().rename(columns={"index": "original_index"})
        )

        _, _, scaler_obj, num_featurized_samples = featurize_and_save_streaming(
            df=df,
            name=split,
            label_cols=tasks,
            cache_dir=features_dir,
            tokenizer=tokenizer,
            model=embedding_model,
            fit_scaler=(split == "train"),
            scaler=scaler,
        )
        if split == "train":
            scaler = scaler_obj
            scaler_path = os.path.join(features_dir, "aux_features_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")

    logger.info("Featurization complete.")


def objective(
    trial: optuna.Trial, args: argparse.Namespace, features_dir: str, tasks: list
) -> float:
    """
    The Optuna objective function for a single trial.

    Args:
        trial: An Optuna Trial object.
        args: Command-line arguments.
        features_dir: Path to the cached features.
        tasks: The list of task names for the dataset.

    Returns:
        The performance metric to be optimized (e.g., negative validation MCC).
    """
    try:
        # ----- 1. Hyperparameter Search Space -----
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 200, 256])
        # You can add more hyperparameters to tune here
        # e.g., dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)

        # ----- 2. Trial-specific Setup -----
        run_id = f"{args.dataset}_seed{args.seed}_{args.splitter}"
        trial_id = f"{run_id}_trial{trial.number}"

        checkpoints_dir = os.path.join(CHECKPOINTS_ROOT, trial_id)
        base_cache_dir = os.path.join(CACHE_ROOT, trial_id)
        output_dir = os.path.join(OUTPUT_ROOT, trial_id)

        # Clean up directories from previous runs of the same trial number
        for d in [checkpoints_dir, base_cache_dir, output_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            ensure_dir(d)

        # ----- 3. Configuration -----
        set_seed(args.seed)  # Ensures model initialization is consistent
        config = get_default_config()
        config.update(
            {
                "num_tasks": len(tasks),
                "tasks": tasks,
                "batch_size": batch_size,
                "num_epochs": args.num_epochs,
                "learning_rate": learning_rate,
                "warmup_fraction": args.warmup_frac,
                "base_cache_dir": base_cache_dir,
                "features_dir": features_dir,
                "checkpoints_dir": checkpoints_dir,
                "output_dir": output_dir,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
        )

        # ----- 4. Data Loaders -----
        h5_paths = {
            split: os.path.join(features_dir, f"{split}_cage_fusion.h5")
            for split in ["train", "val"]
        }
        graph_pkl_paths = {
            split: os.path.join(features_dir, f"{split}_graph_feats.pkl")
            for split in ["train", "val"]
        }

        g = torch.Generator().manual_seed(args.seed)
        train_loader = torch.utils.data.DataLoader(
            CageFusionStreamingDataset(h5_paths["train"], graph_pkl_paths["train"]),
            batch_size=batch_size,
            collate_fn=collate_fn_for_cage_fusion,
            shuffle=True,
            generator=g,
        )
        val_loader = torch.utils.data.DataLoader(
            CageFusionStreamingDataset(h5_paths["val"], graph_pkl_paths["val"]),
            batch_size=batch_size,
            collate_fn=collate_fn_for_cage_fusion,
            shuffle=False,
        )

        # ----- 5. Model, Optimizer, and Training -----
        device = torch.device(config["device"])
        model = CAGEFusionModel(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        pos_weight = compute_pos_weight_from_h5(h5_path=h5_paths["train"]).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        num_training_steps = len(train_loader) * args.num_epochs
        num_warmup_steps = int(num_training_steps * args.warmup_frac)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

        # We need the tokenizer for training callbacks, but not for featurization here
        tokenizer = AutoTokenizer.from_pretrained(
            get_default_config()["model_checkpoint"]
        )

        logger.info(
            f"Starting Trial {trial.number}: LR={learning_rate:.2e}, BS={batch_size}"
        )
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

        # ----- 6. Report Result -----
        hist_path = os.path.join(output_dir, "training_history.csv")
        if os.path.exists(hist_path):
            hist_df = pd.read_csv(hist_path)
            # We want to maximize MCC, so we return its negative value for minimization.
            best_val_mcc = hist_df["val_mcc"].max()
            logger.info(
                f"Trial {trial.number} finished. Best Val MCC: {best_val_mcc:.4f}"
            )
            return -best_val_mcc
        else:
            logger.error(
                f"Training history not found for trial {trial.number}. Pruning."
            )
            raise optuna.TrialPruned()

    except Exception as e:
        logger.error(f"Trial {trial.number} failed with exception: {e}")
        traceback.print_exc()
        # Report failure to Optuna so it can prune the trial.
        raise optuna.TrialPruned()


def main():
    """Main function to setup and run the HPO study."""
    parser = argparse.ArgumentParser(
        description="Run Hyperparameter Optimization with Optuna for CAGE-Fusion."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="MoleculeNet dataset name (e.g., bace_classification).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--splitter",
        type=str,
        default="scaffold",
        choices=["scaffold", "random", "stratified"],
        help="Dataset splitting method.",
    )
    parser.add_argument(
        "--n_trials", type=int, default=25, help="Number of Optuna trials to run."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs per trial.",
    )
    parser.add_argument(
        "--warmup_frac",
        type=float,
        default=DEFAULT_WARMUP_FRAC,
        help="Learning rate warmup fraction.",
    )
    parser.add_argument(
        "--force-featurize",
        action="store_true",
        help="Force re-running featurization even if cache exists.",
    )
    args = parser.parse_args()

    console.rule(f"[bold cyan]HPO for CAGE-Fusion on {args.dataset}[/bold cyan]")

    # Define a shared feature directory for all trials of this HPO run.
    features_dir = os.path.join(
        FEATURES_ROOT, f"{args.dataset}_seed{args.seed}_{args.splitter}"
    )

    # --- Robustness Fix 1: Check for and fix inconsistent feature cache ---
    h5_files = [
        os.path.join(features_dir, f"{split}_cage_fusion.h5")
        for split in ["train", "val", "test"]
    ]
    pkl_files = [
        os.path.join(features_dir, f"{split}_graph_feats.pkl")
        for split in ["train", "val", "test"]
    ]

    h5_all_exist = all(os.path.exists(f) for f in h5_files)
    pkl_any_exist = any(os.path.exists(f) for f in pkl_files)

    # If graph files exist but the final H5 files don't, the cache is corrupt. Force a rebuild.
    if pkl_any_exist and not h5_all_exist and not args.force_featurize:
        logger.warning(
            "Inconsistent feature cache state detected (graph files exist, but H5 files are missing)."
        )
        logger.warning("Forcing re-featurization to ensure cache consistency.")
        args.force_featurize = True

    # Step 1: Prepare features once before starting the study.
    prepare_features_once(args, features_dir)

    # --- Robustness Fix 2: Verify that featurization actually created the files ---
    for f in h5_files:
        if not os.path.exists(f):
            logger.error(
                f"FATAL: Featurization step failed to produce required file: {f}"
            )
            logger.error("Please manually delete the cache directory and try again:")
            logger.error(f"rm -rf {features_dir}")
            sys.exit(1)

    # Step 2: Load tasks list once to pass to all trials, avoiding redundant loads.
    logger.info("Loading dataset tasks list for the HPO study...")
    set_seed(args.seed)
    data_dir = os.path.join(DATA_DIR, args.dataset)
    _, _, _, tasks = load_moleculenet_dataset(
        dataset_name=args.dataset,
        data_dir=data_dir,
        seed=args.seed,
        splitter=args.splitter,
    )

    # Step 3: Define the objective function with fixed arguments.
    objective_fn = lambda trial: objective(trial, args, features_dir, tasks)

    # Step 4: Create and run the Optuna study.
    study = optuna.create_study(
        direction="minimize",  # We minimize the negative of MCC
        pruner=optuna.pruners.MedianPruner(),
        study_name=f"cage-fusion-hpo-{args.dataset}-{args.splitter}-seed{args.seed}",
    )

    console.rule("[bold yellow]Starting Optuna Study[/bold yellow]")
    study.optimize(
        objective_fn, n_trials=args.n_trials, timeout=None
    )  # Set timeout in seconds if needed

    # Step 5: Print results.
    console.rule("[bold green]HPO Study Complete[/bold green]")
    logger.info(f"Number of finished trials: {len(study.trials)}")

    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )

    logger.info(f"  Pruned trials: {len(pruned_trials)}")
    logger.info(f"  Completed trials: {len(complete_trials)}")

    # --- Robustness Fix 3: Check if any trials completed before showing best trial ---
    if complete_trials:
        console.print("\n[bold]Best trial:[/bold]")
        trial = study.best_trial
        console.print(f"  Value (Negative MCC): {-trial.value:.4f}")
        console.print("  Params: ")
        for key, value in trial.params.items():
            console.print(f"    {key}: {value}")
    else:
        console.print(
            "\n[bold red]No trials completed successfully. Cannot determine best trial.[/bold red]"
        )

    # Optionally, save full study results
    results_df = study.trials_dataframe()
    output_path = os.path.join(
        OUTPUT_ROOT, f"hpo_results_{args.dataset}_seed{args.seed}.csv"
    )
    results_df.to_csv(output_path, index=False)
    logger.info(f"Full study results saved to {output_path}")


if __name__ == "__main__":
    main()
