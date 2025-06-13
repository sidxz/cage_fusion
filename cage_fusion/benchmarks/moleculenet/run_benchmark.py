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

# Add project root to the Python path to allow for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# DeepChem and Transformer imports
import deepchem as dc
from deepchem.feat import RawFeaturizer
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

# Rich and logging imports
from rich.console import Console
from rich.table import Table
from rich.traceback import install

# Local library imports
from cage_fusion.configs import get_default_config
from cage_fusion.featurizers import featurize_and_save_streaming
from cage_fusion.models import CAGEFusionModel
from cage_fusion.engine.training import train_model
from cage_fusion.engine.evaluation import evaluate_model
from cage_fusion.engine.dataset import CageFusionStreamingDataset, MiniBatchCacheDataset
from cage_fusion.engine.data_utils import collate_fn_for_cage_fusion
from cage_fusion.engine.utils import move_bmg_to_device
from cage_fusion.utils.logging_utils import logger

# --- Setup Console ---
install()
console = Console()


def set_seed(seed_value=42):
    """Set seed for reproducibility across all libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed_value} for full reproducibility.")


def compute_token_prior(tokenizer, texts, min_freq=10):
    """Computes a token importance prior based on inverse frequency."""
    logger.info("Computing token importance prior...")
    token_freq = Counter(tok for txt in texts for tok in tokenizer.tokenize(txt))
    prior = np.zeros(tokenizer.vocab_size, dtype=np.float32)
    for tok, count in token_freq.items():
        adj_count = max(count, min_freq)
        importance = 1.0 / np.sqrt(adj_count)
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid != tokenizer.unk_token_id:
            prior[tid] = importance
    # Normalize
    prior /= np.max(prior)
    logger.info("Token importance prior computed.")
    return torch.tensor(prior)


def load_moleculenet_dataset(
    dataset_name: str, data_dir="data/molnet", seed=42, splitter="scaffold"
):
    """
    Loads a MoleculeNet dataset using DeepChem's standard scaffold splitter.
    """
    console.rule(
        f"[bold yellow]Loading MoleculeNet Dataset: {dataset_name} with splitter '{splitter}'"
    )
    try:
        loader_fn = getattr(dc.molnet, f"load_{dataset_name}")
        tasks, datasets, transformers = loader_fn(
            featurizer=RawFeaturizer(),
            splitter=splitter,
            reload=True,
            data_dir=data_dir,
            seed=seed,
        )
        train_ds, val_ds, test_ds = datasets

        def process_dataframe(ds, task_list):
            """Converts DeepChem dataset to a clean pandas DataFrame."""
            data = {"SMILES_Canonical": ds.ids}
            for i, task in enumerate(task_list):
                data[task] = ds.y[:, i]
            return pd.DataFrame(data)

        df_train = process_dataframe(train_ds, tasks)
        df_val = process_dataframe(val_ds, tasks)
        df_test = process_dataframe(test_ds, tasks)

        logger.info(f"Loaded dataset: {dataset_name}")
        logger.info(f"Tasks: {tasks}")
        logger.info(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

        # Print label distribution
        def print_distribution(df, split_name):
            logger.info(f"Label distribution per task in {split_name} set:")
            for task in tasks:
                counts = df[task].value_counts(dropna=True).to_dict()
                num_zeros = int(counts.get(0.0, 0))
                num_ones = int(counts.get(1.0, 0))
                logger.info(f"  {task}: 0s = {num_zeros}, 1s = {num_ones}")

        print_distribution(df_train, "train")
        print_distribution(df_val, "val")
        print_distribution(df_test, "test")

        return df_train, df_val, df_test, tasks

    except AttributeError:
        logger.error(f"Dataset loader 'load_{dataset_name}' not found in DeepChem.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        traceback.print_exc()
        sys.exit(1)


def combine_graph_parts(glob_pattern, output_path):
    part_files = sorted(glob.glob(glob_pattern))
    if not part_files:
        raise FileNotFoundError(f"No graph part files found: {glob_pattern}")
    logger.info(f"Combining {len(part_files)} graph parts...")
    all_feats = [feat for pf in part_files for feat in joblib.load(pf)]
    joblib.dump(all_feats, output_path, compress=3)


def run_final_evaluation(checkpoint_path, title, test_loader, device, base_cache_dir):
    console.rule(f"[bold green]Final Evaluation on Test Set ({title})")
    if not os.path.exists(checkpoint_path):
        logger.error(f"{title} checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    best_thresholds = checkpoint.get("best_thresholds")
    logger.info(f"Best thresholds: {best_thresholds}")

    logger.info("Creating a fresh model instance for evaluation...")
    model = CAGEFusionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    model.eval()
    logger.info(f"Loaded {title} model from epoch {checkpoint['epoch']}")

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
            cache_dir=os.path.join(base_cache_dir, f"test_eval_{title.lower()}"),
        )
    )

    console.rule(f"[bold magenta]Final Test Set Results ({title})")
    console.print(
        f"  Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test MCC: {test_mcc:.4f}"
    )
    if per_task_metrics:
        task_table = Table(title=f"Per-Task Test Metrics ({title})")
        for col in ["Task", "AUC", "MCC", "PR-AUC"]:
            task_table.add_column(col)
        for i, (mcc, auc, pr) in enumerate(per_task_metrics):
            task_table.add_row(
                f"{config['tasks'][i]}", f"{auc:.3f}", f"{mcc:.3f}", f"{pr:.3f}"
            )
        console.print(task_table)


def run_benchmark(
    dataset_name: str, seed: int, force_rerun: bool, splitter: str = "scaffold"
):
    console.rule(
        f"[bold cyan]MoleculeNet Benchmark: {dataset_name} (Seed: {seed}), Force Rerun: {force_rerun}"
    )
    set_seed(seed)
    base_cache_dir = os.path.join(
        "data/molnet/bench_cache", f"{dataset_name}_seed{seed}"
    )

    if force_rerun and os.path.exists(base_cache_dir):
        logger.warning(f"Force rerun enabled. Deleting cache: {base_cache_dir}")
        shutil.rmtree(base_cache_dir)

    df_train, df_val, df_test, tasks = load_moleculenet_dataset(dataset_name, seed=seed)

    config = get_default_config()
    config["num_tasks"] = len(tasks)
    config["tasks"] = tasks
    config["base_cache_dir"] = base_cache_dir
    config["batch_size"] = 200
    config["num_epochs"] = 35

    console.rule("[bold yellow]Featurization and Setup")
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    embedding_model = AutoModel.from_pretrained(config["model_checkpoint"]).eval()

    # Compute and add token prior to config
    token_prior = compute_token_prior(tokenizer, df_train.SMILES_Canonical.tolist())
    config["token_importance_prior"] = token_prior.to(torch.device(config["device"]))

    h5_paths, glob_paths = {}, {}
    scaler = None
    for split, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        h5, glob_p, scaler_obj = featurize_and_save_streaming(
            df=df,
            name=split,
            label_cols=tasks,
            cache_dir=base_cache_dir,
            tokenizer=tokenizer,
            model=embedding_model,
            fit_scaler=(split == "train"),
            scaler=scaler,
        )
        if split == "train":
            scaler = scaler_obj
        h5_paths[split], glob_paths[split] = h5, glob_p

    for split, glob_p in glob_paths.items():
        combine_graph_parts(
            glob_p, os.path.join(base_cache_dir, f"{split}_graph_feats.pkl")
        )

    g = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        CageFusionStreamingDataset(
            h5_paths["train"], os.path.join(base_cache_dir, "train_graph_feats.pkl")
        ),
        batch_size=config["batch_size"],
        collate_fn=collate_fn_for_cage_fusion,
        shuffle=True,
        generator=g,
    )
    val_loader = torch.utils.data.DataLoader(
        CageFusionStreamingDataset(
            h5_paths["val"], os.path.join(base_cache_dir, "val_graph_feats.pkl")
        ),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn_for_cage_fusion,
    )
    test_loader = torch.utils.data.DataLoader(
        CageFusionStreamingDataset(
            h5_paths["test"], os.path.join(base_cache_dir, "test_graph_feats.pkl")
        ),
        batch_size=config["batch_size"],
        collate_fn=collate_fn_for_cage_fusion,
        shuffle=False,
    )

    device = torch.device(config["device"])
    model = CAGEFusionModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()
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

    best_model_path = os.path.join(base_cache_dir, "best_model.pt")
    latest_model_path = os.path.join(base_cache_dir, "latest_checkpoint.pt")

    # Clear memory
    del model
    torch.cuda.empty_cache()

    run_final_evaluation(
        checkpoint_path=latest_model_path,
        title="Latest Model",
        test_loader=test_loader,
        device=device,
        base_cache_dir=base_cache_dir,
    )

    run_final_evaluation(
        checkpoint_path=best_model_path,
        title="Best Model",
        test_loader=test_loader,
        device=device,
        base_cache_dir=base_cache_dir,
    )

    console.rule("[bold green]✨ Benchmark Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MoleculeNet benchmark for CAGE-Fusion model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bace_classification",
        help="Name of the MoleculeNet dataset to use (e.g., bace_classification, clintox, sider).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        default=False,
        help="Force rerunning by deleting the cache first.",
    )
    parser.add_argument(
        "--splitter",
        type=str,
        default="scaffold",
        choices=["scaffold", "random", "stratified"],
        help="Dataset splitting method: scaffold (default), random, or stratified.",
    )
    args = parser.parse_args()
    run_benchmark(args.dataset, args.seed, args.force_rerun)
