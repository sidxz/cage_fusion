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
from rich.traceback import install

# Local library imports
from cage_fusion.configs import get_default_config
from cage_fusion.featurizers import featurize_and_save_streaming
from cage_fusion.models import CAGEFusionModel
from cage_fusion.engine.training import train_model
from cage_fusion.engine.evaluation import evaluate_model
from cage_fusion.engine.dataset import CageFusionStreamingDataset, MiniBatchCacheDataset
from cage_fusion.engine.data_utils import collate_fn_for_cage_fusion
from cage_fusion.utils.logging_utils import logger

# --- Setup Console ---
install()
console = Console()


def load_moleculenet_dataset(dataset_name: str, data_dir="data/molnet"):
    """
    Loads a MoleculeNet dataset using DeepChem and splits it into DataFrames.
    """
    console.rule(f"[bold yellow]Loading MoleculeNet Dataset: {dataset_name}")
    try:
        loader_fn = getattr(dc.molnet, f"load_{dataset_name}")
        tasks, datasets, transformers = loader_fn(
            featurizer=RawFeaturizer(),
            splitter="scaffold",
            reload=True,
            data_dir=data_dir,
        )
        train_ds, val_ds, test_ds = datasets

        def process_dataframe(ds, task_list):
            """Converts DeepChem dataset to a clean pandas DataFrame."""
            df = pd.DataFrame(
                {"SMILES_Canonical": ds.ids, task_list[0]: ds.y.flatten()}
            )
            return df

        df_train = process_dataframe(train_ds, tasks)
        df_val = process_dataframe(val_ds, tasks)
        df_test = process_dataframe(test_ds, tasks)

        logger.info(f"Loaded dataset: {dataset_name}")
        logger.info(f"Tasks: {tasks}")
        logger.info(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
        return df_train, df_val, df_test, tasks

    except AttributeError:
        logger.error(f"Dataset loader 'load_{dataset_name}' not found in DeepChem.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        traceback.print_exc()
        sys.exit(1)


def combine_graph_parts(glob_pattern, output_path):
    """Combines chunked graph feature files into a single file."""
    part_files = sorted(glob.glob(glob_pattern))
    if not part_files:
        raise FileNotFoundError(
            f"No graph part files found for pattern: {glob_pattern}"
        )

    logger.info(f"Combining {len(part_files)} graph parts into {output_path}...")
    combined_feats = [
        feat for part_file in part_files for feat in joblib.load(part_file)
    ]
    joblib.dump(combined_feats, output_path, compress=3)
    logger.info(f"Combined graph features saved.")


def run_benchmark(dataset_name: str):
    """
    Main function to run the full benchmark pipeline for a given MoleculeNet dataset.
    """
    console.rule(f"[bold cyan]MoleculeNet Benchmark: {dataset_name}")
    base_cache_dir = os.path.join("data/molnet/bench_cache", dataset_name)

    # --- 1. Load Data ---
    df_train, df_val, df_test, tasks = load_moleculenet_dataset(dataset_name)

    # --- 2. Setup Config ---
    console.rule("[bold yellow]Configuration Setup")
    config = get_default_config()
    config["num_tasks"] = len(tasks)
    config["batch_size"] = 16
    config["num_epochs"] = 30

    # --- 3. Featurization ---
    console.rule("[bold yellow]Featurization Pipeline")
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    embedding_model = AutoModel.from_pretrained(config["model_checkpoint"]).eval()

    h5_train, glob_train, scaler = featurize_and_save_streaming(
        df=df_train,
        name="train",
        label_cols=tasks,
        cache_dir=base_cache_dir,
        tokenizer=tokenizer,
        model=embedding_model,
        fit_scaler=True,
    )
    h5_val, glob_val, _ = featurize_and_save_streaming(
        df=df_val,
        name="val",
        label_cols=tasks,
        cache_dir=base_cache_dir,
        tokenizer=tokenizer,
        model=embedding_model,
        scaler=scaler,
    )
    h5_test, glob_test, _ = featurize_and_save_streaming(
        df=df_test,
        name="test",
        label_cols=tasks,
        cache_dir=base_cache_dir,
        tokenizer=tokenizer,
        model=embedding_model,
        scaler=scaler,
    )

    graph_train_path = os.path.join(base_cache_dir, "train_graph_feats.pkl")
    graph_val_path = os.path.join(base_cache_dir, "val_graph_feats.pkl")
    graph_test_path = os.path.join(base_cache_dir, "test_graph_feats.pkl")

    combine_graph_parts(glob_train, graph_train_path)
    combine_graph_parts(glob_val, graph_val_path)
    combine_graph_parts(glob_test, graph_test_path)

    # --- 4. DataLoaders ---
    console.rule("[bold yellow]Creating DataLoaders")
    train_loader = torch.utils.data.DataLoader(
        MiniBatchCacheDataset(
            CageFusionStreamingDataset(h5_train, graph_train_path), 512
        ),
        batch_size=config["batch_size"],
        collate_fn=collate_fn_for_cage_fusion,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        MiniBatchCacheDataset(CageFusionStreamingDataset(h5_val, graph_val_path), 512),
        batch_size=config["batch_size"],
        collate_fn=collate_fn_for_cage_fusion,
    )
    test_loader = torch.utils.data.DataLoader(
        MiniBatchCacheDataset(
            CageFusionStreamingDataset(h5_test, graph_test_path), 512
        ),
        batch_size=config["batch_size"],
        collate_fn=collate_fn_for_cage_fusion,
    )

    # --- 5. Model Setup ---
    console.rule("[bold yellow]Model Initialization")
    device = torch.device(config["device"])
    model = CAGEFusionModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()
    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config["warmup_fraction"]),
        num_training_steps=total_steps,
    )

    # --- 6. Training ---
    console.rule("[bold yellow]Starting Training")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        num_epochs=config["num_epochs"],
        num_tasks=config["num_tasks"],
        base_cache_dir=base_cache_dir,
        label_names=tasks,
        tokenizer_obj=tokenizer,
    )

    # --- 7. Final Evaluation ---
    console.rule("[bold green]Final Evaluation on Test Set")
    best_model_path = os.path.join(base_cache_dir, "best_model.pt")
    if not os.path.exists(best_model_path):
        logger.error("Best model checkpoint not found. Cannot run final evaluation.")
        sys.exit(1)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        f"Loaded best model from epoch {checkpoint['epoch']} with Val AUC: {checkpoint.get('best_val_auc', -1):.4f}"
    )

    # CORRECTED: Unpack only 4 values when return_thresholds is False
    test_loss, test_mcc, test_auc, test_pr = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        config["num_tasks"],
        tasks,
        return_thresholds=False,
        cache_dir=os.path.join(base_cache_dir, "test_eval"),
    )

    console.rule("[bold magenta]Final Test Set Results")
    console.print(f"  Test Loss: {test_loss:.4f}")
    console.print(f"  Test AUC:  {test_auc:.4f}")
    console.print(f"  Test MCC:  {test_mcc:.4f}")
    console.print(f"  Test PR:   {test_pr:.4f}")
    console.print("------------------------")
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
    args = parser.parse_args()

    run_benchmark(args.dataset)
