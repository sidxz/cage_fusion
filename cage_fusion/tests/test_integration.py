import os
import sys
import torch
import pandas as pd
import shutil
import traceback
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from rich.console import Console
from rich.traceback import install

# Install rich traceback globally
install()
console = Console()

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
from cage_fusion.configs import get_default_config
from cage_fusion.engine.dataset import CageFusionStreamingDataset, MiniBatchCacheDataset
from cage_fusion.engine.training import train_model
from cage_fusion.engine.data_utils import collate_fn_for_cage_fusion
from cage_fusion.featurizers import featurize_and_save_streaming
from cage_fusion.models import CAGEFusionModel
from cage_fusion.utils.logging_utils import logger


def run_library_test():
    """
    A self-contained integration test for the cage_fusion library.
    It uses dummy data to run the entire training pipeline for 5 epochs.
    """
    console.rule("[bold cyan]CAGE Fusion Library Integration Test")
    logger.info("Starting CAGE Fusion Library Integration Test")

    # --- Step 1: Setup Paths and Dummy Data ---
    console.rule("[bold yellow]Step 1/7: Setup")
    test_dir = os.path.dirname(__file__)
    data_path = os.path.join(test_dir, "dummy_data.csv")
    cache_dir = os.path.join(test_dir, "test_cache")

    if not os.path.exists(data_path):
        logger.error(f"Test file 'dummy_data.csv' not found at {data_path}")
        return

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    labels = ["Task_1", "Task_2"]
    logger.info("Environment setup complete")

    # --- Step 2: Load Config and Models ---
    console.rule("[bold yellow]Step 2/7: Configuration and Models")
    config = get_default_config()
    config["num_tasks"] = 2
    config["batch_size"] = 2
    config["num_epochs"] = 5
    config["base_cache_dir"] = cache_dir  # Add the cache dir to the config
    logger.info(
        f"Using Test Config: Epochs={config['num_epochs']}, "
        f"Batch Size={config['batch_size']}, Tasks={config['num_tasks']}"
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    embedding_model = AutoModel.from_pretrained(config["model_checkpoint"])
    logger.info("Configuration and pre-trained models loaded")

    # --- Step 3: Featurization ---
    console.rule("[bold yellow]Step 3/7: Featurization")
    h5_path, _, _, num_featurized_samples = featurize_and_save_streaming(
        df=df,
        name="dummy",
        label_cols=labels,
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        model=embedding_model,
        fit_scaler=True,
    )
    logger.info("Featurization complete")

    # --- Step 4: DataLoader ---
    console.rule("[bold yellow]Step 4/7: DataLoader Setup")
    graph_path = os.path.join(cache_dir, "dummy_graph_feats_part_0.pkl")
    mini_cache_size = 4

    train_dataset = MiniBatchCacheDataset(
        CageFusionStreamingDataset(h5_path, graph_path), cache_size=mini_cache_size
    )
    val_dataset = MiniBatchCacheDataset(
        CageFusionStreamingDataset(h5_path, graph_path), cache_size=mini_cache_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        collate_fn=collate_fn_for_cage_fusion,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        collate_fn=collate_fn_for_cage_fusion,
    )
    logger.info("Train and Validation DataLoaders created")

    # --- Step 5: Initialize Model ---
    console.rule("[bold yellow]Step 5/7: Model Initialization")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAGEFusionModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=total_steps, gamma=1.0)
    logger.info(f"Model initialized on device: {device}")

    # --- Step 6: Training Loop Test ---
    console.rule("[bold yellow]Step 6/7: Training")
    try:
        # CORRECTED: Call train_model with the config dictionary
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            config=config,  # Pass the entire config object
            label_names=labels,
            tokenizer_obj=tokenizer,
        )
        assert (
            "train_loss" in history
            and len(history["train_loss"]) == config["num_epochs"]
        )
        assert (
            "val_loss" in history and len(history["val_loss"]) == config["num_epochs"]
        )
        logger.info("Training completed successfully")
        logger.info(
            f"Final train loss: {history['train_loss'][-1]:.4f}, "
            f"val loss: {history['val_loss'][-1]:.4f}"
        )
    except Exception:
        logger.error("Training loop failed. See traceback below:")
        traceback.print_exc()
        shutil.rmtree(cache_dir)
        return

    # --- Step 7: Cleanup ---
    console.rule("[bold yellow]Step 7/7: Cleanup")
    shutil.rmtree(cache_dir)
    logger.info("Library integration test completed successfully")
    console.rule("[bold green]✅ Integration Test Completed!")


if __name__ == "__main__":
    run_library_test()
