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

# Rich setup
install()
console = Console()

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
from cage_fusion.configs import get_default_config
from cage_fusion.models import CAGEFusionModel
from cage_fusion.featurizers import featurize_and_save_streaming
from cage_fusion.engine.training import train_model
from cage_fusion.engine.data_utils import collate_fn_for_cage_fusion
from cage_fusion.engine.dataset import CageFusionStreamingDataset, MiniBatchCacheDataset
from cage_fusion.utils.logging_utils import logger
from cage_fusion.benchmarks.moleculenet.loader import load_molnet_dataset


def run(dataset_name="bace", epochs=5):
    console.rule(f"[bold cyan]Benchmark: MoleculeNet - {dataset_name.upper()}")

    # Step 1: Load data
    console.rule("[bold yellow]Step 1: Load Dataset")
    dataset, tasks, _ = load_molnet_dataset(dataset_name)
    df = pd.DataFrame({
        "SMILES_Canonical": dataset.X,
        **{task: dataset.y[:, i] for i, task in enumerate(tasks)}
    })

    # Step 2: Config
    console.rule("[bold yellow]Step 2: Configuration")
    config = get_default_config()
    config.update({
        "num_tasks": len(tasks),
        "batch_size": 32,
        "num_epochs": epochs,
    })
    logger.info(f"Loaded config with {len(tasks)} tasks")

    # Step 3: Tokenizer + Model
    console.rule("[bold yellow]Step 3: Model and Tokenizer Setup")
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    encoder_model = AutoModel.from_pretrained(config["model_checkpoint"])

    # Step 4: Featurization
    console.rule("[bold yellow]Step 4: Featurization")
    cache_dir = os.path.join("benchmark_cache", dataset_name)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    h5_path, graph_path, _ = featurize_and_save_streaming(
        df=df, name=dataset_name, label_cols=tasks,
        cache_dir=cache_dir, tokenizer=tokenizer,
        model=encoder_model, fit_scaler=True
    )

    # Step 5: DataLoader
    console.rule("[bold yellow]Step 5: Dataloader")
    dataset = MiniBatchCacheDataset(CageFusionStreamingDataset(h5_path, graph_path), cache_size=64)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config["batch_size"], collate_fn=collate_fn_for_cage_fusion
    )

    # Step 6: Training
    console.rule("[bold yellow]Step 6: Training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAGEFusionModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(loader) * epochs, gamma=1.0)

    try:
        train_model(
            model, train_loader=loader, val_loader=loader,
            optimizer=optimizer, criterion=criterion,
            scheduler=scheduler, device=device,
            num_epochs=epochs, num_tasks=len(tasks),
            base_cache_dir=cache_dir, label_names=tasks,
            tokenizer_obj=tokenizer
        )
    except Exception:
        console.rule("[bold red]❌ Benchmark Failed")
        logger.error("Benchmark training failed. See traceback below:")
        traceback.print_exc()
        return

    # Step 7: Cleanup
    console.rule("[bold green]✅ Benchmark Completed Successfully!")


if __name__ == "__main__":
    run()
