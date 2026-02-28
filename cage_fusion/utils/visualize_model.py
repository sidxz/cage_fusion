#!/usr/bin/env python3
"""
Generates a visual diagram of the CAGEFusionModel architecture using torchviz.

This script initializes the model, creates valid dummy inputs by leveraging the project's
own featurization and data loading pipeline, performs a forward pass, and then uses
the resulting computational graph to render a diagram of the model's structure.
"""

import os
import sys

import torch
import argparse
import numpy as np
import pandas as pd
import tempfile
import shutil
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use the project's own modules to ensure data is created correctly
from cage_fusion.modeling import CAGEFusionModel
from cage_fusion.configuration import CageFusionConfig
from cage_fusion.utils.logging import logger
from cage_fusion.featurization import featurize_and_save_streaming
from cage_fusion.data import CageFusionStreamingDataset, collate_cage_fusion
from cage_fusion.utils import move_bmg_to_device
from cage_fusion.utils.hf_loader import load_tokenizer

# Try to import torchviz, and provide a helpful error message if it's not installed.
try:
    from torchviz import make_dot
except ImportError:
    logger.error("torchviz is not installed. Please install it to use this script:")
    logger.error("pip install torchviz")
    logger.error(
        "You also need to install graphviz. On Debian/Ubuntu: sudo apt-get install graphviz"
    )
    sys.exit(1)


def visualize_model_architecture(output_path="model_architecture"):
    """
    Initializes the CAGEFusionModel, creates valid dummy inputs using the project's
    own data pipeline, and generates a visual graph.
    """
    from rich.console import Console as script_console
    console = script_console()
    
    console.log("[bold blue]Generating model architecture visualization...[/bold blue]")

    # --- 1. Load Model with Default Configuration ---
    config = get_default_config()
    model = CAGEFusionModel(config)
    model.eval()
    device = torch.device(config["device"])
    model.to(device)
    logger.info("CAGEFusionModel initialized with default configuration.")

    # --- 2. Generate a valid batch using the project's data pipeline ---
    console.log("[bold blue]Using existing pipeline to generate valid dummy data...[/bold blue]")
    
    tokenizer = load_tokenizer(config["model_checkpoint"])
    embedding_model = AutoModel.from_pretrained(config["model_checkpoint"]).to(device).eval()

    dummy_scaler = StandardScaler()
    dummy_scaler.fit(np.random.rand(10, config["aux_feature_dim"]))

    temp_dir = tempfile.mkdtemp()
    
    try:
        dummy_df = pd.DataFrame({'SMILES': ['c1ccccc1', 'CC(=O)OC1=CC=CC=C1C(=O)O']})

        h5_path, graph_path, _, num_featurized_samples = featurize_and_save_streaming(
            df=dummy_df,
            name="viz_temp",
            label_cols=['dummy'],
            cache_dir=temp_dir,
            tokenizer=tokenizer,
            model=embedding_model,
            fit_scaler=False,
            scaler=dummy_scaler,
        )

        dataset = CageFusionStreamingDataset(h5_path, graph_path, tokenizer.pad_token_id)
        assert (
            len(dataset) == num_featurized_samples
        ), f"Data integrity check failed: Dataset length ({len(dataset)}) does not match featurized samples ({num_featurized_samples})."
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collate_fn_for_cage_fusion,
            shuffle=False
        )
        
        bmg, sequence_embeddings, attn_mask, aux_feats, _, input_ids_batch, _ = next(iter(loader))
        
        # FIX: Use the project's own utility to move the graph object to the device
        bmg = move_bmg_to_device(bmg, device)
        sequence_embeddings = sequence_embeddings.to(device)
        attn_mask = attn_mask.to(device)
        aux_feats = aux_feats.to(device)
        input_ids_batch = input_ids_batch.to(device)

    finally:
        shutil.rmtree(temp_dir)
        
    logger.info("Valid dummy batch created successfully using the project's data pipeline.")

    # --- 3. Perform a Forward Pass to Get the Computational Graph ---
    model_output = model(
        bmg=bmg,
        sequence_embeddings=sequence_embeddings,
        attn_mask=attn_mask,
        aux_feats=aux_feats,
        input_ids_batch=input_ids_batch,
        return_attn=False,
    )
    logits = model_output[0]

    # --- 4. Generate and Save the Visualization ---
    dot = make_dot(
        logits,
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True,
    )
    
    if '.' in output_path:
        output_path = output_path.split('.')[0]
        
    dot.render(output_path, format="png", view=False, cleanup=True)
    logger.info(
        f"[bold green]✅ Model architecture diagram saved to '{output_path}.png'[/bold green]"
    )


def main():
    """Defines and executes the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate a visual diagram of the CAGE-Fusion model architecture."
    )
    parser.add_argument(
        "--output",
        default="cage_fusion_architecture",
        help="Path and name for the output PNG file (without extension).",
    )
    args = parser.parse_args()
    
    visualize_model_architecture(output_path=args.output)


if __name__ == "__main__":
    main()
