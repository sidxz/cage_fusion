#!/usr/bin/env python3
"""
Provides a programmatic API and a command-line interface for running inference
and generating gradient-based explanations with a trained CAGE-Fusion model.
"""

import os
import sys
import torch
import joblib
import pandas as pd
import numpy as np
import argparse
import traceback
import tempfile
import shutil
from transformers import AutoTokenizer, AutoModel
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cage_fusion.models.cage import CAGEFusionModel
from cage_fusion.engine.dataset import CageFusionStreamingDataset
from cage_fusion.engine.data_utils import collate_fn_for_cage_fusion
from cage_fusion.engine.utils import move_bmg_to_device
from cage_fusion.featurizers.core import featurize_and_save_streaming
from cage_fusion.utils.logging_utils import logger

# Rich console for beautiful output
console = Console()


def generate_saliency_visualization(smiles, tokens, token_saliency, top_n=10):
    """Generates a rich Text object with saliency-highlighted SMILES."""
    if not tokens or token_saliency.size == 0:
        return Text(smiles, style="bold magenta")

    saliency_scores = np.array(token_saliency)
    # Normalize scores to 0-1 range for color mapping
    norm_scores = (saliency_scores - saliency_scores.min()) / (
        saliency_scores.max() - saliency_scores.min() + 1e-9
    )

    text = Text()
    for i, token in enumerate(tokens):
        # Linearly interpolate color from white (low) to bright_red (high)
        clean_token = token.replace("##", "")
        color_val = int(norm_scores[i] * 255)
        style = f"rgb(255,{255-color_val},{255-color_val})"
        text.append(clean_token, style=style)

    return text


def generate_aux_feature_report(aux_saliency, top_n=10):
    """Generates a rich Text object with a ranked list of influential auxiliary features."""
    descriptor_names = [desc[0] for desc in Descriptors._descList]

    if not isinstance(aux_saliency, np.ndarray):
        aux_saliency = aux_saliency.cpu().numpy()

    # Get absolute values for ranking, but keep original sign for reporting
    signed_saliency = aux_saliency.flatten()
    saliency_magnitudes = np.abs(signed_saliency)

    sorted_indices = np.argsort(saliency_magnitudes)[::-1]

    text = Text()
    text.append(
        "Top Influential Auxiliary Features (by gradient magnitude):\n\n",
        style="bold underline cyan",
    )

    for i in range(min(top_n, len(descriptor_names))):
        idx = sorted_indices[i]
        feature_name = descriptor_names[idx]
        saliency_value = signed_saliency[idx]

        # Color based on sign: green for positive (increasing output), red for negative
        style = "green" if saliency_value > 0 else "red"
        sign = "+" if saliency_value > 0 else ""

        text.append(
            f"{i+1}. {feature_name:<20} : {sign}{saliency_value:.4f}\n", style=style
        )

    return text


def predict_and_explain(
    smiles_string: str,
    checkpoint_dir: str,
    target_task: str,
):
    """
    Runs a single prediction and generates gradient-based explanations.
    """
    console.rule(
        f"[bold green]Starting CAGE-Fusion Prediction & Explanation for SMILES: {smiles_string}[/bold green]"
    )

    # === 1. Load Model and Configuration from Checkpoint ===
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    scaler_path = os.path.join(checkpoint_dir, "aux_features_scaler.pkl")

    if not os.path.exists(best_model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Missing 'best_model.pt' or 'aux_features_scaler.pkl' in {checkpoint_dir}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tasks = config["tasks"]

    # Load the best thresholds, providing a default of 0.5 if not found
    all_thresholds = checkpoint.get("best_thresholds", np.full(len(tasks), 0.5))

    if target_task not in tasks:
        raise ValueError(
            f"Target task '{target_task}' not found in model's tasks: {tasks}"
        )
    target_task_index = tasks.index(target_task)

    # Get the specific threshold for our target task
    cutoff_value = all_thresholds[target_task_index]

    model = CAGEFusionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set model to evaluation mode

    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    embedding_model = (
        AutoModel.from_pretrained(config["model_checkpoint"]).to(device).eval()
    )
    scaler = joblib.load(scaler_path)

    # === 2. Featurize the Single SMILES Input ===
    input_df = pd.DataFrame([{"SMILES": smiles_string}])
    dummy_labels = [tasks[0]] if tasks else ["Label"]

    # Use a temporary directory for featurization artifacts
    temp_features_dir = tempfile.mkdtemp()

    try:
        h5_path, graph_path, _ = featurize_and_save_streaming(
            df=input_df,
            name="explain_temp",
            label_cols=dummy_labels,
            cache_dir=temp_features_dir,
            tokenizer=tokenizer,
            model=embedding_model,
            fit_scaler=False,
            scaler=scaler,
        )

        dataset = CageFusionStreamingDataset(
            h5_path, graph_path, tokenizer.pad_token_id
        )
        # We create a loader with batch size 1
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, collate_fn=collate_fn_for_cage_fusion, shuffle=False
        )
        batch = next(iter(loader))

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_features_dir)

    bmg, token_embs, attn_mask, aux_feats, _, input_ids, smiles_batch = batch

    # Move to device
    bmg = move_bmg_to_device(bmg, device)
    token_embs, attn_mask, aux_feats, input_ids = (
        token_embs.to(device),
        attn_mask.to(device),
        aux_feats.to(device),
        input_ids.to(device),
    )

    # === 3. Enable Gradient Tracking on Inputs ===
    token_embs.requires_grad_(True)
    aux_feats.requires_grad_(True)

    # Make sure the model doesn't clear gradients
    model.zero_grad()

    # === 4. Forward Pass to Get Logits ===
    # We call the model directly, not through an evaluation script
    logits, _, _, _, _, _, _, _ = model(
        bmg=bmg,
        sequence_embeddings=token_embs,
        attn_mask=attn_mask,
        aux_feats=aux_feats,
        input_ids_batch=input_ids,
        return_attn=False,  # We don't need attention weights for this
    )

    # Isolate the logit for the target task
    prediction_score = logits[0, target_task_index]

    # The final predicted probability
    final_prob = torch.sigmoid(prediction_score).item()

    # Determine the predicted class based on the cutoff
    predicted_class = 1 if final_prob >= cutoff_value else 0

    # === 5. Backward Pass to Compute Gradients ===
    prediction_score.backward()

    # === 6. Extract and Process Gradients (Saliency) ===
    # Saliency for token embeddings
    token_saliency = token_embs.grad.norm(dim=-1).squeeze(0).cpu().numpy()

    # Saliency for auxiliary features
    aux_saliency = aux_feats.grad.cpu().numpy()

    # === 7. Generate and Display Explanation ===
    active_tokens_mask = input_ids.squeeze(0) != tokenizer.pad_token_id
    tokens = tokenizer.convert_ids_to_tokens(
        input_ids.squeeze(0)[active_tokens_mask].cpu().numpy()
    )
    token_saliency_cleaned = token_saliency[active_tokens_mask.cpu().numpy()]

    # Generate visualization components
    saliency_viz = generate_saliency_visualization(
        smiles_string, tokens, token_saliency_cleaned
    )
    aux_report = generate_aux_feature_report(aux_saliency)

    # Build the summary text
    summary_text = Text(justify="center")
    summary_text.append(f"SMILES: {smiles_string}\n")
    summary_text.append(f"Prediction Score for '{target_task}': {final_prob:.4f}\n")
    summary_text.append(f"Cutoff Threshold: {cutoff_value:.4f}\n\n")
    summary_text.append(
        f"Predicted Class: {predicted_class}",
        style="bold yellow" if predicted_class == 1 else "bold green",
    )

    # Display final report
    console.print(
        Panel(
            summary_text,
            title="[bold blue]Prediction Summary[/bold blue]",
            border_style="blue",
        )
    )

    console.print(
        Panel(
            saliency_viz,
            title="[bold blue]Structural Feature Saliency (SMILES)[/bold blue]",
            subtitle="Brighter tokens have a higher impact on the outcome.",
            border_style="blue",
        )
    )

    console.print(
        Panel(
            aux_report,
            title="[bold blue]Auxiliary Feature Contribution[/bold blue]",
            subtitle="Positive (green) values pushed the prediction higher, negative (red) values pushed it lower.",
            border_style="blue",
        )
    )


def main():
    """Defines and executes the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Predict with a CAGE-Fusion model and explain the result using gradient-based saliency."
    )
    parser.add_argument(
        "--smiles",
        required=True,
        help="The single SMILES string to predict and explain.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to the directory containing model checkpoints ('best_model.pt') and the scaler.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="The name of the target task (column name) to explain the prediction for.",
    )
    args = parser.parse_args()

    try:
        predict_and_explain(
            smiles_string=args.smiles,
            checkpoint_dir=args.checkpoint_dir,
            target_task=args.task,
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
