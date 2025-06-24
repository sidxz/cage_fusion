#!/usr/bin/env python3
"""
Provides a programmatic API and a command-line interface for running inference
with a trained CAGE-Fusion model.
"""

import os
import sys
import torch
import joblib
import pandas as pd
import numpy as np
import argparse
import traceback
import h5py
from transformers import AutoTokenizer, AutoModel
from rich.console import Console
from rich.traceback import install
import shutil
from tqdm import tqdm
from typing import List, Optional

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cage_fusion.configs import get_default_config
from cage_fusion.featurizers import featurize_and_save_streaming
from cage_fusion.models import CAGEFusionModel
from cage_fusion.engine.dataset import CageFusionStreamingDataset
from cage_fusion.engine.data_utils import collate_fn_for_cage_fusion
from cage_fusion.viz.token_viz import (
    visualize_top_token_attentions,
    visualize_attention_weights,
)
from cage_fusion.utils.logging_utils import logger

install()
console = Console()


def predict_smiles(
    input_df: pd.DataFrame,
    checkpoint_dir: str,
    batch_size: int = 200,
    temp_dir: str = None,
    plot_all_attention: bool = False,
    attn_plot_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Core API function to run inference on a DataFrame containing SMILES strings.
    """
    console.rule("[bold green]Starting CAGE-Fusion Prediction[/bold green]")

    if "SMILES" not in input_df.columns:
        raise ValueError("Input DataFrame must contain a 'SMILES' column.")

    if plot_all_attention and not attn_plot_dir:
        raise ValueError(
            "'attn_plot_dir' must be provided if 'plot_all_attention' is True."
        )

    # === 1. Load config and checkpoint ===
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
    best_thresholds = checkpoint.get("best_thresholds", np.full(len(tasks), 0.5))
    logger.info(f"Loaded model for tasks: {tasks} with thresholds: {best_thresholds}")

    # === 2. Load model and components ===
    model = CAGEFusionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    embedding_model = (
        AutoModel.from_pretrained(config["model_checkpoint"]).to(device).eval()
    )
    scaler = joblib.load(scaler_path)
    if scaler and hasattr(scaler, "mean_"):
        logger.info(f"Scaler loaded successfully. Type: {type(scaler)}")
        logger.info(f"Scaler is FITTED. Mean shape: {scaler.mean_.shape}")
    else:
        logger.error("SCALER IS NOT VALID. It might be None or unfitted.")
        raise ValueError("Failed to load a valid, fitted scaler.")
    logger.info("Model and necessary components loaded.")

    # === 3. Featurize the input SMILES ===
    df = input_df.copy().reset_index().rename(columns={"index": "original_index"})
    dummy_labels = [tasks[0]] if tasks else ["Label"]

    if temp_dir:
        temp_features_dir = os.path.join(temp_dir, f"_inference_temp_{os.getpid()}")
    else:
        temp_features_dir = os.path.join(
            checkpoint_dir, f"_inference_temp_{os.getpid()}"
        )
    os.makedirs(temp_features_dir, exist_ok=True)

    logger.info(f"Featurizing {len(df)} SMILES...")
    h5_path, _, _ = featurize_and_save_streaming(
        df=df,
        name="inference",
        label_cols=dummy_labels,
        cache_dir=temp_features_dir,
        tokenizer=tokenizer,
        model=embedding_model,
        fit_scaler=False,
        scaler=scaler,
    )

    final_graph_path = os.path.join(temp_features_dir, "inference_graph_feats.pkl")
    dataset = CageFusionStreamingDataset(h5_path, final_graph_path)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_for_cage_fusion,
        shuffle=False,
    )

    # === 4. Predict ===
    logger.info("Running model predictions...")
    all_preds = []
    sample_idx_offset = 0
    if plot_all_attention:
        os.makedirs(attn_plot_dir, exist_ok=True)
        logger.info(f"Plotting all attentions to: {attn_plot_dir}")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Predicting")):
            if batch is None or batch[0] is None:
                logger.warning(
                    f"Skipping batch {i} due to featurization failure or empty batch."
                )
                continue

            bmg, token_embs, attn_mask, aux_feats, _, input_ids, smiles_batch = batch
            bmg = move_bmg_to_device(bmg, device)
            token_embs, attn_mask = token_embs.to(device), attn_mask.to(device)
            aux_feats, input_ids = aux_feats.to(device), input_ids.to(device)

            model_output = model(
                bmg=bmg,
                sequence_embeddings=token_embs,
                attn_mask=attn_mask,
                aux_feats=aux_feats,
                input_ids_batch=input_ids,
                return_attn=plot_all_attention,
            )

            logits, _, _, g2t_weights, t2a_weights, _, _ = model_output

            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)

            if plot_all_attention:
                for j in range(len(smiles_batch)):
                    with h5py.File(h5_path, "r") as f:
                        original_idx = f["original_indices"][sample_idx_offset + j]

                    safe_fname = "".join(c for c in smiles_batch[j] if c.isalnum())[:50]

                    if g2t_weights is not None:
                        plot_path_g2t = os.path.join(
                            attn_plot_dir,
                            f"g2t_attn_idx_{original_idx}_{safe_fname}.png",
                        )
                        visualize_attention_weights(
                            g2t_weights[j],
                            attn_mask[j],
                            model.num_heads,
                            plot_path_g2t,
                            input_ids[j],
                            tokenizer,
                        )

                    if t2a_weights is not None:
                        for token_idx in range(1, 4):
                            if token_idx < len(input_ids[j]):
                                token_str = tokenizer.convert_ids_to_tokens(
                                    [input_ids[j][token_idx].item()]
                                )[0]
                                plot_path_t2a = os.path.join(
                                    attn_plot_dir,
                                    f"t2a_attn_idx_{original_idx}_token_{token_idx}({token_str})_{safe_fname}.png",
                                )
                                # visualize_token_to_atom_attention(
                                #     smiles_batch[j],
                                #     t2a_weights[j],
                                #     token_idx,
                                #     token_str,
                                #     plot_path_t2a,
                                # )

            sample_idx_offset += len(input_ids)

    # === 5. Format Output ===
    output_df = input_df.copy()
    if all_preds:
        all_preds_np = np.concatenate(all_preds, axis=0)
        with h5py.File(h5_path, "r") as f:
            if "original_indices" in f:
                valid_indices = f["original_indices"][:]
                results_df = pd.DataFrame(index=valid_indices)
                for idx, task in enumerate(tasks):
                    results_df[f"pred_score_{task}"] = all_preds_np[:, idx]
                    results_df[f"pred_label_{task}"] = (
                        all_preds_np[:, idx] > best_thresholds[idx]
                    ).astype(int)
                output_df = output_df.merge(
                    results_df, left_index=True, right_index=True, how="left"
                )
            else:
                logger.error(
                    "Critical: 'original_indices' not found in HDF5. Cannot robustly merge results."
                )

    # === 6. Clean up ===
    try:
        shutil.rmtree(temp_features_dir)
        logger.info(f"Cleaned up temporary directory: {temp_features_dir}")
    except Exception as e:
        logger.warning(f"Could not remove temp features dir: {e}")

    return output_df


def main():
    """Defines and executes the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Predict with CAGE-Fusion model on a new CSV file"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to input CSV (must contain a 'SMILES' column)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to the directory containing model checkpoints",
    )
    parser.add_argument(
        "--output", required=True, help="Path to output CSV for predictions"
    )
    parser.add_argument(
        "--batch-size", type=int, default=200, help="Batch size for inference"
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Optional path for temporary featurization files.",
    )
    parser.add_argument(
        "--plot-all-attention",
        action="store_true",
        help="If set, generate BOTH graph-to-token and token-to-atom attention plots for every SMILES.",
    )
    parser.add_argument(
        "--attn-plot-dir",
        default="./attention_plots_prediction",
        help="Directory to save all attention plots.",
    )
    args = parser.parse_args()

    try:
        input_df = pd.read_csv(args.csv)

        predictions_df = predict_smiles(
            input_df=input_df,
            checkpoint_dir=args.checkpoint_dir,
            batch_size=args.batch_size,
            temp_dir=args.temp_dir,
            plot_all_attention=args.plot_all_attention,
            attn_plot_dir=args.attn_plot_dir,
        )

        predictions_df.to_csv(args.output, index=False)
        # --- CORRECTED: Fixed the variable name in the final log message ---
        logger.info(f"Predictions successfully saved to {args.output}")
        # --------------------------------------------------------------------

    except FileNotFoundError as e:
        logger.error(f"A required file was not found: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
