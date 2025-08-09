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
from rdkit import Chem
from functools import partial


# Set environment variable to handle tokenizer parallelism warning
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cage_fusion.configs import get_default_config
from cage_fusion.featurizers import featurize_and_save_streaming
from cage_fusion.models import CAGEFusionModel
from cage_fusion.engine.dataset import CageFusionStreamingDataset, MiniBatchCacheDataset
from cage_fusion.engine.data_utils import collate_fn_for_cage_fusion
from cage_fusion.viz.token_viz import (
    visualize_top_token_attentions,
    visualize_attention_weights,
    visualize_total_atom_contribution,
    visualize_combined_atom_contribution,
)
from cage_fusion.viz.prompt_viz import visualize_fg_attention
from cage_fusion.utils.logging_utils import logger
from cage_fusion.engine.utils import move_bmg_to_device

install()
console = Console()


def predict_smiles(
    input_df: pd.DataFrame,
    checkpoint_dir: str,
    batch_size: int = 256,
    temp_dir: Optional[str] = None,
    plot_all_attention: bool = False,
    attn_plot_dir: Optional[str] = None,
    model_file_name: Optional[str] = "best_model.pt",
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

    best_model_path = os.path.join(checkpoint_dir, model_file_name)
    scaler_path = os.path.join(checkpoint_dir, "aux_features_scaler.pkl")

    if not os.path.exists(best_model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Missing '{model_file_name}' or 'aux_features_scaler.pkl' in {checkpoint_dir}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tasks = config["tasks"]
    best_thresholds = checkpoint.get("best_thresholds", np.full(len(tasks), 0.5))
    logger.info(f"Loaded model for tasks: {tasks} with thresholds: {best_thresholds}")

    model = CAGEFusionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    embedding_model = (
        AutoModel.from_pretrained(config["model_checkpoint"]).to(device).eval()
    )
    scaler = joblib.load(scaler_path)
    if scaler is None or not hasattr(scaler, "mean_"):
        raise ValueError("Failed to load a valid, fitted scaler.")

    df_with_original_index = (
        input_df.copy().reset_index().rename(columns={"index": "original_index"})
    )
    dummy_labels = [tasks[0]] if tasks else ["Label"]
    temp_features_dir = temp_dir or tempfile.mkdtemp()
    os.makedirs(temp_features_dir, exist_ok=True)

    h5_path, returned_scaler, num_featurized_samples = featurize_and_save_streaming(
        df=df_with_original_index,
        name="inference",
        label_cols=dummy_labels,
        cache_dir=temp_features_dir,
        tokenizer=tokenizer,
        model=embedding_model,
        fit_scaler=False,
        scaler=scaler,
        batch_size=5,
    )

    print(f"Featurization complete. HDF5 path: {h5_path}")

    

    dataset = CageFusionStreamingDataset(h5_path, tokenizer.pad_token_id)
    assert len(dataset) == num_featurized_samples, "Data integrity check failed."
    
    collate_with_pad = partial(collate_fn_for_cage_fusion, pad_token_id=tokenizer.pad_token_id)

    

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_with_pad,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    for batch in loader:
        # print batch contents for debugging
        (
            bmg,
            token_embs,
            attn_mask,
            aux_feats,
            labels_tensor,
            input_ids,
            smiles_batch,
            original_indices_batch,
            ids_list
        ) = batch
        print(f"Batch BMG: {bmg}")
        print(f"Batch token embeddings shape: {token_embs.shape}")
        print(f"Batch attention mask shape: {attn_mask.shape}")
        # print attn mask of first sample
        # print(f"Batch attention mask: {attn_mask[0]}")
        print(f"Batch auxiliary features shape: {aux_feats.shape}")

        print(f"Batch labels shape: {labels_tensor.shape}")
        print(f"Batch input IDs shape: {input_ids.shape}")
        print(f"Batch SMILES: {smiles_batch}")
        print(f"Batch original indices: {original_indices_batch}")
        print(f"Batch IDs: {ids_list}")
        # DEBUG END

    # DEBUG END
    return

    all_preds = []
    all_original_indices = []
    all_top_attention_tokens = []

    if plot_all_attention and attn_plot_dir:
        os.makedirs(attn_plot_dir, exist_ok=True)
        logger.info(f"Attention plots will be saved to: {attn_plot_dir}")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            if batch is None:
                continue

            (
                bmg,
                token_embs,
                attn_mask,
                aux_feats,
                _,
                input_ids,
                smiles_batch,
                original_indices_batch,
            ) = batch

            bmg = move_bmg_to_device(bmg, device)
            token_embs, attn_mask, aux_feats, input_ids = [
                t.to(device) for t in [token_embs, attn_mask, aux_feats, input_ids]
            ]

            model_output = model(
                bmg=bmg,
                sequence_embeddings=token_embs,
                attn_mask=attn_mask,
                aux_feats=aux_feats,
                input_ids_batch=input_ids,
                smiles_batch=smiles_batch,
                return_attn=plot_all_attention,
            )

            logits, _, _, g2t_weights, t2a_weights, _, _, _, prompt_attn_weights = (
                model_output
            )

            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_original_indices.append(original_indices_batch.cpu().numpy())

            if plot_all_attention and attn_plot_dir:
                for j in range(len(smiles_batch)):
                    original_idx = original_indices_batch[j].item()
                    sample_plot_dir = os.path.join(attn_plot_dir, f"idx_{original_idx}")
                    os.makedirs(sample_plot_dir, exist_ok=True)

                    if g2t_weights is not None and t2a_weights is not None:
                        smiles_to_plot = smiles_batch[j]
                        token_scores = g2t_weights[j].sum(dim=(0, 1))
                        special_ids = [
                            tokenizer.pad_token_id,
                            tokenizer.cls_token_id,
                            tokenizer.sep_token_id,
                        ]
                        for special_id in special_ids:
                            token_scores[input_ids[j] == special_id] = -1e9

                        top_token_indices = torch.argsort(
                            token_scores, descending=True
                        )[:3]

                        full_token_list_ids = input_ids[j].cpu().numpy()
                        actual_tokens_mask = (
                            full_token_list_ids != tokenizer.pad_token_id
                        )
                        actual_tokens_ids = full_token_list_ids[actual_tokens_mask]
                        full_token_list_str = tokenizer.convert_ids_to_tokens(
                            actual_tokens_ids
                        )

                        top_tokens = [
                            full_token_list_str[idx]
                            for idx in top_token_indices.cpu().numpy()
                            if idx < len(full_token_list_str)
                        ]
                        all_top_attention_tokens.append(top_tokens)

                        visualize_top_token_attentions(
                            smiles=smiles_to_plot,
                            attention_weights=t2a_weights[j],
                            full_token_list=full_token_list_str,
                            top_token_indices=top_token_indices.cpu().numpy(),
                            output_dir=sample_plot_dir,
                        )

                        # Total contribution
                        attn = t2a_weights[j]
                        if attn.ndim == 3:
                            attn = attn.mean(dim=0)

                        logit_vec = logits[j].detach().cpu().numpy()
                        task_idx = np.argmax(np.abs(logit_vec))
                        pred_logit = logit_vec[task_idx]

                        visualize_total_atom_contribution(
                            smiles=smiles_to_plot,
                            t2a_weights_sample=attn,
                            pred_logit=pred_logit,
                            output_path=os.path.join(
                                sample_plot_dir, "atom_total_contrib.png"
                            ),
                        )

                        # FG Attention (prompt)
                        if prompt_attn_weights and prompt_attn_weights[j] is not None:
                            visualize_fg_attention(
                                smiles=smiles_to_plot,
                                prompt_attn_weights=prompt_attn_weights[j],
                                output_path=os.path.join(
                                    sample_plot_dir, "fg_prompt_attention.png"
                                ),
                                title="Functional Group Attention (PROMPT)",
                            )

                            weight_fg = float(model.alpha.detach().cpu().item())
                            weight_t2a = float(model.scale_graph.detach().cpu().item())

                            visualize_combined_atom_contribution(
                                smiles=smiles_to_plot,
                                t2a_weights_sample=attn,
                                pred_logit=pred_logit,
                                prompt_attn_weights=prompt_attn_weights[j],
                                output_path=os.path.join(
                                    sample_plot_dir, "atom_combined_contrib.png"
                                ),
                                weight_t2a=weight_t2a,
                                weight_fg=weight_fg,
                            )
                        else:
                            logger.warning(
                                f"No prompt attention found for sample idx {original_idx}"
                            )
                    else:
                        all_top_attention_tokens.append(None)
            else:
                all_top_attention_tokens.extend([None] * len(original_indices_batch))

    final_df = input_df.copy()
    if all_preds:
        all_preds_np = np.concatenate(all_preds, axis=0)
        all_original_indices_np = np.concatenate(all_original_indices, axis=0)

        results_df = pd.DataFrame({"original_index": all_original_indices_np})
        for idx, task in enumerate(tasks):
            results_df[f"pred_score_{task}"] = all_preds_np[:, idx]
            results_df[f"pred_label_{task}"] = (
                all_preds_np[:, idx] > best_thresholds[idx]
            ).astype(int)

        results_df["top_attention_tokens"] = all_top_attention_tokens

        final_df = input_df.reset_index().rename(columns={"index": "original_index"})
        final_df = final_df.merge(results_df, on="original_index", how="left")
        final_df.drop(columns=["original_index"], inplace=True)

    try:
        if temp_dir is None:
            shutil.rmtree(temp_features_dir)
            logger.info(f"Cleaned up temporary directory: {temp_features_dir}")
    except Exception as e:
        logger.warning(f"Could not remove temp features dir: {e}")

    return final_df


def main():
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
        "--batch-size", type=int, default=256, help="Batch size for inference"
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Optional path for temporary featurization files.",
    )
    parser.add_argument(
        "--plot-all-attention",
        action="store_true",
        help="If set, generate attention plots for every SMILES.",
    )
    parser.add_argument(
        "--attn-plot-dir",
        default="./attention_plots_prediction",
        help="Directory to save all attention plots.",
    )
    parser.add_argument(
        "--model-file-name",
        default="best_model.pt",
        help="Name of the model file to load.",
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
            model_file_name=args.model_file_name,
        )
        # predictions_df.to_csv(args.output, index=False)
        # logger.info(f"Predictions successfully saved to {args.output}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
