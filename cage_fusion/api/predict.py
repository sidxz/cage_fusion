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
import tempfile
from transformers import AutoTokenizer, AutoModel
from rich.console import Console
from rich.traceback import install
import shutil
from tqdm import tqdm
from typing import List, Optional
from rdkit import Chem
from functools import partial

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


def _worker_init(_):
    # prevent thread oversubscription per worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    # for read-only HDF5 access across processes, this can reduce stalls
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def _plot_batch_attentions(
    j: int,
    tokenizer,
    logits: torch.Tensor,
    g2t_weights: Optional[torch.Tensor],
    t2a_weights: Optional[torch.Tensor],
    input_ids: torch.Tensor,
    smiles_batch: List[str],
    prompt_attn_weights,
    attn_plot_dir: str,
    original_idx: int,
    weight_fg: Optional[float] = None,
    weight_t2a: Optional[float] = None,
) -> List[str]:
    """
    Plot all attention visualizations for a single sample in the batch
    and return top tokens (as strings) for optional inclusion in the CSV.
    """
    if g2t_weights is None or t2a_weights is None:
        return []

    sample_plot_dir = os.path.join(attn_plot_dir, f"idx_{original_idx}")
    os.makedirs(sample_plot_dir, exist_ok=True)

    smiles_to_plot = smiles_batch[j]

    # ----- Top tokens from graph->token -----
    token_scores = g2t_weights[j].sum(dim=(0, 1))
    special_ids = [
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
    ]
    for sid in special_ids:
        token_scores[input_ids[j] == sid] = -1e9

    top_token_indices = torch.argsort(token_scores, descending=True)[:3]
    full_ids = input_ids[j].detach().cpu().numpy()
    actual_ids = full_ids[full_ids != tokenizer.pad_token_id]
    full_tokens = tokenizer.convert_ids_to_tokens(actual_ids)

    # Plot top token attentions (token -> atoms)
    visualize_top_token_attentions(
        smiles=smiles_to_plot,
        attention_weights=t2a_weights[j],
        full_token_list=full_tokens,
        top_token_indices=top_token_indices.detach().cpu().numpy(),
        output_dir=sample_plot_dir,
    )

    # ----- Total atom contribution (token->atom) -----
    attn = t2a_weights[j]
    if attn.ndim == 3:
        attn = attn.mean(dim=0)

    logit_vec = logits[j].detach().cpu().numpy()
    task_idx = int(np.argmax(np.abs(logit_vec)))
    pred_logit = float(logit_vec[task_idx])

    visualize_total_atom_contribution(
        smiles=smiles_to_plot,
        t2a_weights_sample=attn,
        pred_logit=pred_logit,
        output_path=os.path.join(sample_plot_dir, "atom_total_contrib.png"),
    )

    # ----- FG attention (prompt) and combined map -----
    if prompt_attn_weights and prompt_attn_weights[j] is not None:
        visualize_fg_attention(
            smiles=smiles_to_plot,
            prompt_attn_weights=prompt_attn_weights[j],
            output_path=os.path.join(sample_plot_dir, "fg_prompt_attention.png"),
            title="Functional Group Attention (PROMPT)",
        )
        if (weight_fg is not None) and (weight_t2a is not None):
            visualize_combined_atom_contribution(
                smiles=smiles_to_plot,
                t2a_weights_sample=attn,
                pred_logit=pred_logit,
                prompt_attn_weights=prompt_attn_weights[j],
                output_path=os.path.join(sample_plot_dir, "atom_combined_contrib.png"),
                weight_t2a=float(weight_t2a),
                weight_fg=float(weight_fg),
            )

    kept = []
    top_idx_np = set(top_token_indices.detach().cpu().numpy().tolist())
    for idx, tok in enumerate(full_tokens):
        if idx in top_idx_np:
            kept.append(tok)
    return kept


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

    # temp features dir
    temp_features_dir = temp_dir or tempfile.mkdtemp()
    os.makedirs(temp_features_dir, exist_ok=True)

    h5_path, returned_scaler, num_featurized_samples = featurize_and_save_streaming(
        df=input_df,
        name="inference",
        label_cols=[],
        cache_dir=temp_features_dir,
        tokenizer=tokenizer,
        model=embedding_model,
        fit_scaler=False,
        scaler=scaler,
        batch_size=batch_size,
    )

    collate_with_pad = partial(
        collate_fn_for_cage_fusion, pad_token_id=tokenizer.pad_token_id
    )
    num_workers = 0

    common_loader_kwargs = dict(
        collate_fn=collate_with_pad,
        num_workers=num_workers,
        worker_init_fn=_worker_init,
    )
    common_dataset_kwargs = dict(
        tokenizer_pad_id=tokenizer.pad_token_id,
        prefer_normalized_aux=True,
        return_ids=True,
        total_num_workers=num_workers,
        graph_cache="auto",
        single_worker_graph_cache=True,  # only worker 0 caches graphs
        emb_cache_store_dtype=np.float32,  # or np.float16 if your HDF5 is fp16
        return_emb_dtype=torch.float32,  # model expects fp32
    )

    loader = torch.utils.data.DataLoader(
        CageFusionStreamingDataset(
            h5_path,
            **common_dataset_kwargs,
        ),
        batch_size=config.get("batch_size", batch_size),
        shuffle=False,
        **common_loader_kwargs,
    )

    logger.info(f"Loaded {len(loader)} batches for inference.")
    predictions_df = pd.DataFrame()

    if plot_all_attention and attn_plot_dir:
        os.makedirs(attn_plot_dir, exist_ok=True)
        logger.info(f"Attention plots will be saved to: {attn_plot_dir}")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            if batch is None:
                logger.warning("Received an empty batch, skipping.")
                continue

            (
                bmg,
                token_embs,
                attn_mask,
                aux_feats,
                labels,
                input_ids,
                smiles_batch,
                original_indices_batch,
                ids_list,
            ) = batch

            # move to device
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

            if logits.shape[0] != len(smiles_batch):
                raise ValueError(
                    f"Logits shape {logits.shape} does not match batch size {len(smiles_batch)}"
                )

            probabilities = torch.sigmoid(logits).detach().cpu().numpy()

            # Build batch predictions DataFrame
            batch_predictions_df = pd.DataFrame(
                {
                    "Original Index": original_indices_batch.detach().cpu().numpy(),
                    "Id": ids_list,
                    "SMILES": smiles_batch,
                }
            )
            for idx, task in enumerate(tasks):
                batch_predictions_df[f"pred_class_{task}"] = (
                    probabilities[:, idx] > best_thresholds[idx]
                ).astype(int)
                batch_predictions_df[task] = probabilities[:, idx]

            # Optional: per-sample top tokens + attention plots (CSV-friendly string)
            if plot_all_attention and attn_plot_dir:
                weight_fg = (
                    float(model.alpha.detach().cpu().item())
                    if hasattr(model, "alpha")
                    else None
                )
                weight_t2a = (
                    float(model.scale_graph.detach().cpu().item())
                    if hasattr(model, "scale_graph")
                    else None
                )

                batch_top_tokens = []
                for j in range(len(smiles_batch)):
                    original_idx = int(original_indices_batch[j].item())
                    kept_tokens = _plot_batch_attentions(
                        j=j,
                        tokenizer=tokenizer,
                        logits=logits,
                        g2t_weights=g2t_weights,
                        t2a_weights=t2a_weights,
                        input_ids=input_ids,
                        smiles_batch=smiles_batch,
                        prompt_attn_weights=prompt_attn_weights,
                        attn_plot_dir=attn_plot_dir,
                        original_idx=original_idx,
                        weight_fg=weight_fg,
                        weight_t2a=weight_t2a,
                    )
                    batch_top_tokens.append(
                        "|".join(kept_tokens) if kept_tokens else ""
                    )
            else:
                batch_top_tokens = [""] * len(smiles_batch)

            batch_predictions_df["top_tokens"] = batch_top_tokens

            # Reorder columns
            ordered_cols = (
                ["Original Index", "Id", "SMILES"]
                + [f"pred_class_{task}" for task in tasks]
                + list(tasks)
                + ["top_tokens"]
            )
            batch_predictions_df = batch_predictions_df[ordered_cols]

            # Append to global predictions
            predictions_df = pd.concat(
                [predictions_df, batch_predictions_df], ignore_index=True
            )

    # cleanup temp
    try:
        if temp_dir is None:
            shutil.rmtree(temp_features_dir)
            logger.info(f"Cleaned up temporary directory: {temp_features_dir}")
    except Exception as e:
        logger.warning(f"Could not remove temp features dir: {e}")

    return predictions_df


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
        predictions_df.to_csv(args.output, index=False)
        logger.info(f"Predictions successfully saved to {args.output}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
