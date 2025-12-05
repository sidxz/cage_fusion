# cage_fusion/api/predict_service.py

import os
import base64
import sys
import torch
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel
from rich.console import Console
from rich.traceback import install

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cage_fusion.models import CAGEFusionModel
from cage_fusion.engine.dataset import CageFusionStreamingDataset
from cage_fusion.engine.data_utils import collate_fn_for_cage_fusion
from cage_fusion.engine.utils import move_bmg_to_device
from cage_fusion.featurizers import featurize_and_save_streaming
from cage_fusion.utils.logging_utils import logger
from cage_fusion.utils.hf_loader import load_hf_checkpoint

from functools import partial
import tempfile
import shutil
from tqdm import tqdm

from cage_fusion.viz.token_viz import (
    visualize_top_token_attentions,
    visualize_total_atom_contribution,
    visualize_combined_atom_contribution,
)
from cage_fusion.viz.prompt_viz import visualize_fg_attention

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

install()
console = Console()


def _b64_image(path: str) -> Optional[str]:
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None


def _plot_batch_attentions(
    j,
    tokenizer,
    logits,
    g2t_weights,
    t2a_weights,
    input_ids,
    smiles_batch,
    prompt_attn_weights,
    attn_plot_dir,
    original_idx,
    weight_fg=None,
    weight_t2a=None,
    highlight_red: bool = True,
    highlight_blue: bool = False,
):
    if g2t_weights is None or t2a_weights is None:
        return []

    sample_plot_dir = os.path.join(attn_plot_dir, f"idx_{original_idx}")
    os.makedirs(sample_plot_dir, exist_ok=True)

    smiles_to_plot = smiles_batch[j]
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

    # --- Plots ---
    visualize_top_token_attentions(
        smiles=smiles_to_plot,
        attention_weights=t2a_weights[j],
        full_token_list=full_tokens,
        top_token_indices=top_token_indices.detach().cpu().numpy(),
        output_dir=sample_plot_dir,
    )

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
        highlight_red=highlight_red,
        highlight_blue=highlight_blue,
    )

    if prompt_attn_weights and prompt_attn_weights[j] is not None:
        visualize_fg_attention(
            smiles=smiles_to_plot,
            prompt_attn_weights=prompt_attn_weights[j],
            output_path=os.path.join(sample_plot_dir, "fg_prompt_attention.png"),
            title="Functional Group Attention (PROMPT)",
            highlight_red=highlight_red,
            highlight_blue=highlight_blue,
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
                highlight_red=highlight_red,
                highlight_blue=highlight_blue,
            )

    # Collect top token strings
    kept = []
    top_idx_np = set(top_token_indices.detach().cpu().numpy().tolist())
    for idx, tok in enumerate(full_tokens):
        if idx in top_idx_np:
            kept.append(tok)
    return kept


class CAGEFusionPredictor:
    """
    Long-lived predictor that loads model/tokenizer/scaler once.
    Mirrors your CLI's predict_smiles() behavior, but avoids per-request reloads.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        model_file_name: str = "best_model.pt",
        device: Optional[str] = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.model_file_name = model_file_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        best_model_path = os.path.join(self.checkpoint_dir, self.model_file_name)
        scaler_path = os.path.join(self.checkpoint_dir, "aux_features_scaler.pkl")
        if not (os.path.exists(best_model_path) and os.path.exists(scaler_path)):
            raise FileNotFoundError(
                f"Missing '{self.model_file_name}' or 'aux_features_scaler.pkl' in {self.checkpoint_dir}"
            )

        # --- Load checkpoint & config
        ckpt = torch.load(best_model_path, map_location=self.device, weights_only=False)
        self.config = ckpt["config"]
        self.tasks = self.config["tasks"]
        self.best_thresholds = ckpt.get(
            "best_thresholds", np.full(len(self.tasks), 0.5)
        )

        # --- Build model and load weights
        self.model = CAGEFusionModel(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # --- Tokenizer & embedding model
        hf_ckpt = self.config["model_checkpoint"]
        self.tokenizer, self.embedding_model = load_hf_checkpoint(hf_ckpt)
        self.embedding_model = self.embedding_model.to(self.device).eval()

        # --- Scaler
        self.scaler = joblib.load(scaler_path)
        if self.scaler is None or not hasattr(self.scaler, "mean_"):
            raise ValueError("Failed to load a valid, fitted scaler.")

        self.ready = True
        logger.info(
            f"CAGEFusionPredictor initialized on {self.device} with tasks={self.tasks}"
        )

    @torch.inference_mode()
    def predict(
        self,
        input_df: pd.DataFrame,
        batch_size: int = 256,
        plot_all_attention: bool = False,
        attn_plot_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        highlight_red: bool = True,
        highlight_blue: bool = False,
    ) -> pd.DataFrame:
        if plot_all_attention and not attn_plot_dir:
            raise ValueError(
                "'attn_plot_dir' must be provided if 'plot_all_attention' is True."
            )

        temp_features_dir = temp_dir or tempfile.mkdtemp()
        os.makedirs(temp_features_dir, exist_ok=True)
        try:
            h5_path, _, _ = featurize_and_save_streaming(
                df=input_df,
                name="inference",
                label_cols=[],
                cache_dir=temp_features_dir,
                tokenizer=self.tokenizer,
                model=self.embedding_model,
                fit_scaler=False,
                scaler=self.scaler,
                batch_size=batch_size,
            )

            collate_with_pad = partial(
                collate_fn_for_cage_fusion, pad_token_id=self.tokenizer.pad_token_id
            )
            ds = CageFusionStreamingDataset(
                h5_path,
                tokenizer_pad_id=self.tokenizer.pad_token_id,
                prefer_normalized_aux=True,
                return_ids=True,
                total_num_workers=0,
                graph_cache="auto",
                single_worker_graph_cache=True,
                emb_cache_store_dtype=np.float32,
                return_emb_dtype=torch.float32,
            )
            loader = torch.utils.data.DataLoader(
                ds,
                batch_size=min(batch_size, self.config.get("batch_size", batch_size)),
                shuffle=False,
                num_workers=0,
                collate_fn=collate_with_pad,
            )

            predictions_df = pd.DataFrame()
            if plot_all_attention and attn_plot_dir:
                os.makedirs(attn_plot_dir, exist_ok=True)

            all_top_tokens = []
            for batch in tqdm(loader, desc="Predicting", disable=True):
                if batch is None:
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

                bmg = move_bmg_to_device(bmg, self.device)
                token_embs, attn_mask, aux_feats, input_ids = [
                    t.to(self.device)
                    for t in [token_embs, attn_mask, aux_feats, input_ids]
                ]

                model_output = self.model(
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
                probs = torch.sigmoid(logits).detach().cpu().numpy()

                batch_df = pd.DataFrame(
                    {
                        "Original Index": original_indices_batch.detach().cpu().numpy(),
                        "Id": ids_list,
                        "SMILES": smiles_batch,
                    }
                )

                # Per-task class/score
                for i, task in enumerate(self.tasks):
                    batch_df[f"pred_class_{task}"] = (
                        probs[:, i] > self.best_thresholds[i]
                    ).astype(int)
                    batch_df[task] = probs[:, i]

                # Plot attention and collect top_tokens per sample
                if plot_all_attention and attn_plot_dir:
                    weight_fg = (
                        float(self.model.alpha.detach().cpu().item())
                        if hasattr(self.model, "alpha")
                        else None
                    )
                    weight_t2a = (
                        float(self.model.scale_graph.detach().cpu().item())
                        if hasattr(self.model, "scale_graph")
                        else None
                    )
                    batch_top_tokens = []
                    for j in range(len(smiles_batch)):
                        original_idx = int(original_indices_batch[j].item())
                        kept_tokens = _plot_batch_attentions(
                            j=j,
                            tokenizer=self.tokenizer,
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
                            highlight_red=highlight_red,
                            highlight_blue=highlight_blue,
                        )
                        batch_top_tokens.append(
                            "|".join(kept_tokens) if kept_tokens else ""
                        )
                    batch_df["top_tokens"] = batch_top_tokens
                else:
                    batch_df["top_tokens"] = [""] * len(smiles_batch)

                cols = (
                    ["Original Index", "Id", "SMILES"]
                    + [f"pred_class_{t}" for t in self.tasks]
                    + list(self.tasks)
                    + ["top_tokens"]
                )
                predictions_df = pd.concat(
                    [predictions_df, batch_df[cols]], ignore_index=True
                )

            # After all batches: attach base64-encoded images
            if plot_all_attention and attn_plot_dir:
                predictions_df["atom_total_contrib_base64"] = ""
                predictions_df["overall_contrib_base64"] = ""
                predictions_df["prompt_atn_image_base64"] = ""
                predictions_df["attention_summary_image_base64"] = ""

                for i, row in predictions_df.iterrows():
                    idx = int(row["Original Index"])
                    sample_dir = os.path.join(attn_plot_dir, f"idx_{idx}")

                    atom_total = _b64_image(
                        os.path.join(sample_dir, "atom_total_contrib.png")
                    )
                    combined = _b64_image(
                        os.path.join(sample_dir, "atom_combined_contrib.png")
                    )
                    prompt_attn = _b64_image(
                        os.path.join(sample_dir, "fg_prompt_attention.png")
                    )

                    predictions_df.at[i, "atom_total_contrib_base64"] = atom_total or ""
                    predictions_df.at[i, "overall_contrib_base64"] = combined or ""
                    predictions_df.at[i, "prompt_atn_image_base64"] = prompt_attn or ""
                    predictions_df.at[i, "attention_summary_image_base64"] = (
                        combined or ""
                    )

            return predictions_df
        finally:
            if temp_dir is None:
                try:
                    shutil.rmtree(temp_features_dir)
                except Exception:
                    pass
