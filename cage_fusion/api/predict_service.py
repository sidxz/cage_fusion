# cage_fusion/api/predict_service.py
import os

import torch
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel
import sys
from rich.console import Console
from rich.traceback import install
from transformers import AutoTokenizer, AutoModel
# Add project root to the Python path
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

install()
console = Console()


def _load_hf_artifacts(hf_id: str):
    """
    Prefer a pre-downloaded local snapshot (HF_RESOLVED_DIR) when present;
    fall back to online repo id otherwise.
    """
    local_dir = os.getenv("HF_RESOLVED_DIR")
    if local_dir and os.path.isdir(local_dir):
        kw = dict(local_files_only=True)
        src = local_dir
    else:
        kw = {}
        src = hf_id
    tok = AutoTokenizer.from_pretrained(src, **kw)
    emb = AutoModel.from_pretrained(src, **kw)
    return tok, emb


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
        hf_local = os.getenv("HF_RESOLVED_DIR")
        
        hf_ckpt = self.config["model_checkpoint"]
        self.tokenizer, self.embedding_model = load_hf_checkpoint(hf_ckpt)
        self.embedding_model = self.embedding_model.to(self.device).eval()

        # --- Scaler
        self.scaler = joblib.load(scaler_path)
        if self.scaler is None or not hasattr(self.scaler, "mean_"):
            raise ValueError("Failed to load a valid, fitted scaler.")

        # Ready flag
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
    ) -> pd.DataFrame:
        if plot_all_attention and not attn_plot_dir:
            raise ValueError(
                "'attn_plot_dir' must be provided if 'plot_all_attention' is True."
            )

        # --- Temp feature cache
        temp_features_dir = temp_dir or tempfile.mkdtemp()
        os.makedirs(temp_features_dir, exist_ok=True)
        try:
            # --- Featurize (no scaler fit; use loaded scaler)
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
                # class + score columns per task
                for i, task in enumerate(self.tasks):
                    batch_df[f"pred_class_{task}"] = (
                        probs[:, i] > self.best_thresholds[i]
                    ).astype(int)
                    batch_df[task] = probs[:, i]

                cols = (
                    ["Original Index", "Id", "SMILES"]
                    + [f"pred_class_{t}" for t in self.tasks]
                    + list(self.tasks)
                )
                predictions_df = pd.concat(
                    [predictions_df, batch_df[cols]], ignore_index=True
                )

            return predictions_df
        finally:
            if temp_dir is None:
                try:
                    shutil.rmtree(temp_features_dir)
                except Exception:
                    pass
