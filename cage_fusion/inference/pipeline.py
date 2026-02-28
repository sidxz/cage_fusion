"""
cage_fusion/inference/pipeline.py
===================================
High-level inference pipeline for CAGEFusion models.

Three public entry points
-------------------------

:class:`CageFusionPipeline`
    Long-lived service object.  Load once, call repeatedly.
    Accepts a SMILES string, a list of SMILES, or a DataFrame.

:func:`predict_smiles`
    Stateless convenience wrapper — loads and discards the model.

:func:`predict_and_explain`
    Gradient-saliency explanation for a single SMILES.
    Delegates to :mod:`cage_fusion.inference.explainer`.

Quick start
-----------
**Single SMILES string**::

    from cage_fusion import CageFusionPipeline

    pipe   = CageFusionPipeline.from_pretrained("checkpoints/my_run")
    result = pipe("CC(=O)Oc1ccccc1C(=O)O")
    # -> {"SMILES": "CC...", "PAINS_A": 0.12, "pred_class_PAINS_A": 0, ...}

**List of SMILES**::

    results = pipe(["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"])
    # -> list of dicts

**DataFrame (full batch)**::

    import pandas as pd
    df  = pd.DataFrame({"SMILES": ["CC(=O)Oc1ccccc1C(=O)O"]})
    out = pipe(df)   # -> pd.DataFrame

**Stateless**::

    from cage_fusion import predict_smiles
    df_out = predict_smiles(df, checkpoint_dir="checkpoints/my_run")
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import tempfile
from functools import partial
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cage_fusion.configuration import CageFusionConfig
from cage_fusion.data import CageFusionStreamingDataset, collate_cage_fusion
from cage_fusion.featurization import featurize_and_save_streaming
from cage_fusion.modeling import CAGEFusionForMultiLabelClassification
from cage_fusion.utils.device_utils import move_bmg_to_device
from cage_fusion.utils.hf_loader import load_hf_checkpoint, _resolve_pretrained_path
from cage_fusion.visualization import (
    visualize_combined_atom_contribution,
    visualize_fg_attention,
    visualize_top_token_attentions,
    visualize_total_atom_contribution,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _worker_init(_):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def _b64_image(path: str) -> Optional[str]:
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None


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
    highlight_red: bool = True,
    highlight_blue: bool = False,
) -> List[str]:
    """Render all attention visualisations for one sample; return top-token strings."""
    if g2t_weights is None or t2a_weights is None:
        return []

    sample_plot_dir = os.path.join(attn_plot_dir, f"idx_{original_idx}")
    os.makedirs(sample_plot_dir, exist_ok=True)

    smiles_to_plot = smiles_batch[j]

    # Top tokens from graph→token weights
    token_scores = g2t_weights[j].sum(dim=(0, 1))
    special_ids = [
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
    ]
    for sid in special_ids:
        if sid is not None:
            token_scores[input_ids[j] == sid] = -1e9

    top_token_indices = torch.argsort(token_scores, descending=True)[:3]
    full_ids = input_ids[j].detach().cpu().numpy()
    actual_ids = full_ids[full_ids != tokenizer.pad_token_id]
    full_tokens = tokenizer.convert_ids_to_tokens(actual_ids)

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

    kept = [
        tok
        for idx, tok in enumerate(full_tokens)
        if idx in set(top_token_indices.detach().cpu().numpy().tolist())
    ]
    return kept


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------


class CageFusionPipeline:
    """
    Long-lived inference pipeline.  Load the model once, call it many times.

    The pipeline accepts three input formats via :py:meth:`__call__`:

    - A **single SMILES string** → ``dict``
    - A **list of SMILES strings** → ``list[dict]``
    - A **DataFrame** with a ``SMILES`` column → ``pd.DataFrame``

    Construction
    ------------
    Prefer :py:meth:`from_pretrained` over calling ``__init__`` directly::

        pipe = CageFusionPipeline.from_pretrained("checkpoints/my_run")

    Calling
    -------
    ::

        # single string
        out = pipe("CC(=O)Oc1ccccc1C(=O)O")

        # list
        out = pipe(["SMILES1", "SMILES2"])

        # DataFrame
        import pandas as pd
        df  = pd.DataFrame({"SMILES": ["CC(=O)Oc1ccccc1C(=O)O"]})
        out = pipe(df)

    Args:
        checkpoint_dir: Path to a directory containing ``best_model.pt``,
            ``aux_features_scaler.pkl``, and optionally ``config.json``.
        model_file_name: Filename of the PyTorch weights file.
        device: ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
    """

    def __init__(
        self,
        checkpoint_dir: str,
        model_file_name: str = "best_model.pt",
        device: Optional[str] = None,
    ) -> None:
        checkpoint_dir = _resolve_pretrained_path(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        self.model_file_name = model_file_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        best_model_path = os.path.join(checkpoint_dir, model_file_name)
        scaler_path = os.path.join(checkpoint_dir, "aux_features_scaler.pkl")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(
                f"Weights file '{model_file_name}' not found in '{checkpoint_dir}'."
            )
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"'aux_features_scaler.pkl' not found in '{checkpoint_dir}'."
            )

        ckpt = torch.load(best_model_path, map_location=self.device, weights_only=False)

        # Support both old dict-style config and new CageFusionConfig
        raw_config = ckpt["config"]
        if isinstance(raw_config, dict):
            self.config = CageFusionConfig.from_dict(raw_config)
            hf_ckpt = raw_config.get("model_checkpoint", self.config.model_checkpoint)
            _tasks_raw = raw_config.get("tasks") or raw_config.get("label_names") or []
        else:
            self.config = raw_config
            hf_ckpt = self.config.model_checkpoint
            _tasks_raw = getattr(self.config, "label_names", None) or []

        self.tasks: List[str] = list(_tasks_raw)
        self.best_thresholds: np.ndarray = ckpt.get(
            "best_thresholds", np.full(max(len(self.tasks), 1), 0.5)
        )

        # Model
        self.model = CAGEFusionForMultiLabelClassification(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.eval()

        # Tokenizer + embedding model
        self.tokenizer, self.embedding_model = load_hf_checkpoint(hf_ckpt)
        self.embedding_model = self.embedding_model.to(self.device).eval()

        # Aux-features scaler
        self.scaler = joblib.load(scaler_path)
        if self.scaler is None or not hasattr(self.scaler, "mean_"):
            raise ValueError("Loaded scaler is not fitted.")

        logger.info(
            "CageFusionPipeline ready on %s — tasks: %s", self.device, self.tasks
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        model_file_name: str = "best_model.pt",
        device: Optional[str] = None,
    ) -> "CageFusionPipeline":
        """
        Load a pipeline from a saved checkpoint directory.

        This is the recommended way to construct a pipeline::

            pipe = CageFusionPipeline.from_pretrained("checkpoints/my_run")

        Args:
            checkpoint_dir: Local path containing ``best_model.pt``,
                ``aux_features_scaler.pkl``, and (optionally) ``config.json``.
            model_file_name: Name of the PyTorch weights file.
            device: ``"cpu"`` or ``"cuda"``; auto-detected if *None*.

        Returns:
            Initialised :class:`CageFusionPipeline`.
        """
        return cls(
            checkpoint_dir=checkpoint_dir,
            model_file_name=model_file_name,
            device=device,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        inputs: Union[str, List[str], pd.DataFrame],
        batch_size: int = 256,
        **kwargs,
    ) -> Union[Dict, List[Dict], pd.DataFrame]:
        """
        Run inference on one or more SMILES.

        Args:
            inputs: A single SMILES string, a list of SMILES strings, or a
                DataFrame containing a ``SMILES`` column.
            batch_size: Batch size for featurisation and forward passes.
            **kwargs: Extra keyword arguments forwarded to :py:meth:`predict`
                (e.g. ``plot_all_attention``, ``attn_plot_dir``).

        Returns:
            - ``str`` input  → ``dict`` (one result row)
            - ``list`` input → ``list[dict]``
            - ``DataFrame`` input → ``pd.DataFrame``

        Raises:
            TypeError: If *inputs* is not a recognised type.

        Example::

            pipe = CageFusionPipeline.from_pretrained("checkpoints/my_run")

            pipe("CC(=O)Oc1ccccc1C(=O)O")         # -> dict
            pipe(["SMILES1", "SMILES2"])            # -> list[dict]
            pipe(pd.DataFrame({"SMILES": [...]}))   # -> DataFrame
        """
        if isinstance(inputs, str):
            df = pd.DataFrame([{"SMILES": inputs}])
            result = self.predict(df, batch_size=batch_size, **kwargs)
            return result.iloc[0].to_dict()

        if isinstance(inputs, list):
            df = pd.DataFrame([{"SMILES": s} for s in inputs])
            result = self.predict(df, batch_size=batch_size, **kwargs)
            return result.to_dict(orient="records")

        if isinstance(inputs, pd.DataFrame):
            return self.predict(inputs, batch_size=batch_size, **kwargs)

        raise TypeError(
            f"Expected str, List[str], or pd.DataFrame, got {type(inputs).__name__}."
        )

    # ------------------------------------------------------------------
    # Full DataFrame batch inference
    # ------------------------------------------------------------------

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
        """
        Run batch inference on a DataFrame with a ``SMILES`` column.

        Args:
            input_df: DataFrame containing at minimum a ``SMILES`` column.
            batch_size: Featurisation and inference batch size.
            plot_all_attention: Generate attention plots for every molecule.
            attn_plot_dir: Required when ``plot_all_attention=True``.
            temp_dir: Directory for temporary HDF5 features (auto-cleaned if
                *None*).
            highlight_red: Highlight positive atom contributions in plots.
            highlight_blue: Highlight negative atom contributions in plots.

        Returns:
            DataFrame with columns: ``Original Index``, ``Id``, ``SMILES``,
            ``pred_class_<task>``, ``<task_prob>``, ``top_tokens``; and
            optionally base64-encoded attention images.
        """
        if "SMILES" not in input_df.columns:
            raise ValueError("Input DataFrame must contain a 'SMILES' column.")
        if plot_all_attention and not attn_plot_dir:
            raise ValueError(
                "'attn_plot_dir' must be provided when 'plot_all_attention=True'."
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

            collate_fn = partial(
                collate_cage_fusion, pad_token_id=self.tokenizer.pad_token_id
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
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
                worker_init_fn=_worker_init,
            )

            predictions_df = pd.DataFrame()
            if plot_all_attention and attn_plot_dir:
                os.makedirs(attn_plot_dir, exist_ok=True)

            for batch in tqdm(loader, desc="Predicting", disable=not plot_all_attention):
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

                out = self.model(
                    bmg=bmg,
                    sequence_embeddings=token_embs,
                    attn_mask=attn_mask,
                    aux_feats=aux_feats,
                    input_ids_batch=input_ids,
                    smiles_batch=smiles_batch,
                    return_attn=plot_all_attention,
                )

                logits = out.logits
                g2t_weights = out.graph_to_token_weights
                t2a_weights = out.token_to_graph_weights
                prompt_attn_weights = out.prompt_attn_weights

                probs = torch.sigmoid(logits).detach().cpu().numpy()

                batch_df = pd.DataFrame(
                    {
                        "Original Index": original_indices_batch.detach().cpu().numpy(),
                        "Id": ids_list,
                        "SMILES": smiles_batch,
                    }
                )
                for i, task in enumerate(self.tasks):
                    batch_df[f"pred_class_{task}"] = (
                        probs[:, i] > self.best_thresholds[i]
                    ).astype(int)
                    batch_df[task] = probs[:, i]

                if plot_all_attention and attn_plot_dir:
                    weight_fg = (
                        float(
                            self.model.encoder.fg_prompter.alpha.detach().cpu().item()
                        )
                        if hasattr(self.model.encoder, "fg_prompter")
                        and self.model.encoder.fg_prompter is not None
                        else None
                    )
                    weight_t2a = (
                        float(self.model.encoder.scale_graph.detach().cpu().item())
                        if hasattr(self.model.encoder, "scale_graph")
                        else None
                    )
                    batch_top_tokens = []
                    for j in range(len(smiles_batch)):
                        kept = _plot_batch_attentions(
                            j=j,
                            tokenizer=self.tokenizer,
                            logits=logits,
                            g2t_weights=g2t_weights,
                            t2a_weights=t2a_weights,
                            input_ids=input_ids,
                            smiles_batch=smiles_batch,
                            prompt_attn_weights=prompt_attn_weights,
                            attn_plot_dir=attn_plot_dir,
                            original_idx=int(original_indices_batch[j].item()),
                            weight_fg=weight_fg,
                            weight_t2a=weight_t2a,
                            highlight_red=highlight_red,
                            highlight_blue=highlight_blue,
                        )
                        batch_top_tokens.append("|".join(kept) if kept else "")
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

            # Attach base64 images if attention plots were generated
            if plot_all_attention and attn_plot_dir:
                predictions_df["atom_total_contrib_base64"] = ""
                predictions_df["overall_contrib_base64"] = ""
                predictions_df["prompt_atn_image_base64"] = ""

                for i, row in predictions_df.iterrows():
                    idx = int(row["Original Index"])
                    sample_dir = os.path.join(attn_plot_dir, f"idx_{idx}")
                    predictions_df.at[i, "atom_total_contrib_base64"] = (
                        _b64_image(os.path.join(sample_dir, "atom_total_contrib.png"))
                        or ""
                    )
                    predictions_df.at[i, "overall_contrib_base64"] = (
                        _b64_image(os.path.join(sample_dir, "atom_combined_contrib.png"))
                        or ""
                    )
                    predictions_df.at[i, "prompt_atn_image_base64"] = (
                        _b64_image(os.path.join(sample_dir, "fg_prompt_attention.png"))
                        or ""
                    )

            return predictions_df

        finally:
            if temp_dir is None:
                try:
                    shutil.rmtree(temp_features_dir)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Stateless convenience wrapper
# ---------------------------------------------------------------------------


def predict_smiles(
    input_df: pd.DataFrame,
    checkpoint_dir: str,
    batch_size: int = 256,
    temp_dir: Optional[str] = None,
    plot_all_attention: bool = False,
    attn_plot_dir: Optional[str] = None,
    model_file_name: str = "best_model.pt",
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function: load model, run inference, return results.

    Prefer :class:`CageFusionPipeline` for repeated inference to avoid
    reloading weights each time.

    Args:
        input_df: DataFrame with a ``SMILES`` column.
        checkpoint_dir: Path to checkpoint directory.
        batch_size: Featurisation and inference batch size.
        temp_dir: Temporary directory for HDF5 features.
        plot_all_attention: Generate attention plots for every molecule.
        attn_plot_dir: Output directory for attention plots.
        model_file_name: Filename of the weights file.
        device: ``"cpu"`` or ``"cuda"``; auto-detected if *None*.

    Returns:
        Prediction DataFrame (same as :py:meth:`CageFusionPipeline.predict`).
    """
    pipe = CageFusionPipeline(
        checkpoint_dir=checkpoint_dir,
        model_file_name=model_file_name,
        device=device,
    )
    return pipe.predict(
        input_df=input_df,
        batch_size=batch_size,
        temp_dir=temp_dir,
        plot_all_attention=plot_all_attention,
        attn_plot_dir=attn_plot_dir,
    )


# ---------------------------------------------------------------------------
# Gradient-saliency explanation (backwards-compat shim)
# ---------------------------------------------------------------------------


def predict_and_explain(
    smiles_string: str,
    checkpoint_dir: str,
    target_task: str,
    model_file_name: str = "best_model.pt",
    device: Optional[str] = None,
) -> Dict:
    """
    Run a single prediction and compute gradient-based saliency.

    .. deprecated::
        Use :class:`~cage_fusion.inference.explainer.GradientExplainer` or
        :func:`~cage_fusion.inference.explainer.explain_smiles` instead.

    Args:
        smiles_string: Input SMILES.
        checkpoint_dir: Path to checkpoint directory.
        target_task: Name of the task to explain.
        model_file_name: Filename of the weights file.
        device: ``"cpu"`` or ``"cuda"``; auto-detected if *None*.

    Returns:
        Dict with keys ``smiles``, ``task``, ``probability``,
        ``predicted_class``, ``threshold``, ``tokens``,
        ``token_saliency``, ``aux_saliency``.
    """
    from cage_fusion.inference.explainer import explain_smiles

    return explain_smiles(
        smiles_string,
        checkpoint_dir=checkpoint_dir,
        target_task=target_task,
        model_file_name=model_file_name,
        device=device,
    )
