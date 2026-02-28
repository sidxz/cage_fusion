"""
cage_fusion/inference/explainer.py
===================================
Gradient-saliency explainer for CAGEFusion models.

:class:`GradientExplainer` computes input-gradient norms to identify
which SMILES tokens and auxiliary (RDKit) features most influence a
given prediction.

Quick start
-----------
**Attached to a long-lived pipeline** (recommended — model loaded once)::

    from cage_fusion import CageFusionPipeline
    from cage_fusion.inference.explainer import GradientExplainer

    pipe   = CageFusionPipeline.from_pretrained("checkpoints/my_run")
    explainer = GradientExplainer(pipe)

    result = explainer.explain("CC(=O)Oc1ccccc1C(=O)O", target_task="PAINS_A")
    for tok, sal in zip(result["tokens"], result["token_saliency"]):
        print(f"{tok:15s}  {sal:.4f}")

**Stateless one-shot** (convenience wrapper)::

    from cage_fusion.inference.explainer import explain_smiles

    result = explain_smiles(
        "CC(=O)Oc1ccccc1C(=O)O",
        checkpoint_dir="checkpoints/my_run",
        target_task="PAINS_A",
    )
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from functools import partial
from typing import Dict, List, Optional

import torch

from cage_fusion.data import CageFusionStreamingDataset, collate_cage_fusion
from cage_fusion.featurization import featurize_and_save_streaming
from cage_fusion.utils.device_utils import move_bmg_to_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class GradientExplainer:
    """
    Compute input-gradient saliency for a loaded CAGEFusion pipeline.

    Attach this to a :class:`~cage_fusion.inference.pipeline.CageFusionPipeline`
    to explain predictions without reloading the model.

    Args:
        pipeline: A fully initialised
            :class:`~cage_fusion.inference.pipeline.CageFusionPipeline`.

    Example::

        from cage_fusion import CageFusionPipeline
        from cage_fusion.inference.explainer import GradientExplainer

        pipe = CageFusionPipeline.from_pretrained("checkpoints/my_run")
        exp  = GradientExplainer(pipe)
        result = exp.explain("CC(=O)Oc1ccccc1C(=O)O", target_task="PAINS_A")
    """

    def __init__(self, pipeline) -> None:
        self._pipe = pipeline

    # ------------------------------------------------------------------

    def explain(
        self,
        smiles: str,
        *,
        target_task: Optional[str] = None,
        target_idx: Optional[int] = None,
    ) -> Dict:
        """
        Compute gradient-based saliency for a single SMILES.

        Exactly one of *target_task* or *target_idx* must be provided.

        Args:
            smiles: Input SMILES string.
            target_task: Human-readable task name; must match a name in
                ``pipeline.tasks``.
            target_idx: Zero-based task index (alternative to *target_task*).

        Returns:
            Dict with keys:

            - ``"smiles"`` — input SMILES
            - ``"task"`` — resolved task name
            - ``"task_idx"`` — zero-based task index
            - ``"probability"`` — sigmoid probability for this task
            - ``"predicted_class"`` — 0 or 1 (threshold applied)
            - ``"threshold"`` — decision threshold used
            - ``"tokens"`` — list of SMILES tokens (special tokens excluded)
            - ``"token_saliency"`` — gradient-norm per token (same length)
            - ``"aux_saliency"`` — gradient vector over auxiliary features

        Raises:
            ValueError: If neither / both *target_task*/*target_idx* are given,
                or if the task name is not found.
        """
        import pandas as pd

        pipe = self._pipe
        tasks: List[str] = pipe.tasks or []
        idx = _resolve_task(tasks, target_task, target_idx)
        task_name = tasks[idx] if tasks else f"task_{idx}"
        threshold = (
            float(pipe.best_thresholds[idx])
            if pipe.best_thresholds is not None
            else 0.5
        )

        input_df = pd.DataFrame([{"SMILES": smiles}])
        tmp = tempfile.mkdtemp()
        try:
            h5_path, _, _ = featurize_and_save_streaming(
                df=input_df,
                name="explain_temp",
                label_cols=[],
                cache_dir=tmp,
                tokenizer=pipe.tokenizer,
                model=pipe.embedding_model,
                fit_scaler=False,
                scaler=pipe.scaler,
            )
            collate_fn = partial(
                collate_cage_fusion, pad_token_id=pipe.tokenizer.pad_token_id
            )
            ds = CageFusionStreamingDataset(
                h5_path,
                tokenizer_pad_id=pipe.tokenizer.pad_token_id,
                prefer_normalized_aux=True,
                return_ids=False,
                total_num_workers=0,
            )
            loader = torch.utils.data.DataLoader(
                ds, batch_size=1, collate_fn=collate_fn, shuffle=False
            )
            batch = next(iter(loader))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        return _run_gradient_saliency(
            batch=batch,
            model=pipe.model,
            tokenizer=pipe.tokenizer,
            device=pipe.device,
            task_idx=idx,
            task_name=task_name,
            smiles=smiles,
            threshold=threshold,
        )


# ---------------------------------------------------------------------------
# Stateless convenience function
# ---------------------------------------------------------------------------


def explain_smiles(
    smiles: str,
    checkpoint_dir: str,
    target_task: Optional[str] = None,
    target_idx: Optional[int] = None,
    model_file_name: str = "best_model.pt",
    device: Optional[str] = None,
) -> Dict:
    """
    Load a checkpoint, explain a single SMILES prediction, then discard the model.

    For repeated explanations use :class:`GradientExplainer` attached to a
    long-lived :class:`~cage_fusion.inference.pipeline.CageFusionPipeline`
    to avoid reloading weights each time.

    Args:
        smiles: Input SMILES string.
        checkpoint_dir: Path to a checkpoint directory (contains ``config.json``
            + ``best_model.pt`` + ``aux_features_scaler.pkl``).
        target_task: Human-readable task name.
        target_idx: Zero-based task index (alternative to *target_task*).
        model_file_name: Weights filename inside *checkpoint_dir*.
        device: ``"cpu"`` or ``"cuda"``; auto-detected if *None*.

    Returns:
        Same dict as :py:meth:`GradientExplainer.explain`.
    """
    # Lazy import avoids circular dependency at module level
    from cage_fusion.inference.pipeline import CageFusionPipeline

    pipe = CageFusionPipeline(
        checkpoint_dir=checkpoint_dir,
        model_file_name=model_file_name,
        device=device,
    )
    return GradientExplainer(pipe).explain(
        smiles,
        target_task=target_task,
        target_idx=target_idx,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_task(
    tasks: List[str],
    target_task: Optional[str],
    target_idx: Optional[int],
) -> int:
    """Validate inputs and return the resolved zero-based task index."""
    if (target_task is None) == (target_idx is None):
        raise ValueError("Provide exactly one of 'target_task' or 'target_idx'.")

    if target_task is not None:
        if target_task not in tasks:
            raise ValueError(
                f"Task '{target_task}' not found. Available tasks: {tasks}"
            )
        return tasks.index(target_task)

    # target_idx path
    if not (0 <= target_idx < len(tasks)):  # type: ignore[operator]
        raise ValueError(
            f"target_idx={target_idx} is out of range for {len(tasks)} tasks."
        )
    return target_idx  # type: ignore[return-value]


def _run_gradient_saliency(
    *,
    batch,
    model,
    tokenizer,
    device: torch.device,
    task_idx: int,
    task_name: str,
    smiles: str,
    threshold: float,
) -> Dict:
    """
    Core gradient-saliency computation.

    Separated from the public API so it can be unit-tested independently.
    """
    (
        bmg,
        token_embs,
        attn_mask,
        aux_feats,
        _labels,
        input_ids,
        smiles_batch,
        _orig_idx,
        _ids_list,
    ) = batch

    bmg = move_bmg_to_device(bmg, device)
    token_embs = token_embs.to(device).detach().requires_grad_(True)
    attn_mask = attn_mask.to(device)
    aux_feats = aux_feats.to(device).detach().requires_grad_(True)
    input_ids = input_ids.to(device)

    model.zero_grad()
    out = model(
        bmg=bmg,
        sequence_embeddings=token_embs,
        attn_mask=attn_mask,
        aux_feats=aux_feats,
        input_ids_batch=input_ids,
        smiles_batch=smiles_batch,
        return_attn=False,
    )

    score = out.logits[0, task_idx]
    prob = float(torch.sigmoid(score).item())
    predicted_class = int(prob >= threshold)
    score.backward()

    token_saliency = token_embs.grad.norm(dim=-1).squeeze(0).cpu().numpy()
    aux_saliency = aux_feats.grad.squeeze(0).cpu().numpy()

    active_mask = (input_ids.squeeze(0) != tokenizer.pad_token_id).cpu()
    tokens = tokenizer.convert_ids_to_tokens(
        input_ids.squeeze(0)[active_mask].cpu().numpy()
    )
    token_saliency_clean = token_saliency[active_mask.numpy()]

    return {
        "smiles": smiles,
        "task": task_name,
        "task_idx": task_idx,
        "probability": prob,
        "predicted_class": predicted_class,
        "threshold": threshold,
        "tokens": tokens,
        "token_saliency": token_saliency_clean,
        "aux_saliency": aux_saliency,
    }
