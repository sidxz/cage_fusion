"""
Stand-alone evaluator.

``evaluate_model()`` is a thin wrapper around ``Trainer.evaluate()`` that
can be called without a full Trainer instance — useful for one-off
evaluation scripts, benchmarks, or the inference pipeline.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch

from cage_fusion.training.metrics import (
    AUCAccumulator,
    MCCAccumulator,
    PRAccumulator,
)
from cage_fusion.utils.device_utils import move_bmg_to_device
from tqdm import tqdm

logger = logging.getLogger("cagefusion")


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device: torch.device,
    label_names: Optional[List[str]] = None,
    threshold_search: np.ndarray = np.linspace(0.1, 0.9, 20),
    use_precomputed_thresholds: Optional[np.ndarray] = None,
    criterion: Optional[torch.nn.Module] = None,
) -> dict:
    """
    Evaluate *model* on *loader* and return a metrics dict.

    Parameters
    ----------
    model:
        A ``CAGEFusionForMultiLabelClassification`` or compatible model.
    loader:
        DataLoader yielding batches in the standard cage_fusion format.
    device:
        Device to run inference on.
    label_names:
        Optional task names for per-task logging.
    threshold_search:
        Grid of thresholds for MCC optimisation.
    use_precomputed_thresholds:
        If provided, skip threshold search and use these directly.
    criterion:
        Optional external loss function.  If ``None`` and the model
        returns a ``loss`` field, that is used.

    Returns
    -------
    dict with keys:
        ``loss``, ``mcc``, ``auc``, ``pr``,
        ``best_thresholds``, ``per_task``,
        ``norm_graph``, ``norm_attn``, ``norm_aux``
    """
    model.eval()
    num_tasks = getattr(getattr(model, "config", None), "num_labels", 1)

    mcc_agg = MCCAccumulator(num_tasks, label_names)
    auc_agg = AUCAccumulator(num_tasks, label_names)
    pr_agg = PRAccumulator(num_tasks, label_names)

    total_loss = 0.0
    total_graph_norm = total_attn_norm = total_aux_norm = 0.0

    for batch in tqdm(loader, desc="Evaluating"):
        if batch is None:
            continue
        bmg, token_embs, attn_mask, aux_feats, labels, input_ids, smiles_batch, _, _ = batch
        bmg = move_bmg_to_device(bmg, device)
        token_embs = token_embs.to(device)
        attn_mask = attn_mask.to(device)
        aux_feats = aux_feats.to(device)
        labels = labels.to(device)
        input_ids = input_ids.to(device)

        output = model(
            bmg=bmg,
            sequence_embeddings=token_embs,
            attn_mask=attn_mask,
            aux_feats=aux_feats,
            input_ids_batch=input_ids,
            smiles_batch=smiles_batch,
            labels=labels,
            return_attn=False,
        )

        if output.loss is not None:
            total_loss += output.loss.item()
        elif criterion is not None:
            total_loss += criterion(output.logits, labels).item()

        if output.graph_repr is not None:
            total_graph_norm += output.graph_repr.norm(dim=1).mean().item()
        if output.attn_output is not None:
            total_attn_norm += output.attn_output.norm(dim=1).mean().item()
        total_aux_norm += aux_feats.norm(dim=1).mean().item()

        probs = torch.sigmoid(output.logits)
        mcc_agg.update(labels, probs)
        auc_agg.update(labels, probs)
        pr_agg.update(labels, probs)

    n = max(1, len(loader))
    per_task_auc = auc_agg.compute(reduce="none")
    per_task_pr = pr_agg.compute(reduce="none")
    avg_auc = float(np.nanmean(per_task_auc)) if len(per_task_auc) > 0 else 0.0
    avg_pr = float(np.mean(per_task_pr)) if len(per_task_pr) > 0 else 0.0

    if use_precomputed_thresholds is not None:
        avg_mcc, _, per_task_mcc = mcc_agg.compute(thresholds=use_precomputed_thresholds)
        best_thresholds = use_precomputed_thresholds
    else:
        avg_mcc, best_thresholds, per_task_mcc = mcc_agg.compute(threshold_search=threshold_search)

    return {
        "loss": total_loss / n,
        "mcc": avg_mcc,
        "auc": avg_auc,
        "pr": avg_pr,
        "best_thresholds": best_thresholds,
        "per_task": list(zip(per_task_mcc, per_task_auc, per_task_pr)),
        "norm_graph": total_graph_norm / n,
        "norm_attn": total_attn_norm / n,
        "norm_aux": total_aux_norm / n,
    }
