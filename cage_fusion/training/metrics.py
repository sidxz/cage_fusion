"""
In-memory metric accumulators for CAGEFusion training.

Replaces the disk-based aggregators with simple in-memory lists.
Molecular datasets are small enough that accumulating predictions in RAM
is faster and simpler than writing to disk.

Classes
-------
AUCAccumulator  – ROC-AUC per task
PRAccumulator   – PR-AUC per task
MCCAccumulator  – MCC with automatic threshold search per task
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)

logger = logging.getLogger("cagefusion")


def _to_numpy(data):
    return data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)


# ─────────────────────────────────────────────────────────────────────────────
# ROC-AUC
# ─────────────────────────────────────────────────────────────────────────────

class AUCAccumulator:
    def __init__(self, num_tasks: int, label_names=None):
        self.num_tasks = num_tasks
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        self._labels: List[list] = [[] for _ in range(num_tasks)]
        self._probs: List[list] = [[] for _ in range(num_tasks)]

    def update(self, labels_batch, probs_batch):
        labels = _to_numpy(labels_batch)
        probs = _to_numpy(probs_batch)
        for i in range(self.num_tasks):
            self._labels[i].extend(labels[:, i].tolist())
            self._probs[i].extend(probs[:, i].tolist())

    def compute(self, reduce="mean"):
        aucs = []
        for i in range(self.num_tasks):
            labels_np = np.array(self._labels[i])
            probs_np = np.array(self._probs[i])
            try:
                auc = roc_auc_score(labels_np, probs_np) if labels_np.size else float("nan")
            except Exception as e:
                logger.warning("AUC failed for task %d: %s", i, e)
                auc = float("nan")
            aucs.append(auc)
        aucs_np = np.array(aucs)
        return float(np.nanmean(aucs_np)) if reduce == "mean" else aucs_np

    def reset(self):
        self._labels = [[] for _ in range(self.num_tasks)]
        self._probs = [[] for _ in range(self.num_tasks)]


# ─────────────────────────────────────────────────────────────────────────────
# MCC with threshold search
# ─────────────────────────────────────────────────────────────────────────────

class MCCAccumulator:
    def __init__(self, num_tasks: int, label_names=None):
        self.num_tasks = num_tasks
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        self._labels: List[list] = [[] for _ in range(num_tasks)]
        self._probs: List[list] = [[] for _ in range(num_tasks)]

    def update(self, labels_batch, probs_batch):
        labels = _to_numpy(labels_batch)
        probs = _to_numpy(probs_batch)
        for i in range(self.num_tasks):
            self._labels[i].extend(labels[:, i].tolist())
            self._probs[i].extend(probs[:, i].tolist())

    def compute(
        self,
        threshold_search: np.ndarray = np.linspace(0.1, 0.9, 20),
        thresholds: Optional[np.ndarray] = None,
    ):
        """Returns (mean_mcc, best_thresholds, per_task_mccs)."""
        if thresholds is not None and len(thresholds) != self.num_tasks:
            logger.warning("Threshold count mismatch – ignoring provided thresholds.")
            thresholds = None

        mccs, best_thresholds = [], []

        for i in range(self.num_tasks):
            labels_np = np.array(self._labels[i])
            probs_np = np.array(self._probs[i])

            if len(labels_np) == 0:
                mccs.append(0.0)
                best_thresholds.append(thresholds[i] if thresholds is not None else 0.5)
                continue

            try:
                if thresholds is not None:
                    bin_preds = (probs_np > thresholds[i]).astype(int)
                    mcc = matthews_corrcoef(labels_np, bin_preds)
                    best_thresholds.append(float(thresholds[i]))
                else:
                    mcc, best_t = -1.0, 0.5
                    for t in threshold_search:
                        try:
                            score = matthews_corrcoef(labels_np, (probs_np > t).astype(int))
                            if score > mcc:
                                mcc, best_t = score, float(t)
                        except Exception:
                            continue
                    best_thresholds.append(best_t)
            except Exception as e:
                logger.warning("MCC failed for task %d: %s", i, e)
                mcc = 0.0
                best_thresholds.append(0.5)

            mccs.append(mcc)

        return float(np.mean(mccs)), best_thresholds, mccs

    def reset(self):
        self._labels = [[] for _ in range(self.num_tasks)]
        self._probs = [[] for _ in range(self.num_tasks)]


# ─────────────────────────────────────────────────────────────────────────────
# PR-AUC
# ─────────────────────────────────────────────────────────────────────────────

class PRAccumulator:
    def __init__(self, num_tasks: int, label_names=None):
        self.num_tasks = num_tasks
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        self._labels: List[list] = [[] for _ in range(num_tasks)]
        self._probs: List[list] = [[] for _ in range(num_tasks)]

    def update(self, labels_batch, probs_batch):
        labels = _to_numpy(labels_batch)
        probs = _to_numpy(probs_batch)
        for i in range(self.num_tasks):
            self._labels[i].extend(labels[:, i].tolist())
            self._probs[i].extend(probs[:, i].tolist())

    def compute(self, reduce="mean"):
        pr_aucs = []
        for i in range(self.num_tasks):
            labels_np = np.array(self._labels[i])
            probs_np = np.array(self._probs[i])
            try:
                pr_aucs.append(
                    average_precision_score(labels_np, probs_np) if len(labels_np) > 0 else 0.0
                )
            except Exception as e:
                logger.warning("PR-AUC failed for task %d: %s", i, e)
                pr_aucs.append(0.0)
        pr_np = np.array(pr_aucs)
        return float(np.mean(pr_np)) if reduce == "mean" else pr_np

    def reset(self):
        self._labels = [[] for _ in range(self.num_tasks)]
        self._probs = [[] for _ in range(self.num_tasks)]
