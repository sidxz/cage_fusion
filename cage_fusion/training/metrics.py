"""
In-memory metric accumulators for CAGEFusion training.

Replaces the disk-based aggregators with simple in-memory lists.
Molecular datasets are small enough that accumulating predictions in RAM
is faster and simpler than writing to disk.

Classes
-------
AUCAccumulator        – ROC-AUC per task          (classification)
PRAccumulator         – PR-AUC per task            (classification)
MCCAccumulator        – MCC with threshold search  (classification)
RegressionAccumulator – RMSE / MAE / R² per task  (regression)
MARAEAccumulator      – MA-RAE per task            (regression, OpenADMET leaderboard metric)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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
                if labels_np.size == 0 or len(np.unique(labels_np)) < 2:
                    auc = float("nan")
                else:
                    auc = roc_auc_score(labels_np, probs_np)
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
                if len(labels_np) == 0 or len(np.unique(labels_np)) < 2:
                    pr_aucs.append(float("nan"))
                else:
                    pr_aucs.append(average_precision_score(labels_np, probs_np))
            except Exception as e:
                logger.warning("PR-AUC failed for task %d: %s", i, e)
                pr_aucs.append(float("nan"))
        pr_np = np.array(pr_aucs)
        return float(np.nanmean(pr_np)) if reduce == "mean" else pr_np

    def reset(self):
        self._labels = [[] for _ in range(self.num_tasks)]
        self._probs = [[] for _ in range(self.num_tasks)]


# ─────────────────────────────────────────────────────────────────────────────
# Regression metrics: RMSE / MAE / R²
# ─────────────────────────────────────────────────────────────────────────────

class RegressionAccumulator:
    """
    Accumulates raw continuous predictions and targets; computes per-task
    RMSE, MAE, and R².  No sigmoid is applied — use directly with
    ``CAGEFusionForRegression`` logits.
    """

    def __init__(self, num_tasks: int, label_names=None):
        self.num_tasks = num_tasks
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        self._preds: List[list] = [[] for _ in range(num_tasks)]
        self._targets: List[list] = [[] for _ in range(num_tasks)]

    def update(self, targets_batch, preds_batch):
        targets = _to_numpy(targets_batch)
        preds = _to_numpy(preds_batch)
        for i in range(self.num_tasks):
            self._targets[i].extend(targets[:, i].tolist())
            self._preds[i].extend(preds[:, i].tolist())

    def compute(self, reduce="mean"):
        """
        Returns:
            ``reduce="mean"`` → ``(mean_rmse, mean_mae, mean_r2)`` floats.
            ``reduce="none"`` → ``(rmse_array, mae_array, r2_array)`` numpy arrays.
        """
        rmses, maes, r2s = [], [], []
        for i in range(self.num_tasks):
            t = np.array(self._targets[i])
            p = np.array(self._preds[i])
            if len(t) == 0:
                rmses.append(float("nan"))
                maes.append(float("nan"))
                r2s.append(float("nan"))
                continue
            try:
                rmse = float(mean_squared_error(t, p) ** 0.5)
                mae = float(mean_absolute_error(t, p))
                r2 = float(r2_score(t, p)) if len(t) > 1 else float("nan")
            except Exception as e:
                logger.warning("Regression metrics failed for task %d: %s", i, e)
                rmse, mae, r2 = float("nan"), float("nan"), float("nan")
            rmses.append(rmse)
            maes.append(mae)
            r2s.append(r2)

        rmse_np = np.array(rmses)
        mae_np = np.array(maes)
        r2_np = np.array(r2s)

        if reduce == "mean":
            return (
                float(np.nanmean(rmse_np)),
                float(np.nanmean(mae_np)),
                float(np.nanmean(r2_np)),
            )
        return rmse_np, mae_np, r2_np

    def reset(self):
        self._preds = [[] for _ in range(self.num_tasks)]
        self._targets = [[] for _ in range(self.num_tasks)]


# ─────────────────────────────────────────────────────────────────────────────
# MA-RAE  (OpenADMET leaderboard metric)
# ─────────────────────────────────────────────────────────────────────────────

class MARAEAccumulator:
    """
    Macro-Averaged Relative Absolute Error — matches the OpenADMET leaderboard
    scoring formula.

    Per-endpoint RAE::

        RAE_i = MAE_i / mean(|y_true_i − mean(y_true_i)|)

    MA-RAE is the mean of RAE_i across all tasks that have ≥ 2 valid (non-NaN)
    samples.  NaN targets are masked per-task, making this safe for sparse
    multi-task labels.

    Expected inputs
    ---------------
    Both ``targets_batch`` and ``preds_batch`` should be in the **same scale
    as training** (i.e. log-transformed if that was applied before training).
    Do *not* inverse-transform before computing MA-RAE.
    """

    def __init__(self, num_tasks: int, label_names=None):
        self.num_tasks  = num_tasks
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        self._preds:   List[list] = [[] for _ in range(num_tasks)]
        self._targets: List[list] = [[] for _ in range(num_tasks)]

    def update(self, targets_batch, preds_batch):
        """Accumulate one batch.  NaN targets are stored as NaN and masked at
        ``compute()`` time."""
        targets = _to_numpy(targets_batch)
        preds   = _to_numpy(preds_batch)
        for i in range(self.num_tasks):
            self._targets[i].extend(targets[:, i].tolist())
            self._preds[i].extend(preds[:, i].tolist())

    def compute(self, reduce: str = "mean"):
        """
        Compute MA-RAE.

        Args:
            reduce: ``"mean"`` → returns ``(ma_rae_float, per_task_rae_list)``.
                    ``"none"`` → returns only ``per_task_rae_list`` (np.ndarray,
                    NaN for tasks with < 2 valid samples).

        Returns:
            ``(ma_rae, per_task_raes)`` when ``reduce="mean"``, or
            ``per_task_raes`` array when ``reduce="none"``.
        """
        raes = []
        for i in range(self.num_tasks):
            t = np.array(self._targets[i], dtype=float)
            p = np.array(self._preds[i],   dtype=float)
            # mask NaN targets
            mask = ~np.isnan(t)
            t, p = t[mask], p[mask]
            if len(t) < 2:
                raes.append(float("nan"))
                continue
            try:
                mae        = float(mean_absolute_error(t, p))
                denominator = float(np.mean(np.abs(t - np.mean(t))))
                rae        = mae / denominator if denominator > 1e-12 else float("nan")
            except Exception as e:
                logger.warning("MA-RAE failed for task %d (%s): %s", i, self.label_names[i], e)
                rae = float("nan")
            raes.append(rae)

        raes_np = np.array(raes)
        if reduce == "none":
            return raes_np
        ma_rae = float(np.nanmean(raes_np))
        return ma_rae, raes

    def reset(self):
        self._preds   = [[] for _ in range(self.num_tasks)]
        self._targets = [[] for _ in range(self.num_tasks)]
