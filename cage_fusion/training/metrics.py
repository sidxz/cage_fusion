"""
Streaming metric aggregators that write per-batch predictions to disk,
then compute metrics in a single pass at epoch end.

This avoids accumulating all predictions in RAM during long epochs.

Classes
-------
AUCBatchAggregatorToDisk  – ROC-AUC per task
MCCBatchAggregatorToDisk  – MCC with automatic threshold search per task
PRBatchAggregatorToDisk   – PR-AUC per task
"""

# Logic identical to the original engine/metrics.py;
# import updated from cage_fusion.utils.logging_utils → logging

from __future__ import annotations

import glob
import logging
import os
import pickle
import tempfile
import uuid
from pathlib import Path
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
    return data.cpu().numpy() if isinstance(data, torch.Tensor) else data


# ─────────────────────────────────────────────────────────────────────────────
# ROC-AUC
# ─────────────────────────────────────────────────────────────────────────────

class AUCBatchAggregatorToDisk:
    def __init__(self, num_tasks: int, cache_dir: str = "eval_cache", label_names=None):
        self.num_tasks = num_tasks
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def update(self, labels_batch, preds_batch):
        labels = _to_numpy(labels_batch)
        preds = _to_numpy(preds_batch)
        for i in range(self.num_tasks):
            uid = uuid.uuid4().hex
            np.save(self.cache_dir / f"task_{i}_labels_{uid}.npy", labels[:, i])
            np.save(self.cache_dir / f"task_{i}_probs_{uid}.npy", preds[:, i])

    def compute(self, reduce: str = "mean"):
        aucs = []
        for i in range(self.num_tasks):
            try:
                labels = self._load_all(f"task_{i}_labels_*.npy")
                preds = self._load_all(f"task_{i}_probs_*.npy")
                auc = roc_auc_score(labels, preds) if labels.size else float("nan")
            except Exception as e:
                logger.warning("AUC failed for task %d: %s", i, e)
                auc = float("nan")
            aucs.append(auc)
            self._cleanup(f"task_{i}_labels_*.npy")
            self._cleanup(f"task_{i}_probs_*.npy")

        return float(np.nanmean(aucs)) if reduce == "mean" else aucs

    def _load_all(self, pattern: str) -> np.ndarray:
        files = sorted(glob.glob(str(self.cache_dir / pattern)))
        arrays = [np.load(f, allow_pickle=True) for f in files]
        return np.concatenate(arrays) if arrays else np.array([])

    def _cleanup(self, pattern: str):
        for f in glob.glob(str(self.cache_dir / pattern)):
            os.remove(f)


# ─────────────────────────────────────────────────────────────────────────────
# MCC with threshold search
# ─────────────────────────────────────────────────────────────────────────────

class MCCBatchAggregatorToDisk:
    def __init__(self, num_tasks: int, cache_dir: str = "eval_cache", label_names=None):
        self.num_tasks = num_tasks
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.task_files = [
            open(self.cache_dir / f"task_{i}.pkl", "ab") for i in range(num_tasks)
        ]

    def update(self, labels_batch, preds_batch):
        labels = _to_numpy(labels_batch)
        preds = _to_numpy(preds_batch)
        for i in range(self.num_tasks):
            try:
                pickle.dump((labels[:, i].tolist(), preds[:, i].tolist()), self.task_files[i])
            except Exception as e:
                logger.warning("Failed to cache task %d: %s", i, e)

    def compute(
        self,
        threshold_search: np.ndarray = np.linspace(0.1, 0.9, 20),
        thresholds: Optional[np.ndarray] = None,
    ):
        self._close_files()
        mccs: List[float] = []
        best_thresholds: List[float] = []

        if thresholds is not None and len(thresholds) != self.num_tasks:
            logger.warning("Threshold count mismatch – ignoring provided thresholds.")
            thresholds = None

        for i in range(self.num_tasks):
            path = self.cache_dir / f"task_{i}.pkl"
            labels, preds = self._load(path)
            if not labels:
                mccs.append(0.0)
                best_thresholds.append(thresholds[i] if thresholds is not None else 0.5)
                continue

            labels_np = np.array(labels)
            preds_np = np.array(preds)

            try:
                if thresholds is not None:
                    bin_preds = (preds_np > thresholds[i]).astype(int)
                    mcc = matthews_corrcoef(labels_np, bin_preds) if labels_np.size else 0.0
                    best_thresholds.append(float(thresholds[i]))
                else:
                    mcc, best_t = -1.0, 0.5
                    for t in threshold_search:
                        try:
                            score = matthews_corrcoef(labels_np, (preds_np > t).astype(int))
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
            self._cleanup(path)

        return float(np.mean(mccs)), (thresholds if thresholds is not None else best_thresholds), mccs

    def _load(self, path: Path):
        labels, preds = [], []
        if path.exists():
            with open(path, "rb") as f:
                while True:
                    try:
                        l, p = pickle.load(f)
                        labels.extend(l)
                        preds.extend(p)
                    except EOFError:
                        break
                    except Exception as e:
                        logger.warning("Error reading %s: %s", path, e)
                        break
        return labels, preds

    def _close_files(self):
        for fh in self.task_files:
            try:
                fh.close()
            except Exception:
                pass

    def _cleanup(self, path: Path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Failed to remove %s: %s", path, e)


# ─────────────────────────────────────────────────────────────────────────────
# PR-AUC
# ─────────────────────────────────────────────────────────────────────────────

class PRBatchAggregatorToDisk:
    def __init__(self, num_tasks: int, cache_dir: Optional[str] = None, label_names=None):
        self.num_tasks = num_tasks
        self.cache_dir = Path(cache_dir or tempfile.mkdtemp())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]

    def update(self, labels_batch, preds_batch):
        labels = _to_numpy(labels_batch)
        preds = _to_numpy(preds_batch)
        for i in range(self.num_tasks):
            uid = uuid.uuid4().hex
            with (
                open(self.cache_dir / f"task_{i}_labels_{uid}.pkl", "wb") as lf,
                open(self.cache_dir / f"task_{i}_probs_{uid}.pkl", "wb") as pf,
            ):
                pickle.dump(list(labels[:, i]), lf)
                pickle.dump(list(preds[:, i]), pf)

    def compute(self, reduce: str = "mean"):
        pr_aucs = []
        for i in range(self.num_tasks):
            try:
                labels, preds = self._load_all(i)
                pr_aucs.append(average_precision_score(labels, preds) if labels else 0.0)
            except Exception as e:
                logger.warning("PR-AUC failed for task %d: %s", i, e)
                pr_aucs.append(0.0)
            self._cleanup(f"task_{i}_labels_*.pkl")
            self._cleanup(f"task_{i}_probs_*.pkl")
        return float(np.mean(pr_aucs)) if reduce == "mean" else np.array(pr_aucs)

    def _load_all(self, task_index: int):
        labels, preds = [], []
        label_files = sorted(glob.glob(str(self.cache_dir / f"task_{task_index}_labels_*.pkl")))
        prob_files = sorted(glob.glob(str(self.cache_dir / f"task_{task_index}_probs_*.pkl")))
        for lf, pf in zip(label_files, prob_files):
            with open(lf, "rb") as fl, open(pf, "rb") as fp:
                labels.extend(pickle.load(fl))
                preds.extend(pickle.load(fp))
        return labels, preds

    def _cleanup(self, pattern: str):
        for f in glob.glob(str(self.cache_dir / pattern)):
            os.remove(f)
