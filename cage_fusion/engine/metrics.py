import os
import torch
import pickle
import numpy as np
import tempfile
from pathlib import Path
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score
from cage_fusion.utils.logging_utils import logger


def to_numpy(data):
    return data.cpu().numpy() if isinstance(data, torch.Tensor) else data


class AUCBatchAggregatorToDisk:
    """Streams batch-wise predictions to disk and computes ROC-AUC per task."""

    def __init__(self, num_tasks: int, cache_dir="eval_cache", label_names=None):
        self.num_tasks = num_tasks
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.label_files = [
            open(self.cache_dir / f"task_{i}_labels.npy", "wb")
            for i in range(num_tasks)
        ]
        self.prob_files = [
            open(self.cache_dir / f"task_{i}_probs.npy", "wb") for i in range(num_tasks)
        ]

    def update(self, labels_batch, preds_batch):
        labels = to_numpy(labels_batch)
        preds = to_numpy(preds_batch)
        for i in range(self.num_tasks):
            np.save(self.label_files[i], labels[:, i])
            np.save(self.prob_files[i], preds[:, i])

    def compute(self, reduce="mean"):
        self._close_files()
        aucs = []

        for i in range(self.num_tasks):
            try:
                labels = self._load_npy_sequence(
                    self.cache_dir / f"task_{i}_labels.npy"
                )
                preds = self._load_npy_sequence(self.cache_dir / f"task_{i}_probs.npy")
                auc = (
                    roc_auc_score(labels, preds)
                    if labels.size and preds.size
                    else float("nan")
                )
            except Exception as e:
                logger.warning(f"AUC computation failed for task {i}: {e}")
                auc = float("nan")
            aucs.append(auc)
            self._safe_remove(f"task_{i}_labels.npy")
            self._safe_remove(f"task_{i}_probs.npy")

        return np.nanmean(aucs) if reduce == "mean" else aucs

    def _load_npy_sequence(self, path):
        data = []
        with open(path, "rb") as f:
            while True:
                try:
                    data.append(np.load(f, allow_pickle=True))
                except Exception:
                    break
        return np.concatenate(data) if data else np.array([])

    def _close_files(self):
        for f in self.label_files + self.prob_files:
            f.close()

    def _safe_remove(self, filename):
        try:
            os.remove(self.cache_dir / filename)
        except OSError:
            pass


class MCCBatchAggregatorToDisk:
    """Streams predictions and labels for MCC computation with threshold optimization."""

    def __init__(self, num_tasks: int, cache_dir="eval_cache", label_names=None):
        self.num_tasks = num_tasks
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.task_files = [
            open(self.cache_dir / f"task_{i}.pkl", "ab") for i in range(num_tasks)
        ]

    def update(self, labels_batch, preds_batch):
        labels = to_numpy(labels_batch)
        preds = to_numpy(preds_batch)
        for i in range(self.num_tasks):
            pickle.dump(
                (labels[:, i].tolist(), preds[:, i].tolist()), self.task_files[i]
            )

    def compute(self, threshold_search=np.linspace(0.1, 0.9, 20)):
        self._close_files()
        mccs, best_thresholds = [], []

        for i in range(self.num_tasks):
            path = self.cache_dir / f"task_{i}.pkl"
            labels, preds = self._load_pickle_sequence(path)
            if not labels:
                mccs.append(0.0)
                best_thresholds.append(0.5)
                continue

            best_mcc, best_thresh = -1.0, 0.5
            for t in threshold_search:
                try:
                    bin_preds = (np.array(preds) > t).astype(int)
                    mcc = matthews_corrcoef(labels, bin_preds)
                    if mcc > best_mcc:
                        best_mcc, best_thresh = mcc, t
                except ValueError:
                    continue

            mccs.append(best_mcc)
            best_thresholds.append(best_thresh)
            self._safe_remove(f"task_{i}.pkl")

        return float(np.mean(mccs)), best_thresholds, mccs

    def _close_files(self):
        for f in self.task_files:
            f.close()

    def _load_pickle_sequence(self, path):
        labels, preds = [], []
        if not path.exists():
            return labels, preds
        with open(path, "rb") as f:
            while True:
                try:
                    l, p = pickle.load(f)
                    labels.extend(l)
                    preds.extend(p)
                except EOFError:
                    break
        return labels, preds

    def _safe_remove(self, filename):
        try:
            os.remove(self.cache_dir / filename)
        except OSError:
            pass


class PRBatchAggregatorToDisk:
    """Streams batch outputs to compute PR-AUC (Average Precision) per task."""

    def __init__(self, num_tasks: int, cache_dir=None, label_names=None):
        self.num_tasks = num_tasks
        self.cache_dir = Path(cache_dir or tempfile.mkdtemp())
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.label_files = [
            open(self.cache_dir / f"task_{i}_labels.pkl", "wb")
            for i in range(num_tasks)
        ]
        self.prob_files = [
            open(self.cache_dir / f"task_{i}_probs.pkl", "wb") for i in range(num_tasks)
        ]

    def update(self, labels_batch, preds_batch):
        labels = to_numpy(labels_batch)
        preds = to_numpy(preds_batch)
        for i in range(self.num_tasks):
            pickle.dump(list(labels[:, i]), self.label_files[i])
            pickle.dump(list(preds[:, i]), self.prob_files[i])

    def compute(self, reduce="mean"):
        self._close_files()
        pr_aucs = []

        for i in range(self.num_tasks):
            try:
                labels, preds = self._load_pickle_sequence(i)
                pr_aucs.append(
                    average_precision_score(labels, preds) if labels else 0.0
                )
            except Exception as e:
                logger.warning(f"PR-AUC computation failed for task {i}: {e}")
                pr_aucs.append(0.0)
            self._safe_remove(f"task_{i}_labels.pkl")
            self._safe_remove(f"task_{i}_probs.pkl")

        return float(np.mean(pr_aucs)) if reduce == "mean" else np.array(pr_aucs)

    def _close_files(self):
        for f in self.label_files + self.prob_files:
            f.close()

    def _load_pickle_sequence(self, task_index):
        labels, preds = [], []
        label_path = self.cache_dir / f"task_{task_index}_labels.pkl"
        prob_path = self.cache_dir / f"task_{task_index}_probs.pkl"
        with open(label_path, "rb") as f_l, open(prob_path, "rb") as f_p:
            while True:
                try:
                    labels.extend(pickle.load(f_l))
                    preds.extend(pickle.load(f_p))
                except EOFError:
                    break
        return labels, preds

    def _safe_remove(self, filename):
        try:
            os.remove(self.cache_dir / filename)
        except OSError:
            pass
