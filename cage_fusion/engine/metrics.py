import os
import torch
import pickle
import numpy as np
import tempfile
import warnings
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score
from cage_fusion.utils.logging_utils import logger


class AUCBatchAggregatorToDisk:
    """Streams batch-wise predictions to disk and computes ROC-AUC per task."""

    def __init__(self, num_tasks: int, cache_dir="eval_cache", label_names=None):
        self.num_tasks = num_tasks
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.label_files = [open(os.path.join(cache_dir, f"task_{i}_labels.npy"), "wb") for i in range(num_tasks)]
        self.prob_files = [open(os.path.join(cache_dir, f"task_{i}_probs.npy"), "wb") for i in range(num_tasks)]

    def update(self, labels_batch, preds_batch):
        labels = labels_batch.cpu().numpy() if isinstance(labels_batch, torch.Tensor) else labels_batch
        preds = preds_batch.cpu().numpy() if isinstance(preds_batch, torch.Tensor) else preds_batch
        for i in range(self.num_tasks):
            np.save(self.label_files[i], labels[:, i])
            np.save(self.prob_files[i], preds[:, i])

    def compute(self, reduce="mean"):
        self._close_files()
        aucs = []

        for i in range(self.num_tasks):
            try:
                labels = self._load_npy_sequence(f"task_{i}_labels.npy")
                probs = self._load_npy_sequence(f"task_{i}_probs.npy")
                auc = roc_auc_score(labels, probs) if labels.size and probs.size else float("nan")
            except Exception as e:
                logger.warning("Failed to compute AUC for task %d: %s", i, e)
                auc = float("nan")
            aucs.append(auc)
            self._safe_remove(f"task_{i}_labels.npy")
            self._safe_remove(f"task_{i}_probs.npy")

        return np.nanmean(aucs) if reduce == "mean" else aucs

    def _load_npy_sequence(self, filename):
        data = []
        path = os.path.join(self.cache_dir, filename)
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
            os.remove(os.path.join(self.cache_dir, filename))
        except OSError:
            pass


class MCCBatchAggregatorToDisk:
    """Streams predictions and labels for MCC computation with threshold search."""

    def __init__(self, num_tasks: int, cache_dir="eval_cache", label_names=None):
        self.num_tasks = num_tasks
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.task_files = [open(os.path.join(cache_dir, f"task_{i}.pkl"), "ab") for i in range(num_tasks)]

    def update(self, labels_batch, preds_batch):
        labels = labels_batch.cpu().numpy() if isinstance(labels_batch, torch.Tensor) else labels_batch
        preds = preds_batch.cpu().numpy() if isinstance(preds_batch, torch.Tensor) else preds_batch
        for i in range(self.num_tasks):
            pickle.dump((labels[:, i].tolist(), preds[:, i].tolist()), self.task_files[i])

    def compute(self, threshold_search=np.linspace(0.1, 0.9, 20)):
        self._close_files()
        mccs, best_thresholds = [], []

        for i in range(self.num_tasks):
            path = os.path.join(self.cache_dir, f"task_{i}.pkl")
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
        if not os.path.exists(path):
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
            os.remove(os.path.join(self.cache_dir, filename))
        except OSError:
            pass


class PRBatchAggregatorToDisk:
    """Streams batch outputs to compute PR-AUC (Average Precision) later."""

    def __init__(self, num_tasks: int, cache_dir=None, label_names=None):
        self.num_tasks = num_tasks
        self.cache_dir = cache_dir or tempfile.mkdtemp()
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        os.makedirs(self.cache_dir, exist_ok=True)
        self.label_files = [open(os.path.join(self.cache_dir, f"task_{i}_labels.pkl"), "wb") for i in range(num_tasks)]
        self.prob_files = [open(os.path.join(self.cache_dir, f"task_{i}_probs.pkl"), "wb") for i in range(num_tasks)]

    def update(self, labels_batch, preds_batch):
        labels = labels_batch.cpu().numpy() if isinstance(labels_batch, torch.Tensor) else labels_batch
        preds = preds_batch.cpu().numpy() if isinstance(preds_batch, torch.Tensor) else preds_batch
        for i in range(self.num_tasks):
            pickle.dump(list(labels[:, i]), self.label_files[i])
            pickle.dump(list(preds[:, i]), self.prob_files[i])

    def compute(self, reduce="mean"):
        self._close_files()
        pr_aucs = []

        for i in range(self.num_tasks):
            try:
                labels, preds = self._load_pickle_sequence(i)
                if labels:
                    pr_aucs.append(average_precision_score(labels, preds))
                else:
                    pr_aucs.append(0.0)
            except Exception as e:
                logger.warning("PR-AUC failed for task %d: %s", i, e)
                pr_aucs.append(0.0)
            self._safe_remove(f"task_{i}_labels.pkl")
            self._safe_remove(f"task_{i}_probs.pkl")

        return float(np.mean(pr_aucs)) if reduce == "mean" else np.array(pr_aucs)

    def _close_files(self):
        for f in self.label_files + self.prob_files:
            f.close()

    def _load_pickle_sequence(self, task_index):
        labels, preds = [], []
        with open(os.path.join(self.cache_dir, f"task_{task_index}_labels.pkl"), "rb") as f_l, \
             open(os.path.join(self.cache_dir, f"task_{task_index}_probs.pkl"), "rb") as f_p:
            while True:
                try:
                    labels.extend(pickle.load(f_l))
                    preds.extend(pickle.load(f_p))
                except EOFError:
                    break
        return labels, preds

    def _safe_remove(self, filename):
        try:
            os.remove(os.path.join(self.cache_dir, filename))
        except OSError:
            pass
