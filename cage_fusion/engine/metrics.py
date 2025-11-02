import os
import torch
import pickle
import numpy as np
import tempfile
import warnings
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score
from cage_fusion.utils.logging_utils import logger
import uuid
import glob
from pathlib import Path


def to_numpy(data):
    return data.cpu().numpy() if isinstance(data, torch.Tensor) else data


# class AUCBatchAggregatorToDisk:
#     def __init__(self, num_tasks: int, cache_dir="eval_cache", label_names=None):
#         self.num_tasks = num_tasks
#         self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
#         self.cache_dir = Path(cache_dir)
#         self.cache_dir.mkdir(parents=True, exist_ok=True)

#     def update(self, labels_batch, preds_batch):
#         labels = to_numpy(labels_batch)
#         preds = to_numpy(preds_batch)
#         for i in range(self.num_tasks):
#             uid = uuid.uuid4().hex
#             np.save(self.cache_dir / f"task_{i}_labels_{uid}.npy", labels[:, i])
#             np.save(self.cache_dir / f"task_{i}_probs_{uid}.npy", preds[:, i])

#     def compute(self, reduce="mean"):
#         aucs = []
#         for i in range(self.num_tasks):
#             try:
#                 labels = self._load_all(f"task_{i}_labels_*.npy")
#                 preds = self._load_all(f"task_{i}_probs_*.npy")
#                 auc = roc_auc_score(labels, preds) if labels.size else float("nan")
#             except Exception as e:
#                 logger.warning(f"AUC computation failed for task {i}: {e}")
#                 auc = float("nan")
#             aucs.append(auc)
#             self._cleanup(f"task_{i}_labels_*.npy")
#             self._cleanup(f"task_{i}_probs_*.npy")

#         return np.nanmean(aucs) if reduce == "mean" else aucs

#     def _load_all(self, pattern):
#         files = sorted(glob.glob(str(self.cache_dir / pattern)))
#         arrays = [np.load(f, allow_pickle=True) for f in files]
#         return np.concatenate(arrays) if arrays else np.array([])

#     def _cleanup(self, pattern):
#         for f in glob.glob(str(self.cache_dir / pattern)):
#             os.remove(f)


# With debug prints
class AUCBatchAggregatorToDisk:
    def __init__(self, num_tasks: int, cache_dir="eval_cache", label_names=None):
        self.num_tasks = num_tasks
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def update(self, labels_batch, preds_batch):
        labels = to_numpy(labels_batch)
        preds = to_numpy(preds_batch)
        for i in range(self.num_tasks):
            uid = uuid.uuid4().hex
            np.save(self.cache_dir / f"task_{i}_labels_{uid}.npy", labels[:, i])
            np.save(self.cache_dir / f"task_{i}_probs_{uid}.npy", preds[:, i])

    def compute(self, reduce="mean"):
        aucs = []
        for i in range(self.num_tasks):
            try:
                labels = self._load_all(f"task_{i}_labels_*.npy")
                preds = self._load_all(f"task_{i}_probs_*.npy")

                # Print values for debugging
                logger.debug(f"\nTask {i}:")
                logger.debug(f"Labels shape: {labels.shape}")
                logger.debug(f"Preds shape:  {preds.shape}")
                logger.debug(f"Labels: {labels}")
                logger.debug(f"Preds:  {preds}")

                # Count 0s, 1s, and NaNs
                label_flat = labels.flatten()
                num_zeros = int((label_flat == 0).sum())
                num_ones = int((label_flat == 1).sum())
                num_nans = int(np.isnan(label_flat).sum())
                logger.debug(
                    f"Label counts — 0s: {num_zeros}, 1s: {num_ones}, NaNs: {num_nans}"
                )

                auc = roc_auc_score(labels, preds) if labels.size else float("nan")
                logger.debug(f"AUC: {auc}")
            except Exception as e:
                logger.warning(f"AUC computation failed for task {i}: {e}")
                auc = float("nan")
            aucs.append(auc)
            self._cleanup(f"task_{i}_labels_*.npy")
            self._cleanup(f"task_{i}_probs_*.npy")

        mean_auc = np.nanmean(aucs)
        logger.debug(f"\nMean AUC across all tasks: {mean_auc}")

        return mean_auc if reduce == "mean" else aucs

    def _load_all(self, pattern):
        files = sorted(glob.glob(str(self.cache_dir / pattern)))
        arrays = [np.load(f, allow_pickle=True) for f in files]
        return np.concatenate(arrays) if arrays else np.array([])

    def _cleanup(self, pattern):
        for f in glob.glob(str(self.cache_dir / pattern)):
            os.remove(f)


class MCCBatchAggregatorToDisk:
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
            try:
                pickle.dump(
                    (labels[:, i].tolist(), preds[:, i].tolist()), self.task_files[i]
                )
            except Exception as e:
                logger.warning(f"Failed to cache task {i} batch: {e}")

    def compute(self, threshold_search=np.linspace(0.1, 0.9, 20), thresholds=None):
        self._close_files()
        mccs, best_thresholds = [], []

        # Defensive check for threshold size
        if thresholds is not None:
            if not isinstance(thresholds, (list, np.ndarray)):
                logger.warning("Provided thresholds must be a list or numpy array.")
                thresholds = None
            elif len(thresholds) != self.num_tasks:
                logger.warning(
                    f"Expected {self.num_tasks} thresholds, but got {len(thresholds)}. Ignoring thresholds."
                )
                thresholds = None

        for i in range(self.num_tasks):
            path = self.cache_dir / f"task_{i}.pkl"
            labels, preds = self._load(path)
            if not labels:
                logger.warning(f"No labels for task {i}; skipping MCC.")
                mccs.append(0.0)
                best_thresholds.append(thresholds[i] if thresholds is not None else 0.5)
                continue

            labels_np, preds_np = np.array(labels), np.array(preds)

            try:
                if thresholds is not None:
                    bin_preds = (preds_np > thresholds[i]).astype(int)
                    mcc = (
                        matthews_corrcoef(labels_np, bin_preds)
                        if labels_np.size
                        else 0.0
                    )
                    best_thresholds.append(thresholds[i])
                else:
                    mcc, best_t = -1.0, 0.5
                    for t in threshold_search:
                        try:
                            bp = (preds_np > t).astype(int)
                            score = matthews_corrcoef(labels_np, bp)
                            if score > mcc:
                                mcc, best_t = score, t
                        except Exception as inner_e:
                            logger.debug(
                                f"Threshold {t:.3f} failed for task {i}: {inner_e}"
                            )
                            continue
                    best_thresholds.append(best_t)
            except Exception as e:
                logger.warning(f"MCC computation failed for task {i}: {e}")
                mcc = 0.0
                best_thresholds.append(0.5)

            mccs.append(mcc)
            self._cleanup(path)

        return float(np.mean(mccs)), best_thresholds if thresholds is None else thresholds, mccs


    def _load(self, path):
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
                        logger.warning(f"Error loading cached file {path}: {e}")
                        break
        return labels, preds

    def _close_files(self):
        for f in self.task_files:
            try:
                f.close()
            except Exception as e:
                logger.warning(f"Failed to close file: {e}")

    def _cleanup(self, path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Failed to remove {path}: {e}")


class PRBatchAggregatorToDisk:
    def __init__(self, num_tasks: int, cache_dir=None, label_names=None):
        self.num_tasks = num_tasks
        self.cache_dir = Path(cache_dir or tempfile.mkdtemp())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]

    def update(self, labels_batch, preds_batch):
        labels = to_numpy(labels_batch)
        preds = to_numpy(preds_batch)
        for i in range(self.num_tasks):
            uid = uuid.uuid4().hex
            with open(self.cache_dir / f"task_{i}_labels_{uid}.pkl", "wb") as lf, open(
                self.cache_dir / f"task_{i}_probs_{uid}.pkl", "wb"
            ) as pf:
                pickle.dump(list(labels[:, i]), lf)
                pickle.dump(list(preds[:, i]), pf)

    def compute(self, reduce="mean"):
        pr_aucs = []
        for i in range(self.num_tasks):
            try:
                labels, preds = self._load_all(i)
                pr_aucs.append(
                    average_precision_score(labels, preds) if labels else 0.0
                )
            except Exception as e:
                logger.warning(f"PR-AUC computation failed for task {i}: {e}")
                pr_aucs.append(0.0)
            self._cleanup(f"task_{i}_labels_*.pkl")
            self._cleanup(f"task_{i}_probs_*.pkl")

        return float(np.mean(pr_aucs)) if reduce == "mean" else np.array(pr_aucs)

    def _load_all(self, task_index):
        labels, preds = [], []
        label_files = sorted(
            glob.glob(str(self.cache_dir / f"task_{task_index}_labels_*.pkl"))
        )
        prob_files = sorted(
            glob.glob(str(self.cache_dir / f"task_{task_index}_probs_*.pkl"))
        )
        for lf, pf in zip(label_files, prob_files):
            with open(lf, "rb") as f_l, open(pf, "rb") as f_p:
                labels.extend(pickle.load(f_l))
                preds.extend(pickle.load(f_p))
        return labels, preds

    def _cleanup(self, pattern):
        for f in glob.glob(str(self.cache_dir / pattern)):
            os.remove(f)
