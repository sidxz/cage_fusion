import os
import torch
import numpy as np
import warnings
import pickle
import tempfile
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score

class AUCBatchAggregatorToDisk:
    """Calculates ROC AUC scores by streaming batch results to disk to save RAM."""
    def __init__(self, num_tasks, cache_dir="eval_cache", label_names=None):
        self.num_tasks = num_tasks; self.cache_dir = cache_dir
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        os.makedirs(cache_dir, exist_ok=True)
        self.label_files = [open(os.path.join(cache_dir, f"task_{i}_labels.npy"), "wb") for i in range(num_tasks)]
        self.prob_files = [open(os.path.join(cache_dir, f"task_{i}_probs.npy"), "wb") for i in range(num_tasks)]

    def update(self, labels_batch, preds_batch):
        labels = labels_batch.cpu().numpy() if isinstance(labels_batch, torch.Tensor) else labels_batch
        preds = preds_batch.cpu().numpy() if isinstance(preds_batch, torch.Tensor) else preds_batch
        for i in range(self.num_tasks):
            np.save(self.label_files[i], labels[:, i]); np.save(self.prob_files[i], preds[:, i])

    def close_files(self):
        for f in self.label_files + self.prob_files:
            if not f.closed: f.close()

    def compute(self, reduce="mean"):
        self.close_files()
        def load_all_arrays(path):
            arrays = []
            with open(path, "rb") as f:
                while True:
                    try: arrays.append(np.load(f, allow_pickle=True))
                    except (EOFError, ValueError): break
            return np.concatenate(arrays) if arrays else np.array([])
        aucs = []
        for i in range(self.num_tasks):
            label_path = os.path.join(self.cache_dir, f"task_{i}_labels.npy")
            prob_path = os.path.join(self.cache_dir, f"task_{i}_probs.npy")
            labels = load_all_arrays(label_path)
            probs = load_all_arrays(prob_path)
            if labels.size == 0 or probs.size == 0: aucs.append(float("nan")); continue
            try: aucs.append(roc_auc_score(labels, probs))
            except ValueError: aucs.append(float("nan"))
        for i in range(self.num_tasks):
            try:
                os.remove(os.path.join(self.cache_dir, f"task_{i}_labels.npy"))
                os.remove(os.path.join(self.cache_dir, f"task_{i}_probs.npy"))
            except OSError: pass
        return np.nanmean(aucs) if reduce == "mean" else aucs

class MCCBatchAggregatorToDisk:
    """Calculates Matthews Correlation Coefficient by streaming to disk."""
    def __init__(self, num_tasks, cache_dir="eval_cache", label_names=None):
        self.num_tasks = num_tasks; self.cache_dir = cache_dir
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        os.makedirs(cache_dir, exist_ok=True)
        self.task_files = [open(os.path.join(cache_dir, f"task_{i}.pkl"), "ab") for i in range(num_tasks)]

    def update(self, labels_batch, preds_batch):
        # Convert tensors to CPU numpy arrays first
        labels_np = labels_batch.cpu().numpy() if isinstance(labels_batch, torch.Tensor) else labels_batch
        preds_np = preds_batch.cpu().numpy() if isinstance(preds_batch, torch.Tensor) else preds_batch
        
        # CORRECTED: Use the numpy arrays (labels_np, preds_np) for pickling
        for i in range(self.num_tasks):
            data = (labels_np[:, i].tolist(), preds_np[:, i].tolist())
            pickle.dump(data, self.task_files[i])

    def close_files(self):
        for f in self.task_files:
            if not f.closed: f.close()

    def compute(self, threshold_search=np.linspace(0.1, 0.9, 20)):
        self.close_files()
        best_thresholds, per_task_mcc = [], []
        for i in range(self.num_tasks):
            path = os.path.join(self.cache_dir, f"task_{i}.pkl")
            all_labels, all_preds = [], []
            if not os.path.exists(path): per_task_mcc.append(0.0); best_thresholds.append(0.5); continue
            with open(path, "rb") as f:
                while True:
                    try: 
                        labels, preds = pickle.load(f)
                        all_labels.extend(labels); all_preds.extend(preds)
                    except EOFError: break
            if not all_labels: per_task_mcc.append(0.0); best_thresholds.append(0.5); continue
            labels, preds = np.array(all_labels), np.array(all_preds)
            best_mcc, best_thresh = -1.0, 0.5
            for t in threshold_search:
                bin_preds = (preds > t).astype(int)
                try: mcc = matthews_corrcoef(labels, bin_preds)
                except ValueError: mcc = 0.0
                if mcc > best_mcc: best_mcc, best_thresh = mcc, t
            per_task_mcc.append(best_mcc); best_thresholds.append(best_thresh)
            try: os.remove(path)
            except OSError: pass
        return float(np.mean(per_task_mcc)), best_thresholds, per_task_mcc

class PRBatchAggregatorToDisk:
    """Calculates PR-AUC (Average Precision) by streaming to disk."""
    def __init__(self, num_tasks, cache_dir=None, label_names=None):
        self.num_tasks = num_tasks; self.cache_dir = cache_dir or tempfile.mkdtemp()
        self.label_names = label_names or [f"Task {i}" for i in range(num_tasks)]
        os.makedirs(self.cache_dir, exist_ok=True)
        self.label_files = [open(os.path.join(self.cache_dir, f"task_{i}_labels_pr.pkl"), "wb") for i in range(num_tasks)]
        self.prob_files = [open(os.path.join(self.cache_dir, f"task_{i}_probs_pr.pkl"), "wb") for i in range(num_tasks)]

    def update(self, labels_batch, preds_batch):
        labels_np = labels_batch.cpu().numpy() if isinstance(labels_batch, torch.Tensor) else labels_batch
        preds_np = preds_batch.cpu().numpy() if isinstance(preds_batch, torch.Tensor) else preds_batch
        for i in range(self.num_tasks):
            pickle.dump(list(labels_np[:, i]), self.label_files[i]); pickle.dump(list(preds_np[:, i]), self.prob_files[i])

    def compute(self, reduce="mean"):
        for f in self.label_files + self.prob_files: f.close()
        pr_aucs = []
        for i in range(self.num_tasks):
            labels, preds = [], []
            label_path = os.path.join(self.cache_dir, f"task_{i}_labels_pr.pkl")
            prob_path  = os.path.join(self.cache_dir, f"task_{i}_probs_pr.pkl")
            if not os.path.exists(label_path): pr_aucs.append(0.0); continue
            with open(label_path, "rb") as f_l, open(prob_path, "rb") as f_p:
                while True:
                    try: labels.extend(pickle.load(f_l)); preds.extend(pickle.load(f_p))
                    except EOFError: break
            if not labels: pr_aucs.append(0.0); continue
            try: pr_aucs.append(average_precision_score(labels, preds))
            except ValueError: pr_aucs.append(0.0)
            try: os.remove(label_path); os.remove(prob_path)
            except OSError: pass
        return np.mean(pr_aucs) if reduce == "mean" else np.array(pr_aucs)
