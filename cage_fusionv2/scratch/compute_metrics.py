import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)
from tabulate import tabulate
from transformers import TrainerCallback


# -----------------------------
# Pretty Table Callback
# -----------------------------
class PrettyPrintCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        skip = {"total_flos", "train_loss", "grad_norm"}
        clean_logs = {k: v for k, v in logs.items() if k not in skip}

        table = []
        for k, v in clean_logs.items():
            table.append([k, f"{v:.4f}" if isinstance(v, float) else v])

        print(
            "\n"
            + tabulate(table, headers=["Metric", "Value"], tablefmt="github")
            + "\n"
        )


# ---------------------------------------------------
#  CORE METRIC COMPUTATION (general, reusable)
# ---------------------------------------------------
def compute_all_metrics(logits, labels, label_names, thresholds=None):
    """
    Compute metrics for multi-label classification.
    If thresholds is None → uses 0.5 for all labels.

    logits: (N, L)
    labels: (N, L)
    thresholds: dict[label_name] => float
    """

    logits = np.asarray(logits)
    labels = np.asarray(labels)

    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    y_true = labels.astype(int)

    # Thresholds
    if thresholds is None:
        y_pred = (probs >= 0.5).astype(int)
    else:
        thr_vec = np.array([thresholds[name] for name in label_names])[None, :]
        y_pred = (probs >= thr_vec).astype(int)

    metrics = {}

    # AUC metrics (independent of threshold)
    try:
        metrics["roc_auc_macro"] = roc_auc_score(y_true, probs, average="macro")
        metrics["roc_auc_micro"] = roc_auc_score(y_true, probs, average="micro")
    except ValueError:
        metrics["roc_auc_macro"] = float("nan")
        metrics["roc_auc_micro"] = float("nan")

    # PR AUC
    try:
        metrics["avg_precision_macro"] = average_precision_score(
            y_true, probs, average="macro"
        )
        metrics["avg_precision_micro"] = average_precision_score(
            y_true, probs, average="micro"
        )
    except ValueError:
        metrics["avg_precision_macro"] = float("nan")
        metrics["avg_precision_micro"] = float("nan")

    # F1/precision/recall
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    metrics["precision_macro"] = p_macro
    metrics["recall_macro"] = r_macro
    metrics["f1_macro"] = f_macro
    metrics["precision_micro"] = p_micro
    metrics["recall_micro"] = r_micro
    metrics["f1_micro"] = f_micro

    # Per-label F1
    _, _, f1_per_label, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    for name, f1 in zip(label_names, f1_per_label):
        metrics[f"f1_{name}"] = f1

    # MCC + accuracy
    metrics["mcc"] = matthews_corrcoef(y_true.ravel(), y_pred.ravel())
    metrics["accuracy"] = accuracy_score(y_true.ravel(), y_pred.ravel())

    return metrics


# ---------------------------------------------------
#  HF Trainer-compatible wrapper
#  (always uses threshold=0.5)
# ---------------------------------------------------
def make_compute_metrics(label_names):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return compute_all_metrics(
            logits=logits,
            labels=labels,
            label_names=label_names,
            thresholds=None,  # ALWAYS 0.5 inside Trainer
        )

    return compute_metrics
