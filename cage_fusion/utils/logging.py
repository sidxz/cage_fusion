"""
cage_fusion/utils/logging.py

Centralized logging setup for the cage_fusion library.

Provides:
  - ``logger``  : standard ``logging.Logger`` with Rich console handler.
  - ``log_epoch_results``   : rich per-epoch training summary table.
  - ``plot_training_history``: training curve plots + CSV export.
  - ``plot_confusion_matrix``: per-task confusion matrix heatmap.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from sklearn.metrics import confusion_matrix

try:
    from loguru import logger as _loguru_logger
    _LOGURU_AVAILABLE = True
except ImportError:
    _LOGURU_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

LOG_DIR = os.getenv("CAGE_FUSION_LOG_DIR", "/logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "cagefusion.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)],
)

logger = logging.getLogger("cagefusion")

if _LOGURU_AVAILABLE:
    _loguru_logger.remove()
    _loguru_logger.add(
        LOG_FILE,
        rotation="10 MB",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        enqueue=True,
    )

    class _PropagateToLoguru(logging.Handler):
        def emit(self, record):
            try:
                level = _loguru_logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            _loguru_logger.log(level, record.getMessage())

    logging.getLogger().addHandler(_PropagateToLoguru())

# ---------------------------------------------------------------------------
# Console instance (shared across the package)
# ---------------------------------------------------------------------------

console = Console()


# ---------------------------------------------------------------------------
# Epoch summary logging
# ---------------------------------------------------------------------------

def log_epoch_results(
    epoch: int,
    num_epochs: int,
    history: dict,
    label_names: Optional[List[str]],
    per_task_metrics: list,
):
    """
    Print rich-formatted training / validation statistics for one epoch.

    Args:
        epoch: Current epoch number (1-based).
        num_epochs: Total number of epochs.
        history: Dict of lists keyed by metric name (e.g. ``"train_loss"``).
        label_names: Task / label names used as row labels.
        per_task_metrics: List of ``(mcc, auc, pr)`` tuples, one per task.
    """

    def _delta(current, history_list, is_loss=False):
        if len(history_list) < 2:
            return ""
        delta = current - history_list[-2]
        improved = (delta < 0) if is_loss else (delta > 0)
        color = "green" if improved else "red"
        sign = "+" if delta >= 0 else ""
        return f"[{color}]({sign}{delta:.4f})[/{color}]"

    console.rule(f"[bold blue]Epoch {epoch}/{num_epochs} Summary")

    train_loss = history["train_loss"][-1]
    console.print(
        f"[bold magenta]Train Loss:[/bold magenta] {train_loss:.4f} "
        f"{_delta(train_loss, history['train_loss'], is_loss=True)}"
    )

    # Validation summary table
    val_table = Table(title="Validation Metrics", show_header=True, header_style="bold cyan")
    val_table.add_column("Metric")
    val_table.add_column("Value", justify="right")
    val_table.add_column("Δ", justify="right")

    for key, name, is_loss in [
        ("val_loss", "Loss", True),
        ("val_mcc", "MCC", False),
        ("val_auc", "AUC", False),
        ("val_pr", "PR-AUC", False),
    ]:
        val = history[key][-1]
        val_table.add_row(name, f"{val:.4f}", _delta(val, history[key], is_loss))
    console.print(val_table)

    # Per-task table
    task_table = Table(
        title="Per-Task Validation Metrics",
        header_style="bold green",
        show_footer=False,
    )
    task_table.add_column("Task", style="cyan")
    task_table.add_column("ROC-AUC", style="magenta", justify="right")
    task_table.add_column("MCC", style="yellow", justify="right")
    task_table.add_column("PR-AUC", style="green", justify="right")

    prev_tasks = history["per_task"][-2] if len(history["per_task"]) > 1 else None
    for i, (mcc, auc, pr) in enumerate(per_task_metrics):
        task = label_names[i] if label_names and i < len(label_names) else f"Task {i}"
        auc_d = _delta(auc, [p[i][1] for p in history["per_task"]]) if prev_tasks else ""
        mcc_d = _delta(mcc, [p[i][0] for p in history["per_task"]]) if prev_tasks else ""
        pr_d  = _delta(pr,  [p[i][2] for p in history["per_task"]]) if prev_tasks else ""
        task_table.add_row(task, f"{auc:.3f} {auc_d}", f"{mcc:.3f} {mcc_d}", f"{pr:.3f} {pr_d}")

    macro_auc = history["val_auc"][-1]
    macro_mcc = history["val_mcc"][-1]
    macro_pr  = history["val_pr"][-1]
    task_table.add_section()
    task_table.add_row(
        "[bold]Macro-Avg[/bold]",
        f"[bold]{macro_auc:.4f}[/bold]",
        f"[bold]{macro_mcc:.4f}[/bold]",
        f"[bold]{macro_pr:.4f}[/bold]",
    )
    console.print(task_table)

    # Learned scaler table
    scale_table = Table(title="Learned Modality Scalers", header_style="bold yellow")
    scale_table.add_column("Scaler", justify="left")
    scale_table.add_column("Value", justify="right")
    scale_table.add_column("Avg. Rep Norm", justify="right")
    scale_table.add_column("Scaled Norm", justify="right")
    scale_table.add_column("Δ (Value)", justify="right")

    for scale_key, norm_key in [
        ("scale_graph", "val_norm_graph"),
        ("scale_attn", "val_norm_attn"),
        ("scale_aux", "val_norm_aux"),
    ]:
        val = history[scale_key][-1]
        norm = history.get(norm_key, [0])[-1]
        scale_table.add_row(
            scale_key,
            f"{val:.4f}",
            f"{norm:.4f}",
            f"{val * norm:.4f}",
            _delta(val, history[scale_key]),
        )
    console.print(scale_table)


# ---------------------------------------------------------------------------
# Training curve plots
# ---------------------------------------------------------------------------

def plot_training_history(history: dict, output_dir: Optional[str] = None):
    """
    Plot train/val curves for loss, MCC, AUC, and PR-AUC, and optionally save
    them as PNGs and a CSV.

    Args:
        history: Training history dict produced by :class:`~cage_fusion.training.Trainer`.
        output_dir: If given, saves plots and CSV there; otherwise shows interactively.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    metrics = [
        ("loss", "train_loss", "val_loss"),
        ("mcc",  "train_mcc",  "val_mcc"),
        ("auc",  "train_auc",  "val_auc"),
        ("pr",   "train_pr",   "val_pr"),
    ]

    for title, train_key, val_key in metrics:
        plt.figure()
        plt.plot(history[train_key], label="Train")
        plt.plot(history[val_key],   label="Validation")
        plt.title(f"{title.upper()} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(title.upper())
        plt.legend()
        plt.grid(True)
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{title}_curve.png"))
        else:
            plt.show()
        plt.close()

    df = pd.DataFrame(
        {k: v for k, v in history.items() if isinstance(v, (list, float))}
    )
    if output_dir:
        csv_path = os.path.join(output_dir, "training_history.csv")
        df.to_csv(csv_path, index_label="epoch")
        logger.info(f"Saved training history to {csv_path}")
    else:
        print(df.head())


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str,
    save_path: str,
):
    """Save a seaborn confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
