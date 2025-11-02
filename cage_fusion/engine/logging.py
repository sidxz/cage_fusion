import numpy as np
import matplotlib.pyplot as plt
from cage_fusion.utils.logging_utils import logger
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

console = Console()


def plot_confusion_matrix(y_true, y_pred, title, save_path):
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


def plot_training_history(history, output_dir=None):
    """
    Plot training and validation curves, and save training history as CSV.

    Args:
        history (dict): Training history with keys like 'train_loss', 'val_loss', etc.
        output_dir (str, optional): If provided, saves plots and CSV there.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    metrics = [
        ("loss", "train_loss", "val_loss"),
        ("mcc", "train_mcc", "val_mcc"),
        ("auc", "train_auc", "val_auc"),
        ("pr", "train_pr", "val_pr"),
    ]

    for title, train_key, val_key in metrics:
        plt.figure()
        plt.plot(history[train_key], label="Train")
        plt.plot(history[val_key], label="Validation")
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

    # Save history to CSV
    df = pd.DataFrame(
        {
            k: v
            for k, v in history.items()
            if isinstance(v, list) or isinstance(v, float)
        }
    )
    if output_dir:
        csv_path = os.path.join(output_dir, "training_history.csv")
        df.to_csv(csv_path, index_label="epoch")
        print(f"✅ Saved training history to {csv_path}")
    else:
        print(df.head())


def log_epoch_results(epoch, num_epochs, history, label_names, per_task_metrics):
    """
    Logs training and validation statistics at each epoch using rich formatting.
    """

    def get_colored_delta(current, history_list, is_loss=False):
        if len(history_list) < 2:
            return ""
        delta = current - history_list[-2]
        improved = (delta < 0) if is_loss else (delta > 0)
        color = "green" if improved else "red"
        sign = "+" if delta >= 0 else ""
        return f"[{color}]({sign}{delta:.4f})[/{color}]"

    console.rule(f"[bold blue]Epoch {epoch}/{num_epochs} Summary")

    # --- Train Loss ---
    train_loss = history["train_loss"][-1]
    console.print(
        f"[bold magenta]Train Loss:[/bold magenta] {train_loss:.4f} {get_colored_delta(train_loss, history['train_loss'], is_loss=True)}"
    )

    # --- Validation Summary ---
    val_table = Table(
        title="Validation Metrics", show_header=True, header_style="bold cyan"
    )
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
        val_table.add_row(
            name, f"{val:.4f}", get_colored_delta(val, history[key], is_loss)
        )
    console.print(val_table)

    # --- Per Task Metrics ---
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
        auc_d = (
            get_colored_delta(auc, [p[i][1] for p in history["per_task"]])
            if prev_tasks
            else ""
        )
        mcc_d = (
            get_colored_delta(mcc, [p[i][0] for p in history["per_task"]])
            if prev_tasks
            else ""
        )
        pr_d = (
            get_colored_delta(pr, [p[i][2] for p in history["per_task"]])
            if prev_tasks
            else ""
        )
        task_table.add_row(
            task, f"{auc:.3f} {auc_d}", f"{mcc:.3f} {mcc_d}", f"{pr:.3f} {pr_d}"
        )

    # --- MODIFIED: Add Macro-Average Row ---
    macro_auc = history["val_auc"][-1]
    macro_mcc = history["val_mcc"][-1]
    macro_pr = history["val_pr"][-1]

    task_table.add_section()  # Adds a separator line
    task_table.add_row(
        "[bold]Macro-Avg[/bold]",
        f"[bold]{macro_auc:.4f}[/bold]",
        f"[bold]{macro_mcc:.4f}[/bold]",
        f"[bold]{macro_pr:.4f}[/bold]",
    )
    # --- End of Modification ---

    console.print(task_table)

    # --- Scalers ---
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
        scaled_norm = val * norm
        scale_table.add_row(
            scale_key,
            f"{val:.4f}",
            f"{norm:.4f}",
            f"{scaled_norm:.4f}",
            get_colored_delta(val, history[scale_key]),
        )
    console.print(scale_table)
