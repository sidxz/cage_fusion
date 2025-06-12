import numpy as np
import matplotlib.pyplot as plt
from cage_fusion.utils.logging_utils import logger


def plot_training_history(history):
    """
    Generate a 3-panel plot showing model training history including:
    - Training and validation loss
    - AUC and MCC metrics
    - Learned modality scalers (graph, attention, auxiliary)

    Args:
        history (dict): Dictionary containing lists of training statistics per epoch.
    """
    epochs = range(1, len(history["val_loss"]) + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 20), sharex=True)
    fig.suptitle("Model Training History", fontsize=20, weight="bold")

    # --- Panel 1: Loss ---
    ax1.plot(
        epochs,
        history["train_loss"],
        "o--",
        color="dodgerblue",
        label="Train Loss",
        lw=2,
    )
    ax1.plot(
        epochs,
        history["val_loss"],
        "o-",
        color="darkorange",
        label="Validation Loss",
        lw=2,
    )
    ax1.set_title("Training and Validation Loss", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # --- Panel 2: Metrics ---
    ax2.plot(
        epochs, history["val_auc"], "o-", color="green", label="Validation AUC", lw=2
    )
    ax2.plot(
        epochs, history["val_mcc"], "o-", color="purple", label="Validation MCC", lw=2
    )
    ax2.set_title("Validation Performance Metrics", fontsize=16)
    ax2.set_ylabel("Score", fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_ylim(bottom=max(0, np.min(history["val_auc"] + history["val_mcc"]) - 0.1))

    # --- Panel 3: Scalers ---
    ax3.plot(
        epochs, history["scale_graph"], "o-", color="crimson", label="Graph Scale", lw=2
    )
    ax3.plot(
        epochs, history["scale_attn"], "o-", color="teal", label="Attention Scale", lw=2
    )
    ax3.plot(
        epochs, history["scale_aux"], "o-", color="gold", label="Auxiliary Scale", lw=2
    )
    ax3.set_title("Learned Modality Importance Scalers", fontsize=16)
    ax3.set_ylabel("Scaler Value", fontsize=14)
    ax3.set_xlabel("Epoch", fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # Configure x-ticks
    step = max(1, len(epochs) // 15)
    plt.xticks(np.arange(1, len(epochs) + 1, step=step))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


from cage_fusion.utils.logging_utils import logger
from rich.console import Console
from rich.table import Table

console = Console()


def log_epoch_results(epoch, num_epochs, history, label_names, per_task_metrics):
    """
    Logs training and validation statistics at each epoch using rich formatting.

    Args:
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
        history (dict): Historical training data.
        label_names (List[str]): List of task labels.
        per_task_metrics (List[Tuple[float]]): Tuple of (MCC, AUC, PR) per task.
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
    task_table = Table(title="Per-Task Validation Metrics", header_style="bold green")
    task_table.add_column("Task")
    task_table.add_column("AUC", justify="right")
    task_table.add_column("MCC", justify="right")
    task_table.add_column("PR-AUC", justify="right")

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
    console.print(task_table)

    # --- Scalers ---
    scale_table = Table(title="Learned Modality Scalers", header_style="bold yellow")
    scale_table.add_column("Scaler")
    scale_table.add_column("Value", justify="right")
    scale_table.add_column("Δ", justify="right")

    for scale_key in ["scale_graph", "scale_attn", "scale_aux"]:
        val = history[scale_key][-1]
        scale_table.add_row(
            scale_key, f"{val:.4f}", get_colored_delta(val, history[scale_key])
        )
    console.print(scale_table)
