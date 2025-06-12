import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Generates a multi-panel plot to visualize the training history.
    """
    epochs = range(1, len(history['val_loss']) + 1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 20), sharex=True)
    fig.suptitle('Model Training History', fontsize=20, weight='bold')

    # Panel 1: Loss
    ax1.plot(epochs, history['train_loss'], 'o--', color='dodgerblue', label='Train Loss', lw=2)
    ax1.plot(epochs, history['val_loss'], 'o-', color='darkorange', label='Validation Loss', lw=2)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.set_title('Training and Validation Loss', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=12)

    # Panel 2: Key Performance Metrics (AUC & MCC)
    ax2.plot(epochs, history['val_auc'], 'o-', color='green', label='Validation AUC', lw=2)
    ax2.plot(epochs, history['val_mcc'], 'o-', color='purple', label='Validation MCC', lw=2)
    ax2.set_ylabel('Score', fontsize=14)
    ax2.set_title('Validation Performance Metrics', fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=12)
    ax2.set_ylim(bottom=max(0, np.min(history['val_auc'] + history['val_mcc']) - 0.1))

    # Panel 3: Learned Modality Scalers
    ax3.plot(epochs, history['scale_graph'], 'o-', color='crimson', label='Graph Scale', lw=2)
    ax3.plot(epochs, history['scale_attn'], 'o-', color='teal', label='Attention Scale', lw=2)
    ax3.plot(epochs, history['scale_aux'], 'o-', color='gold', label='Auxiliary Scale', lw=2)
    ax3.set_ylabel('Scaler Value', fontsize=14)
    ax3.set_title('Learned Modality Importance Scalers', fontsize=16)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(fontsize=12)

    ax3.set_xlabel('Epoch', fontsize=14)
    
    x_ticks = np.arange(1, len(history['val_loss']) + 1, step=max(1, (len(history['val_loss']) // 15)))
    plt.xticks(ticks=x_ticks)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def log_epoch_results(epoch, num_epochs, history, label_names, per_task_metrics):
    """
    Handles the logging of all metrics for a given epoch, including deltas.
    """
    print(f"\n--- Epoch {epoch}/{num_epochs} Summary ---")

    def get_delta_str(current_val, history_list, is_loss=False):
        if len(history_list) < 2:
            return ""
        prev_val = history_list[-2]
        delta = current_val - prev_val
        
        color_green, color_red, color_end = '\033[92m', '\033[91m', '\033[0m'
        is_improvement = (delta < 0) if is_loss else (delta > 0)
        color = color_green if is_improvement else color_red
        
        return f" ({color}{delta:+.4f}{color_end})"

    # Overall Metrics
    train_loss, val_loss = history["train_loss"][-1], history["val_loss"][-1]
    val_mcc, val_auc = history["val_mcc"][-1], history["val_auc"][-1]
    
    val_loss_delta = get_delta_str(val_loss, history["val_loss"], is_loss=True)
    val_mcc_delta = get_delta_str(val_mcc, history["val_mcc"])
    val_auc_delta = get_delta_str(val_auc, history["val_auc"])
    
    print(f"🔻 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{val_loss_delta}")
    print(f"🧪 Val Metrics | ✅ MCC: {val_mcc:.4f}{val_mcc_delta} | 📈 AUC: {val_auc:.4f}{val_auc_delta}")

    # Per-Task Metrics
    print("🔍 Per-task metrics (Validation):")
    prev_per_task = history["per_task"][-2] if len(history["per_task"]) > 1 else None
    for i, (task_mcc, task_auc, task_pr) in enumerate(per_task_metrics):
        task_name = label_names[i] if label_names and i < len(label_names) else f"Task {i}"
        mcc_delta_str = get_delta_str(task_mcc, [prev[i][0] for prev in history["per_task"] if prev]) if prev_per_task else ""
        auc_delta_str = get_delta_str(task_auc, [prev[i][1] for prev in history["per_task"] if prev]) if prev_per_task else ""
        print(f"  • {task_name:25s} | AUC: {task_auc:.3f}{auc_delta_str} | MCC: {task_mcc:.3f}{mcc_delta_str}")

    # Modality Scalers
    print("🧮 Learned modality scalers:")
    scale_graph, scale_attn, scale_aux = history["scale_graph"][-1], history["scale_attn"][-1], history["scale_aux"][-1]
    sg_delta = get_delta_str(scale_graph, history["scale_graph"])
    sa_delta = get_delta_str(scale_attn, history["scale_attn"])
    sr_delta = get_delta_str(scale_aux, history["scale_aux"])
    print(f"   scale_graph = {scale_graph:.4f}{sg_delta}")
    print(f"   scale_attn  = {scale_attn:.4f}{sa_delta}")
    print(f"   scale_aux   = {scale_aux:.4f}{sr_delta}")
