import pandas as pd
import matplotlib.pyplot as plt

# Function to plot class distribution
def plot_class_distribution(df: pd.DataFrame, label_names: list, save_path: str = None):
    """
    Plot counts of positive samples (value=1) per label.
    """

    # Count positive samples for each label
    class_counts = df[label_names].sum().sort_values(ascending=False)
    print("Label frequencies:\n", class_counts)

    # Plot
    plt.figure(figsize=(10, 5))
    class_counts.plot(kind="bar")

    plt.xlabel("Labels")
    plt.ylabel("Count (value=1)")
    plt.title("Label Distribution (Positive Samples)")

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        # Print to console if no save path provided as well
        plt.show()


from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score


def plot_multilabel_confusion(y_true, y_pred, label_names, save_path=None):
    cm_all = multilabel_confusion_matrix(y_true, y_pred)
    num_labels = len(label_names)

    fig, axes = plt.subplots(1, num_labels, figsize=(4*num_labels, 4))
    if num_labels == 1:
        axes = [axes]

    for i, label in enumerate(label_names):
        cm = cm_all[i]
        ax = axes[i]
        im = ax.imshow(cm, cmap="Blues")

        ax.set_title(label)
        ax.set_xticks([0,1])
        ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])

        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r,c]), ha="center", va="center",
                        color="white" if cm[r,c] > cm.max()/2 else "black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrices to {save_path}")
    else:
        plt.show()




def find_optimal_thresholds(probs, y_true, label_names, grid=None):
    """
    Find per-label thresholds that maximize F1 on validation set.

    probs: (N, L) sigmoid outputs
    y_true: (N, L) ground truth (0/1)
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)  # 0.05, 0.10, ..., 0.95

    best_thresholds = {}

    for i, name in enumerate(label_names):
        y = y_true[:, i]

        # If label is constant, threshold doesn't matter much
        if y.sum() == 0 or y.sum() == len(y):
            best_thresholds[name] = 0.5
            continue

        best_f1 = -1.0
        best_t = 0.5

        for t in grid:
            y_pred = (probs[:, i] >= t).astype(int)
            f1 = f1_score(y, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds[name] = best_t

    return best_thresholds
