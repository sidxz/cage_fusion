import os
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from chemprop.data import BatchMolGraph
from cage_fusion.utils.logging_utils import logger


def move_bmg_to_device(bmg: BatchMolGraph, device: torch.device) -> BatchMolGraph:
    """
    Transfers a BatchMolGraph object to the specified device.

    Args:
        bmg (BatchMolGraph): Graph object with V, E, edge_index, and batch tensors.
        device (torch.device): Target device (e.g., torch.device("cuda")).

    Returns:
        BatchMolGraph: Graph object with tensors moved to device.
    """
    for attr in ["V", "E", "edge_index", "batch"]:
        setattr(bmg, attr, getattr(bmg, attr).to(device))
    return bmg


def visualize_attention_weights(
    attn_weights: torch.Tensor,
    mask: torch.Tensor,
    num_heads: int,
    output_path: str,
    input_ids: torch.Tensor = None,
    tokenizer_obj=None,
):
    """
    Visualizes attention weights across heads and saves heatmaps and distributions.

    Args:
        attn_weights (torch.Tensor): [num_heads, seq_len] or [1, num_heads, seq_len] attention weights.
        mask (torch.Tensor): Boolean tensor marking valid tokens.
        num_heads (int): Number of attention heads.
        output_path (str): Path to save the visualization.
        input_ids (torch.Tensor, optional): Token IDs for label display.
        tokenizer_obj (Tokenizer, optional): Tokenizer for token-to-string conversion.
    """
    if attn_weights.ndim == 3 and attn_weights.shape[1] > 1:
        attn_weights = torch.mean(attn_weights, dim=1)

    seq_len = int(mask.sum().item())
    attn_weights = attn_weights[:, :seq_len]
    attn_np = torch.clamp(attn_weights, min=0.0).cpu().numpy()

    normalized_heads = []
    for head in attn_np:
        if np.isnan(head).all():
            normalized_heads.append(np.full_like(head, np.nan))
        else:
            normalized_heads.append(head / (head.sum() + 1e-8))
    normalized_heads = np.array(normalized_heads)

    averaged_attn = np.nanmean(normalized_heads, axis=0)
    if np.isnan(averaged_attn).all():
        logger.warning("Skipped attention visualization: All values are NaN.")
        return

    xtick_labels = None
    if tokenizer_obj and input_ids is not None:
        tokens = tokenizer_obj.convert_ids_to_tokens(input_ids.cpu().numpy()[:seq_len])
        tick_skip = 2 if seq_len > 100 else 1
        xtick_labels = [
            tok if i % tick_skip == 0 else "" for i, tok in enumerate(tokens)
        ]

    # --- Plotting ---
    num_rows = 1 + num_heads
    fig, axs = plt.subplots(
        num_rows, 2, figsize=(20, 3.5 * num_rows), gridspec_kw={"width_ratios": [4, 2]}
    )
    axs = np.array(axs).reshape(num_rows, 2)

    def plot_head(ax_row, data, title):
        sns.heatmap(
            data.reshape(1, -1), cmap="viridis", ax=ax_row[0], xticklabels=xtick_labels
        )
        ax_row[0].set_title(title)
        ax_row[1].hist(data, bins=20, color="skyblue")
        ax_row[1].set_title(f"{title} Distribution")

    plot_head(axs[0], averaged_attn, "Averaged Attention")

    for i, head_data in enumerate(normalized_heads):
        if not np.isnan(head_data).all():
            plot_head(axs[i + 1], head_data, f"Head {i}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    #logger.info(f"Saved attention visualization to: {output_path}")
    
    
    
    
    # file: cage_fusion/engine/utils.py

import torch
import h5py
from typing import Tuple

def compute_pos_weight_from_h5(
    h5_path: str,
    chunk_size: int = 10_000,
    epsilon: float = 1e-6,
    verbose: bool = True
) -> torch.Tensor:
    """
    Stream and compute class-wise pos_weight from labels in an HDF5 file.
    
    Args:
        h5_path (str): Path to the HDF5 file containing a 'labels' dataset.
        chunk_size (int): Number of rows to read at once.
        epsilon (float): Small constant to avoid division by zero.
        verbose (bool): Whether to print counts and weights.

    Returns:
        torch.Tensor: pos_weight tensor of shape [num_classes]
    """
    with h5py.File(h5_path, "r") as f:
        labels_dset = f["labels"]
        num_samples, num_classes = labels_dset.shape

        pos_counts = torch.zeros(num_classes, dtype=torch.float64)
        neg_counts = torch.zeros(num_classes, dtype=torch.float64)

        for i in range(0, num_samples, chunk_size):
            labels = torch.tensor(labels_dset[i:i+chunk_size], dtype=torch.float32)
            pos_counts += labels.sum(dim=0)
            neg_counts += (1.0 - labels).sum(dim=0)

    pos_weight = (neg_counts / (pos_counts + epsilon)).to(torch.float32)

    if verbose:
        logger.info(f"Positive counts per class: {pos_counts.tolist()}")
        logger.info(f"Negative counts per class: {neg_counts.tolist()}")
        logger.info(f"Calculated pos_weight: {pos_weight.tolist()}")

    return pos_weight

