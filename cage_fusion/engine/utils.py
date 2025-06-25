import torch
import matplotlib
import h5py
matplotlib.use("Agg")
from chemprop.data import BatchMolGraph
from cage_fusion.utils.logging_utils import logger


def move_bmg_to_device(bmg: BatchMolGraph, device: torch.device) -> BatchMolGraph:
    """
    Transfers a BatchMolGraph object to the specified device.
    """
    for attr in ["V", "E", "edge_index", "batch"]:
        setattr(bmg, attr, getattr(bmg, attr).to(device))
    return bmg


def compute_pos_weight_from_h5(
    h5_path: str, chunk_size: int = 10_000, epsilon: float = 1e-6, verbose: bool = True
) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        labels_dset = f["labels"]
        num_samples, num_classes = labels_dset.shape

        pos_counts = torch.zeros(num_classes, dtype=torch.float64)
        neg_counts = torch.zeros(num_classes, dtype=torch.float64)

        for i in range(0, num_samples, chunk_size):
            labels = torch.tensor(labels_dset[i : i + chunk_size], dtype=torch.float32)
            pos_counts += labels.sum(dim=0)
            neg_counts += (1.0 - labels).sum(dim=0)

    pos_weight = (neg_counts / (pos_counts + epsilon)).to(torch.float32)

    if verbose:
        logger.info(f"Positive counts per class: {pos_counts.tolist()}")
        logger.info(f"Negative counts per class: {neg_counts.tolist()}")
        logger.info(f"Calculated pos_weight: {pos_weight.tolist()}")

    return pos_weight
