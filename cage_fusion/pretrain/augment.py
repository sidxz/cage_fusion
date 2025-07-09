# === cage_fusion/pretrain/augment.py ===
import torch
import random
from torch_geometric.data import Data
import copy


def mask_node_features(data: Data, mask_rate=0.15):
    x = data.x.clone()
    num_nodes = x.size(0)
    num_mask = int(mask_rate * num_nodes)
    mask_idx = random.sample(range(num_nodes), num_mask)
    x[mask_idx] = 0  # TODO: replace with learnable mask token if needed
    data.x = x
    return data


def drop_edges(data: Data, drop_rate=0.2):
    edge_index = data.edge_index.clone()
    num_edges = edge_index.size(1)
    num_drop = int(drop_rate * num_edges)
    if num_drop == 0:
        return data
    keep_idx = torch.randperm(num_edges)[: num_edges - num_drop]
    data.edge_index = edge_index[:, keep_idx]
    return data


def random_graph_augment(data: Data) -> Data:
    # Clone to avoid in-place ops
    data = copy.deepcopy(data)

    # Randomly select 1–2 augmentations
    if random.random() < 0.5:
        data = mask_node_features(data)
    if random.random() < 0.5:
        data = drop_edges(data)
    return data
