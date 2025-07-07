# FILE: cage_fusion/pretrain/utils.py
import torch
import random
import torch.nn.functional as F


def augment_graph(data, node_mask_rate=0.15, edge_drop_rate=0.15):
    """
    Creates an augmented view of a graph by masking node features and dropping edges.
    """
    new_data = data.clone()
    num_nodes = new_data.num_nodes

    # Node Feature Masking: Replace some node features with a special "mask" token.
    # The ZINC dataset has 28 atom types, so we use index 28 as our mask token.
    mask_feature_index = 28
    mask = torch.rand(num_nodes) < node_mask_rate
    new_data.x[mask, 0] = mask_feature_index

    # Edge Dropping: Randomly remove some edges.
    num_edges = new_data.num_edges
    edges_to_keep = torch.rand(num_edges) > edge_drop_rate
    new_data.edge_index = new_data.edge_index[:, edges_to_keep]

    return new_data


def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Calculates the NT-Xent loss for contrastive learning.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(
        representations.unsqueeze(1), representations.unsqueeze(0), dim=2
    )

    batch_size = z1.shape[0]
    labels = torch.cat([torch.arange(batch_size) for i in range(2)]).to(
        similarity_matrix.device
    )
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(similarity_matrix.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(similarity_matrix.device)

    logits = logits / temperature
    return F.cross_entropy(logits, labels)
