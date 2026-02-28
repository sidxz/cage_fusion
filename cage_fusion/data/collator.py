"""
DataLoader collate function for cage_fusion batches.
"""

import torch
from chemprop.data import BatchMolGraph


def collate_cage_fusion(batch, pad_token_id: int = 0):
    """
    Collate a list of dataset items into model-ready tensors.

    Each item is the 8-tuple returned by ``CageFusionStreamingDataset.__getitem__``.

    Returns
    -------
    batched_graph : BatchMolGraph
    embeddings    : Tensor [B, T, D]
    attn_mask     : BoolTensor [B, T]   True = real token
    aux_features  : Tensor [B, F]
    labels        : Tensor [B, L]  (empty when no labels)
    input_ids     : Tensor [B, T]
    smiles        : List[str]
    original_indices : LongTensor [B]
    ids_list      : List[str | None]
    """
    if pad_token_id is None:
        pad_token_id = 0

    mol_graphs, embeddings, aux_features, labels, input_ids_list, \
        smiles, original_indices, *maybe_ids = zip(*batch)

    batched_graph = BatchMolGraph(list(mol_graphs))
    embeddings = torch.stack(embeddings)          # [B, T, D]
    input_ids = torch.stack(input_ids_list)       # [B, T]
    attn_mask = input_ids != pad_token_id         # bool [B, T]
    aux_features = torch.stack(aux_features)      # [B, F]

    if labels and labels[0].numel() > 0:
        labels = torch.stack(labels)              # [B, L]
    else:
        labels = torch.empty((len(batch), 0), dtype=torch.float32)

    original_indices = torch.as_tensor(original_indices, dtype=torch.long)
    ids_list = list(maybe_ids[0]) if maybe_ids else [None] * len(batch)

    return (
        batched_graph,
        embeddings,
        attn_mask,
        aux_features,
        labels,
        input_ids,
        list(smiles),
        original_indices,
        ids_list,
    )
