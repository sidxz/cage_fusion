# collate.py
import torch
from chemprop.data import BatchMolGraph


def collate_fn_for_cage_fusion(batch, pad_token_id):
    if pad_token_id is None:
        pad_token_id = 0

    (
        mol_graphs,
        embeddings,
        aux_features,
        labels,
        input_ids_list,
        smiles,
        original_indices,
        *maybe_ids,  # list[str] or None from dataset
    ) = zip(*batch)

    # Build batched graph in worker
    batched_graph = BatchMolGraph(list(mol_graphs))

    # Tensors: already fixed length -> just stack
    embeddings = torch.stack(embeddings)  # [B, T, D]
    input_ids = torch.stack(input_ids_list)  # [B, T]
    attn_mask = input_ids != pad_token_id  # bool [B, T]
    aux_features = torch.stack(aux_features)  # [B, F]

    # Labels may be empty (inference)
    if labels and labels[0].numel() > 0:
        labels = torch.stack(labels)  # [B, L]
    else:
        labels = torch.empty((len(batch), 0), dtype=torch.float32)

    smiles = list(smiles)
    original_indices = torch.as_tensor(original_indices, dtype=torch.long)

    # Optional ids: dataset should already return str or None
    ids_list = list(maybe_ids[0]) if maybe_ids else [None] * len(batch)

    return (
        batched_graph,
        embeddings,
        attn_mask,
        aux_features,
        labels,
        input_ids,
        smiles,
        original_indices,
        ids_list,
    )
