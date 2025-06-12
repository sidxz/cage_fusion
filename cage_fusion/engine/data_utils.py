import torch
from torch.nn.utils.rnn import pad_sequence
from chemprop.data import BatchMolGraph

def collate_fn_for_cage_fusion(batch):
    (mol_graphs, embeddings, aux_features, labels, input_ids_list) = zip(*batch)
    batched_graph = BatchMolGraph(list(mol_graphs))
    padded_embeddings = pad_sequence(
        [e.contiguous() for e in embeddings], batch_first=True
    )
    pad_token_id = 0
    padded_input_ids = pad_sequence(
        [ids.contiguous() for ids in input_ids_list],
        batch_first=True,
        padding_value=pad_token_id,
    )
    seq_lens = torch.tensor([(ids != pad_token_id).sum() for ids in input_ids_list])
    max_len = padded_embeddings.size(1)
    attn_mask = torch.arange(max_len).unsqueeze(0) < seq_lens.unsqueeze(1)
    attn_mask = attn_mask.to(dtype=torch.bool)
    aux_features_tensor = torch.stack(aux_features)
    labels_tensor = torch.stack(labels)
    return (
        batched_graph,
        padded_embeddings,
        attn_mask,
        aux_features_tensor,
        labels_tensor,
        padded_input_ids,
    )
