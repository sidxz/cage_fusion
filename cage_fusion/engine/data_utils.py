import torch
from torch.nn.utils.rnn import pad_sequence
from chemprop.data import BatchMolGraph


def collate_fn_for_cage_fusion(batch):
    """
    Custom collate function for batching data samples used in the CAGEFusion model.

    This function performs the following:
    - Constructs a batched molecular graph
    - Pads token embeddings and input IDs to uniform sequence length
    - Computes attention masks based on actual sequence lengths
    - Stacks auxiliary features and labels for downstream processing

    Args:
        batch (list of tuples): Each tuple contains:
            - mol_graph: Molecular graph representation
            - embeddings: Token-level embeddings (Tensor)
            - aux_features: Auxiliary feature vector (Tensor)
            - labels: Target labels (Tensor)
            - input_ids: Token IDs from the tokenizer (Tensor)

    Returns:
        tuple: A batch consisting of:
            - Batched molecular graph
            - Padded token embeddings
            - Attention masks
            - Stacked auxiliary features
            - Stacked labels
            - Padded input IDs
    """
    try:
        # Unzip batch into individual components
        mol_graphs, embeddings, aux_features, labels, input_ids_list = zip(*batch)

        # Create a batched molecular graph
        batched_graph = BatchMolGraph(list(mol_graphs))

        # Pad the embeddings and input IDs to match the longest sequence
        padded_embeddings = pad_sequence(
            [e.contiguous() for e in embeddings], batch_first=True
        )

        pad_token_id = 0
        padded_input_ids = pad_sequence(
            [ids.contiguous() for ids in input_ids_list],
            batch_first=True,
            padding_value=pad_token_id,
        )

        # Compute sequence lengths (excluding padding)
        seq_lengths = torch.tensor(
            [(ids != pad_token_id).sum() for ids in input_ids_list]
        )

        # Create boolean attention masks where True indicates non-padded tokens
        max_len = padded_embeddings.size(1)
        attn_mask = torch.arange(max_len).unsqueeze(0) < seq_lengths.unsqueeze(1)
        attn_mask = attn_mask.to(dtype=torch.bool)

        # Stack auxiliary features and labels into tensors
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

    except Exception as e:
        raise RuntimeError(f"Error during batch collation: {e}")
