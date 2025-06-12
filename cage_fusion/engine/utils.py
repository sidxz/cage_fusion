import os
import torch
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for script-based execution
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from chemprop.data import BatchMolGraph

def move_bmg_to_device(bmg: BatchMolGraph, device: torch.device) -> BatchMolGraph:
    """Moves all tensor components of a BatchMolGraph object to the specified device."""
    bmg.V = bmg.V.to(device)
    bmg.E = bmg.E.to(device)
    bmg.edge_index = bmg.edge_index.to(device)
    bmg.batch = bmg.batch.to(device)
    return bmg

def visualize_attention_weights(attn_weights: torch.Tensor, mask: torch.Tensor, num_heads: int,
                                  output_path: str,
                                  input_ids: torch.Tensor = None, tokenizer_obj=None):
    """
    Visualizes attention weights and saves the plot to a file.
    """
    if attn_weights.ndim == 3 and attn_weights.shape[1] > 1:
        attn_weights = torch.mean(attn_weights, dim=1)
    
    mask = mask.cpu().numpy()
    seq_len = int(mask.sum())
    
    attn_weights = attn_weights[:, :seq_len]
    attn_weights_np = torch.clamp(attn_weights, min=0.0).cpu().numpy()

    normalized_attn_heads = []
    for h_idx in range(num_heads):
        head_attn = attn_weights_np[h_idx]
        if np.isnan(head_attn).all():
            normalized_attn_heads.append(np.full_like(head_attn, np.nan))
            continue
        normalized_attn_heads.append(head_attn / (head_attn.sum() + 1e-8))

    normalized_attn_heads = np.array(normalized_attn_heads)
    averaged_attn = np.nanmean(normalized_attn_heads, axis=0)

    if np.isnan(averaged_attn).all():
        print("⚠️ Skipping visualization: All averaged values are NaN.")
        return

    xtick_labels, tokens_in_sample = None, []
    if input_ids is not None and tokenizer_obj is not None:
        tokens_in_sample = tokenizer_obj.convert_ids_to_tokens(input_ids.cpu().numpy()[:seq_len])
        tick_skip = 2 if seq_len > 100 else 1
        xtick_labels = [tok if i % tick_skip == 0 else '' for i, tok in enumerate(tokens_in_sample)]

    num_rows = 1 + num_heads
    fig, axs = plt.subplots(num_rows, 2, figsize=(20, 3.5 * num_rows), gridspec_kw={'width_ratios': [4, 2]})
    if num_rows == 1: axs = np.expand_dims(axs, axis=0)

    sns.heatmap(averaged_attn.reshape(1, -1), cmap="viridis", ax=axs[0, 0], xticklabels=xtick_labels)
    axs[0, 0].set_title("Averaged Attention Heatmap")
    axs[0, 1].hist(averaged_attn, bins=20, color="skyblue")
    axs[0, 1].set_title("Averaged Attention Distribution")

    for h_idx in range(num_heads):
        head_attn = normalized_attn_heads[h_idx]
        if np.isnan(head_attn).all(): continue
        sns.heatmap(head_attn.reshape(1, -1), cmap="viridis", ax=axs[h_idx + 1, 0], xticklabels=xtick_labels)
        axs[h_idx + 1, 0].set_title(f"Head {h_idx} Attention")
        axs[h_idx + 1, 1].hist(head_attn, bins=20, color="lightcoral")
        axs[h_idx + 1, 1].set_title(f"Head {h_idx} Distribution")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory
    print(f"📊 Attention plot saved to: {output_path}")

def collate_fn_for_cage_fusion(batch):
    (mol_graphs, embeddings, aux_features, labels, input_ids_list) = zip(*batch)
    batched_graph = BatchMolGraph(list(mol_graphs))
    padded_embeddings = pad_sequence([e.contiguous() for e in embeddings], batch_first=True)
    pad_token_id = 0 
    padded_input_ids = pad_sequence([ids.contiguous() for ids in input_ids_list], batch_first=True, padding_value=pad_token_id)
    seq_lens = torch.tensor([(ids != pad_token_id).sum() for ids in input_ids_list])
    max_len = padded_embeddings.size(1)
    attn_mask = torch.arange(max_len).unsqueeze(0) < seq_lens.unsqueeze(1)
    attn_mask = attn_mask.to(dtype=torch.bool)
    aux_features_tensor = torch.stack(aux_features)
    labels_tensor = torch.stack(labels)
    return batched_graph, padded_embeddings, attn_mask, aux_features_tensor, labels_tensor, padded_input_ids
