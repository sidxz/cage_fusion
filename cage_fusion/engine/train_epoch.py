import os
import torch
import numpy as np
import random
from tqdm import tqdm

# Import from other engine modules
from .metrics import AUCBatchAggregatorToDisk, MCCBatchAggregatorToDisk, PRBatchAggregatorToDisk
from .utils import move_bmg_to_device, visualize_attention_weights

def train_one_epoch(
    model, loader, optimizer, criterion, scheduler, device, num_tasks,
    cache_dir=None, tokenizer_obj=None, log_attn_random_batch: bool = True,
    lambda_entropy: float = 0.0, lambda_prior: float = 0.0, label_names=None
):
    """
    Runs a single epoch of training for the CAGEFusionModel.
    """
    model.train()
    total_loss = 0.0
    has_logged_attention = False

    mcc_agg = MCCBatchAggregatorToDisk(num_tasks=num_tasks, cache_dir=f"{cache_dir}/mcc_train", label_names=label_names)
    auc_agg = AUCBatchAggregatorToDisk(num_tasks=num_tasks, cache_dir=f"{cache_dir}/auc_train", label_names=label_names)
    pr_agg  = PRBatchAggregatorToDisk(num_tasks=num_tasks,  cache_dir=f"{cache_dir}/pr_train", label_names=label_names)

    batch_to_log_idx = -1
    if log_attn_random_batch and len(loader) > 0:
        batch_to_log_idx = random.randint(0, len(loader) - 1)
        print(f"💡 Attention visualization for batch index: {batch_to_log_idx}")

    for batch_idx, batch_data in enumerate(tqdm(loader, desc="🔄 Training")):
        # --- Start of New Debugging ---
        print(f"\n--- Batch {batch_idx} Debug ---")
        if batch_data is None:
            print("🚨 CRITICAL: Dataloader returned None for this batch. Skipping.")
            continue
        
        bmg, token_embs, attn_mask, aux_feats, labels, input_ids_batch = batch_data
        print(f"  [Input] token_embs shape: {token_embs.shape}, labels shape: {labels.shape}")
        # --- End of New Debugging ---

        bmg = move_bmg_to_device(bmg, device)
        token_embs, attn_mask = token_embs.to(device), attn_mask.to(device)
        aux_feats, labels = aux_feats.to(device), labels.to(device)
        input_ids_batch = input_ids_batch.to(device)
        optimizer.zero_grad()
        
        model_output = model(
            bmg, token_embs, attn_mask, aux_feats,
            input_ids_batch=input_ids_batch, return_attn=True
        )

        # --- More Debugging ---
        print(f"  [Model Output] Type: {type(model_output)}")
        if model_output is None:
            raise ValueError(f"🚨 CRITICAL: Model returned None for batch {batch_idx}")
        print(f"  [Model Output] Length: {len(model_output)}")
        # --- End of Debugging ---

        logits, attn_entropy_loss, token_prior_loss, attn_weights, attn_output, graph_repr = model_output

        loss = criterion(logits, labels)
        loss += lambda_entropy * attn_entropy_loss
        loss += lambda_prior * token_prior_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        probs = torch.sigmoid(logits).detach()
        mcc_agg.update(labels.detach(), probs)
        auc_agg.update(labels.detach(), probs)
        pr_agg.update(labels.detach(), probs)

        if batch_idx == batch_to_log_idx and not has_logged_attention:
            with torch.no_grad():
                random_idx = random.randint(0, labels.shape[0] - 1)
                print(f"\nINFO: Visualizing attention for random molecule at index {random_idx} in batch {batch_idx}.")
                if attn_weights is not None:
                    plot_dir = os.path.join(cache_dir, "attention_plots")
                    os.makedirs(plot_dir, exist_ok=True)
                    plot_path = os.path.join(plot_dir, f"batch_{batch_idx}_idx_{random_idx}.png")
                    
                    visualize_attention_weights(
                        attn_weights[random_idx], attn_mask[random_idx], model.num_heads,
                        output_path=plot_path,
                        input_ids=input_ids_batch[random_idx], tokenizer_obj=tokenizer_obj
                    )
                print(f"📏 graph_repr norm: {graph_repr.norm(dim=1).mean():.4f}")
                print(f"📏 attn_output norm: {attn_output.norm(dim=1).mean():.4f}")
                print(f"🧮 Learned modality scalers:")
                print(f"   scale_graph = {model.scale_graph.item():.4f}")
                print(f"   scale_attn  = {model.scale_attn.item():.4f}")
                print(f"   scale_aux   = {model.scale_aux.item():.4f}")
            has_logged_attention = True

    avg_loss = total_loss / len(loader)
    avg_mcc, *_ = mcc_agg.compute()
    avg_auc = auc_agg.compute()
    avg_pr  = pr_agg.compute()
    return avg_loss, avg_mcc, avg_auc, avg_pr
