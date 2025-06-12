import os
import torch
import numpy as np
from tqdm import tqdm

# Import from other engine modules
from .metrics import AUCBatchAggregatorToDisk, MCCBatchAggregatorToDisk, PRBatchAggregatorToDisk
from .utils import move_bmg_to_device, visualize_attention_weights

@torch.no_grad()
def evaluate_model(
    model, loader, criterion, device, num_tasks, label_names,
    threshold_search=np.linspace(0.1, 0.9, 20), return_thresholds=True,
    plot_attn=False, cache_dir="val_cache", tokenizer_obj=None
):
    """
    Evaluates the model on a given dataset using memory-efficient metric aggregators.
    """
    model.eval()
    total_loss = 0.0
    has_plotted = False

    # Initialize metric aggregators
    mcc_agg = MCCBatchAggregatorToDisk(num_tasks=num_tasks, cache_dir=f"{cache_dir}/mcc", label_names=label_names)
    auc_agg = AUCBatchAggregatorToDisk(num_tasks=num_tasks, cache_dir=f"{cache_dir}/auc", label_names=label_names)
    pr_agg  = PRBatchAggregatorToDisk(num_tasks=num_tasks, cache_dir=f"{cache_dir}/pr", label_names=label_names)

    for bmg, token_embs, attn_mask, rdkit_feats, labels, input_ids_batch in tqdm(loader, desc="🧪 Evaluating"):
        # Move data to device
        bmg = move_bmg_to_device(bmg, device)
        token_embs, attn_mask = token_embs.to(device), attn_mask.to(device)
        rdkit_feats, labels = rdkit_feats.to(device), labels.to(device)
        input_ids_batch = input_ids_batch.to(device)
        return_attn = plot_attn and not has_plotted
        
        # Forward pass
        model_output = model(bmg, token_embs, attn_mask, rdkit_feats, input_ids_batch, return_attn=return_attn)
        
        # CORRECTED: Unpack the model output correctly in both scenarios
        if return_attn:
            # Unpack all 6 values when attention is returned
            logits, _, _, attn_weights, _, _ = model_output
            if attn_weights is not None:
                plot_dir = os.path.join(cache_dir, "attention_plots")
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, "evaluation_attention.png")
                visualize_attention_weights(
                    attn_weights[0], attn_mask[0], model.num_heads,
                    output_path=plot_path,
                    input_ids=input_ids_batch[0], tokenizer_obj=tokenizer_obj
                )
            has_plotted = True
        else:
            # Unpack the 3 values returned when attention is not requested
            logits, _, _ = model_output

        # Calculate loss and update metric aggregators
        loss = criterion(logits, labels)
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        mcc_agg.update(labels, probs)
        auc_agg.update(labels, probs)
        pr_agg.update(labels, probs)

    # Compute final metrics from aggregators
    avg_loss = total_loss / len(loader)
    avg_mcc, best_thresholds, per_task_mcc = mcc_agg.compute(threshold_search=threshold_search)
    per_task_auc = auc_agg.compute(reduce="none")
    per_task_pr  = pr_agg.compute(reduce="none")
    avg_auc = float(np.nanmean(per_task_auc))
    avg_pr  = float(np.mean(np.array(per_task_pr)))
    per_task_metrics = list(zip(per_task_mcc, per_task_auc, per_task_pr))

    if return_thresholds:
        return avg_loss, avg_mcc, avg_auc, avg_pr, np.array(best_thresholds), per_task_metrics
    else:
        return avg_loss, avg_mcc, avg_auc, avg_pr
