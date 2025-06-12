import os
import torch
import numpy as np
from tqdm import tqdm
from cage_fusion.utils.logging_utils import logger
from .metrics import (
    AUCBatchAggregatorToDisk,
    MCCBatchAggregatorToDisk,
    PRBatchAggregatorToDisk,
)
from .utils import move_bmg_to_device, visualize_attention_weights


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    criterion,
    device,
    num_tasks,
    label_names,
    threshold_search=np.linspace(0.1, 0.9, 20),
    return_thresholds=True,
    plot_attn=False,
    cache_dir="val_cache",
    tokenizer_obj=None,
):
    """
    Evaluates the model using batched metrics aggregation with minimal memory overhead.

    Args:
        model (nn.Module): Trained model instance.
        loader (DataLoader): DataLoader for validation or test data.
        criterion (callable): Loss function.
        device (torch.device): Target computation device (CPU/GPU).
        num_tasks (int): Number of classification tasks.
        label_names (List[str]): Task names for logging and plotting.
        threshold_search (np.ndarray): Thresholds for MCC evaluation.
        return_thresholds (bool): Whether to return optimal per-task thresholds.
        plot_attn (bool): Whether to plot attention maps for the first batch.
        cache_dir (str): Path to store cached predictions and plots.
        tokenizer_obj (PreTrainedTokenizer): Tokenizer for attention plots.

    Returns:
        Tuple:
            - avg_loss (float): Average evaluation loss.
            - avg_mcc (float): Average Matthews correlation coefficient.
            - avg_auc (float): Average AUC score.
            - avg_pr (float): Average Precision-Recall AUC.
            - best_thresholds (np.ndarray): Optimal thresholds per task.
            - per_task_metrics (List[Tuple[float]]): MCC, AUC, PR per task.
    """
    model.eval()
    total_loss = 0.0
    has_plotted = False

    # Initialize disk-backed metric aggregators
    mcc_agg = MCCBatchAggregatorToDisk(
        num_tasks, cache_dir=os.path.join(cache_dir, "mcc"), label_names=label_names
    )
    auc_agg = AUCBatchAggregatorToDisk(
        num_tasks, cache_dir=os.path.join(cache_dir, "auc"), label_names=label_names
    )
    pr_agg = PRBatchAggregatorToDisk(
        num_tasks, cache_dir=os.path.join(cache_dir, "pr"), label_names=label_names
    )

    for batch in tqdm(loader, desc="Evaluating"):
        bmg, token_embs, attn_mask, rdkit_feats, labels, input_ids_batch = batch

        # Transfer tensors to device
        bmg = move_bmg_to_device(bmg, device)
        token_embs = token_embs.to(device)
        attn_mask = attn_mask.to(device)
        rdkit_feats = rdkit_feats.to(device)
        labels = labels.to(device)
        input_ids_batch = input_ids_batch.to(device)

        return_attn = plot_attn and not has_plotted

        # Model inference
        outputs = model(
            bmg,
            token_embs,
            attn_mask,
            rdkit_feats,
            input_ids_batch,
            return_attn=return_attn,
        )

        if return_attn:
            logits, _, _, attn_weights, _, _ = outputs
            if attn_weights is not None:
                plot_dir = os.path.join(cache_dir, "attention_plots")
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, "evaluation_attention.png")
                visualize_attention_weights(
                    attn_weights[0],
                    attn_mask[0],
                    model.num_heads,
                    output_path=plot_path,
                    input_ids=input_ids_batch[0],
                    tokenizer_obj=tokenizer_obj,
                )
            has_plotted = True
        else:
            logits, _, _ = outputs

        # Compute loss and update metrics
        loss = criterion(logits, labels)
        total_loss += loss.item()
        probs = torch.sigmoid(logits)

        mcc_agg.update(labels, probs)
        auc_agg.update(labels, probs)
        pr_agg.update(labels, probs)

    # Final aggregation of metrics
    avg_loss = total_loss / len(loader)
    avg_mcc, best_thresholds, per_task_mcc = mcc_agg.compute(
        threshold_search=threshold_search
    )
    per_task_auc = auc_agg.compute(reduce="none")
    per_task_pr = pr_agg.compute(reduce="none")

    avg_auc = float(np.nanmean(per_task_auc))
    avg_pr = float(np.mean(per_task_pr))
    per_task_metrics = list(zip(per_task_mcc, per_task_auc, per_task_pr))

    # logger.info(f"Evaluation complete: loss={avg_loss:.4f} | MCC={avg_mcc:.4f} | AUC={avg_auc:.4f} | PR-AUC={avg_pr:.4f}")

    if return_thresholds:
        return (
            avg_loss,
            avg_mcc,
            avg_auc,
            avg_pr,
            np.array(best_thresholds),
            per_task_metrics,
        )
    else:
        return avg_loss, avg_mcc, avg_auc, avg_pr
