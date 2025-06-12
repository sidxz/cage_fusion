import os
import torch
import numpy as np
import random
from tqdm import tqdm
from cage_fusion.utils.logging_utils import logger
from .metrics import (
    AUCBatchAggregatorToDisk,
    MCCBatchAggregatorToDisk,
    PRBatchAggregatorToDisk,
)
from .utils import move_bmg_to_device, visualize_attention_weights


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scheduler,
    device,
    num_tasks,
    cache_dir=None,
    tokenizer_obj=None,
    log_attn_random_batch: bool = True,
    lambda_entropy: float = 0.0,
    lambda_prior: float = 0.0,
    label_names=None,
):
    """
    Runs a single training epoch with logging and streaming metric computation.

    Args:
        model (torch.nn.Module): The model being trained.
        loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer instance.
        criterion (callable): Primary loss function.
        scheduler (_LRScheduler): Learning rate scheduler.
        device (torch.device): Device to run the training on.
        num_tasks (int): Number of prediction tasks.
        cache_dir (str): Path to store temporary metric data.
        tokenizer_obj (Tokenizer): Tokenizer for attention visualization.
        log_attn_random_batch (bool): Enable attention visualization for one batch.
        lambda_entropy (float): Weight for attention entropy regularization.
        lambda_prior (float): Weight for token prior regularization.
        label_names (List[str]): Optional names for each task.

    Returns:
        Tuple: (avg_loss, avg_mcc, avg_auc, avg_pr)
    """
    model.train()
    total_loss = 0.0
    has_logged_attention = False

    # Metric aggregators (streaming to disk)
    mcc_agg = MCCBatchAggregatorToDisk(
        num_tasks, os.path.join(cache_dir, "mcc_train"), label_names
    )
    auc_agg = AUCBatchAggregatorToDisk(
        num_tasks, os.path.join(cache_dir, "auc_train"), label_names
    )
    pr_agg = PRBatchAggregatorToDisk(
        num_tasks, os.path.join(cache_dir, "pr_train"), label_names
    )

    # Pick a random batch for attention visualization
    batch_to_log_idx = (
        random.randint(0, len(loader) - 1)
        if log_attn_random_batch and len(loader) > 0
        else -1
    )
    if batch_to_log_idx >= 0:
        logger.info(
            f"Attention visualization will be logged for batch index: {batch_to_log_idx}"
        )

    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        if batch is None:
            logger.warning(f"Batch {batch_idx} is None. Skipping.")
            continue

        bmg, token_embs, attn_mask, aux_feats, labels, input_ids = batch

        # Move tensors to device
        bmg = move_bmg_to_device(bmg, device)
        token_embs = token_embs.to(device)
        attn_mask = attn_mask.to(device)
        aux_feats = aux_feats.to(device)
        labels = labels.to(device)
        input_ids = input_ids.to(device)

        optimizer.zero_grad()

        output = model(
            bmg,
            token_embs,
            attn_mask,
            aux_feats,
            input_ids_batch=input_ids,
            return_attn=True,
        )

        if output is None:
            raise ValueError(f"Model returned None for batch {batch_idx}")

        (
            logits,
            attn_entropy_loss,
            token_prior_loss,
            attn_weights,
            attn_output,
            graph_repr,
        ) = output

        loss = criterion(logits, labels)
        loss += lambda_entropy * attn_entropy_loss
        loss += lambda_prior * token_prior_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        # Detach predictions for metric logging
        probs = torch.sigmoid(logits).detach()
        mcc_agg.update(labels.detach(), probs)
        auc_agg.update(labels.detach(), probs)
        pr_agg.update(labels.detach(), probs)

        # Attention visualization
        if (
            batch_idx == batch_to_log_idx
            and not has_logged_attention
            and attn_weights is not None
        ):
            random_idx = random.randint(0, labels.size(0) - 1)
            plot_dir = os.path.join(cache_dir, "attention_plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(
                plot_dir, f"batch_{batch_idx}_idx_{random_idx}.png"
            )

            visualize_attention_weights(
                attn_weights[random_idx].detach(),
                attn_mask[random_idx],
                model.num_heads,
                output_path=plot_path,
                input_ids=input_ids[random_idx],
                tokenizer_obj=tokenizer_obj,
            )

            logger.debug(f"Graph norm: {graph_repr.norm(dim=1).mean():.4f}")
            logger.debug(f"Attention norm: {attn_output.norm(dim=1).mean():.4f}")
            logger.debug(
                f"Modality scalers: "
                f"scale_graph={model.scale_graph.item():.4f}, "
                f"scale_attn={model.scale_attn.item():.4f}, "
                f"scale_aux={model.scale_aux.item():.4f}"
            )

            has_logged_attention = True

    # Final metric aggregation
    avg_loss = total_loss / len(loader)
    avg_mcc, *_ = mcc_agg.compute()
    avg_auc = auc_agg.compute()
    avg_pr = pr_agg.compute()

    logger.info(
        f"Training complete. Avg Loss: {avg_loss:.4f}, MCC: {avg_mcc:.4f}, AUC: {avg_auc:.4f}, PR-AUC: {avg_pr:.4f}"
    )
    return avg_loss, avg_mcc, avg_auc, avg_pr
