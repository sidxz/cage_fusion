import os
import torch
from .evaluation import evaluate_model
from .train_epoch import train_one_epoch
from .logging import log_epoch_results, plot_training_history
from cage_fusion.utils.logging_utils import logger


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device,
    num_epochs=50,
    num_tasks=6,
    base_cache_dir="metric_cache",
    label_names=None,
    tokenizer_obj=None,
    lambda_entropy=0.0,
    lambda_prior=0.01,
):
    """
    Full training loop with checkpointing, evaluation, and visual logging.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (callable): Loss function.
        scheduler (torch.optim.lr_scheduler): Scheduler.
        device (torch.device): CPU or CUDA device.
        num_epochs (int): Number of training epochs.
        num_tasks (int): Number of prediction tasks.
        base_cache_dir (str): Directory for metrics, checkpoints, and logs.
        label_names (List[str]): Optional task labels for visualization.
        tokenizer_obj (PreTrainedTokenizer): Tokenizer for attention plots.
        lambda_entropy (float): Weight for attention entropy regularization.
        lambda_prior (float): Weight for prior loss from token importance.

    Returns:
        dict: Training history containing loss and metric trends.
    """
    os.makedirs(base_cache_dir, exist_ok=True)
    checkpoint_path = os.path.join(base_cache_dir, "latest_checkpoint.pt")
    best_model_path = os.path.join(base_cache_dir, "best_model.pt")
    start_epoch = 1
    best_val_auc = -1.0

    # Resume from checkpoint if available
    if os.path.exists(checkpoint_path):
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        # CORRECTED: Added weights_only=False to allow loading of history dict
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = checkpoint["history"]
        best_val_auc = checkpoint.get("best_val_auc", -1.0)
        start_epoch = checkpoint["epoch"] + 1

        # Ensure all keys exist
        for key in ["scale_graph", "scale_attn", "scale_aux"]:
            history.setdefault(key, [])

        # Move optimizer tensors to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        logger.info("No checkpoint found. Starting training from scratch.")
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_mcc": [],
            "val_mcc": [],
            "train_auc": [],
            "val_auc": [],
            "train_pr": [],
            "val_pr": [],
            "per_task": [],
            "scale_graph": [],
            "scale_attn": [],
            "scale_aux": [],
        }

    logger.info(f"Starting training from epoch {start_epoch} to {num_epochs}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    logger.info(
        f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"{'='*25} Epoch {epoch}/{num_epochs} {'='*25}")

        train_cache = os.path.join(base_cache_dir, f"epoch_{epoch}_train")
        val_cache = os.path.join(base_cache_dir, f"epoch_{epoch}_val")

        # --- Train ---
        train_loss, train_mcc, train_auc, train_pr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            num_tasks=num_tasks,
            cache_dir=train_cache,
            tokenizer_obj=tokenizer_obj,
            log_attn_random_batch=True,
            lambda_entropy=lambda_entropy,
            lambda_prior=lambda_prior,
            label_names=label_names,
        )

        # --- Evaluate ---
        val_loss, val_mcc, val_auc, val_pr, _, per_task_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_tasks=num_tasks,
            label_names=label_names,
            plot_attn=(epoch % 5 == 0),
            cache_dir=val_cache,
            tokenizer_obj=tokenizer_obj,
        )

        # --- History ---
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mcc"].append(train_mcc)
        history["val_mcc"].append(val_mcc)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)
        history["train_pr"].append(train_pr)
        history["val_pr"].append(val_pr)
        history["per_task"].append(per_task_metrics)
        history["scale_graph"].append(model.scale_graph.item())
        history["scale_attn"].append(model.scale_attn.item())
        history["scale_aux"].append(model.scale_aux.item())

        # --- Log Results ---
        log_epoch_results(epoch, num_epochs, history, label_names, per_task_metrics)

        # --- Save Checkpoint ---
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_val_auc": best_val_auc,
        }
        torch.save(checkpoint_data, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            checkpoint_data["best_val_auc"] = best_val_auc
            torch.save(checkpoint_data, best_model_path)
            logger.info(
                f"New best model saved at {best_model_path} with AUC: {best_val_auc:.4f}"
            )

    logger.info("Training completed.")
    plot_training_history(history)
    return history
