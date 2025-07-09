# training.py
import os
import torch
from .evaluation import evaluate_model
from .train_epoch import train_one_epoch
from .logging import log_epoch_results, plot_training_history
from cage_fusion.utils.logging_utils import logger
from cage_fusion.engine.fg_utils import FG_NAMES  # Import FG_NAMES
from rich.console import Console
from rich.traceback import install

install()
console = Console()


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device,
    config,  # Pass the full config dictionary
    label_names=None,
    tokenizer_obj=None,
):
    """
    Full training loop with checkpointing, evaluation, and visual logging.
    """
    base_cache_dir = config["base_cache_dir"]
    checkpoint_dir = config["checkpoints_dir"]
    output_dir = config["output_dir"]

    num_epochs = config["num_epochs"]
    num_tasks = config["num_tasks"]
    lambda_entropy = config["lambda_entropy"]
    lambda_prior = config["lambda_prior"]

    os.makedirs(base_cache_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    start_epoch = 1
    best_val_auc = -1.0

    # Resume from checkpoint if available
    if os.path.exists(checkpoint_path):
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        resume_with_new_arch = config.get("resume_with_new_arch", False)
        if resume_with_new_arch:
            console.log(
                "[bold yellow]Resuming with new architecture. Loading compatible weights only.[/bold yellow]"
            )
            checkpoint_state_dict = checkpoint["model_state_dict"]
            current_model_state_dict = model.state_dict()

            # Create a new state dict to hold the weights we can transfer
            new_state_dict = {}

            for name, param in checkpoint_state_dict.items():
                # Check if the layer exists in the new model
                if name in current_model_state_dict:
                    # Check if the weights have the same shape
                    if current_model_state_dict[name].shape == param.shape:
                        new_state_dict[name] = param
                    else:
                        console.log(
                            f"   -> Skipping [red]{name}[/red] due to size mismatch: "
                            f"Checkpoint shape: {param.shape}, "
                            f"Model shape: {current_model_state_dict[name].shape}"
                        )

            # Load the compatible weights
            model.load_state_dict(new_state_dict, strict=False)

            console.log(
                "[bold yellow]Optimizer state not loaded due to architecture change.[/bold yellow]"
            )
            # Do NOT load the optimizer state. The optimizer must be re-initialized
            # because the model's parameters have changed.

            # You can still resume from the last epoch number
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_metric = checkpoint.get("best_metric", -1)

            # set config.get("resume_with_new_arch", True)
            config["resume_with_new_arch"] = False

            if "history" in checkpoint:
                history = checkpoint["history"]
                # Make sure all expected keys exist
                for key in [
                    "scale_graph",
                    "scale_attn",
                    "scale_aux",
                    "val_norm_graph",
                    "val_norm_attn",
                    "val_norm_aux",
                    "train_loss",
                    "val_loss",
                    "train_mcc",
                    "val_mcc",
                    "train_auc",
                    "val_auc",
                    "train_pr",
                    "val_pr",
                    "per_task",
                ]:
                    history.setdefault(key, [])
            else:
                logger.warning(
                    "Checkpoint does not contain history. Initializing empty history."
                )
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
                    "val_norm_graph": [],
                    "val_norm_attn": [],
                    "val_norm_aux": [],
                }

        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            history = checkpoint["history"]
            best_val_auc = checkpoint.get("best_val_auc", -1.0)
            start_epoch = checkpoint["epoch"] + 1

            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

            for key in [
                "scale_graph",
                "scale_attn",
                "scale_aux",
                "val_norm_graph",
                "val_norm_attn",
                "val_norm_aux",
            ]:
                history.setdefault(key, [])

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
            "val_norm_graph": [],
            "val_norm_attn": [],
            "val_norm_aux": [],
        }

    logger.info(f"Starting training from epoch {start_epoch} to {num_epochs}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    logger.info(
        f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"{'='*25} Epoch {epoch}/{num_epochs} {'='*25}")

        train_cache = os.path.join(base_cache_dir, f"epoch_{epoch}_train")
        val_cache = os.path.join(base_cache_dir, f"epoch_{epoch}_val")

        train_loss, train_mcc, train_auc, train_pr, top_fgs = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            num_tasks=num_tasks,
            cache_dir=train_cache,
            tokenizer_obj=tokenizer_obj,
            lambda_entropy=lambda_entropy,
            lambda_prior=lambda_prior,
            label_names=label_names,
        )

        if top_fgs:
            logger.info("--- Top 5 Attended Functional Groups (Overall this Epoch) ---")
            for i, (fg_id, avg_weight) in enumerate(top_fgs[:5]):
                if fg_id < len(FG_NAMES):
                    fg_name = FG_NAMES[fg_id]
                    logger.info(
                        f"  {i+1}. {fg_name:<25} | Average Attention: {avg_weight:.4f}"
                    )
                else:
                    logger.warning(
                        f"  Functional group ID {fg_id} is out of bounds for FG_NAMES."
                    )
            logger.info("-" * 65)

        (
            val_loss,
            val_mcc,
            val_auc,
            val_pr,
            best_thresholds,
            per_task_metrics,
            val_norm_graph,
            val_norm_attn,
            val_norm_aux,
        ) = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_tasks=num_tasks,
            label_names=label_names,
            plot_attn=(epoch % 1 == 0),
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
        history["val_norm_graph"].append(val_norm_graph)
        history["val_norm_attn"].append(val_norm_attn)
        history["val_norm_aux"].append(val_norm_aux)

        log_epoch_results(epoch, num_epochs, history, label_names, per_task_metrics)

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_val_auc": best_val_auc,
            "config": config,
            "best_thresholds": best_thresholds,
        }
        torch.save(checkpoint_data, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_checkpoint_data = dict(checkpoint_data)  # shallow copy is OK here
            best_checkpoint_data["best_val_auc"] = best_val_auc
            best_checkpoint_data["config"] = dict(config)

            torch.save(best_checkpoint_data, best_model_path)
            logger.info(
                f"New best model saved at {best_model_path} with AUC: {best_val_auc:.4f}"
            )

    logger.info("Training completed.")
    plot_training_history(history, output_dir=output_dir)
    return history
