import os
import torch
from .evaluation import evaluate_model
from .train_epoch import train_one_epoch
from .logging import log_epoch_results, plot_training_history
from cage_fusion.utils.logging_utils import logger
from cage_fusion.engine.fg_utils import FG_NAMES
from rich.console import Console
from rich.traceback import install
import shutil

install()
console = Console()


def freeze_aux_and_output(model):
    """Freeze aux layers, output, fusion MLP; unfreeze rest."""
    frozen, trainable = [], []
    for name, param in model.named_parameters():
        if any(key in name for key in ["fusion_mlp", "output", "scale_aux", "aux_mlp"]):
            param.requires_grad = False
            frozen.append(name)
        else:
            param.requires_grad = True
            trainable.append(name)
    logger.info("🔒 [PHASE 1] Freezing AUX/Fusion/Output layers.")
    logger.info(f"Trainable layers:\n  - " + "\n  - ".join(trainable))
    logger.info(f"Frozen layers:\n  - " + "\n  - ".join(frozen))


def freeze_aux_and_fusion(model):
    """Freeze aux layers, fusion MLP; unfreeze rest. PHASE 1 Alternative"""
    frozen, trainable = [], []
    for name, param in model.named_parameters():
        if any(key in name for key in ["fusion_mlp", "scale_aux", "aux_mlp"]):
            param.requires_grad = False
            frozen.append(name)
        else:
            param.requires_grad = True
            trainable.append(name)
    logger.info("🔒 [PHASE 1] Freezing AUX/Fusion layers PHASE 1 Alt 1.")
    logger.info(f"Trainable layers:\n  - " + "\n  - ".join(trainable))
    logger.info(f"Frozen layers:\n  - " + "\n  - ".join(frozen))


def freeze_all_but_aux(model):
    """Freeze all except aux, fusion_mlp, output, scale_aux (AUX warmup phase)."""
    frozen, trainable = [], []
    for name, param in model.named_parameters():
        if any(key in name for key in ["fusion_mlp", "output", "scale_aux", "aux_mlp"]):
            param.requires_grad = True
            trainable.append(name)
        else:
            param.requires_grad = False
            frozen.append(name)
    logger.info("🟢 [WARMUP] Unfreezing AUX/Fusion/Output (core frozen).")
    logger.info(f"Trainable layers:\n  - " + "\n  - ".join(trainable))
    logger.info(f"Frozen layers:\n  - " + "\n  - ".join(frozen))


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    logger.info("🔓 [PHASE 2] Unfroze all model layers.")


def rebuild_optimizer(model, config):
    lr = config.get("learning_rate", 0.00052)
    wd = config.get("weight_decay", 0.0)
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device,
    config,
    label_names=None,
    tokenizer_obj=None,
):
    """
    Training loop with checkpointing, evaluation, and visual logging.
    """
    base_cache_dir = config["base_cache_dir"]
    checkpoint_dir = config["checkpoints_dir"]
    output_dir = config["output_dir"]
    num_epochs = config["num_epochs"]
    num_tasks = config["num_tasks"]
    lambda_entropy = config.get("lambda_entropy", 0)
    lambda_prior = config.get("lambda_prior", 0)

    os.makedirs(base_cache_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    best_model_mcc_path = os.path.join(checkpoint_dir, "best_model_mcc.pt")
    start_epoch = 1
    best_val_auc = -1.0
    best_val_mcc = -1.0

    # Resume logic...
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
            new_state_dict = {}
            for name, param in checkpoint_state_dict.items():
                if (
                    name in current_model_state_dict
                    and current_model_state_dict[name].shape == param.shape
                ):
                    new_state_dict[name] = param
            model.load_state_dict(new_state_dict, strict=False)
            config["resume_with_new_arch"] = False
            start_epoch = checkpoint.get("epoch", 0) + 1
            history = checkpoint.get("history", {})
            best_val_auc = checkpoint.get("best_val_auc", -1.0)
            best_val_mcc = checkpoint.get("best_val_mcc", -1.0)
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            # PATCH: Only load optimizer and scheduler state dicts if param groups match!
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info(
                    "Loaded optimizer and scheduler state dict from checkpoint."
                )
            except ValueError as e:
                logger.warning(
                    f"Skipping loading optimizer/scheduler state: {e}\n"
                    f"This is expected if parameter set changed (e.g. after freezing/unfreezing). "
                    "Optimizer was reinitialized."
                )
            history = checkpoint["history"]
            best_val_auc = checkpoint.get("best_val_auc", -1.0)
            best_val_mcc = checkpoint.get("best_val_mcc", -1.0)
            start_epoch = checkpoint["epoch"] + 1
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    else:
        history = {
            key: []
            for key in [
                "train_loss",
                "val_loss",
                "train_mcc",
                "val_mcc",
                "train_auc",
                "val_auc",
                "train_pr",
                "val_pr",
                "per_task",
                "scale_graph",
                "scale_attn",
                "scale_aux",
                "val_norm_graph",
                "val_norm_attn",
                "val_norm_aux",
            ]
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
                fg_name = FG_NAMES[fg_id] if fg_id < len(FG_NAMES) else f"FG_{fg_id}"
                logger.info(
                    f"  {i+1}. {fg_name:<25} | Average Attention: {avg_weight:.4f}"
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
        history["scale_graph"].append(
            getattr(model, "scale_graph", torch.tensor(0.0)).item()
        )
        history["scale_attn"].append(
            getattr(model, "scale_attn", torch.tensor(0.0)).item()
        )
        history["scale_aux"].append(
            getattr(model, "scale_aux", torch.tensor(0.0)).item()
        )
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
            "best_val_mcc": best_val_mcc,
            "config": config,
            "best_thresholds": best_thresholds,
        }
        torch.save(checkpoint_data, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_checkpoint_data = dict(checkpoint_data)
            best_checkpoint_data["best_val_auc"] = best_val_auc
            best_checkpoint_data["config"] = dict(config)
            torch.save(best_checkpoint_data, best_model_path)
            logger.info(
                f"New best model saved at {best_model_path} with AUC: {best_val_auc:.4f}"
            )
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_checkpoint_data = dict(checkpoint_data)
            best_checkpoint_data["best_val_mcc"] = best_val_mcc
            best_checkpoint_data["config"] = dict(config)
            torch.save(best_checkpoint_data, best_model_mcc_path)
            logger.info(
                f"New best model saved at {best_model_mcc_path} with MCC: {best_val_mcc:.4f}"
            )
    logger.info("Training completed.")
    plot_training_history(history, output_dir=output_dir)
    return history


def staged_finetune(
    model,
    train_loader,
    val_loader,
    device,
    config,
    label_names,
    tokenizer_obj,
    pos_weight,
    scheduler_fn,
    num_epochs_warmup,
    num_epochs_phase1,
    num_epochs_aux_warmup,
    num_epochs_phase2,
    freeze_fn=freeze_aux_and_fusion,
    aux_warmup_fn=freeze_all_but_aux,
    unfreeze_fn=unfreeze_all,
):
    """
    RRun 3 rounds of warmup with everything trainable
    """
    """
    Three-phase fine-tuning:
    1. Phase 1: Train core (aux/fusion/output frozen) or Alternative Phase 1 (aux/fusion/ frozen, output unfrozen)
    2. Phase 2a: AUX-only warmup (core frozen)
    3. Phase 2b: Train all (full unfreeze)
    """

    # Warmup with everything trainable
    # -- Phase 0: Warmup --
    logger.info("🔄 Warming up")
    config["num_epochs"] = num_epochs_warmup
    optimizer = rebuild_optimizer(model, config)
    scheduler = scheduler_fn(optimizer, phase="warmup")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Console().rule("[bold yellow]Phase WARMUP: Training with all layers trainable")
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        device,
        config,
        label_names,
        tokenizer_obj,
    )

    warmup_ckpt = os.path.join(config["checkpoints_dir"], "checkpoint_phase1.pt")
    latest_ckpt = os.path.join(config["checkpoints_dir"], "latest_checkpoint.pt")
    shutil.copyfile(latest_ckpt, warmup_ckpt)
    logger.info(f"[WARMUP] Saved checkpoint to {warmup_ckpt}")

    # ---- Phase 1 ----
    freeze_fn(model)
    config["num_epochs"] = num_epochs_phase1
    optimizer = rebuild_optimizer(model, config)
    scheduler = scheduler_fn(optimizer, phase=1)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Console().rule("[bold yellow]Phase 1: Train core model (AUX/Fusion/Output frozen)")
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        device,
        config,
        label_names,
        tokenizer_obj,
    )

    phase1_ckpt = os.path.join(config["checkpoints_dir"], "checkpoint_phase1.pt")
    latest_ckpt = os.path.join(config["checkpoints_dir"], "latest_checkpoint.pt")
    shutil.copyfile(latest_ckpt, phase1_ckpt)
    logger.info(f"[PHASE 1] Saved checkpoint to {phase1_ckpt}")

    # ---- Phase 2a: AUX warmup ----
    if num_epochs_aux_warmup > 0:
        aux_warmup_fn(model)
        config["num_epochs"] = num_epochs_aux_warmup
        optimizer = rebuild_optimizer(model, config)
        scheduler = scheduler_fn(optimizer, phase="aux")
        Console().rule("[bold cyan]Phase 2a: AUX-only warmup (core frozen)")
        train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            scheduler,
            device,
            config,
            label_names,
            tokenizer_obj,
        )
        phase2a_ckpt = os.path.join(config["checkpoints_dir"], "checkpoint_phase2a.pt")
        latest_ckpt = os.path.join(config["checkpoints_dir"], "latest_checkpoint.pt")
        shutil.copyfile(latest_ckpt, phase2a_ckpt)
        logger.info(f"[PHASE 2a] Saved checkpoint to {phase2a_ckpt}")

    # ---- Phase 2b: FULL unfreeze ----
    unfreeze_fn(model)
    config["num_epochs"] = num_epochs_phase2
    optimizer = rebuild_optimizer(model, config)
    scheduler = scheduler_fn(optimizer, phase=2)
    Console().rule("[bold yellow]Phase 2b: Train full model (all layers unfrozen)")
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        device,
        config,
        label_names,
        tokenizer_obj,
    )
