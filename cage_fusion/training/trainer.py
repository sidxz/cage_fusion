"""
Trainer — HuggingFace-inspired training loop for CAGEFusion models.

Key features
────────────
- Single ``.train()`` entry point with automatic checkpointing
- ``train_epoch()`` / ``evaluate()`` usable independently
- Phased training via ``freeze_phase()`` + ``rebuild_optimizer()``
- ``staged_finetune()`` for the full warmup → phase1 → aux-warmup → full-unfreeze workflow
- Streaming disk-based metrics (no RAM accumulation)
"""

from __future__ import annotations

import logging
import os
import shutil
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from tqdm import tqdm

from cage_fusion.training.metrics import (
    AUCAccumulator,
    MARAEAccumulator,
    MCCAccumulator,
    PRAccumulator,
    RegressionAccumulator,
)
from cage_fusion.training.training_args import TrainingArguments
from cage_fusion.utils.device_utils import move_bmg_to_device

logger = logging.getLogger("cagefusion")
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Freeze helpers
# ─────────────────────────────────────────────────────────────────────────────

_FREEZE_STRATEGIES: Dict[str, List[str]] = {
    # Phase 1: train graph + attention; freeze aux / fusion / output
    "freeze_aux_and_output": ["fusion_mlp", "fusion", "output", "classifier", "regressor", "scale_aux", "aux_mlp"],
    # Phase 1-alt: train graph + attention + output; freeze aux / fusion
    "freeze_aux_and_fusion": ["fusion_mlp", "fusion", "scale_aux", "aux_mlp"],
    # AUX warmup: train only aux / fusion / output
    "aux_only": None,  # handled specially below
}


def freeze_phase(model: nn.Module, phase: str) -> None:
    """
    Freeze / unfreeze parameters according to a named strategy.

    Phases
    ------
    ``"freeze_aux_and_output"``  – train graph encoder + attention, freeze the rest
    ``"freeze_aux_and_fusion"``  – train graph + attention + output head, freeze aux + fusion
    ``"aux_only"``               – train aux MLP + fusion + head only (core frozen)
    ``"unfreeze_all"``           – unfreeze everything
    """
    frozen, trainable = [], []

    if phase == "unfreeze_all":
        for name, p in model.named_parameters():
            p.requires_grad = True
            trainable.append(name)

    elif phase == "aux_only":
        aux_keys = ["fusion_mlp", "fusion", "output", "classifier", "regressor", "scale_aux", "aux_mlp"]
        for name, p in model.named_parameters():
            if any(k in name for k in aux_keys):
                p.requires_grad = True
                trainable.append(name)
            else:
                p.requires_grad = False
                frozen.append(name)

    elif phase in _FREEZE_STRATEGIES:
        freeze_keys = _FREEZE_STRATEGIES[phase]
        for name, p in model.named_parameters():
            if any(k in name for k in freeze_keys):
                p.requires_grad = False
                frozen.append(name)
            else:
                p.requires_grad = True
                trainable.append(name)
    else:
        raise ValueError(
            f"Unknown freeze phase '{phase}'. "
            f"Valid: {list(_FREEZE_STRATEGIES.keys()) + ['unfreeze_all']}"
        )

    logger.info("[freeze_phase=%s] trainable=%d frozen=%d", phase, len(trainable), len(frozen))


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Training orchestrator for CAGEFusion models.

    Parameters
    ----------
    model:
        A ``CAGEFusionForMultiLabelClassification`` or
        ``CAGEFusionForRegression`` instance (or any nn.Module that
        accepts the same forward signature).
    args:
        ``TrainingArguments`` with all hyperparameters.
    train_loader, val_loader:
        PyTorch DataLoaders.
    criterion:
        Loss function.  ``BCEWithLogitsLoss`` for classification,
        ``MSELoss`` for regression.  If ``None``, the model's built-in
        ``loss`` field (returned when ``labels`` are passed) is used.
    label_names:
        List of task names for metric logging.
    tokenizer_obj:
        Tokenizer (used by visualisation helpers during evaluation).
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_loader,
        val_loader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        label_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve label names and task type from model config
        cfg = getattr(model, "config", None)
        if label_names is not None:
            self.label_names = label_names
        else:
            self.label_names = getattr(cfg, "label_names", None) or []
        self.task: str = getattr(cfg, "model_task", "classification")

        # Build a default Adam optimizer if none was provided
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            trainable = [p for p in model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.Adam(
                trainable,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            logger.info(
                "Auto-built Adam optimizer  lr=%.2e  wd=%.2e  params=%d",
                args.learning_rate, args.weight_decay, len(trainable),
            )

        self.scheduler = scheduler

    # ── Public API ──────────────────────────────────────────────────────────

    def train(self) -> dict:
        """Run the full training loop and return the history dict."""
        args = self.args
        os.makedirs(args.checkpoints_dir, exist_ok=True)

        checkpoint_path = os.path.join(args.checkpoints_dir, "latest_checkpoint.pt")
        best_auc_path = os.path.join(args.checkpoints_dir, "best_model.pt")
        best_mcc_path = os.path.join(args.checkpoints_dir, "best_model_mcc.pt")

        start_epoch = 1
        # Initialise best-metric trackers from TrainingArguments.
        # primary_metric_direction "min" → start at +inf; "max" → start at -inf.
        _dir = args.primary_metric_direction
        best_primary   = float("inf")  if _dir == "min" else -float("inf")
        # Secondary metric keeps the old defaults for backward compat.
        if self.task == "regression":
            best_secondary = float("inf")   # MAE
        else:
            best_secondary = -1.0           # MCC
        history = self._empty_history()

        # Resume from checkpoint
        if os.path.exists(checkpoint_path):
            start_epoch, best_primary, best_secondary, history = self._resume(checkpoint_path)

        logger.info(
            "Training from epoch %d to %d | train batches: %d | val batches: %d",
            start_epoch, args.num_epochs,
            len(self.train_loader), len(self.val_loader),
        )
        logger.info(
            "Trainable params: %s",
            f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}",
        )

        for epoch in range(start_epoch, args.num_epochs + 1):
            console.rule(f"[bold blue]Epoch {epoch}/{args.num_epochs}")

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()

            self._update_history(history, train_metrics, val_metrics)
            self._log_epoch(epoch, history, val_metrics)

            ckpt = self._make_checkpoint(epoch, history, best_primary, best_secondary, val_metrics)
            torch.save(ckpt, checkpoint_path)

            # Primary metric — driven by args.primary_metric + direction.
            p_key   = args.primary_metric
            p_val   = val_metrics.get(p_key, float("nan"))
            p_dir   = args.primary_metric_direction
            p_label = p_key.upper()
            if p_dir == "min":
                p_improved = p_val < best_primary
            else:
                p_improved = p_val > best_primary

            # Secondary metric keeps old defaults.
            if self.task == "regression":
                s_val      = val_metrics["mae"]
                s_improved = s_val < best_secondary
                s_label    = "MAE"
            else:
                s_val      = val_metrics["mcc"]
                s_improved = s_val > best_secondary
                s_label    = "MCC"

            if p_improved:
                best_primary = p_val
                ckpt["best_primary"] = best_primary
                torch.save(ckpt, best_auc_path)
                self.model.save_pretrained(args.checkpoints_dir)
                logger.info("New best %s=%.4f → %s", p_label, best_primary, best_auc_path)

            if s_improved:
                best_secondary = s_val
                ckpt["best_secondary"] = best_secondary
                torch.save(ckpt, best_mcc_path)
                logger.info("New best %s=%.4f → %s", s_label, best_secondary, best_mcc_path)

        logger.info("Training complete.")
        self._plot_history(history)
        return history

    def train_epoch(self, epoch: int) -> dict:
        """Run one training epoch. Returns a metrics dict."""
        args = self.args
        model = self.model
        model.train()

        num_tasks = getattr(getattr(model, "config", None), "num_labels", 1)

        if self.task == "regression":
            reg_agg = RegressionAccumulator(num_tasks, self.label_names)
        else:
            mcc_agg = MCCAccumulator(num_tasks, self.label_names)
            auc_agg = AUCAccumulator(num_tasks, self.label_names)
            pr_agg  = PRAccumulator(num_tasks, self.label_names)

        total_loss = 0.0
        fg_attention: dict = defaultdict(float)
        fg_count: dict = defaultdict(int)

        for batch in tqdm(self.train_loader, desc=f"Train epoch {epoch}"):
            if batch is None:
                continue

            bmg, token_embs, attn_mask, aux_feats, labels, input_ids, smiles_batch, _, _ = batch
            bmg = move_bmg_to_device(bmg, self.device)
            token_embs = token_embs.to(self.device, non_blocking=True)
            attn_mask = attn_mask.to(self.device, non_blocking=True)
            aux_feats = aux_feats.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            input_ids = input_ids.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            output = model(
                bmg=bmg,
                sequence_embeddings=token_embs,
                attn_mask=attn_mask,
                aux_feats=aux_feats,
                input_ids_batch=input_ids,
                smiles_batch=smiles_batch,
                labels=labels,
                lambda_entropy=args.lambda_entropy,
                lambda_prior=args.lambda_prior,
                return_attn=True,
            )

            loss = output.loss
            if loss is None:
                raise RuntimeError("Model returned loss=None. Pass labels to forward().")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()

            preds = output.logits.detach()
            if self.task == "regression":
                reg_agg.update(labels.detach(), preds)
            else:
                probs = torch.sigmoid(preds)
                mcc_agg.update(labels.detach(), probs)
                auc_agg.update(labels.detach(), probs)
                pr_agg.update(labels.detach(), probs)

            # Accumulate FG prompt attention stats
            if output.prompt_attn_weights:
                for item in output.prompt_attn_weights:
                    if isinstance(item, dict):
                        for fg_id, w in zip(item.get("fg_ids", []), item.get("weights", [])):
                            if fg_id != -1:
                                fg_attention[fg_id] += float(w)
                                fg_count[fg_id] += 1

        avg_loss = total_loss / max(1, len(self.train_loader))

        # Log top-5 attended functional groups
        if fg_attention:
            avg_fg = {k: fg_attention[k] / fg_count[k] for k in fg_attention if fg_count[k] > 0}
            top5 = sorted(avg_fg.items(), key=lambda x: x[1], reverse=True)[:5]
            try:
                from cage_fusion.chemistry.fg_utils import FG_NAMES
                lines = ["Top-5 attended FGs:"]
                for rank, (fg_id, w) in enumerate(top5, 1):
                    name = FG_NAMES[fg_id] if fg_id < len(FG_NAMES) else f"FG_{fg_id}"
                    lines.append(f"  {rank}. {name:<25} avg={w:.4f}")
                logger.info("\n".join(lines))
            except Exception:
                pass

        if self.task == "regression":
            avg_rmse, avg_mae, avg_r2 = reg_agg.compute()
            logger.info(
                "Epoch train | loss=%.4f rmse=%.4f mae=%.4f r2=%.4f",
                avg_loss, avg_rmse, avg_mae, avg_r2,
            )
            return {"loss": avg_loss, "rmse": avg_rmse, "mae": avg_mae, "r2": avg_r2}
        else:
            avg_mcc, *_ = mcc_agg.compute()
            avg_auc = auc_agg.compute()
            avg_pr  = pr_agg.compute()
            logger.info(
                "Epoch train | loss=%.4f mcc=%.4f auc=%.4f pr=%.4f",
                avg_loss, avg_mcc, avg_auc, avg_pr,
            )
            return {"loss": avg_loss, "mcc": avg_mcc, "auc": avg_auc, "pr": avg_pr}

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on val_loader. Returns a metrics dict."""
        model = self.model
        model.eval()
        num_tasks = getattr(getattr(model, "config", None), "num_labels", 1)

        if self.task == "regression":
            reg_agg   = RegressionAccumulator(num_tasks, self.label_names)
            marae_agg = MARAEAccumulator(num_tasks, self.label_names)
        else:
            mcc_agg = MCCAccumulator(num_tasks, self.label_names)
            auc_agg = AUCAccumulator(num_tasks, self.label_names)
            pr_agg  = PRAccumulator(num_tasks, self.label_names)

        total_loss = 0.0
        total_graph_norm = total_attn_norm = total_aux_norm = 0.0

        for batch in tqdm(self.val_loader, desc="Evaluate"):
            if batch is None:
                continue
            bmg, token_embs, attn_mask, aux_feats, labels, input_ids, smiles_batch, _, _ = batch
            bmg = move_bmg_to_device(bmg, self.device)
            token_embs = token_embs.to(self.device)
            attn_mask = attn_mask.to(self.device)
            aux_feats = aux_feats.to(self.device)
            labels = labels.to(self.device)
            input_ids = input_ids.to(self.device)

            output = model(
                bmg=bmg,
                sequence_embeddings=token_embs,
                attn_mask=attn_mask,
                aux_feats=aux_feats,
                input_ids_batch=input_ids,
                smiles_batch=smiles_batch,
                labels=labels,
                return_attn=False,
            )

            if output.loss is not None:
                total_loss += output.loss.item()

            if output.graph_repr is not None:
                total_graph_norm += output.graph_repr.norm(dim=1).mean().item()
            if output.attn_output is not None:
                total_attn_norm += output.attn_output.norm(dim=1).mean().item()
            total_aux_norm += aux_feats.norm(dim=1).mean().item()

            preds = output.logits
            if self.task == "regression":
                reg_agg.update(labels, preds)
                marae_agg.update(labels, preds)
            else:
                probs = torch.sigmoid(preds)
                mcc_agg.update(labels, probs)
                auc_agg.update(labels, probs)
                pr_agg.update(labels, probs)

        n = max(1, len(self.val_loader))
        avg_loss = total_loss / n
        norms = {
            "norm_graph": total_graph_norm / n,
            "norm_attn":  total_attn_norm / n,
            "norm_aux":   total_aux_norm / n,
        }

        if self.task == "regression":
            avg_rmse, avg_mae, avg_r2 = reg_agg.compute()
            pt_rmse, pt_mae, pt_r2   = reg_agg.compute(reduce="none")
            avg_marae, _             = marae_agg.compute()
            logger.info(
                "Epoch val | loss=%.4f rmse=%.4f mae=%.4f r2=%.4f marae=%.4f",
                avg_loss, avg_rmse, avg_mae, avg_r2, avg_marae,
            )
            return {
                "loss": avg_loss, "rmse": avg_rmse, "mae": avg_mae, "r2": avg_r2,
                "marae": avg_marae,
                "per_task": list(zip(pt_rmse.tolist(), pt_mae.tolist(), pt_r2.tolist())),
                **norms,
            }
        else:
            per_task_auc                       = auc_agg.compute(reduce="none")
            per_task_pr                        = pr_agg.compute(reduce="none")
            avg_auc = float(np.nanmean(per_task_auc)) if len(per_task_auc) > 0 else 0.0
            avg_pr  = float(np.nanmean(per_task_pr))  if len(per_task_pr) > 0 else 0.0
            avg_mcc, best_thresholds, per_task_mcc = mcc_agg.compute()
            logger.info(
                "Epoch val | loss=%.4f mcc=%.4f auc=%.4f pr=%.4f",
                avg_loss, avg_mcc, avg_auc, avg_pr,
            )
            return {
                "loss": avg_loss, "mcc": avg_mcc, "auc": avg_auc, "pr": avg_pr,
                "best_thresholds": best_thresholds,
                "per_task": list(zip(per_task_mcc, per_task_auc, per_task_pr)),
                **norms,
            }

    # ── Phased training ──────────────────────────────────────────────────────

    def freeze_phase(self, phase: str) -> None:
        """Apply a named freeze strategy to the model. Rebuilds optimizer."""
        freeze_phase(self.model, phase)

    def rebuild_optimizer(self, lr: Optional[float] = None) -> None:
        """Re-create Adam with only currently trainable parameters."""
        lr = lr or self.args.learning_rate
        wd = self.args.weight_decay
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    def staged_finetune(
        self,
        num_epochs_warmup: int,
        num_epochs_phase1: int,
        num_epochs_aux_warmup: int,
        num_epochs_phase2: int,
        scheduler_fn: Optional[Callable] = None,
        freeze_phase1: str = "freeze_aux_and_fusion",
        criterion: Optional[nn.Module] = None,
    ) -> None:
        """
        Run the canonical 4-phase staged fine-tuning protocol:

        0. Warmup     – all layers trainable
        1. Phase 1    – freeze aux / fusion; train encoder + attention (+ head)
        2. AUX warmup – freeze encoder + attention; train aux + fusion + head
        3. Phase 2    – full unfreeze

        ``scheduler_fn(optimizer, phase)`` should return a scheduler instance.
        """
        if criterion is not None:
            self.criterion = criterion

        def _run(n_epochs: str, phase_label: str):
            self.args.num_epochs = n_epochs
            console.rule(f"[bold yellow]{phase_label}")
            self.train()
            ckpt_src = os.path.join(self.args.checkpoints_dir, "latest_checkpoint.pt")
            ckpt_dst = os.path.join(self.args.checkpoints_dir, f"checkpoint_{phase_label.replace(' ', '_')}.pt")
            shutil.copyfile(ckpt_src, ckpt_dst)
            logger.info("Saved phase checkpoint → %s", ckpt_dst)

        # Warmup
        self.freeze_phase("unfreeze_all")
        self.rebuild_optimizer()
        if scheduler_fn:
            self.scheduler = scheduler_fn(self.optimizer, phase="warmup")
        _run(num_epochs_warmup, "Phase 0 Warmup")

        # Phase 1
        self.freeze_phase(freeze_phase1)
        self.rebuild_optimizer()
        if scheduler_fn:
            self.scheduler = scheduler_fn(self.optimizer, phase=1)
        _run(num_epochs_phase1, "Phase 1 core training")

        # AUX warmup
        if num_epochs_aux_warmup > 0:
            self.freeze_phase("aux_only")
            self.rebuild_optimizer()
            if scheduler_fn:
                self.scheduler = scheduler_fn(self.optimizer, phase="aux")
            _run(num_epochs_aux_warmup, "Phase 2a AUX warmup")

        # Full unfreeze
        self.freeze_phase("unfreeze_all")
        self.rebuild_optimizer()
        if scheduler_fn:
            self.scheduler = scheduler_fn(self.optimizer, phase=2)
        _run(num_epochs_phase2, "Phase 2b full fine-tune")

    # ── Internals ────────────────────────────────────────────────────────────

    def _empty_history(self) -> dict:
        common = [
            "train_loss", "val_loss", "per_task",
            "scale_graph", "scale_attn", "scale_aux",
            "val_norm_graph", "val_norm_attn", "val_norm_aux",
        ]
        if self.task == "regression":
            task_keys = [
                "train_rmse", "val_rmse",
                "train_mae",  "val_mae",
                "train_r2",   "val_r2",
                "train_marae","val_marae",
            ]
        else:
            task_keys = ["train_mcc", "val_mcc", "train_auc", "val_auc", "train_pr", "val_pr"]
        return {k: [] for k in common + task_keys}

    def _resume(self, checkpoint_path: str):
        logger.info("Resuming from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if self.args.resume_with_new_arch:
            cur = self.model.state_dict()
            compat = {
                k: v for k, v in ckpt["model_state_dict"].items()
                if k in cur and cur[k].shape == v.shape
            }
            self.model.load_state_dict(compat, strict=False)
        else:
            self.model.load_state_dict(ckpt["model_state_dict"])
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if self.scheduler:
                    self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except ValueError as e:
                logger.warning("Skipping optimizer/scheduler restore: %s", e)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        history = ckpt.get("history", self._empty_history())
        start_epoch = ckpt.get("epoch", 0) + 1
        _dir = self.args.primary_metric_direction
        _default_primary = float("inf") if _dir == "min" else -float("inf")
        best_primary   = ckpt.get("best_primary", _default_primary)
        if self.task == "regression":
            best_secondary = ckpt.get("best_secondary", float("inf"))
        else:
            best_secondary = ckpt.get("best_secondary", ckpt.get("best_val_mcc", -1.0))
        return start_epoch, best_primary, best_secondary, history

    def _update_history(self, history: dict, train: dict, val: dict):
        metrics = ("loss", "rmse", "mae", "r2", "marae") if self.task == "regression" \
                  else ("loss", "mcc", "auc", "pr")
        for metric in metrics:
            history[f"train_{metric}"].append(train[metric])
            history[f"val_{metric}"].append(val[metric])
        history["per_task"].append(val.get("per_task", []))
        m = self.model
        # Scalers live on CAGEFusionModel; task-head models (Classification/Regression)
        # wrap it as `.encoder`, so check there first.
        enc = getattr(m, "encoder", m)
        history["scale_graph"].append(getattr(enc, "scale_graph", torch.tensor(0.0)).item() if hasattr(enc, "scale_graph") else 0.0)
        history["scale_attn"].append(getattr(enc, "scale_attn", torch.tensor(0.0)).item() if hasattr(enc, "scale_attn") else 0.0)
        history["scale_aux"].append(getattr(enc, "scale_aux", torch.tensor(0.0)).item() if hasattr(enc, "scale_aux") else 0.0)
        history["val_norm_graph"].append(val.get("norm_graph", 0.0))
        history["val_norm_attn"].append(val.get("norm_attn", 0.0))
        history["val_norm_aux"].append(val.get("norm_aux", 0.0))

    def _log_epoch(self, epoch: int, history: dict, val: dict):
        from cage_fusion.utils.logging import log_epoch_results
        num_epochs = self.args.num_epochs
        per_task = val.get("per_task", [])
        log_epoch_results(epoch, num_epochs, history, self.label_names, per_task)

    def _make_checkpoint(self, epoch, history, best_primary, best_secondary, val):
        sched_state = self.scheduler.state_dict() if self.scheduler else {}
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": sched_state,
            "history": history,
            "best_primary":   best_primary,
            "best_secondary": best_secondary,
            "config": self.model.config.to_dict() if hasattr(self.model, "config") else {},
        }
        if self.task != "regression":
            ckpt["best_thresholds"] = val.get("best_thresholds", [])
        return ckpt

    def _plot_history(self, history: dict):
        try:
            from cage_fusion.utils.logging import plot_training_history
            plot_training_history(history, output_dir=self.args.output_dir)
        except Exception as e:
            logger.warning("Could not plot training history: %s", e)
