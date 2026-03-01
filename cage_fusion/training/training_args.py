"""
Training hyperparameter container.

``TrainingArguments`` is a plain dataclass (no framework dependencies)
that holds every knob needed by ``Trainer``.  It is saved alongside
checkpoints so experiments are fully reproducible.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TrainingArguments:
    """
    All hyperparameters and directory settings for a training run.

    Example::

        args = TrainingArguments(
            output_dir="runs/exp1",
            num_epochs=50,
            learning_rate=1e-3,
        )
        args.save(args.output_dir)
    """

    # ── Directories ───────────────────────────────────────────────────────
    output_dir: str = "outputs"
    checkpoints_dir: str = "checkpoints"
    base_cache_dir: str = ".cache"

    # ── Optimisation ──────────────────────────────────────────────────────
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 50
    batch_size: int = 128
    warmup_fraction: float = 0.09
    max_grad_norm: float = 1.0

    # ── Attention regularisation ──────────────────────────────────────────
    lambda_entropy: float = 0.0   # weight for attention entropy loss
    lambda_prior: float = 0.0     # weight for token-importance prior loss

    # ── Data loading ──────────────────────────────────────────────────────
    num_workers: int = 4

    # ── Checkpoint / resume ───────────────────────────────────────────────
    resume_with_new_arch: bool = False

    # ── Checkpoint metric ─────────────────────────────────────────────────
    primary_metric: str = "rmse"
    """Metric used to select ``best_model.pt``.
    Regression options : ``"rmse"`` | ``"mae"`` | ``"r2"`` | ``"marae"``
    Classification options: ``"auc"`` | ``"mcc"`` | ``"pr"``
    """
    primary_metric_direction: str = "min"
    """``"min"`` if lower is better (rmse, mae, marae);
    ``"max"`` if higher is better (r2, auc, mcc, pr).
    """

    # ── Mixed precision ───────────────────────────────────────────────────
    bf16: bool = False
    """Use BF16 autocast during training (recommended for A6000 Ada / Ampere+)."""

    # ── Misc ─────────────────────────────────────────────────────────────
    seed: int = 42

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingArguments":
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in valid})

    def save(self, directory: str) -> None:
        """Write ``training_args.json`` into *directory*."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "training_args.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "TrainingArguments":
        path = os.path.join(directory, "training_args.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No training_args.json at '{directory}'.")
        with open(path) as f:
            return cls.from_dict(json.load(f))
