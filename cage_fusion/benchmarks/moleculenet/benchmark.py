"""
cage_fusion/benchmarks/moleculenet/benchmark.py
================================================
Single-call MoleculeNet benchmark runner.

Usage::

    from cage_fusion.benchmarks import run_moleculenet_benchmark

    results = run_moleculenet_benchmark(
        dataset="bace_classification",
        output_dir="runs/bace",
    )
    print(f"Test ROC-AUC: {results['test_auc']:.4f}")
    print(f"Test MCC:     {results['test_mcc']:.4f}")
"""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, Optional

import numpy as np
import torch

from cage_fusion.configuration import CageFusionConfig
from cage_fusion.auto import AutoCageFusion
from cage_fusion.data import CageFusionDataModule
from cage_fusion.training import Trainer, TrainingArguments
from cage_fusion.evaluation import evaluate_model

logger = logging.getLogger(__name__)


def run_moleculenet_benchmark(
    dataset: str = "bace_classification",
    *,
    output_dir: str = "runs/benchmark",
    model_checkpoint: str = "DeepChem/ChemBERTa-77M-MTR",
    splitter: str = "scaffold",
    seed: int = 42,
    num_epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    attn_mode: str = "self_graph",
    use_fg_prompt: bool = True,
    config: Optional[CageFusionConfig] = None,
    training_args: Optional[TrainingArguments] = None,
    data_dir: str = "data/molnet",
    cache_dir: Optional[str] = None,
) -> Dict:
    """
    Load a MoleculeNet dataset, train CAGEFusion, and return test metrics.

    All intermediate artefacts (HDF5 features, model checkpoints) are written
    to *output_dir*.  The best model by validation ROC-AUC is used for the
    final test evaluation.

    Args:
        dataset: DeepChem dataset name, e.g. ``"bace_classification"``,
            ``"tox21"``, ``"sider"``, ``"hiv"``.  The full list is at
            https://deepchem.io/docs/api_reference/moleculenet.html
        output_dir: Root directory for checkpoints, logs, and caches.
        model_checkpoint: HuggingFace sequence-encoder checkpoint.
        splitter: ``"scaffold"`` (default), ``"random"``, or ``"stratified"``.
        seed: Global random seed.
        num_epochs: Number of training epochs.
        batch_size: DataLoader batch size.
        learning_rate: Adam learning rate.
        attn_mode: Co-attention strategy — ``"cross"``, ``"self_tokens"``,
            ``"self_graph"``, or ``"self_both"``.
        use_fg_prompt: Enable functional-group chemical prompting.
        config: Optional :class:`~cage_fusion.configuration.CageFusionConfig`
            override (all architecture settings).
        training_args: Optional :class:`~cage_fusion.training.TrainingArguments`
            override (all training settings).
        data_dir: Directory for DeepChem dataset downloads.
        cache_dir: Directory for HDF5 feature caches (defaults to
            ``<output_dir>/features``).

    Returns:
        Dict with keys ``test_auc``, ``test_mcc``, ``test_pr``,
        ``per_task_metrics``, ``label_names``, ``checkpoint_dir``.

    Example::

        from cage_fusion.benchmarks import run_moleculenet_benchmark

        results = run_moleculenet_benchmark("tox21", num_epochs=30)
        for task, (mcc, auc, pr) in zip(results["label_names"],
                                         results["per_task_metrics"]):
            print(f"{task:30s}  AUC={auc:.4f}  MCC={mcc:.4f}")
    """
    _set_seed(seed)

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    cache_dir = cache_dir or os.path.join(output_dir, "features")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── 1. Data ────────────────────────────────────────────────────────────
    logger.info("Preparing data module for '%s'…", dataset)
    dm = CageFusionDataModule.from_moleculenet(
        dataset_name=dataset,
        model_checkpoint=model_checkpoint,
        splitter=splitter,
        seed=seed,
        data_dir=data_dir,
        cache_dir=cache_dir,
        batch_size=batch_size,
    )
    # Persist scaler for inference
    dm.save_scaler(checkpoint_dir)

    # ── 2. Model ───────────────────────────────────────────────────────────
    if config is None:
        config = CageFusionConfig(
            num_labels=len(dm.label_names),
            model_task="classification",
            label_names=dm.label_names,
            model_checkpoint=model_checkpoint,
            attn_mode=attn_mode,
            use_fg_prompt=use_fg_prompt,
        )

    logger.info("Building model: %s", config)
    model = AutoCageFusion.from_config(config).to(device)
    logger.info(
        "Parameters: %s total / %s trainable",
        f"{sum(p.numel() for p in model.parameters()):,}",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )

    # ── 3. Training ────────────────────────────────────────────────────────
    if training_args is None:
        training_args = TrainingArguments(
            output_dir=output_dir,
            checkpoints_dir=checkpoint_dir,
            base_cache_dir=os.path.join(output_dir, "train_cache"),
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_loader=dm.train_loader,
        val_loader=dm.val_loader,
        device=device,
    )
    history = trainer.train()

    # ── 4. Test evaluation ─────────────────────────────────────────────────
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(best_model_path):
        logger.warning("No best_model.pt found; using latest checkpoint.")
        best_model_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")

    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    best_thresholds = ckpt.get("best_thresholds", [0.5] * len(dm.label_names))

    if dm.test_loader is None:
        logger.warning("No test split available; returning validation metrics.")
        return {
            "test_auc": history["val_auc"][-1],
            "test_mcc": history["val_mcc"][-1],
            "test_pr": history["val_pr"][-1],
            "per_task_metrics": [],
            "label_names": dm.label_names,
            "checkpoint_dir": checkpoint_dir,
            "history": history,
        }

    criterion = torch.nn.BCEWithLogitsLoss()
    test_metrics = evaluate_model(
        model=model,
        loader=dm.test_loader,
        criterion=criterion,
        device=device,
        num_tasks=len(dm.label_names),
        label_names=dm.label_names,
        use_precomputed_thresholds=best_thresholds,
        cache_dir=os.path.join(output_dir, "test_cache"),
    )

    # evaluate_model returns a named tuple / dict-like
    (test_loss, test_mcc, test_auc, test_pr, *rest) = test_metrics
    per_task = rest[1] if len(rest) > 1 else []

    logger.info(
        "Test results | AUC=%.4f  MCC=%.4f  PR=%.4f", test_auc, test_mcc, test_pr
    )

    return {
        "test_auc": float(test_auc),
        "test_mcc": float(test_mcc),
        "test_pr": float(test_pr),
        "per_task_metrics": per_task,
        "label_names": dm.label_names,
        "checkpoint_dir": checkpoint_dir,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
