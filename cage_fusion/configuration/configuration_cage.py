"""
cage_fusion/configuration/configuration_cage.py
=================================================
Configuration class for the CAGEFusion model family.

:class:`CageFusionConfig` is a ``@dataclass`` that holds every model
hyper-parameter. It follows HuggingFace ``PretrainedConfig`` conventions:

- ``save_pretrained(directory)`` — write ``config.json``
- ``from_pretrained(directory)`` — load ``config.json``
- ``to_dict()`` / ``from_dict()`` — plain-dict serialisation

Quick start
-----------
**Create and save a config**::

    from cage_fusion import CageFusionConfig

    config = CageFusionConfig(
        num_labels=4,
        model_task="classification",
        label_names=["PAINS_A", "PAINS_B", "Aggregator", "Chelator"],
        attn_mode="self_graph",
    )
    config.save_pretrained("my_model/")

**Load it back**::

    config = CageFusionConfig.from_pretrained("my_model/")
    print(config.label_names)   # ["PAINS_A", ...]

**For ADMET regression**::

    config = CageFusionConfig(
        num_labels=12,
        model_task="regression",
        label_names=["logP", "logD", "solubility", "permeability", ...],
    )
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional


@dataclass
class CageFusionConfig:
    """
    Configuration for the CAGEFusion model family.

    Args:
        graph_dim: Hidden dimension of the D-MPNN graph encoder output.
        embedding_dim: Dimension of the ChemBERTa token embeddings.
        model_checkpoint: HuggingFace model ID for the sequence encoder.
        aux_feature_dim: Number of RDKit physicochemical descriptors.
        attn_mode: Attention strategy — one of ``"cross"``, ``"self_tokens"``,
            ``"self_graph"``, ``"self_both"``.
        num_heads: Number of attention heads.  Must evenly divide
            ``embedding_dim``.
        co_attention_layers: Number of stacked co-attention layers (cross mode).
        cross_attn_dropout: Dropout applied inside attention layers.
        proj_dropout: Dropout applied in projection layers.
        use_co_attention: Enable the attention pathway.
        use_aux_features: Enable the auxiliary-features pathway.
        use_fg_prompt: Enable functional-group chemical prompting.
        use_embedding_proj: Project token embeddings before attention.
        norm_type: Normalisation layer — ``"layer"`` (recommended) or
            ``"batch"``.
        fusion_residual: Add a skip connection in the fusion gate.
        fusion_dropout_1: Dropout after the first fusion MLP layer.
        fusion_dropout_2: Dropout after the second fusion MLP layer.
        scaled_graph_factor: Initial value for the learnable graph scale.
        scale_attn_factor: Initial value for the learnable attention scale.
        scale_aux_factor: Initial value for the learnable aux scale.
        scaled_fg_factor: Initial value for the learnable FG-prompt scale.
        num_labels: Number of output neurons (tasks or ADMET targets).
        hidden_size: Width of the pre-head representation.
        label_names: Optional list of task / property names, length
            ``num_labels``.  Stored in ``config.json`` and used by the
            pipeline to label output columns.
        model_task: ``"classification"`` (sigmoid / BCE) or
            ``"regression"`` (linear / MSE).  Used by
            :class:`~cage_fusion.auto.AutoCageFusion` to dispatch to the
            correct task head.
        lambda_entropy: Attention entropy regularisation weight.
        lambda_prior: Token-importance prior regularisation weight.
        model_type: Identifier string.  Always ``"cage_fusion"``.
    """

    # ── Encoder: Graph (D-MPNN via ChemProp) ─────────────────────────────────
    graph_dim: int = 300

    # ── Encoder: Sequence (ChemBERTa / BERT-SMILES) ──────────────────────────
    embedding_dim: int = 384
    model_checkpoint: str = "DeepChem/ChemBERTa-77M-MTR"

    # ── Encoder: Auxiliary (RDKit physicochemical descriptors) ───────────────
    aux_feature_dim: int = 217

    # ── Co-Attention ─────────────────────────────────────────────────────────
    attn_mode: Literal["cross", "self_tokens", "self_graph", "self_both"] = "self_graph"
    num_heads: int = 8
    co_attention_layers: int = 1
    cross_attn_dropout: float = 0.15
    proj_dropout: float = 0.10

    # ── Module toggles ────────────────────────────────────────────────────────
    use_co_attention: bool = True
    use_aux_features: bool = True
    use_fg_prompt: bool = True
    use_embedding_proj: bool = True

    # ── Normalisation ─────────────────────────────────────────────────────────
    norm_type: Literal["batch", "layer"] = "layer"

    # ── Fusion gate ───────────────────────────────────────────────────────────
    fusion_residual: bool = False
    fusion_dropout_1: float = 0.3
    fusion_dropout_2: float = 0.2

    # ── Modality scale initialisations ───────────────────────────────────────
    scaled_graph_factor: float = 10.0
    scale_attn_factor: float = 1.0
    scale_aux_factor: float = 0.5
    scaled_fg_factor: float = 1.0

    # ── Task head ─────────────────────────────────────────────────────────────
    num_labels: int = 4
    hidden_size: int = 128

    # ── Task metadata ─────────────────────────────────────────────────────────
    label_names: Optional[List[str]] = field(default=None)
    """Human-readable names for output labels, length ``num_labels``."""

    model_task: Literal["classification", "regression"] = "classification"
    """Determines the task head used by :class:`~cage_fusion.auto.AutoCageFusion`."""

    # ── Attention regularisation ──────────────────────────────────────────────
    lambda_entropy: float = 0.0
    lambda_prior: float = 0.0

    # ── Misc ──────────────────────────────────────────────────────────────────
    model_type: str = "cage_fusion"

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate parameter combinations after construction."""
        if not (0.0 <= self.cross_attn_dropout <= 1.0):
            raise ValueError(
                f"cross_attn_dropout must be in [0, 1], got {self.cross_attn_dropout}"
            )
        if not (0.0 <= self.proj_dropout <= 1.0):
            raise ValueError(
                f"proj_dropout must be in [0, 1], got {self.proj_dropout}"
            )
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.attn_mode not in {"cross", "self_tokens", "self_graph", "self_both"}:
            raise ValueError(f"Unknown attn_mode: '{self.attn_mode}'")
        if self.model_task not in {"classification", "regression"}:
            raise ValueError(f"Unknown model_task: '{self.model_task}'")
        if self.label_names is not None and len(self.label_names) != self.num_labels:
            raise ValueError(
                f"label_names has {len(self.label_names)} entries but "
                f"num_labels={self.num_labels}"
            )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def fusion_dim(self) -> int:
        """Total width of the concatenated pre-fusion vector.

        Equal to ``graph_dim + embedding_dim + aux_feature_dim``.
        """
        return self.graph_dim + self.embedding_dim + self.aux_feature_dim

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain JSON-compatible ``dict``."""
        d = asdict(self)
        d.pop("token_importance_prior", None)
        return d

    @classmethod
    def from_dict(cls, config_dict: dict) -> "CageFusionConfig":
        """
        Construct from a plain dict, ignoring unknown keys.

        Handles the legacy ``num_tasks`` key present in old checkpoints.

        Args:
            config_dict: Dictionary of configuration values.

        Returns:
            :class:`CageFusionConfig` instance.
        """
        d = dict(config_dict)

        # Legacy key mapping
        if "num_tasks" in d and "num_labels" not in d:
            d["num_labels"] = d.pop("num_tasks")

        # Also carry label_names from old 'tasks' list if present
        if "tasks" in d and "label_names" not in d and d.get("tasks"):
            d["label_names"] = d.pop("tasks")

        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in valid})

    def save_pretrained(self, save_directory: str) -> None:
        """
        Write ``config.json`` into *save_directory*.

        Args:
            save_directory: Local directory path. Created if missing.
        """
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, "config.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "CageFusionConfig":
        """
        Load config from a local directory containing ``config.json``.

        Args:
            pretrained_model_name_or_path: Path to the checkpoint directory.

        Raises:
            FileNotFoundError: If ``config.json`` does not exist.

        Returns:
            :class:`CageFusionConfig` instance.
        """
        path = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No config.json found at '{pretrained_model_name_or_path}'."
            )
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def __repr__(self) -> str:  # noqa: D105
        fields = [
            f"num_labels={self.num_labels}",
            f"model_task={self.model_task!r}",
            f"attn_mode={self.attn_mode!r}",
            f"hidden_size={self.hidden_size}",
            f"fusion_dim={self.fusion_dim}",
        ]
        return f"CageFusionConfig({', '.join(fields)})"
