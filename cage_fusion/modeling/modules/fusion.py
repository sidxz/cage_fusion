"""
Fusion gate and MLP.

Takes the three modality vectors (graph, attention, auxiliary) and
produces the pre-head ``hidden_size`` representation used by all task heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Literal


def _norm_layer(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm1d(dim)
    return nn.LayerNorm(dim)


class FusionHead(nn.Module):
    """
    Gated multimodal fusion followed by a 3-layer MLP.

    Architecture::

        raw = cat([graph_part, attn_part, aux_part])   # [B, fusion_dim]
        gate = sigmoid(linear(raw))
        fused = raw * gate  [+ raw if fusion_residual]
        hidden = MLP(fused)  →  [B, hidden_size]

    Parameters
    ----------
    fusion_dim:
        ``graph_dim + embedding_dim + aux_feature_dim``
    hidden_size:
        Output dimensionality fed to the task head.
    dropout_1, dropout_2:
        Dropout rates applied after the first and second MLP layers.
    norm_type:
        ``"layer"`` (recommended) or ``"batch"`` (original cage.py style).
    fusion_residual:
        When ``True`` the raw pre-gate vector is added back after gating,
        matching the ``cage_cross_attn.py`` variant.
    """

    def __init__(
        self,
        fusion_dim: int,
        hidden_size: int = 128,
        dropout_1: float = 0.3,
        dropout_2: float = 0.2,
        norm_type: Literal["layer", "batch"] = "layer",
        fusion_residual: bool = False,
    ):
        super().__init__()
        self.fusion_residual = fusion_residual
        self.gate = nn.Linear(fusion_dim, fusion_dim)

        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, 768),
            _norm_layer(norm_type, 768),
            nn.ReLU(),
            nn.Dropout(dropout_1),
            nn.Linear(768, 384),
            _norm_layer(norm_type, 384),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            nn.Linear(384, hidden_size),
            _norm_layer(norm_type, hidden_size),
            nn.ReLU(),
        )

    def forward(
        self,
        graph_part: torch.Tensor,   # [B, graph_dim]
        attn_part: torch.Tensor,    # [B, embedding_dim]
        aux_part: torch.Tensor,     # [B, aux_feature_dim]
    ) -> torch.Tensor:
        """Returns ``[B, hidden_size]``."""
        raw = torch.cat([graph_part, attn_part, aux_part], dim=1)
        gate = torch.sigmoid(self.gate(raw))
        fused = raw * gate
        if self.fusion_residual:
            fused = fused + raw
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e3, neginf=-1e3)
        return self.mlp(fused)
