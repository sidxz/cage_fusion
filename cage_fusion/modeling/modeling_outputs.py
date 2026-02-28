from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import torch


@dataclass
class CageFusionEncoderOutput:
    """
    Output of the CAGEFusion *backbone encoder* (no task head attached).

    ``hidden_states`` is the ``[B, hidden_size]`` representation produced
    by the fusion MLP.  Task-specific heads operate on this tensor.

    All interpretability fields are ``None`` when ``return_attn=False``.
    Regularisation losses are ``0.0`` (scalar tensor) when not computed.
    """

    # Core representation fed into task heads
    hidden_states: torch.Tensor                          # [B, hidden_size]

    # Regularisation auxiliary losses
    attn_entropy_loss: torch.Tensor = field(
        default_factory=lambda: torch.tensor(0.0)
    )
    token_prior_loss: torch.Tensor = field(
        default_factory=lambda: torch.tensor(0.0)
    )

    # Interpretability / visualisation (populated when return_attn=True)
    graph_to_token_weights: Optional[torch.Tensor] = None  # [B, heads, N_atoms, T]
    token_to_graph_weights: Optional[torch.Tensor] = None  # [B, heads, T, N_atoms]
    attn_output: Optional[torch.Tensor] = None             # [B, embedding_dim]
    graph_repr: Optional[torch.Tensor] = None              # [B, graph_dim]
    atom_features: Optional[torch.Tensor] = None           # [total_atoms, graph_dim]
    prompt_attn_weights: Optional[List] = None             # list[dict] per molecule


@dataclass
class CageFusionModelOutput:
    """
    Output of a CAGEFusion model **with a task head** attached.

    When ``labels`` are supplied to ``forward()``, the ``loss`` field is
    populated with the combined task loss (including any regularisation).

    This is the standard output returned by
    ``CAGEFusionForMultiLabelClassification`` and
    ``CAGEFusionForRegression``.
    """

    logits: torch.Tensor                                   # [B, num_labels]
    loss: Optional[torch.Tensor] = None                    # scalar

    # Encoder passthrough (same semantics as CageFusionEncoderOutput)
    hidden_states: Optional[torch.Tensor] = None
    attn_entropy_loss: Optional[torch.Tensor] = None
    token_prior_loss: Optional[torch.Tensor] = None
    graph_to_token_weights: Optional[torch.Tensor] = None
    token_to_graph_weights: Optional[torch.Tensor] = None
    attn_output: Optional[torch.Tensor] = None
    graph_repr: Optional[torch.Tensor] = None
    atom_features: Optional[torch.Tensor] = None
    prompt_attn_weights: Optional[List] = None
