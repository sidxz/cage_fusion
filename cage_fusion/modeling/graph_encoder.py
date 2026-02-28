"""
cage_fusion/modeling/graph_encoder.py
======================================
Self-contained D-MPNN molecular graph encoder.

The :class:`MolGraphEncoder` wraps ChemProp's ``BondMessagePassing`` +
``MeanAggregation`` into a single ``nn.Module`` so the rest of the model does
not depend on ChemProp internals directly.

Typical usage::

    from cage_fusion.modeling.graph_encoder import MolGraphEncoder
    from cage_fusion.configuration import CageFusionConfig

    cfg = CageFusionConfig(graph_dim=300)
    encoder = MolGraphEncoder(cfg)

    atom_features, graph_repr = encoder(bmg)
    # atom_features : [total_atoms, graph_dim]
    # graph_repr    : [batch_size,  graph_dim]
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
from chemprop.nn.agg import MeanAggregation
from chemprop.nn.message_passing import BondMessagePassing
from chemprop.nn.predictors import BinaryClassificationFFN
from chemprop.models.model import MPNN

from cage_fusion.configuration.configuration_cage import CageFusionConfig

logger = logging.getLogger(__name__)


class MolGraphEncoder(nn.Module):
    """
    Molecular graph encoder based on ChemProp's D-MPNN architecture.

    Runs bond-message-passing over a :class:`~chemprop.data.BatchMolGraph`,
    then mean-pools atom features to produce a fixed-size molecule-level
    representation.

    Args:
        config: A :class:`~cage_fusion.configuration.CageFusionConfig` instance.
            The ``graph_dim``, ``num_labels`` fields are used to set up the
            underlying MPNN.

    Inputs:
        bmg: A :class:`~chemprop.data.BatchMolGraph` batch.

    Returns:
        A 2-tuple ``(atom_features, graph_repr)`` where:

        - ``atom_features``: ``[total_atoms, graph_dim]`` — per-atom hidden states
          after message passing. Used by attention layers.
        - ``graph_repr``: ``[batch_size, graph_dim]`` — mean-pooled molecule
          representations.

    Example::

        encoder = MolGraphEncoder(config)
        atom_features, graph_repr = encoder(bmg)
        print(atom_features.shape)   # [N_atoms, 300]
        print(graph_repr.shape)      # [B, 300]
    """

    def __init__(self, config: CageFusionConfig) -> None:
        super().__init__()
        self.config = config

        # A minimal FFN predictor is required by ChemProp's MPNN constructor
        # but is never used in our forward pass — we extract internal features directly.
        _dummy_predictor = BinaryClassificationFFN(
            input_dim=config.graph_dim,
            n_tasks=config.num_labels,
            hidden_dim=128,
            n_layers=1,
            dropout=0.1,
            activation="ReLU",
        )
        self._mpnn = MPNN(
            message_passing=BondMessagePassing(),
            agg=MeanAggregation(),
            predictor=_dummy_predictor,
        )
        self._agg = MeanAggregation()

    def forward(self, bmg) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of molecules.

        Args:
            bmg: :class:`~chemprop.data.BatchMolGraph` on the correct device.

        Returns:
            Tuple of:
            - ``atom_features``: ``[total_atoms, graph_dim]``
            - ``graph_repr``   : ``[batch_size,  graph_dim]``
        """
        atom_features = self._mpnn.message_passing(bmg)        # [N, D]
        graph_repr = self._agg(atom_features, bmg.batch)       # [B, D]
        return atom_features, graph_repr
