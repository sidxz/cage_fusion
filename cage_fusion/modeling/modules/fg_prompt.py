"""
Functional-Group Prompt module.

Generates a chemically-aware prompt vector for each molecule by:
  1. Detecting SMARTS-defined functional groups in the SMILES.
  2. Looking up learnable FG embeddings and fusing them with the mean
     atom-feature signal from the GNN.
  3. Using attention (learnable CLS query over FG keys) to summarise
     the FG information into a single prompt vector.

The prompt is added to the graph representation in the main model with
a learnable scale factor (alpha).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from typing import List, Tuple, Optional

_rdlog = RDLogger.logger()


def _mol_from_smiles_quiet(smiles: str):
    """Parse SMILES while suppressing RDKit sanitisation warnings.

    SMILES arriving here were already validated during featurisation, so any
    remaining warnings (e.g. 'not removing hydrogen atom without neighbors')
    are cosmetic and should not pollute training logs.
    """
    _rdlog.setLevel(RDLogger.CRITICAL)
    try:
        mol = Chem.MolFromSmiles(smiles)
    finally:
        _rdlog.setLevel(RDLogger.WARNING)
    return mol


class FunctionalGroupPrompt(nn.Module):
    """
    Chemically-aware prompt enrichment for the graph representation.

    Parameters
    ----------
    num_functional_groups:
        Size of the FG vocabulary (number of SMARTS patterns loaded).
    feature_dim:
        Dimension of atom/graph features (must match graph_dim in config).
    """

    def __init__(self, num_functional_groups: int, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        # Learnable embedding table: one vector per FG in the vocabulary
        self.fg_embedding = nn.Embedding(num_functional_groups, feature_dim)

        # Fuse static FG embedding with dynamic atom signal from the GNN
        self.fusion_layer = nn.Linear(feature_dim * 2, feature_dim)

        # Attention: CLS-like query summarises all present FGs
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4, batch_first=True
        )

        # Learnable [CLS]-like query vector (shared across the batch)
        self.query_vector = nn.Parameter(torch.randn(1, 1, feature_dim))

    def forward(
        self,
        smiles_batch: List[str],
        atom_features: torch.Tensor,   # (total_atoms, feature_dim)
        bmg,                           # BatchMolGraph; bmg.batch maps atom→mol
        fg_detector,                   # callable: (rdkit.Mol) -> List[int]
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Parameters
        ----------
        smiles_batch:
            List of SMILES strings, one per molecule in the batch.
        atom_features:
            Flat tensor of atom features from the GNN message passing step.
        bmg:
            BatchMolGraph whose ``.batch`` attribute maps each atom to its
            molecule index in the batch.
        fg_detector:
            Callable that takes an RDKit Mol and returns a list of integer FG
            indices (0-indexed into the FG vocabulary).
        return_attn:
            If True, also return per-molecule attention weight dicts.

        Returns
        -------
        final_prompts:
            Tensor of shape ``(batch_size, feature_dim)``.
        batch_attn_weights:
            If ``return_attn=True``, a list (len=batch) of dicts::

                {"fg_ids": List[int], "weights": np.ndarray shape (num_FGs,)}

            Otherwise ``None``.
        """
        device = atom_features.device
        batch_prompts: List[torch.Tensor] = []
        batch_attn_weights: Optional[list] = [] if return_attn else None

        for i, smiles in enumerate(smiles_batch):
            try:
                mol = _mol_from_smiles_quiet(smiles)
                if mol is None:
                    batch_prompts.append(
                        torch.zeros(1, self.feature_dim, device=device)
                    )
                    if return_attn:
                        batch_attn_weights.append({"fg_ids": [], "weights": []})
                    continue

                fg_ids = fg_detector(mol)
                if not fg_ids:
                    batch_prompts.append(
                        torch.zeros(1, self.feature_dim, device=device)
                    )
                    if return_attn:
                        batch_attn_weights.append({"fg_ids": [], "weights": []})
                    continue

                # Gather atom features for this molecule
                mol_atom_idx = (bmg.batch == i).nonzero(as_tuple=True)[0]
                mol_atom_feats = atom_features[mol_atom_idx]
                if mol_atom_feats.numel() == 0:
                    batch_prompts.append(
                        torch.zeros(1, self.feature_dim, device=device)
                    )
                    if return_attn:
                        batch_attn_weights.append({"fg_ids": fg_ids, "weights": []})
                    continue

                # Static FG embeddings
                fg_ids_t = torch.tensor(fg_ids, dtype=torch.long, device=device)
                static_fg_embeds = self.fg_embedding(fg_ids_t)          # (m, d)

                # Dynamic atom signal: mean-pool over atoms
                atom_signal = mol_atom_feats.mean(dim=0, keepdim=True)  # (1, d)
                if torch.isnan(atom_signal).any():
                    batch_prompts.append(
                        torch.zeros(1, self.feature_dim, device=device)
                    )
                    if return_attn:
                        batch_attn_weights.append({"fg_ids": fg_ids, "weights": []})
                    continue

                # Fuse static FG embedding with dynamic atom signal
                atom_signal_exp = atom_signal.expand_as(static_fg_embeds)  # (m, d)
                fused_input = torch.cat(
                    [static_fg_embeds, atom_signal_exp], dim=1
                )                                                           # (m, 2d)
                fused_fg = torch.relu(self.fusion_layer(fused_input))       # (m, d)

                # Attention: CLS query attends over FG keys/values
                query = self.query_vector.to(device)                        # (1, 1, d)
                keys = fused_fg.unsqueeze(0)                                # (1, m, d)
                attn_out, attn_w = self.attention(
                    query=query, key=keys, value=keys
                )                                                           # (1,1,d), (1,1,m)
                prompt_vector = attn_out[:, 0, :]                          # (1, d)
                batch_prompts.append(prompt_vector)

                if return_attn:
                    weights_np = attn_w[0, 0, :].detach().cpu().numpy()   # (m,)
                    s = weights_np.sum()
                    if s > 0:
                        weights_np = weights_np / s
                    batch_attn_weights.append(
                        {"fg_ids": fg_ids, "weights": weights_np}
                    )

            except Exception:
                batch_prompts.append(torch.zeros(1, self.feature_dim, device=device))
                if return_attn:
                    batch_attn_weights.append({"fg_ids": [], "weights": []})

        if not batch_prompts:
            empty = torch.zeros(0, self.feature_dim, device=device)
            return empty, (batch_attn_weights if return_attn else None)

        return (
            torch.cat(batch_prompts, dim=0),
            batch_attn_weights if return_attn else None,
        )
