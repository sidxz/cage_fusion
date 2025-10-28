import torch
import torch.nn as nn
from rdkit import Chem
from typing import List, Tuple, Optional

# Make sure this import path is correct for your project structure
from cage_fusion.engine.fg_utils import get_functional_groups, NUM_FUNCTIONAL_GROUPS


class FunctionalGroupPrompt(nn.Module):
    """
    A module to generate a chemically-aware prompt based on functional groups
    present in a molecule. This prompt enriches the graph representation before
    the final prediction.

    Key change vs. previous version:
    - Attention query is the learnable CLS-like vector.
    - Keys/values are ONLY the FG embeddings (CLS is NOT in the key set).
      => Reported attention weights sum to 1 over present FGs.
    """

    def __init__(self, num_functional_groups: int, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

        # 1) Learnable embedding table for all functional groups in the vocabulary.
        self.fg_embedding = nn.Embedding(num_functional_groups, feature_dim)

        # 2) Fuse static FG embedding with dynamic atom signals from the GNN.
        self.fusion_layer = nn.Linear(feature_dim * 2, feature_dim)

        # 3) Attention to summarize FGs into a single prompt vector.
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4, batch_first=True
        )

        # 4) Learnable query vector acting like a [CLS] token (one per batch).
        self.query_vector = nn.Parameter(torch.randn(1, 1, feature_dim))

    def forward(
        self,
        smiles_batch: List[str],
        atom_features: torch.Tensor,  # shape: (total_atoms, d)
        bmg,  # batch vector for atoms: bmg.batch (len = total_atoms)
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Returns
        -------
        final_prompts : Tensor
            Shape (batch_size, feature_dim)
        batch_attn_weights : Optional[list]
            If return_attn=True, a list (len=batch) of dicts:
              {"fg_ids": List[int], "weights": np.ndarray of shape (num_FGs,)}
            else None.
        """
        device = atom_features.device
        batch_prompts = []
        batch_attn_weights = [] if return_attn else None

        for i, smiles in enumerate(smiles_batch):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    # Invalid SMILES → zero prompt
                    batch_prompts.append(
                        torch.zeros(1, self.feature_dim, device=device)
                    )
                    if return_attn:
                        batch_attn_weights.append({"fg_ids": [], "weights": []})
                    continue

                fg_ids = get_functional_groups(mol)  # List[int] of FG indices
                if not fg_ids:
                    # No FGs found → zero prompt
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
                fg_ids_tensor = torch.tensor(fg_ids, dtype=torch.long, device=device)
                static_fg_embeds = self.fg_embedding(fg_ids_tensor)  # (m, d)

                # Dynamic atom signal (simple mean pool over atoms)
                atom_signal = mol_atom_feats.mean(dim=0, keepdim=True)  # (1, d)
                if torch.isnan(atom_signal).any():
                    batch_prompts.append(
                        torch.zeros(1, self.feature_dim, device=device)
                    )
                    if return_attn:
                        batch_attn_weights.append({"fg_ids": fg_ids, "weights": []})
                    continue

                # Expand atom signal to one per FG and fuse with static FG embedding
                atom_signal_expanded = atom_signal.expand_as(static_fg_embeds)  # (m, d)
                fused_input = torch.cat(
                    [static_fg_embeds, atom_signal_expanded], dim=1
                )  # (m, 2d)
                fused_fg_embeds = torch.relu(self.fusion_layer(fused_input))  # (m, d)

                # ---- Attention summarization (CLS queries FG-only keys) ----
                # query: (1, 1, d), keys/values: (1, m, d)
                query = self.query_vector.to(device)
                keys = fused_fg_embeds.unsqueeze(0)  # add batch dim: (1, m, d)

                attn_output, attn_weights = self.attention(
                    query=query,  # (1, 1, d)
                    key=keys,  # (1, m, d)
                    value=keys,  # (1, m, d)
                    # average_attn_weights=True by default → (1, 1, m)
                )
                # Prompt vector for this molecule
                prompt_vector = attn_output[:, 0, :]  # (1, d)
                batch_prompts.append(prompt_vector)

                if return_attn:
                    # Head-averaged weights over FGs only; sums to 1.0
                    weights_to_fgs = (
                        attn_weights[0, 0, :].detach().cpu().numpy()
                    )  # shape (m,)
                    # (Optional safety) tiny renorm to handle rare FP rounding:
                    s = weights_to_fgs.sum()
                    if s > 0:
                        weights_to_fgs = weights_to_fgs / s
                    batch_attn_weights.append(
                        {"fg_ids": fg_ids, "weights": weights_to_fgs}
                    )

            except Exception as e:
                # Robust fallback on any per-molecule error
                batch_prompts.append(torch.zeros(1, self.feature_dim, device=device))
                if return_attn:
                    batch_attn_weights.append({"fg_ids": [], "weights": []})
                continue

        if not batch_prompts:
            return torch.zeros(0, self.feature_dim, device=device), (
                batch_attn_weights if return_attn else None
            )

        final_prompts = torch.cat(batch_prompts, dim=0)  # (B, d)
        return final_prompts, (batch_attn_weights if return_attn else None)
