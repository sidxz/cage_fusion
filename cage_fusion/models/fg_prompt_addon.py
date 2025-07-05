import torch
import torch.nn as nn
from rdkit import Chem

# Make sure this import path is correct for your project structure
from cage_fusion.engine.fg_utils import get_functional_groups, NUM_FUNCTIONAL_GROUPS


class FunctionalGroupPrompt(nn.Module):
    """
    A module to generate a chemically-aware prompt based on functional groups
    present in a molecule. This prompt enriches the graph representation before
    the final prediction.
    """

    def __init__(self, num_functional_groups: int, feature_dim: int):
        super(FunctionalGroupPrompt, self).__init__()
        self.feature_dim = feature_dim

        # 1. Learnable embedding table for all functional groups in the vocabulary.
        self.fg_embedding = nn.Embedding(num_functional_groups, feature_dim)

        # 2. A linear layer to fuse the static FG embedding with dynamic atom signals from the GNN.
        self.fusion_layer = nn.Linear(feature_dim * 2, feature_dim)

        # 3. Self-attention mechanism to create a single, weighted prompt vector per molecule.
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4, batch_first=True
        )

        # 4. A learnable query vector that acts like a [CLS] token to summarize the FGs.
        self.query_vector = nn.Parameter(torch.randn(1, 1, feature_dim))

    def forward(
        self,
        smiles_batch: list[str],
        atom_features: torch.Tensor,
        bmg,
        return_attn: bool = False,
    ):
        """
        Generates a functional group prompt for a batch of molecules.

        Args:
            smiles_batch: A list of SMILES strings for the batch.
            atom_features: The tensor of atom-level features from the GNN.
            bmg: The BatchMolGraph object, used to map atoms to molecules.
            return_attn: Flag to return attention weights for visualization.

        Returns:
            A tensor of prompt vectors for the batch. If return_attn is True, also
            returns a list of attention weight dictionaries.
        """
        batch_prompts = []
        batch_attn_weights = []  # Initialize list to store attention weights

        # --- FINAL PATCH: Iterate over the provided SMILES list ---
        for i, smiles in enumerate(smiles_batch):
            # a. Create an RDKit molecule object from the SMILES string.
            mol_object = Chem.MolFromSmiles(smiles)

            # If RDKit fails to parse the SMILES, use a zero prompt.
            if mol_object is None:
                batch_prompts.append(
                    torch.zeros(1, self.feature_dim, device=atom_features.device)
                )
                if return_attn:
                    batch_attn_weights.append({"fg_ids": [], "weights": []})
                continue

            fg_ids = get_functional_groups(mol_object)

            # If no functional groups are found, use a zero prompt.
            if not fg_ids:
                batch_prompts.append(
                    torch.zeros(1, self.feature_dim, device=atom_features.device)
                )
                if return_attn:
                    batch_attn_weights.append({"fg_ids": [], "weights": []})
                continue

            # b. Get the atom features corresponding to only the i-th molecule.
            # The `batch` attribute of bmg is a tensor mapping each atom to its molecule index.
            mol_atom_indices = (bmg.batch == i).nonzero(as_tuple=True)[0]
            mol_atom_features = atom_features[mol_atom_indices]

            # c. Look up embeddings for the functional groups present in this molecule.
            fg_ids_tensor = torch.tensor(
                fg_ids, dtype=torch.long, device=atom_features.device
            )
            static_fg_embeds = self.fg_embedding(fg_ids_tensor)

            # d. Use the mean of the molecule's atom features as the "atom signal".
            atom_signal = mol_atom_features.mean(dim=0, keepdim=True).expand_as(
                static_fg_embeds
            )

            # e. Fuse the static FG embeddings with the dynamic atom signal.
            fused_input = torch.cat([static_fg_embeds, atom_signal], dim=1)
            fused_fg_embeds = torch.relu(self.fusion_layer(fused_input))

            # f. Use self-attention to get a single prompt vector for the molecule.
            query = self.query_vector
            attn_input = torch.cat([query, fused_fg_embeds.unsqueeze(0)], dim=1)

            attn_output, attn_weights = self.attention(
                query=attn_input, key=attn_input, value=attn_input
            )

            prompt_vector = attn_output[:, 0, :]
            batch_prompts.append(prompt_vector)

            # g. If requested, store the attention weights for visualization.
            if return_attn:
                weights_to_fgs = attn_weights[0, 0, 1:].detach().cpu().numpy()
                batch_attn_weights.append({"fg_ids": fg_ids, "weights": weights_to_fgs})

        # Concatenate the list of prompts into a single tensor for the batch.
        final_prompts = torch.cat(batch_prompts, dim=0)

        if return_attn:
            return final_prompts, batch_attn_weights

        return final_prompts, None
