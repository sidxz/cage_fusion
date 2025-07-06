# In cage_fusion/viz/prompt_viz.py

import os
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.cm as cm
from PIL import Image
import matplotlib.pyplot as plt

from ..engine.fg_utils import FG_NAMES, FG_SMARTS
from cage_fusion.utils.logging_utils import logger


def visualize_fg_attention(
    smiles: str,
    prompt_attn_weights: dict,
    output_path: str,
    title: str = "Top 3 Functional Group Attentions",
    top_n: int = 3,
):
    """
    Generates an image of the molecule with the top N most attended
    functional groups highlighted.

    Args:
        smiles (str): The SMILES string of the molecule.
        prompt_attn_weights (dict): A dictionary from the model containing
                                    'fg_ids' and 'weights'.
        output_path (str): Path to save the visualization.
        title (str): The title for the plot (used as legend).
        top_n (int): The number of top functional groups to highlight.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.warning(f"Could not generate molecule from SMILES: {smiles}")
        return

    fg_ids = prompt_attn_weights.get("fg_ids", [])
    weights = prompt_attn_weights.get("weights", [])

    if not fg_ids:
        logger.info("No functional groups to visualize for this molecule.")
        return

    # --- Find Top N Functional Groups ---
    top_indices = np.argsort(weights)[-top_n:][::-1]
    top_fg_ids = [fg_ids[i] for i in top_indices]
    top_fg_names = [FG_NAMES[i] for i in top_fg_ids]

    # --- Prepare for Drawing ---
    highlight_atoms = []
    atom_colors = {}
    legend_entries = []

    # Use a color map to assign distinct colors
    colors = cm.get_cmap("viridis", top_n)

    for i, fg_id in enumerate(top_fg_ids):
        fg_name = top_fg_names[i]
        fg_pattern = FG_SMARTS.get(fg_name)
        if not fg_pattern:
            continue

        # Get the atom indices for the current functional group
        matches = mol.GetSubstructMatches(fg_pattern)
        fg_atoms = [atom_idx for match in matches for atom_idx in match]

        if fg_atoms:
            color = colors(i / (top_n - 1) if top_n > 1 else 1)
            legend_entries.append(f"{fg_name}")
            for atom_idx in fg_atoms:
                if atom_idx not in highlight_atoms:
                    highlight_atoms.append(atom_idx)
                    atom_colors[atom_idx] = color

    # --- Generate Image ---
    if not highlight_atoms:
        logger.info("No atoms to highlight for the top functional groups.")
        img = Draw.MolToImage(mol, size=(800, 600), legend=title)
        img.save(output_path)
        logger.info(f"Saved molecule image (no highlights) to {output_path}")
        return

    drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
    drawer.drawOptions().legendFontSize = 20
    drawer.drawOptions().addAtomIndices = True
    drawer.drawOptions().setHighlightColour((0.5, 0.5, 0.5, 0.5))

    d_mol = rdMolDraw2D.PrepareMolForDrawing(mol)

    drawer.DrawMolecule(
        d_mol,
        legend=", ".join(legend_entries),
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
    )

    drawer.FinishDrawing()
    drawer.WriteDrawingText(output_path)
    logger.info(f"Saved functional group attention visualization to {output_path}")
