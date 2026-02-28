"""
cage_fusion/visualization/functional_groups.py

Functional-group attention visualizations.

Public API
----------
- ``visualize_fg_attention``: Atom-highlighted molecule with FG coefficient legend.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from cage_fusion.chemistry.fg_utils import FG_NAMES, _FG_SMARTS

logger = logging.getLogger(__name__)


def _minmax_neg1_1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    minv, maxv = x.min(), x.max()
    if np.isclose(maxv, minv):
        return np.zeros_like(x)
    return 2 * (x - minv) / (maxv - minv + 1e-8) - 1


def visualize_fg_attention(
    smiles: str,
    prompt_attn_weights: dict,
    output_path: str,
    title: str = "Functional Group Attention Coefficients",
    top_n: int = 5,
    highlight_red: bool = True,
    highlight_blue: bool = False,
):
    """
    Visualize functional-group attention with polarity toggles.

    Atoms belonging to highlighted functional groups are colored:
    - Red  = coefficient > 0 (above-average attention).
    - Blue = coefficient < 0 (below-average attention).

    If ``top_n > 0``, selects the ``top_n`` groups per enabled polarity.

    Args:
        smiles: SMILES string for the molecule.
        prompt_attn_weights: Dict with keys ``fg_ids`` (list[int]) and
                             ``weights`` (list[float]).
        output_path: PNG save path.
        title: Figure title.
        top_n: Maximum functional groups to highlight per polarity.
        highlight_red: Show groups with positive coefficients.
        highlight_blue: Show groups with negative coefficients.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.warning(f"Could not parse SMILES: {smiles}")
        return

    fg_ids = prompt_attn_weights.get("fg_ids", [])
    weights = np.array(prompt_attn_weights.get("weights", []))
    if len(fg_ids) == 0 or len(weights) == 0:
        logger.info("No functional groups to visualize for this molecule.")
        return

    # Center and normalize weights to coefficients
    weights = _minmax_neg1_1(weights)
    avg = float(np.mean(weights))
    coeffs = weights - avg

    # Build full FG list
    all_fg = []
    for i, fg_id in enumerate(fg_ids):
        fg_name = FG_NAMES[fg_id] if fg_id < len(FG_NAMES) else f"FG_{fg_id}"
        patt = _FG_SMARTS.get(fg_name)
        all_fg.append({
            "id": int(fg_id),
            "name": fg_name,
            "pattern": patt,
            "coefficient": float(coeffs[i]),
        })

    red_items = [d for d in all_fg if d["coefficient"] > 0] if highlight_red else []
    blue_items = [d for d in all_fg if d["coefficient"] < 0] if highlight_blue else []

    if top_n and top_n > 0:
        red_items = sorted(red_items, key=lambda d: d["coefficient"], reverse=True)[:top_n]
        blue_items = sorted(blue_items, key=lambda d: d["coefficient"])[:top_n]

    selected = red_items + blue_items

    # Atom / bond highlight prep
    atom_colors: dict = {}
    bond_colors: dict = {}
    highlight_atoms: list = []
    highlight_bonds: list = []
    colormap = cm.get_cmap("bwr")

    if selected:
        max_abs = max(abs(d["coefficient"]) for d in selected) or 1.0

        for data in selected:
            patt = (
                Chem.MolFromSmarts(data["pattern"])
                if isinstance(data["pattern"], str)
                else (data["pattern"] if isinstance(data["pattern"], Chem.Mol) else None)
            )
            if not patt:
                continue
            matches = mol.GetSubstructMatches(patt)
            if not matches:
                continue

            norm_coeff = 0.5 * (data["coefficient"] / (max_abs + 1e-8) + 1.0)
            color = colormap(norm_coeff)

            for match in matches:
                for a in match:
                    if a not in atom_colors:
                        atom_colors[a] = color
                for bond in mol.GetBonds():
                    a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if a1 in match and a2 in match and bond.GetIdx() not in bond_colors:
                        bond_colors[bond.GetIdx()] = color

        highlight_atoms = list(atom_colors.keys())
        highlight_bonds = list(bond_colors.keys())

    # Draw molecule
    mol_drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
    opts = mol_drawer.drawOptions()
    opts.addAtomIndices = True
    opts.setHighlightColour((0.8, 0.8, 0.8, 0.4))
    opts.highlightBondWidthMultiplier = 12
    opts.clearBackground = False

    rdMolDraw2D.PrepareAndDrawMolecule(
        mol_drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    mol_drawer.FinishDrawing()
    mol_image = Image.open(io.BytesIO(mol_drawer.GetDrawingText()))

    # Compose legend
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        text_font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
        legend_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except IOError:
        title_font = text_font = legend_font = ImageFont.load_default()

    header_height = 100
    legend_height_per_item = 45
    footer_height = (len(selected) * legend_height_per_item + 60) if selected else 40
    total_width = 800
    total_height = mol_image.height + header_height + footer_height

    final_image = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(final_image)
    final_image.paste(mol_image, (0, header_height), mol_image)

    y_cursor = header_height + mol_image.height + 20
    draw.text(
        (30, y_cursor),
        "Top Influential Functional Groups" if selected else "No groups highlighted",
        font=title_font,
        fill="black",
    )
    y_cursor += 35

    if selected:
        max_abs = max(abs(d["coefficient"]) for d in selected) or 1.0
        for data in selected:
            norm_coeff = 0.5 * (data["coefficient"] / (max_abs + 1e-8) + 1.0)
            rgba = colormap(norm_coeff)
            color_rgb = tuple(int(c * 255) for c in rgba[:3])

            draw.rectangle([30, y_cursor, 50, y_cursor + 20], fill=color_rgb, outline="dimgray")
            sign = "+" if data["coefficient"] >= 0 else ""
            info_text = f"{data['name']} (Coefficient: {sign}{data['coefficient']:.3f})"
            draw.text((65, y_cursor), info_text, font=legend_font, fill="black")
            y_cursor += 22

            patt = (
                Chem.MolFromSmarts(data["pattern"])
                if isinstance(data["pattern"], str)
                else (data["pattern"] if isinstance(data["pattern"], Chem.Mol) else None)
            )
            smarts_str = Chem.MolToSmarts(patt) if patt else "N/A"
            draw.text((65, y_cursor), f"SMARTS: {smarts_str}", font=text_font, fill="gray")
            y_cursor += 25

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final_image.save(output_path)
