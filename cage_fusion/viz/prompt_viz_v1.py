import os
import io
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont

# Make sure to adjust the import paths if they differ in your project
from ..engine.fg_utils import FG_NAMES, FG_SMARTS
from ..utils.logging_utils import logger


def visualize_fg_attention(
    smiles: str,
    prompt_attn_weights: dict,
    output_path: str,
    title: str = "Top Functional Group Attentions",
    top_n: int = 3,
):
    """
    Generates a composite image showing the molecule with top N attended
    functional groups highlighted. Highlight color is a continuous block
    proportional to attention weight.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.warning(f"Could not generate molecule from SMILES: {smiles}")
        return

    fg_ids = prompt_attn_weights.get("fg_ids", [])
    weights = np.array(prompt_attn_weights.get("weights", []))

    if len(fg_ids) == 0 or len(weights) == 0:
        logger.info("No functional groups to visualize for this molecule.")
        return

    # --- 1. Top N Functional Groups and Normalize Weights ---
    # Ensure top_n is not greater than the number of available functional groups
    num_fgs = len(fg_ids)
    actual_top_n = min(top_n, num_fgs)

    top_indices = np.argsort(weights)[-actual_top_n:][::-1]
    top_weights = weights[top_indices]

    # Handle the case where all top weights are the same to avoid division by zero
    if top_weights.max() == top_weights.min():
        norm_weights = np.full_like(top_weights, 0.5)
    else:
        norm_weights = 0.2 + 0.8 * (
            (top_weights - top_weights.min()) / (top_weights.max() - top_weights.min())
        )

    top_fg_data = []
    for i, idx in enumerate(top_indices):
        fg_id = fg_ids[idx]
        # --- FIX: Access FG_NAMES by index, not with .get() ---
        fg_name = FG_NAMES[fg_id] if fg_id < len(FG_NAMES) else f"FG_{fg_id}"
        fg_pattern = FG_SMARTS.get(fg_name)

        if fg_pattern:
            top_fg_data.append(
                {
                    "name": fg_name,
                    "pattern": fg_pattern,
                    "attention": float(weights[idx]),
                    "norm_weight": float(norm_weights[i]),
                }
            )

    # --- 2. Drawing Highlights ---
    highlight_atoms, highlight_bonds = [], []
    atom_colors, bond_colors = {}, {}
    colormap = cm.get_cmap("Greens")

    for data in top_fg_data:
        # Ensure pattern is a Mol object
        patt = (
            Chem.MolFromSmarts(data["pattern"])
            if isinstance(data["pattern"], str)
            else data["pattern"]
        )
        if not patt:
            continue

        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue

        color = colormap(data["norm_weight"])
        data["display_color"] = color

        for match in matches:
            for atom_idx in match:
                if atom_idx not in atom_colors:  # Prioritize higher attention colors
                    atom_colors[atom_idx] = color

            for bond in mol.GetBonds():
                a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a1 in match and a2 in match and bond.GetIdx() not in bond_colors:
                    bond_colors[bond.GetIdx()] = color

    highlight_atoms = list(atom_colors.keys())
    highlight_bonds = list(bond_colors.keys())

    # --- 3. Render Molecule Image ---
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

    # --- 4. Composite with Legend ---
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        text_font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
        legend_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except IOError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        legend_font = ImageFont.load_default()

    header_height = 100
    legend_height_per_item = 45
    footer_height = len(top_fg_data) * legend_height_per_item + 60
    total_width = 800
    total_height = mol_image.height + header_height + footer_height

    final_image = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(final_image)

    # --- Header ---
    draw.text((30, 20), title, font=title_font, fill="black")
    draw.text((30, 60), f"SMILES: {smiles}", font=text_font, fill="dimgray")

    # --- Molecule Image ---
    final_image.paste(mol_image, (0, header_height), mol_image)

    # --- Legend ---
    y_cursor = header_height + mol_image.height + 20
    draw.text(
        (30, y_cursor),
        "Top Functional Group Highlights",
        font=title_font.font_variant(size=18),
        fill="black",
    )
    y_cursor += 35

    for data in top_fg_data:
        color_rgb = tuple(
            int(c * 255) for c in data.get("display_color", (0, 0, 0, 0))[:3]
        )
        draw.rectangle(
            [30, y_cursor, 50, y_cursor + 20], fill=color_rgb, outline="dimgray"
        )

        info_text = f"{data['name']} (Attention: {data['attention']:.1%})"
        draw.text((65, y_cursor), info_text, font=legend_font, fill="black")
        y_cursor += 22
        patt = (
            Chem.MolFromSmarts(data["pattern"])
            if isinstance(data["pattern"], str)
            else data["pattern"]
        )
        smarts_str = Chem.MolToSmarts(patt)
        draw.text(
            (65, y_cursor),
            f"SMARTS: {smarts_str}",
            font=text_font.font_variant(size=14),
            fill="gray",
        )
        y_cursor += 25

    # --- Save ---
    final_image.save(output_path)
    # logger.info(f"Saved detailed functional group attention to {output_path}")
