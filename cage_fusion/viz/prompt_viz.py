import os
import io
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont

# Make sure to adjust the import paths if they differ in your project
from ..engine.fg_utils import FG_NAMES, FG_SMARTS
from ..utils.logging_utils import logger


def visualize_fg_attention(
    smiles: str,
    prompt_attn_weights: dict,
    output_path: str,
    title: str = "Functional Group Attention Coefficients",
    top_n: int = 5,
):
    """
    Generates a composite image showing the molecule with functional groups
    highlighted based on their attention coefficients, as described in
    Tang et al. J Cheminform (2020) 12:15.

    The color indicates whether a group's contribution is above (red) or
    below (blue) the average attention for this molecule.
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

    # --- 1. Calculate Attention Coefficients ---
    average_attention = np.mean(weights)
    attention_coefficients = weights - average_attention

    num_fgs = len(fg_ids)
    actual_top_n = min(top_n, num_fgs)

    # Sort by the *magnitude* of the coefficient to find the most influential FGs
    top_indices = np.argsort(np.abs(attention_coefficients))[-actual_top_n:][::-1]

    top_fg_data = []
    for idx in top_indices:
        fg_id = fg_ids[idx]
        fg_name = FG_NAMES[fg_id] if fg_id < len(FG_NAMES) else f"FG_{fg_id}"
        fg_pattern = FG_SMARTS.get(fg_name)

        if fg_pattern:
            top_fg_data.append(
                {
                    "name": fg_name,
                    "pattern": fg_pattern,
                    "coefficient": float(attention_coefficients[idx]),
                }
            )

    # --- 2. Drawing Highlights using a Diverging Colormap ---
    highlight_atoms, highlight_bonds = [], []
    atom_colors, bond_colors = {}, {}
    # Use a blue-white-red colormap, perfect for showing deviation from a central point (zero)
    colormap = cm.get_cmap("bwr")

    # Find the maximum absolute coefficient to normalize the color scale from -max_abs to +max_abs
    max_abs_coeff = (
        max(abs(d["coefficient"]) for d in top_fg_data) if top_fg_data else 1.0
    )

    for data in top_fg_data:
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

        # Normalize coefficient from -max_abs to +max_abs -> 0 to 1 for the colormap
        # 0 -> blue, 0.5 -> white, 1.0 -> red
        norm_coeff = 0.5 * (data["coefficient"] / (max_abs_coeff + 0.00000001) + 1.0)
        color = colormap(norm_coeff)
        data["display_color"] = color

        for match in matches:
            for atom_idx in match:
                if atom_idx not in atom_colors:
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
    #draw.text((30, 20), title, font=title_font, fill="black")
    #draw.text((30, 60), f"SMILES: {smiles}", font=text_font, fill="dimgray")

    # --- Molecule Image ---
    final_image.paste(mol_image, (0, header_height), mol_image)

    # --- Legend ---
    y_cursor = header_height + mol_image.height + 20
    draw.text(
        (30, y_cursor),
        "Top Influential Functional Groups (by Attention Coefficient)",
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

        sign = "+" if data["coefficient"] >= 0 else ""
        info_text = f"{data['name']} (Coefficient: {sign}{data['coefficient']:.3f})"
        draw.text((65, y_cursor), info_text, font=legend_font, fill="black")
        y_cursor += 22

        # Display SMARTS for clarity
        patt = (
            Chem.MolFromSmarts(data["pattern"])
            if isinstance(data["pattern"], str)
            else data["pattern"]
        )
        smarts_str = Chem.MolToSmarts(patt) if patt else "N/A"
        draw.text(
            (65, y_cursor),
            f"SMARTS: {smarts_str}",
            font=text_font.font_variant(size=14),
            fill="gray",
        )
        y_cursor += 25

    # --- Save ---
    final_image.save(output_path)
