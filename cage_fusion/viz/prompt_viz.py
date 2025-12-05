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


def minmax_neg1_1(x):
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
    highlight_red: bool = True,   # show groups with positive coeffs
    highlight_blue: bool = False,  # show groups with negative coeffs
):
    """
    Visualize functional-group attention with optional polarity toggles.
    - Red  = coefficient > 0 (above-average attention)
    - Blue = coefficient < 0 (below-average attention)
    If top_n > 0, picks top_n per enabled polarity (most + and most -).
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

    # Normalize and center to get coefficients
    weights = minmax_neg1_1(weights)  # [-1, 1]
    avg = float(np.mean(weights))
    coeffs = weights - avg

    # Build full FG list (id, name, pattern, coeff)
    top_fg_data_all = []
    for i, fg_id in enumerate(fg_ids):
        fg_name = FG_NAMES[fg_id] if fg_id < len(FG_NAMES) else f"FG_{fg_id}"
        patt = FG_SMARTS.get(fg_name)
        top_fg_data_all.append(
            {"id": int(fg_id), "name": fg_name, "pattern": patt, "coefficient": float(coeffs[i])}
        )

    # Filter by polarity toggles
    red_items = [d for d in top_fg_data_all if d["coefficient"] > 0] if highlight_red else []
    blue_items = [d for d in top_fg_data_all if d["coefficient"] < 0] if highlight_blue else []

    # Polarity-aware top_n selection
    if top_n and top_n > 0:
        red_items = sorted(red_items, key=lambda d: d["coefficient"], reverse=True)[:top_n]
        blue_items = sorted(blue_items, key=lambda d: d["coefficient"])[:top_n]  # most negative
    # If both toggles are off, show nothing (just molecule)
    selected = red_items + blue_items

    # --- Color + highlight prep ---
    atom_colors, bond_colors = {}, {}
    highlight_atoms, highlight_bonds = [], []
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

            # Map coefficient to [0,1] for bwr
            norm_coeff = 0.5 * (data["coefficient"] / (max_abs + 1e-8) + 1.0)
            color = colormap(norm_coeff)

            for match in matches:
                for a in match:
                    if a not in atom_colors:
                        atom_colors[a] = color
                # bonds fully inside the match
                for bond in mol.GetBonds():
                    a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if a1 in match and a2 in match and bond.GetIdx() not in bond_colors:
                        bond_colors[bond.GetIdx()] = color

        highlight_atoms = list(atom_colors.keys())
        highlight_bonds = list(bond_colors.keys())

    # --- Draw molecule ---
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

    # --- Compose legend only for selected items ---
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
    footer_height = (len(selected) * legend_height_per_item + 60) if selected else 40
    total_width = 800
    total_height = mol_image.height + header_height + footer_height

    final_image = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(final_image)

    # Molecule
    final_image.paste(mol_image, (0, header_height), mol_image)

    # Legend
    y_cursor = header_height + mol_image.height + 20
    draw.text(
        (30, y_cursor),
        "Top Influential Functional Groups" if selected else "No groups highlighted",
        font=title_font.font_variant(size=18) if hasattr(title_font, "font_variant") else title_font,
        fill="black",
    )
    y_cursor += 35

    for data in selected:
        # Recompute color chip (same mapping used above)
        # Safer to rebuild to avoid carrying display state in dict
        max_abs = max(abs(d["coefficient"]) for d in selected) or 1.0
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
        draw.text((65, y_cursor), f"SMARTS: {smarts_str}",
                  font=text_font if not hasattr(text_font, "font_variant") else text_font.font_variant(size=14),
                  fill="gray")
        y_cursor += 25

    # Save
    outdir = os.path.dirname(output_path) or "."
    os.makedirs(outdir, exist_ok=True)
    final_image.save(output_path)

