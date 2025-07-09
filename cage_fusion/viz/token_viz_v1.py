import os
import torch
import numpy as np
import matplotlib
import h5py
import re

# --- Corrected RDKit and added Pillow imports ---
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from chemprop.data import BatchMolGraph
from cage_fusion.utils.logging_utils import logger


def visualize_attention_weights(
    attn_weights: torch.Tensor,
    mask: torch.Tensor,
    num_heads: int,
    output_path: str,
    input_ids: torch.Tensor = None,
    tokenizer_obj=None,
    smiles: str = None,  # New argument to accept the SMILES string
):
    """
    Visualizes graph-to-token attention weights across heads.
    """
    if attn_weights.ndim == 3 and attn_weights.shape[1] > 1:
        attn_weights = torch.mean(attn_weights, dim=1)

    seq_len = int(mask.sum().item())
    attn_weights = attn_weights[:, :seq_len]
    attn_np = torch.clamp(attn_weights, min=0.0).cpu().numpy()

    normalized_heads = []
    for head in attn_np:
        if np.isnan(head).all():
            normalized_heads.append(np.full_like(head, np.nan))
        else:
            normalized_heads.append(head / (head.sum() + 1e-8))
    normalized_heads = np.array(normalized_heads)

    averaged_attn = np.nanmean(normalized_heads, axis=0)
    if np.isnan(averaged_attn).all():
        logger.warning("Skipped attention visualization: All values are NaN.")
        return

    xtick_labels = None
    if tokenizer_obj and input_ids is not None:
        tokens = tokenizer_obj.convert_ids_to_tokens(input_ids.cpu().numpy()[:seq_len])
        tick_skip = 2 if seq_len > 100 else 1
        xtick_labels = [
            tok if i % tick_skip == 0 else "" for i, tok in enumerate(tokens)
        ]

    num_rows = 1 + num_heads
    fig, axs = plt.subplots(
        num_rows, 2, figsize=(20, 3.5 * num_rows), gridspec_kw={"width_ratios": [4, 2]}
    )
    axs = np.array(axs).reshape(num_rows, 2)

    # --- ADDED: Display the SMILES string as a title ---
    if smiles:
        fig.suptitle(f"Graph-to-Token Attention for: {smiles}", fontsize=16)

    def plot_head(ax_row, data, title):
        sns.heatmap(
            data.reshape(1, -1), cmap="viridis", ax=ax_row[0], xticklabels=xtick_labels
        )
        ax_row[0].set_title(title)
        ax_row[1].hist(data, bins=20, color="skyblue")
        ax_row[1].set_title(f"{title} Distribution")

    plot_head(axs[0], averaged_attn, "Averaged Attention")

    for i, head_data in enumerate(normalized_heads):
        if not np.isnan(head_data).all():
            plot_head(axs[i + 1], head_data, f"Head {i}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the title
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def visualize_top_token_attentions(
    smiles: str,
    attention_weights: torch.Tensor,
    full_token_list: list,
    top_token_indices: np.ndarray,
    output_dir: str,
    top_k: int = 5,
):
    """
    Generates a final, high-resolution composite image summarizing attention.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.warning(f"Could not generate molecule from SMILES: {smiles}")
        return
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return

    # --- Step 1: Gather data for all plots ---
    grid_plots_data = []
    combined_atom_weights = {i: 0.0 for i in range(num_atoms)}
    all_source_atoms = set()
    attention_colormap = cm.get_cmap("Greens")
    source_token_color = (0.8, 0.4, 0.4)  # Red

    for token_idx in top_token_indices:
        token_str = full_token_list[token_idx].replace("##", "")
        weights = attention_weights[:, token_idx, :num_atoms].mean(axis=0).cpu().numpy()
        if weights.size == 0:
            continue

        for i, w in enumerate(weights):
            combined_atom_weights[i] = max(combined_atom_weights[i], w)

        k = min(top_k, num_atoms)
        top_atoms = np.argsort(weights)[-k:].tolist() if k > 0 else []

        plot_data = _prepare_plot_data(
            mol, token_str, top_atoms, weights, attention_colormap, source_token_color
        )
        grid_plots_data.append(plot_data)
        all_source_atoms.update(plot_data["source_atoms"])

    # --- Step 2: Generate intermediate images ---

    # Image 1: Grid of individual attention plots
    grid_img_path = os.path.join(output_dir, "temp_grid.png")
    mols_for_grid = [Chem.Mol(mol) for _ in grid_plots_data]
    legends_for_grid = [
        f"From token: '{full_token_list[idx].replace('##','')}'"
        for idx in top_token_indices
    ]

    Draw.MolsToGridImage(
        mols_for_grid,
        molsPerRow=len(mols_for_grid),
        subImgSize=(500, 500),
        legends=legends_for_grid,
        highlightAtomLists=[p["atoms"] for p in grid_plots_data],
        highlightBondLists=[p["bonds"] for p in grid_plots_data],
        highlightAtomColors=[p["atom_colors"] for p in grid_plots_data],
        highlightBondColors=[p["bond_colors"] for p in grid_plots_data],
    ).save(grid_img_path)

    # Image 2: Combined summary plot
    combined_img_path = os.path.join(output_dir, "temp_combined.png")
    all_top_attention_atoms = set(a for p in grid_plots_data for a in p["top_atoms"])

    combined_data = _prepare_plot_data(
        mol,
        None,
        list(all_top_attention_atoms),
        np.array(list(combined_atom_weights.values())),
        attention_colormap,
        source_token_color,
        all_source_atoms,
    )

    drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
    drawer.drawOptions().addAtomIndices = True

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        legend="Combined Attention from Top Tokens",
        highlightAtoms=combined_data["atoms"],
        highlightBonds=combined_data["bonds"],
        highlightAtomColors=combined_data["atom_colors"],
        highlightBondColors=combined_data["bond_colors"],
    )
    drawer.FinishDrawing()
    drawer.WriteDrawingText(combined_img_path)

    # Image 3: Text Image with Highlighted SMILES
    text_img_path = os.path.join(output_dir, "temp_text.png")
    _create_highlighted_smiles_image(full_token_list, top_token_indices, text_img_path)

    # --- Stitch images together ---
    _stitch_images(combined_img_path, text_img_path, grid_img_path, output_dir)


def _prepare_plot_data(
    mol,
    token_str,
    top_atoms,
    weights,
    attention_cmap,
    source_color,
    precomputed_source_atoms=None,
):
    """Helper to calculate highlight details for a single RDKit plot."""
    atom_colors = {}
    bond_colors = {}

    # Apply graded green color for top attention atoms
    if top_atoms:
        top_weights = weights[top_atoms]
        norm_top_weights = (
            (top_weights - top_weights.min())
            / (top_weights.max() - top_weights.min() + 1e-8)
            if top_weights.size > 1
            else np.array([1.0])
        )
        for i, atom_idx in enumerate(top_atoms):
            atom_colors[atom_idx] = attention_cmap(norm_top_weights[i])

    # Apply red highlight for source token atoms, overwriting green if necessary
    source_atoms = set()
    if precomputed_source_atoms is not None:
        source_atoms = precomputed_source_atoms
    elif token_str and source_color:
        try:
            smarts_query = (
                f"[{token_str}]"
                if len(token_str) == 1 and token_str.isalpha()
                else token_str
            )
            query_mol = Chem.MolFromSmarts(smarts_query)
            if query_mol:
                match = mol.GetSubstructMatch(query_mol)
                if match:
                    source_atoms.update(match)
        except Exception:
            pass

    for atom_idx in source_atoms:
        atom_colors[atom_idx] = source_color

    highlight_atoms = list(atom_colors.keys())

    # Find all bonds connecting any two highlighted atoms
    highlight_bonds = [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in highlight_atoms
        and b.GetEndAtomIdx() in highlight_atoms
    ]

    # Color bonds based on the atoms they connect
    for bond_idx in highlight_bonds:
        begin_idx = mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx()
        end_idx = mol.GetBondWithIdx(bond_idx).GetEndAtomIdx()

        is_source_bond = begin_idx in source_atoms and end_idx in source_atoms
        is_attention_bond = begin_idx in top_atoms and end_idx in top_atoms

        if is_source_bond:
            bond_colors[bond_idx] = source_color
        elif is_attention_bond:
            c1 = mcolors.to_rgb(atom_colors.get(begin_idx, (1, 1, 1)))
            c2 = mcolors.to_rgb(atom_colors.get(end_idx, (1, 1, 1)))
            bond_colors[bond_idx] = tuple(np.mean([c1, c2], axis=0))

    return {
        "atoms": highlight_atoms,
        "bonds": highlight_bonds,
        "atom_colors": atom_colors,
        "bond_colors": bond_colors,
        "top_atoms": top_atoms,
        "source_atoms": source_atoms,
    }


def _create_highlighted_smiles_image(full_token_list, top_token_indices, output_path):
    """Helper to create an image of the SMILES string with top tokens highlighted."""
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    top_indices_set = set(top_token_indices)
    segments = [
        (token, "red" if i in top_indices_set else "black")
        for i, token in enumerate(full_token_list)
    ]

    reconstructed_smiles = "".join([s.replace("##", "") for s in full_token_list])
    text_img_width = font.getbbox(f"SMILES: {reconstructed_smiles}")[2] + 40
    text_img_height = font.getbbox("SMILES:")[3] + 20

    text_img = Image.new("RGB", (text_img_width, text_img_height), "white")
    draw_text = ImageDraw.Draw(text_img)

    x_cursor = 10
    y_cursor = 10
    draw_text.text((x_cursor, y_cursor), "SMILES: ", font=font, fill="black")
    x_cursor += font.getbbox("SMILES: ")[2]

    for text, color in segments:
        display_text = text.replace("##", "")
        draw_text.text((x_cursor, y_cursor), display_text, font=font, fill=color)
        x_cursor += font.getbbox(display_text)[2]

    text_img.save(output_path)


def _stitch_images(img1_path, img2_path, img3_path, output_dir):
    """Helper to stitch three images vertically."""
    try:
        paths = [img1_path, img2_path, img3_path]
        images = [Image.open(p) for p in paths if os.path.exists(p)]
        if not images:
            logger.warning("No images found for stitching.")
            return

        total_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)

        new_im = Image.new("RGB", (total_width, total_height), "white")

        current_y = 0
        for img in images:
            new_im.paste(img, (int((total_width - img.width) / 2), current_y))
            current_y += img.height

        final_output_path = os.path.join(output_dir, "attention_summary.png")
        new_im.save(final_output_path)
        #logger.info(f"Saved final composite visualization to: {final_output_path}")

    finally:
        # Clean up temporary files
        for p in [img1_path, img2_path, img3_path]:
            if os.path.exists(p):
                os.remove(p)
