import os
import torch
import numpy as np
import matplotlib
import h5py

# --- Corrected RDKit imports ---
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont

# -------------------------------

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from chemprop.data import BatchMolGraph
from cage_fusion.utils.logging_utils import logger


def move_bmg_to_device(bmg: BatchMolGraph, device: torch.device) -> BatchMolGraph:
    """
    Transfers a BatchMolGraph object to the specified device.
    """
    for attr in ["V", "E", "edge_index", "batch"]:
        setattr(bmg, attr, getattr(bmg, attr).to(device))
    return bmg


def visualize_attention_weights(
    attn_weights: torch.Tensor,
    mask: torch.Tensor,
    num_heads: int,
    output_path: str,
    input_ids: torch.Tensor = None,
    tokenizer_obj=None,
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

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
    Generates a final composite image showing a summary of attention.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.warning(f"Could not generate molecule from SMILES: {smiles}")
        return
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return

    # --- Setup for collecting data for all plots ---
    mols_for_grid = []
    legends_for_grid = []
    grid_highlights = []
    combined_atom_weights = {}

    attention_colormap = cm.get_cmap("Greens")

    for token_idx in top_token_indices:
        token_str = full_token_list[token_idx]
        weights = attention_weights[:, token_idx, :num_atoms].mean(axis=0).cpu().numpy()
        if weights.size == 0:
            continue

        for i, w in enumerate(weights):
            combined_atom_weights[i] = max(combined_atom_weights.get(i, 0.0), w)

        k = min(top_k, num_atoms)
        top_atoms_indices = np.argsort(weights)[-k:].tolist() if k > 0 else []
        top_atoms_set = set(top_atoms_indices)
        top_bonds_indices = [
            b.GetIdx()
            for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in top_atoms_set
            and b.GetEndAtomIdx() in top_atoms_set
        ]

        mols_for_grid.append(Chem.Mol(mol))
        legends_for_grid.append(f"From token: '{token_str}'")
        grid_highlights.append({"atoms": top_atoms_indices, "bonds": top_bonds_indices})

    # --- Generate Intermediate Images ---

    # Image 1: Grid of individual plots
    grid_img_path = os.path.join(output_dir, "temp_grid.png")
    if mols_for_grid:
        grid_image = Draw.MolsToGridImage(
            mols_for_grid,
            molsPerRow=len(mols_for_grid),
            subImgSize=(400, 400),
            legends=legends_for_grid,
            highlightAtomLists=[h["atoms"] for h in grid_highlights],
            highlightBondLists=[h["bonds"] for h in grid_highlights],
        )
        grid_image.save(grid_img_path)

    # Image 2: Combined plot with graded colors
    combined_img_path = os.path.join(output_dir, "temp_combined.png")
    all_highlighted_atoms = sorted(
        list(set(a for h in grid_highlights for a in h["atoms"]))
    )
    all_highlighted_bonds = sorted(
        list(set(b for h in grid_highlights for b in h["bonds"]))
    )

    if all_highlighted_atoms:
        weights_of_highlighted_atoms = np.array(
            [combined_atom_weights.get(i, 0) for i in all_highlighted_atoms]
        )
        norm_combined_weights = (
            weights_of_highlighted_atoms - weights_of_highlighted_atoms.min()
        ) / (
            weights_of_highlighted_atoms.max()
            - weights_of_highlighted_atoms.min()
            + 1e-8
        )

        combined_atom_colors = {
            atom_idx: attention_colormap(norm_combined_weights[i])[:3]
            for i, atom_idx in enumerate(all_highlighted_atoms)
        }

        drawer = rdMolDraw2D.MolDraw2DCairo(600, 400)
        drawer.drawOptions().addAtomIndices = True
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            legend="Combined Attention from Top Tokens",
            highlightAtoms=all_highlighted_atoms,
            highlightBonds=all_highlighted_bonds,
            highlightAtomColors=combined_atom_colors,
        )
        drawer.FinishDrawing()
        drawer.WriteDrawingText(combined_img_path)

    # --- Generate Text Image with Highlighted SMILES ---
    text_img_path = os.path.join(output_dir, "temp_text.png")
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    top_indices_set = set(top_token_indices)
    segments = [
        (token, "red" if i in top_indices_set else "black")
        for i, token in enumerate(full_token_list)
    ]

    text_img_width = (
        font.getbbox("SMILES: ")[2]
        + sum(font.getbbox(seg[0])[2] for seg in segments)
        + 20
    )
    text_img_height = font.getbbox("SMILES:")[3] + 10

    text_img = Image.new("RGB", (text_img_width, text_img_height), "white")
    draw_text = ImageDraw.Draw(text_img)

    x_cursor = 10
    y_cursor = 5
    draw_text.text((x_cursor, y_cursor), "SMILES: ", font=font, fill="black")
    x_cursor += font.getbbox("SMILES: ")[2]

    for text, color in segments:
        display_text = text.replace("##", "")
        draw_text.text((x_cursor, y_cursor), display_text, font=font, fill=color)
        x_cursor += font.getbbox(display_text)[2]

    text_img.save(text_img_path)

    # --- Stitch images together ---
    try:
        if (
            os.path.exists(combined_img_path)
            and os.path.exists(grid_img_path)
            and os.path.exists(text_img_path)
        ):
            img_mol = Image.open(combined_img_path)
            img_text = Image.open(text_img_path)
            img_grid = Image.open(grid_img_path)

            total_width = max(img_mol.width, img_text.width, img_grid.width)
            total_height = img_mol.height + img_text.height + img_grid.height

            new_im = Image.new("RGB", (total_width, total_height), "white")

            new_im.paste(img_mol, (int((total_width - img_mol.width) / 2), 0))
            new_im.paste(
                img_text, (int((total_width - img_text.width) / 2), img_mol.height)
            )
            new_im.paste(
                img_grid,
                (
                    int((total_width - img_grid.width) / 2),
                    img_mol.height + img_text.height,
                ),
            )

            final_output_path = os.path.join(output_dir, "attention_summary.png")
            new_im.save(final_output_path)
            logger.info(f"Saved final composite visualization to: {final_output_path}")
        else:
            logger.warning(
                "One or more temporary images for stitching were not created."
            )

    finally:
        # Clean up temporary files
        if os.path.exists(grid_img_path):
            os.remove(grid_img_path)
        if os.path.exists(combined_img_path):
            os.remove(combined_img_path)
        if os.path.exists(text_img_path):
            os.remove(text_img_path)





def visualize_token_to_atom_attention(
    smiles: str,
    attention_weights: torch.Tensor,
    token_idx: int,
    token_str: str,
    output_path: str,
    top_k: int = 5,
):
    """
    Generates a 2D depiction of a molecule highlighting:
    1. The atoms AND bonds corresponding to the token itself (in red), if the token
       is a valid chemical substructure.
    2. The top_k atoms receiving the most attention from that token (in green).

    Note: Not all tokens from a SMILES tokenizer are valid chemical substructures.
    In such cases, only the green attention highlights will be shown.
    """
    logger.info(
        f"Visualizing attention from token '{token_str}' (index {token_idx}) for SMILES: {smiles}"
    )

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.warning(
            f"Could not generate molecule from SMILES for plotting: {smiles}"
        )
        return

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        logger.warning(
            f"SMILES '{smiles}' resulted in a molecule with 0 atoms. Skipping."
        )
        return

    weights_for_token = (
        attention_weights[:, token_idx, :num_atoms].mean(axis=0).cpu().numpy()
    )

    if weights_for_token.size == 0:
        logger.warning(
            f"Attention weights array for token {token_idx} ('{token_str}') is empty. Skipping."
        )
        return

    # --- Identify atoms and bonds for the SOURCE token ---
    token_substructure_atoms = []
    token_substructure_bonds = []
    try:
        # Treat the token string as a SMARTS query to find its location
        query_mol = Chem.MolFromSmarts(token_str)
        if query_mol:
            matches = mol.GetSubstructMatches(query_mol)
            if matches:
                # Flatten the list of matched atom indices
                token_substructure_atoms = sorted(
                    list(set(idx for match in matches for idx in match))
                )

                # Find bonds that connect the matched atoms
                token_atoms_set = set(token_substructure_atoms)
                for bond in mol.GetBonds():
                    if (
                        bond.GetBeginAtomIdx() in token_atoms_set
                        and bond.GetEndAtomIdx() in token_atoms_set
                    ):
                        # Ensure the bond exists in the query mol to be more precise
                        # This part is complex; a simpler heuristic is to just include all internal bonds.
                        token_substructure_bonds.append(bond.GetIdx())
    except Exception:
        # This is expected for tokens that are not valid SMARTS, e.g., 'Cc1cc'
        logger.debug(f"Could not parse token '{token_str}' as a SMARTS pattern.")

    # --- Identify top-k atoms by attention score ---
    k = min(top_k, num_atoms)
    top_attention_atoms = np.argsort(weights_for_token)[-k:].tolist() if k > 0 else []

    # --- Define colors and combine highlights ---
    atom_colors = {}
    bond_colors = {}
    attention_color = (0.0, 1.0, 0.0)  # Green for attention targets
    token_color = (1.0, 0.0, 0.0)  # Red for the token's own atoms/bonds

    # Apply attention highlights first
    for atom_idx in top_attention_atoms:
        atom_colors[atom_idx] = attention_color

    # Apply token substructure highlights, overwriting if they overlap
    for atom_idx in token_substructure_atoms:
        atom_colors[atom_idx] = token_color
    for bond_idx in token_substructure_bonds:
        bond_colors[bond_idx] = token_color

    all_highlight_atoms = list(atom_colors.keys())
    all_highlight_bonds = list(bond_colors.keys())

    # Create a Cairo drawer for PNG output
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    drawer.drawOptions().addAtomIndices = True
    drawer.drawOptions().legendFontSize = 20

    # Draw the molecule with atom and bond highlights
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        legend=f"Attention from token '{token_str}'",
        highlightAtoms=all_highlight_atoms,
        highlightBonds=all_highlight_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )

    drawer.FinishDrawing()
    drawer.WriteDrawingText(output_path)

    logger.info(f"Saved token-to-atom attention plot to: {output_path}")


# def visualize_top_token_attentions(
#     smiles: str,
#     attention_weights: torch.Tensor,
#     top_tokens_info: list,
#     output_dir: str,
#     top_k: int = 5,
# ):
#     """
#     Generates two summary plots for the attention from top tokens:
#     1. A grid image showing individual attention highlights for each top token.
#     2. A single heatmap image showing the combined attention from all top tokens.
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     if not mol:
#         logger.warning(f"Could not generate molecule from SMILES: {smiles}")
#         return
#     num_atoms = mol.GetNumAtoms()
#     if num_atoms == 0:
#         return

#     mols_for_grid = []
#     legends_for_grid = []
#     highlights_for_grid = []
#     combined_attention = np.zeros(num_atoms)

#     for token_idx, token_str in top_tokens_info:
#         weights_for_token = (
#             attention_weights[:, token_idx, :num_atoms].mean(axis=0).cpu().numpy()
#         )
#         if weights_for_token.size == 0:
#             continue

#         combined_attention += weights_for_token

#         # --- Prepare highlights for the grid image ---
#         k = min(top_k, num_atoms)
#         top_attention_atoms = (
#             np.argsort(weights_for_token)[-k:].tolist() if k > 0 else []
#         )

#         atom_colors = {
#             i: (0.0, 1.0, 0.0) for i in top_attention_atoms
#         }  # Green for top attention

#         mols_for_grid.append(Chem.Mol(mol))  # Add a copy for the grid
#         legends_for_grid.append(f"From token: '{token_str}'")
#         highlights_for_grid.append(
#             {"atoms": top_attention_atoms, "colors": atom_colors}
#         )

#     # --- Plot 1: Grid of individual top token attentions ---
#     if mols_for_grid:
#         grid_image = Draw.MolsToGridImage(
#             mols_for_grid,
#             molsPerRow=len(mols_for_grid),
#             subImgSize=(400, 400),
#             legends=legends_for_grid,
#             highlightAtomLists=[h["atoms"] for h in highlights_for_grid],
#             highlightAtomColors=[h["colors"] for h in highlights_for_grid],
#         )
#         grid_output_path = os.path.join(output_dir, "top_tokens_attention_grid.png")
#         grid_image.save(grid_output_path)
#         logger.info(f"Saved top-tokens grid visualization to: {grid_output_path}")

#     # --- Plot 2: Combined attention heatmap ---
#     if np.any(combined_attention):
#         norm_weights = (combined_attention - np.min(combined_attention)) / (
#             np.max(combined_attention) - np.min(combined_attention) + 1e-8
#         )

#         cmap = cm.get_cmap("viridis")
#         atom_colors_heatmap = {
#             i: tuple(cmap(norm_weights[i])[:3]) for i in range(num_atoms)
#         }

#         drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
#         drawer.drawOptions().addAtomIndices = True
#         drawer.drawOptions().legendFontSize = 20

#         rdMolDraw2D.PrepareAndDrawMolecule(
#             drawer,
#             mol,
#             legend="Combined Attention from Top Tokens",
#             highlightAtoms=list(range(num_atoms)),
#             highlightAtomColors=atom_colors_heatmap,
#         )

#         drawer.FinishDrawing()
#         heatmap_output_path = os.path.join(output_dir, "combined_attention_heatmap.png")
#         drawer.WriteDrawingText(heatmap_output_path)
#         logger.info(f"Saved combined attention heatmap to: {heatmap_output_path}")


# def visualize_token_to_atom_attention(
#     smiles: str,
#     attention_weights: torch.Tensor,
#     token_idx: int,
#     token_str: str,
#     output_path: str,
#     top_k: int = 5,
# ):
#     """
#     Generates a 2D depiction of a molecule highlighting the top_k atoms
#     and their interconnecting bonds based on attention scores from a specific query token.
#     """
#     logger.info(
#         f"Visualizing top {top_k} atom attention from token '{token_str}' (index {token_idx}) for SMILES: {smiles}"
#     )

#     mol = Chem.MolFromSmiles(smiles)
#     if not mol:
#         logger.warning(
#             f"Could not generate molecule from SMILES for plotting: {smiles}"
#         )
#         return

#     num_atoms = mol.GetNumAtoms()

#     if num_atoms == 0:
#         logger.warning(
#             f"SMILES '{smiles}' resulted in a molecule with 0 atoms. Skipping visualization."
#         )
#         return

#     weights_for_token = (
#         attention_weights[:, token_idx, :num_atoms].mean(axis=0).cpu().numpy()
#     )

#     if weights_for_token.size == 0:
#         logger.warning(
#             f"Attention weights array for token {token_idx} ('{token_str}') is empty. Skipping visualization."
#         )
#         return

#     # --- NEW LOGIC: Find top_k atoms and their bonds ---
#     k = min(top_k, num_atoms)
#     if k == 0:
#         return

#     # Get the indices of the top k atoms by sorting the weights in descending order
#     highlight_atoms = np.argsort(weights_for_token)[-k:].tolist()

#     # Find bonds that connect two highlighted atoms
#     highlight_bonds = []
#     highlight_atoms_set = set(highlight_atoms)
#     for bond in mol.GetBonds():
#         if (
#             bond.GetBeginAtomIdx() in highlight_atoms_set
#             and bond.GetEndAtomIdx() in highlight_atoms_set
#         ):
#             highlight_bonds.append(bond.GetIdx())

#     # Use a single, distinct color for highlighting
#     highlight_color = (0.0, 1.0, 0.0, 0.7)  # A bright, semi-transparent green
#     atom_colors = {i: highlight_color for i in highlight_atoms}
#     bond_colors = {i: highlight_color for i in highlight_bonds}

#     # Create a Cairo drawer for PNG output
#     drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
#     drawer.drawOptions().addAtomIndices = True
#     drawer.drawOptions().legendFontSize = 20
#     drawer.drawOptions().padding = 0.1
#     drawer.drawOptions().setHighlightColour(highlight_color)
#     drawer.drawOptions().fillHighlights = True

#     # Draw the molecule with highlights for the top atoms and their bonds
#     rdMolDraw2D.PrepareAndDrawMolecule(
#         drawer,
#         mol,
#         legend=f"Top {k} atom attention from token '{token_str}'",
#         highlightAtoms=highlight_atoms,
#         highlightBonds=highlight_bonds,
#     )

#     drawer.FinishDrawing()
#     drawer.WriteDrawingText(output_path)

#     logger.info(f"Saved top-{k} attention plot to: {output_path}")


def compute_pos_weight_from_h5(
    h5_path: str, chunk_size: int = 10_000, epsilon: float = 1e-6, verbose: bool = True
) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        labels_dset = f["labels"]
        num_samples, num_classes = labels_dset.shape

        pos_counts = torch.zeros(num_classes, dtype=torch.float64)
        neg_counts = torch.zeros(num_classes, dtype=torch.float64)

        for i in range(0, num_samples, chunk_size):
            labels = torch.tensor(labels_dset[i : i + chunk_size], dtype=torch.float32)
            pos_counts += labels.sum(dim=0)
            neg_counts += (1.0 - labels).sum(dim=0)

    pos_weight = (neg_counts / (pos_counts + epsilon)).to(torch.float32)

    if verbose:
        logger.info(f"Positive counts per class: {pos_counts.tolist()}")
        logger.info(f"Negative counts per class: {neg_counts.tolist()}")
        logger.info(f"Calculated pos_weight: {pos_weight.tolist()}")

    return pos_weight
