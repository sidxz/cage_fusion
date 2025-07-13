import os
import torch
import numpy as np
import matplotlib
import h5py
import re

# --- RDKit and Pillow imports ---
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from chemprop.data import BatchMolGraph
from cage_fusion.utils.logging_utils import logger
from typing import List, Optional


def visualize_attention_weights(
    attn_weights: torch.Tensor,
    mask: torch.Tensor,
    num_heads: int,
    output_path: str,
    input_ids: torch.Tensor = None,
    tokenizer_obj=None,
    smiles: str = None,
):
    """
    Visualizes graph-to-token attention weights across heads.
    (This function remains unchanged)
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

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def visualize_top_token_attentions(
    smiles: str,
    attention_weights: torch.Tensor,
    full_token_list: list,
    top_token_indices: np.ndarray,
    output_dir: str,
    top_k_atoms_per_token: int = 5,
):
    """
    Generates a high-resolution composite image summarizing atom-to-token attention
    using attention coefficients. This now includes the combined summary plot.
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
    attention_colormap = cm.get_cmap("bwr")  # Blue-White-Red colormap
    avg_head_attn = attention_weights.mean(axis=0).cpu().numpy()

    # To calculate the combined plot, we track the maximum *absolute* coefficient for each atom
    combined_atom_coeffs = {i: 0.0 for i in range(num_atoms)}
    all_top_atoms = set()

    for token_idx in top_token_indices:
        token_str = full_token_list[token_idx].replace("##", "")
        atom_weights_for_token = avg_head_attn[token_idx, :num_atoms]
        if atom_weights_for_token.size == 0:
            continue

        avg_atom_attention = np.mean(atom_weights_for_token)
        attention_coeffs = atom_weights_for_token - avg_atom_attention

        # Update the combined coefficients
        for i, coeff in enumerate(attention_coeffs):
            # Take the one with the largest magnitude
            if abs(coeff) > abs(combined_atom_coeffs[i]):
                combined_atom_coeffs[i] = coeff

        k = min(top_k_atoms_per_token, num_atoms)
        top_atom_indices = (
            np.argsort(np.abs(attention_coeffs))[-k:].tolist() if k > 0 else []
        )
        all_top_atoms.update(top_atom_indices)

        plot_data = _prepare_plot_data_coeff(
            mol, attention_coeffs, top_atom_indices, attention_colormap
        )
        grid_plots_data.append(plot_data)

    # --- Step 2: Generate intermediate images ---

    # Image 1: Grid of individual token attention plots
    grid_img_path = os.path.join(output_dir, "temp_grid.png")
    mols_for_grid = [Chem.Mol(mol) for _ in grid_plots_data]
    legends_for_grid = [
        f"Attention to token: '{full_token_list[idx].replace('##','')}'"
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

    # Image 2: Combined summary plot (RESTORED)
    combined_img_path = os.path.join(output_dir, "temp_combined.png")
    combined_data = _prepare_plot_data_coeff(
        mol,
        np.array(list(combined_atom_coeffs.values())),
        list(all_top_atoms),
        attention_colormap,
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

    # --- Step 3: Stitch all three images together ---
    _stitch_images(combined_img_path, text_img_path, grid_img_path, output_dir)


def _prepare_plot_data_coeff(mol, attention_coeffs, top_atom_indices, attention_cmap):
    """
    Helper to calculate highlight details for a single RDKit plot using coefficients.
    """
    atom_colors = {}
    bond_colors = {}

    max_abs_coeff = (
        np.max(np.abs(attention_coeffs)) if attention_coeffs.size > 0 else 1.0
    )
    norm = mcolors.Normalize(vmin=-max_abs_coeff, vmax=max_abs_coeff)

    # Color only the most influential atoms for clarity
    for atom_idx in top_atom_indices:
        coeff = attention_coeffs[atom_idx]
        atom_colors[atom_idx] = attention_cmap(norm(coeff))

    highlight_atoms = list(atom_colors.keys())

    highlight_bonds = [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in highlight_atoms
        and b.GetEndAtomIdx() in highlight_atoms
    ]

    for bond_idx in highlight_bonds:
        begin_idx = mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx()
        end_idx = mol.GetBondWithIdx(bond_idx).GetEndAtomIdx()
        c1 = mcolors.to_rgb(atom_colors.get(begin_idx, (1, 1, 1)))
        c2 = mcolors.to_rgb(atom_colors.get(end_idx, (1, 1, 1)))
        bond_colors[bond_idx] = tuple(np.mean([c1, c2], axis=0))

    return {
        "atoms": highlight_atoms,
        "bonds": highlight_bonds,
        "atom_colors": atom_colors,
        "bond_colors": bond_colors,
    }


def _create_highlighted_smiles_image(full_token_list, top_token_indices, output_path):
    """
    Helper to create an image of the tokenized string with top tokens highlighted.
    """
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    top_indices_set = set(top_token_indices)
    segments = [
        (token, "red" if i in top_indices_set else "black")
        for i, token in enumerate(full_token_list)
    ]

    x_cursor = 10
    prompt_width = font.getbbox("Top Tokens: ")[2]
    x_cursor += prompt_width
    for text, color in segments:
        x_cursor += font.getbbox(text)[2]

    text_img_width = x_cursor + 20
    text_img_height = font.getbbox("Top Tokens:")[3] + 20

    text_img = Image.new("RGB", (text_img_width, text_img_height), "white")
    draw_text = ImageDraw.Draw(text_img)

    x_cursor = 10
    y_cursor = 10
    draw_text.text((x_cursor, y_cursor), "Top Tokens: ", font=font, fill="black")
    x_cursor += prompt_width

    for text, color in segments:
        draw_text.text((x_cursor, y_cursor), text, font=font, fill=color)
        x_cursor += font.getbbox(text)[2]

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
            paste_x = int((total_width - img.width) / 2)
            new_im.paste(img, (paste_x, current_y))
            current_y += img.height

        final_output_path = os.path.join(output_dir, "attention_summary.png")
        new_im.save(final_output_path)

    finally:
        for p in [img1_path, img2_path, img3_path]:
            if os.path.exists(p):
                os.remove(p)


def visualize_contributions(
    smiles: str,
    token_ids: list,
    attention_weights,
    tokenizer_obj,
    predicted_class,
    output_dir: str,
    direction: str = "token_to_atom",  # or "atom_to_token"
    token_mask: "Optional[np.ndarray]" = None,
    atom_mask: "Optional[np.ndarray]" = None,
    top_k: "Optional[int]" = 6,
):

    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Prepare attention array and orient to [n_tokens, n_atoms] ---
    if isinstance(attention_weights, torch.Tensor):
        attn = attention_weights.detach().cpu().numpy()
    else:
        attn = np.array(attention_weights)
    attn = np.squeeze(attn)
    if attn.ndim != 2:
        raise ValueError(f"Expected attention matrix to be 2D, got shape {attn.shape}")

    if direction == "token_to_atom":
        n_tokens, n_atoms = attn.shape
    elif direction == "atom_to_token":
        n_atoms, n_tokens = attn.shape
        attn = attn.T
    else:
        raise ValueError("direction must be 'token_to_atom' or 'atom_to_token'")

    # --- 2. Get molecule and ensure we never use indices out of bounds ---
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[❌] Could not parse SMILES: {smiles}")
        return
    num_atoms = mol.GetNumAtoms()
    max_atoms = min(num_atoms, n_atoms)
    max_tokens = n_tokens  # Do not index beyond attn.shape[0] anyway

    # --- 3. Build valid index arrays, never out of bounds ---
    valid_atom_idxs = np.arange(max_atoms)
    valid_token_idxs = (
        np.arange(max_tokens)
        if token_mask is None
        else np.where(token_mask[:max_tokens])[0]
    )

    # --- 4. Center over valid atoms for each token (row-wise mean) ---
    centered_attn = attn.copy()
    for t in valid_token_idxs:
        mean_over_valid_atoms = (
            np.mean(attn[t, valid_atom_idxs]) if len(valid_atom_idxs) > 0 else 0
        )
        centered_attn[t, :] -= mean_over_valid_atoms

    # --- 5. Class sign (positive:1/negative:-1) ---
    sign = 1 if np.any(predicted_class) else -1
    signed_attn = sign * centered_attn

    # --- 6. Contributions (never index out of bounds!) ---
    atom_contribs = np.zeros(n_atoms)
    if len(valid_token_idxs) > 0:
        # Only sum over valid atoms!
        atom_contribs[valid_atom_idxs] = signed_attn[valid_token_idxs][
            :, valid_atom_idxs
        ].sum(axis=0)
    token_contribs = np.zeros(n_tokens)
    if len(valid_atom_idxs) > 0:
        token_contribs[valid_token_idxs] = signed_attn[valid_token_idxs][
            :, valid_atom_idxs
        ].sum(axis=1)

    # --- 7. Normalize ONLY over valid entries ---
    def _normalize(x, mask):
        maxval = np.max(np.abs(x[mask])) if len(mask) > 0 else 1.0
        if maxval < 1e-8:
            return x  # avoid division by zero
        x = x.copy()
        x[mask] = x[mask] / maxval
        return x

    atom_contribs = _normalize(atom_contribs, valid_atom_idxs)
    token_contribs = _normalize(token_contribs, valid_token_idxs)

    # --- 8. Draw molecule with colored atoms (only valid atom indices, all int) ---
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
    drawer.drawOptions().addAtomIndices = True
    atom_colors = {}
    for i in valid_atom_idxs:
        idx = int(i)
        val = float(atom_contribs[idx])
        color = (1.0, 0.0, 0.0) if val > 0 else (0.0, 0.0, 1.0)
        atom_colors[idx] = tuple(abs(val) * np.array(color))
    highlight_atoms = list(atom_colors.keys())

    drawer.DrawMolecule(
        mol, highlightAtoms=highlight_atoms, highlightAtomColors=atom_colors
    )
    drawer.FinishDrawing()
    mol_img_path = os.path.join(output_dir, f"mol_contribution_{direction}.png")
    with open(mol_img_path, "wb") as f:
        f.write(drawer.GetDrawingText())

    # --- 9. Bar plot for token contributions (only valid tokens) ---
    tokens_for_labels = tokenizer_obj.convert_ids_to_tokens(
        [tid for idx, tid in enumerate(token_ids) if idx in valid_token_idxs]
    )
    contribs_for_plot = token_contribs[valid_token_idxs]
    plt.figure(figsize=(max(6, 0.3 * len(tokens_for_labels)), 1.7))
    plt.bar(
        range(len(contribs_for_plot)),
        contribs_for_plot,
        color=["red" if x > 0 else "blue" for x in contribs_for_plot],
    )
    plt.xticks(
        range(len(tokens_for_labels)), tokens_for_labels, rotation=90, fontsize=8
    )
    plt.title(f"Token Contribution ({direction.replace('_', '→')})")
    plt.tight_layout()
    token_img_path = os.path.join(output_dir, f"token_contribution_{direction}.png")
    plt.savefig(token_img_path)
    plt.close()

    # --- 10. Save contributions ---
    # np.save(os.path.join(output_dir, f"atom_contribs_{direction}.npy"), atom_contribs)
    # np.save(os.path.join(output_dir, f"token_contribs_{direction}.npy"), token_contribs)

    print(f"[✅] Saved visualization to {output_dir}")

    return atom_contribs, token_contribs

    import os


import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.cm as cm

def minmax_neg1_1(x):
    x = np.asarray(x)
    minv, maxv = x.min(), x.max()
    return 2 * (x - minv) / (maxv - minv + 1e-8) - 1

def visualize_total_atom_contribution(
    smiles,
    t2a_weights_sample,  # shape: [n_tokens, n_atoms] (numpy, or torch.cpu().numpy())
    pred_logit,          # model output scalar, for sign
    output_path="atom_total_contrib.png",
    top_n=None,             # If None, show all; else, show top N pos and N neg atoms
):
    """
    Visualize total per-atom contributions by summing over all tokens.
    - Red: atom contributed positively (toward prediction)
    - Blue: atom contributed negatively (against prediction)
    If top_n is set, only the N most positive and N most negative atoms are shown.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Could not parse SMILES: {smiles}")
        return

    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    if n_atoms == 0:
        print("No atoms in molecule.")
        return

    # 1. Aggregate all tokens
    if hasattr(t2a_weights_sample, "cpu"):
        attn = t2a_weights_sample.cpu().numpy()
    else:
        attn = t2a_weights_sample
    atom_scores = attn.sum(axis=0)  # [n_atoms]
    atom_scores -= atom_scores.mean()
    sign = 1 if pred_logit > 0 else -1
    atom_contribs = sign * atom_scores
    vmax = np.abs(atom_contribs).max() if n_atoms > 0 else 1.0
    norm = minmax_neg1_1(atom_contribs)
    cmap = cm.get_cmap("bwr")
    atom_colors = {}

    # Make sure top_n does not exceed n_atoms
    n_top = min(top_n or n_atoms, n_atoms)

    # Select top_n positive and top_n negative atoms if requested
    if top_n is not None and top_n > 0:
        pos_atoms = np.argsort(-atom_contribs)[:n_top]  # N most positive
        neg_atoms = np.argsort(atom_contribs)[:n_top]   # N most negative
        highlight_atoms = list(set(int(i) for i in np.concatenate([pos_atoms, neg_atoms])))
    else:
        highlight_atoms = list(range(n_atoms))

    # Only color those atoms (ensure Python int and in range)
    for i in highlight_atoms:
        idx = int(i)
        if 0 <= idx < n_atoms:
            atom_colors[idx] = cmap(0.5 + 0.5 * norm[idx])[:3]

    # --- Bond coloring: only for bonds between highlighted atoms ---
    highlight_bonds = [
        int(b.GetIdx())
        for b in mol.GetBonds()
        if int(b.GetBeginAtomIdx()) in highlight_atoms and int(b.GetEndAtomIdx()) in highlight_atoms
    ]
    bond_colors = {}
    for bond_idx in highlight_bonds:
        begin_idx = int(mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx())
        end_idx = int(mol.GetBondWithIdx(bond_idx).GetEndAtomIdx())
        c1 = atom_colors.get(begin_idx, (1, 1, 1))
        c2 = atom_colors.get(end_idx, (1, 1, 1))
        bond_colors[bond_idx] = tuple(np.mean([c1, c2], axis=0))

    # Ensure indices are all Python ints and in-bounds
    highlight_atoms = [int(i) for i in highlight_atoms if 0 <= int(i) < n_atoms]
    atom_colors = {int(i): c for i, c in atom_colors.items() if 0 <= int(i) < n_atoms}
    highlight_bonds = [int(i) for i in highlight_bonds if 0 <= int(i) < n_bonds]
    bond_colors = {int(i): c for i, c in bond_colors.items() if 0 <= int(i) < n_bonds}

    drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
    drawer.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(drawer.GetDrawingText())
        
        
def visualize_combined_atom_contribution(
    smiles,
    t2a_weights_sample,      # [n_tokens, n_atoms_padded] (numpy or torch)
    pred_logit,              # scalar (float)
    prompt_attn_weights,     # dict as in visualize_fg_attention
    output_path="atom_combined_contrib.png",
    weight_t2a=1.0,
    weight_fg=1.0,
    top_n=None,              # highlight top N positive/negative if wanted
):
    print(f"weight_t2a: {weight_t2a}, weight_fg: {weight_fg}")
    # print("="*10, "ENTER visualize_combined_atom_contribution", "="*10)
    # print(f"SMILES: {smiles}")
    # print(f"t2a_weights_sample type: {type(t2a_weights_sample)}")
    # print(f"t2a_weights_sample.shape: {getattr(t2a_weights_sample, 'shape', 'N/A')}")
    from ..engine.fg_utils import FG_NAMES, FG_SMARTS

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        print(f"Could not parse or empty SMILES: {smiles}")
        return

    n_atoms = mol.GetNumAtoms()
    #print(f"Num atoms in molecule: {n_atoms}")

    # --- 1. Model contribution: as before ---
    if hasattr(t2a_weights_sample, "cpu"):
        attn = t2a_weights_sample.cpu().numpy()
    else:
        attn = t2a_weights_sample
    #print(f"attn.shape after cpu().numpy(): {attn.shape}")
    # if attn.shape[-1] != n_atoms:
    #     print(f"WARNING: attn.shape[-1] = {attn.shape[-1]} does not match n_atoms = {n_atoms}")
    #     print(f"Trimming model_atom_scores from {attn.shape[-1]} to {n_atoms}")
    model_atom_scores = attn.sum(axis=0)  # [n_atoms_padded]
    model_atom_scores -= model_atom_scores.mean()
    sign = 1 if pred_logit > 0 else -1
    model_atom_contribs = sign * model_atom_scores

    # PATCH: trim to only real atoms
    model_atom_contribs = model_atom_contribs[:n_atoms]
    #print(f"model_atom_contribs.shape after trim: {model_atom_contribs.shape}")

    # --- 2. FG prompt per-atom contribution ---
    fg_atom_contribs = np.zeros(n_atoms)
    fg_ids = prompt_attn_weights.get("fg_ids", [])
    weights = np.array(prompt_attn_weights.get("weights", []))
    #print(f"FG ids: {fg_ids}")
    #print(f"FG weights: {weights}")
    if len(fg_ids) > 0 and len(weights) > 0:
        average_attention = np.mean(weights)
        attention_coefficients = weights - average_attention
        #print(f"FG attention coefficients: {attention_coefficients}")
        for fg_idx, fg_id in enumerate(fg_ids):
            coeff = attention_coefficients[fg_idx]
            fg_name = FG_NAMES[fg_id] if fg_id < len(FG_NAMES) else f"FG_{fg_id}"
            smarts = FG_SMARTS.get(fg_name)
            if isinstance(smarts, str):
                patt = Chem.MolFromSmarts(smarts)
            elif isinstance(smarts, Chem.Mol):
                patt = smarts
            else:
                patt = None
            if not patt:
                continue
            matches = mol.GetSubstructMatches(patt)
            #print(f"  FG {fg_name} (id {fg_id}), SMARTS: {smarts}, coeff: {coeff}, matches: {matches}")
            for match in matches:
                for atom_idx in match:
                    if atom_idx < n_atoms:
                        fg_atom_contribs[atom_idx] += coeff / len(match)
                        #print(f"    Atom {atom_idx}: FG contrib += {coeff / len(match)}")
    #print(f"Final model_atom_contribs.shape: {model_atom_contribs.shape}")
    #print(f"Final fg_atom_contribs.shape: {fg_atom_contribs.shape}")

    # --- Defensive check after population ---
    if model_atom_contribs.shape != fg_atom_contribs.shape:
        print(f"Shape mismatch! Model: {model_atom_contribs.shape}, FG: {fg_atom_contribs.shape}")
        print(f"SMILES: {smiles}")
        print(f"fg_ids: {fg_ids}")
        print(f"weights: {weights}")
        print(f"Mol atoms: {n_atoms}")
        print("Aborting combined visualization for this molecule.")
        return

    # --- 3. Weighted sum ---
    model_atom_contribs_norm = minmax_neg1_1(model_atom_contribs)
    fg_atom_contribs_norm = minmax_neg1_1(fg_atom_contribs)

    total_atom_contribs = weight_t2a * model_atom_contribs_norm + weight_fg * fg_atom_contribs_norm
    vmax = np.abs(total_atom_contribs).max() if n_atoms > 0 else 1.0
    norm = total_atom_contribs / (vmax + 1e-8)
    cmap = cm.get_cmap("bwr")

    # --- Top-n highlighting ---
    if top_n is not None and top_n > 0:
        pos_atoms = np.argsort(-total_atom_contribs)[:top_n]
        neg_atoms = np.argsort(total_atom_contribs)[:top_n]
        highlight_atoms = list(set(int(i) for i in np.concatenate([pos_atoms, neg_atoms])))
    else:
        highlight_atoms = list(range(n_atoms))

    atom_colors = {}
    for i in highlight_atoms:
        idx = int(i)
        if 0 <= idx < n_atoms:
            atom_colors[idx] = cmap(0.5 + 0.5 * norm[idx])[:3]

    # Bond colors: highlight if both atoms are highlighted
    highlight_bonds = [
        int(b.GetIdx())
        for b in mol.GetBonds()
        if int(b.GetBeginAtomIdx()) in highlight_atoms and int(b.GetEndAtomIdx()) in highlight_atoms
    ]
    bond_colors = {}
    for bond_idx in highlight_bonds:
        begin_idx = int(mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx())
        end_idx = int(mol.GetBondWithIdx(bond_idx).GetEndAtomIdx())
        c1 = atom_colors.get(begin_idx, (1, 1, 1))
        c2 = atom_colors.get(end_idx, (1, 1, 1))
        bond_colors[bond_idx] = tuple(np.mean([c1, c2], axis=0))

    drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
    drawer.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(drawer.GetDrawingText())