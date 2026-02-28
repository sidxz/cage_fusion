"""
cage_fusion/visualization/attention.py

Attention-weight visualizations for CAGEFusion models.

Public API
----------
- ``visualize_attention_weights``       : per-head graph→token heatmap
- ``visualize_top_token_attentions``    : composite atom-highlight image
- ``visualize_contributions``           : general token/atom contribution plot
- ``visualize_total_atom_contribution`` : summed token→atom contribution
- ``visualize_combined_atom_contribution``: model + FG-prompt combined map
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph→Token heatmap
# ---------------------------------------------------------------------------

def visualize_attention_weights(
    attn_weights: torch.Tensor,
    mask: torch.Tensor,
    num_heads: int,
    output_path: str,
    input_ids: Optional[torch.Tensor] = None,
    tokenizer_obj=None,
    smiles: Optional[str] = None,
):
    """
    Visualize graph-to-token attention weights across heads as a heatmap.

    Args:
        attn_weights: Tensor of shape ``[n_atoms, num_heads, seq_len]`` or
                      ``[n_atoms, seq_len]``.
        mask: Boolean / integer mask of length ``seq_len``; non-padding positions.
        num_heads: Number of attention heads.
        output_path: PNG save path.
        input_ids: Token IDs used to annotate x-axis ticks.
        tokenizer_obj: HuggingFace tokenizer for label conversion.
        smiles: SMILES string used as figure title.
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
        logger.warning("Skipped attention visualization: all values are NaN.")
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
        fig.suptitle(f"Graph-to-Token Attention: {smiles}", fontsize=16)

    def _plot_head(ax_row, data, title):
        sns.heatmap(
            data.reshape(1, -1), cmap="viridis", ax=ax_row[0], xticklabels=xtick_labels
        )
        ax_row[0].set_title(title)
        ax_row[1].hist(data, bins=20, color="skyblue")
        ax_row[1].set_title(f"{title} Distribution")

    _plot_head(axs[0], averaged_attn, "Averaged Attention")
    for i, head_data in enumerate(normalized_heads):
        if not np.isnan(head_data).all():
            _plot_head(axs[i + 1], head_data, f"Head {i}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-token atom-highlight composite
# ---------------------------------------------------------------------------

def visualize_top_token_attentions(
    smiles: str,
    attention_weights: torch.Tensor,
    full_token_list: list,
    top_token_indices: np.ndarray,
    output_dir: str,
    top_k_atoms_per_token: int = 5,
):
    """
    Composite image: per-token atom-highlight grid + combined summary + token labels.

    Saves ``attention_summary.png`` in ``output_dir``.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.warning(f"Could not parse SMILES: {smiles}")
        return
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return

    attention_colormap = cm.get_cmap("bwr")
    avg_head_attn = attention_weights.mean(axis=0).cpu().numpy()
    combined_atom_coeffs = {i: 0.0 for i in range(num_atoms)}
    all_top_atoms: set = set()
    grid_plots_data = []

    for token_idx in top_token_indices:
        atom_weights_for_token = avg_head_attn[token_idx, :num_atoms]
        if atom_weights_for_token.size == 0:
            continue

        avg_atom_attention = np.mean(atom_weights_for_token)
        attention_coeffs = atom_weights_for_token - avg_atom_attention

        for i, coeff in enumerate(attention_coeffs):
            if abs(coeff) > abs(combined_atom_coeffs[i]):
                combined_atom_coeffs[i] = coeff

        k = min(top_k_atoms_per_token, num_atoms)
        top_atom_indices = (
            np.argsort(np.abs(attention_coeffs))[-k:].tolist() if k > 0 else []
        )
        all_top_atoms.update(top_atom_indices)

        grid_plots_data.append(
            _prepare_plot_data_coeff(mol, attention_coeffs, top_atom_indices, attention_colormap)
        )

    # Grid image
    grid_img_path = os.path.join(output_dir, "temp_grid.png")
    mols_for_grid = [Chem.Mol(mol) for _ in grid_plots_data]
    legends_for_grid = [
        f"Token: '{full_token_list[idx].replace('##', '')}'"
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

    # Combined summary image
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

    # Text image
    text_img_path = os.path.join(output_dir, "temp_text.png")
    _create_highlighted_smiles_image(full_token_list, top_token_indices, text_img_path)

    # Stitch
    _stitch_images(combined_img_path, text_img_path, grid_img_path, output_dir)


# ---------------------------------------------------------------------------
# General contribution visualization
# ---------------------------------------------------------------------------

def visualize_contributions(
    smiles: str,
    token_ids: list,
    attention_weights,
    tokenizer_obj,
    predicted_class,
    output_dir: str,
    direction: str = "token_to_atom",
    token_mask: Optional[np.ndarray] = None,
    atom_mask: Optional[np.ndarray] = None,
    top_k: Optional[int] = 6,
):
    """
    General-purpose token/atom contribution plot (molecule + bar chart).

    Args:
        direction: ``"token_to_atom"`` or ``"atom_to_token"``.
    """
    os.makedirs(output_dir, exist_ok=True)

    attn = (
        attention_weights.detach().cpu().numpy()
        if isinstance(attention_weights, torch.Tensor)
        else np.array(attention_weights)
    )
    attn = np.squeeze(attn)
    if attn.ndim != 2:
        raise ValueError(f"Expected 2-D attention matrix, got shape {attn.shape}")

    if direction == "token_to_atom":
        n_tokens, n_atoms = attn.shape
    elif direction == "atom_to_token":
        n_atoms, n_tokens = attn.shape
        attn = attn.T
    else:
        raise ValueError("direction must be 'token_to_atom' or 'atom_to_token'")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.error(f"Could not parse SMILES: {smiles}")
        return
    num_atoms = mol.GetNumAtoms()
    max_atoms = min(num_atoms, n_atoms)

    valid_atom_idxs = np.arange(max_atoms)
    valid_token_idxs = (
        np.arange(n_tokens) if token_mask is None
        else np.where(token_mask[:n_tokens])[0]
    )

    centered_attn = attn.copy()
    for t in valid_token_idxs:
        mean_val = np.mean(attn[t, valid_atom_idxs]) if len(valid_atom_idxs) > 0 else 0
        centered_attn[t, :] -= mean_val

    sign = 1 if np.any(predicted_class) else -1
    signed_attn = sign * centered_attn

    atom_contribs = np.zeros(n_atoms)
    if len(valid_token_idxs) > 0:
        atom_contribs[valid_atom_idxs] = signed_attn[valid_token_idxs][
            :, valid_atom_idxs
        ].sum(axis=0)
    token_contribs = np.zeros(n_tokens)
    if len(valid_atom_idxs) > 0:
        token_contribs[valid_token_idxs] = signed_attn[valid_token_idxs][
            :, valid_atom_idxs
        ].sum(axis=1)

    def _normalize(x, mask):
        maxval = np.max(np.abs(x[mask])) if len(mask) > 0 else 1.0
        if maxval < 1e-8:
            return x
        x = x.copy()
        x[mask] = x[mask] / maxval
        return x

    atom_contribs = _normalize(atom_contribs, valid_atom_idxs)
    token_contribs = _normalize(token_contribs, valid_token_idxs)

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
    plt.xticks(range(len(tokens_for_labels)), tokens_for_labels, rotation=90, fontsize=8)
    plt.title(f"Token Contribution ({direction.replace('_', '→')})")
    plt.tight_layout()
    token_img_path = os.path.join(output_dir, f"token_contribution_{direction}.png")
    plt.savefig(token_img_path)
    plt.close()

    return atom_contribs, token_contribs


# ---------------------------------------------------------------------------
# Total atom contribution (token→atom sum)
# ---------------------------------------------------------------------------

def _minmax_neg1_1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    minv, maxv = x.min(), x.max()
    return 2 * (x - minv) / (maxv - minv + 1e-8) - 1


def visualize_total_atom_contribution(
    smiles: str,
    t2a_weights_sample,
    pred_logit: float,
    output_path: str = "atom_total_contrib.png",
    top_n: Optional[int] = None,
    highlight_red: bool = True,
    highlight_blue: bool = False,
):
    """
    Visualize total per-atom contributions by summing token→atom attention.

    - Red  = atom contributed positively toward the prediction.
    - Blue = atom contributed negatively against the prediction.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        logger.warning(f"Could not parse or empty SMILES: {smiles}")
        return

    n_atoms = mol.GetNumAtoms()
    attn = (
        t2a_weights_sample.cpu().numpy()
        if hasattr(t2a_weights_sample, "cpu")
        else np.asarray(t2a_weights_sample)
    )
    atom_scores = attn.sum(axis=0)
    atom_scores -= atom_scores.mean()
    sign = 1 if pred_logit > 0 else -1
    atom_contribs = sign * atom_scores
    norm = _minmax_neg1_1(atom_contribs)
    cmap = cm.get_cmap("bwr")

    n_top = min(top_n or n_atoms, n_atoms)
    pos_atoms = np.argsort(-atom_contribs)[:n_top] if highlight_red else np.array([])
    neg_atoms = np.argsort(atom_contribs)[:n_top] if highlight_blue else np.array([])

    highlight_atoms = (
        list(set(map(int, np.concatenate([pos_atoms, neg_atoms]))))
        if top_n
        else list(range(n_atoms))
    )
    atom_colors: dict = {}
    for i in highlight_atoms:
        val = norm[i]
        if atom_contribs[i] > 0 and highlight_red:
            atom_colors[i] = cmap(0.5 + 0.5 * val)[:3]
        elif atom_contribs[i] < 0 and highlight_blue:
            atom_colors[i] = cmap(0.5 + 0.5 * val)[:3]

    highlight_bonds = [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in atom_colors and b.GetEndAtomIdx() in atom_colors
    ]
    bond_colors = {
        b_idx: tuple(
            np.mean([atom_colors[b.GetBeginAtomIdx()], atom_colors[b.GetEndAtomIdx()]], axis=0)
        )
        for b_idx, b in [(b.GetIdx(), b) for b in mol.GetBonds()]
        if b.GetBeginAtomIdx() in atom_colors and b.GetEndAtomIdx() in atom_colors
    }

    drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
    drawer.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(drawer.GetDrawingText())


# ---------------------------------------------------------------------------
# Combined model + FG-prompt atom contribution
# ---------------------------------------------------------------------------

def visualize_combined_atom_contribution(
    smiles: str,
    t2a_weights_sample,
    pred_logit: float,
    prompt_attn_weights: dict,
    output_path: str = "atom_combined_contrib.png",
    weight_t2a: float = 1.0,
    weight_fg: float = 1.0,
    top_n: Optional[int] = None,
    highlight_red: bool = True,
    highlight_blue: bool = False,
):
    """
    Weighted combination of token→atom attention and FG-prompt attention.
    """
    from cage_fusion.chemistry.fg_utils import FG_NAMES, _FG_SMARTS

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        logger.warning(f"Could not parse or empty SMILES: {smiles}")
        return

    n_atoms = mol.GetNumAtoms()

    attn = (
        t2a_weights_sample.cpu().numpy()
        if hasattr(t2a_weights_sample, "cpu")
        else np.asarray(t2a_weights_sample)
    )
    model_atom_scores = attn.sum(axis=0)
    model_atom_scores -= model_atom_scores.mean()
    sign = 1 if pred_logit > 0 else -1
    model_atom_contribs = (sign * model_atom_scores)[:n_atoms]

    fg_atom_contribs = np.zeros(n_atoms, dtype=float)
    fg_ids = prompt_attn_weights.get("fg_ids", [])
    weights = np.array(prompt_attn_weights.get("weights", []))

    if len(fg_ids) > 0 and len(weights) > 0:
        avg_attn = np.mean(weights)
        attn_coeffs = weights - avg_attn
        for fg_idx, fg_id in enumerate(fg_ids):
            coeff = attn_coeffs[fg_idx]
            fg_name = FG_NAMES[fg_id] if fg_id < len(FG_NAMES) else f"FG_{fg_id}"
            smarts = _FG_SMARTS.get(fg_name)
            patt = (
                Chem.MolFromSmarts(smarts) if isinstance(smarts, str)
                else (smarts if isinstance(smarts, Chem.Mol) else None)
            )
            if not patt:
                continue
            for match in mol.GetSubstructMatches(patt):
                share = coeff / max(len(match), 1)
                for atom_idx in match:
                    if atom_idx < n_atoms:
                        fg_atom_contribs[atom_idx] += share

    model_norm = _minmax_neg1_1(model_atom_contribs)
    fg_norm = _minmax_neg1_1(fg_atom_contribs)
    total_atom_contribs = weight_t2a * model_norm + weight_fg * fg_norm

    vmax = np.abs(total_atom_contribs).max() if n_atoms > 0 else 1.0
    norm = total_atom_contribs / (vmax + 1e-8)
    cmap = cm.get_cmap("bwr")

    pos_idx = np.where(total_atom_contribs > 0)[0]
    neg_idx = np.where(total_atom_contribs < 0)[0]

    if top_n and top_n > 0:
        pos_sel = (
            pos_idx[np.argsort(-total_atom_contribs[pos_idx])[:top_n]]
            if highlight_red else np.array([], dtype=int)
        )
        neg_sel = (
            neg_idx[np.argsort(total_atom_contribs[neg_idx])[:top_n]]
            if highlight_blue else np.array([], dtype=int)
        )
        selected = np.unique(np.concatenate([pos_sel, neg_sel]))
    else:
        red_all = pos_idx if highlight_red else np.array([], dtype=int)
        blue_all = neg_idx if highlight_blue else np.array([], dtype=int)
        selected = np.unique(np.concatenate([red_all, blue_all]))

    highlight_atoms = [int(i) for i in selected.tolist()]
    atom_colors = {i: cmap(0.5 + 0.5 * norm[i])[:3] for i in highlight_atoms}

    highlight_bonds = [
        int(b.GetIdx())
        for b in mol.GetBonds()
        if int(b.GetBeginAtomIdx()) in atom_colors and int(b.GetEndAtomIdx()) in atom_colors
    ]
    bond_colors = {}
    for b_idx in highlight_bonds:
        b = mol.GetBondWithIdx(b_idx)
        c1 = atom_colors[int(b.GetBeginAtomIdx())]
        c2 = atom_colors[int(b.GetEndAtomIdx())]
        bond_colors[b_idx] = tuple(np.mean([c1, c2], axis=0))

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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(drawer.GetDrawingText())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_plot_data_coeff(mol, attention_coeffs, top_atom_indices, attention_cmap):
    atom_colors = {}
    bond_colors = {}

    max_abs_coeff = np.max(np.abs(attention_coeffs)) if attention_coeffs.size > 0 else 1.0
    if max_abs_coeff < 1e-8:
        max_abs_coeff = 1.0
    norm = mcolors.Normalize(vmin=-max_abs_coeff, vmax=max_abs_coeff)

    for atom_idx in top_atom_indices:
        coeff = attention_coeffs[atom_idx]
        atom_colors[atom_idx] = attention_cmap(norm(coeff))

    highlight_atoms = list(atom_colors.keys())
    highlight_bonds = [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in highlight_atoms and b.GetEndAtomIdx() in highlight_atoms
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
    for text, _ in segments:
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


def _stitch_images(img1_path: str, img2_path: str, img3_path: str, output_dir: str):
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

        new_im.save(os.path.join(output_dir, "attention_summary.png"))
    finally:
        for p in [img1_path, img2_path, img3_path]:
            if os.path.exists(p):
                os.remove(p)
