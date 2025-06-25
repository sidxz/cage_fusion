#!/usr/bin/env python3
"""
Corrected and Enhanced CAGE-Fusion Global Model Analyzer (v14 - Final)

This script analyzes a trained CAGE-Fusion model to extract global insights.
This definitive version uses a robust Matplotlib layout and includes all
requested visualization features: contextual coloring, attachment bond
highlighting, and descriptive titles for both rows and individual plots.
"""

import os
import sys
import torch
import joblib
import pandas as pd
import numpy as np
import argparse
import traceback
import tempfile
import shutil
from collections import Counter, defaultdict
from io import BytesIO
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from typing import Optional


# Plotting and visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use the project's own modules
from cage_fusion.models.cage import CAGEFusionModel
from cage_fusion.utils.logging_utils import logger
from cage_fusion.featurizers.core import featurize_and_save_streaming
from cage_fusion.engine.dataset import CageFusionStreamingDataset
from cage_fusion.engine.data_utils import collate_fn_for_cage_fusion
from cage_fusion.engine.utils import move_bmg_to_device

# Rich console for beautiful output
from rich.console import Console
from tqdm import tqdm

console = Console()


def analyze_and_visualize(
    csv_path: str,
    checkpoint_dir: str,
    target_task: str,
    output_dir: str,
    top_n_features: int = 15,
    top_n_fragments: int = 8,
    batch_size: int = 32,
    provided_h5_path: Optional[str] = None,
    provided_graph_path: Optional[str] = None,
):
    """
    Main function to analyze a dataset and generate global insight visualizations.
    """
    console.rule(f"[bold green]Starting CAGE-Fusion Global Model Analysis[/bold green]")
    os.makedirs(output_dir, exist_ok=True)

    # === 1. Load Model and Configuration ===
    console.log("Loading model and configuration...")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    scaler_path = os.path.join(checkpoint_dir, "aux_features_scaler.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tasks = config["tasks"]
    if target_task not in tasks:
        raise ValueError(
            f"Target task '{target_task}' not found in model's tasks: {tasks}"
        )
    target_task_index = tasks.index(target_task)
    all_thresholds = checkpoint.get("best_thresholds", np.full(len(tasks), 0.5))
    threshold = all_thresholds[target_task_index]
    # log the threshold for the target task
    console.log(
        f"Using threshold {threshold} for task '{target_task}'"
    )

    model = CAGEFusionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    embedding_model = (
        AutoModel.from_pretrained(config["model_checkpoint"]).to(device).eval()
    )
    scaler = joblib.load(scaler_path)

    console.log(
        f"✅ Model and components loaded. Using device: [bold cyan]{device}[/bold cyan]"
    )

    # === 2. Featurize the Dataset ===
    
    temp_dir = tempfile.mkdtemp()

    try:
        if provided_h5_path and provided_graph_path:
            console.log(
                f"Using provided H5 path: [cyan]{provided_h5_path}[/cyan] and graph path: [cyan]{provided_graph_path}[/cyan]"
            )
            h5_path = provided_h5_path
            graph_path = provided_graph_path
        else:
            console.log(
                "No pre-featurized data provided. Featurizing dataset from scratch..."
            )
            console.log(f"Loading and featurizing data from {csv_path}...")
            input_df = pd.read_csv(csv_path)
            

            h5_path, graph_path, _ = featurize_and_save_streaming(
                df=input_df,
                name="analysis_temp",
                label_cols=tasks,
                cache_dir=temp_dir,
                tokenizer=tokenizer,
                model=embedding_model,
                fit_scaler=False,
                scaler=scaler,
            )

        dataset = CageFusionStreamingDataset(
            h5_path, graph_path, tokenizer.pad_token_id
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_for_cage_fusion,
            shuffle=False,
            num_workers=0,
        )

        console.log("✅ Dataset featurized. Aggregating model insights...")
        total_aux_grads = np.zeros(config["aux_feature_dim"])
        fragment_counter = Counter()
        fragment_examples = {}

        # === 3. Aggregate Gradients Across the Dataset ===
        for batch in tqdm(loader, desc="Analyzing Dataset Batches"):
            bmg, token_embs, attn_mask, aux_feats, _, input_ids, smiles_batch = batch
            bmg = move_bmg_to_device(bmg, device)
            token_embs, attn_mask, aux_feats, input_ids = (
                token_embs.to(device),
                attn_mask.to(device),
                aux_feats.to(device),
                input_ids.to(device),
            )
            aux_feats.requires_grad_(True)
            model.zero_grad()

            logits, _, _, _, _, _, _, atom_features = model(
                bmg=bmg,
                sequence_embeddings=token_embs,
                attn_mask=attn_mask,
                aux_feats=aux_feats,
                input_ids_batch=input_ids,
                return_attn=True,
            )

            probs = torch.sigmoid(logits[:, target_task_index])
            positive_indices = torch.where(probs >= threshold)[0]
            if len(positive_indices) == 0:
                continue

            atom_grads_storage = []
            if atom_features is not None:
                atom_features.register_hook(
                    lambda grad: atom_grads_storage.append(grad)
                )
            score_to_backprop = logits[positive_indices, target_task_index].sum()
            score_to_backprop.backward()

            if aux_feats.grad is not None:
                total_aux_grads += (
                    aux_feats.grad[positive_indices].abs().sum(dim=0).cpu().numpy()
                )

            if atom_grads_storage:
                batch_atom_grads = atom_grads_storage[0]
                for j_idx in positive_indices:
                    smiles = smiles_batch[j_idx.item()]
                    mol = Chem.MolFromSmiles(smiles)
                    if not mol:
                        continue
                    mol_atom_grads_norm = (
                        batch_atom_grads[bmg.batch == j_idx.item()]
                        .norm(dim=-1)
                        .cpu()
                        .numpy()
                    )
                    top_atom_indices = np.argsort(mol_atom_grads_norm)[-3:]
                    for atom_idx in top_atom_indices:
                        fp = AllChem.GetMorganFingerprint(
                            mol, 2, fromAtoms=[int(atom_idx)]
                        )
                        for frag_id, _ in fp.GetNonzeroElements().items():
                            fragment_counter.update([frag_id])
                            if frag_id not in fragment_examples:
                                fragment_examples[frag_id] = smiles

        # === 4. Generate Visualizations ===
        console.log("✅ Gradient aggregation complete. Generating visualizations...")

        # --- Auxiliary Features Plot ---
        descriptor_names = [desc[0] for desc in Descriptors._descList]
        feature_importance_df = (
            pd.DataFrame({"feature": descriptor_names, "importance": total_aux_grads})
            .sort_values("importance", ascending=False)
            .head(top_n_features)
        )
        plt.figure(figsize=(12, 10))
        sns.barplot(
            x="importance", y="feature", data=feature_importance_df, palette="viridis"
        )
        plt.title(
            f"Top {top_n_features} Influential Auxiliary Features for '{target_task}'",
            fontsize=16,
            weight="bold",
        )
        plt.xlabel("Aggregated Absolute Gradient (Feature Importance)", fontsize=12)
        plt.ylabel("Physicochemical Descriptor", fontsize=12)
        plt.tight_layout()
        aux_fig_path = os.path.join(output_dir, f"top_aux_features_{target_task}.png")
        plt.savefig(aux_fig_path, dpi=300)
        plt.close()
        console.log(
            f"✅ Saved auxiliary feature analysis to [cyan]{aux_fig_path}[/cyan]"
        )

        # --- Final, Robust Contextual Substructure Visualization ---
        if fragment_counter:
            console.log(
                "Grouping fragments and generating advanced contextual visualization..."
            )

            grouped_fragments = defaultdict(list)
            for frag_id, count in fragment_counter.most_common(100):
                if frag_id not in fragment_examples:
                    continue
                mol = Chem.MolFromSmiles(fragment_examples[frag_id])
                if not mol:
                    continue
                bit_info = {}
                AllChem.GetMorganFingerprint(mol, 2, bitInfo=bit_info)
                if frag_id in bit_info:
                    core_atoms = [idx for idx, rad in bit_info[frag_id] if rad == 0]
                    try:
                        core_smiles = Chem.MolFragmentToSmiles(
                            mol,
                            atomsToUse=core_atoms,
                            isomericSmiles=True,
                            canonical=True,
                        )
                        grouped_fragments[core_smiles].append(
                            {
                                "frag_id": frag_id,
                                "count": count,
                                "example_mol": mol,
                                "bit_info": bit_info[frag_id],
                            }
                        )
                    except Exception:
                        continue

            sorted_groups = sorted(
                grouped_fragments.items(),
                key=lambda item: sum(frag["count"] for frag in item[1]),
                reverse=True,
            )

            num_groups_to_plot = min(len(sorted_groups), top_n_fragments)
            if num_groups_to_plot > 0:
                max_contexts = (
                    max(len(g[1]) for g in sorted_groups[:num_groups_to_plot])
                    if sorted_groups
                    else 1
                )

                fig, axes = plt.subplots(
                    num_groups_to_plot,
                    max_contexts,
                    figsize=(max_contexts * 5, num_groups_to_plot * 4.5),
                    squeeze=False,
                )
                fig.suptitle(
                    f"Top Substructure Motifs Associated with '{target_task}'",
                    fontsize=24,
                    weight="bold",
                )

                for i, (core_smiles, fragments) in enumerate(
                    sorted_groups[:num_groups_to_plot]
                ):
                    fragments.sort(key=lambda x: x["count"], reverse=True)
                    total_group_count = sum(f["count"] for f in fragments)

                    # Use the Y-label of the first axis in a row to act as a robust row title
                    row_title = f"Motif: '{core_smiles}'\n(Total Occurrences: {total_group_count})"
                    # Set the label, make it horizontal, and adjust padding to give it space
                    axes[i, 0].set_ylabel(
                        row_title,
                        rotation=0,
                        size=14,
                        labelpad=100,
                        ha="center",
                        va="center",
                    )

                    for j, frag_info in enumerate(fragments):
                        ax = axes[i, j]
                        mol_to_draw, bit_info_frag = (
                            frag_info["example_mol"],
                            frag_info["bit_info"],
                        )
                        core_atoms = [idx for idx, rad in bit_info_frag if rad == 0]
                        env_atoms = [idx for idx, rad in bit_info_frag if rad > 0]

                        atom_colors = {
                            idx: (0.8, 0.8, 1.0) for idx in env_atoms
                        }  # Blue for environment
                        for idx in core_atoms:
                            atom_colors[idx] = (1.0, 0.6, 0.4)  # Orange for core

                        all_highlight_atoms_set = set(core_atoms + env_atoms)
                        attachment_bonds = []
                        for bond in mol_to_draw.GetBonds():
                            if (bond.GetBeginAtomIdx() in all_highlight_atoms_set) != (
                                bond.GetEndAtomIdx() in all_highlight_atoms_set
                            ):
                                attachment_bonds.append(bond.GetIdx())
                        bond_colors = {
                            bond_idx: (0.6, 0.2, 0.6) for bond_idx in attachment_bonds
                        }  # Purple for attachments

                        drawer = rdMolDraw2D.MolDraw2DCairo(1600, 1400)
                        drawer.drawOptions().addAtomIndices = True
                        rdMolDraw2D.PrepareAndDrawMolecule(
                            drawer,
                            mol_to_draw,
                            highlightAtoms=core_atoms + env_atoms,
                            highlightAtomColors=atom_colors,
                            highlightBonds=attachment_bonds,
                            highlightBondColors=bond_colors,
                        )
                        drawer.FinishDrawing()

                        img = Image.open(BytesIO(drawer.GetDrawingText()))
                        ax.imshow(img)
                        # FINAL TEXT FIX: Add the core SMILES back to the subplot title
                        ax.set_title(
                            f"SMILES: '{core_smiles}'\nContext {j+1} | Count: {frag_info['count']}",
                            fontsize=12,
                        )
                        ax.axis("off")

                    # Turn off unused axes in the row
                    for j in range(len(fragments), max_contexts):
                        axes[i, j].axis("off")

                plt.tight_layout(rect=[0, 0, 1, 0.96])

                frag_fig_path = os.path.join(
                    output_dir, f"top_substructures_contextual_{target_task}.png"
                )
                plt.savefig(frag_fig_path, dpi=300)
                plt.close(fig)
                console.log(
                    f"✅ Saved advanced contextual substructure analysis to [cyan]{frag_fig_path}[/cyan]"
                )

    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        traceback.print_exc()
        console.log(
            f"[bold red]Analysis failed. Temporary files are preserved for debugging at: {temp_dir}[/bold red]"
        )
    else:
        shutil.rmtree(temp_dir)

    console.rule("[bold green]Analysis Complete[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="Run CAGE-Fusion Global Model Analysis for publication insights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", help="Path to input CSV dataset for analysis.")
    parser.add_argument(
        "--h5-path",
        help="Path to h5 path, if provided along with graph path, will skip featurization.",
    )
    parser.add_argument(
        "--graph-path",
        help="Path to graph path, if provided along with h5 path, will skip featurization.",
    )

    parser.add_argument(
        "--checkpoint-dir", required=True, help="Path to the trained model directory."
    )
    parser.add_argument(
        "--task",
        required=True,
        help="The name of the target task (e.g., 'IsNuisance').",
    )
    parser.add_argument(
        "--output-dir",
        default="./model_insights",
        help="Directory to save analysis plots.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for analysis."
    )
    args = parser.parse_args()

    if not args.csv and (not args.h5_path or not args.graph_path):
        parser.error(
            "You must provide either a CSV file for analysis or both h5 and graph paths."
        )
    if args.h5_path and not args.graph_path:
        parser.error("If h5 path is provided, graph path must also be provided.")
    if args.graph_path and not args.h5_path:
        parser.error("If graph path is provided, h5 path must also be provided.")

    analyze_and_visualize(
        csv_path=args.csv,
        checkpoint_dir=args.checkpoint_dir,
        target_task=args.task,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        provided_h5_path=args.h5_path,
        provided_graph_path=args.graph_path,
    )


if __name__ == "__main__":
    main()
