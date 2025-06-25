#!/usr/bin/env python3
"""
Corrected CAGE-Fusion Global Model Analyzer

This script analyzes a trained CAGE-Fusion model to extract global insights,
suitable for research publication. It processes an entire dataset to:

1.  Identify and rank the top auxiliary features (e.g., physicochemical
    descriptors) that consistently contribute to a positive prediction (e.g.,
    classifying a compound as a "nuisance compound"). Importance is measured
    by the aggregated magnitude of the feature's gradient.

2.  Identify and visualize the most common molecular substructures (via Morgan
    fingerprints) that the model's graph encoder learns to associate with a
    positive outcome. It highlights the most salient atoms based on their
    aggregated gradient norms.

This version corrects a critical flaw in the original script's gradient
calculation by performing a single, efficient backward pass per batch rather
than an incorrect, iterative approach. This ensures the resulting feature
attributions are valid and reliable.
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
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use the project's own modules to ensure data is created correctly
from cage_fusion.models.cage import CAGEFusionModel
from cage_fusion.utils.logging_utils import logger
from cage_fusion.featurizers.core import featurize_and_save_streaming
from cage_fusion.engine.dataset import CageFusionStreamingDataset
from cage_fusion.engine.data_utils import collate_fn_for_cage_fusion
from cage_fusion.engine.utils import move_bmg_to_device

from rich.console import Console

console = Console()


def analyze_and_visualize(
    csv_path: str,
    checkpoint_dir: str,
    target_task: str,
    output_dir: str,
    top_n_features: int = 15,
    top_n_fragments: int = 12,
    batch_size: int = 32,
):
    """
    Main function to analyze a dataset and generate global insight visualizations.
    """
    console.rule(f"[bold green]Starting CAGE-Fusion Global Model Analysis[/bold green]")
    os.makedirs(output_dir, exist_ok=True)

    # === 1. Load Model and Configuration ===
    console.log("Loading model and configuration...")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")  #
    scaler_path = os.path.join(checkpoint_dir, "aux_features_scaler.pkl")  #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Required model file not found: {best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)  #
    config = checkpoint["config"]  #
    tasks = config["tasks"]  #

    if target_task not in tasks:
        raise ValueError(
            f"Target task '{target_task}' not found in model's tasks: {tasks}"
        )
    target_task_index = tasks.index(target_task)  #

    model = CAGEFusionModel(config).to(device)  #
    model.load_state_dict(checkpoint["model_state_dict"])  #
    model.eval()  #

    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])  #
    embedding_model = (
        AutoModel.from_pretrained(config["model_checkpoint"]).to(device).eval()
    )  #
    scaler = joblib.load(scaler_path)  #
    console.log(
        f"✅ Model, tokenizer, and scaler loaded. Using device: [bold cyan]{device}[/bold cyan]"
    )

    # === 2. Featurize the Dataset ===
    console.log(f"Loading and featurizing data from {csv_path}...")
    input_df = pd.read_csv(csv_path)  #
    temp_dir = tempfile.mkdtemp()

    try:
        h5_path, graph_path, _ = featurize_and_save_streaming(
            df=input_df,
            name="analysis_temp",
            label_cols=tasks,
            cache_dir=temp_dir,
            tokenizer=tokenizer,
            model=embedding_model,
            fit_scaler=False,
            scaler=scaler,
        )  #
        dataset = CageFusionStreamingDataset(
            h5_path, graph_path, tokenizer.pad_token_id
        )  #
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_for_cage_fusion,
            shuffle=False,
            num_workers=0,
        )  #

        console.log("✅ Dataset featurized. Aggregating model insights...")
        total_aux_grads = np.zeros(config["aux_feature_dim"])
        fragment_counter = Counter()
        fragment_examples = {}

        # === 3. CORRECTED: Aggregate Gradients Across the Dataset ===
        for batch in tqdm(loader, desc="Analyzing Dataset Batches"):
            bmg, token_embs, attn_mask, aux_feats, _, input_ids, smiles_batch = batch  #

            # Move data to device and enable gradient tracking on inputs
            bmg = move_bmg_to_device(bmg, device)
            token_embs, attn_mask, aux_feats, input_ids = (
                token_embs.to(device),
                attn_mask.to(device),
                aux_feats.to(device),
                input_ids.to(device),
            )
            aux_feats.requires_grad_(True)
            model.zero_grad()

            # --- SINGLE FORWARD PASS ---
            # We must set return_attn=True to get intermediate `atom_features`
            logits, _, _, _, _, _, _, atom_features = model(
                bmg=bmg,
                sequence_embeddings=token_embs,
                attn_mask=attn_mask,
                aux_feats=aux_feats,
                input_ids_batch=input_ids,
                return_attn=True,
            )

            # --- BATCH-LEVEL GRADIENT CALCULATION (THE CORRECTION) ---
            probs = torch.sigmoid(logits[:, target_task_index])
            positive_indices = torch.where(probs >= 0.5)[0]

            # If no positive predictions in this batch, skip to the next
            if len(positive_indices) == 0:
                continue

            # Register hook to capture gradients of intermediate `atom_features`
            atom_grads_storage = []
            if atom_features is not None:

                def atom_grad_hook(grad):
                    atom_grads_storage.append(grad)

                atom_features.register_hook(atom_grad_hook)

            # Sum the logits of positive predictions to create a single scalar score
            score_to_backprop = logits[positive_indices, target_task_index].sum()

            # --- SINGLE BACKWARD PASS ---
            # This correctly computes gradients for all inputs w.r.t the total score
            score_to_backprop.backward()

            # --- AGGREGATE CORRECT GRADIENTS ---
            # 1. Aggregate Auxiliary Feature Gradients
            if aux_feats.grad is not None:
                # Sum the absolute gradients for the positive samples in the batch
                batch_aux_grads = (
                    aux_feats.grad[positive_indices].abs().sum(dim=0).cpu().numpy()
                )
                total_aux_grads += batch_aux_grads

            # 2. Aggregate Atom-Level (Graph) Gradients and Identify Fragments
            if atom_grads_storage:
                batch_atom_grads = atom_grads_storage[0]
                for j_idx in positive_indices:
                    smiles = smiles_batch[j_idx.item()]
                    mol = Chem.MolFromSmiles(smiles)
                    if not mol:
                        continue

                    # Isolate the atom gradients for the j-th molecule
                    mol_atom_grads_tensor = batch_atom_grads[bmg.batch == j_idx.item()]
                    mol_atom_grads_norm = (
                        mol_atom_grads_tensor.norm(dim=-1).cpu().numpy()
                    )

                    # Find the top 3 most salient atoms based on gradient norm
                    top_atom_indices = np.argsort(mol_atom_grads_norm)[-3:]

                    # Generate fragments centered on these salient atoms
                    for atom_idx in top_atom_indices:
                        fp = AllChem.GetMorganFingerprint(
                            mol, 2, fromAtoms=[int(atom_idx)]
                        )
                        for frag_id, _ in fp.GetNonzeroElements().items():
                            fragment_counter.update([frag_id])
                            # Store the first SMILES example we see for this fragment
                            if frag_id not in fragment_examples:
                                fragment_examples[frag_id] = smiles

        # === 4. Generate Publication-Quality Visualizations ===
        console.log("✅ Gradient aggregation complete. Generating visualizations...")

        # --- Auxiliary Features Bar Plot ---
        descriptor_names = [desc[0] for desc in Descriptors._descList]  #
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
            f"Top {top_n_features} Influential Auxiliary Features for '{target_task}'\n(Model's Association with Nuisance Compounds)",
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

        # --- Top Substructures Grid Image ---
        if fragment_counter:
            top_fragments = fragment_counter.most_common(top_n_fragments)

            mols_to_draw, legends, highlight_atom_lists = [], [], []
            for i, (frag_id, count) in enumerate(top_fragments):
                if frag_id not in fragment_examples:
                    continue

                smiles = fragment_examples[frag_id]
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    continue

                # Find the atoms that correspond to this fragment ID
                bit_info = {}
                AllChem.GetMorganFingerprint(mol, 2, bitInfo=bit_info)
                if frag_id in bit_info:
                    atoms_to_highlight = list(
                        {a for a, r in bit_info[frag_id]}
                    )  # Use a list

                    # --- NEW: Generate a SMILES string for the fragment ---
                    try:
                        # RDKit function to get a SMILES string from a list of atom indices
                        fragment_smiles = Chem.MolFragmentToSmiles(
                            mol,
                            atomsToUse=atoms_to_highlight,
                            isomericSmiles=True,
                            canonical=True,
                        )
                    except Exception:
                        fragment_smiles = "[Fragment Error]"  # Add a fallback
                    # --------------------------------------------------------

                    mols_to_draw.append(mol)

                    # --- MODIFIED: Update the legend to use the fragment SMILES ---
                    # Use a newline character '\n' for better formatting
                    legends.append(f"Fragment: {fragment_smiles}\nCount: {count}")
                    # -------------------------------------------------------------

                    highlight_atom_lists.append(atoms_to_highlight)

            # Generate a clean grid image using RDKit
            if mols_to_draw:
                img = Draw.MolsToGridImage(
                    mols_to_draw,
                    molsPerRow=4,
                    subImgSize=(300, 300),
                    legends=legends,
                    highlightAtomLists=highlight_atom_lists,
                    useSVG=False,
                )
                frag_fig_path = os.path.join(
                    output_dir, f"top_substructures_{target_task}.png"
                )
                img.save(frag_fig_path)
                console.log(
                    f"✅ Saved substructure analysis to [cyan]{frag_fig_path}[/cyan]"
                )

    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        traceback.print_exc()
        console.log(
            f"[bold red]Analysis failed. Temporary files are preserved for debugging at: {temp_dir}[/bold red]"
        )
    else:
        # Clean up temporary directory only on success
        shutil.rmtree(temp_dir)

    console.rule("[bold green]Analysis Complete[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="Run CAGE-Fusion Global Model Analysis for publication insights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv", required=True, help="Path to input CSV dataset for analysis."
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

    analyze_and_visualize(
        csv_path=args.csv,
        checkpoint_dir=args.checkpoint_dir,
        target_task=args.task,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
