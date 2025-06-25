#!/usr/bin/env python3
"""
Analyzes a trained CAGE-Fusion model to extract global insights.

This script processes an entire dataset to:
1.  Identify the top auxiliary features that consistently contribute to model predictions.
2.  Identify and visualize the most common molecular substructures (fragments)
    that the model learns to associate with a positive outcome.
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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use the project's own modules to ensure data is created correctly
from cage_fusion.models.cage import CAGEFusionModel
from cage_fusion.configs import get_default_config
from cage_fusion.utils.logging_utils import logger
from cage_fusion.featurizers.core import featurize_and_save_streaming
from cage_fusion.engine.dataset import CageFusionStreamingDataset
from cage_fusion.engine.data_utils import collate_fn_for_cage_fusion
from cage_fusion.engine.utils import move_bmg_to_device

# Rich console for beautiful output
from rich.console import Console
console = Console()

def analyze_and_visualize(
    csv_path: str,
    checkpoint_dir: str,
    target_task: str,
    output_dir: str,
    top_n_features: int = 15,
    top_n_fragments: int = 12,
):
    """
    Main function to analyze a dataset and generate global insight visualizations.
    """
    console.rule(f"[bold green]Starting CAGE-Fusion Global Model Analysis[/bold green]")
    os.makedirs(output_dir, exist_ok=True)

    # === 1. Load Model and Configuration ===
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    scaler_path = os.path.join(checkpoint_dir, "aux_features_scaler.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tasks = config["tasks"]
    target_task_index = tasks.index(target_task)
    
    model = CAGEFusionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    embedding_model = AutoModel.from_pretrained(config["model_checkpoint"]).to(device).eval()
    scaler = joblib.load(scaler_path)

    # === 2. Process the entire dataset ===
    logger.info(f"Loading and featurizing data from {csv_path}...")
    input_df = pd.read_csv(csv_path)
    temp_dir = tempfile.mkdtemp()
    
    try:
        h5_path, graph_path, _ = featurize_and_save_streaming(
            df=input_df, name="analysis_temp", label_cols=tasks, cache_dir=temp_dir,
            tokenizer=tokenizer, model=embedding_model, fit_scaler=False, scaler=scaler
        )
        dataset = CageFusionStreamingDataset(h5_path, graph_path, tokenizer.pad_token_id)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, collate_fn=collate_fn_for_cage_fusion, shuffle=False, num_workers=0
        )

        logger.info("Aggregating model insights across the dataset...")
        total_aux_grads = np.zeros(config["aux_feature_dim"])
        fragment_counter = Counter()
        fragment_examples = {}

        for i, batch in enumerate(tqdm(loader, desc="Analyzing Dataset")):
            bmg, token_embs, attn_mask, aux_feats, labels, input_ids, smiles_batch = batch
            bmg = move_bmg_to_device(bmg, device)
            token_embs, attn_mask, aux_feats, input_ids = (
                token_embs.to(device), attn_mask.to(device),
                aux_feats.to(device), input_ids.to(device)
            )
            
            token_embs.requires_grad_(True)
            aux_feats.requires_grad_(True)
            model.zero_grad()

            logits, _, _, g2t_weights, _, _, _ = model(
                bmg=bmg, sequence_embeddings=token_embs, attn_mask=attn_mask,
                aux_feats=aux_feats, input_ids_batch=input_ids, return_attn=True
            )
            
            for j in range(logits.shape[0]):
                prob = torch.sigmoid(logits[j, target_task_index]).item()
                if prob < 0.5:
                    continue

                model.zero_grad()
                logits[j, target_task_index].backward(retain_graph=True)

                if aux_feats.grad is not None:
                    total_aux_grads += np.abs(aux_feats.grad[j].cpu().numpy())
                
                if g2t_weights is not None:
                    atom_importance = g2t_weights[j].sum(dim=(0, 1)).cpu().detach().numpy()
                    
                    mol_atom_indices = np.where(bmg.batch.cpu().numpy() == j)[0]
                    num_atoms = len(mol_atom_indices)
                    
                    mol = Chem.MolFromSmiles(smiles_batch[j])
                    if not mol or num_atoms == 0:
                        continue

                    top_atom_indices_local = np.argsort(atom_importance[:num_atoms])[-3:]

                    for atom_idx in top_atom_indices_local:
                        fp = AllChem.GetMorganFingerprint(mol, 2, fromAtoms=[int(atom_idx)])
                        for frag_id in fp.GetNonzeroElements().keys():
                            fragment_counter.update([frag_id])
                            if frag_id not in fragment_examples:
                                fragment_examples[frag_id] = smiles_batch[j]

        # === 4. Visualize Results (after the loop completes) ===
        logger.info("Generating visualizations...")
        
        # Auxiliary Features Visualization
        descriptor_names = [desc[0] for desc in Descriptors._descList]
        feature_importance_df = pd.DataFrame({
            'feature': descriptor_names,
            'importance': total_aux_grads
        }).sort_values('importance', ascending=False).head(top_n_features)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
        plt.title(f'Top {top_n_features} Influential Auxiliary Features for Task: {target_task}', fontsize=16)
        plt.xlabel('Aggregated Absolute Gradient', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        aux_fig_path = os.path.join(output_dir, f'top_aux_features_{target_task}.png')
        plt.savefig(aux_fig_path)
        plt.close()
        console.log(f"✅ Saved auxiliary feature analysis to [cyan]{aux_fig_path}[/cyan]")

        # Substructure Visualization
        if fragment_counter:
            top_fragments = fragment_counter.most_common(top_n_fragments)
            
            # FIX: Manually draw each fragment and assemble into a grid for robustness.
            img_size = 500
            mols_per_row = 4
            num_rows = (len(top_fragments) + mols_per_row - 1) // mols_per_row
            
            # Create a blank canvas for the final grid image
            full_img = Image.new('RGB', (mols_per_row * img_size, num_rows * (img_size + 40)), 'white')
            
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            for i, (frag_id, count) in enumerate(top_fragments):
                if frag_id in fragment_examples:
                    example_smiles = fragment_examples[frag_id]
                    mol = Chem.MolFromSmiles(example_smiles)
                    if not mol: continue
                    
                    # Find the atoms and bonds involved in this fragment ID
                    info = {}
                    AllChem.GetMorganFingerprint(mol, 2, bitInfo=info)
                    if frag_id in info:
                        atoms_to_highlight = {a for a, r in info[frag_id]}
                        
                        # Draw the molecule with the highlighted fragment
                        drawer = rdMolDraw2D.MolDraw2DCairo(img_size, img_size)
                        drawer.drawOptions().addAtomIndices = True
                        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=list(atoms_to_highlight))
                        drawer.FinishDrawing()
                        
                        # Convert drawing to a PIL image
                        pil_img = Image.open(io.BytesIO(drawer.GetDrawingText()))
                        
                        # Paste the image onto the canvas
                        row = i // mols_per_row
                        col = i % mols_per_row
                        x_offset = col * img_size
                        y_offset = row * (img_size + 40)
                        full_img.paste(pil_img, (x_offset, y_offset))
                        
                        # Add the legend
                        draw = ImageDraw.Draw(full_img)
                        legend = f"Fragment ID: {frag_id} (Count: {count})"
                        text_bbox = draw.textbbox((0, 0), legend, font=font)
                        text_x = x_offset + (img_size - (text_bbox[2] - text_bbox[0])) / 2
                        text_y = y_offset + img_size + 10
                        draw.text((text_x, text_y), legend, font=font, fill="black")

            frag_fig_path = os.path.join(output_dir, f'top_substructures_{target_task}.png')
            full_img.save(frag_fig_path)
            console.log(f"✅ Saved substructure analysis to [cyan]{frag_fig_path}[/cyan]")


    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        traceback.print_exc()
        logger.warning(f"Temporary files are preserved for debugging at: {temp_dir}")
    else:
        # Only clean up if no errors occurred
        shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description="Analyze a CAGE-Fusion model for global insights.")
    parser.add_argument("--csv", required=True, help="Path to input CSV for analysis (e.g., test set).")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to the directory with the trained model.")
    parser.add_argument("--task", required=True, help="The name of the target task to analyze.")
    parser.add_argument("--output-dir", default="./model_insights", help="Directory to save analysis plots.")
    args = parser.parse_args()
    analyze_and_visualize(args.csv, args.checkpoint_dir, args.task, args.output_dir)

if __name__ == "__main__":
    import io # Add this import for the fix
    main()
