import os
import gc
import joblib
import h5py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer
from transformers import AutoTokenizer, AutoModel

# Set environment variable to prevent tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clean_descriptors(x: np.ndarray) -> np.ndarray:
    """Clips and cleans auxiliary descriptor values, ensuring no NaNs or Infs."""
    if np.isnan(x).any() or np.isinf(x).any():
        print("⚠️ Warning: NaN or Inf found in auxiliary descriptors. Cleaning...")
        x = np.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    return np.clip(x, -1e4, 1e4)

def featurize_and_save_streaming(
    df: pd.DataFrame,
    name: str,
    label_cols: list,
    cache_dir: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    fit_scaler: bool = False,
    scaler: StandardScaler = None,
    batch_size: int = 32,
    graph_dump_interval: int = 10000
):
    """
    Featurizes a DataFrame in batches with enhanced error checking and saves to HDF5.
    
    This function creates datasets for token embeddings, input IDs, graph features,
    auxiliary features (like RDKit descriptors), and labels.
    """
    os.makedirs(cache_dir, exist_ok=True)
    h5_path = os.path.join(cache_dir, f"{name}_cage_fusion.h5")
    graph_path_base = os.path.join(cache_dir, f"{name}_graph_feats_part")
    scaler_path = os.path.join(cache_dir, "aux_features_scaler.pkl")
    bad_smiles_path = os.path.join(cache_dir, f"{name}_bad_smiles.csv")

    D_embedding = model.config.hidden_size
    D_seq_len = min(512, getattr(model.config, "max_position_embeddings", 512))
    VOCAB_SIZE = tokenizer.vocab_size
    print(f"✅ Featurizing '{name}'. Embedding Dim: {D_embedding}, Max Length: {D_seq_len}, Vocab Size: {VOCAB_SIZE}")

    descriptor_names = [desc[0] for desc in Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    graph_featurizer = SimpleMoleculeMolGraphFeaturizer()
    D_aux_feats = len(descriptor_names)

    # Filter out any SMILES that RDKit can't handle from the start
    df["mol"] = df["SMILES_Canonical"].apply(Chem.MolFromSmiles)
    if df["mol"].isnull().any():
        print(f"⚠️ Found {df['mol'].isnull().sum()} invalid SMILES strings. Removing them.")
        df = df.dropna(subset=['mol']).reset_index(drop=True)
    
    N = len(df)
    L = len(label_cols)

    # Check if featurization is already complete
    needs_featurization_loop = True
    if os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, "r") as f:
                if "embedding" in f and f["embedding"].shape[0] == N:
                    print(f"📦 Main featurization for '{name}' appears complete. Skipping main loop.")
                    needs_featurization_loop = False
        except Exception as e:
            print(f"HDF5 file for {name} might be corrupted. Re-running featurization. Error: {e}")
            os.remove(h5_path)
    
    if needs_featurization_loop:
        print(f"\n🔬 Starting main featurization for '{name}' ({N} rows).")
        
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("embedding", shape=(N, D_seq_len, D_embedding), dtype=np.float32)
            f.create_dataset("input_ids", shape=(N, D_seq_len), dtype=np.int32)
            f.create_dataset("auxiliary_features", shape=(N, D_aux_feats), dtype=np.float32)
            f.create_dataset("labels", shape=(N, L), dtype=np.float32)

        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(model_device)
        print(f"🤖 Model moved to: {model_device}")

        current_scaler = StandardScaler() if fit_scaler else scaler
        graph_feats = []
        graph_part = 0

        for i in tqdm(range(0, N, batch_size), desc=f"Featurizing {name}"):
            batch_df = df.iloc[i:i + batch_size]
            smiles_batch = batch_df["SMILES_Canonical"].tolist()

            try:
                inputs = tokenizer(smiles_batch, return_tensors="pt", padding="max_length", truncation=True, max_length=D_seq_len)
                input_ids = inputs["input_ids"]
                
                if (input_ids >= VOCAB_SIZE).any() or (input_ids < 0).any():
                    invalid_indices = torch.where((input_ids >= VOCAB_SIZE) | (input_ids < 0))[0].unique()
                    bad_smiles_in_batch = [smiles_batch[idx] for idx in invalid_indices]
                    raise ValueError(f"Invalid token ID found. Vocab size is {VOCAB_SIZE}. Bad SMILES: {bad_smiles_in_batch}")

                inputs_on_device = {k: v.to(model_device) for k, v in inputs.items()}
                with torch.no_grad():
                    output = model(**inputs_on_device)
                    embeddings = output.last_hidden_state
                    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                        raise ValueError(f"NaN or Inf in model output at batch starting at index {i}")

                with h5py.File(h5_path, "a") as f:
                    f["input_ids"][i:i + len(batch_df)] = input_ids.cpu().to(torch.int32)
                    f["embedding"][i:i + len(batch_df)] = embeddings.cpu().to(torch.float32)

            except Exception as e:
                print(f"🚨 CRITICAL ERROR during tokenization/inference at batch {i}-{i+batch_size}. Skipping. Error: {e}")
                bad_batch_df = pd.DataFrame({"SMILES_Canonical": smiles_batch})
                bad_batch_df.to_csv(bad_smiles_path, mode='a', header=not os.path.exists(bad_smiles_path), index=False)
                continue 

            with h5py.File(h5_path, "a") as f:
                batch_aux_feats = []
                for j, row in batch_df.iterrows():
                    current_global_idx = i + (j - batch_df.index[0])
                    graph_feats.append(graph_featurizer(row["mol"]))
                    
                    aux_desc = clean_descriptors(np.array(desc_calc.CalcDescriptors(row["mol"])))
                    f["auxiliary_features"][current_global_idx] = aux_desc
                    batch_aux_feats.append(aux_desc)

                    label_values = row[label_cols].values.astype(np.float32)
                    f["labels"][current_global_idx] = label_values

                if fit_scaler and batch_aux_feats:
                    current_scaler.partial_fit(np.array(batch_aux_feats))

            if len(graph_feats) >= graph_dump_interval:
                joblib.dump(graph_feats, f"{graph_path_base}_{graph_part}.pkl", compress=3)
                print(f"\n... Saved graph part {graph_part} ...")
                graph_feats.clear()
                graph_part += 1
            
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        if graph_feats:
            joblib.dump(graph_feats, f"{graph_path_base}_{graph_part}.pkl", compress=3)

        if fit_scaler:
            joblib.dump(current_scaler, scaler_path)
            print(f"✅ Scaler fitted and saved to {scaler_path}")
    
    # --- Normalization Step ---
    scaler_to_use = scaler if not fit_scaler else joblib.load(scaler_path)
    
    with h5py.File(h5_path, "a") as f:
        if scaler_to_use and ("auxiliary_features_normalized" not in f or f["auxiliary_features_normalized"].shape[0] != N):
            print(f"\n⚙️ Applying scaler to create 'auxiliary_features_normalized' for '{name}'...")
            if "auxiliary_features_normalized" in f:
                del f["auxiliary_features_normalized"]
            f.create_dataset("auxiliary_features_normalized", shape=(N, D_aux_feats), dtype=np.float32)
            
            for i_norm in tqdm(range(0, N, batch_size), desc=f"Normalizing '{name}'"):
                desc_batch = f["auxiliary_features"][i_norm:i_norm + batch_size]
                if desc_batch.shape[0] > 0:
                    f["auxiliary_features_normalized"][i_norm:i_norm + batch_size] = scaler_to_use.transform(desc_batch)
            print(f"✅ Finished normalizing '{name}'.")
        elif not scaler_to_use:
            print(f"⚠️ No scaler available for '{name}', cannot create 'auxiliary_features_normalized'.")
        else:
            print(f"✅ 'auxiliary_features_normalized' for '{name}' already exists and is correctly sized.")

    final_scaler_to_return = scaler_to_use
    return h5_path, graph_path_base + "_*.pkl", final_scaler_to_return
