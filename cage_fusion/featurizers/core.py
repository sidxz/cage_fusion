import os
import gc
import h5py
import joblib
import pandas as pd
import numpy as np
import glob
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer
from cage_fusion.utils.logging_utils import logger

from .helpers import (
    initialize_hdf5_file,
    featurize_batch,
    process_auxiliary_features,
    save_graph_features,
    normalize_auxiliary_features,
)


def clean_descriptors(x: np.ndarray) -> np.ndarray:
    """Sanitize and clip descriptor values."""
    if np.isnan(x).any() or np.isinf(x).any():
        # logger.warning("NaN or Inf found in auxiliary descriptors")
        x = np.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    return np.clip(x, -1e4, 1e4)


def featurize_and_save_streaming(
    df: pd.DataFrame,
    name: str,
    label_cols: list,
    cache_dir: str,
    tokenizer,
    model,
    fit_scaler: bool = False,
    scaler: StandardScaler = None,
    batch_size: int = 32,
    graph_dump_interval: int = 10000,
):
    """
    Stream featurization: saves embeddings, graph features, auxiliary descriptors, and labels.
    """
    os.makedirs(cache_dir, exist_ok=True)
    h5_path = os.path.join(cache_dir, f"{name}_cage_fusion.h5")
    graph_path_base = os.path.join(cache_dir, f"{name}_graph_feats_part")
    scaler_path = os.path.join(cache_dir, "aux_features_scaler.pkl")
    bad_smiles_path = os.path.join(cache_dir, f"{name}_bad_smiles.csv")

    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(model_device)

    D_embedding = model.config.hidden_size
    D_seq_len = min(512, getattr(model.config, "max_position_embeddings", 512))
    vocab_size = tokenizer.vocab_size

    descriptor_names = [desc[0] for desc in Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    graph_featurizer = SimpleMoleculeMolGraphFeaturizer()
    D_aux_feats = len(descriptor_names)

    # --- FIX: Ensure SMILES column is string type ---
    df["SMILES_Canonical"] = df["SMILES_Canonical"].astype(str)

    df["mol"] = df["SMILES_Canonical"].apply(Chem.MolFromSmiles)
    if df["mol"].isnull().any():
        n_bad = df["mol"].isnull().sum()
        logger.warning(f"Found {n_bad} invalid SMILES. Dropping.")
        df = df.dropna(subset=["mol"]).reset_index(drop=True)

    N = len(df)
    L = len(label_cols)

    run_featurization = True
    if os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, "r") as f:
                if "embedding" in f and f["embedding"].shape[0] == N:
                    logger.info(f"Featurization already exists for '{name}'. Skipping.")
                    run_featurization = False
        except Exception as e:
            logger.warning(f"HDF5 issue for '{name}': {str(e)}. Re-running.")
            os.remove(h5_path)

    if run_featurization:
        logger.info(f"Running featurization for {N} samples (name='{name}')")
        initialize_hdf5_file(h5_path, N, D_seq_len, D_embedding, D_aux_feats, L)

        current_scaler = StandardScaler() if fit_scaler else scaler
        graph_feats, graph_part = [], 0

        for i in tqdm(range(0, N, batch_size), desc=f"Featurizing {name}"):
            batch_df = df.iloc[i : i + batch_size]
            smiles_batch = batch_df["SMILES_Canonical"].tolist()

            try:
                input_ids, embeddings = featurize_batch(
                    tokenizer, model, smiles_batch, D_seq_len, model_device, vocab_size
                )
                with h5py.File(h5_path, "a") as f:
                    f["input_ids"][i : i + len(batch_df)] = input_ids
                    f["embedding"][i : i + len(batch_df)] = embeddings

            except Exception as e:
                logger.error(f"Batch {i}-{i + batch_size} failed: {str(e)}")
                pd.DataFrame({"SMILES_Canonical": smiles_batch}).to_csv(
                    bad_smiles_path,
                    mode="a",
                    header=not os.path.exists(bad_smiles_path),
                    index=False,
                )
                continue

            with h5py.File(h5_path, "a") as f:
                graph_feats = process_auxiliary_features(
                    batch_df,
                    i,
                    graph_feats,
                    graph_featurizer,
                    desc_calc,
                    label_cols,
                    current_scaler,
                    f,
                    fit_scaler,
                    clean_descriptors,
                )

            if len(graph_feats) >= graph_dump_interval:
                save_graph_features(graph_feats, graph_path_base, graph_part)
                graph_feats.clear()
                graph_part += 1

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if graph_feats:
            save_graph_features(graph_feats, graph_path_base, graph_part)

        if fit_scaler:
            joblib.dump(current_scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")

    final_scaler = scaler if not fit_scaler else joblib.load(scaler_path)

    if final_scaler:
        try:
            normalize_auxiliary_features(
                h5_path, final_scaler, D_aux_feats, batch_size, name
            )
        except NotFittedError as e:
            logger.error("Scaler must be fitted before normalization.")
            raise e
        
    # ---- Combine part files here ----
    glob_pattern = graph_path_base + "_*.pkl"
    combined_graph_path = os.path.join(cache_dir, f"{name}_graph_feats.pkl")
    part_files = sorted(glob.glob(glob_pattern))
    if not part_files:
        raise FileNotFoundError(f"No graph part files found: {glob_pattern}")
    logger.info(f"Combining {len(part_files)} graph parts into {combined_graph_path} ...")
    all_feats = [feat for pf in part_files for feat in joblib.load(pf)]
    joblib.dump(all_feats, combined_graph_path, compress=3)

    return h5_path, graph_path_base + "_*.pkl", final_scaler
