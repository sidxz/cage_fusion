"""
End-to-end featurisation: SMILES DataFrame → HDF5 file.

The main entry point is ``featurize_and_save_streaming()``.  It converts a
pandas DataFrame of SMILES (with optional labels) into a single HDF5 file
containing:

- ``embedding``                       – ChemBERTa token embeddings [N, T, D]
- ``input_ids``                       – tokeniser IDs [N, T]
- ``auxiliary_features``              – raw RDKit descriptors [N, 217]
- ``auxiliary_features_normalized``   – StandardScaler-normalised [N, 217]
- ``graph_bytes``                     – pickled ChemProp MolGraph objects [N]
- ``labels``                          – binary/continuous targets [N, L]  (optional)
- ``original_indices``, ``smiles``, ``ids`` – provenance
"""

from __future__ import annotations

import gc
import logging
import os
import pickle
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rich.console import Console
from rich.table import Table
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer

from cage_fusion.featurization.featurizer_utils import (
    initialize_hdf5_file,
    featurize_batch,
    process_aux_feats,
    process_graphs,
    process_labels,
    normalize_auxiliary_features,
)

logger = logging.getLogger("cagefusion")


def _clean_descriptors(x: np.ndarray) -> np.ndarray:
    if np.isnan(x).any() or np.isinf(x).any():
        x = np.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    return np.clip(x, -1e4, 1e4)


def featurize_and_save_streaming(
    df: pd.DataFrame,
    name: str,
    label_cols: List[str],
    cache_dir: str,
    tokenizer,
    model,
    fit_scaler: bool = False,
    scaler: Optional[StandardScaler] = None,
    batch_size: int = 32,
    id_col: Optional[str] = "Id",
) -> Tuple[str, Optional[StandardScaler], int]:
    """
    Featurise a DataFrame of SMILES and write results to HDF5.

    Parameters
    ----------
    df:
        DataFrame with a ``SMILES`` column and optional label columns.
    name:
        Prefix for the output ``.h5`` file (e.g. ``"train"``).
    label_cols:
        Column names to use as labels.  Pass ``[]`` for inference mode.
    cache_dir:
        Directory where the HDF5 file is written.
    tokenizer, model:
        Pre-trained HuggingFace tokenizer and model for embedding.
    fit_scaler:
        If ``True``, fit a new ``StandardScaler`` on auxiliary features.
        If ``False``, *scaler* must be a fitted scaler (prediction mode).
    scaler:
        An existing ``StandardScaler`` (required when ``fit_scaler=False``).
    batch_size:
        Molecules per featurisation batch.
    id_col:
        DataFrame column to store as the ``ids`` field.  ``None`` disables.

    Returns
    -------
    h5_path:
        Absolute path to the created HDF5 file.
    scaler:
        The fitted or provided scaler (``None`` if neither was available).
    n_written:
        Number of molecules successfully written.
    """
    os.makedirs(cache_dir, exist_ok=True)
    h5_path = os.path.join(cache_dir, f"{name}_cage_fusion.h5")
    bad_smiles_path = os.path.join(cache_dir, f"{name}_bad_smiles.csv")

    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(model_device)

    D_embedding = int(model.config.hidden_size)
    if D_embedding <= 0:
        raise ValueError("Model embedding size must be > 0.")
    D_seq_len = min(512, int(getattr(model.config, "max_position_embeddings", 512)))

    descriptor_names = [desc[0] for desc in Descriptors.descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    graph_featurizer = SimpleMoleculeMolGraphFeaturizer()
    D_aux_feats = len(descriptor_names)

    if "original_index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "original_index"})

    has_ids = bool(id_col) and (id_col in df.columns)
    df["SMILES"] = df["SMILES"].astype(str)
    df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)

    n_bad = int(df["mol"].isnull().sum())
    if n_bad:
        logger.warning("Dropping %d invalid SMILES.", n_bad)
        df = df.dropna(subset=["mol"]).reset_index(drop=True)

    N = len(df)
    label_cols_present = [c for c in label_cols if c in df.columns]
    if len(label_cols_present) != len(label_cols):
        missing = sorted(set(label_cols) - set(label_cols_present))
        logger.warning("Label columns missing – prediction mode. Missing: %s", missing)
    L = len(label_cols)

    table = Table(title="Featurisation parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    for k, v in [
        ("HDF5 path", h5_path), ("N samples", N), ("seq_len", D_seq_len),
        ("embed_dim", D_embedding), ("aux_dim", D_aux_feats),
        ("num_labels", L), ("ids", has_ids), ("batch_size", batch_size),
    ]:
        table.add_row(str(k), str(v))
    Console().print(table)

    initialize_hdf5_file(h5_path, N, D_seq_len, D_embedding, D_aux_feats, L, ids_enabled=has_ids)

    current_scaler = StandardScaler() if fit_scaler else scaler
    returned_scaler = current_scaler if fit_scaler else scaler
    write_idx = 0

    with h5py.File(h5_path, "a") as f:
        f["auxiliary_features"].attrs["descriptor_names"] = np.array(descriptor_names, dtype=object)
        f["input_ids"].attrs["tokenizer"] = getattr(tokenizer, "name_or_path", "unknown")
        if has_ids:
            f["ids"].attrs["source_column"] = id_col

        for i in tqdm(range(0, N, batch_size), desc=f"Featurising {name}"):
            batch_df = df.iloc[i : i + batch_size]
            bs = len(batch_df)
            # Use RDKit canonical SMILES (mol objects already parsed & validated).
            # This eliminates "not removing hydrogen atom without neighbors" warnings
            # at training time and ensures ChemBERTa sees its native canonical form.
            smiles_batch = [Chem.MolToSmiles(mol) for mol in batch_df["mol"]]
            orig_idx_batch = batch_df["original_index"].tolist()
            ids_batch = batch_df[id_col].astype(str).tolist() if has_ids else None

            try:
                input_ids, embeddings = featurize_batch(
                    tokenizer, model, smiles_batch, D_seq_len, model_device, 0
                )
                if input_ids.shape != (bs, D_seq_len):
                    raise ValueError(f"input_ids shape mismatch: {input_ids.shape}")
                if embeddings.shape != (bs, D_seq_len, D_embedding):
                    raise ValueError(f"embeddings shape mismatch: {embeddings.shape}")

                batch_aux = process_aux_feats(
                    batch_df, desc_calc, current_scaler, fit_scaler, _clean_descriptors
                )
                if len(batch_aux) != bs:
                    raise ValueError("Aux-feat count mismatch")

                if L > 0:
                    batch_labels = process_labels(batch_df, label_cols)
                    if batch_labels.shape != (bs, L):
                        raise ValueError(f"Labels shape mismatch: {batch_labels.shape}")

                batch_graphs = process_graphs(batch_df, graph_featurizer)
                batch_graph_bytes = [
                    np.frombuffer(pickle.dumps(g, protocol=5), dtype=np.uint8)
                    for g in batch_graphs
                ]

                f["original_indices"][write_idx : write_idx + bs] = np.asarray(orig_idx_batch, dtype=np.int64)
                f["smiles"][write_idx : write_idx + bs] = smiles_batch
                if has_ids:
                    f["ids"][write_idx : write_idx + bs] = ids_batch
                f["input_ids"][write_idx : write_idx + bs] = input_ids
                f["embedding"][write_idx : write_idx + bs] = embeddings
                f["auxiliary_features"][write_idx : write_idx + bs] = np.asarray(batch_aux, dtype=np.float32)
                f["graph_bytes"][write_idx : write_idx + bs] = batch_graph_bytes
                if L > 0:
                    f["labels"][write_idx : write_idx + bs] = batch_labels
                write_idx += bs

            except Exception as e:
                logger.error("Batch at index %d failed (skipped): %s", i, e)
                bad = {"original_index": orig_idx_batch, "SMILES": smiles_batch}
                if has_ids:
                    bad[id_col] = ids_batch
                pd.DataFrame(bad).to_csv(
                    bad_smiles_path, mode="a",
                    header=not os.path.exists(bad_smiles_path), index=False,
                )
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Resize datasets to actual written count
        for dset in ["embedding", "input_ids", "auxiliary_features",
                     "auxiliary_features_normalized", "labels",
                     "original_indices", "smiles", "graph_bytes", "ids"]:
            if dset in f and f[dset].shape[0] != write_idx:
                f[dset].resize(write_idx, axis=0)

    if returned_scaler is not None and write_idx > 0:
        try:
            normalize_auxiliary_features(h5_path, returned_scaler, D_aux_feats, batch_size, name)
        except NotFittedError:
            logger.error("Scaler not fitted – cannot normalise aux features.")
            raise
    elif returned_scaler is None:
        logger.warning("No scaler available; skipping normalisation.")

    return h5_path, returned_scaler, write_idx
