# In cage_fusion/featurizers/core.py

from __future__ import annotations

import os, gc, h5py, pandas as pd, numpy as np, torch, pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer
from cage_fusion.utils.logging_utils import logger
from rich.table import Table
from rich.console import Console
from typing import List, Tuple, Optional

from .helpers import (
    initialize_hdf5_file,
    featurize_batch,  # returns np.int32 input_ids, np.float32 embeddings
    process_aux_feats,
    process_graphs,
    process_labels,
    normalize_auxiliary_features,
)


def clean_descriptors(x: np.ndarray) -> np.ndarray:
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
    id_col: Optional[
        str
    ] = "Id",  # which column to copy into HDF5 as 'ids' (if present)
) -> Tuple[str, Optional[StandardScaler], int]:

    os.makedirs(cache_dir, exist_ok=True)
    h5_path = os.path.join(cache_dir, f"{name}_cage_fusion.h5")
    bad_smiles_path = os.path.join(cache_dir, f"{name}_bad_smiles.csv")

    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(model_device)

    D_embedding = int(model.config.hidden_size)
    if D_embedding <= 0:
        raise ValueError("Model embedding size must be greater than 0.")
    D_seq_len = min(512, int(getattr(model.config, "max_position_embeddings", 512)))

    # RDKit descriptors
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    graph_featurizer = SimpleMoleculeMolGraphFeaturizer()
    D_aux_feats = len(descriptor_names)

    # Track original order (before dropping invalid mols)
    if "original_index" not in df.columns:
        df = df.reset_index().rename(columns={"index": "original_index"})

    # Normalize id column name if present
    has_ids = bool(id_col) and (id_col in df.columns)

    df["SMILES"] = df["SMILES"].astype(str)
    df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)

    if df["mol"].isnull().any():
        n_bad = int(df["mol"].isnull().sum())
        logger.warning(f"Found {n_bad} invalid SMILES. Dropping.")
        df = df.dropna(subset=["mol"]).reset_index(drop=True)

    N = len(df)
    label_cols = label_cols or []
    label_cols_present = [c for c in label_cols if c in df.columns]
    if len(label_cols_present) != len(label_cols):
        missing = sorted(set(label_cols) - set(label_cols_present))
        if missing:
            logger.warning(
                "Label columns missing in input DataFrame. "
                "Switching to prediction mode without labels. Missing: %s",
                ", ".join(missing),
            )
    L = len(label_cols)

    table = Table(title="HDF5 Initialization Parameters")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_row("HDF5 Path", h5_path)
    table.add_row("Num Samples (N)", str(N))
    table.add_row("Sequence Length (D_seq_len)", str(D_seq_len))
    table.add_row("Embedding Size (D_embedding)", str(D_embedding))
    table.add_row("Auxiliary Features (D_aux_feats)", str(D_aux_feats))
    table.add_row("Num Labels (L)", str(L))
    table.add_row("IDs present", str(has_ids))
    table.add_row("Batch Size", str(batch_size))
    Console().print(table)

    # Initialize HDF5; pass ids_enabled flag
    # NOTE: initialize_hdf5_file should create "embedding" as float32 (no AMP/fp16).
    initialize_hdf5_file(
        h5_path, N, D_seq_len, D_embedding, D_aux_feats, L, ids_enabled=has_ids
    )

    current_scaler = StandardScaler() if fit_scaler else scaler
    returned_scaler = current_scaler if fit_scaler else scaler

    write_idx = 0

    with h5py.File(h5_path, "a") as f:
        # provenance
        f["auxiliary_features"].attrs["descriptor_names"] = np.array(
            descriptor_names, dtype=object
        )
        f["input_ids"].attrs["tokenizer"] = getattr(
            tokenizer, "name_or_path", "unknown"
        )
        if has_ids:
            f["ids"].attrs["source_column"] = id_col

        for i in tqdm(range(0, N, batch_size), desc=f"Featurizing {name}"):
            batch_df = df.iloc[i : i + batch_size]
            bs = len(batch_df)

            smiles_batch = batch_df["SMILES"].tolist()
            original_indices_batch = batch_df["original_index"].tolist()
            ids_batch = batch_df[id_col].astype(str).tolist() if has_ids else None

            try:
                # featurize_batch returns np.int32 input_ids and np.float32 embeddings
                input_ids, embeddings = featurize_batch(
                    tokenizer, model, smiles_batch, D_seq_len, model_device, 0
                )
                if input_ids.shape != (bs, D_seq_len):
                    raise ValueError(
                        f"Input IDs shape mismatch: expected {(bs, D_seq_len)}, got {input_ids.shape}"
                    )
                if embeddings.shape != (bs, D_seq_len, D_embedding):
                    raise ValueError(
                        f"Embeddings shape mismatch: expected {(bs, D_seq_len, D_embedding)}, got {embeddings.shape}"
                    )

                batch_aux = process_aux_feats(
                    batch_df=batch_df,
                    desc_calc=desc_calc,
                    scaler=current_scaler,
                    fit_scaler=fit_scaler,
                    clean_descriptors=clean_descriptors,
                )
                if len(batch_aux) != bs:
                    raise ValueError(
                        f"Auxiliary features shape mismatch: expected {bs}, got {len(batch_aux)}"
                    )
                if L > 0:
                    batch_labels = process_labels(
                        batch_df=batch_df, label_cols=label_cols
                    )
                    if L > 0 and batch_labels.shape != (bs, L):
                        raise ValueError(
                            f"Labels shape mismatch: expected {(bs, L)}, got {batch_labels.shape}"
                        )

                batch_graph_feats = process_graphs(
                    batch_df=batch_df, graph_featurizer=graph_featurizer
                )
                batch_graph_bytes = [
                    np.frombuffer(pickle.dumps(g, protocol=5), dtype=np.uint8)
                    for g in batch_graph_feats
                ]
                if len(batch_graph_bytes) != bs:
                    raise ValueError("Graph features count mismatch vs batch size")

                # aligned writes
                f["original_indices"][write_idx : write_idx + bs] = np.asarray(
                    original_indices_batch, dtype=np.int64
                )
                f["smiles"][write_idx : write_idx + bs] = smiles_batch
                if has_ids:
                    f["ids"][write_idx : write_idx + bs] = ids_batch
                f["input_ids"][write_idx : write_idx + bs] = input_ids
                # embeddings are np.float32; dataset "embedding" is float32
                f["embedding"][write_idx : write_idx + bs] = embeddings
                f["auxiliary_features"][write_idx : write_idx + bs] = np.asarray(
                    batch_aux, dtype=np.float32
                )
                f["graph_bytes"][write_idx : write_idx + bs] = batch_graph_bytes
                if L > 0:
                    f["labels"][write_idx : write_idx + bs] = batch_labels

                write_idx += bs

            except Exception as e:
                logger.error(
                    "Batch starting at index %d failed and was SKIPPED: %s", i, str(e)
                )
                # keep Id + original_index in the bad list if present
                bad = {"original_index": original_indices_batch, "SMILES": smiles_batch}
                if has_ids:
                    bad[id_col] = ids_batch
                pd.DataFrame(bad).to_csv(
                    bad_smiles_path,
                    mode="a",
                    header=not os.path.exists(bad_smiles_path),
                    index=False,
                )
                continue

            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Resize to actual count (if some batches were skipped)
        for dset in [
            "embedding",
            "input_ids",
            "auxiliary_features",
            "auxiliary_features_normalized",
            "labels",
            "original_indices",
            "smiles",
            "graph_bytes",
            "ids",  # 'ids' may or may not exist
        ]:
            if dset in f and f[dset].shape[0] != write_idx:
                f[dset].resize(write_idx, axis=0)

    # Normalize AFTER resize; only if we actually wrote rows
    if returned_scaler is not None and write_idx > 0:
        try:
            normalize_auxiliary_features(
                h5_path, returned_scaler, D_aux_feats, batch_size, name
            )
        except NotFittedError as e:
            logger.error(
                "Scaler must be fitted before normalization. This should not happen in prediction mode."
            )
            raise e
    elif returned_scaler is None:
        logger.warning("No scaler available. Skipping normalization.")

    # Final integrity asserts
    with h5py.File(h5_path, "r") as f:
        n = write_idx
        assert f["embedding"].shape[0] == n
        assert f["input_ids"].shape[0] == n
        assert f["auxiliary_features"].shape[0] == n
        if "auxiliary_features_normalized" in f:
            assert f["auxiliary_features_normalized"].shape[0] == n
        assert f["original_indices"].shape[0] == n
        assert f["smiles"].shape[0] == n
        assert f["graph_bytes"].shape[0] == n
        if "labels" in f:
            assert f["labels"].shape[0] == n
        if "ids" in f:
            assert f["ids"].shape[0] == n

    return h5_path, returned_scaler, write_idx
