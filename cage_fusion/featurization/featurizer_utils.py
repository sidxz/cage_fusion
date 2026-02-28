"""
Low-level featurisation helpers (HDF5 I/O, batch tokenisation, aux features).

These are internal utilities consumed by ``molecular_featurizer.py``.
"""

# Identical logic to the original featurizers/helpers.py
# with the import path updated to cage_fusion.utils.logging

from __future__ import annotations

import logging
import h5py
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Optional

logger = logging.getLogger("cagefusion")


def initialize_hdf5_file(
    h5_path: str,
    num_samples: int,
    seq_len: int,
    embed_dim: int,
    aux_dim: int,
    num_labels: int,
    chunk_size: int = 256,
    ids_enabled: bool = False,
    token_vocab_size: Optional[int] = None,
    use_lzf_for_small: bool = True,
    emb_dtype: np.dtype = np.float32,
) -> None:
    ids_dtype = np.uint16 if (token_vocab_size is not None and token_vocab_size <= 65535) else np.int32
    small_comp = "lzf" if use_lzf_for_small else "gzip"
    small_comp_opts = None if use_lzf_for_small else 4

    c = min(chunk_size, num_samples)

    with h5py.File(h5_path, "w", libver="latest") as f:
        f.create_dataset(
            "embedding",
            shape=(num_samples, seq_len, embed_dim), maxshape=(None, seq_len, embed_dim),
            dtype=emb_dtype, chunks=(c, seq_len, embed_dim),
            compression=None, shuffle=False, fletcher32=False,
        )
        f["embedding"].attrs["stored_dtype"] = str(np.dtype(emb_dtype))

        f.create_dataset(
            "input_ids",
            shape=(num_samples, seq_len), maxshape=(None, seq_len),
            dtype=ids_dtype, chunks=(c, seq_len),
            compression=None, shuffle=False, fletcher32=False,
        )
        f["input_ids"].attrs["stored_dtype"] = str(np.dtype(ids_dtype))
        if token_vocab_size is not None:
            f["input_ids"].attrs["token_vocab_size"] = int(token_vocab_size)

        f.create_dataset(
            "auxiliary_features",
            shape=(num_samples, aux_dim), maxshape=(None, aux_dim),
            dtype=np.float32, chunks=(c, aux_dim),
            compression=small_comp, compression_opts=small_comp_opts,
            shuffle=True, fletcher32=False,
        )

        if num_labels > 0:
            f.create_dataset(
                "labels",
                shape=(num_samples, num_labels), maxshape=(None, num_labels),
                dtype=np.float32, chunks=(c, max(num_labels, 1)),
                compression=small_comp, compression_opts=small_comp_opts,
                shuffle=True, fletcher32=False,
            )

        f.create_dataset(
            "original_indices",
            shape=(num_samples,), maxshape=(None,),
            dtype=np.int64, chunks=(c,),
            compression=small_comp, compression_opts=small_comp_opts,
            shuffle=True, fletcher32=False,
        )

        str_dt = h5py.string_dtype(encoding="utf-8")
        for name in (["smiles"] + (["ids"] if ids_enabled else [])):
            f.create_dataset(
                name, shape=(num_samples,), maxshape=(None,),
                dtype=str_dt, chunks=(c,),
                compression=small_comp, compression_opts=small_comp_opts,
                shuffle=True, fletcher32=False,
            )

        vlen_bytes = h5py.vlen_dtype(np.dtype("uint8"))
        f.create_dataset("graph_bytes", shape=(num_samples,), maxshape=(None,), dtype=vlen_bytes)

    logger.info(
        "Initialised HDF5 at %s | N=%d emb=%s ids=%s aux_dim=%d labels=%d",
        h5_path, num_samples, str(np.dtype(emb_dtype)), str(np.dtype(ids_dtype)),
        aux_dim, num_labels,
    )


def featurize_batch(
    tokenizer,
    model,
    smiles_batch: List[str],
    seq_len: int,
    device: torch.device,
    _vocab_size_unused: int,
):
    """
    Tokenise SMILES and extract token embeddings.

    Returns
    -------
    input_ids : np.int32  [B, seq_len]
    embeddings : np.float32  [B, seq_len, D]
    """
    inputs = tokenizer(
        smiles_batch,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    input_ids = inputs["input_ids"]
    num_embeddings = model.get_input_embeddings().num_embeddings
    if (input_ids >= num_embeddings).any() or (input_ids < 0).any():
        raise ValueError("Token IDs out of range for model embeddings.")

    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model(**inputs)
        embeddings = output.last_hidden_state
    del output, inputs

    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        raise ValueError("Embeddings contain NaN or Inf")

    return (
        input_ids.cpu().numpy().astype(np.int32, copy=False),
        embeddings.cpu().numpy().astype(np.float32, copy=False),
    )


def process_aux_feats(batch_df, desc_calc, scaler, fit_scaler: bool = True, clean_descriptors=lambda x: x):
    if fit_scaler and scaler is None:
        raise ValueError("fit_scaler=True but no scaler was provided.")
    batch_aux = []
    for row in batch_df.itertuples(index=False):
        desc = clean_descriptors(np.array(desc_calc.CalcDescriptors(row.mol)))
        batch_aux.append(desc)
    if fit_scaler and batch_aux:
        scaler.partial_fit(np.asarray(batch_aux))
    return batch_aux


def process_labels(batch_df, label_cols: List[str]) -> np.ndarray:
    if not label_cols:
        return np.empty((len(batch_df), 0), dtype=np.float32)
    arr = batch_df[label_cols].to_numpy(copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32, copy=False)


def process_graphs(batch_df, graph_featurizer) -> List[Dict[str, Any]]:
    return [graph_featurizer(row.mol) for row in batch_df.itertuples(index=False)]


def normalize_auxiliary_features(
    h5_path: str, scaler, aux_dim: int, batch_size: int = 512, name: str = "features"
) -> None:
    logger.info("Normalising auxiliary features in %s ...", h5_path)
    with h5py.File(h5_path, "a") as f:
        if "auxiliary_features" not in f:
            raise KeyError("Missing dataset 'auxiliary_features'")
        N = f["auxiliary_features"].shape[0]
        dst = "auxiliary_features_normalized"
        if dst in f:
            del f[dst]
        dset = f.create_dataset(
            dst, shape=(N, aux_dim), maxshape=(N, aux_dim), dtype=np.float32,
            chunks=(min(batch_size, N), aux_dim), compression="gzip",
            compression_opts=4, shuffle=True,
        )
        for i in tqdm(range(0, N, batch_size), desc=f"Normalising {name}"):
            batch = f["auxiliary_features"][i : i + batch_size]
            dset[i : i + batch_size] = scaler.transform(batch).astype(np.float32, copy=False)
        dset.attrs["normalized"] = True
        for k in ("mean_", "scale_", "var_"):
            if hasattr(scaler, k):
                dset.attrs[f"scaler_{k}"] = getattr(scaler, k).astype(np.float32)
    logger.info("Wrote 'auxiliary_features_normalized' to %s", h5_path)
