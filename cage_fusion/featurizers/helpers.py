# In cage_fusion/featurizers/helpers.py

from __future__ import annotations
import h5py
import numpy as np
import torch
from tqdm import tqdm
from cage_fusion.utils.logging_utils import logger
from typing import List, Dict, Any, Optional



def initialize_hdf5_file(
    h5_path: str,
    num_samples: int,
    seq_len: int,
    embed_dim: int,
    aux_dim: int,
    num_labels: int,
    chunk_size: int = 256,
    ids_enabled: bool = False,
    token_vocab_size: Optional[int] = None,  # NEW: choose smaller dtype for input_ids
    use_lzf_for_small: bool = True,  # NEW: faster than gzip if available
    emb_dtype: np.dtype = np.float32,  # NEW: store embeddings as float32
) -> None:
    """
    Initializes a resizable HDF5 file to store:
      - embedding [N, T, D] (float32 by default, no compression for speed)
      - input_ids [N, T] (uint16 if vocab <= 65535 else int32)
      - auxiliary_features [N, F]
      - optional labels [N, L]
      - original_indices [N]
      - smiles [N] (utf-8)
      - optional ids [N] (utf-8)
      - graph_bytes [N] (vlen uint8)

    Notes:
      * Use batch-aligned chunking for fast sequential reads.
      * Avoid compression on embeddings; it’s the biggest I/O and gzip is CPU-bound.
      * Strings can stay gzip (tiny) or lzf.
    """
    # Decide dtype for input_ids
    if token_vocab_size is not None and token_vocab_size <= 65535:
        ids_dtype = np.uint16
    else:
        ids_dtype = np.int32

    # Filters
    small_comp = "lzf" if use_lzf_for_small else "gzip"
    small_comp_opts = None if use_lzf_for_small else 4

    # Batch-aligned chunk shapes
    c = min(chunk_size, num_samples)
    emb_chunks = (c, seq_len, embed_dim)
    ids_chunks = (c, seq_len)
    aux_chunks = (c, aux_dim)
    lbl_chunks = (c, max(num_labels, 1))
    vec1d_chunks = (c,)

    # Open with latest format (nicer for concurrent reading later)
    with h5py.File(h5_path, "w", libver="latest") as f:
        # === Embeddings: biggest hot path ===
        # No compression for speed; store fp16 to cut I/O in half
        f.create_dataset(
            "embedding",
            shape=(num_samples, seq_len, embed_dim),
            maxshape=(None, seq_len, embed_dim),
            dtype=emb_dtype,
            chunks=emb_chunks,
            compression=None,
            shuffle=False,
            fletcher32=False,
        )
        f["embedding"].attrs["stored_dtype"] = str(np.dtype(emb_dtype))

        # === Token IDs ===
        f.create_dataset(
            "input_ids",
            shape=(num_samples, seq_len),
            maxshape=(None, seq_len),
            dtype=ids_dtype,
            chunks=ids_chunks,
            compression=None,  # small enough, keep fast
            shuffle=False,
            fletcher32=False,
        )
        f["input_ids"].attrs["stored_dtype"] = str(np.dtype(ids_dtype))
        if token_vocab_size is not None:
            f["input_ids"].attrs["token_vocab_size"] = int(token_vocab_size)

        # === Aux features (raw) ===
        f.create_dataset(
            "auxiliary_features",
            shape=(num_samples, aux_dim),
            maxshape=(None, aux_dim),
            dtype=np.float32,
            chunks=aux_chunks,
            compression=small_comp,
            compression_opts=small_comp_opts,
            shuffle=True,
            fletcher32=False,
        )

        # === Labels (optional) ===
        if num_labels > 0:
            f.create_dataset(
                "labels",
                shape=(num_samples, num_labels),
                maxshape=(None, num_labels),
                dtype=np.float32,
                chunks=lbl_chunks,
                compression=small_comp,
                compression_opts=small_comp_opts,
                shuffle=True,
                fletcher32=False,
            )

        # === Original indices ===
        f.create_dataset(
            "original_indices",
            shape=(num_samples,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=vec1d_chunks,
            compression=small_comp,
            compression_opts=small_comp_opts,
            shuffle=True,
            fletcher32=False,
        )

        # === Strings ===
        str_dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset(
            "smiles",
            shape=(num_samples,),
            maxshape=(None,),
            dtype=str_dt,
            chunks=vec1d_chunks,
            compression=small_comp,
            compression_opts=small_comp_opts,
            shuffle=True,
            fletcher32=False,
        )
        if ids_enabled:
            f.create_dataset(
                "ids",
                shape=(num_samples,),
                maxshape=(None,),
                dtype=str_dt,
                chunks=vec1d_chunks,
                compression=small_comp,
                compression_opts=small_comp_opts,
                shuffle=True,
                fletcher32=False,
            )

        # === Graphs as vlen bytes ===
        vlen_bytes = h5py.vlen_dtype(np.dtype("uint8"))
        f.create_dataset(
            "graph_bytes",
            shape=(num_samples,),
            maxshape=(None,),
            dtype=vlen_bytes,
            # h5py typically ignores compression for vlen; fine.
        )

    logger.info(
        "Initialized HDF5 at %s | emb=%s chunks=%s ids=%s aux=%s labels=%s strings=%s",
        h5_path,
        str(np.dtype(emb_dtype)),
        emb_chunks,
        str(np.dtype(ids_dtype)),
        aux_chunks,
        ("enabled" if num_labels > 0 else "disabled"),
        small_comp,
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
    Tokenizes SMILES and extracts embeddings using a pretrained transformer model.
    Returns:
      - input_ids: np.int32 [B, seq_len]
      - embeddings: np.float32 [B, seq_len, D]
    """
    inputs = tokenizer(
        smiles_batch,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    input_ids = inputs["input_ids"]

    # Safer than tokenizer.vocab_size: check against model's embedding table
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


def process_aux_feats(
    batch_df,
    desc_calc,
    scaler,
    fit_scaler: bool = True,
    clean_descriptors=lambda x: x,
):
    """
    Processes auxiliary features for a batch of molecules.
    Returns a Python list of np.float64 arrays (we cast to float32 at write).
    """
    if fit_scaler and scaler is None:
        raise ValueError("fit_scaler=True but no scaler was provided.")

    batch_aux = []
    for row in batch_df.itertuples(index=False):
        mol = row.mol
        desc = clean_descriptors(np.array(desc_calc.CalcDescriptors(mol)))
        batch_aux.append(desc)

    if fit_scaler and batch_aux:
        scaler.partial_fit(np.asarray(batch_aux))

    return batch_aux


def process_labels(batch_df, label_cols: List[str]) -> np.ndarray:
    """
    Processes labels for a batch of molecules.
    Robust to NaNs and awkward column names (uses DataFrame indexing).
    """
    if not label_cols:
        return np.empty((len(batch_df), 0), dtype=np.float32)

    arr = batch_df[label_cols].to_numpy(copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def process_graphs(batch_df, graph_featurizer) -> List[Dict[str, Any]]:
    """
    Processes graphs for a batch of molecules (Chemprop style dicts).
    """
    graph_feats = []
    for row in batch_df.itertuples(index=False):
        mol = row.mol
        graph_feats.append(graph_featurizer(mol))
    return graph_feats


def normalize_auxiliary_features(
    h5_path: str,
    scaler,
    aux_dim: int,
    batch_size: int = 512,
    name: str = "features",
) -> None:
    """
    Create/refresh 'auxiliary_features_normalized' from 'auxiliary_features'
    and always KEEP the raw dataset. Safe to call multiple times.
    """
    logger.info(f"Normalizing auxiliary features in {h5_path} ...")
    with h5py.File(h5_path, "a") as f:
        src_key = "auxiliary_features"
        dst_key = "auxiliary_features_normalized"
        if src_key not in f:
            raise KeyError(f"Missing dataset '{src_key}'")

        N = f[src_key].shape[0]

        # Recreate normalized dataset (refreshable)
        if dst_key in f:
            del f[dst_key]

        dset_norm = f.create_dataset(
            dst_key,
            shape=(N, aux_dim),
            maxshape=(N, aux_dim),
            dtype=np.float32,
            chunks=(min(batch_size, N), aux_dim),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        for i in tqdm(range(0, N, batch_size), desc=f"Normalizing {name}"):
            batch = f[src_key][i : i + batch_size]
            norm = scaler.transform(batch).astype(np.float32, copy=False)
            dset_norm[i : i + batch_size] = norm

        # Attach scaler metadata
        dset_norm.attrs["normalized"] = True
        dset_norm.attrs["source"] = src_key
        for k in ("mean_", "scale_", "var_"):
            if hasattr(scaler, k):
                dset_norm.attrs[f"scaler_{k}"] = getattr(scaler, k).astype(np.float32)
        if hasattr(scaler, "n_samples_seen_"):
            dset_norm.attrs["scaler_n_samples_seen_"] = int(scaler.n_samples_seen_)

        logger.info("Wrote 'auxiliary_features_normalized' to HDF5 file.")
