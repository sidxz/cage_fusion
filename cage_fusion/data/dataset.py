"""
Streaming HDF5 dataset for cage_fusion.

``CageFusionStreamingDataset`` reads molecular features from HDF5 files
produced by ``featurization.molecular_featurizer`` with aggressive RAM-aware
caching of small arrays and optional graph caching.  Token embeddings are
always read directly from disk (never cached in RAM).
"""

# Logic identical to the original engine/dataset.py;
# import updated from cage_fusion.utils.logging_utils → logging

from __future__ import annotations

import logging
import os
import pickle
from collections import OrderedDict
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

logger = logging.getLogger("cagefusion")


def _available_ram_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return 8 * 1024 ** 3


def _log_rss(msg: str) -> None:
    try:
        import psutil
        rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        logger.info("%s: RSS ≈ %.2f GB", msg, rss_gb)
    except Exception:
        pass


def _sizeof_string_list(xs) -> int:
    if xs is None:
        return 0
    return sum(len(s) for s in xs if s is not None) + 32 * len(xs)


class CageFusionStreamingDataset(Dataset):
    """
    Memory-efficient streaming dataset backed by HDF5.

    Token embeddings are always loaded from disk.  Graphs, token IDs,
    auxiliary features, labels, and string fields are cached in RAM
    subject to a per-worker budget derived from available system memory.

    Parameters
    ----------
    h5_path:
        Path to the HDF5 file produced by ``featurize_and_save_streaming``.
    total_num_workers:
        **Required** – total number of DataLoader workers.  Used for RAM
        budget partitioning across processes.
    """

    def __init__(
        self,
        h5_path: str,
        *,
        total_num_workers: int,
        worker_id: int | None = None,
        tokenizer_pad_id: int = 0,
        prefer_normalized_aux: bool = True,
        return_ids: bool = True,
        graph_cache: str = "auto",
        max_ram_fraction: float = 0.5,
        sample_for_estimate: int = 2000,
        cache_input_ids: bool = True,
        cache_aux: bool = True,
        cache_labels: bool = True,
        cache_strings: bool = True,
        emb_cache_store_dtype: np.dtype = np.float32,
        return_emb_dtype: torch.dtype = torch.float32,
        rdcc_nbytes: int = 128 << 20,
        rdcc_nslots: int = 1_000_003,
        overall_max_ram_fraction: float = 0.8,
        single_worker_graph_cache: bool = False,
    ):
        self.h5_path = h5_path
        self.pad_token_id = tokenizer_pad_id
        self.prefer_normalized_aux = prefer_normalized_aux
        self.return_ids = return_ids
        self.graph_cache = graph_cache
        self.max_ram_fraction = float(max_ram_fraction)
        self.sample_for_estimate = int(sample_for_estimate)
        self.cache_input_ids = bool(cache_input_ids)
        self.cache_aux = bool(cache_aux)
        self.cache_labels = bool(cache_labels)
        self.cache_strings = bool(cache_strings)
        self.emb_cache_store_dtype = np.dtype(emb_cache_store_dtype)
        self.return_emb_dtype = return_emb_dtype
        self.rdcc_nbytes = int(max(8 << 20, rdcc_nbytes))
        self.rdcc_nslots = int(max(10_007, rdcc_nslots))
        self.overall_max_ram_fraction = float(overall_max_ram_fraction)
        self.single_worker_graph_cache = bool(single_worker_graph_cache)

        wi = get_worker_info()
        detected_id = wi.id if wi is not None else 0
        self._worker_id = detected_id if worker_id is None else int(worker_id)
        self._num_workers = int(max(1, total_num_workers))

        self._h5_handle: Optional[h5py.File] = None
        self._graph_bytes_cache = None
        self._graph_obj_cache = None
        self._ids_int = None
        self._aux = None
        self._labels = None
        self._smiles = None
        self._orig_idx = None
        self._ids_str = None
        self._seq_len = 0
        self._emb_dim = 0

        avail = _available_ram_bytes()
        total_budget = int(self.overall_max_ram_fraction * avail)
        per_worker_budget = max(0, total_budget // self._num_workers)
        used_graphs = 0
        used_small = 0

        with h5py.File(h5_path, "r") as f:
            required = ["graph_bytes", "embedding", "auxiliary_features",
                        "input_ids", "smiles", "original_indices"]
            missing = [k for k in required if k not in f]
            if missing:
                raise KeyError(f"Missing required datasets in {h5_path}: {missing}")

            self.length = int(f["embedding"].shape[0])
            self.has_labels = "labels" in f
            self.has_aux_norm = "auxiliary_features_normalized" in f
            self.has_ids = self.return_ids and ("ids" in f)

            self._emb_shape = f["embedding"].shape
            self._seq_len = int(self._emb_shape[1])
            self._emb_dim = int(self._emb_shape[2])

            # Graph cache decision
            chosen = self._decide_cache_mode(f) if self.graph_cache == "auto" else self.graph_cache
            if self.single_worker_graph_cache and self._worker_id != 0:
                chosen = "none"

            if chosen == "objects":
                gb_arr = f["graph_bytes"][:]
                total_bytes = sum(len(b) for b in gb_arr)
                est_needed = int(total_bytes * 2.5)
                if est_needed <= per_worker_budget - used_graphs - used_small:
                    self._graph_obj_cache = [pickle.loads(bytes(b)) for b in gb_arr]
                    used_graphs += est_needed
                    chosen = "objects"
                elif total_bytes <= per_worker_budget - used_graphs - used_small:
                    self._graph_bytes_cache = list(gb_arr)
                    used_graphs += total_bytes
                    chosen = "bytes"
                else:
                    chosen = "none"
            elif chosen == "bytes":
                gb_arr = f["graph_bytes"][:]
                total_bytes = sum(len(b) for b in gb_arr)
                if total_bytes <= per_worker_budget - used_graphs - used_small:
                    self._graph_bytes_cache = list(gb_arr)
                    used_graphs += total_bytes
                else:
                    chosen = "none"

            self.graph_cache = chosen

            # Small array caches
            if self.cache_input_ids:
                ids_arr = f["input_ids"][:]
                if used_graphs + used_small + ids_arr.nbytes <= per_worker_budget:
                    self._ids_int = ids_arr
                    used_small += ids_arr.nbytes

            if self.cache_aux:
                aux_key = "auxiliary_features_normalized" if (self.prefer_normalized_aux and self.has_aux_norm) else "auxiliary_features"
                aux_arr = f[aux_key][:].astype(np.float32, copy=False)
                if used_graphs + used_small + aux_arr.nbytes <= per_worker_budget:
                    self._aux = aux_arr
                    used_small += aux_arr.nbytes

            if self.cache_labels and self.has_labels:
                lbl_arr = f["labels"][:].astype(np.float32, copy=False)
                if used_graphs + used_small + lbl_arr.nbytes <= per_worker_budget:
                    self._labels = lbl_arr
                    used_small += lbl_arr.nbytes

            if self.cache_strings:
                smiles_list = [
                    s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s)
                    for s in f["smiles"][:]
                ]
                orig_idx_arr = f["original_indices"][:].astype(np.int64, copy=False)
                ids_str_list = None
                if self.has_ids:
                    ids_str_list = [
                        s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s)
                        for s in f["ids"][:]
                    ]
                strings_total = _sizeof_string_list(smiles_list) + orig_idx_arr.nbytes + _sizeof_string_list(ids_str_list)
                if used_graphs + used_small + strings_total <= per_worker_budget:
                    self._smiles = smiles_list
                    self._orig_idx = orig_idx_arr
                    self._ids_str = ids_str_list
                    used_small += strings_total
                elif used_graphs + used_small + orig_idx_arr.nbytes <= per_worker_budget:
                    self._orig_idx = orig_idx_arr
                    used_small += orig_idx_arr.nbytes

            logger.info(
                "[wk%d/%d] Dataset ready: N=%d graph_cache=%s emb_shape=%s",
                self._worker_id, self._num_workers, self.length,
                self.graph_cache, str(self._emb_shape),
            )

    def _decide_cache_mode(self, f: h5py.File) -> str:
        N = int(f["graph_bytes"].shape[0])
        if N == 0:
            return "none"
        k = min(N, max(1, self.sample_for_estimate))
        step = max(1, N // k)
        total_bytes = sum(len(f["graph_bytes"][i]) for i in range(0, N, step))
        avg = total_bytes / max(1, N // step)
        est_bytes = avg * N
        avail = _available_ram_bytes()
        budget = avail * self.max_ram_fraction
        if est_bytes * 2.5 <= budget:
            return "objects"
        if est_bytes <= budget:
            return "bytes"
        return "none"

    def _get_h5(self) -> h5py.File:
        if self._h5_handle is None:
            self._h5_handle = h5py.File(
                self.h5_path, "r",
                rdcc_nbytes=self.rdcc_nbytes, rdcc_nslots=self.rdcc_nslots,
            )
        return self._h5_handle

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        h5 = self._get_h5()

        # Graph
        if self._graph_obj_cache is not None:
            graph = self._graph_obj_cache[idx]
        elif self._graph_bytes_cache is not None:
            graph = pickle.loads(bytes(self._graph_bytes_cache[idx]))
        else:
            graph = pickle.loads(bytes(h5["graph_bytes"][idx]))

        # Embedding (always from disk)
        emb_np = np.ascontiguousarray(h5["embedding"][idx]).astype(self.emb_cache_store_dtype, copy=False)
        if self.return_emb_dtype == torch.float32 and emb_np.dtype == np.float32:
            token_embs = torch.from_numpy(emb_np)
        elif self.return_emb_dtype == torch.float16 and emb_np.dtype == np.float16:
            token_embs = torch.from_numpy(emb_np)
        else:
            token_embs = torch.as_tensor(emb_np).to(self.return_emb_dtype)

        # Token IDs
        token_ids = (
            torch.from_numpy(self._ids_int[idx]).long()
            if self._ids_int is not None
            else torch.tensor(h5["input_ids"][idx], dtype=torch.long)
        )

        # Aux
        if self._aux is not None:
            aux = torch.from_numpy(self._aux[idx])
        elif self.prefer_normalized_aux and "auxiliary_features_normalized" in h5:
            aux = torch.tensor(h5["auxiliary_features_normalized"][idx], dtype=torch.float32)
        else:
            aux = torch.tensor(h5["auxiliary_features"][idx], dtype=torch.float32)

        # Labels
        if self.has_labels:
            labels = (
                torch.from_numpy(self._labels[idx])
                if self._labels is not None
                else torch.tensor(h5["labels"][idx], dtype=torch.float32)
            )
        else:
            labels = torch.empty(0, dtype=torch.float32)

        # SMILES
        if self._smiles is not None:
            smiles = self._smiles[idx]
        else:
            raw = h5["smiles"][idx]
            smiles = raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else str(raw)

        # Original index
        original_index = int(self._orig_idx[idx]) if self._orig_idx is not None else int(h5["original_indices"][idx])

        # Optional molecule ID
        id_str = None
        if self.has_ids:
            if self._ids_str is not None:
                id_str = self._ids_str[idx]
            else:
                raw = h5["ids"][idx]
                id_str = raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else str(raw)

        return graph, token_embs, aux, labels, token_ids, smiles, original_index, id_str

    def __del__(self):
        if self._h5_handle is not None:
            try:
                self._h5_handle.close()
            except Exception:
                pass
            finally:
                self._h5_handle = None


class MiniBatchCacheDataset(Dataset):
    """Thin LRU wrapper for graph-uncached scenarios."""

    def __init__(self, dataset: Dataset, cache_size: int = 1024):
        self.dataset = dataset
        self.cache: OrderedDict = OrderedDict()
        self.cache_size = cache_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        item = self.dataset[idx]
        self.cache[idx] = item
        self.cache.move_to_end(idx)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return item
