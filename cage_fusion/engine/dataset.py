# cage_fusion/engine/dataset.py

from __future__ import annotations

import os
import h5py
import torch
import pickle
import numpy as np
from typing import Optional
from collections import OrderedDict
from torch.utils.data import Dataset, get_worker_info
from cage_fusion.utils.logging_utils import logger


# ---------------------- RAM helpers ---------------------- #
def _available_ram_bytes() -> int:
    """Best-effort estimate of currently available RAM in bytes."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        sz = os.sysconf("SC_PAGE_SIZE")
        return int(pages * sz)
    except Exception:
        return 8 * 1024**3  # 8 GB fallback


def _log_rss(msg: str) -> None:
    try:
        import psutil

        rss_gb = psutil.Process().memory_info().rss / (1024**3)
        logger.info("%s: RSS ≈ %.2f GB", msg, rss_gb)
    except Exception:
        pass


def _sizeof_string_list(xs) -> int:
    """Rough size estimate for list of strings in bytes (content + tiny overhead)."""
    if xs is None:
        return 0
    total = 0
    for s in xs:
        if s is None:
            continue
        try:
            total += len(s)
        except Exception:
            pass
    total += 32 * len(xs)  # tiny per-entry overhead
    return total


# ---------------------- Dataset ---------------------- #
class CageFusionStreamingDataset(Dataset):
    """
    Streams tensors from HDF5 with aggressive caching of small arrays and optional
    graph caching. **Embeddings are always read directly from the HDF5 file** (no LRU).
    Graphs may be cached on ALL workers (subject to each worker's RAM budget).

    Global constraint (per worker): combined caches (graphs + small arrays)
    will not exceed ~overall_max_ram_fraction of *currently available* RAM divided by
    the *user-provided* total_num_workers. (We do NOT guess the worker count.)
    """

    def __init__(
        self,
        h5_path: str,
        *,
        total_num_workers: int,  # REQUIRED: used for ALL budgeting
        worker_id: int | None = None,  # if None we'll use get_worker_info().id or 0
        tokenizer_pad_id: int = 0,
        prefer_normalized_aux: bool = True,
        return_ids: bool = True,
        graph_cache: str = "auto",  # "auto" | "bytes" | "objects" | "none"
        max_ram_fraction: float = 0.5,  # used only to choose a graph cache *mode*
        sample_for_estimate: int = 2000,
        # cache toggles
        cache_input_ids: bool = True,
        cache_aux: bool = True,
        cache_labels: bool = True,
        cache_strings: bool = True,
        # embedding dtype handling
        emb_cache_store_dtype: np.dtype = np.float32,  # dtype used when reading from HDF5
        return_emb_dtype: torch.dtype = torch.float32,  # tensor dtype returned to model
        # HDF5 read cache tuning
        rdcc_nbytes: int = 128 << 20,
        rdcc_nslots: int = 1_000_003,
        # overall cap (of currently available RAM)
        overall_max_ram_fraction: float = 0.8,  # e.g., 0.8 => leave ~20% to the system
        # duplication control (now default False so ALL workers may cache graphs)
        single_worker_graph_cache: bool = False,
    ):
        # --- core config ---
        self.h5_path = h5_path
        self.pad_token_id = tokenizer_pad_id
        self.prefer_normalized_aux = prefer_normalized_aux
        self.return_ids = return_ids
        self.graph_cache = graph_cache
        self.max_ram_fraction = float(max_ram_fraction)  # only for mode choice
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

        # --- worker identity (for logging/graph caching) ---
        wi = get_worker_info()
        detected_worker_id = wi.id if wi is not None else 0
        self._worker_id = detected_worker_id if worker_id is None else int(worker_id)

        # --- IMPORTANT: use ONLY user-provided total_num_workers for budgeting ---
        self._num_workers = int(max(1, total_num_workers))

        # handles and caches
        self._h5_handle: Optional[h5py.File] = None
        self._graph_bytes_cache = None  # list[np.uint8]
        self._graph_obj_cache = None  # list[object]

        # small arrays full-cache (numpy or lists)
        self._ids_int = None
        self._aux = None
        self._labels = None
        self._smiles = None
        self._orig_idx = None
        self._ids_str = None

        # embedding shape info (no LRU now)
        self._seq_len = 0
        self._emb_dim = 0

        # ---- Build caches respecting a per-worker cap ----
        avail_bytes_now = _available_ram_bytes()
        total_budget = int(self.overall_max_ram_fraction * avail_bytes_now)
        per_worker_budget = max(0, total_budget // self._num_workers)

        used_by_graphs = 0
        used_by_small = 0

        with h5py.File(h5_path, "r") as f:
            # basic checks
            required = [
                "graph_bytes",
                "embedding",
                "auxiliary_features",
                "input_ids",
                "smiles",
                "original_indices",
            ]
            missing = [k for k in required if k not in f]
            if missing:
                raise KeyError(f"Missing required datasets in {h5_path}: {missing}")

            self.length = int(f["embedding"].shape[0])
            self.has_labels = "labels" in f
            self.has_aux_norm = "auxiliary_features_normalized" in f
            self.has_ids = self.return_ids and ("ids" in f)

            # embedding shape
            emb_dset = f["embedding"]
            self._emb_shape = emb_dset.shape  # (N, T, D)
            self._seq_len = int(self._emb_shape[1])
            self._emb_dim = int(self._emb_shape[2])

            # ---------- Graph cache (counts against per-worker budget) ----------
            chosen = (
                self._decide_cache_mode(f)
                if (self.graph_cache == "auto")
                else self.graph_cache
            )
            # NEW: allow caching on ALL workers by default (no forced 'none' for non-zero workers)
            if self.single_worker_graph_cache and self._worker_id != 0:
                chosen = "none"

            if chosen == "objects":
                gb_arr = f["graph_bytes"][:]  # bulk read once
                total_bytes = int(sum(len(b) for b in gb_arr))  # raw bytes
                estimated_needed = int(total_bytes * 2.5)  # object overhead
                if estimated_needed <= max(
                    0, per_worker_budget - used_by_graphs - used_by_small
                ):
                    self._graph_obj_cache = [pickle.loads(bytes(b)) for b in gb_arr]
                    self._graph_bytes_cache = None
                    used_by_graphs += estimated_needed
                    _log_rss(f"[wk{self._worker_id}] After caching graph objects")
                else:
                    if total_bytes <= max(
                        0, per_worker_budget - used_by_graphs - used_by_small
                    ):
                        self._graph_bytes_cache = list(gb_arr)
                        self._graph_obj_cache = None
                        used_by_graphs += total_bytes
                        _log_rss(
                            f"[wk{self._worker_id}] After caching graph bytes (fallback)"
                        )
                        chosen = "bytes"
                    else:
                        self._graph_bytes_cache = None
                        self._graph_obj_cache = None
                        chosen = "none"

            elif chosen == "bytes":
                gb_arr = f["graph_bytes"][:]
                total_bytes = int(sum(len(b) for b in gb_arr))
                if total_bytes <= max(
                    0, per_worker_budget - used_by_graphs - used_by_small
                ):
                    self._graph_bytes_cache = list(gb_arr)
                    self._graph_obj_cache = None
                    used_by_graphs += total_bytes
                    _log_rss(f"[wk{self._worker_id}] After caching graph bytes")
                else:
                    self._graph_bytes_cache = None
                    self._graph_obj_cache = None
                    chosen = "none"

            elif chosen == "none":
                self._graph_bytes_cache = None
                self._graph_obj_cache = None
            else:
                logger.warning(
                    "Unknown graph_cache='%s'; falling back to 'none'", chosen
                )
                self._graph_bytes_cache = None
                self._graph_obj_cache = None
                chosen = "none"
            self.graph_cache = chosen

            # ---------- Small arrays full-cache (per-worker budget) ----------
            # input_ids
            if self.cache_input_ids:
                ids_arr = f["input_ids"][:]
                nbytes = ids_arr.nbytes
                if used_by_graphs + used_by_small + nbytes <= per_worker_budget:
                    self._ids_int = ids_arr
                    used_by_small += nbytes
                else:
                    self._ids_int = None  # fallback: read on demand

            # aux (normalized preferred)
            if self.cache_aux:
                aux_key = (
                    "auxiliary_features_normalized"
                    if (self.prefer_normalized_aux and self.has_aux_norm)
                    else "auxiliary_features"
                )
                aux_arr = f[aux_key][:].astype(np.float32, copy=False)
                nbytes = aux_arr.nbytes
                if used_by_graphs + used_by_small + nbytes <= per_worker_budget:
                    self._aux = aux_arr
                    used_by_small += nbytes
                else:
                    self._aux = None

            # labels
            if self.cache_labels and self.has_labels:
                lbl_arr = f["labels"][:].astype(np.float32, copy=False)
                nbytes = lbl_arr.nbytes
                if used_by_graphs + used_by_small + nbytes <= per_worker_budget:
                    self._labels = lbl_arr
                    used_by_small += nbytes
                else:
                    self._labels = None

            # strings & original indices
            if self.cache_strings:
                smiles_list = [
                    s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s)
                    for s in f["smiles"][:]
                ]
                smiles_bytes = _sizeof_string_list(smiles_list)

                orig_idx_arr = f["original_indices"][:].astype(np.int64, copy=False)
                orig_bytes = orig_idx_arr.nbytes

                ids_str_list = None
                ids_bytes = 0
                if self.has_ids:
                    ids_str_list = [
                        (
                            s.decode("utf-8")
                            if isinstance(s, (bytes, np.bytes_))
                            else str(s)
                        )
                        for s in f["ids"][:]
                    ]
                    ids_bytes = _sizeof_string_list(ids_str_list)

                strings_total = smiles_bytes + orig_bytes + ids_bytes
                if used_by_graphs + used_by_small + strings_total <= per_worker_budget:
                    self._smiles = smiles_list
                    self._orig_idx = orig_idx_arr
                    self._ids_str = ids_str_list if self.has_ids else None
                    used_by_small += strings_total
                else:
                    # keep original_indices only if we can afford them
                    if used_by_graphs + used_by_small + orig_bytes <= per_worker_budget:
                        self._orig_idx = orig_idx_arr
                        used_by_small += orig_bytes
                        self._smiles = None
                        self._ids_str = None
                    else:
                        self._smiles = None
                        self._orig_idx = None
                        self._ids_str = None

            logger.info(
                "[wk%d/%d] H5 ready (disk-only embeddings): N=%d | labels=%s | aux_norm=%s | "
                "ids=%s | graph_cache=%s | emb_shape=%s | return_emb=%s",
                self._worker_id,
                self._num_workers,
                self.length,
                self.has_labels,
                self.has_aux_norm,
                self.has_ids,
                self.graph_cache,
                str(self._emb_shape),
                str(self.return_emb_dtype).split(".")[-1],
            )
            logger.info(
                "[wk%d/%d] Budgets -> per-worker cap: %.2f GB (%.0f%% of avail / %d workers) | "
                "used_graphs: %.2f GB | used_small: %.2f GB | emb_mode: disk-only",
                self._worker_id,
                self._num_workers,
                per_worker_budget / (1024**3),
                self.overall_max_ram_fraction * 100.0,
                self._num_workers,
                used_by_graphs / (1024**3),
                used_by_small / (1024**3),
            )

    # ----- Graph cache mode decision (only to pick objects/bytes/none) -----
    def _decide_cache_mode(self, f: h5py.File) -> str:
        N = int(f["graph_bytes"].shape[0])
        if N == 0:
            return "none"

        k = min(N, max(1, self.sample_for_estimate))
        if k == N:
            sample_idxs = range(N)
        else:
            step = max(1, N // k)
            sample_idxs = range(0, N, step)

        total_sample_bytes = 0
        cnt = 0
        for i in sample_idxs:
            gb = f["graph_bytes"][i]
            total_sample_bytes += int(len(gb))
            cnt += 1
            if cnt >= k:
                break

        avg_bytes = total_sample_bytes / max(1, cnt)
        est_total_bytes = avg_bytes * N
        est_objects_bytes = est_total_bytes * 2.5  # rough Python object overhead

        # NOTE: This picks a *mode* only; actual allocation is capped later per worker.
        avail = _available_ram_bytes()
        budget = avail * self.max_ram_fraction
        logger.info(
            "Graph mem estimate: avg=%.1f KB, total≈%.2f GB (bytes), objects≈%.2f GB; "
            "avail≈%.2f GB; mode_budget≈%.2f GB",
            avg_bytes / 1024.0,
            est_total_bytes / (1024**3),
            est_objects_bytes / (1024**3),
            avail / (1024**3),
            budget / (1024**3),
        )
        if est_objects_bytes <= budget:
            return "objects"
        if est_total_bytes <= budget:
            return "bytes"
        return "none"

    # ---------------------- HDF5 handle ---------------------- #
    def _get_h5(self) -> h5py.File:
        if self._h5_handle is None:
            self._h5_handle = h5py.File(
                self.h5_path,
                "r",
                rdcc_nbytes=self.rdcc_nbytes,
                rdcc_nslots=self.rdcc_nslots,
            )
        return self._h5_handle

    # ---------------------- Public API ---------------------- #
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        h5 = self._get_h5()

        # Graph
        if self._graph_obj_cache is not None:
            graph = self._graph_obj_cache[idx]
        elif self._graph_bytes_cache is not None:
            gb = self._graph_bytes_cache[idx]
            graph = pickle.loads(bytes(gb))
        else:
            gb = h5["graph_bytes"][idx]
            graph = pickle.loads(bytes(gb))

        # Embedding: ALWAYS read directly from file (disk-only, no LRU)
        emb_np = np.ascontiguousarray(h5["embedding"][idx]).astype(
            self.emb_cache_store_dtype, copy=False
        )

        # Convert to tensor with requested return dtype
        if self.return_emb_dtype == torch.float32 and emb_np.dtype == np.float32:
            token_embs = torch.from_numpy(emb_np)
        elif self.return_emb_dtype == torch.float16 and emb_np.dtype == np.float16:
            token_embs = torch.from_numpy(emb_np)
        else:
            token_embs = torch.as_tensor(emb_np).to(self.return_emb_dtype)

        # Token IDs
        if self._ids_int is not None:
            token_ids = torch.from_numpy(self._ids_int[idx]).long()
        else:
            token_ids = torch.tensor(h5["input_ids"][idx], dtype=torch.long)

        # Aux
        if self._aux is not None:
            aux = torch.from_numpy(self._aux[idx])
        else:
            if self.prefer_normalized_aux and ("auxiliary_features_normalized" in h5):
                aux = torch.tensor(
                    h5["auxiliary_features_normalized"][idx], dtype=torch.float32
                )
            else:
                aux = torch.tensor(h5["auxiliary_features"][idx], dtype=torch.float32)

        # Labels
        if self.has_labels:
            if self._labels is not None:
                labels = torch.from_numpy(self._labels[idx])
            else:
                labels = torch.tensor(h5["labels"][idx], dtype=torch.float32)
        else:
            labels = torch.empty(0, dtype=torch.float32)

        # Strings / indices
        if self._smiles is not None:
            smiles = self._smiles[idx]
        else:
            raw_smiles = h5["smiles"][idx]
            smiles = (
                raw_smiles.decode("utf-8")
                if isinstance(raw_smiles, (bytes, np.bytes_))
                else str(raw_smiles)
            )

        if self._orig_idx is not None:
            original_index = int(self._orig_idx[idx])
        else:
            original_index = int(h5["original_indices"][idx])

        id_str = None
        if self.has_ids:
            if self._ids_str is not None:
                id_str = self._ids_str[idx]
            else:
                raw_id = h5["ids"][idx]
                id_str = (
                    raw_id.decode("utf-8")
                    if isinstance(raw_id, (bytes, np.bytes_))
                    else str(raw_id)
                )

        return graph, token_embs, aux, labels, token_ids, smiles, original_index, id_str

    def __del__(self):
        if self._h5_handle is not None:
            try:
                self._h5_handle.close()
            except Exception:
                pass
            finally:
                self._h5_handle = None


# Optional: tiny LRU for graph-uncached scenarios; generally not needed now.
class MiniBatchCacheDataset(Dataset):
    def __init__(self, dataset: Dataset, cache_size: int = 1024):
        self.dataset = dataset
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def __len__(self) -> int:
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
