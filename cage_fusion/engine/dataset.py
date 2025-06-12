import h5py
import joblib
import torch
import threading
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset
from cage_fusion.utils.logging_utils import logger

# Thread-local cache for HDF5 handles (per worker in multiprocessing)
_worker_cache = threading.local()


class CageFusionStreamingDataset(Dataset):
    """
    Dataset for streaming token embeddings, auxiliary features, labels, and molecular graphs
    from disk using multiprocessing-safe HDF5 handles.
    """

    def __init__(self, h5_path: str, graph_path: str, tokenizer_pad_id: int = 0):
        self.h5_path = h5_path
        self.pad_token_id = tokenizer_pad_id

        # Load precomputed graph features
        try:
            self.graphs = joblib.load(graph_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load graph features from {graph_path}: {e}")

        # Validate presence of key datasets in HDF5
        with h5py.File(h5_path, "r") as f:
            if "input_ids" not in f or "labels" not in f:
                raise KeyError(
                    f"Missing required datasets ('input_ids' or 'labels') in: {h5_path}"
                )
            self.length = f["labels"].shape[0]

        if self.length != len(self.graphs):
            logger.error(
                "Mismatch between HDF5 samples (%d) and graph entries (%d).",
                self.length,
                len(self.graphs),
            )
            raise ValueError(
                "Mismatch in sample counts between HDF5 and graph features."
            )

    def __len__(self) -> int:
        return self.length

    def _get_h5_file_handle(self):
        """Returns a per-worker cached HDF5 file handle."""
        if not hasattr(_worker_cache, "h5_file"):
            _worker_cache.h5_file = h5py.File(self.h5_path, "r")
        return _worker_cache.h5_file

    def __getitem__(self, idx: int) -> tuple:
        h5 = self._get_h5_file_handle()

        return (
            self.graphs[idx],
            torch.tensor(h5["embedding"][idx], dtype=torch.float32),
            torch.tensor(h5["auxiliary_features_normalized"][idx], dtype=torch.float32),
            torch.tensor(h5["labels"][idx], dtype=torch.float32),
            torch.tensor(h5["input_ids"][idx], dtype=torch.long),
        )

    def __del__(self):
        """Ensure HDF5 handle is closed when dataset is garbage collected."""
        if hasattr(_worker_cache, "h5_file"):
            try:
                _worker_cache.h5_file.close()
            except Exception:
                logger.warning("Failed to close HDF5 file cleanly.")
            finally:
                del _worker_cache.h5_file


class MiniBatchCacheDataset(Dataset):
    """
    Dataset wrapper that implements an LRU cache for recently accessed samples to reduce I/O.

    This is particularly useful for scenarios where the same indices are repeatedly accessed
    across different epochs or batches (e.g., in evaluation or shuffling with replacement).
    """

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

        # Maintain LRU cache by popping oldest entry if over capacity
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return item
