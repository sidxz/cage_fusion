import h5py
import joblib
import torch
import threading
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset
from cage_fusion.utils.logging_utils import logger

# Thread-local cache to manage HDF5 handles in multiprocessing workers
_worker_cache = threading.local()

class CageFusionStreamingDataset(Dataset):
    """
    Streams embeddings, auxiliary features, and labels from HDF5 along with graph features.

    This dataset is multiprocessing-safe by assigning a dedicated HDF5 handle to each worker.
    """
    def __init__(self, h5_path: str, graph_path: str, tokenizer_pad_id: int = 0):
        self.h5_path = h5_path
        self.pad_token_id = tokenizer_pad_id

        # Load graph features from joblib
        try:
            self.graphs = joblib.load(graph_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load graph features from {graph_path}: {e}")

        with h5py.File(h5_path, "r") as f:
            if "input_ids" not in f:
                raise KeyError(
                    f"Dataset 'input_ids' not found in HDF5 file: {h5_path}. "
                    "Ensure featurization stores 'input_ids'."
                )
            self.length = f["labels"].shape[0]

        if self.length != len(self.graphs):
            logger.error("Mismatch between HDF5 (%d) and graph features (%d)", self.length, len(self.graphs))
            raise ValueError("Graph feature count does not match HDF5 samples.")

    def __len__(self) -> int:
        return self.length

    def _get_h5_file_handle(self):
        """Returns a cached HDF5 file handle per worker."""
        if not hasattr(_worker_cache, "h5_file"):
            _worker_cache.h5_file = h5py.File(self.h5_path, "r")
        return _worker_cache.h5_file

    def __getitem__(self, idx: int) -> tuple:
        h5 = self._get_h5_file_handle()
        graph = self.graphs[idx]

        embedding = torch.tensor(h5["embedding"][idx], dtype=torch.float32)
        aux_features = torch.tensor(h5["auxiliary_features_normalized"][idx], dtype=torch.float32)
        label = torch.tensor(h5["labels"][idx], dtype=torch.float32)
        input_ids = torch.tensor(h5["input_ids"][idx], dtype=torch.long)

        return graph, embedding, aux_features, label, input_ids

    def __del__(self):
        if hasattr(_worker_cache, "h5_file"):
            _worker_cache.h5_file.close()
            del _worker_cache.h5_file


class MiniBatchCacheDataset(Dataset):
    """
    Wraps a Dataset with an in-memory LRU cache to reduce disk I/O for recently used samples.
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

        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return item
