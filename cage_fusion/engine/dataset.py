import h5py
import joblib
import torch
import threading
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset

# A thread-local cache for HDF5 file handles to avoid issues with multiprocessing
_worker_cache = threading.local()

class CageFusionStreamingDataset(Dataset):
    """
    A PyTorch Dataset for streaming data from HDF5 and graph feature files.
    
    This dataset is optimized for use with multiprocessing in PyTorch DataLoaders
    by opening a file handle for the HDF5 file on each worker process.
    """
    def __init__(self, h5_path: str, graph_path: str, tokenizer_pad_id: int = 0):
        self.h5_path = h5_path
        self.graphs = joblib.load(graph_path)
        self.pad_token_id = tokenizer_pad_id

        with h5py.File(h5_path, "r") as f:
            self.length = f["labels"].shape[0]
            if "input_ids" not in f:
                raise KeyError(f"Dataset 'input_ids' not found in HDF5 file: {h5_path}. "
                               "Please ensure featurization saves this dataset.")

        assert self.length == len(self.graphs), "❌ Mismatch between graph and HDF5 length."

    def __len__(self):
        return self.length

    def _get_h5_file_handle(self):
        """Opens and caches a file handle for the HDF5 file on the current worker."""
        if not hasattr(_worker_cache, "h5_file"):
            _worker_cache.h5_file = h5py.File(self.h5_path, "r")
        return _worker_cache.h5_file

    def __getitem__(self, idx):
        h5 = self._get_h5_file_handle()
        graph = self.graphs[idx]

        # Retrieve data from HDF5 and convert to tensors
        embedding = torch.tensor(h5["embedding"][idx], dtype=torch.float32)
        aux_features = torch.tensor(h5["auxiliary_features_normalized"][idx], dtype=torch.float32)
        label = torch.tensor(h5["labels"][idx], dtype=torch.float32)
        input_ids = torch.tensor(h5["input_ids"][idx], dtype=torch.long)

        return graph, embedding, aux_features, label, input_ids


class MiniBatchCacheDataset(Dataset):
    """
    Wraps around an existing Dataset and caches recently used samples in RAM
    using a Least Recently Used (LRU) eviction strategy.
    """
    def __init__(self, dataset: Dataset, cache_size: int = 1024):
        self.dataset = dataset
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]

        item = self.dataset[idx]
        self.cache[idx] = item
        self.cache.move_to_end(idx)

        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return item
