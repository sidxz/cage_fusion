import h5py
import joblib
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset
from cage_fusion.utils.logging_utils import logger


class CageFusionStreamingDataset(Dataset):
    """
    Dataset for streaming token embeddings, auxiliary features, labels, and molecular graphs
    from disk using safe, per-instance HDF5 handles.
    MODIFIED to also stream SMILES strings for interpretability analysis.
    """

    def __init__(self, h5_path: str, graph_path: str, tokenizer_pad_id: int = 0):
        self.h5_path = h5_path
        self.pad_token_id = tokenizer_pad_id
        self._h5_handle = None  # Instance-level HDF5 handle (lazy-loaded)

        logger.debug(f"Loading HDF5 dataset from: {h5_path}")
        logger.debug(f"Loading graph features from: {graph_path}")
        logger.debug(f"Using pad token ID: {self.pad_token_id}")

        try:
            self.graphs = joblib.load(graph_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load graph features from {graph_path}: {e}")

        # Validate dataset structure and get length
        with h5py.File(h5_path, "r") as f:
            required_keys = ["input_ids", "labels", "smiles", "original_indices"]
            if not all(key in f for key in required_keys):
                raise KeyError(
                    f"HDF5 file at {h5_path} must contain all required keys: {required_keys}"
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
        """Returns a per-instance HDF5 file handle (lazy-loaded)."""
        if self._h5_handle is None:
            self._h5_handle = h5py.File(self.h5_path, "r")
        return self._h5_handle

    def __getitem__(self, idx: int) -> tuple:
        h5 = self._get_h5_file_handle()

        # --- FIXED: Decode the SMILES string from bytes to a proper string ---
        raw_smiles = h5["smiles"][idx]
        # h5py often returns bytes; we decode to utf-8 for universal compatibility.
        smiles_str = (
            raw_smiles.decode("utf-8")
            if isinstance(raw_smiles, bytes)
            else str(raw_smiles)
        )
        # ---------------------------------------------------------------------

        return (
            self.graphs[idx],
            torch.tensor(h5["embedding"][idx], dtype=torch.float32),
            torch.tensor(h5["auxiliary_features_normalized"][idx], dtype=torch.float32),
            torch.tensor(h5["labels"][idx], dtype=torch.float32),
            torch.tensor(h5["input_ids"][idx], dtype=torch.long),
            smiles_str,  # Return the properly decoded SMILES string
        )

    def __del__(self):
        """Ensure the per-instance HDF5 handle is properly closed."""
        if self._h5_handle is not None:
            try:
                self._h5_handle.close()
            except Exception:
                logger.warning("Failed to close HDF5 file cleanly.")
            finally:
                self._h5_handle = None


class MiniBatchCacheDataset(Dataset):
    """
    Dataset wrapper that implements an LRU cache for recently accessed samples to reduce I/O.
    This class does not need to be changed.
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
