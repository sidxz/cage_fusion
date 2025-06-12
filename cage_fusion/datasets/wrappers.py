from collections import OrderedDict
from torch.utils.data import Dataset
from typing import TypeVar

# A generic type variable for the items returned by the dataset.
T_co = TypeVar('T_co', covariant=True)

class MiniBatchCacheDataset(Dataset[T_co]):
    """
    A Dataset wrapper that provides an in-memory LRU (Least Recently Used) cache.

    This wrapper is designed to speed up training when data loading from the
    underlying dataset (e.g., from disk or a slow network) is a bottleneck.
    It stores the most recently accessed data points in an OrderedDict, which
    acts as a fast in-memory cache.

    Args:
        dataset: The dataset object to wrap (e.g., CageFusionStreamingDataset).
        cache_size: The maximum number of samples to store in the cache.
    """
    def __init__(self, dataset: Dataset[T_co], cache_size: int = 1024):
        self.dataset = dataset
        self.cache_size = cache_size
        self.cache: OrderedDict[int, T_co] = OrderedDict()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> T_co:
        """
        Retrieves an item by index.

        First, it checks if the item is in the cache. If so, it returns the
        cached item and moves it to the end of the OrderedDict to mark it as
        recently used. If not, it fetches the item from the underlying dataset,
        adds it to the cache, and then returns it. If the cache is full, the
        least recently used item is evicted.
        """
        # Check if the item is already in the in-memory cache
        if idx in self.cache:
            # Move the accessed item to the end to mark it as recently used
            self.cache.move_to_end(idx)
            return self.cache[idx]

        # If not in cache, load the item from the underlying dataset on disk
        item = self.dataset[idx]

        # Add the new item to our cache
        self.cache[idx] = item
        
        # Evict the least recently used item if the cache has exceeded its size
        if len(self.cache) > self.cache_size:
            # popitem(last=False) removes the first item (the least recently used)
            self.cache.popitem(last=False)

        return item
