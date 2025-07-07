# FILE: cage_fusion/pretrain/contrastive_dataset.py
from torch.utils.data import Dataset
from .utils import augment_graph


class ContrastiveDataset(Dataset):
    def __init__(self, pyg_dataset):
        self.pyg_dataset = pyg_dataset

    def __len__(self):
        return len(self.pyg_dataset)

    def __getitem__(self, idx):
        original_data = self.pyg_dataset[idx]
        view_1 = augment_graph(original_data)
        view_2 = augment_graph(original_data)
        return view_1, view_2
