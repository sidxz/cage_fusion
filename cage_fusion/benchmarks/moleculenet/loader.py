# loader.py
import os
from deepchem.molnet import load_dataset
from deepchem.feat import MolGraphConvFeaturizer

def load_molnet_dataset(name, data_dir=None, featurizer=None):
    data_dir = data_dir or os.path.join("data", "molnet")
    os.makedirs(data_dir, exist_ok=True)
    featurizer = featurizer or MolGraphConvFeaturizer()

    dataset, tasks, transformers = load_dataset(
        name=name,
        data_dir=data_dir,
        featurizer=featurizer
    )
    return dataset, tasks, transformers
