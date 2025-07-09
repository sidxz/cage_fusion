from cage_fusion.pretrain.dataset import GraphPretrainDataset
from cage_fusion.pretrain.model import GraphContrastiveModel
from cage_fusion.pretrain.config import get_config


def test_graph_dataset():
    config = get_config()
    dataset = GraphPretrainDataset(config["dataset_path"])
    print(f"✅ Dataset loaded with {len(dataset)} molecules")

    sample1, sample2 = dataset[0]
    print("Sample 1:", sample1)
    print("Sample 2:", sample2)
    return dataset


def test_augmentation():
    dataset = test_graph_dataset()
    g1, g2 = dataset[0]
    print("Augmented view 1:", g1)
    print("Augmented view 2:", g2)
    
def test_model_forward():
    config = get_config()
    dataset = dataset = test_graph_dataset()
    g1, g2 = dataset[0]
    model = GraphContrastiveModel(config)
    z1, z2 = model(g1, g2)
    print("✅ Embeddings:", z1.shape, z2.shape)    
    
if __name__ == "__main__":
    dataset = test_model_forward()