# Entrypoint for pretraining

# TODO [Phase 9]: Connect everything and run training
from config import get_config
from dataset import GraphPretrainDataset
from model import GraphContrastiveModel
from trainer import PretrainTrainer

if __name__ == "__main__":
    config = get_config()
    dataset = GraphPretrainDataset(config["dataset_path"])
    model = GraphContrastiveModel(config)
    trainer = PretrainTrainer(model, dataset, config)
    trainer.train()
