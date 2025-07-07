# FILE: cage_fusion/pretrain/pretraining_model.py
import torch.nn as nn
from cage_fusion.models.cage import CAGEFusionModel


class PretrainingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # This now receives a Namespace object, so CAGEFusionModel is robust
        # to how it accesses config values (e.g., config.key).
        cage_model = CAGEFusionModel(config)
        self.encoder = cage_model.encoder

        # Use attribute-style access, which is now safe because we
        # pass a Namespace object instead of a dict.
        encoder_output_dim = config["graph_dim"]

        self.projection_head = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_output_dim, 128),
        )

    def forward(self, data):
        graph_embedding = self.encoder(data)
        projection = self.projection_head(graph_embedding)
        return projection
