import torch
import torch.nn as nn
from chemprop.nn.message_passing import BondMessagePassing
from chemprop.nn.agg import AttentiveAggregation
from chemprop.nn.predictors import BinaryClassificationFFN
from chemprop.models.model import MPNN

class GraphContrastiveModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = MPNN(
            message_passing=BondMessagePassing(),
            agg=AttentiveAggregation(config["graph_dim"], config["graph_dim"]),
            predictor=BinaryClassificationFFN(
                input_dim=config["graph_dim"],
                n_tasks=1,
                hidden_dim=128,
                n_layers=1,
                dropout=0.1,
                activation="ReLU"
            )
        )
        self.projector = nn.Sequential(
            nn.Linear(config["graph_dim"], config["proj_hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["proj_hidden_dim"], config["proj_dim"]),
        )

    def forward(self, g1, g2):
        return self.encode(g1), self.encode(g2)

    def encode(self, graph):
        x = self.encoder.message_passing(graph)
        x = self.encoder.agg(x, graph.batch)
        return nn.functional.normalize(self.projector(x), dim=-1)