import torch
import torch.nn as nn
from chemprop.nn.message_passing import BondMessagePassing
from chemprop.nn.agg import AttentiveAggregation
from chemprop.nn.predictors import BinaryClassificationFFN
from chemprop.models.model import MPNN
from cage_fusion.utils.logging_utils import logger


class ChempropFNN(nn.Module):
    """
    Drop-in replacement for CAGEFusionModel using Chemprop MPNN only.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.graph_dim = config["graph_dim"]
        self.num_tasks = config["num_tasks"]

        # Graph encoder and predictor
        message_passing = BondMessagePassing()
        aggregation = AttentiveAggregation(
            input_size=self.graph_dim, output_size=self.graph_dim
        )
        predictor = BinaryClassificationFFN(
            input_dim=self.graph_dim,
            n_tasks=self.num_tasks,
            hidden_dim=128,
            n_layers=2,
            dropout=0.2,
            activation="ReLU",
        )
        self.model = MPNN(
            message_passing=message_passing,
            agg=aggregation,
            predictor=predictor,
        )

        logger.info("Initialized ChempropOnlyModel")

    def forward(
        self,
        bmg,
        embedding_tokens=None,
        attn_mask=None,
        aux_feats=None,
        input_ids_batch=None,
        return_attn=False,
    ):
        return self.model(bmg)  # returns logits of shape [B, num_tasks]
