import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from chemprop.nn.message_passing import BondMessagePassing
from chemprop.nn.agg import AttentiveAggregation, MeanAggregation
from chemprop.nn.predictors import BinaryClassificationFFN
from chemprop.models.model import MPNN
from torch.nn.utils.rnn import pad_sequence
from cage_fusion.utils.logging_utils import logger


class CAGEFusionModel(nn.Module):
    """
    CAGEFusionModel (Co‑Attention Graph Embedding) for molecular property prediction.
    """

    def __init__(self, config):
        super().__init__()
        logger.info("Initializing CAGEFusionModel with config: {}", config)

        # Save shapes and hyperparameters
        self.config = config
        self.graph_dim = config["graph_dim"]
        self.embedding_dim = config["embedding_dim"]
        self.aux_feature_dim = config["aux_feature_dim"]
        self.num_tasks = config["num_tasks"]
        self.num_heads = config["num_heads"]
        self.cross_attn_dropout = config["cross_attn_dropout"]
        self.proj_dropout = config["proj_dropout"]
        self.use_atom_level_queries = config["use_atom_level_queries"]
        self.use_advanced_features = config["use_advanced_features"]

        # Load tokenizer and store special token IDs
        tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
        self.register_buffer(
            "PAD_TOKEN_ID", torch.tensor(tokenizer.pad_token_id, dtype=torch.long)
        )
        self.register_buffer(
            "CLS_TOKEN_ID", torch.tensor(tokenizer.cls_token_id, dtype=torch.long)
        )
        self.register_buffer(
            "SEP_TOKEN_ID", torch.tensor(tokenizer.sep_token_id, dtype=torch.long)
        )
        logger.debug("Stored PAD/CLS/SEP token IDs")

        # Optional prior over token importance (used for regularization)
        tip = config.get("token_importance_prior")
        if tip is not None:
            self.register_buffer("token_importance_prior", tip)
            logger.debug("Loaded token importance prior")
        else:
            self.register_buffer("token_importance_prior", None)

        # Graph encoder using message passing + attentive aggregation
        self.message_passing = BondMessagePassing()
        self.global_aggregation = AttentiveAggregation(
            input_size=self.graph_dim, output_size=self.graph_dim
        )
        dummy_pred = BinaryClassificationFFN(
            input_dim=self.graph_dim,
            n_tasks=self.num_tasks,
            hidden_dim=128,
            n_layers=1,
            dropout=0.1,
            activation="ReLU",
        )
        self.encoder = MPNN(
            message_passing=self.message_passing,
            agg=self.global_aggregation,
            predictor=dummy_pred,
        )
        logger.debug("Graph encoder initialized")

        # Projection layers for graph and token-space embeddings
        self.attention_aggregation = MeanAggregation()

        self.graph_proj = nn.Sequential(
            nn.Linear(self.graph_dim, self.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(self.proj_dropout),
        )
        logger.debug("Graph projection layer created")

        self.embedding_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Dropout(self.proj_dropout),
        )
        logger.debug("Embedding projection layer created")

        # Cross-attention and gating modules for two fusion layers
        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    dropout=self.cross_attn_dropout,
                    batch_first=True,
                )
                for _ in range(2)
            ]
        )
        self.co_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    dropout=self.cross_attn_dropout,
                    batch_first=True,
                )
                for _ in range(2)
            ]
        )
        self.gate_graph = nn.ModuleList(
            [nn.Linear(2 * self.embedding_dim, self.embedding_dim) for _ in range(2)]
        )
        self.gate_embedding = nn.ModuleList(
            [nn.Linear(2 * self.embedding_dim, self.embedding_dim) for _ in range(2)]
        )
        self.cross_attn_norms_attn = nn.ModuleList(
            [nn.LayerNorm(self.embedding_dim) for _ in range(2)]
        )
        self.cross_attn_norms_ffn = nn.ModuleList(
            [nn.LayerNorm(self.embedding_dim) for _ in range(2)]
        )
        self.cross_attn_ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embedding_dim, self.embedding_dim),
                    nn.GELU(),
                    nn.Dropout(self.cross_attn_dropout),
                    nn.Linear(self.embedding_dim, self.embedding_dim),
                )
                for _ in range(2)
            ]
        )
        logger.debug("Cross-attention and gating modules set up")

        # Learnable scaling factors for fusion contributions
        self.scale_graph = nn.Parameter(torch.tensor(1.0))
        self.scale_attn = nn.Parameter(torch.tensor(0.1))
        self.scale_aux = nn.Parameter(torch.tensor(0.1))

        # Final MLP fusion network combining graph, text, and auxiliary features
        fusion_dim = self.graph_dim + self.embedding_dim + self.aux_feature_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.output = nn.Linear(128, self.num_tasks)
        logger.info("CAGEFusionModel initialization complete")

    def forward(
        self,
        bmg,
        embedding_tokens,
        attn_mask,
        aux_feats,
        input_ids_batch,
        return_attn=False,
    ):
        """
        Forward pass of CAGEFusionModel.
        Logs input dimensions for debugging, performs attention-based fusion of molecular graph and sequence embeddings.
        """

        # Log input shapes and types
        logger.debug("===== Forward Pass Inputs =====")
        logger.debug("bmg (BatchedMolecularGraph): type = {}", type(bmg))
        logger.debug("embedding_tokens: shape = {}", embedding_tokens.shape)
        logger.debug("attn_mask: shape = {}", attn_mask.shape)
        logger.debug("aux_feats: shape = {}", aux_feats.shape)
        logger.debug("input_ids_batch: shape = {}", input_ids_batch.shape)
        logger.debug("return_attn: {}", return_attn)

        # Consistency checks for batch size
        batch_size = embedding_tokens.shape[0]
        assert attn_mask.shape[0] == batch_size, "attn_mask batch size mismatch"
        assert (
            input_ids_batch.shape[0] == batch_size
        ), "input_ids_batch batch size mismatch"
        assert aux_feats.shape[0] == batch_size, "aux_feats batch size mismatch"
        assert bmg.batch.max().item() + 1 == batch_size, "bmg batch size mismatch"
        # Assertions to catch mismatches early
        assert (
            embedding_tokens.dim() == 3
        ), "embedding_tokens must be 3D (batch_size, seq_len, embedding_dim)"
        assert (
            attn_mask.shape == input_ids_batch.shape
        ), "attn_mask and input_ids_batch must have the same shape"
        assert (
            aux_feats.dim() == 2 and aux_feats.shape[1] == self.aux_feature_dim
        ), f"aux_feats should be (batch_size, {self.aux_feature_dim})"
        assert (
            embedding_tokens.shape[0] == aux_feats.shape[0]
        ), "Batch size mismatch between tokens and aux_feats"

        # Project and mask special tokens
        embedding_proj = self.embedding_proj(embedding_tokens)
        mask_pad_cls = (
            (input_ids_batch == self.PAD_TOKEN_ID)
            | (input_ids_batch == self.CLS_TOKEN_ID)
        ).unsqueeze(-1)
        embedding_proj = embedding_proj.masked_fill(mask_pad_cls, 0.0)

        # Create key padding mask including PAD, CLS, SEP
        special_ids = [self.CLS_TOKEN_ID, self.PAD_TOKEN_ID, self.SEP_TOKEN_ID]
        explicit_special_mask = torch.zeros_like(input_ids_batch, dtype=torch.bool)
        for tok in special_ids:
            explicit_special_mask |= input_ids_batch == tok
        key_padding_mask = (attn_mask == 0) | explicit_special_mask

        # Message passing and graph representation
        atom_features = self.encoder.message_passing(bmg)
        graph_repr = self.encoder.agg(atom_features, bmg.batch)

        # Prepare graph queries (atom-level or graph-level)
        if self.use_atom_level_queries:
            atom_lengths = torch.bincount(bmg.batch)
            segments = torch.split(atom_features, atom_lengths.tolist())
            padded = pad_sequence(segments, batch_first=True, padding_value=0.0)
            x = self.graph_proj(padded)
        else:
            x = self.graph_proj(graph_repr).unsqueeze(1)

        # Initialize attention outputs
        attn_entropy_loss = 0.0
        token_prior_loss = 0.0
        attn_weights_final = None

        for i in range(2):
            attn_out, attn_weights = self.cross_attn[i](
                x,
                embedding_proj,
                embedding_proj,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            if i == 0:
                attn_weights_final = attn_weights

            # Regularize attention (entropy + prior)
            with torch.no_grad():
                attn_log = torch.log(attn_weights + 1e-8)
                entropy = -torch.sum(attn_weights * attn_log, dim=-1).mean()
                attn_entropy_loss += entropy

                if self.token_importance_prior is not None:
                    prior_scores = self.token_importance_prior[input_ids_batch]
                    prior_scores = (
                        prior_scores.unsqueeze(1).unsqueeze(1).expand_as(attn_weights)
                    )
                    token_prior_loss += (
                        -(attn_weights * prior_scores).sum(dim=-1).mean()
                    )

            # Apply gated co-attention if enabled
            if self.use_advanced_features:
                c2g_out, _ = self.co_attn[i](embedding_proj, x, x)
                gate_g = torch.sigmoid(
                    self.gate_graph[i](torch.cat([x, attn_out], dim=-1))
                )
                x = self.cross_attn_norms_attn[i](x + gate_g * attn_out)

                gate_e = torch.sigmoid(
                    self.gate_embedding[i](torch.cat([embedding_proj, c2g_out], dim=-1))
                )
                embedding_proj = self.cross_attn_norms_attn[i](
                    embedding_proj + gate_e * c2g_out
                )
            else:
                x = self.cross_attn_norms_attn[i](x + attn_out)

            x = self.cross_attn_norms_ffn[i](x + self.cross_attn_ffn[i](x))

        # Aggregate final attention output
        if self.use_atom_level_queries:
            unpadded = [x[j, :l] for j, l in enumerate(atom_lengths)]
            flat = torch.cat(unpadded, dim=0)
            attn_output = self.attention_aggregation(flat, bmg.batch)
        else:
            attn_output = x.squeeze(1)

        # Fuse graph, attention, and aux features
        fused = torch.cat(
            [
                self.scale_graph * graph_repr,
                self.scale_attn * attn_output,
                self.scale_aux * aux_feats,
            ],
            dim=1,
        )
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e3, neginf=-1e3)
        logits = self.output(self.fusion_mlp(fused))

        # Check for NaNs in output
        if torch.isnan(logits).any():
            logger.error(
                "Logits contain NaNs. Investigate input or network instability."
            )
            raise ValueError("Output logits contain NaNs")

        logger.debug("===== Forward Pass Complete =====")
        return (
            (
                logits,
                attn_entropy_loss,
                token_prior_loss,
                attn_weights_final,
                attn_output,
                graph_repr,
            )
            if return_attn
            else (logits, attn_entropy_loss, token_prior_loss)
        )
