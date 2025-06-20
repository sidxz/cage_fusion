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
import json


class CAGEFusionModel(nn.Module):
    """
    CAGEFusionModel (Co‑Attention Graph Embedding) for molecular property prediction.
    This model fuses information from a molecular graph, a sequence-based language model,
    and auxiliary physicochemical features to make predictions. The core innovation is a
    gated co-attention mechanism that allows the graph and sequence representations to
    bidirectionally inform and refine each other.
    """

    def __init__(self, config):
        super().__init__()

        # --- Configuration and Hyperparameters ---
        def make_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return f"<Tensor shape={obj.shape} device={obj.device}>"
            if isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            return obj

        serializable_config = make_serializable(config)
        logger.debug(
            "Initializing CAGEFusionModel with config:\n%s",
            json.dumps(serializable_config, indent=2),
        )

        assert (
            isinstance(config, dict) and len(config) > 0
        ), "[XXXXX] Received empty config in model init!"

        # Save shapes and hyperparameters
        self.config = config
        self.graph_dim = config["graph_dim"]
        self.embedding_dim = config["embedding_dim"]
        self.aux_feature_dim = config["aux_feature_dim"]
        self.num_tasks = config["num_tasks"]
        self.num_heads = config["num_heads"]
        self.cross_attn_dropout = config["cross_attn_dropout"]
        self.proj_dropout = config["proj_dropout"]

        # --- Control Flags ---
        # Flag to enable graph-only prediction mode, bypassing fusion.
        self.graph_only_mode = config.get("graph_only_mode", False)
        # Flag to use the advanced gated co-attention mechanism.
        self.use_co_attention = config.get("use_co_attention", True)

        # --- Tokenizer and Special Tokens ---
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

        # --- Optional Regularization Prior ---
        tip = config.get("token_importance_prior")
        if tip is not None:
            self.register_buffer("token_importance_prior", tip)
            logger.debug("Loaded token importance prior")
        else:
            self.register_buffer("token_importance_prior", None)

        # --- Graph Encoder (Modality 1) ---
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

        # --- Graph-Only Mode Predictor (Mimics Chemprop) ---
        if self.graph_only_mode:
            self.graph_only_predictor = BinaryClassificationFFN(
                input_dim=self.graph_dim,
                n_tasks=self.num_tasks,
                hidden_dim=self.graph_dim,  # Common practice in Chemprop
                n_layers=2,
                dropout=0.2,
                activation="ReLU",
            )
            # Add placeholder attributes for logging consistency
            self.scale_graph = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            self.scale_attn = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.scale_aux = nn.Parameter(torch.tensor(0.0), requires_grad=False)

            logger.info("Model initialized in GRAPH-ONLY mode (Chemprop architecture).")
            return  # Skip initializing fusion components if in graph-only mode

        # --- Projection and Aggregation Layers (for Fusion Mode) ---
        self.attention_aggregation = MeanAggregation()
        self.graph_proj = nn.Sequential(
            nn.Linear(self.graph_dim, self.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(self.proj_dropout),
        )
        self.embedding_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Dropout(self.proj_dropout),
        )

        # --- Co-Attention Dialogue Block ---
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
        self.attention_norm_layers = nn.ModuleList(
            [nn.LayerNorm(self.embedding_dim) for _ in range(2)]
        )
        self.ffn_norm_layers = nn.ModuleList(
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

        # --- Final Fusion Parameters ---
        self.scale_graph = nn.Parameter(torch.tensor(1.0))
        self.scale_attn = nn.Parameter(torch.tensor(0.05))
        self.scale_aux = nn.Parameter(torch.tensor(0.1))

        # --- Prediction Head (for Fusion Mode) ---
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
        sequence_embeddings,
        attn_mask,
        aux_feats,
        input_ids_batch,
        return_attn=False,
    ):
        """
        Forward pass of CAGEFusionModel.
        """
        # --- Initial Modality Representations ---
        # Get atom-level features and a single graph-level representation from the MPNN.
        atom_features = self.encoder.message_passing(bmg)
        graph_repr = self.encoder.agg(atom_features, bmg.batch)

        # --- Graph-Only Mode Logic ---
        if self.graph_only_mode:
            logits = self.graph_only_predictor(graph_repr)
            # Return a tuple that matches the expected output structure to avoid unpacking errors.
            if return_attn:
                # Return zero tensors as placeholders for attn_output to prevent errors in evaluation.
                attn_output_placeholder = torch.zeros_like(graph_repr)
                return (logits, 0.0, 0.0, None, attn_output_placeholder, graph_repr)
            else:
                return (logits, 0.0, 0.0)

        # --- Fusion Mode Logic ---
        logger.debug("===== Forward Pass Inputs (Fusion Mode) =====")
        batch_size = sequence_embeddings.shape[0]

        # --- Preparation and Masking ---
        embedding_proj = self.embedding_proj(sequence_embeddings)
        mask_pad_cls = (
            (input_ids_batch == self.PAD_TOKEN_ID)
            | (input_ids_batch == self.CLS_TOKEN_ID)
        ).unsqueeze(-1)
        embedding_proj = embedding_proj.masked_fill(mask_pad_cls, 0.0)

        special_ids = [self.CLS_TOKEN_ID, self.PAD_TOKEN_ID, self.SEP_TOKEN_ID]
        explicit_special_mask = torch.zeros_like(input_ids_batch, dtype=torch.bool)
        for tok in special_ids:
            explicit_special_mask |= input_ids_batch == tok
        key_padding_mask = (attn_mask == 0) | explicit_special_mask

        # --- Prepare Queries (Always Atom-Level) ---
        atom_lengths = torch.bincount(bmg.batch)
        segments = torch.split(atom_features, atom_lengths.tolist())
        padded = pad_sequence(segments, batch_first=True, padding_value=0.0)
        graph_queries = self.graph_proj(padded)

        # Initialize regularization losses and a placeholder for attention weights.
        attn_entropy_loss = 0.0
        token_prior_loss = 0.0
        attn_weights_final = None

        # --- Co-Attention Dialogue Loop ---
        for i in range(2):
            attn_out, attn_weights = self.cross_attn[i](
                graph_queries,
                embedding_proj,
                embedding_proj,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            if i == 0:
                attn_weights_final = attn_weights

            # Calculate regularization losses
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

            # The bidirectional, gated update.
            if self.use_co_attention:
                c2g_out, _ = self.co_attn[i](
                    embedding_proj, graph_queries, graph_queries
                )
                gate_g = torch.sigmoid(
                    self.gate_graph[i](torch.cat([graph_queries, attn_out], dim=-1))
                )
                graph_queries = self.attention_norm_layers[i](
                    graph_queries + gate_g * attn_out
                )
                gate_e = torch.sigmoid(
                    self.gate_embedding[i](torch.cat([embedding_proj, c2g_out], dim=-1))
                )
                embedding_proj = self.attention_norm_layers[i](
                    embedding_proj + gate_e * c2g_out
                )
            else:
                graph_queries = self.attention_norm_layers[i](graph_queries + attn_out)

            graph_queries = self.ffn_norm_layers[i](
                graph_queries + self.cross_attn_ffn[i](graph_queries)
            )

        # --- Aggregation and Final Fusion ---
        unpadded = [graph_queries[j, :l] for j, l in enumerate(atom_lengths)]
        flat = torch.cat(unpadded, dim=0)
        attn_output = self.attention_aggregation(flat, bmg.batch)

        fused = torch.cat(
            [
                self.scale_graph * graph_repr,
                self.scale_attn * attn_output,
                self.scale_aux * aux_feats,
            ],
            dim=1,
        )
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e3, neginf=-1e3)

        # --- Prediction ---
        logits = self.output(self.fusion_mlp(fused))

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
