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
        self.graph_only_mode = config.get("graph_only_mode", False)
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
                hidden_dim=self.graph_dim,
                n_layers=2,
                dropout=0.2,
                activation="ReLU",
            )
            scaled_graph_factor = config.get("scaled_graph_factor", 1.0)
            self.scale_graph = nn.Parameter(
                torch.tensor(scaled_graph_factor), requires_grad=False
            )
            scale_attn_factor = config.get("scale_attn_factor", 0.05)
            self.scale_attn = nn.Parameter(
                torch.tensor(scale_attn_factor), requires_grad=False
            )
            scale_aux_factor = config.get("scale_aux_factor", 0.1)
            self.aux_feature_dim = config.get("aux_feature_dim", 0)
            logger.info(
                "Graph-only predictor initialized with scaled factors: "
                f"graph={self.scale_graph.item()}, attn={self.scale_attn.item()}, aux={self.scale_aux_item()}"
            )
            logger.info("Model initialized in GRAPH-ONLY mode (Chemprop architecture).")
            return

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
        co_attention_layers = config.get("co_attention_layers", 2)
        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    dropout=self.cross_attn_dropout,
                    batch_first=True,
                )
                for _ in range(co_attention_layers)
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
                for _ in range(co_attention_layers)
            ]
        )
        
        self.gate_graph = nn.ModuleList(
            [nn.Linear(2 * self.embedding_dim, self.embedding_dim) for _ in range(co_attention_layers)]
        )
        self.gate_embedding = nn.ModuleList(
            [nn.Linear(2 * self.embedding_dim, self.embedding_dim) for _ in range(co_attention_layers)]
        )
        self.attention_norm_layers = nn.ModuleList(
            [nn.LayerNorm(self.embedding_dim) for _ in range(co_attention_layers)]
        )
        self.ffn_norm_layers = nn.ModuleList(
            [nn.LayerNorm(self.embedding_dim) for _ in range(co_attention_layers)]
        )
        self.cross_attn_ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embedding_dim, self.embedding_dim),
                    nn.GELU(),
                    nn.Dropout(self.cross_attn_dropout),
                    nn.Linear(self.embedding_dim, self.embedding_dim),
                )
                for _ in range(co_attention_layers)
            ]
        )

        # --- Final Fusion Parameters ---
        self.scale_graph = nn.Parameter(torch.tensor(1.0))
        self.scale_attn = nn.Parameter(torch.tensor(0.05))
        self.scale_aux = nn.Parameter(torch.tensor(0.1))

        # --- Prediction Head (for Fusion Mode) ---
        fusion_dropout_1 = config.get("fusion_dropout_1", 0.3)
        fusion_dropout_2 = config.get("fusion_dropout_2", 0.2)

        fusion_dim = self.graph_dim + self.embedding_dim + self.aux_feature_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(fusion_dropout_1),
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(fusion_dropout_2),
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
        atom_features = self.encoder.message_passing(bmg)
        graph_repr = self.encoder.agg(atom_features, bmg.batch)

        # --- Graph-Only Mode Logic ---
        if self.graph_only_mode:
            logits = self.graph_only_predictor(graph_repr)
            if return_attn:
                # Return a tuple that matches the new, longer expected output structure
                attn_output_placeholder = torch.zeros_like(graph_repr)
                return (
                    logits,
                    0.0,
                    0.0,
                    None,
                    None,
                    attn_output_placeholder,
                    graph_repr,
                )
            else:
                return (logits, 0.0, 0.0, None, None, None, None)

        # --- Fusion Mode Logic ---
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

        # Initialize placeholders for attention weights.
        attn_entropy_loss = 0.0
        token_prior_loss = 0.0
        graph_to_token_weights = None
        # --- ADDED: Placeholder for the new attention weights ---
        token_to_graph_weights = None
        # --------------------------------------------------------

        # --- Co-Attention Dialogue Loop ---
        for i in range(len(self.cross_attn)):
            # Graph queries tokens
            attn_out, g2t_weights = self.cross_attn[i](
                query=graph_queries,
                key=embedding_proj,
                value=embedding_proj,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            # Store the first loop's weights for plotting
            if i == 0:
                graph_to_token_weights = g2t_weights

            # Calculate regularization losses (only for graph-to-token for now)
            with torch.no_grad():
                attn_log = torch.log(g2t_weights + 1e-8)
                entropy = -torch.sum(g2t_weights * attn_log, dim=-1).mean()
                attn_entropy_loss += entropy
                if self.token_importance_prior is not None:
                    prior_scores = self.token_importance_prior[input_ids_batch]
                    prior_scores = (
                        prior_scores.unsqueeze(1).unsqueeze(1).expand_as(g2t_weights)
                    )
                    token_prior_loss += -(g2t_weights * prior_scores).sum(dim=-1).mean()

            # The bidirectional, gated update.
            if self.use_co_attention:
                # --- MODIFIED: Capture token-to-graph attention weights ---
                # Tokens query graph
                c2g_out, t2g_weights = self.co_attn[i](
                    query=embedding_proj,
                    key=graph_queries,
                    value=graph_queries,
                    need_weights=True,  # Ask for the weights
                    average_attn_weights=False,
                )
                if i == 0:
                    token_to_graph_weights = t2g_weights
                # ---------------------------------------------------------

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

        # --- MODIFIED: Update the return signature ---
        if return_attn:
            return (
                logits,
                attn_entropy_loss,
                token_prior_loss,
                graph_to_token_weights,  # The original weights
                token_to_graph_weights,  # The new weights
                attn_output,
                graph_repr,
                atom_features
            )
        else:
            # Return None for all extra values if not returning attention
            return (logits, attn_entropy_loss, token_prior_loss, None, None, None, None, None)
