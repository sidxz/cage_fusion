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
        # Dimensionality of the graph representation from the MPNN.
        self.graph_dim = config["graph_dim"]

        # The common latent space dimension for the attention mechanism.
        self.embedding_dim = config["embedding_dim"]

        # Dimensionality of the auxiliary feature vector (e.g., RDKit descriptors).
        self.aux_feature_dim = config["aux_feature_dim"]

        # Number of output tasks for the model (e.g., 1 for binary classification).
        self.num_tasks = config["num_tasks"]

        # Number of heads in the multi-head attention mechanism.
        self.num_heads = config["num_heads"]

        # Dropout rate for the attention layers.
        self.cross_attn_dropout = config["cross_attn_dropout"]

        # Dropout rate for the projection layers.
        self.proj_dropout = config["proj_dropout"]

        # Flag to use atom-level features as queries (True) or a single graph-level feature (False).
        self.use_atom_level_queries = config["use_atom_level_queries"]

        # Flag to use the advanced gated co-attention mechanism (True) or simple cross-attention (False).
        self.use_advanced_features = config["use_advanced_features"]

        # --- Tokenizer and Special Tokens ---

        # Loads a pre-trained tokenizer to get IDs for special tokens used in masking.
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
        # A vector that assigns a prior importance to each token in the vocabulary.
        # Used for a regularization loss to guide the attention mechanism.
        tip = config.get("token_importance_prior")
        if tip is not None:
            self.register_buffer("token_importance_prior", tip)
            logger.debug("Loaded token importance prior")
        else:
            self.register_buffer("token_importance_prior", None)

        # --- Graph Encoder (Modality 1) ---
        # This MPNN processes the molecular graph to learn structural features.
        self.message_passing = BondMessagePassing()
        self.global_aggregation = AttentiveAggregation(
            input_size=self.graph_dim, output_size=self.graph_dim
        )

        # A dummy predictor is required by the Chemprop MPNN class structure but is not used.
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

        # --- Projection and Aggregation Layers ---
        # Used to aggregate the final atom-level representations after the attention dialogue.
        self.attention_aggregation = MeanAggregation()

        # Projects the graph representation into the common embedding dimension for the attention dialogue.
        self.graph_proj = nn.Sequential(
            nn.Linear(self.graph_dim, self.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(self.proj_dropout),
        )
        logger.debug("Graph projection layer created")

        # Projects the sequence embeddings, allowing the model to adapt them for the fusion task.
        self.embedding_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Dropout(self.proj_dropout),
        )
        logger.debug("Embedding projection layer created")

        # --- Co-Attention Dialogue Block ---
        # A list of attention layers for the graph-to-sequence information flow. (Cross-attention)
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

        # A list of attention layers for the sequence-to-graph information flow. (Co-attention)
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

        # Gating layers that learn to control the update of the graph representation.
        self.gate_graph = nn.ModuleList(
            [nn.Linear(2 * self.embedding_dim, self.embedding_dim) for _ in range(2)]
        )

        # Gating layers that learn to control the update of the sequence representation.
        self.gate_embedding = nn.ModuleList(
            [nn.Linear(2 * self.embedding_dim, self.embedding_dim) for _ in range(2)]
        )

        # Normalization layers applied after the attention operation.
        self.attention_norm_layers = nn.ModuleList(
            [nn.LayerNorm(self.embedding_dim) for _ in range(2)]
        )

        # Normalization layers applied after the feed-forward network operation.
        self.ffn_norm_layers = nn.ModuleList(
            [nn.LayerNorm(self.embedding_dim) for _ in range(2)]
        )

        # The feed-forward network, a standard component of a Transformer block.
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

        # --- Final Fusion Parameters ---
        # Learnable scalar weights that control the contribution of each modality to the final prediction.
        self.scale_graph = nn.Parameter(torch.tensor(1.0))
        self.scale_attn = nn.Parameter(torch.tensor(0.05))
        self.scale_aux = nn.Parameter(torch.tensor(0.1))

        # --- Prediction Head ---
        # The final MLP that takes the fused representation and produces the output logits.
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
        bmg,  # BatchedMolecularGraph: A batch of molecular graphs from Chemprop.
        sequence_embeddings,  # Tensor [batch, seq_len, embed_dim]: Pre-computed embeddings from a language model.
        attn_mask,  # Tensor [batch, seq_len]: The attention mask for the sequence embeddings.
        aux_feats,  # Tensor [batch, aux_dim]: The vector of auxiliary physicochemical features.
        input_ids_batch,  # Tensor [batch, seq_len]: The raw token IDs for the sequences, used for masking.
        return_attn=False,
    ):
        """
        Forward pass of CAGEFusionModel.
        Logs input dimensions for debugging, performs attention-based fusion of molecular graph and sequence embeddings.
        """

        # Log input shapes and types
        logger.debug("===== Forward Pass Inputs =====")
        logger.debug("bmg (BatchedMolecularGraph): type = {}", type(bmg))
        logger.debug("sequence_embeddings: shape = {}", sequence_embeddings.shape)
        logger.debug("attn_mask: shape = {}", attn_mask.shape)
        logger.debug("aux_feats: shape = {}", aux_feats.shape)
        logger.debug("input_ids_batch: shape = {}", input_ids_batch.shape)
        logger.debug("return_attn: {}", return_attn)

        # Consistency checks for batch size
        batch_size = sequence_embeddings.shape[0]
        assert attn_mask.shape[0] == batch_size, "attn_mask batch size mismatch"
        assert (
            input_ids_batch.shape[0] == batch_size
        ), "input_ids_batch batch size mismatch"
        assert aux_feats.shape[0] == batch_size, "aux_feats batch size mismatch"
        assert bmg.batch.max().item() + 1 == batch_size, "bmg batch size mismatch"
        # Assertions to catch mismatches early
        assert (
            sequence_embeddings.dim() == 3
        ), "sequence_embeddings must be 3D (batch_size, seq_len, embedding_dim)"
        assert (
            attn_mask.shape == input_ids_batch.shape
        ), "attn_mask and input_ids_batch must have the same shape"
        assert (
            aux_feats.dim() == 2 and aux_feats.shape[1] == self.aux_feature_dim
        ), f"aux_feats should be (batch_size, {self.aux_feature_dim})"
        assert (
            sequence_embeddings.shape[0] == aux_feats.shape[0]
        ), "Batch size mismatch between tokens and aux_feats"

        # --- Preparation and Masking ---
        # Project sequence embeddings and mask out special tokens like [PAD] and [CLS].
        embedding_proj = self.embedding_proj(sequence_embeddings)
        mask_pad_cls = (
            (input_ids_batch == self.PAD_TOKEN_ID)
            | (input_ids_batch == self.CLS_TOKEN_ID)
        ).unsqueeze(-1)
        embedding_proj = embedding_proj.masked_fill(mask_pad_cls, 0.0)

        # Create the final key padding mask for the attention mechanism.
        special_ids = [self.CLS_TOKEN_ID, self.PAD_TOKEN_ID, self.SEP_TOKEN_ID]
        explicit_special_mask = torch.zeros_like(input_ids_batch, dtype=torch.bool)
        for tok in special_ids:
            explicit_special_mask |= input_ids_batch == tok
        key_padding_mask = (attn_mask == 0) | explicit_special_mask

        # --- Initial Modality Representations ---
        # Get atom-level features and a single graph-level representation from the MPNN.
        atom_features = self.encoder.message_passing(bmg)
        graph_repr = self.encoder.agg(atom_features, bmg.batch)

        # Prepare the initial queries for the attention dialogue, derived from the graph.
        if self.use_atom_level_queries:
            atom_lengths = torch.bincount(bmg.batch)
            segments = torch.split(atom_features, atom_lengths.tolist())
            padded = pad_sequence(segments, batch_first=True, padding_value=0.0)
            graph_queries = self.graph_proj(padded)
        else:
            graph_queries = self.graph_proj(graph_repr).unsqueeze(1)

        # Initialize regularization losses and a placeholder for attention weights.
        attn_entropy_loss = 0.0
        attn_entropy_loss = 0.0
        token_prior_loss = 0.0
        attn_weights_final = None

        # --- Co-Attention Dialogue Loop ---
        for i in range(2):
            # Graph queries attend to sequence embeddings.
            attn_out, attn_weights = self.cross_attn[i](
                graph_queries,  # RENAMED
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
            if self.use_advanced_features:
                # Sequence embeddings attend to graph queries.
                c2g_out, _ = self.co_attn[i](
                    embedding_proj, graph_queries, graph_queries
                )
                # Calculate the gate for the graph update.
                gate_g = torch.sigmoid(
                    self.gate_graph[i](torch.cat([graph_queries, attn_out], dim=-1))
                )
                # Update the graph queries.
                graph_queries = self.attention_norm_layers[i](
                    graph_queries + gate_g * attn_out
                )
                # Calculate the gate for the embedding update.
                gate_e = torch.sigmoid(
                    self.gate_embedding[i](torch.cat([embedding_proj, c2g_out], dim=-1))
                )
                # Update the sequence embeddings.
                embedding_proj = self.attention_norm_layers[i](
                    embedding_proj + gate_e * c2g_out
                )
            else:
                # Simple cross-attention update if advanced features are disabled.
                graph_queries = self.attention_norm_layers[i](graph_queries + attn_out)

            # Apply the feed-forward network part of the Transformer block.
            graph_queries = self.ffn_norm_layers[i](
                graph_queries + self.cross_attn_ffn[i](graph_queries)
            )

        # --- Aggregation and Final Fusion ---
        # Aggregate the final enriched graph queries into a single vector per molecule.
        if self.use_atom_level_queries:
            unpadded = [graph_queries[j, :l] for j, l in enumerate(atom_lengths)]
            flat = torch.cat(unpadded, dim=0)
            attn_output = self.attention_aggregation(flat, bmg.batch)
        else:
            attn_output = graph_queries.squeeze(1)

        # Concatenate the three information streams, scaled by their learnable weights.
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
        # Pass the fused vector through the final MLP to get the logits.
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
