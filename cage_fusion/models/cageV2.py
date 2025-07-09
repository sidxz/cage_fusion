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
from cage_fusion.models.fg_prompt_addon import FunctionalGroupPrompt
from cage_fusion.engine.fg_utils import get_functional_groups, NUM_FUNCTIONAL_GROUPS
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
        self.scaled_fg_factor = torch.tensor(config.get("scaled_fg_factor", 0.1))

        # --- Control Flags ---
        self.graph_only_mode = config.get("graph_only_mode", False)
        self.use_co_attention = config.get("use_co_attention", True)
        self.use_aux_features = config.get("use_aux_features", True)
        self.use_fg_prompt = config.get("use_fg_prompt", False)

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
        scaled_graph_factor = config.get("scaled_graph_factor", 1.0)
        scale_attn_factor = config.get("scale_attn_factor", 1.0)
        scale_aux_factor = config.get("scale_aux_factor", 1.0)

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
            self.scale_graph = nn.Parameter(
                torch.tensor(scaled_graph_factor), requires_grad=False
            )
            self.scale_attn = nn.Parameter(
                torch.tensor(scale_attn_factor), requires_grad=False
            )
            self.scale_aux = nn.Parameter(
                torch.tensor(scale_aux_factor), requires_grad=False
            )
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
            [
                nn.Linear(2 * self.embedding_dim, self.embedding_dim)
                for _ in range(co_attention_layers)
            ]
        )
        self.gate_embedding = nn.ModuleList(
            [
                nn.Linear(2 * self.embedding_dim, self.embedding_dim)
                for _ in range(co_attention_layers)
            ]
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

        if self.use_fg_prompt:
            # Only initialize these layers if the feature is enabled
            self.fg_prompter = FunctionalGroupPrompt(
                feature_dim=self.graph_dim, num_functional_groups=NUM_FUNCTIONAL_GROUPS
            )
            self.alpha = nn.Parameter(
                self.scaled_fg_factor,
                requires_grad=True,
            )
            logger.info(
                "Functional Group Prompt initialized with alpha: %.4f",
                self.alpha.item(),
            )

        # --- Final Fusion Parameters ---
        self.scale_graph = nn.Parameter(torch.tensor(scaled_graph_factor))
        self.scale_attn = nn.Parameter(torch.tensor(scale_attn_factor))
        if self.use_aux_features:
            self.scale_aux = nn.Parameter(torch.tensor(scale_aux_factor))
        else:
            self.register_buffer("scale_aux", torch.tensor(0.0))

        # --- Prediction Head (for Fusion Mode) ---
        fusion_dropout_1 = config.get("fusion_dropout_1", 0.3)
        fusion_dropout_2 = config.get("fusion_dropout_2", 0.2)

        fusion_dim = self.graph_dim + self.embedding_dim
        if self.use_aux_features:
            fusion_dim += self.aux_feature_dim

        # --- ADDED: GATING LAYER ---
        self.fusion_gate = nn.Linear(fusion_dim, fusion_dim)
        # ---------------------------

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
        smiles_batch,
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
        token_to_graph_weights = None

        # --- Co-Attention Dialogue Loop ---
        for i in range(len(self.cross_attn)):
            attn_out, g2t_weights = self.cross_attn[i](
                query=graph_queries,
                key=embedding_proj,
                value=embedding_proj,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            if i == 0:
                graph_to_token_weights = g2t_weights

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

            if self.use_co_attention:
                c2g_out, t2g_weights = self.co_attn[i](
                    query=embedding_proj,
                    key=graph_queries,
                    value=graph_queries,
                    need_weights=True,
                    average_attn_weights=False,
                )
                if i == 0:
                    token_to_graph_weights = t2g_weights

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

        prompted_graph_repr = graph_repr
        prompt_attn_weights = None

        if self.use_fg_prompt:
            # Generate functional group prompts
            fg_prompt_tensor, prompt_attn_weights = self.fg_prompter(
                smiles_batch=smiles_batch,  # <--- PASS THE NEW ARGUMENT
                atom_features=atom_features,
                bmg=bmg,  # <--- PASS THE BMG OBJECT
                return_attn=return_attn,
            )

            # Scale the prompts by alpha
            fg_prompt_tensor = fg_prompt_tensor * self.alpha

            if torch.isnan(fg_prompt_tensor).any():
                logger.error("Functional Group Prompt contains NaNs!")
                fg_prompt_tensor = torch.nan_to_num(fg_prompt_tensor, nan=0.0)
            # Add the prompts to the graph representation
            prompted_graph_repr = graph_repr + fg_prompt_tensor

        tensors_to_fuse = [
            self.scale_graph * prompted_graph_repr,
            self.scale_attn * attn_output,
        ]
        if self.use_aux_features:
            tensors_to_fuse.append(self.scale_aux * aux_feats)

        raw_fused = torch.cat(tensors_to_fuse, dim=1)

        # --- APPLY GATING MECHANISM ---
        gate = torch.sigmoid(self.fusion_gate(raw_fused))
        fused = raw_fused * gate  # Element-wise multiplication
        # ----------------------------

        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e3, neginf=-1e3)
        if torch.isnan(fused).any():
            logger.error("Fused tensor contains NaNs BEFORE MLP!")

        # --- Prediction ---
        logits = self.output(self.fusion_mlp(fused))

        if torch.isnan(logits).any():
            logger.error(
                "Logits contain NaNs. Investigate input or network instability."
            )
            raise ValueError("Output logits contain NaNs")

        if return_attn:
            return (
                logits,
                attn_entropy_loss,
                token_prior_loss,
                graph_to_token_weights,
                token_to_graph_weights,
                attn_output,
                graph_repr,
                atom_features,
                prompt_attn_weights,
            )
        else:
            return (
                logits,
                attn_entropy_loss,
                token_prior_loss,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    def initialize_modality_scalers(self, data_loader, device):
        """
        Calculates the initial norms of each modality and sets the scalers
        such that the initial effective scaled norm is 1.0 for each.
        This should be called once before training begins.
        """
        logger.info("Initializing modality scalers for balanced contribution...")
        self.eval()

        try:
            from cage_fusion.engine.utils import move_bmg_to_device

            bmg, sequence_embeddings, attn_mask, aux_feats, _, input_ids_batch, _ = (
                next(iter(data_loader))
            )
        except StopIteration:
            logger.warning("Data loader is empty, cannot initialize scalers.")
            return

        bmg = move_bmg_to_device(bmg, device)
        sequence_embeddings = sequence_embeddings.to(device)
        attn_mask = attn_mask.to(device)
        aux_feats = aux_feats.to(device)
        input_ids_batch = input_ids_batch.to(device)

        with torch.no_grad():
            _, _, _, _, _, attn_output, graph_repr, _ = self.forward(
                bmg=bmg,
                sequence_embeddings=sequence_embeddings,
                attn_mask=attn_mask,
                aux_feats=aux_feats,
                input_ids_batch=input_ids_batch,
                return_attn=True,
            )

            norm_graph = graph_repr.norm(p=2, dim=1).mean()
            norm_attn = attn_output.norm(p=2, dim=1).mean()

            epsilon = 1e-8

            self.scale_graph.data.fill_(1.0 / (norm_graph + epsilon))
            self.scale_attn.data.fill_(1.0 / (norm_attn + epsilon))

            logger.info("Scaler initialization complete:")
            logger.info(
                f"  Initial Graph Norm: {norm_graph:.4f} -> New Scaler: {self.scale_graph.item():.4f}"
            )
            logger.info(
                f"  Initial Attn Norm: {norm_attn:.4f} -> New Scaler: {self.scale_attn.item():.4f}"
            )

            if self.use_aux_features:
                norm_aux = aux_feats.norm(p=2, dim=1).mean()
                self.scale_aux.data.fill_(1.0 / (norm_aux + epsilon))
                logger.info(
                    f"  Initial Aux Norm: {norm_aux:.4f} -> New Scaler: {self.scale_aux.item():.4f}"
                )

        self.train()
