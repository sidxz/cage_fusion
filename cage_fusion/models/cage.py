import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from chemprop.nn.message_passing import BondMessagePassing
from chemprop.nn.agg import AttentiveAggregation, MeanAggregation
from chemprop.nn.predictors import BinaryClassificationFFN
from chemprop.models.model import MPNN
from torch.nn.utils.rnn import pad_sequence
from cage_fusion.utils.logging_utils import logger
from cage_fusion.models.fg_prompt_addon import FunctionalGroupPrompt
from cage_fusion.engine.fg_utils import NUM_FUNCTIONAL_GROUPS
from cage_fusion.utils.hf_loader import load_hf_checkpoint, load_tokenizer
import json


class CAGEFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

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

        self.config = config
        self.graph_dim = config["graph_dim"]
        self.embedding_dim = config["embedding_dim"]
        self.aux_feature_dim = config["aux_feature_dim"]
        self.num_tasks = config["num_tasks"]
        self.num_heads = config["num_heads"]
        self.cross_attn_dropout = config["cross_attn_dropout"]
        self.proj_dropout = config["proj_dropout"]
        self.scaled_fg_factor = torch.tensor(config.get("scaled_fg_factor", 0.1))

        self.use_co_attention = config.get("use_co_attention", True)
        self.use_aux_features = config.get("use_aux_features", True)
        self.use_fg_prompt = config.get("use_fg_prompt", False)

        # NEW: attention mode toggle (default keeps current behavior)
        self.attn_mode = config.get(
            "attn_mode", "cross"
        )  # cross | self_tokens | self_graph | self_both
        logger.info(f"[]Attention mode set to: {self.attn_mode}")

        #tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
        tokenizer = load_tokenizer(config["model_checkpoint"])
        self.register_buffer(
            "PAD_TOKEN_ID", torch.tensor(tokenizer.pad_token_id, dtype=torch.long)
        )
        self.register_buffer(
            "CLS_TOKEN_ID", torch.tensor(tokenizer.cls_token_id, dtype=torch.long)
        )
        self.register_buffer(
            "SEP_TOKEN_ID", torch.tensor(tokenizer.sep_token_id, dtype=torch.long)
        )

        tip = config.get("token_importance_prior")
        if tip is not None:
            self.register_buffer("token_importance_prior", tip)
        else:
            self.register_buffer("token_importance_prior", None)

        self.message_passing = BondMessagePassing()
        # Oct 3 2025: Previous
        # self.global_aggregation = AttentiveAggregation(
        #     input_size=self.graph_dim, output_size=self.graph_dim
        # )
        # New Test with Mean Aggregation
        self.global_aggregation = MeanAggregation()

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

        # NEW: self-attention blocks (used only if attn_mode != "cross")
        self.self_attn_tok = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            dropout=self.cross_attn_dropout,
            batch_first=True,
        )
        self.self_attn_graph = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            dropout=self.cross_attn_dropout,
            batch_first=True,
        )
        self.self_tok_ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.cross_attn_dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.self_tok_norm = nn.LayerNorm(self.embedding_dim)
        self.self_graph_ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.cross_attn_dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        self.self_graph_norm = nn.LayerNorm(self.embedding_dim)
        self.both_proj = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.cross_attn_dropout),
        )

        if self.use_fg_prompt:
            self.fg_prompter = FunctionalGroupPrompt(
                feature_dim=self.graph_dim, num_functional_groups=NUM_FUNCTIONAL_GROUPS
            )
            self.alpha = nn.Parameter(self.scaled_fg_factor, requires_grad=True)

        self.scale_graph = nn.Parameter(
            torch.tensor(config.get("scaled_graph_factor", 1.0))
        )
        self.scale_attn = nn.Parameter(
            torch.tensor(config.get("scale_attn_factor", 1.0))
        )
        if self.use_aux_features:
            self.scale_aux = nn.Parameter(
                torch.tensor(config.get("scale_aux_factor", 1.0))
            )
            # --- Aux features MLP (ADDED) ---
            self.aux_mlp = nn.Sequential(
                nn.Linear(self.aux_feature_dim, self.aux_feature_dim),
                nn.ReLU(),
                # 3 Oct 2025 Previous
                nn.BatchNorm1d(self.aux_feature_dim),
                # New Experiment with layer norm
                # nn.LayerNorm(self.aux_feature_dim),
                nn.Linear(self.aux_feature_dim, self.aux_feature_dim),
                nn.ReLU(),
            )
        else:
            self.register_buffer("scale_aux", torch.tensor(0.0))
            self.aux_mlp = None

        # Always use full fusion_dim regardless of config flags
        fusion_dim = self.graph_dim + self.embedding_dim + self.aux_feature_dim

        self.fusion_gate = nn.Linear(fusion_dim, fusion_dim)
        fusion_dropout_1 = config.get("fusion_dropout_1", 0.3)
        fusion_dropout_2 = config.get("fusion_dropout_2", 0.2)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 768),
            # 3 Oct 2025 Previous
            nn.BatchNorm1d(768),
            # New Experiment with layer norm
            # nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(fusion_dropout_1),
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            # nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(fusion_dropout_2),
            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            # nn.LayerNorm(128),
            nn.ReLU(),
        )
        # Original output layer is linear to num_tasks
        # self.output = nn.Linear(128, self.num_tasks)
        # Test with Predictor MLP Dec 19, 2025
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(config.get("head_dropout", 0.1)),
                nn.Linear(64, 1)
            ) for _ in range(self.num_tasks)
        ])
        
        self.attention_aggregation = MeanAggregation()
        

        logger.info("CAGEFusionModel initialization complete")

    def forward(
        self,
        bmg,
        sequence_embeddings=None,
        attn_mask=None,
        aux_feats=None,
        input_ids_batch=None,
        smiles_batch=None,
        return_attn=False,
    ):
        atom_features = self.encoder.message_passing(bmg)
        graph_repr = self.encoder.agg(atom_features, bmg.batch)

        # Apply functional group prompts if enabled
        prompted_graph_repr = graph_repr
        prompt_attn_weights = None
        if self.use_fg_prompt:
            fg_prompt_tensor, prompt_attn_weights = self.fg_prompter(
                smiles_batch=smiles_batch,
                atom_features=atom_features,
                bmg=bmg,
                return_attn=return_attn,
            )
            fg_prompt_tensor = fg_prompt_tensor * self.alpha
            fg_prompt_tensor = torch.nan_to_num(fg_prompt_tensor, nan=0.0)
            prompted_graph_repr = graph_repr + fg_prompt_tensor

        # Projected graph part (needed for fallback below)
        graph_part = self.scale_graph * prompted_graph_repr

        # Attention output (if co-attention enabled)
        if self.use_co_attention and sequence_embeddings is not None:
            # ====== ORIGINAL CROSS-ATTN PATH (unchanged) ======
            if self.attn_mode == "cross":
                embedding_proj = self.embedding_proj(sequence_embeddings)
                # Previous Oct 3 2025
                mask_pad_cls = (
                    (input_ids_batch == self.PAD_TOKEN_ID)
                    | (input_ids_batch == self.CLS_TOKEN_ID)
                ).unsqueeze(-1)
                embedding_proj = embedding_proj.masked_fill(mask_pad_cls, 0.0)

                special_ids = [self.CLS_TOKEN_ID, self.PAD_TOKEN_ID, self.SEP_TOKEN_ID]
                explicit_special_mask = torch.zeros_like(
                    input_ids_batch, dtype=torch.bool
                )
                for tok in special_ids:
                    explicit_special_mask |= input_ids_batch == tok
                key_padding_mask = (attn_mask == 0) | explicit_special_mask

                atom_lengths = torch.bincount(bmg.batch)
                segments = torch.split(atom_features, atom_lengths.tolist())
                padded = pad_sequence(segments, batch_first=True, padding_value=0.0)
                graph_queries = self.graph_proj(padded)

                attn_entropy_loss = 0.0
                token_prior_loss = 0.0
                graph_to_token_weights = None
                token_to_graph_weights = None

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
                                prior_scores.unsqueeze(1)
                                .unsqueeze(1)
                                .expand_as(g2t_weights)
                            )
                            token_prior_loss += (
                                -(g2t_weights * prior_scores).sum(dim=-1).mean()
                            )

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
                        self.gate_embedding[i](
                            torch.cat([embedding_proj, c2g_out], dim=-1)
                        )
                    )
                    embedding_proj = self.attention_norm_layers[i](
                        embedding_proj + gate_e * c2g_out
                    )
                    graph_queries = self.ffn_norm_layers[i](
                        graph_queries + self.cross_attn_ffn[i](graph_queries)
                    )

                unpadded = [graph_queries[j, :l] for j, l in enumerate(atom_lengths)]
                flat = torch.cat(unpadded, dim=0)
                attn_output = self.attention_aggregation(flat, bmg.batch)

            # ====== NEW: SELF-ATTN PATHS (tokens / graph / both) ======
            else:
                # Common prep for self-attn modes (PAD-only mask; CLS/SEP kept)
                embedding_proj = self.embedding_proj(sequence_embeddings)  # [B,T,D]
                pad_mask = input_ids_batch == self.PAD_TOKEN_ID  # [B,T]
                if attn_mask is not None:
                    pad_mask = pad_mask | (attn_mask == 0)
                embedding_proj = embedding_proj.masked_fill(pad_mask.unsqueeze(-1), 0.0)
                token_keep_counts = (~pad_mask).sum(dim=1).clamp(min=1)  # [B]

                atom_lengths = torch.bincount(bmg.batch)  # [B]
                segments = torch.split(atom_features, atom_lengths.tolist())
                padded_nodes = pad_sequence(
                    segments, batch_first=True, padding_value=0.0
                )  # [B,Nmax,H]
                graph_queries = self.graph_proj(padded_nodes)  # [B,Nmax,D]
                max_nodes = graph_queries.size(1)
                node_idx = torch.arange(
                    max_nodes, device=graph_queries.device
                ).unsqueeze(0)
                graph_pad_mask = node_idx >= atom_lengths.unsqueeze(1)  # [B,Nmax]
                graph_queries = graph_queries.masked_fill(
                    graph_pad_mask.unsqueeze(-1), 0.0
                )

                attn_entropy_loss = 0.0
                token_prior_loss = 0.0
                graph_to_token_weights = None
                token_to_graph_weights = None

                if self.attn_mode == "self_tokens":
                    tok_out, tok_weights = self.self_attn_tok(
                        query=embedding_proj,
                        key=embedding_proj,
                        value=embedding_proj,
                        key_padding_mask=pad_mask,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                    tok_out = self.self_tok_norm(tok_out + self.self_tok_ffn(tok_out))
                    tok_out = tok_out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
                    attn_output = tok_out.sum(dim=1) / token_keep_counts.unsqueeze(-1)
                    graph_to_token_weights = tok_weights

                elif self.attn_mode == "self_graph":
                    g_out, g_weights = self.self_attn_graph(
                        query=graph_queries,
                        key=graph_queries,
                        value=graph_queries,
                        key_padding_mask=graph_pad_mask,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                    g_out = self.self_graph_norm(g_out + self.self_graph_ffn(g_out))
                    g_out = g_out.masked_fill(graph_pad_mask.unsqueeze(-1), 0.0)
                    valid_counts = (~graph_pad_mask).sum(dim=1).clamp(min=1)
                    attn_output = g_out.sum(dim=1) / valid_counts.unsqueeze(-1)
                    token_to_graph_weights = g_weights

                elif self.attn_mode == "self_both":
                    # tokens
                    tok_out, tok_weights = self.self_attn_tok(
                        query=embedding_proj,
                        key=embedding_proj,
                        value=embedding_proj,
                        key_padding_mask=pad_mask,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                    tok_out = self.self_tok_norm(tok_out + self.self_tok_ffn(tok_out))
                    tok_out = tok_out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
                    tok_pooled = tok_out.sum(dim=1) / token_keep_counts.unsqueeze(-1)

                    # graph nodes
                    g_out, g_weights = self.self_attn_graph(
                        query=graph_queries,
                        key=graph_queries,
                        value=graph_queries,
                        key_padding_mask=graph_pad_mask,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                    g_out = self.self_graph_norm(g_out + self.self_graph_ffn(g_out))
                    g_out = g_out.masked_fill(graph_pad_mask.unsqueeze(-1), 0.0)
                    valid_counts = (~graph_pad_mask).sum(dim=1).clamp(min=1)
                    graph_pooled = g_out.sum(dim=1) / valid_counts.unsqueeze(-1)

                    both = torch.cat([tok_pooled, graph_pooled], dim=-1)  # [B,2D]
                    attn_output = self.both_proj(both)  # [B,D]

                    graph_to_token_weights = tok_weights
                    token_to_graph_weights = g_weights

                else:
                    # fallback (shouldn't hit if attn_mode is valid)
                    attn_output = torch.zeros(
                        graph_part.size(0), self.embedding_dim, device=graph_part.device
                    )

        else:
            attn_output = torch.zeros(
                graph_part.size(0), self.embedding_dim, device=graph_part.device
            )
            attn_entropy_loss = 0.0
            token_prior_loss = 0.0
            graph_to_token_weights = None
            token_to_graph_weights = None

        # Validate aux_feat shape
        if aux_feats is not None and aux_feats.size(1) != self.aux_feature_dim:
            raise ValueError(
                f"aux_feats has dim {aux_feats.size(1)}, but model expects {self.aux_feature_dim}"
            )

        # Fixed-size tensors for fusion
        attn_part = (
            self.scale_attn * attn_output
            if self.use_co_attention
            else torch.zeros_like(attn_output)
        )
        if aux_feats is not None:
            if self.use_aux_features and self.aux_mlp is not None:
                aux_part = self.scale_aux * self.aux_mlp(aux_feats)
            else:
                aux_part = torch.zeros_like(aux_feats)
        else:
            aux_part = torch.zeros(
                graph_part.size(0), self.aux_feature_dim, device=graph_part.device
            )

        raw_fused = torch.cat([graph_part, attn_part, aux_part], dim=1)
        gate = torch.sigmoid(self.fusion_gate(raw_fused))
        # Previous Oct 3 2025
        fused = raw_fused * gate
        # New Try with Residual Connection
        #fused = raw_fused * gate + raw_fused
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e3, neginf=-1e3)

        
        # Original output layer is linear to num_tasks
        #logits = self.output(self.fusion_mlp(fused))
        
        # Test with Predictor MLP Dec 19, 2025
        # 1. Generate the shared fused representation
        fused_repr = self.fusion_mlp(fused)
        # 2. Iterate through task-specific heads
        # Each head outputs [Batch, 1], so we concatenate to [Batch, num_tasks]
        logits = torch.cat([head(fused_repr) for head in self.output_heads], dim=1)

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
