import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from chemprop.nn.message_passing import BondMessagePassing
from chemprop.nn.agg import AttentiveAggregation, MeanAggregation
from chemprop.nn.predictors import BinaryClassificationFFN
from chemprop.models.model import MPNN
from torch.nn.utils.rnn import pad_sequence

class CAGEFusionModel(nn.Module):
    """
    CAGEFusionModel (Co-Attention Graph Embedding) for predicting molecular properties.
    """
    def __init__(self, config):
        super().__init__()
        
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
        
        tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
        self.register_buffer("PAD_TOKEN_ID", torch.tensor(tokenizer.pad_token_id, dtype=torch.long))
        self.register_buffer("CLS_TOKEN_ID", torch.tensor(tokenizer.cls_token_id, dtype=torch.long))
        self.register_buffer("SEP_TOKEN_ID", torch.tensor(tokenizer.sep_token_id, dtype=torch.long))

        if config.get("token_importance_prior") is not None:
            self.register_buffer("token_importance_prior", config["token_importance_prior"])
        else:
            self.register_buffer("token_importance_prior", None)

        self.message_passing = BondMessagePassing()
        self.global_aggregation = AttentiveAggregation(input_size=self.graph_dim, output_size=self.graph_dim)
        dummy_predictor = BinaryClassificationFFN(input_dim=self.graph_dim, n_tasks=self.num_tasks, hidden_dim=128, n_layers=1, dropout=0.1, activation="ReLU")
        self.encoder = MPNN(message_passing=self.message_passing, agg=self.global_aggregation, predictor=dummy_predictor)
        
        self.attention_aggregation = MeanAggregation()
        self.graph_proj = nn.Sequential(nn.Linear(self.graph_dim, self.embedding_dim), nn.GELU(), nn.LayerNorm(self.embedding_dim), nn.Dropout(self.proj_dropout))
        self.embedding_proj = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Dropout(self.proj_dropout))
        
        self.cross_attn = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.num_heads, dropout=self.cross_attn_dropout, batch_first=True) for _ in range(2)])
        self.co_attn = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.num_heads, dropout=self.cross_attn_dropout, batch_first=True) for _ in range(2)])
        self.gate_graph = nn.ModuleList([nn.Linear(2 * self.embedding_dim, self.embedding_dim) for _ in range(2)])
        self.gate_embedding = nn.ModuleList([nn.Linear(2 * self.embedding_dim, self.embedding_dim) for _ in range(2)])
        self.cross_attn_norms_attn = nn.ModuleList([nn.LayerNorm(self.embedding_dim) for _ in range(2)])
        self.cross_attn_norms_ffn = nn.ModuleList([nn.LayerNorm(self.embedding_dim) for _ in range(2)])
        self.cross_attn_ffn = nn.ModuleList([nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.GELU(), nn.Dropout(self.cross_attn_dropout), nn.Linear(self.embedding_dim, self.embedding_dim)) for _ in range(2)])
        
        self.scale_graph = nn.Parameter(torch.tensor(1.0))
        self.scale_attn = nn.Parameter(torch.tensor(0.1))
        self.scale_aux = nn.Parameter(torch.tensor(0.1))
        
        fusion_dim = self.graph_dim + self.embedding_dim + self.aux_feature_dim
        self.fusion_mlp = nn.Sequential(nn.Linear(fusion_dim, 768), nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(0.3), nn.Linear(768, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(0.2), nn.Linear(384, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.output = nn.Linear(128, self.num_tasks)

    def forward(self, bmg, embedding_tokens, attn_mask, aux_feats, input_ids_batch, return_attn=False):

        embedding_proj = self.embedding_proj(embedding_tokens)
        padding_token_mask_for_emb = ((input_ids_batch == self.PAD_TOKEN_ID) | (input_ids_batch == self.CLS_TOKEN_ID)).unsqueeze(-1).expand_as(embedding_proj)
        embedding_proj = embedding_proj.masked_fill(padding_token_mask_for_emb, 0.0)
        
        special_token_ids_to_mask = [self.CLS_TOKEN_ID, self.PAD_TOKEN_ID, self.SEP_TOKEN_ID]
        explicit_special_token_mask = torch.zeros_like(input_ids_batch, dtype=torch.bool)
        for token_id in special_token_ids_to_mask:
            explicit_special_token_mask |= (input_ids_batch == token_id)
        key_padding_mask = (attn_mask == 0) | explicit_special_token_mask
        
        atom_features = self.encoder.message_passing(bmg)
        graph_repr = self.encoder.agg(atom_features, bmg.batch)
        
        if self.use_atom_level_queries:
            atom_lengths = torch.bincount(bmg.batch)
            unpadded_atom_features = torch.split(atom_features, atom_lengths.tolist())
            padded_atom_features = pad_sequence(unpadded_atom_features, batch_first=True, padding_value=0.0)
            graph_queries = self.graph_proj(padded_atom_features)
            x = graph_queries
        else: 
            x = self.graph_proj(graph_repr).unsqueeze(1)
        
        attn_weights_final, attn_entropy_loss, token_prior_loss = None, 0.0, 0.0
        for i in range(2):
            attn_out, attn_weights = self.cross_attn[i](x, embedding_proj, embedding_proj, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=False)
            if i == 0: attn_weights_final = attn_weights

            with torch.no_grad():
                attn_log = torch.log(attn_weights + 1e-8)
                entropy = torch.sum(attn_weights * attn_log, dim=-1)
                attn_entropy_loss += -entropy.mean()
                if self.token_importance_prior is not None:
                    prior_scores = self.token_importance_prior[input_ids_batch].unsqueeze(1).unsqueeze(1).expand_as(attn_weights)
                    token_prior_loss += -(attn_weights * prior_scores).sum(dim=-1).mean()

            if self.use_advanced_features:
                c2g_attn_out, _ = self.co_attn[i](embedding_proj, x, x)
                gate_graph = torch.sigmoid(self.gate_graph[i](torch.cat([x, attn_out], dim=-1)))
                x = self.cross_attn_norms_attn[i](x + gate_graph * attn_out)
                gate_embedding = torch.sigmoid(self.gate_embedding[i](torch.cat([embedding_proj, c2g_attn_out], dim=-1)))
                embedding_proj = self.cross_attn_norms_attn[i](embedding_proj + gate_embedding * c2g_attn_out)
            else:
                x = self.cross_attn_norms_attn[i](x + attn_out)
            
            x = self.cross_attn_norms_ffn[i](x + self.cross_attn_ffn[i](x))

        if self.use_atom_level_queries:
            unpadded_x_list = [x[i, :length] for i, length in enumerate(atom_lengths)]
            flattened_x = torch.cat(unpadded_x_list, dim=0)
            attn_output = self.attention_aggregation(flattened_x, bmg.batch)
        else:
            attn_output = x.squeeze(1)
        
        fused = torch.cat([self.scale_graph * graph_repr, self.scale_attn * attn_output, self.scale_aux * aux_feats], dim=1)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e3, neginf=-1e3)
        logits = self.output(self.fusion_mlp(fused))

        if torch.isnan(logits).any(): raise ValueError("Output logits contain NaNs")
        
        if return_attn:
            # Return additional tensors for debugging and visualization
            return (logits, attn_entropy_loss, token_prior_loss, attn_weights_final, attn_output, graph_repr)
        else:
            return (logits, attn_entropy_loss, token_prior_loss)
