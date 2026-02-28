"""
Co-attention layer and self-attention helpers.

A single ``CoAttentionLayer`` implements one round of bidirectional
graph ↔ token cross-attention with gated residual connections and a
position-wise FFN, matching the architecture described in the original
cage_fusion paper.

Self-attention variants (``SelfAttentionBlock``) are used when the model
is configured with ``attn_mode`` in {``self_tokens``, ``self_graph``,
``self_both``}.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CoAttentionLayer(nn.Module):
    """
    One layer of bidirectional graph-node ↔ token cross-attention.

    Graph nodes attend to token embeddings (query=graph, key/value=tokens)
    and tokens attend to graph nodes (query=tokens, key/value=graph) in
    the same layer.  Both updates use a sigmoid gate before the residual
    addition, then a shared LayerNorm and a position-wise FFN.

    Parameters
    ----------
    embedding_dim:
        Hidden dimension for both graph and token representations.
    num_heads:
        Number of attention heads.
    dropout:
        Dropout applied inside ``MultiheadAttention``.
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.co_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.gate_graph = nn.Linear(2 * embedding_dim, embedding_dim)
        self.gate_embedding = nn.Linear(2 * embedding_dim, embedding_dim)

        self.attn_norm = nn.LayerNorm(embedding_dim)
        self.ffn_norm = nn.LayerNorm(embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        graph_queries: torch.Tensor,     # [B, N_atoms_padded, D]
        embedding_proj: torch.Tensor,    # [B, T, D]
        key_padding_mask: torch.Tensor,  # [B, T] – True = ignore position
        need_weights: bool = False,
    ):
        """
        Returns
        -------
        graph_queries:
            Updated graph node tensor ``[B, N_atoms_padded, D]``.
        embedding_proj:
            Updated token embedding tensor ``[B, T, D]``.
        g2t_weights:
            Graph-to-token attention weights ``[B, heads, N_atoms_padded, T]``
            or ``None`` when ``need_weights=False``.
        t2g_weights:
            Token-to-graph attention weights ``[B, heads, T, N_atoms_padded]``
            or ``None`` when ``need_weights=False``.
        """
        # Graph nodes attend to tokens
        attn_out, g2t_weights = self.cross_attn(
            query=graph_queries,
            key=embedding_proj,
            value=embedding_proj,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )

        # Tokens attend to graph nodes
        c2g_out, t2g_weights = self.co_attn(
            query=embedding_proj,
            key=graph_queries,
            value=graph_queries,
            need_weights=need_weights,
            average_attn_weights=False,
        )

        # Gated residual: graph
        gate_g = torch.sigmoid(
            self.gate_graph(torch.cat([graph_queries, attn_out], dim=-1))
        )
        graph_queries = self.attn_norm(graph_queries + gate_g * attn_out)

        # Gated residual: tokens
        gate_e = torch.sigmoid(
            self.gate_embedding(torch.cat([embedding_proj, c2g_out], dim=-1))
        )
        embedding_proj = self.attn_norm(embedding_proj + gate_e * c2g_out)

        # FFN on graph side
        graph_queries = self.ffn_norm(graph_queries + self.ffn(graph_queries))

        return graph_queries, embedding_proj, g2t_weights, t2g_weights


class SelfAttentionBlock(nn.Module):
    """
    A single-modality self-attention block with residual + FFN.

    Used by the ``self_tokens`` and ``self_graph`` attention modes.

    Parameters
    ----------
    embedding_dim, num_heads, dropout:
        Standard transformer hyper-parameters.
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ):
        """
        Returns
        -------
        x:
            Updated tensor (same shape as input).
        weights:
            Attention weights or ``None`` when ``need_weights=False``.
        """
        out, weights = self.attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = self.norm(x + out)
        x = self.norm(x + self.ffn(x))
        return x, weights
