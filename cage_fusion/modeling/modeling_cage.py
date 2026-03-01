"""
cage_fusion/modeling/modeling_cage.py
=======================================
CAGEFusion model family.

Class hierarchy
---------------
::

    CAGEFusionPreTrainedModel          <- save/from_pretrained helpers
        ├── CAGEFusionModel            <- backbone encoder  (no task head)
        ├── CAGEFusionForMultiLabelClassification  <- BCEWithLogits head
        └── CAGEFusionForRegression                <- MSE head

The backbone (:class:`CAGEFusionModel`) is intentionally head-free so that
pre-trained weights can be transferred to any task head without key mismatches.

Quick start
-----------
**Multi-label classification (e.g. nuisance compound detection)**::

    from cage_fusion import CageFusionConfig, CAGEFusionForMultiLabelClassification

    config = CageFusionConfig(num_labels=4, label_names=["PAINS", "Aggregator"])
    model  = CAGEFusionForMultiLabelClassification(config)

    # Load from a checkpoint directory:
    model = CAGEFusionForMultiLabelClassification.from_pretrained("checkpoints/my_run")

**Regression (ADMET values)**::

    from cage_fusion import CageFusionConfig, CAGEFusionForRegression

    config = CageFusionConfig(num_labels=12, model_task="regression",
                              label_names=["logP", "solubility"])
    model  = CAGEFusionForRegression(config)

**Backbone only** (feature extraction / fine-tuning)::

    from cage_fusion import CAGEFusionModel
    backbone = CAGEFusionModel.from_pretrained("checkpoints/my_run")
    hidden   = backbone(bmg, sequence_embeddings, ...).hidden_states  # [B, 128]
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from cage_fusion.configuration.configuration_cage import CageFusionConfig
from cage_fusion.modeling.graph_encoder import MolGraphEncoder
from cage_fusion.modeling.modeling_outputs import (
    CageFusionEncoderOutput,
    CageFusionModelOutput,
)
from cage_fusion.modeling.modules import (
    CoAttentionLayer,
    FunctionalGroupPrompt,
    FusionHead,
    SelfAttentionBlock,
)
from cage_fusion.utils.hf_loader import load_tokenizer, _resolve_pretrained_path

logger = logging.getLogger("cagefusion")


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class CAGEFusionPreTrainedModel(nn.Module):
    """
    Base class providing HuggingFace-style save / load helpers.

    All CAGEFusion model variants inherit from this class.

    Persistence
    -----------
    .. code-block:: python

        # Save weights + config
        model.save_pretrained("my_model_dir/")

        # Reload later
        model = MyCAGEFusionModel.from_pretrained("my_model_dir/")

        # Load from a legacy .pt checkpoint
        model = MyCAGEFusionModel.from_checkpoint("best_model.pt")
    """

    config_class = CageFusionConfig

    def __init__(self, config: CageFusionConfig) -> None:
        super().__init__()
        self.config = config

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save model weights (``pytorch_model.bin``) and config (``config.json``)
        into *save_directory*.

        Args:
            save_directory: Local path. Created if it does not exist.

        Example::

            model.save_pretrained("checkpoints/my_run")
        """
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        weight_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), weight_path)
        logger.info("Saved model to %s", save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: Optional[CageFusionConfig] = None,
        strict: bool = True,
        **kwargs,
    ) -> "CAGEFusionPreTrainedModel":
        """
        Load a model from a local directory.

        The directory should contain ``config.json`` and ``pytorch_model.bin``
        (created by :meth:`save_pretrained`).

        Args:
            pretrained_model_name_or_path: Path to a local checkpoint directory.
            config: Optional pre-built config; loaded from ``config.json`` if omitted.
            strict: Passed to :meth:`~torch.nn.Module.load_state_dict`.
                    Use ``False`` when swapping task heads.
            **kwargs: Forwarded to the model constructor.

        Returns:
            Initialised model with loaded weights.

        Example::

            model = CAGEFusionForMultiLabelClassification.from_pretrained(
                "checkpoints/nuisance_model",
                strict=False,   # ignore head mismatch when fine-tuning
            )
        """
        pretrained_model_name_or_path = _resolve_pretrained_path(pretrained_model_name_or_path)

        if config is None:
            config = CageFusionConfig.from_pretrained(pretrained_model_name_or_path)

        model = cls(config, **kwargs)

        weight_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if os.path.isfile(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(state_dict, strict=strict)
            if missing:
                logger.warning("Missing weights: %s", missing[:5])
            if unexpected:
                logger.warning("Unexpected weights: %s", unexpected[:5])
        else:
            logger.warning(
                "No pytorch_model.bin in '%s'; returning randomly initialised model.",
                pretrained_model_name_or_path,
            )
        return model

    # ── Transfer-learning helpers ─────────────────────────────────────────────

    #: Parameter name prefixes that belong to the task head, not the backbone.
    _HEAD_PREFIXES: tuple = ("classifier.", "regressor.")

    def freeze_backbone(self) -> None:
        """Freeze all encoder weights; leave only the task head trainable.

        Use this during the head-warmup phase of fine-tuning so the pretrained
        backbone is not disturbed before the new head has stabilised.

        Example::

            model = CAGEFusionForRegression.from_pretrained(
                "cage-fusion/cage-fusion-pretrained",
                config=finetune_config, strict=False,
            )
            model.freeze_backbone()   # only regressor.weight / .bias trainable
            trainer.train()           # phase A
            model.unfreeze_backbone() # phase B — full fine-tuning
        """
        frozen = 0
        for name, param in self.named_parameters():
            if not any(name.startswith(p) for p in self._HEAD_PREFIXES):
                param.requires_grad_(False)
                frozen += 1
        logger.info("freeze_backbone: froze %d parameter tensors.", frozen)

    def unfreeze_backbone(self) -> None:
        """Re-enable gradients for all parameters (backbone + head)."""
        for param in self.parameters():
            param.requires_grad_(True)
        logger.info("unfreeze_backbone: all parameters trainable.")

    def save_backbone(self, save_directory: str) -> None:
        """Save only encoder weights (no task head) as ``backbone.bin``.

        Safe to load into any task head variant via
        ``from_pretrained(..., strict=False)``.  Also saves ``config.json``
        so the backbone width / architecture is recorded alongside the weights.

        Args:
            save_directory: Local path. Created if it does not exist.

        Example::

            model.save_backbone("/data-1/cage-fusion-pretrain/checkpoints/best/")
            # produces: backbone.bin  +  config.json

            # Fine-tune on a different task later:
            new_model = CAGEFusionForMultiLabelClassification.from_pretrained(
                "/data-1/cage-fusion-pretrain/checkpoints/best/",
                config=CageFusionConfig(num_labels=4, model_task="classification", ...),
                strict=False,   # head key mismatch is expected
            )
        """
        os.makedirs(save_directory, exist_ok=True)
        backbone_state = {
            k: v for k, v in self.state_dict().items()
            if not any(k.startswith(p) for p in self._HEAD_PREFIXES)
        }
        torch.save(backbone_state, os.path.join(save_directory, "backbone.bin"))
        self.config.save_pretrained(save_directory)
        logger.info(
            "Saved backbone (%d tensors) to %s",
            len(backbone_state), save_directory,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[CageFusionConfig] = None,
        strict: bool = False,
        **kwargs,
    ) -> "CAGEFusionPreTrainedModel":
        """
        Load from a legacy ``.pt`` checkpoint file.

        The file must have been saved with keys ``model_state_dict`` and ``config``.

        Args:
            checkpoint_path: Path to the ``.pt`` file.
            config: Override config instead of reading from the checkpoint.
            strict: Passed to :meth:`~torch.nn.Module.load_state_dict`.
            **kwargs: Forwarded to the model constructor.

        Returns:
            Model loaded with checkpoint weights.

        Example::

            model = CAGEFusionForMultiLabelClassification.from_checkpoint(
                "checkpoints/best_model.pt"
            )
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if config is None:
            raw_cfg = ckpt.get("config", {})
            config = (
                CageFusionConfig.from_dict(raw_cfg)
                if isinstance(raw_cfg, dict)
                else raw_cfg
            )

        model = cls(config, **kwargs)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if missing:
            logger.warning("Missing keys from checkpoint: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys from checkpoint: %s", unexpected[:5])
        return model


# ──────────────────────────────────────────────────────────────────────────────
# Backbone encoder
# ──────────────────────────────────────────────────────────────────────────────

class CAGEFusionModel(CAGEFusionPreTrainedModel):
    """
    CAGEFusion backbone encoder — no task head attached.

    Encodes a molecule through three parallel pathways:

    1. **Graph** — D-MPNN (:class:`~cage_fusion.modeling.graph_encoder.MolGraphEncoder`)
    2. **Sequence** — pre-computed ChemBERTa token embeddings
    3. **Auxiliary** — 217 normalised RDKit physicochemical descriptors

    Pathways are combined via a configurable attention mechanism followed by a
    gated fusion MLP, producing a ``[B, hidden_size]`` tensor (``hidden_states``).
    Task-specific heads are attached on top.

    Attention modes (``config.attn_mode``)
    ----------------------------------------
    - ``"cross"``       — bidirectional graph <-> token cross-attention
    - ``"self_tokens"`` — token self-attention only
    - ``"self_graph"``  — graph-node self-attention only *(default)*
    - ``"self_both"``   — independent self-attention on both, then project

    Args:
        config: :class:`~cage_fusion.configuration.CageFusionConfig`

    Returns:
        :class:`~cage_fusion.modeling.modeling_outputs.CageFusionEncoderOutput`

    Example::

        from cage_fusion import CAGEFusionModel, CageFusionConfig

        config   = CageFusionConfig(num_labels=4)
        backbone = CAGEFusionModel(config)

        enc_out = backbone(
            bmg=bmg,
            sequence_embeddings=token_embs,   # [B, T, D]
            attn_mask=mask,                    # [B, T]
            aux_feats=aux,                     # [B, 217]
            input_ids_batch=input_ids,         # [B, T]
            smiles_batch=smiles_list,
        )
        print(enc_out.hidden_states.shape)     # [B, 128]
    """

    def __init__(self, config: CageFusionConfig) -> None:
        super().__init__(config)

        self.graph_encoder = MolGraphEncoder(config)
        self._build_projections()
        self._build_attention()
        self._build_aux()
        self._build_fusion()
        self._build_fg_prompt()
        self._register_tokenizer_buffers()

        logger.info("CAGEFusionModel initialised (attn_mode=%s)", config.attn_mode)

    # ── Component builders ───────────────────────────────────────────────────

    def _build_projections(self) -> None:
        """Linear projections that align graph and token embedding spaces."""
        cfg = self.config
        self.graph_proj = nn.Sequential(
            nn.Linear(cfg.graph_dim, cfg.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.embedding_dim),
            nn.Dropout(cfg.proj_dropout),
        )
        self.embedding_proj = nn.Sequential(
            nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
            nn.Dropout(cfg.proj_dropout),
        )
        # Used only when attn_mode == "self_both"
        self.both_proj = nn.Sequential(
            nn.Linear(2 * cfg.embedding_dim, cfg.embedding_dim),
            nn.GELU(),
            nn.Dropout(cfg.cross_attn_dropout),
        )

    def _build_attention(self) -> None:
        """Cross-attention and self-attention layers."""
        cfg = self.config
        d, h, dp = cfg.embedding_dim, cfg.num_heads, cfg.cross_attn_dropout

        self.co_attn_layers = nn.ModuleList(
            [CoAttentionLayer(d, h, dp) for _ in range(cfg.co_attention_layers)]
        )
        self.self_attn_tok   = SelfAttentionBlock(d, h, dp)
        self.self_attn_graph = SelfAttentionBlock(d, h, dp)

    def _build_aux(self) -> None:
        """Two-layer MLP for auxiliary RDKit descriptor processing."""
        cfg = self.config
        if cfg.use_aux_features:
            norm = (
                nn.BatchNorm1d(cfg.aux_feature_dim)
                if cfg.norm_type == "batch"
                else nn.LayerNorm(cfg.aux_feature_dim)
            )
            self.aux_mlp = nn.Sequential(
                nn.Linear(cfg.aux_feature_dim, cfg.aux_feature_dim),
                nn.ReLU(), norm,
                nn.Linear(cfg.aux_feature_dim, cfg.aux_feature_dim),
                nn.ReLU(),
            )
            self.scale_aux = nn.Parameter(torch.tensor(cfg.scale_aux_factor))
        else:
            self.aux_mlp = None
            self.register_buffer("scale_aux", torch.tensor(0.0))

    def _build_fusion(self) -> None:
        """Gated multi-modal fusion MLP."""
        cfg = self.config
        self.scale_graph = nn.Parameter(torch.tensor(cfg.scaled_graph_factor))
        self.scale_attn  = nn.Parameter(torch.tensor(cfg.scale_attn_factor))
        self.fusion = FusionHead(
            fusion_dim=cfg.fusion_dim,
            hidden_size=cfg.hidden_size,
            dropout_1=cfg.fusion_dropout_1,
            dropout_2=cfg.fusion_dropout_2,
            norm_type=cfg.norm_type,
            fusion_residual=cfg.fusion_residual,
        )

    def _build_fg_prompt(self) -> None:
        """Optional functional-group chemical prompt module."""
        cfg = self.config
        if cfg.use_fg_prompt:
            from cage_fusion.chemistry.fg_utils import NUM_FUNCTIONAL_GROUPS
            self.fg_prompter = FunctionalGroupPrompt(
                num_functional_groups=NUM_FUNCTIONAL_GROUPS,
                feature_dim=cfg.graph_dim,
            )
            self.alpha = nn.Parameter(torch.tensor(cfg.scaled_fg_factor))
        else:
            self.fg_prompter = None
            self.alpha = None

    def _register_tokenizer_buffers(self) -> None:
        """Store special-token IDs as non-trainable buffers."""
        tok = load_tokenizer(self.config.model_checkpoint)
        self.register_buffer("PAD_TOKEN_ID",
                             torch.tensor(tok.pad_token_id, dtype=torch.long))
        self.register_buffer("CLS_TOKEN_ID",
                             torch.tensor(tok.cls_token_id, dtype=torch.long))
        self.register_buffer("SEP_TOKEN_ID",
                             torch.tensor(tok.sep_token_id, dtype=torch.long))
        self.register_buffer("token_importance_prior", None)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        bmg,
        sequence_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        aux_feats: Optional[torch.Tensor] = None,
        input_ids_batch: Optional[torch.Tensor] = None,
        smiles_batch: Optional[List[str]] = None,
        return_attn: bool = False,
    ) -> CageFusionEncoderOutput:
        """
        Encode a batch of molecules.

        Args:
            bmg: :class:`~chemprop.data.BatchMolGraph` on the correct device.
            sequence_embeddings: Pre-computed token embeddings ``[B, T, D]``.
            attn_mask: Attention mask ``[B, T]`` (1 attend / 0 ignore).
            aux_feats: Normalised RDKit descriptors ``[B, aux_feature_dim]``.
            input_ids_batch: Token IDs ``[B, T]`` for masking special tokens.
            smiles_batch: Raw SMILES strings, required when ``use_fg_prompt=True``.
            return_attn: If ``True``, populate attention weight fields in the output.

        Returns:
            :class:`~cage_fusion.modeling.modeling_outputs.CageFusionEncoderOutput`
            Always contains ``hidden_states [B, hidden_size]``.
            Interpretability fields are ``None`` unless ``return_attn=True``.
        """
        cfg = self.config
        dev = next(self.parameters()).device

        # 1 — Graph encoding
        atom_features, graph_repr = self.graph_encoder(bmg)      # [N,D], [B,D]

        # 2 — Optional functional-group prompt
        prompted_graph_repr = graph_repr
        prompt_attn_weights = None
        if cfg.use_fg_prompt and self.fg_prompter is not None and smiles_batch:
            from cage_fusion.chemistry.fg_utils import get_functional_groups
            fg_prompt, prompt_attn_weights = self.fg_prompter(
                smiles_batch=smiles_batch,
                atom_features=atom_features,
                bmg=bmg,
                fg_detector=get_functional_groups,
                return_attn=return_attn,
            )
            fg_prompt = torch.nan_to_num(fg_prompt * self.alpha, nan=0.0)
            prompted_graph_repr = graph_repr + fg_prompt

        graph_part = self.scale_graph * prompted_graph_repr       # [B, graph_dim]

        # 3 — Attention pathway
        attn_output, g2t_weights, t2g_weights = self._attention_pathway(
            atom_features, graph_part, sequence_embeddings,
            attn_mask, input_ids_batch, bmg, return_attn,
        )

        # 4 — Auxiliary features
        if aux_feats is not None and cfg.use_aux_features and self.aux_mlp is not None:
            if aux_feats.size(1) != cfg.aux_feature_dim:
                raise ValueError(
                    f"aux_feats dim {aux_feats.size(1)} != expected {cfg.aux_feature_dim}"
                )
            aux_part = self.scale_aux * self.aux_mlp(aux_feats)
        else:
            aux_part = torch.zeros(graph_part.size(0), cfg.aux_feature_dim, device=dev)

        attn_part = (
            self.scale_attn * attn_output
            if cfg.use_co_attention
            else torch.zeros_like(attn_output)
        )

        # 5 — Fusion MLP
        hidden_states = self.fusion(graph_part, attn_part, aux_part)  # [B, hidden]

        return CageFusionEncoderOutput(
            hidden_states=hidden_states,
            attn_entropy_loss=torch.tensor(0.0, device=dev),
            token_prior_loss=torch.tensor(0.0, device=dev),
            # graph_repr and attn_output are compact [B, dim] tensors always needed
            # for norm tracking; only gate the large weight matrices behind return_attn.
            graph_repr=graph_repr,
            attn_output=attn_output,
            graph_to_token_weights=g2t_weights  if return_attn else None,
            token_to_graph_weights=t2g_weights  if return_attn else None,
            atom_features=atom_features         if return_attn else None,
            prompt_attn_weights=prompt_attn_weights,
        )

    # ── Attention routing ────────────────────────────────────────────────────

    def _attention_pathway(
        self, atom_features, graph_part, sequence_embeddings,
        attn_mask, input_ids_batch, bmg, return_attn,
    ):
        """Dispatch to the correct attention implementation based on attn_mode."""
        cfg  = self.config
        B    = graph_part.size(0)
        zero = torch.zeros(B, cfg.embedding_dim, device=graph_part.device)

        if (
            not cfg.use_co_attention
            or sequence_embeddings is None
            or not cfg.use_embedding_proj
        ):
            return zero, None, None

        mode = cfg.attn_mode
        if mode == "cross":
            return self._cross_attention(
                atom_features, sequence_embeddings, attn_mask,
                input_ids_batch, bmg, return_attn,
            )
        elif mode == "self_tokens":
            return self._self_attn_tokens(sequence_embeddings, attn_mask, input_ids_batch)
        elif mode == "self_graph":
            return self._self_attn_graph(atom_features, bmg)
        elif mode == "self_both":
            return self._self_attn_both(
                atom_features, sequence_embeddings, attn_mask, input_ids_batch, bmg
            )
        return zero, None, None

    def _pad_atoms(self, atom_features: torch.Tensor, bmg):
        """Pad per-molecule atom sequences into [B, max_N, D]."""
        atom_lengths = torch.bincount(bmg.batch)
        segments = torch.split(atom_features, atom_lengths.tolist())
        padded   = pad_sequence(segments, batch_first=True, padding_value=0.0)
        return padded, atom_lengths

    def _cross_attention(
        self, atom_features, sequence_embeddings, attn_mask,
        input_ids_batch, bmg, return_attn,
    ):
        embedding_proj = self.embedding_proj(sequence_embeddings)

        # Only mask PAD tokens as cross-attention keys.
        # CLS carries global sequence context and SEP marks sentence boundaries —
        # both are valid keys for graph nodes to attend to.
        # Masking them was unnecessary and risked all-masked softmax → NaN.
        key_padding_mask = (attn_mask == 0)  # True = ignore (PAD only)

        # Zero out PAD positions in the value stream so they contribute nothing.
        mask_pad = (input_ids_batch == self.PAD_TOKEN_ID).unsqueeze(-1)
        embedding_proj = embedding_proj.masked_fill(mask_pad, 0.0)

        padded, atom_lengths = self._pad_atoms(atom_features, bmg)
        graph_queries = self.graph_proj(padded)

        g2t_weights = t2g_weights = None
        for i, layer in enumerate(self.co_attn_layers):
            graph_queries, embedding_proj, gw, tw = layer(
                graph_queries=graph_queries,
                embedding_proj=embedding_proj,
                key_padding_mask=key_padding_mask,
                need_weights=return_attn,
            )
            if i == 0:
                g2t_weights, t2g_weights = gw, tw

        unpadded = [graph_queries[j, :l] for j, l in enumerate(atom_lengths)]
        flat = torch.cat(unpadded, dim=0)
        attn_output = self.graph_encoder._agg(flat, bmg.batch)
        return attn_output, g2t_weights, t2g_weights

    def _self_attn_tokens(self, sequence_embeddings, attn_mask, input_ids_batch):
        embedding_proj = self.embedding_proj(sequence_embeddings)
        pad_mask = input_ids_batch == self.PAD_TOKEN_ID
        if attn_mask is not None:
            pad_mask = pad_mask | (attn_mask == 0)
        embedding_proj = embedding_proj.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        out, weights = self.self_attn_tok(
            x=embedding_proj, key_padding_mask=pad_mask, need_weights=True
        )
        out = out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        token_counts = (~pad_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
        attn_output  = out.sum(dim=1) / token_counts
        return attn_output, weights, None

    def _self_attn_graph(self, atom_features, bmg):
        padded, atom_lengths = self._pad_atoms(atom_features, bmg)
        graph_queries = self.graph_proj(padded)

        max_nodes = graph_queries.size(1)
        node_idx  = torch.arange(max_nodes, device=graph_queries.device).unsqueeze(0)
        graph_pad_mask = node_idx >= atom_lengths.unsqueeze(1)
        graph_queries  = graph_queries.masked_fill(graph_pad_mask.unsqueeze(-1), 0.0)

        out, weights = self.self_attn_graph(
            x=graph_queries, key_padding_mask=graph_pad_mask, need_weights=True
        )
        out = out.masked_fill(graph_pad_mask.unsqueeze(-1), 0.0)
        valid_counts = (~graph_pad_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
        attn_output  = out.sum(dim=1) / valid_counts
        return attn_output, None, weights

    def _self_attn_both(
        self, atom_features, sequence_embeddings, attn_mask, input_ids_batch, bmg
    ):
        # Token branch
        embedding_proj = self.embedding_proj(sequence_embeddings)
        pad_mask = input_ids_batch == self.PAD_TOKEN_ID
        if attn_mask is not None:
            pad_mask = pad_mask | (attn_mask == 0)
        embedding_proj = embedding_proj.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        tok_out, tok_w = self.self_attn_tok(
            x=embedding_proj, key_padding_mask=pad_mask, need_weights=True
        )
        tok_out = tok_out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        tok_counts  = (~pad_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
        tok_pooled  = tok_out.sum(dim=1) / tok_counts

        # Graph branch
        padded, atom_lengths = self._pad_atoms(atom_features, bmg)
        graph_queries = self.graph_proj(padded)
        max_nodes = graph_queries.size(1)
        node_idx  = torch.arange(max_nodes, device=graph_queries.device).unsqueeze(0)
        graph_pad_mask = node_idx >= atom_lengths.unsqueeze(1)
        graph_queries  = graph_queries.masked_fill(graph_pad_mask.unsqueeze(-1), 0.0)
        g_out, g_w = self.self_attn_graph(
            x=graph_queries, key_padding_mask=graph_pad_mask, need_weights=True
        )
        g_out = g_out.masked_fill(graph_pad_mask.unsqueeze(-1), 0.0)
        valid_counts = (~graph_pad_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
        graph_pooled = g_out.sum(dim=1) / valid_counts

        both        = torch.cat([tok_pooled, graph_pooled], dim=-1)
        attn_output = self.both_proj(both)
        return attn_output, tok_w, g_w


# ──────────────────────────────────────────────────────────────────────────────
# Task heads
# ──────────────────────────────────────────────────────────────────────────────

class CAGEFusionForMultiLabelClassification(CAGEFusionPreTrainedModel):
    """
    CAGEFusion with a multi-label binary classification head.

    Suitable for nuisance-compound detection, toxicity prediction, or any
    scenario requiring independent sigmoid predictions per label.

    Loss (when ``labels`` are provided):
    ``BCEWithLogitsLoss`` + optional attention regularisation.

    Args:
        config: ``config.num_labels`` sets the number of binary output tasks.

    Example::

        config = CageFusionConfig(num_labels=4, label_names=["PAINS_A", "PAINS_B"])
        model  = CAGEFusionForMultiLabelClassification(config)

        out = model(bmg, sequence_embeddings, attn_mask, aux_feats,
                    input_ids_batch, smiles_batch, labels=labels)
        print(out.logits.shape)   # [B, 4]
        print(out.loss)           # scalar
    """

    def __init__(self, config: CageFusionConfig) -> None:
        super().__init__(config)
        self.encoder    = CAGEFusionModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        bmg,
        sequence_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        aux_feats: Optional[torch.Tensor] = None,
        input_ids_batch: Optional[torch.Tensor] = None,
        smiles_batch: Optional[List[str]] = None,
        labels: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        lambda_entropy: float = 0.0,
        lambda_prior: float = 0.0,
        return_attn: bool = False,
    ) -> CageFusionModelOutput:
        enc    = self.encoder(
            bmg=bmg, sequence_embeddings=sequence_embeddings,
            attn_mask=attn_mask, aux_feats=aux_feats,
            input_ids_batch=input_ids_batch, smiles_batch=smiles_batch,
            return_attn=return_attn,
        )
        logits = self.classifier(enc.hidden_states)
        loss   = None
        if labels is not None:
            mask = ~torch.isnan(labels)          # [B, L] — False where label is missing
            if mask.any():
                safe_labels = labels.clone()
                safe_labels[~mask] = 0.0         # NaN → 0 for safe BCE computation
                bce_elem = nn.functional.binary_cross_entropy_with_logits(
                    logits, safe_labels, reduction="none"
                )                                # [B, L]
                bce_loss = (bce_elem * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
            else:
                bce_loss = torch.zeros(1, device=logits.device).squeeze()
            loss = (
                bce_loss
                + lambda_entropy * enc.attn_entropy_loss
                + lambda_prior   * enc.token_prior_loss
            )
        return CageFusionModelOutput(
            logits=logits, loss=loss,
            hidden_states=enc.hidden_states,
            attn_entropy_loss=enc.attn_entropy_loss,
            token_prior_loss=enc.token_prior_loss,
            graph_to_token_weights=enc.graph_to_token_weights,
            token_to_graph_weights=enc.token_to_graph_weights,
            attn_output=enc.attn_output,
            graph_repr=enc.graph_repr,
            atom_features=enc.atom_features,
            prompt_attn_weights=enc.prompt_attn_weights,
        )


class CAGEFusionForRegression(CAGEFusionPreTrainedModel):
    """
    CAGEFusion with a regression head.

    Suitable for continuous property prediction: ADMET values, pIC50,
    solubility, permeability, clearance, etc.

    Loss (when ``labels`` are provided): MSELoss over all targets.

    Args:
        config: ``config.num_labels`` sets the number of regression targets.

    Example::

        config = CageFusionConfig(
            num_labels=12, model_task="regression",
            label_names=["logP", "logD", "solubility", ...],
        )
        model = CAGEFusionForRegression(config)

        out = model(bmg, sequence_embeddings, ..., labels=labels)
        predictions = out.logits   # [B, 12] — continuous values
    """

    def __init__(self, config: CageFusionConfig) -> None:
        super().__init__(config)
        self.encoder   = CAGEFusionModel(config)
        self.regressor = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        bmg,
        sequence_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        aux_feats: Optional[torch.Tensor] = None,
        input_ids_batch: Optional[torch.Tensor] = None,
        smiles_batch: Optional[List[str]] = None,
        labels: Optional[torch.Tensor] = None,
        lambda_entropy: float = 0.0,
        lambda_prior: float = 0.0,
        return_attn: bool = False,
    ) -> CageFusionModelOutput:
        enc         = self.encoder(
            bmg=bmg, sequence_embeddings=sequence_embeddings,
            attn_mask=attn_mask, aux_feats=aux_feats,
            input_ids_batch=input_ids_batch, smiles_batch=smiles_batch,
            return_attn=return_attn,
        )
        predictions = self.regressor(enc.hidden_states)
        loss        = None
        if labels is not None:
            # Masked MSE — supports NaN targets (sparse multi-task labels).
            # Only positions where the label is finite contribute to the loss.
            mask = ~torch.isnan(labels)
            if mask.any():
                diff = (predictions - labels.nan_to_num(0.0)) ** 2
                mse  = (diff * mask).sum() / mask.sum().clamp(min=1)
            else:
                mse  = predictions.sum() * 0.0   # no valid targets in batch
            loss = (
                mse
                + lambda_entropy * enc.attn_entropy_loss
                + lambda_prior   * enc.token_prior_loss
            )
        return CageFusionModelOutput(
            logits=predictions, loss=loss,
            hidden_states=enc.hidden_states,
            attn_entropy_loss=enc.attn_entropy_loss,
            token_prior_loss=enc.token_prior_loss,
            graph_to_token_weights=enc.graph_to_token_weights,
            token_to_graph_weights=enc.token_to_graph_weights,
            attn_output=enc.attn_output,
            graph_repr=enc.graph_repr,
            atom_features=enc.atom_features,
            prompt_attn_weights=enc.prompt_attn_weights,
        )
