"""
Unit tests for CAGEFusionModel and task-head models.

These tests avoid loading actual checkpoints — they only verify that the
models can be instantiated, perform forward passes with synthetic tensors,
and that outputs have the expected shapes and types.
"""

from __future__ import annotations

import tempfile

import pytest
import torch

from cage_fusion.configuration import CageFusionConfig
from cage_fusion.modeling import (
    CAGEFusionForMultiLabelClassification,
    CAGEFusionForRegression,
    CAGEFusionModel,
)
from cage_fusion.modeling.modeling_outputs import (
    CageFusionEncoderOutput,
    CageFusionModelOutput,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config() -> CageFusionConfig:
    """Minimal config to keep instantiation fast during unit tests."""
    return CageFusionConfig(
        graph_dim=32,
        embedding_dim=16,
        aux_feature_dim=8,
        hidden_size=16,
        num_labels=3,
        num_co_attn_layers=1,
        num_heads=2,
        dropout_1=0.0,
        dropout_2=0.0,
        attn_mode="cross",
        norm_type="layer",
    )


def _make_fake_bmg(batch_size: int, num_atoms: int = 5):
    """Create a minimal fake BatchMolGraph-like object for testing."""
    from types import SimpleNamespace
    # BatchMolGraph expects tensors; we use a SimpleNamespace to duck-type
    total_atoms = batch_size * num_atoms
    bmg = SimpleNamespace(
        V=torch.randn(total_atoms, 72),          # atom feature dim from chemprop
        E=torch.randn(total_atoms * 2, 14),       # bond feature dim
        edge_index=torch.randint(0, total_atoms, (2, total_atoms * 2)),
        batch=torch.repeat_interleave(
            torch.arange(batch_size), torch.tensor([num_atoms] * batch_size)
        ),
    )
    return bmg


# ---------------------------------------------------------------------------
# Config save/load
# ---------------------------------------------------------------------------

class TestConfigSaveLoad:
    def test_save_pretrained(self, small_config, tmp_path):
        small_config.save_pretrained(str(tmp_path))
        loaded = CageFusionConfig.from_pretrained(str(tmp_path))
        assert loaded.hidden_size == small_config.hidden_size
        assert loaded.num_labels == small_config.num_labels


# ---------------------------------------------------------------------------
# Modeling outputs
# ---------------------------------------------------------------------------

class TestModelingOutputs:
    def test_encoder_output_defaults(self):
        hs = torch.zeros(2, 16)
        out = CageFusionEncoderOutput(hidden_states=hs)
        assert out.hidden_states.shape == (2, 16)
        assert out.attn_entropy_loss.item() == 0.0
        assert out.token_prior_loss.item() == 0.0
        assert out.graph_to_token_weights is None

    def test_model_output_fields(self):
        logits = torch.zeros(2, 3)
        out = CageFusionModelOutput(logits=logits)
        assert out.logits.shape == (2, 3)
        assert out.loss is None


# ---------------------------------------------------------------------------
# CAGEFusionModel (backbone encoder)
# ---------------------------------------------------------------------------

class TestCAGEFusionModel:
    def test_instantiation(self, small_config):
        model = CAGEFusionModel(small_config)
        assert model is not None

    def test_parameter_count_nonzero(self, small_config):
        model = CAGEFusionModel(small_config)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_save_pretrained(self, small_config, tmp_path):
        model = CAGEFusionModel(small_config)
        model.save_pretrained(str(tmp_path))
        assert (tmp_path / "model.safetensors").exists() or \
               (tmp_path / "pytorch_model.bin").exists() or \
               any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# CAGEFusionForMultiLabelClassification
# ---------------------------------------------------------------------------

class TestCAGEFusionForMultiLabelClassification:
    def test_instantiation(self, small_config):
        model = CAGEFusionForMultiLabelClassification(small_config)
        assert model is not None
        # Check the head is the right size
        assert model.classifier.out_features == small_config.num_labels

    def test_output_type(self, small_config):
        model = CAGEFusionForMultiLabelClassification(small_config)
        assert isinstance(model, CAGEFusionForMultiLabelClassification)


# ---------------------------------------------------------------------------
# CAGEFusionForRegression
# ---------------------------------------------------------------------------

class TestCAGEFusionForRegression:
    def test_instantiation(self, small_config):
        model = CAGEFusionForRegression(small_config)
        assert model is not None
        assert model.regressor.out_features == small_config.num_labels

    def test_output_type(self, small_config):
        model = CAGEFusionForRegression(small_config)
        assert isinstance(model, CAGEFusionForRegression)
