"""
Unit tests for CageFusionConfig.
"""

import json
import os
import tempfile

import pytest

from cage_fusion.configuration import CageFusionConfig


class TestCageFusionConfig:

    def test_defaults(self):
        cfg = CageFusionConfig()
        assert cfg.graph_dim == 300
        assert cfg.embedding_dim == 384
        assert cfg.aux_feature_dim == 217
        assert cfg.hidden_size == 128
        assert cfg.num_labels == 4
        assert cfg.attn_mode == "cross"
        assert cfg.model_task == "classification"
        assert cfg.label_names is None

    def test_fusion_dim_property(self):
        cfg = CageFusionConfig(graph_dim=100, embedding_dim=200, aux_feature_dim=50)
        assert cfg.fusion_dim == 350

    def test_to_dict_round_trip(self):
        cfg = CageFusionConfig(num_labels=8, hidden_size=64, attn_mode="self_both")
        d = cfg.to_dict()
        assert d["num_labels"] == 8
        assert d["hidden_size"] == 64
        assert d["attn_mode"] == "self_both"

        cfg2 = CageFusionConfig.from_dict(d)
        assert cfg2.num_labels == 8
        assert cfg2.hidden_size == 64
        assert cfg2.attn_mode == "self_both"

    def test_from_dict_legacy_num_tasks(self):
        """Old checkpoints use 'num_tasks' instead of 'num_labels'."""
        d = {"num_tasks": 6}
        cfg = CageFusionConfig.from_dict(d)
        assert cfg.num_labels == 6

    def test_save_and_load_pretrained(self, tmp_path):
        cfg = CageFusionConfig(
            num_labels=2,
            model_task="regression",
            label_names=["logP", "solubility"],
        )
        cfg.save_pretrained(str(tmp_path))
        assert (tmp_path / "config.json").exists()

        cfg2 = CageFusionConfig.from_pretrained(str(tmp_path))
        assert cfg2.num_labels == 2
        assert cfg2.model_task == "regression"
        assert cfg2.label_names == ["logP", "solubility"]

    def test_from_dict_legacy_tasks_key(self):
        """Old checkpoints carry 'tasks' list — should map to label_names."""
        d = {"num_labels": 3, "tasks": ["A", "B", "C"]}
        cfg = CageFusionConfig.from_dict(d)
        assert cfg.label_names == ["A", "B", "C"]

    def test_post_init_label_names_length(self):
        with pytest.raises(ValueError, match="label_names"):
            CageFusionConfig(num_labels=2, label_names=["only_one"])

    def test_post_init_invalid_model_task(self):
        with pytest.raises(ValueError, match="model_task"):
            CageFusionConfig(model_task="unknown")  # type: ignore

    def test_repr(self):
        cfg = CageFusionConfig(num_labels=4, model_task="classification")
        r = repr(cfg)
        assert "num_labels=4" in r
        assert "model_task='classification'" in r

    def test_unknown_keys_ignored(self):
        """from_dict should silently ignore unknown keys from old configs."""
        d = {"unknown_future_key": 999, "num_labels": 3}
        cfg = CageFusionConfig.from_dict(d)
        assert cfg.num_labels == 3
        assert not hasattr(cfg, "unknown_future_key")
