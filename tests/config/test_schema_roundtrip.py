"""tests/config/test_schema_roundtrip.py — Round-trip schema tests for new arch YAMLs.

Loads experiments/transformer_g1.yaml and experiments/xlstm_g1.yaml and asserts:
  - ExperimentConfig resolves to the correct model subclass via arch discriminator.
  - Key HP fields survive the YAML → Pydantic round-trip unchanged.
  - model_dump() → model_validate() round-trip produces an identical config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_experiment
from src.config.schema import (
    ExperimentConfig,
    TransformerModelConfig,
    XLSTMModelConfig,
)

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "experiments"


# ---------------------------------------------------------------------------
# Transformer G1
# ---------------------------------------------------------------------------

class TestTransformerG1Schema:
    @pytest.fixture(scope="class")
    def cfg(self) -> ExperimentConfig:
        return load_experiment(EXPERIMENTS / "transformer_g1.yaml")

    def test_model_subclass(self, cfg: ExperimentConfig) -> None:
        assert isinstance(cfg.model, TransformerModelConfig), (
            f"Expected TransformerModelConfig, got {type(cfg.model).__name__}"
        )

    def test_arch_discriminator(self, cfg: ExperimentConfig) -> None:
        assert cfg.model.arch == "transformer"

    def test_d_model(self, cfg: ExperimentConfig) -> None:
        assert cfg.model.d_model == 64

    def test_nhead(self, cfg: ExperimentConfig) -> None:
        assert cfg.model.nhead == 4

    def test_num_layers(self, cfg: ExperimentConfig) -> None:
        assert cfg.model.num_layers == 2

    def test_dim_feedforward(self, cfg: ExperimentConfig) -> None:
        assert cfg.model.dim_feedforward == 128

    def test_dropout(self, cfg: ExperimentConfig) -> None:
        assert cfg.model.dropout == pytest.approx(0.1)

    def test_labeller(self, cfg: ExperimentConfig) -> None:
        assert cfg.data.labeller == "fvg_valid"

    def test_device_cpu(self, cfg: ExperimentConfig) -> None:
        assert cfg.train.device == "cpu"

    def test_model_dump_roundtrip(self, cfg: ExperimentConfig) -> None:
        """model_dump() → model_validate() must produce an equivalent config."""
        dumped = cfg.model_dump()
        restored = ExperimentConfig.model_validate(dumped)
        assert isinstance(restored.model, TransformerModelConfig)
        assert restored.model.d_model == cfg.model.d_model
        assert restored.model.nhead == cfg.model.nhead
        assert restored.model.num_layers == cfg.model.num_layers
        assert restored.model.dim_feedforward == cfg.model.dim_feedforward


# ---------------------------------------------------------------------------
# xLSTM G1
# ---------------------------------------------------------------------------

class TestXLSTMG1Schema:
    @pytest.fixture(scope="class")
    def cfg(self) -> ExperimentConfig:
        return load_experiment(EXPERIMENTS / "xlstm_g1.yaml")

    def test_model_subclass(self, cfg: ExperimentConfig) -> None:
        assert isinstance(cfg.model, XLSTMModelConfig), (
            f"Expected XLSTMModelConfig, got {type(cfg.model).__name__}"
        )

    def test_arch_discriminator(self, cfg: ExperimentConfig) -> None:
        assert cfg.model.arch == "xlstm"

    def test_embedding_dim(self, cfg: ExperimentConfig) -> None:
        assert cfg.model.embedding_dim == 64

    def test_num_blocks(self, cfg: ExperimentConfig) -> None:
        assert cfg.model.num_blocks == 2

    def test_num_heads(self, cfg: ExperimentConfig) -> None:
        assert cfg.model.num_heads == 4

    def test_dropout(self, cfg: ExperimentConfig) -> None:
        assert cfg.model.dropout == pytest.approx(0.1)

    def test_labeller(self, cfg: ExperimentConfig) -> None:
        assert cfg.data.labeller == "fvg_valid"

    def test_device_cpu(self, cfg: ExperimentConfig) -> None:
        assert cfg.train.device == "cpu"

    def test_model_dump_roundtrip(self, cfg: ExperimentConfig) -> None:
        """model_dump() → model_validate() must produce an equivalent config."""
        dumped = cfg.model_dump()
        restored = ExperimentConfig.model_validate(dumped)
        assert isinstance(restored.model, XLSTMModelConfig)
        assert restored.model.embedding_dim == cfg.model.embedding_dim
        assert restored.model.num_blocks == cfg.model.num_blocks
        assert restored.model.num_heads == cfg.model.num_heads


# ---------------------------------------------------------------------------
# Discriminator correctness — inline model_validate (no file I/O)
# ---------------------------------------------------------------------------

class TestDiscriminatorNewArchs:
    def test_transformer_discriminated_union(self) -> None:
        exp = ExperimentConfig.model_validate(
            {"model": {"arch": "transformer", "d_model": 32, "nhead": 2}}
        )
        assert isinstance(exp.model, TransformerModelConfig)
        assert exp.model.d_model == 32

    def test_xlstm_discriminated_union(self) -> None:
        exp = ExperimentConfig.model_validate(
            {"model": {"arch": "xlstm", "embedding_dim": 32, "num_blocks": 1}}
        )
        assert isinstance(exp.model, XLSTMModelConfig)
        assert exp.model.embedding_dim == 32

    def test_transformer_arch_field_immutable(self) -> None:
        cfg = TransformerModelConfig()
        assert cfg.arch == "transformer"

    def test_xlstm_arch_field_immutable(self) -> None:
        cfg = XLSTMModelConfig()
        assert cfg.arch == "xlstm"
