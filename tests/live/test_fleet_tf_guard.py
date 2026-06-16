"""tests/live/test_fleet_tf_guard.py — H2: fleet TF-mismatch enforcement.

A model must only trade on the timeframe it was trained on. FleetRouter
asserts, before a cell joins routing, that each cell's declared ``tf`` equals
the timeframe stamped in the model's checkpoint ``.meta.json``
(missing meta → "h1").
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.inspect.base import ModelAdapter
from src.live.fleet import CellSpec, FleetRouter, make_cell_key
from src.live.fleet_state import FleetState
from src.live.logger import FleetSessionLogger
from src.strategy.exits import ExitConfig


class _StubAdapter(ModelAdapter):
    """Adapter exposing a checkpoint_path (for TF resolution) and a flat signal."""

    name = "_tf_stub"

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        n = windows.shape[0]
        out = np.zeros((n, 3), dtype=np.float32)
        out[:, 0] = 0.99
        return out


def _write_ckpt_with_tf(tmp_path: Path, token: str) -> Path:
    """Create a dummy checkpoint file with a meta.json declaring *token*."""
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"\x00")
    (tmp_path / "model.meta.json").write_text(json.dumps({"timeframe": token}))
    return ckpt


def _registry(specs, strategy_configs, fleet_state):
    from src.live.sim_executor import SimFillExecutor

    reg = {}
    for spec in specs:
        key = make_cell_key(spec.ticker, spec.model, spec.strategy)
        reg[key] = SimFillExecutor(
            realism=strategy_configs[spec.strategy], state=fleet_state, cell_key=key
        )
    return reg


def _build(tmp_path, spec_tf: str, meta_tf: str):
    ckpt = _write_ckpt_with_tf(tmp_path, meta_tf)
    adapters = {"_tf_stub": _StubAdapter(ckpt)}
    strategy_configs = {"fixed_2r": ExitConfig(strategy="fixed_2r", fill_mode="optimistic")}
    fleet_state = FleetState(fleet_id="tf_guard", base_dir=tmp_path / "logs")
    logger = FleetSessionLogger(session_id="s", base_dir=str(tmp_path / "logs" / "paper"))
    specs = [
        CellSpec(
            ticker="SPY",
            model="_tf_stub",
            strategy="fixed_2r",
            executor_type="sim",
            tf=spec_tf,
        )
    ]
    registry = _registry(specs, strategy_configs, fleet_state)
    return dict(
        tickers=["SPY"],
        adapters=adapters,
        strategy_configs=strategy_configs,
        executor_registry=registry,
        logger=logger,
        specs=specs,
    )


def test_fleet_cell_tf_mismatch_raises(tmp_path):
    """Cell declares tf=5m but checkpoint meta says h1 → hard error."""
    kwargs = _build(tmp_path, spec_tf="5m", meta_tf="h1")
    with pytest.raises(ValueError, match="trade only on the timeframe"):
        FleetRouter(**kwargs)


def test_fleet_cell_tf_match_ok(tmp_path):
    """Cell tf agrees with checkpoint meta → constructs fine."""
    kwargs = _build(tmp_path, spec_tf="h1", meta_tf="h1")
    router = FleetRouter(**kwargs)
    assert router._cell_tf[make_cell_key("SPY", "_tf_stub", "fixed_2r")] == "h1"


def test_fleet_cell_missing_meta_defaults_h1(tmp_path):
    """No meta sidecar → resolves to h1; an h1 cell is accepted."""
    ckpt = tmp_path / "bare.pt"
    ckpt.write_bytes(b"\x00")  # no meta.json
    adapters = {"_tf_stub": _StubAdapter(ckpt)}
    strategy_configs = {"fixed_2r": ExitConfig(strategy="fixed_2r", fill_mode="optimistic")}
    fleet_state = FleetState(fleet_id="tf_guard", base_dir=tmp_path / "logs")
    logger = FleetSessionLogger(session_id="s", base_dir=str(tmp_path / "logs" / "paper"))
    specs = [
        CellSpec(ticker="SPY", model="_tf_stub", strategy="fixed_2r",
                 executor_type="sim", tf="h1")
    ]
    registry = _registry(specs, strategy_configs, fleet_state)
    router = FleetRouter(
        tickers=["SPY"], adapters=adapters, strategy_configs=strategy_configs,
        executor_registry=registry, logger=logger, specs=specs,
    )
    assert router._cell_tf[make_cell_key("SPY", "_tf_stub", "fixed_2r")] == "h1"
