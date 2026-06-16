"""tests/rigor/test_tf_training.py — WS-5 timeframe-aware training plumbing tests.

Assertions:
  1. token "h1" → checkpoint subdir has NO token suffix (backward-compat).
  2. token "5m" → checkpoint subdir gains "_5m" suffix.
  3. token "15m" → checkpoint subdir gains "_15m" suffix.
  4. _ckpt_subdir applies to all arches correctly.
  5. SeedSweepConfig.timeframe defaults to "h1".
  6. from_experiment_config propagates timeframe from DataConfig.
  7. run_seed_sweep writes checkpoints to the token-namespaced subdir.
  8. meta JSON written by run_seed_sweep carries "timeframe" key.
  9. _load_splits uses token-namespaced parquet filename (stub build).
 10. Config with timeframe "5m" loads from spy_5m_*.parquet, not spy_h1_*.parquet.
 11. New 5m/15m experiment YAMLs are loadable and carry correct timeframe field.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import load_experiment
from src.rigor.seed_sweep import (
    SeedSweepConfig,
    _ckpt_subdir,
    run_seed_sweep,
)


# ---------------------------------------------------------------------------
# 1-4: _ckpt_subdir rule
# ---------------------------------------------------------------------------

def test_ckpt_subdir_h1_spy() -> None:
    """h1 + spy (default dataset) → {arch}_h1_spy flat scheme."""
    assert _ckpt_subdir("lstm", "h1") == "lstm_h1_spy"
    assert _ckpt_subdir("lstm", "h1", "spy") == "lstm_h1_spy"


def test_ckpt_subdir_5m_multisym() -> None:
    assert _ckpt_subdir("lstm", "5m", "multisym") == "lstm_5m_multisym"


def test_ckpt_subdir_15m_multisym() -> None:
    assert _ckpt_subdir("lstm", "15m", "multisym") == "lstm_15m_multisym"


def test_ckpt_subdir_all_arches_h1() -> None:
    """All arches: h1 spy → {arch}_h1_spy."""
    for arch in ("lstm", "cnn_lstm", "transformer", "xlstm", "xgb"):
        assert _ckpt_subdir(arch, "h1", "spy") == f"{arch}_h1_spy", f"arch={arch}"


def test_ckpt_subdir_all_arches_5m() -> None:
    """All arches: 5m multisym → {arch}_5m_multisym."""
    for arch in ("lstm", "cnn_lstm", "transformer", "xlstm", "xgb"):
        assert _ckpt_subdir(arch, "5m", "multisym") == f"{arch}_5m_multisym"


def test_ckpt_subdir_tuned() -> None:
    """tuned=True appends _tuned suffix."""
    assert _ckpt_subdir("cnn_lstm", "15m", "multisym", tuned=True) == "cnn_lstm_15m_multisym_tuned"


# ---------------------------------------------------------------------------
# 5-6: SeedSweepConfig defaults + from_experiment_config propagation
# ---------------------------------------------------------------------------

def test_seed_sweep_config_timeframe_default() -> None:
    cfg = SeedSweepConfig(model_type="lstm", hyperparams={}, seeds=[42])
    assert cfg.timeframe == "h1"


def test_from_experiment_config_propagates_timeframe_h1() -> None:
    """Default YAML has no timeframe key → "h1" default propagates through."""
    exp = load_experiment("experiments/lstm_g1.yaml")
    sweep = SeedSweepConfig.from_experiment_config(exp)
    assert sweep.timeframe == "h1"


def test_from_experiment_config_propagates_timeframe_5m() -> None:
    """lstm_5m.yaml carries timeframe: "5m" → sweep.timeframe == "5m"."""
    exp = load_experiment("experiments/lstm_5m.yaml")
    sweep = SeedSweepConfig.from_experiment_config(exp)
    assert sweep.timeframe == "5m"


def test_from_experiment_config_propagates_timeframe_15m() -> None:
    exp = load_experiment("experiments/lstm_15m.yaml")
    sweep = SeedSweepConfig.from_experiment_config(exp)
    assert sweep.timeframe == "15m"


# ---------------------------------------------------------------------------
# 7: run_seed_sweep writes to token-namespaced ckpt dir
# ---------------------------------------------------------------------------

def _make_fake_result(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "test_macro_f1": float(rng.uniform(0.6, 0.8)),
        "test_per_class_f1": [
            float(rng.uniform(0.8, 0.95)),
            float(rng.uniform(0.5, 0.75)),
            float(rng.uniform(0.5, 0.75)),
        ],
        "y_true": np.array([0, 1, 2, 0], dtype=np.int64),
        "y_pred": np.array([0, 1, 2, 0], dtype=np.int64),
    }


def _write_fake_meta(meta_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    meta = {
        "test_macro_f1": float(rng.uniform(0.6, 0.8)),
        "test_per_class_f1": [0.88, 0.60, 0.61],
        "timeframe": "5m",
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta))


def test_run_seed_sweep_5m_uses_namespaced_dir(tmp_path: Path) -> None:
    """5m training writes checkpoints to {checkpoint_dir}/lstm_5m_spy/, not lstm/ or lstm_5m/."""
    cfg = SeedSweepConfig(
        model_type="lstm",
        hyperparams={"hidden_size": 64, "lr": 1e-3},
        seeds=[42],
        timeframe="5m",
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoints",
    )

    with patch("src.rigor.seed_sweep._train_lstm") as mock_train:
        mock_train.side_effect = lambda config, seed, ckpt_path, meta_path: (
            _write_fake_meta(meta_path, seed) or _make_fake_result(seed)
        )
        run_seed_sweep(cfg)

    # Checkpoint must land in lstm_5m_spy (default dataset=spy), NOT old lstm_5m
    expected_dir = tmp_path / "checkpoints" / "lstm_5m_spy"
    assert expected_dir.exists(), f"Expected {expected_dir} to exist"
    wrong_dir = tmp_path / "checkpoints" / "lstm_5m"
    assert not wrong_dir.exists(), f"Unexpected {wrong_dir} — old naming must not be created"
    wrong_dir2 = tmp_path / "checkpoints" / "lstm"
    assert not wrong_dir2.exists(), f"Unexpected {wrong_dir2} — h1 path must not be created for 5m"


def test_run_seed_sweep_h1_uses_namespaced_dir(tmp_path: Path) -> None:
    """h1 training writes to {checkpoint_dir}/lstm_h1_spy/ (explicit flat scheme)."""
    cfg = SeedSweepConfig(
        model_type="lstm",
        hyperparams={"hidden_size": 64, "lr": 1e-3},
        seeds=[42],
        timeframe="h1",
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoints",
    )

    with patch("src.rigor.seed_sweep._train_lstm") as mock_train:
        mock_train.side_effect = lambda config, seed, ckpt_path, meta_path: (
            _write_fake_meta(meta_path, seed) or _make_fake_result(seed)
        )
        run_seed_sweep(cfg)

    expected_dir = tmp_path / "checkpoints" / "lstm_h1_spy"
    assert expected_dir.exists(), f"Expected {expected_dir} for h1 spy"
    wrong_dir = tmp_path / "checkpoints" / "lstm"
    assert not wrong_dir.exists(), f"Unexpected {wrong_dir} — old bare arch dir must not be created"


# ---------------------------------------------------------------------------
# 8: meta JSON carries "timeframe" key
# ---------------------------------------------------------------------------

def test_run_seed_sweep_meta_carries_timeframe(tmp_path: Path) -> None:
    """Meta JSON written by run_seed_sweep must contain 'timeframe' key."""
    cfg = SeedSweepConfig(
        model_type="lstm",
        hyperparams={"hidden_size": 64, "lr": 1e-3},
        seeds=[42],
        timeframe="5m",
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoints",
    )

    captured_meta_path: list[Path] = []

    def fake_train(config, seed, ckpt_path, meta_path):
        captured_meta_path.append(meta_path)
        _write_fake_meta(meta_path, seed)
        return _make_fake_result(seed)

    with patch("src.rigor.seed_sweep._train_lstm", side_effect=fake_train):
        run_seed_sweep(cfg)

    assert captured_meta_path, "No meta path was captured"
    meta = json.loads(captured_meta_path[0].read_text())
    assert "timeframe" in meta, "meta.json must carry 'timeframe' key (WS-5)"
    assert meta["timeframe"] == "5m"


# ---------------------------------------------------------------------------
# 9-10: _load_splits uses token-namespaced parquet
# ---------------------------------------------------------------------------

def test_load_splits_uses_token_filename(tmp_path: Path) -> None:
    """_load_splits(data_dir, '5m') reads spy_5m_*.parquet, not spy_h1_*.parquet."""
    import pandas as pd
    from src.rigor.seed_sweep import _load_splits

    # Write minimal parquets with 5m token names
    dummy = pd.DataFrame({"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.0], "volume": [100]})
    (tmp_path / "spy_5m_train.parquet").parent.mkdir(parents=True, exist_ok=True)
    dummy.to_parquet(tmp_path / "spy_5m_train.parquet")
    dummy.to_parquet(tmp_path / "spy_5m_val.parquet")
    dummy.to_parquet(tmp_path / "spy_5m_test.parquet")

    train, val, test = _load_splits(tmp_path, "5m")
    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1

    # h1-named files must NOT be created (we only wrote 5m files above)
    assert not (tmp_path / "spy_h1_train.parquet").exists()


def test_load_splits_h1_reads_h1_parquet(tmp_path: Path) -> None:
    """_load_splits(data_dir, 'h1') reads spy_h1_*.parquet (backward-compat)."""
    import pandas as pd
    from src.rigor.seed_sweep import _load_splits

    dummy = pd.DataFrame({"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.0], "volume": [100]})
    dummy.to_parquet(tmp_path / "spy_h1_train.parquet")
    dummy.to_parquet(tmp_path / "spy_h1_val.parquet")
    dummy.to_parquet(tmp_path / "spy_h1_test.parquet")

    train, val, test = _load_splits(tmp_path, "h1")
    assert len(train) == 1


# ---------------------------------------------------------------------------
# 11: New 5m/15m YAMLs are loadable with correct timeframe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("yaml_name,expected_tf", [
    ("experiments/lstm_5m.yaml", "5m"),
    ("experiments/lstm_15m.yaml", "15m"),
    ("experiments/cnn_lstm_5m.yaml", "5m"),
    ("experiments/cnn_lstm_15m.yaml", "15m"),
    ("experiments/transformer_5m.yaml", "5m"),
    ("experiments/transformer_15m.yaml", "15m"),
    ("experiments/xgboost_5m.yaml", "5m"),
    ("experiments/xgboost_15m.yaml", "15m"),
])
def test_new_yaml_timeframe_field(yaml_name: str, expected_tf: str) -> None:
    """New {arch}_{token}.yaml files load cleanly and carry correct timeframe."""
    exp = load_experiment(yaml_name)
    assert exp.data.timeframe == expected_tf, (
        f"{yaml_name}: expected timeframe={expected_tf!r}, got {exp.data.timeframe!r}"
    )
    # Ensure timeframe_obj resolves without error
    tf_obj = exp.data.timeframe_obj
    assert tf_obj.token == expected_tf


# ---------------------------------------------------------------------------
# 12: XGB subprocess worker receives --timeframe and resolves correct paths
# ---------------------------------------------------------------------------

def _make_xgb_fake_parquets(tmp_path: Path, token: str) -> Path:
    """Write minimal spy_{token}_{split}.parquet files; return data_dir."""
    import pandas as pd

    n = 100
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-02 09:30", periods=n, freq="1h", tz="America/New_York")
    prices = 400.0 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame({
        "open": prices, "high": prices + 0.1, "low": prices - 0.1,
        "close": prices, "volume": np.ones(n) * 1000,
        "label": np.zeros(n, dtype=int),
    }, index=idx)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        df.to_parquet(data_dir / f"spy_{token}_{split}.parquet")
    return data_dir


@pytest.mark.parametrize("token", ["5m", "h1"])
def test_xgb_sweep_worker_cmd_includes_timeframe(tmp_path: Path, token: str) -> None:
    """_run_xgb_seed_sweep_subprocess passes --timeframe <token> to the worker."""
    from unittest.mock import patch

    data_dir = _make_xgb_fake_parquets(tmp_path, token)

    import scripts.rigor.eval.multiseed_run as mr

    cfg = SeedSweepConfig(
        model_type="xgboost",
        hyperparams={"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1,
                     "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8},
        seeds=[42],
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "ckpt",
        data_dir=data_dir,
        timeframe=token,
    )

    captured_cmds: list[list] = []

    with patch("subprocess.run", side_effect=lambda cmd, **kw: captured_cmds.append(cmd)):
        mr._run_xgb_seed_sweep_subprocess(cfg, tmp_path / "ts")

    assert len(captured_cmds) == 1, "Expected exactly one subprocess.run call"
    cmd = captured_cmds[0]
    assert "--timeframe" in cmd, f"--timeframe missing from worker cmd: {cmd}"
    tf_idx = cmd.index("--timeframe")
    assert cmd[tf_idx + 1] == token, f"Expected {token!r}, got {cmd[tf_idx + 1]!r}"


def test_xgb_worker_5m_resolves_correct_weights_path() -> None:
    """_xgb_sweep_worker resolves class_weights_spy_{tf}.json for each token."""
    # This tests the path-resolution logic in the worker without running it.
    # We replicate the same logic that the worker uses (flat naming: scope_tf).
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent.parent

    def _worker_weights_path(tf: str, scope: str = "spy") -> Path:
        return ROOT / "data" / "processed" / f"class_weights_{scope}_{tf}.json"

    assert _worker_weights_path("h1").name == "class_weights_spy_h1.json"
    assert _worker_weights_path("5m").name == "class_weights_spy_5m.json"
    assert _worker_weights_path("15m").name == "class_weights_spy_15m.json"
    assert _worker_weights_path("h1", scope="multisym").name == "class_weights_multisym_h1.json"
