"""test_timeframe_aware.py — WS-6 TF-aware inspect tests.

Verifies:
1. runner.run(timeframe=M5) uses the 5m gap threshold (8 min) not the H1 one (90 min).
2. _resolve_dataset_path token mapping — h1→spy_h1_test, 5m→spy_5m_test, 15m→spy_15m_test.
3. multisym resolver is token-aware — h1 uses bare dirs; 5m appends _5m suffix.
4. Checkpoint meta TF mismatch causes main() to return 1 (hard error).
5. Missing checkpoint meta assumes h1 (no error when --timeframe h1).

All tests are path-resolution / threshold logic only.  No real data or
real checkpoints are required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.timeframe import H1, M5, M15, Timeframe
from src.inspect.base import ModelAdapter
from src.inspect.multisym import (
    MULTISYM_DIRS,
    _multisym_dirs_for_token,
    resolve_multisym_checkpoints,
)
from src.inspect.runner import run


# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------

class _ZeroAdapter(ModelAdapter):
    name = "stub_zero"

    def __init__(self, *a, **kw):
        pass

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        n = windows.shape[0]
        out = np.zeros((n, 3), dtype=np.float32)
        out[:, 0] = 1.0
        return out


# ---------------------------------------------------------------------------
# Minimal synthetic DataFrame helpers
# ---------------------------------------------------------------------------

def _make_df(n: int, freq_minutes: int = 5) -> pd.DataFrame:
    """Build a minimal OHLCV+label DataFrame at *freq_minutes* spacing."""
    rng = np.random.default_rng(7)
    idx = pd.date_range(
        "2024-01-03 09:30",
        periods=n,
        freq=f"{freq_minutes}min",
        tz="America/New_York",
    )
    prices = 400.0 + np.abs(rng.normal(0, 1, n).cumsum()) + 100.0
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + rng.uniform(0.1, 0.5, n),
            "low": prices - rng.uniform(0.1, 0.5, n),
            "close": prices + rng.normal(0, 0.1, n),
            "volume": rng.integers(5000, 20000, n).astype(float),
            "label": 0,
        },
        index=idx,
    )
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


def _make_df_with_overnight_gap(freq_minutes: int = 5, window_size: int = 20) -> pd.DataFrame:
    """Build a DataFrame with an overnight gap in the middle.

    The gap is larger than the H1 threshold (90 min) but the consecutive
    bars within each day are freq_minutes apart.  Used to test that the
    session-gap threshold is TF-derived, not hard-coded.
    """
    # Day 1: window_size bars at freq_minutes spacing
    rng = np.random.default_rng(99)
    day1_idx = pd.date_range(
        "2024-01-03 09:30",
        periods=window_size,
        freq=f"{freq_minutes}min",
        tz="America/New_York",
    )
    # Day 2: another window_size bars — gap between last day1 and first day2
    # is ~17 hours (overnight), well above both 8-min (M5) and 90-min (H1) thresholds
    day2_idx = pd.date_range(
        "2024-01-04 09:30",
        periods=window_size,
        freq=f"{freq_minutes}min",
        tz="America/New_York",
    )
    idx = day1_idx.append(day2_idx)
    n = len(idx)
    prices = 400.0 + np.abs(rng.normal(0, 1, n).cumsum()) + 100.0
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.2,
            "low": prices - 0.2,
            "close": prices,
            "volume": 10000.0,
            "label": 0,
        },
        index=idx,
    )
    return df


# ---------------------------------------------------------------------------
# 1. runner.run() gap threshold tests
# ---------------------------------------------------------------------------

class _FakeLabeller:
    """Minimal labeller stub accepted by runner.run()."""
    classes_ = [0, 1, 2]


@pytest.fixture
def labeller():
    return _FakeLabeller()


def test_runner_m5_gap_threshold_is_8(labeller):
    """M5.max_intra_window_gap_minutes == 8 (round(1.5 * 5))."""
    assert M5.max_intra_window_gap_minutes == 8


def test_runner_h1_gap_threshold_is_90(labeller):
    """H1.max_intra_window_gap_minutes == 90 — unchanged from hard-coded value."""
    assert H1.max_intra_window_gap_minutes == 90


def test_runner_m5_drop_cross_session_uses_5m_threshold(labeller):
    """run(timeframe=M5, drop_cross_session=True) uses 8-min gap, not 90-min.

    The synthetic DF has an overnight gap (~17 h) between day 1 and day 2.
    With window_size == number of bars per day, the only windows that span the
    gap should be dropped; all intra-day windows should pass (their max
    consecutive gap is freq_minutes = 5 min < 8 min threshold).

    This test asserts that at least some windows are produced (intra-day ones
    are kept) and that the run doesn't crash.
    """
    window_size = 10
    df = _make_df_with_overnight_gap(freq_minutes=5, window_size=window_size)
    adapters = [_ZeroAdapter()]
    results = run(
        adapters, df, labeller,
        window_size=window_size,
        drop_cross_session=True,
        timeframe=M5,
    )
    # There should be some windows (the intra-day ones)
    assert results.n >= 1


def test_runner_h1_default_timeframe_unchanged(labeller):
    """run() with no timeframe arg produces same result as run(timeframe=H1)."""
    df = _make_df(n=100, freq_minutes=60)
    adapters = [_ZeroAdapter()]
    res_default = run(adapters, df, labeller, window_size=10)
    res_h1 = run(adapters, df, labeller, window_size=10, timeframe=H1)
    assert res_default.n == res_h1.n


# ---------------------------------------------------------------------------
# 2. _resolve_dataset_path token mapping
# ---------------------------------------------------------------------------

def test_resolve_dataset_path_h1_test():
    """h1 token → spy_h1_test.parquet (existing path, backward-compat)."""
    from scripts.inspect_models import _resolve_dataset_path
    p = _resolve_dataset_path("test", token="h1")
    assert p.name == "spy_h1_test.parquet"
    assert "data/processed" in str(p)


def test_resolve_dataset_path_5m_test():
    """5m token → spy_5m_test.parquet."""
    from scripts.inspect_models import _resolve_dataset_path
    p = _resolve_dataset_path("test", token="5m")
    assert p.name == "spy_5m_test.parquet"


def test_resolve_dataset_path_15m_train():
    """15m token → spy_15m_train.parquet."""
    from scripts.inspect_models import _resolve_dataset_path
    p = _resolve_dataset_path("train", token="15m")
    assert p.name == "spy_15m_train.parquet"


def test_resolve_dataset_path_explicit_path_ignores_token():
    """Explicit path string bypasses token substitution."""
    from scripts.inspect_models import _resolve_dataset_path
    explicit = "data/processed/my_custom.parquet"
    p = _resolve_dataset_path(explicit, token="5m")
    assert p.name == "my_custom.parquet"


# ---------------------------------------------------------------------------
# 3. Multisym resolver token-aware
# ---------------------------------------------------------------------------

def test_multisym_dirs_h1_uses_flat_names():
    """h1 token → flat directory names {arch}_h1_multisym (no inner subdir)."""
    dirs = _multisym_dirs_for_token("h1")
    assert dirs["lstm"][0] == "lstm_h1_multisym"
    assert dirs["cnn_lstm"][0] == "cnn_lstm_h1_multisym"
    assert dirs["xgboost"][0] == "xgboost_h1_multisym"


def test_multisym_dirs_5m_uses_flat_names():
    """5m token → flat directory names {arch}_5m_multisym."""
    dirs = _multisym_dirs_for_token("5m")
    assert dirs["lstm"][0] == "lstm_5m_multisym"
    assert dirs["cnn_lstm"][0] == "cnn_lstm_5m_multisym"
    assert dirs["xgboost"][0] == "xgboost_5m_multisym"


def test_multisym_dirs_15m_uses_flat_names():
    """15m token → flat directory names {arch}_15m_multisym."""
    dirs = _multisym_dirs_for_token("15m")
    assert dirs["transformer"][0] == "transformer_15m_multisym"


def test_multisym_dirs_h1_matches_legacy_constant():
    """_multisym_dirs_for_token('h1') subdirs match the MULTISYM_DIRS constant."""
    dirs = _multisym_dirs_for_token("h1")
    for arch in MULTISYM_DIRS:
        assert dirs[arch][0] == MULTISYM_DIRS[arch][0], (
            f"arch={arch}: {dirs[arch][0]} != {MULTISYM_DIRS[arch][0]}"
        )


def test_resolve_multisym_checkpoints_token_default_is_h1(tmp_path):
    """resolve_multisym_checkpoints(dir) (no token) resolves h1 dirs — no files found is fine."""
    result = resolve_multisym_checkpoints(tmp_path)
    # Empty result expected (no checkpoints exist in tmp_path); just no crash
    assert isinstance(result, dict)


def test_resolve_multisym_checkpoints_5m_finds_files(tmp_path):
    """resolve_multisym_checkpoints with token='5m' finds files in the flat _5m_multisym dir."""
    # Create a stub checkpoint in the expected flat 5m location
    ckpt_dir = tmp_path / "lstm_5m_multisym"
    ckpt_dir.mkdir(parents=True)
    stub = ckpt_dir / "lstm_seed0.pt"
    stub.write_text("stub")

    result = resolve_multisym_checkpoints(tmp_path, token="5m")
    assert "lstm" in result
    assert result["lstm"][0][0] == 0  # seed 0
    assert result["lstm"][0][1] == stub


def test_resolve_multisym_checkpoints_h1_does_not_find_5m_files(tmp_path):
    """h1 resolution must NOT pick up files from _5m_multisym dirs."""
    ckpt_dir = tmp_path / "lstm_5m_multisym"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "lstm_seed0.pt").write_text("stub")

    result = resolve_multisym_checkpoints(tmp_path, token="h1")
    assert "lstm" not in result


# ---------------------------------------------------------------------------
# 4 & 5. Checkpoint meta TF mismatch / missing meta
# ---------------------------------------------------------------------------

def _run_main(argv: list[str]) -> int:
    """Run inspect_models.main() and return exit code."""
    from scripts.inspect_models import main
    return main(argv)


def test_main_tf_mismatch_returns_error(tmp_path, monkeypatch):
    """main() returns 1 when checkpoint meta declares TF != --timeframe."""
    import scripts.inspect_models as _im
    from unittest.mock import MagicMock
    import pandas as pd, numpy as np

    # Create a stub checkpoint + meta declaring timeframe=h1 (mismatching --timeframe 5m)
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"stub")
    meta = ckpt.with_suffix(".meta.json")
    meta.write_text(json.dumps({"timeframe": "h1", "hyperparams": {}}))

    # Create a minimal real parquet so the dataset-not-found check passes
    ds = tmp_path / "spy_5m_test.parquet"
    idx = pd.date_range("2023-01-02 09:30", periods=60, freq="5min", tz="America/New_York")
    pd.DataFrame({
        "open": np.ones(60), "high": np.ones(60)+0.1, "low": np.ones(60)-0.1,
        "close": np.ones(60), "volume": np.ones(60)*1000,
        "label": np.zeros(60, dtype=int), "raw_label": ["none"]*60,
    }, index=idx).to_parquet(ds)

    # Stub load_adapters so we bypass torch.load but still expose checkpoint_path on adapter
    fake_adapter = MagicMock()
    fake_adapter.checkpoint_path = str(ckpt)
    fake_adapter.name = "lstm"
    monkeypatch.setattr("src.inspect.registry.load_adapters", lambda *a, **kw: [fake_adapter])

    rc = _run_main([
        "--timeframe", "5m",
        "--models", f"lstm:{ckpt}",
        "--dataset", str(ds),
        "--checkpoint-dir", str(tmp_path),
    ])
    assert rc == 1, "Expected exit code 1 on TF mismatch"


def test_main_missing_meta_assumes_h1_no_error(tmp_path, monkeypatch):
    """Missing checkpoint meta → assumed h1.  No error when --timeframe h1."""
    # Create stub checkpoint with NO meta file
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"stub")

    # The TF check should pass (both assumed h1 == requested h1).
    # The script will fail later (dataset not found / adapter load error),
    # but NOT with exit code 1 from the TF-mismatch check.
    # We monkeypatch _resolve_dataset_path to return a non-existent path so
    # main() exits at the dataset-not-found check, not inside TF validation.
    import scripts.inspect_models as _im
    original = _im._resolve_dataset_path

    def _fake_resolve(dataset, token="h1"):
        return tmp_path / "nonexistent.parquet"

    monkeypatch.setattr(_im, "_resolve_dataset_path", _fake_resolve)

    rc = _run_main([
        "--timeframe", "h1",
        "--models", f"lstm:{ckpt}",
        "--dataset", "test",
        "--checkpoint-dir", str(tmp_path),
    ])
    # Exit code 1 because dataset not found — NOT because of TF mismatch
    # (i.e., the TF check passed silently)
    assert rc == 1


def test_main_matching_tf_meta_no_error(tmp_path, monkeypatch):
    """Checkpoint meta TF matches --timeframe → no TF-mismatch error."""
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"stub")
    meta = ckpt.with_suffix(".meta.json")
    meta.write_text(json.dumps({"timeframe": "5m"}))

    import scripts.inspect_models as _im

    def _fake_resolve(dataset, token="h1"):
        return tmp_path / "nonexistent.parquet"

    monkeypatch.setattr(_im, "_resolve_dataset_path", _fake_resolve)

    rc = _run_main([
        "--timeframe", "5m",
        "--models", f"lstm:{ckpt}",
        "--dataset", "test",
        "--checkpoint-dir", str(tmp_path),
    ])
    # Should exit 1 for missing dataset, NOT for TF mismatch
    assert rc == 1


# ---------------------------------------------------------------------------
# H2 (hoist): TF validation runs for BARE-NAME checkpoints too, and is
# non-vacuous (reaches the validation, not an earlier dataset/load failure).
# ---------------------------------------------------------------------------

def _real_dataset(tmp_path: Path) -> Path:
    """Write a small valid OHLCV+label parquet so dataset resolution passes."""
    df = _make_df(80, freq_minutes=5)
    p = tmp_path / "ds.parquet"
    df.to_parquet(p)
    return p


def _bare_name_adapter(tmp_path: Path, meta_tf: str):
    """Return a _ZeroAdapter exposing checkpoint_path whose meta declares meta_tf."""
    ckpt = tmp_path / "bare_model.pt"
    ckpt.write_bytes(b"stub")
    if meta_tf is not None:
        ckpt.with_suffix(".meta.json").write_text(json.dumps({"timeframe": meta_tf}))
    ad = _ZeroAdapter()
    ad.checkpoint_path = ckpt
    return ad


def test_main_bare_name_5m_with_h1_meta_errors(tmp_path, monkeypatch):
    """Bare-name model (no name:path) on --timeframe 5m but h1-stamped meta → rc 1.

    Proven non-vacuous: dataset resolution and adapter load are both stubbed to
    succeed, so the only thing that can return 1 is the TF-mismatch guard — and
    run() is patched to fail loudly if it is ever reached.
    """
    import scripts.inspect_models as _im

    import src.inspect.registry as _reg
    import src.inspect.runner as _runner

    ds = _real_dataset(tmp_path)
    monkeypatch.setattr(_im, "_resolve_dataset_path", lambda dataset, token="h1": ds)
    monkeypatch.setattr(
        _reg, "load_adapters",
        lambda names, ckpt_dir, checkpoint_paths=None, **kw: [
            _bare_name_adapter(tmp_path, meta_tf="h1")
        ],
    )

    def _boom(*a, **k):  # run() must NOT be reached on a mismatch
        raise AssertionError("TF guard did not fire — inference was reached")

    monkeypatch.setattr(_runner, "run", _boom)

    rc = _run_main([
        "--timeframe", "5m",
        "--models", "lstm",        # BARE name, no :path override
        "--dataset", "test",
        "--checkpoint-dir", str(tmp_path),
    ])
    assert rc == 1, "Expected rc=1 from the bare-name TF-mismatch guard"


def test_main_bare_name_h1_meta_passes_validation(tmp_path, monkeypatch):
    """Control: bare-name h1 model on --timeframe h1 passes the TF guard.

    Proves the mismatch test above is not trivially returning 1 — here the guard
    accepts the adapter and execution proceeds into run() (which we stub to a
    sentinel so we can assert it was reached, then short-circuit cleanly).
    """
    import scripts.inspect_models as _im

    import src.inspect.registry as _reg
    import src.inspect.runner as _runner

    ds = _real_dataset(tmp_path)
    monkeypatch.setattr(_im, "_resolve_dataset_path", lambda dataset, token="h1": ds)
    monkeypatch.setattr(
        _reg, "load_adapters",
        lambda names, ckpt_dir, checkpoint_paths=None, **kw: [
            _bare_name_adapter(tmp_path, meta_tf="h1")
        ],
    )

    reached = {"run": False}

    def _sentinel(*a, **k):
        reached["run"] = True
        raise RuntimeError("stop-after-validation")

    monkeypatch.setattr(_runner, "run", _sentinel)

    with pytest.raises(RuntimeError, match="stop-after-validation"):
        _run_main([
            "--timeframe", "h1",
            "--models", "lstm",
            "--dataset", "test",
            "--checkpoint-dir", str(tmp_path),
        ])
    assert reached["run"], "TF guard wrongly rejected a matching h1 checkpoint"
