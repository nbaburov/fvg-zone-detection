"""test_transformer_adapter.py — Tests for TransformerAdapter.

Covers:
  - Registry discovery: "transformer" appears in list_available()
  - predict_proba contract: (N,3) float32, rows sum to ~1.0, values in [0,1]
  - Empty-batch edge case: N=0 returns (0,3) without error
  - predict() argmax: returns (N,) ints in {0,1,2}
  - Checkpoint loading from real committed checkpoint + meta.json filtering
    (Optuna keys lr/warmup_steps/n_trials_completed/batch_size must not break ctor)
  - Determinism: same input -> same output across two calls in eval mode
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints"
_TRANSFORMER_CKPT = _CHECKPOINT_DIR / "transformer_h1_spy" / "transformer_seed0.pt"
_TRANSFORMER_META = _CHECKPOINT_DIR / "transformer_h1_spy" / "transformer_seed0.meta.json"

_HAS_REAL_CKPT = _TRANSFORMER_CKPT.exists()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def random_windows() -> np.ndarray:
    """Synthetic normalised (4, 60, 5) float32 batch."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((4, 60, 5)).astype(np.float32)


@pytest.fixture()
def real_adapter():
    """TransformerAdapter loaded from the committed checkpoint — skip if absent."""
    if not _HAS_REAL_CKPT:
        pytest.skip("Real transformer checkpoint not found — skipping")
    from src.inspect.adapters.transformer_adapter import TransformerAdapter
    return TransformerAdapter(checkpoint_dir=_CHECKPOINT_DIR)


# ---------------------------------------------------------------------------
# 1. Registry discovery
# ---------------------------------------------------------------------------


def test_registry_includes_transformer():
    from src.inspect.registry import list_available
    assert "transformer" in list_available()


# ---------------------------------------------------------------------------
# 2. predict_proba contract — real checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_REAL_CKPT, reason="No real checkpoint")
def test_predict_proba_shape(real_adapter, random_windows):
    out = real_adapter.predict_proba(random_windows)
    assert out.shape == (4, 3)


@pytest.mark.skipif(not _HAS_REAL_CKPT, reason="No real checkpoint")
def test_predict_proba_dtype(real_adapter, random_windows):
    out = real_adapter.predict_proba(random_windows)
    assert out.dtype == np.float32


@pytest.mark.skipif(not _HAS_REAL_CKPT, reason="No real checkpoint")
def test_predict_proba_rows_sum_to_one(real_adapter, random_windows):
    out = real_adapter.predict_proba(random_windows)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(4, dtype=np.float32), atol=1e-5)


@pytest.mark.skipif(not _HAS_REAL_CKPT, reason="No real checkpoint")
def test_predict_proba_values_in_unit_interval(real_adapter, random_windows):
    out = real_adapter.predict_proba(random_windows)
    assert (out >= 0.0).all() and (out <= 1.0).all()


# ---------------------------------------------------------------------------
# 3. Empty-batch edge case
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_REAL_CKPT, reason="No real checkpoint")
def test_predict_proba_empty_batch(real_adapter):
    empty = np.zeros((0, 60, 5), dtype=np.float32)
    out = real_adapter.predict_proba(empty)
    assert out.shape == (0, 3)
    assert out.ndim == 2


# ---------------------------------------------------------------------------
# 4. predict() argmax
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_REAL_CKPT, reason="No real checkpoint")
def test_predict_argmax_shape(real_adapter, random_windows):
    preds = real_adapter.predict(random_windows)
    assert preds.shape == (4,)


@pytest.mark.skipif(not _HAS_REAL_CKPT, reason="No real checkpoint")
def test_predict_argmax_values_in_valid_classes(real_adapter, random_windows):
    preds = real_adapter.predict(random_windows)
    assert set(preds.tolist()).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# 5. Checkpoint loading + Optuna key filtering
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_REAL_CKPT, reason="No real checkpoint")
def test_real_checkpoint_loads_without_error():
    """Loading seed0 checkpoint must succeed — architecture must match state_dict.

    Revert-sensitivity: if FVGTransformerClassifier ctor signature changes or
    the meta.json HP keys diverge from the model, this will raise either a
    TypeError (unexpected kwarg) or RuntimeError (state_dict key mismatch).
    """
    from src.inspect.adapters.transformer_adapter import TransformerAdapter
    adapter = TransformerAdapter(checkpoint_dir=_CHECKPOINT_DIR)
    assert adapter is not None


@pytest.mark.skipif(not _HAS_REAL_CKPT, reason="No real checkpoint")
def test_optuna_keys_in_meta_do_not_break_ctor():
    """meta.json contains lr, warmup_steps, batch_size, n_trials_completed —
    none of these are FVGTransformerClassifier ctor params; the adapter must
    filter them before construction.

    Revert-sensitivity: removing the HP-filter step in TransformerAdapter.__init__
    causes TypeError: unexpected keyword argument.
    """
    assert _TRANSFORMER_META.exists(), "meta.json must exist alongside checkpoint"
    with open(_TRANSFORMER_META) as f:
        meta = json.load(f)
    hp = meta.get("hyperparams", {})
    # Confirm Optuna keys are present in the sidecar (so filter is exercised)
    assert "lr" in hp or "warmup_steps" in hp or "n_trials_completed" in hp, (
        "meta.json no longer contains Optuna metadata — test premise changed"
    )
    # If loading succeeds the filter worked
    from src.inspect.adapters.transformer_adapter import TransformerAdapter
    TransformerAdapter(checkpoint_dir=_CHECKPOINT_DIR)


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_REAL_CKPT, reason="No real checkpoint")
def test_determinism(real_adapter, random_windows):
    """Eval mode + no_grad — same input must yield bitwise-identical output.

    Revert-sensitivity: if model is accidentally left in train() mode dropout
    will differ between calls and this assertion fails.
    """
    out1 = real_adapter.predict_proba(random_windows)
    out2 = real_adapter.predict_proba(random_windows)
    np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# 7. Missing checkpoint raises FileNotFoundError
# ---------------------------------------------------------------------------


def test_missing_checkpoint_raises(tmp_path):
    from src.inspect.adapters.transformer_adapter import TransformerAdapter
    with pytest.raises(FileNotFoundError):
        TransformerAdapter(checkpoint_dir=tmp_path)
