"""test_xlstm_adapter.py — Tests for XLSTMAdapter.

NOTE — Synthetic checkpoint caveat:
    No trained xLSTM checkpoint is committed to this repository. All tests that
    require a loaded adapter construct a fresh FVGxLSTMClassifier with default
    hyperparameters, save its state_dict + a matching meta.json to tmp_path,
    and point XLSTMAdapter at it. Real-weights inference is therefore unverified
    by this suite. If a trained xlstm_seed42.pt is committed in future, add a
    test mirroring test_real_checkpoint_loads_without_error from
    test_transformer_adapter.py.

Covers:
  - Registry discovery: "xlstm" appears in list_available()
  - predict_proba contract: (N,3) float32, rows sum to ~1.0, values in [0,1]
  - Empty-batch edge case: N=0 returns (0,3) without error
  - predict() argmax: returns (N,) ints in {0,1,2}
  - Synthetic round-trip: build model → save state_dict + meta.json → load via
    adapter → predict_proba works (exercises full ctor + state_dict loading path)
  - Optuna-key filtering (via synthetic meta.json carrying extraneous keys)
  - Determinism: same input -> same output across two calls in eval mode
  - Missing checkpoint raises FileNotFoundError
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_synthetic_adapter(tmp_path: Path, extra_meta: dict | None = None):
    """Build FVGxLSTMClassifier with default HP, save state_dict + meta.json,
    return an XLSTMAdapter pointing at tmp_path.

    extra_meta: keys merged into hyperparams to simulate Optuna sidecar noise.
    """
    from src.models.xlstm_model import FVGxLSTMClassifier
    from src.inspect.adapters.xlstm_adapter import XLSTMAdapter

    # Build model with default HP
    model = FVGxLSTMClassifier()
    model.eval()

    # Save checkpoint (bare state_dict)
    ckpt_dir = tmp_path / "xlstm"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "xlstm_seed42.pt"
    torch.save(model.state_dict(), str(ckpt_path))

    # Save matching meta.json
    hp: dict = {
        "input_size": 5,
        "embedding_dim": 64,
        "num_blocks": 2,
        "num_heads": 4,
        "num_classes": 3,
        "dropout": 0.1,
        "head_dropout": 0.3,
        "context_length": 60,
    }
    if extra_meta:
        hp.update(extra_meta)
    meta = {"hyperparams": hp}
    meta_path = ckpt_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return XLSTMAdapter(checkpoint_dir=tmp_path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def random_windows() -> np.ndarray:
    """Synthetic normalised (4, 60, 5) float32 batch."""
    rng = np.random.default_rng(1)
    return rng.standard_normal((4, 60, 5)).astype(np.float32)


@pytest.fixture()
def synthetic_adapter(tmp_path):
    """XLSTMAdapter built from a synthetic round-tripped checkpoint."""
    return _build_synthetic_adapter(tmp_path)


# ---------------------------------------------------------------------------
# 1. Registry discovery
# ---------------------------------------------------------------------------


def test_registry_includes_xlstm():
    from src.inspect.registry import list_available
    assert "xlstm" in list_available()


# ---------------------------------------------------------------------------
# 2. predict_proba contract
# ---------------------------------------------------------------------------


def test_predict_proba_shape(synthetic_adapter, random_windows):
    out = synthetic_adapter.predict_proba(random_windows)
    assert out.shape == (4, 3)


def test_predict_proba_dtype(synthetic_adapter, random_windows):
    out = synthetic_adapter.predict_proba(random_windows)
    assert out.dtype == np.float32


def test_predict_proba_rows_sum_to_one(synthetic_adapter, random_windows):
    """Softmax output must form a valid probability distribution per row.

    Revert-sensitivity: removing F.softmax in XLSTMAdapter.predict_proba causes
    raw logits to be returned — row sums will differ from 1.0 by an arbitrary
    amount and this assertion fails.
    """
    out = synthetic_adapter.predict_proba(random_windows)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(4, dtype=np.float32), atol=1e-5)


def test_predict_proba_values_in_unit_interval(synthetic_adapter, random_windows):
    out = synthetic_adapter.predict_proba(random_windows)
    assert (out >= 0.0).all() and (out <= 1.0).all()


# ---------------------------------------------------------------------------
# 3. Empty-batch edge case
# ---------------------------------------------------------------------------


def test_predict_proba_empty_batch(synthetic_adapter):
    """N=0 must return (0,3) without raising — matching lstm_adapter behaviour."""
    empty = np.zeros((0, 60, 5), dtype=np.float32)
    out = synthetic_adapter.predict_proba(empty)
    assert out.shape == (0, 3)
    assert out.ndim == 2


# ---------------------------------------------------------------------------
# 4. predict() argmax
# ---------------------------------------------------------------------------


def test_predict_argmax_shape(synthetic_adapter, random_windows):
    preds = synthetic_adapter.predict(random_windows)
    assert preds.shape == (4,)


def test_predict_argmax_values_in_valid_classes(synthetic_adapter, random_windows):
    preds = synthetic_adapter.predict(random_windows)
    assert set(preds.tolist()).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# 5. Synthetic round-trip + Optuna-key filtering
# ---------------------------------------------------------------------------


def test_synthetic_roundtrip_loads_and_predicts(tmp_path, random_windows):
    """Build model → save state_dict + meta.json → load via adapter → predict.

    Revert-sensitivity: if XLSTMAdapter breaks the save/load path (wrong subdir
    name "xlstm/", wrong default checkpoint filename, or state_dict key mismatch)
    this test raises FileNotFoundError or RuntimeError.
    """
    adapter = _build_synthetic_adapter(tmp_path)
    out = adapter.predict_proba(random_windows)
    assert out.shape == (4, 3)


def test_optuna_keys_filtered_from_meta(tmp_path):
    """Extraneous keys (lr, warmup_steps, n_trials_completed) in meta.json
    hyperparams must be stripped before FVGxLSTMClassifier ctor is called.

    Revert-sensitivity: removing the HP-filter in XLSTMAdapter.__init__ will
    cause TypeError: unexpected keyword argument 'lr'.
    """
    extra = {"lr": 1e-3, "warmup_steps": 100, "n_trials_completed": 50, "batch_size": 32}
    adapter = _build_synthetic_adapter(tmp_path, extra_meta=extra)
    rng = np.random.default_rng(2)
    windows = rng.standard_normal((2, 60, 5)).astype(np.float32)
    out = adapter.predict_proba(windows)
    assert out.shape == (2, 3)


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


def test_determinism(synthetic_adapter, random_windows):
    """Eval mode + no_grad — identical inputs must yield bitwise-identical output.

    Revert-sensitivity: if model.train(False) is omitted in XLSTMAdapter.__init__
    dropout will randomise outputs and the array-equal check fails.
    """
    out1 = synthetic_adapter.predict_proba(random_windows)
    out2 = synthetic_adapter.predict_proba(random_windows)
    np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# 7. Missing checkpoint raises FileNotFoundError
# ---------------------------------------------------------------------------


def test_missing_checkpoint_raises(tmp_path):
    from src.inspect.adapters.xlstm_adapter import XLSTMAdapter
    with pytest.raises(FileNotFoundError):
        XLSTMAdapter(checkpoint_dir=tmp_path)
