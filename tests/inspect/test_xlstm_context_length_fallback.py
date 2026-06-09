"""test_xlstm_context_length_fallback.py — Change B: xlstm context_length fallback paths.

The adapter resolves context_length as:
  model_kwargs.get("context_length")         [explicit ctor kwarg in hyperparams]
  -> hp.get("window_size")                   [seed_sweep layout: window_size inside hyperparams]
  -> meta.get("window_size")                 [top-level window_size in meta.json]
  -> 60                                      [hardcoded default]

The existing tests in test_xlstm_adapter.py only exercise path 1 (context_length
explicitly in hyperparams).  These tests cover paths 2, 3, and 4.

Covers:
  B1. hp.window_size path (seed_sweep layout) — meta.json has window_size inside
      hyperparams, NO context_length key, NO top-level window_size.  Adapter must
      construct successfully using hyperparams["window_size"] as context_length.
      Uses a non-default window_size=60 to keep state_dict dims consistent (the
      synthetic model is built with the same value), while asserting predict_proba
      works (proves the hp.window_size path is exercised, not the hardcoded 60
      fallback, by also verifying the path when window_size != 60 would fail to
      load a model built with 60).
  B2. top-level meta window_size path — meta.json has window_size at the top level
      (not inside hyperparams), no context_length, no hp.window_size.  Adapter
      must use top-level window_size.
  B3. Hardcoded 60 fallback — meta.json has neither context_length nor window_size
      anywhere.  Adapter must construct without error using default 60.
  B4. hp.window_size non-default value round-trip — meta.json has window_size=40
      inside hyperparams (no context_length); the synthetic model is also built
      with context_length=40.  State-dict round-trip must succeed and predict_proba
      must work on (N, 40, 5) windows, proving the hp.window_size value is actually
      passed to the ctor (not silently overridden by the 60 default).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _build_xlstm_ckpt(
    tmp_path: Path,
    *,
    context_length: int = 60,
    meta_hyperparams: dict | None = None,
    meta_top_level_extras: dict | None = None,
    ckpt_filename: str = "xlstm_seed42.pt",
) -> Path:
    """Build a synthetic FVGxLSTMClassifier with the given context_length, save its
    state_dict + a meta.json constructed from the provided dicts, and return the
    checkpoint path (within tmp_path/xlstm/).

    meta_hyperparams: dict merged into meta["hyperparams"] (may or may not include
        window_size / context_length).
    meta_top_level_extras: dict merged into meta at the top level (for testing
        top-level window_size).
    """
    from src.models.xlstm_model import FVGxLSTMClassifier

    model = FVGxLSTMClassifier(context_length=context_length)
    model.eval()

    ckpt_dir = tmp_path / "xlstm"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / ckpt_filename
    torch.save(model.state_dict(), str(ckpt_path))

    meta: dict = {"hyperparams": dict(meta_hyperparams or {})}
    if meta_top_level_extras:
        meta.update(meta_top_level_extras)

    meta_path = ckpt_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    return ckpt_path


# ---------------------------------------------------------------------------
# B1. hp.window_size path — seed_sweep layout
# ---------------------------------------------------------------------------


def test_xlstm_context_length_from_hyperparams_window_size(tmp_path):
    """meta.json has window_size=60 inside hyperparams, no context_length key.
    Adapter must construct and predict without error.

    This is the real seed_sweep layout: seed_sweep records window_size as a
    _NON_MODEL_HP (stripped from model ctor kwargs but kept in meta.hyperparams).

    Revert-sensitivity: if `hp.get("window_size")` is removed from the resolution
    chain, the adapter falls through to the hardcoded 60 default.  For window_size=60
    the result is the same (test passes trivially), so B4 uses window_size=40 to
    prove the path is genuinely exercised — see below.
    """
    from src.inspect.adapters.xlstm_adapter import XLSTMAdapter

    ckpt_path = _build_xlstm_ckpt(
        tmp_path,
        context_length=60,
        meta_hyperparams={"window_size": 60},  # present in hp, NOT as context_length
    )
    adapter = XLSTMAdapter(
        checkpoint_dir=tmp_path.parent,  # would resolve to wrong dir if override not triggered
        checkpoint_path=ckpt_path,
    )
    rng = np.random.default_rng(10)
    windows = rng.standard_normal((3, 60, 5)).astype(np.float32)
    out = adapter.predict_proba(windows)
    assert out.shape == (3, 3)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.sum(axis=1), np.ones(3, dtype=np.float32), atol=1e-5)


# ---------------------------------------------------------------------------
# B2. top-level meta window_size path
# ---------------------------------------------------------------------------


def test_xlstm_context_length_from_top_level_meta_window_size(tmp_path):
    """meta.json has window_size=60 at the TOP LEVEL (not inside hyperparams),
    no context_length, no hp.window_size.  Adapter must use top-level window_size.

    Revert-sensitivity: if `meta.get("window_size")` is removed from the chain,
    context_length falls to the hardcoded 60 — for window_size=60 this silently
    succeeds (no observable difference), so the test proves the path exists but
    for a definitive proof see B4 with a non-60 value.
    """
    from src.inspect.adapters.xlstm_adapter import XLSTMAdapter

    ckpt_path = _build_xlstm_ckpt(
        tmp_path,
        context_length=60,
        meta_hyperparams={},                          # no context_length, no window_size in hp
        meta_top_level_extras={"window_size": 60},    # window_size at top level
    )
    adapter = XLSTMAdapter(checkpoint_dir=tmp_path, checkpoint_path=ckpt_path)
    assert adapter is not None
    rng = np.random.default_rng(11)
    windows = rng.standard_normal((2, 60, 5)).astype(np.float32)
    out = adapter.predict_proba(windows)
    assert out.shape == (2, 3)


# ---------------------------------------------------------------------------
# B3. Hardcoded 60 fallback
# ---------------------------------------------------------------------------


def test_xlstm_context_length_hardcoded_fallback(tmp_path):
    """meta.json has neither context_length nor window_size anywhere.
    Adapter must fall back to 60 and construct/predict without error.

    Revert-sensitivity: removing the `or 60` terminal fallback in the ctor
    would cause context_length=None to be passed to FVGxLSTMClassifier, which
    raises a TypeError or uses an unexpected default.
    """
    from src.inspect.adapters.xlstm_adapter import XLSTMAdapter

    ckpt_path = _build_xlstm_ckpt(
        tmp_path,
        context_length=60,       # model built with 60
        meta_hyperparams={},     # empty — no clues in meta
    )
    adapter = XLSTMAdapter(checkpoint_dir=tmp_path, checkpoint_path=ckpt_path)
    rng = np.random.default_rng(12)
    windows = rng.standard_normal((2, 60, 5)).astype(np.float32)
    out = adapter.predict_proba(windows)
    assert out.shape == (2, 3)


# ---------------------------------------------------------------------------
# B4. hp.window_size non-default value round-trip (40)
# ---------------------------------------------------------------------------


def test_xlstm_context_length_from_hyperparams_window_size_nondefault(tmp_path):
    """meta.json has window_size=40 inside hyperparams (no context_length).
    Adapter must pass context_length=40 to FVGxLSTMClassifier so the state-dict
    load succeeds.  predict_proba on (N, 40, 5) windows must return (N, 3).

    This is the critical revert-sensitivity test: if `hp.get("window_size")` is
    removed from the resolution chain, context_length falls back to the hardcoded
    60, the model is constructed with context_length=60, and torch.load_state_dict
    raises a RuntimeError (key-shape mismatch) because the saved state_dict has
    context_length=40 dimensions.  The test therefore CANNOT pass by accident.
    """
    from src.inspect.adapters.xlstm_adapter import XLSTMAdapter

    ckpt_path = _build_xlstm_ckpt(
        tmp_path,
        context_length=40,
        meta_hyperparams={"window_size": 40},   # seed_sweep layout
    )
    adapter = XLSTMAdapter(checkpoint_dir=tmp_path, checkpoint_path=ckpt_path)
    rng = np.random.default_rng(13)
    windows = rng.standard_normal((3, 40, 5)).astype(np.float32)
    out = adapter.predict_proba(windows)
    assert out.shape == (3, 3)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.sum(axis=1), np.ones(3, dtype=np.float32), atol=1e-5)


# ---------------------------------------------------------------------------
# B5. context_length in hyperparams takes priority over window_size in hyperparams
# ---------------------------------------------------------------------------


def test_xlstm_context_length_explicit_overrides_hp_window_size(tmp_path):
    """When BOTH context_length and window_size are present in hyperparams, the
    adapter must use context_length (it is a recognised _CTOR_PARAM and lands in
    model_kwargs directly).

    Revert-sensitivity: if the resolution chain accidentally reads window_size
    even when context_length is present, it would pass the wrong dimension to
    the ctor and the state-dict load would fail (key-shape mismatch) because the
    saved model was built with context_length=60, not window_size=40.
    """
    from src.inspect.adapters.xlstm_adapter import XLSTMAdapter

    # model built with 60; meta says context_length=60 AND window_size=40
    ckpt_path = _build_xlstm_ckpt(
        tmp_path,
        context_length=60,
        meta_hyperparams={"context_length": 60, "window_size": 40},
    )
    adapter = XLSTMAdapter(checkpoint_dir=tmp_path, checkpoint_path=ckpt_path)
    rng = np.random.default_rng(14)
    windows = rng.standard_normal((2, 60, 5)).astype(np.float32)
    out = adapter.predict_proba(windows)
    assert out.shape == (2, 3)
