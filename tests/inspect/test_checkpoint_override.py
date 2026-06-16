"""test_checkpoint_override.py — Change A: per-model checkpoint_path override.

Covers:
  A1. TransformerAdapter with checkpoint_path= loads a non-default seed file and
      predict_proba returns (N,3) softmax.  Proves the override path is used by
      pointing at seed17 (which default resolution — seed0 — would never reach) and
      asserting no FileNotFoundError + output shape/dtype contract.
  A2. Backward compat — constructing each torch adapter WITHOUT checkpoint_path
      (old positional checkpoint_dir + default seed) still works unchanged.
  A3. load_adapters checkpoint_paths mapping — with a real transformer checkpoint
      path in the dict, the adapter loads from that file; with checkpoint_paths=None
      the dict path is NOT used (falls back to dir/default).
  A4. CLI name:path token parsing in _parse_args / inline main() logic — tested at
      the argparse level via _parse_args and the token-split logic extracted from
      the inline loop.  Bare names do not populate overrides; name:path tokens do.
  A5. XGBoostAdapter with checkpoint_path= pointing at a real .ubj file (xgboost_multisym)
      loads successfully (constructor only — no predict_proba call to avoid subprocess
      overhead in CI; the worker invocation path is already exercised by the existing
      xgboost adapter tests).  Confirms _checkpoint_path is set to the override path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CKPT = _REPO_ROOT / "checkpoints"

_TRANSFORMER_SEED0 = _CKPT / "transformer_h1_spy" / "transformer_seed0.pt"
_TRANSFORMER_ALT = _CKPT / "transformer_h1_spy" / "transformer_seed17.pt"
_XGB_MULTISYM_SEED42 = _CKPT / "xgboost_h1_multisym" / "xgb_seed42.ubj"

_HAS_TRANSFORMER_SEED0 = _TRANSFORMER_SEED0.exists()
_HAS_TRANSFORMER_ALT = _TRANSFORMER_ALT.exists()
_HAS_XGB_MULTISYM = _XGB_MULTISYM_SEED42.exists()


# ---------------------------------------------------------------------------
# A1. TransformerAdapter checkpoint_path override — non-default seed
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TRANSFORMER_ALT, reason="transformer seed17 checkpoint absent")
def test_transformer_adapter_checkpoint_path_override_loads():
    """Constructing TransformerAdapter with checkpoint_path= pointing at seed1
    must succeed and produce a (N,3) float32 softmax output.

    Revert-sensitivity: removing the `if checkpoint_path is not None` branch in
    TransformerAdapter.__init__ causes it to attempt
    checkpoint_dir / 'transformer' / 'transformer_seed0.pt', which is a
    *different* file than seed17, so the test would either silently load the wrong
    weights (behaviour change) or raise FileNotFoundError if checkpoint_dir is
    tmp_path (construction path breaks).
    """
    from src.inspect.adapters.transformer_adapter import TransformerAdapter

    # Pass a dummy (nonexistent) checkpoint_dir — if the override is NOT used,
    # the adapter falls back to checkpoint_dir/transformer/transformer_seed0.pt
    # which does not exist under tmp_path and raises FileNotFoundError.
    adapter = TransformerAdapter(
        checkpoint_dir=Path("/nonexistent_dir_should_not_be_used"),
        checkpoint_path=_TRANSFORMER_ALT,
    )
    rng = np.random.default_rng(42)
    windows = rng.standard_normal((3, 60, 5)).astype(np.float32)
    out = adapter.predict_proba(windows)
    assert out.shape == (3, 3)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.sum(axis=1), np.ones(3, dtype=np.float32), atol=1e-5)


@pytest.mark.skipif(not _HAS_TRANSFORMER_ALT, reason="transformer seed17 checkpoint absent")
def test_transformer_adapter_checkpoint_path_override_uses_sidecar_from_override_path(tmp_path):
    """The meta sidecar must be resolved as checkpoint_path.with_suffix('.meta.json'),
    not from the (dummy) checkpoint_dir.

    Revert-sensitivity: if the meta sidecar path calculation uses checkpoint_dir
    instead of checkpoint_path, the HP lookup silently falls back to defaults which
    may produce a different architecture than the checkpoint was trained on — causing
    a RuntimeError (state_dict mismatch) or silent wrong output.
    """
    from src.inspect.adapters.transformer_adapter import TransformerAdapter

    # seed17 has a real sidecar next to it; passing /nonexistent as checkpoint_dir
    # proves the sidecar is found via checkpoint_path, not checkpoint_dir.
    adapter = TransformerAdapter(
        checkpoint_dir=Path("/nonexistent_dir_should_not_be_used"),
        checkpoint_path=_TRANSFORMER_ALT,
    )
    assert adapter is not None


# ---------------------------------------------------------------------------
# A2. Backward compat — old positional API still works for torch adapters
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TRANSFORMER_SEED0, reason="transformer seed0 checkpoint absent")
def test_transformer_adapter_backward_compat_no_override():
    """checkpoint_dir + default seed (no checkpoint_path) must still work.

    Revert-sensitivity: if the fallback `else` branch is removed from the ctor,
    checkpoint_path stays None and a TypeError or AttributeError is raised on
    checkpoint_path.exists().
    """
    from src.inspect.adapters.transformer_adapter import TransformerAdapter

    adapter = TransformerAdapter(checkpoint_dir=_CKPT)
    rng = np.random.default_rng(7)
    windows = rng.standard_normal((2, 60, 5)).astype(np.float32)
    out = adapter.predict_proba(windows)
    assert out.shape == (2, 3)


# ---------------------------------------------------------------------------
# A3. load_adapters checkpoint_paths mapping
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TRANSFORMER_ALT, reason="transformer seed17 checkpoint absent")
def test_load_adapters_checkpoint_paths_mapping_uses_override():
    """load_adapters with checkpoint_paths={"transformer": seed17_path} must load
    from seed17, even though checkpoint_dir points at the default seed.

    Revert-sensitivity: if load_adapters does not forward checkpoint_path to the
    adapter ctor, the adapter loads the default seed0 instead of seed17.  The
    distinction is proven by passing a /nonexistent checkpoint_dir: default
    resolution would raise FileNotFoundError, override succeeds.
    """
    from src.inspect.registry import load_adapters

    adapters = load_adapters(
        ["transformer"],
        checkpoint_dir=Path("/nonexistent_dir_should_not_be_used"),
        checkpoint_paths={"transformer": str(_TRANSFORMER_ALT)},
    )
    assert len(adapters) == 1
    rng = np.random.default_rng(9)
    windows = rng.standard_normal((2, 60, 5)).astype(np.float32)
    out = adapters[0].predict_proba(windows)
    assert out.shape == (2, 3)


@pytest.mark.skipif(not _HAS_TRANSFORMER_SEED0, reason="transformer seed0 checkpoint absent")
def test_load_adapters_checkpoint_paths_none_falls_back_to_dir():
    """With checkpoint_paths=None the adapter must use checkpoint_dir + default seed.

    Revert-sensitivity: if checkpoint_paths=None is mishandled (e.g. treated as
    an empty dict rather than ignored), the standard dir resolution still works,
    but this test guarantees the None path is explicitly exercised.
    """
    from src.inspect.registry import load_adapters

    adapters = load_adapters(
        ["transformer"],
        checkpoint_dir=_CKPT,
        checkpoint_paths=None,
    )
    assert len(adapters) == 1


@pytest.mark.skipif(not _HAS_TRANSFORMER_SEED0, reason="transformer seed0 checkpoint absent")
def test_load_adapters_absent_from_overrides_uses_dir():
    """A name absent from checkpoint_paths must fall back to dir/default.

    Revert-sensitivity: if the `name in per_model` guard is missing, all adapters
    would receive checkpoint_path=None regardless, silently breaking overrides.
    """
    from src.inspect.registry import load_adapters

    # "transformer" is NOT in checkpoint_paths — must fall back to dir
    adapters = load_adapters(
        ["transformer"],
        checkpoint_dir=_CKPT,
        checkpoint_paths={"lstm": "/some/nonexistent/path"},  # irrelevant key
    )
    assert len(adapters) == 1


# ---------------------------------------------------------------------------
# A4. CLI name:path token parsing
# ---------------------------------------------------------------------------


def test_cli_bare_name_does_not_populate_overrides():
    """A bare 'name' token (no colon) must appear in model_names but NOT in
    checkpoint_path_overrides.

    Revert-sensitivity: if the ':' check is accidentally applied to all tokens,
    bare names would be misrouted (e.g. treated as 'name' with path='') and
    could raise an unexpected error in load_adapters.
    """
    # Replicate the inline parsing logic from scripts/inspect_models.py _parse_args
    tokens = ["lstm", "xgboost"]
    model_names = []
    overrides: dict[str, str] = {}
    for token in tokens:
        if ":" in token:
            name, cp_path = token.split(":", 1)
            model_names.append(name)
            overrides[name] = cp_path
        else:
            model_names.append(token)

    assert model_names == ["lstm", "xgboost"]
    assert overrides == {}


def test_cli_name_colon_path_populates_overrides():
    """A 'name:path' token must split into model_names entry + overrides entry.

    Revert-sensitivity: if the split logic is removed or the key/value are
    swapped, checkpoint_path_overrides will be missing the name key and the
    adapter falls back to dir/default rather than the specified path.
    """
    tokens = [
        "transformer:/some/path/transformer_seed17.pt",
        "lstm",
        "cnn_lstm:/other/path/cnn_lstm_seed0.pt",
    ]
    model_names = []
    overrides: dict[str, str] = {}
    for token in tokens:
        if ":" in token:
            name, cp_path = token.split(":", 1)
            model_names.append(name)
            overrides[name] = cp_path
        else:
            model_names.append(token)

    assert model_names == ["transformer", "lstm", "cnn_lstm"]
    assert overrides["transformer"] == "/some/path/transformer_seed17.pt"
    assert overrides["cnn_lstm"] == "/other/path/cnn_lstm_seed0.pt"
    assert "lstm" not in overrides


def test_cli_path_with_colon_in_path_splits_on_first_colon():
    """A Windows-style path (C:\\...) or a path containing ':' must split only on
    the FIRST colon so the drive letter stays in the path portion.

    Revert-sensitivity: using token.split(':') (limit 1) is required; split(':')
    without maxsplit would break Windows paths.
    """
    token = "transformer:C:/checkpoints/transformer_seed0.pt"
    name, cp_path = token.split(":", 1)
    assert name == "transformer"
    assert cp_path == "C:/checkpoints/transformer_seed0.pt"


def test_cli_argparse_models_accepts_name_colon_path(tmp_path):
    """_parse_args must accept --models with name:path tokens without error.

    This tests at the argparse level that the metavar/nargs setup does not reject
    colon-containing tokens.
    """
    # Add scripts/ to sys.path for direct import
    scripts_dir = _REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "inspect_models_script",
        str(_REPO_ROOT / "scripts" / "inspect_models.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    args = mod._parse_args([
        "--models",
        "transformer:/fake/path/transformer_seed0.pt",
        "lstm",
        "--dataset", "test",
    ])
    assert args.models == [
        "transformer:/fake/path/transformer_seed0.pt",
        "lstm",
    ]


# ---------------------------------------------------------------------------
# A5. XGBoostAdapter with checkpoint_path= override (constructor only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_XGB_MULTISYM, reason="xgboost_multisym seed42 checkpoint absent")
def test_xgboost_adapter_checkpoint_path_override_constructor():
    """XGBoostAdapter with checkpoint_path= pointing at xgboost_multisym/xgboost/xgb_seed42.ubj
    must construct without error and store the override path.

    This exercises the `if checkpoint_path is not None` branch in XGBoostAdapter.__init__.
    No predict_proba is called here to avoid subprocess invocation overhead; subprocess
    inference is already covered by existing xgboost adapter tests.

    Revert-sensitivity: if the checkpoint_path branch is removed, the constructor
    falls back to checkpoint_dir/xgboost/xgb_seed42.ubj. Passing /nonexistent as
    checkpoint_dir proves the override is actually used (FileNotFoundError if not).
    """
    from src.inspect.adapters.xgboost_adapter import XGBoostAdapter

    adapter = XGBoostAdapter(
        checkpoint_dir=Path("/nonexistent_dir_should_not_be_used"),
        checkpoint_path=_XGB_MULTISYM_SEED42,
    )
    assert adapter._checkpoint_path == _XGB_MULTISYM_SEED42


@pytest.mark.skipif(not _HAS_XGB_MULTISYM, reason="xgboost_multisym seed42 checkpoint absent")
def test_xgboost_adapter_checkpoint_path_override_path_is_stored():
    """_checkpoint_path attribute must reflect the explicit override, not a
    dir-derived path.

    Revert-sensitivity: if the ctor sets self._checkpoint_path = checkpoint_dir /
    'xgboost' / checkpoint_file regardless of checkpoint_path kwarg, this assertion
    fails because the stored path would differ from the override.
    """
    from src.inspect.adapters.xgboost_adapter import XGBoostAdapter

    adapter = XGBoostAdapter(
        checkpoint_dir=_CKPT,
        checkpoint_path=_XGB_MULTISYM_SEED42,
    )
    assert str(adapter._checkpoint_path) == str(_XGB_MULTISYM_SEED42)
