"""test_optuna_ns.py — Unit tests for _optuna_storage_and_study namespace helper.

Verifies:
  - h1 → legacy (unmodified) path and study name — backward-compat
  - 5m  → TF-namespaced path and study name
  - 15m → TF-namespaced path and study name
  - all four archs produce correct study names for h1 and non-h1
  - xgboost uses study_prefix="xgb" (legacy study name "xgb_fvg") while dir
    is still "xgboost" / "xgboost_5m" etc.

Filesystem: tmp_path is used as root; mkdir is allowed on tmp.
Optuna and training code are NOT imported.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helper import (no heavy deps)
# ---------------------------------------------------------------------------

from src.rigor.optuna_ns import _optuna_storage_and_study

_TODAY = date(2026, 6, 12)


# ---------------------------------------------------------------------------
# Parametrised: (arch, timeframe, study_prefix_override, expected_subdir, expected_study)
# study_prefix_override=None → use arch as prefix (default)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch,timeframe,study_prefix,expected_subdir,expected_study", [
    # h1 spy (default dataset) — flat scheme {arch}_h1_spy
    ("cnn_lstm",    "h1",  None,  "cnn_lstm_h1_spy",     "cnn_lstm_fvg"),
    ("lstm",        "h1",  None,  "lstm_h1_spy",         "lstm_fvg"),
    ("transformer", "h1",  None,  "transformer_h1_spy",  "transformer_fvg"),
    # xgboost: dir="xgboost_h1_spy", study prefix="xgb" (legacy study name)
    ("xgboost",     "h1",  "xgb", "xgboost_h1_spy",      "xgb_fvg"),
    # 5m spy — {arch}_5m_spy
    ("cnn_lstm",    "5m",  None,  "cnn_lstm_5m_spy",  "cnn_lstm_5m_fvg"),
    ("lstm",        "5m",  None,  "lstm_5m_spy",      "lstm_5m_fvg"),
    ("transformer", "5m",  None,  "transformer_5m_spy", "transformer_5m_fvg"),
    ("xgboost",     "5m",  "xgb", "xgboost_5m_spy",   "xgb_5m_fvg"),
    # 15m spy — {arch}_15m_spy
    ("cnn_lstm",    "15m", None,  "cnn_lstm_15m_spy", "cnn_lstm_15m_fvg"),
    ("lstm",        "15m", None,  "lstm_15m_spy",     "lstm_15m_fvg"),
    ("transformer", "15m", None,  "transformer_15m_spy", "transformer_15m_fvg"),
    ("xgboost",     "15m", "xgb", "xgboost_15m_spy",  "xgb_15m_fvg"),
])
def test_storage_and_study(
    tmp_path: Path,
    arch: str,
    timeframe: str,
    study_prefix: str | None,
    expected_subdir: str,
    expected_study: str,
) -> None:
    kwargs = {"today": _TODAY}
    if study_prefix is not None:
        kwargs["study_prefix"] = study_prefix

    storage_path, storage_url, study_name = _optuna_storage_and_study(
        arch, timeframe, tmp_path, **kwargs
    )

    # --- storage path ---
    expected_path = tmp_path / "checkpoints" / expected_subdir / f"optuna_{_TODAY}.db"
    assert storage_path == expected_path, (
        f"arch={arch} tf={timeframe}: expected path {expected_path}, got {storage_path}"
    )

    # --- storage URL ---
    assert storage_url == f"sqlite:///{expected_path}"

    # --- study name ---
    assert study_name == expected_study, (
        f"arch={arch} tf={timeframe}: expected study '{expected_study}', got '{study_name}'"
    )


# ---------------------------------------------------------------------------
# Explicit h1 backward-compat check: subdir must be exactly arch (no suffix)
# ---------------------------------------------------------------------------

def test_h1_flat_name(tmp_path: Path) -> None:
    """h1 timeframe uses flat name {arch}_h1_spy; study name has no _h1 suffix."""
    for arch in ("cnn_lstm", "lstm", "transformer", "xgboost"):
        storage_path, _, study_name = _optuna_storage_and_study(
            arch, "h1", tmp_path, today=_TODAY
        )
        expected_subdir = f"{arch}_h1_spy"
        assert storage_path.parent.name == expected_subdir, (
            f"h1 ckpt subdir should be '{expected_subdir}', got '{storage_path.parent.name}'"
        )
        assert "_h1" not in study_name, f"h1 study name must not contain '_h1': {study_name}"
        assert study_name.endswith("_fvg")


# ---------------------------------------------------------------------------
# Mkdir side-effect: directory is created
# ---------------------------------------------------------------------------

def test_ckpt_dir_created(tmp_path: Path) -> None:
    storage_path, _, _ = _optuna_storage_and_study("lstm", "5m", tmp_path, today=_TODAY)
    assert storage_path.parent.is_dir()
