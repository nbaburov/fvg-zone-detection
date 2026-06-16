"""optuna_ns.py — Namespace helpers for Optuna storage + study naming.

Ensures that tuning runs at different timeframes use isolated SQLite databases
and study names, preventing a 5m or 15m run from resuming an h1 study.

Rule (backward-compat):
  - timeframe "h1"  → no suffix on either path or name (preserves existing runs)
  - any other token → suffix appended (e.g. "lstm_5m", study "lstm_5m_fvg")
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from src.rigor.seed_sweep import _ckpt_subdir


def _optuna_storage_and_study(
    arch: str,
    timeframe: str,
    root: Path,
    today: date | None = None,
    study_prefix: str | None = None,
    dataset: str = "spy",
) -> tuple[Path, str, str]:
    """Return ``(storage_path, storage_url, study_name)`` for *arch* + *timeframe*.

    Parameters
    ----------
    arch:
        Model architecture token used for the checkpoint subdirectory, e.g.
        ``"cnn_lstm"``, ``"lstm"``, ``"transformer"``, ``"xgboost"``.
    timeframe:
        Timeframe token, e.g. ``"h1"``, ``"5m"``, ``"15m"``.
    root:
        Project root ``Path`` (the directory that contains ``checkpoints/``).
    today:
        Date used for the db filename; defaults to ``date.today()``.
        Injected in tests for determinism.
    study_prefix:
        Base token used in the study name instead of *arch*.  Needed when the
        legacy study name differs from the arch dir name (e.g. XGBoost uses
        ``"xgb"`` not ``"xgboost"``).  Defaults to *arch*.
    dataset:
        Dataset token (``"spy"`` or ``"multisym"``).  Used to build the flat
        checkpoint subdir name ``{arch}_{tf}_{dataset}``.  Defaults to ``"spy"``.

    Returns
    -------
    storage_path : Path
        Absolute path to the SQLite file,
        e.g. ``<root>/checkpoints/cnn_lstm_5m_spy/optuna_2026-06-12.db``.
    storage_url : str
        SQLAlchemy URL, e.g. ``sqlite:///<storage_path>``.
    study_name : str
        Optuna study name, e.g. ``"cnn_lstm_5m_fvg"`` or ``"cnn_lstm_fvg"``
        for h1.
    """
    if today is None:
        today = date.today()
    if study_prefix is None:
        study_prefix = arch

    subdir = _ckpt_subdir(arch, timeframe, dataset)   # e.g. "cnn_lstm_h1_spy", "lstm_5m_spy"
    ckpt_dir = root / "checkpoints" / subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    storage_path = ckpt_dir / f"optuna_{today}.db"
    storage_url = f"sqlite:///{storage_path}"

    # h1 → legacy name (no suffix); others get prefix_tf suffix
    if timeframe == "h1":
        study_name = f"{study_prefix}_fvg"
    else:
        study_name = f"{study_prefix}_{timeframe}_fvg"

    return storage_path, storage_url, study_name
