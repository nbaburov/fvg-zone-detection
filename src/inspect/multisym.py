"""multisym.py — Single naming authority for checkpoint resolution.

Canonical checkpoint directory scheme
--------------------------------------
Every checkpoint directory follows:
    checkpoints/{arch}_{tf}_{dataset}[_tuned]/

- arch  in {cnn_lstm, lstm, transformer, xgboost, xlstm}
- tf    in {h1, 5m, 15m}
- dataset in {spy, multisym}
- optional _tuned suffix (only cnn_lstm_{5m,15m} are tuned)

Files DIRECTLY inside (FLAT — no inner /<arch>/ level):
    {arch}_seed{N}.pt   +   {arch}_seed{N}.meta.json     (torch)
    xgb_seed{N}.ubj     +   xgb_seed{N}.ubj.meta.json    (xgboost)

Examples:
    checkpoints/cnn_lstm_h1_spy/cnn_lstm_seed42.pt
    checkpoints/lstm_h1_multisym/lstm_seed0.pt
    checkpoints/xgboost_15m_multisym/xgb_seed42.ubj
    checkpoints/cnn_lstm_15m_multisym_tuned/cnn_lstm_seed42.pt
"""

from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

ALL_SEEDS: list[int] = [0, 17, 42, 123, 2024]

# Maps arch_name -> filename template only (no subdir — that is now derived
# from arch+tf+dataset+tuned via _dir_name()).
_ARCH_FILE_TEMPLATE: dict[str, str] = {
    "lstm":        "lstm_seed{seed}.pt",
    "cnn_lstm":    "cnn_lstm_seed{seed}.pt",
    "transformer": "transformer_seed{seed}.pt",
    "xgboost":     "xgb_seed{seed}.ubj",
    "xlstm":       "xlstm_seed{seed}.pt",
}

# Default seed used when no explicit path is given (bare name → seed0).
_DEFAULT_SEED = 0


# -----------------------------------------------------------------------
# Naming helpers
# -----------------------------------------------------------------------


def _dir_name(arch: str, tf: str, dataset: str, tuned: bool = False) -> str:
    """Return the flat directory name for a checkpoint set.

    Parameters
    ----------
    arch : str
        Architecture name (e.g. ``"lstm"``, ``"xgboost"``).
    tf : str
        Timeframe token (``"h1"``, ``"5m"``, ``"15m"``).
    dataset : str
        Dataset token (``"spy"`` or ``"multisym"``).
    tuned : bool
        When True, appends ``_tuned`` suffix.

    Returns
    -------
    str
        E.g. ``"lstm_h1_multisym"``, ``"cnn_lstm_15m_multisym_tuned"``.
    """
    name = f"{arch}_{tf}_{dataset}"
    if tuned:
        name = f"{name}_tuned"
    return name


# -----------------------------------------------------------------------
# Backward-compatibility helpers (used by tests + inspect_models.py)
# -----------------------------------------------------------------------


def _multisym_dirs_for_token(token: str, dataset: str = "multisym") -> dict[str, tuple[str, str]]:
    """Return the arch -> (subdir, filename_template) map for *token* and *dataset*.

    This is the single source of truth for directory names.  The returned
    subdir is the FLAT directory name (no inner arch subdir).

    Parameters
    ----------
    token : str
        Timeframe token (``"h1"``, ``"5m"``, ``"15m"``).
    dataset : str
        Dataset token (``"spy"`` or ``"multisym"``).  Defaults to ``"multisym"``
        to preserve backward-compatible behaviour in ``--all-seeds`` resolution.
    """
    return {
        arch: (_dir_name(arch, token, dataset), tmpl)
        for arch, tmpl in _ARCH_FILE_TEMPLATE.items()
    }


# H1-multisym MULTISYM_DIRS — kept for backward-compatibility with callers
# that imported this dict directly (e.g. eval_spy_test.py).
# Now points at new flat h1-multisym names.
MULTISYM_DIRS: dict[str, tuple[str, str]] = _multisym_dirs_for_token("h1", "multisym")


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------


def resolve_multisym_checkpoints(
    checkpoint_dir: Path,
    token: str = "h1",
    dataset: str = "multisym",
    tuned: bool = False,
) -> dict[str, list[tuple[int, Path]]]:
    """Resolve all available checkpoints per arch for *token* and *dataset*.

    Parameters
    ----------
    checkpoint_dir : Path
        Root checkpoints directory (e.g. ``Path("checkpoints")``).
    token : str
        Timeframe token (``"h1"``, ``"5m"``, ``"15m"``).  Defaults to ``"h1"``.
    dataset : str
        Dataset token (``"spy"`` or ``"multisym"``).  Defaults to ``"multisym"``.
    tuned : bool
        When True, resolves the ``_tuned`` variant (only cnn_lstm 5m/15m have these).

    Returns
    -------
    dict[str, list[tuple[int, Path]]]
        ``arch_name -> [(seed, path), ...]`` for every checkpoint file that
        exists on disk.  Archs with zero found files are omitted; callers
        should warn and skip them.
    """
    result: dict[str, list[tuple[int, Path]]] = {}
    for arch, tmpl in _ARCH_FILE_TEMPLATE.items():
        dir_name = _dir_name(arch, token, dataset, tuned=tuned)
        found: list[tuple[int, Path]] = []
        for seed in ALL_SEEDS:
            fname = tmpl.format(seed=seed)
            p = checkpoint_dir / dir_name / fname
            if p.exists():
                found.append((seed, p))
        if found:
            result[arch] = found
    return result


def default_checkpoint_path(
    arch: str,
    checkpoint_dir: Path,
    token: str = "h1",
    dataset: str = "spy",
    tuned: bool = False,
) -> Path | None:
    """Return the seed-0 checkpoint path for *arch* at *token*/*dataset*, or None.

    This is the path used when a bare arch name is specified without an explicit
    ``name:path`` override.  Default dataset is ``"spy"`` — bare names resolve
    to the H1 SPY checkpoints.

    Parameters
    ----------
    arch : str
        Architecture name (e.g. ``"lstm"``, ``"xgboost"``).
    checkpoint_dir : Path
        Root checkpoints directory.
    token : str
        Timeframe token.  Defaults to ``"h1"``.
    dataset : str
        Dataset token.  Defaults to ``"spy"`` (bare-name = H1 SPY default).
    tuned : bool
        When True, resolves the ``_tuned`` variant.

    Returns
    -------
    Path | None
        Absolute path to the default checkpoint file, or ``None`` if the arch
        is unknown or the file does not exist.
    """
    if arch not in _ARCH_FILE_TEMPLATE:
        return None
    dir_name = _dir_name(arch, token, dataset, tuned=tuned)
    tmpl = _ARCH_FILE_TEMPLATE[arch]
    fname = tmpl.format(seed=_DEFAULT_SEED)
    p = checkpoint_dir / dir_name / fname
    return p if p.exists() else None
