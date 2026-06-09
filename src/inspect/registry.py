"""registry.py — Auto-discovery and instantiation of ModelAdapter subclasses.

Scans ``src/inspect/adapters/`` at import time. No central if/else — adding a new
adapter is: create the file, set ``name``, implement ``predict_proba``. Done.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.inspect.base import ModelAdapter

logger = logging.getLogger(__name__)

# Populated by _discover() on first access
_REGISTRY: dict[str, type["ModelAdapter"]] | None = None


def _discover() -> dict[str, type["ModelAdapter"]]:
    """Import all modules in src/inspect/adapters/ and collect ModelAdapter subclasses."""
    from src.inspect.base import ModelAdapter  # local import to avoid circular

    adapters_pkg = "src.inspect.adapters"
    adapters_path = Path(__file__).parent / "adapters"

    for _finder, module_name, _ispkg in pkgutil.iter_modules([str(adapters_path)]):
        full_name = f"{adapters_pkg}.{module_name}"
        try:
            importlib.import_module(full_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skipping adapter %r — import failed: %s: %s",
                full_name,
                type(exc).__name__,
                exc,
            )
            continue

    registry: dict[str, type[ModelAdapter]] = {}
    for cls in _all_subclasses(ModelAdapter):
        if hasattr(cls, "name") and isinstance(cls.name, str):
            registry[cls.name] = cls
    return registry


def _all_subclasses(cls: type) -> list[type]:
    """Recursively collect all subclasses."""
    result = []
    for sub in cls.__subclasses__():
        result.append(sub)
        result.extend(_all_subclasses(sub))
    return result


def _get_registry() -> dict[str, type["ModelAdapter"]]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _discover()
    return _REGISTRY


def list_available() -> list[str]:
    """Return sorted list of registered adapter names."""
    return sorted(_get_registry().keys())


def load_adapters(
    names: list[str],
    checkpoint_dir: Path,
    checkpoint_paths: dict[str, str] | None = None,
    **kwargs,
) -> list["ModelAdapter"]:
    """Instantiate adapters by name.

    Parameters
    ----------
    names : list[str]
        Adapter names to load. Each must be registered.
    checkpoint_dir : Path
        Passed to each adapter constructor as the first positional argument.
        Used for any adapter whose name is not present in ``checkpoint_paths``.
    checkpoint_paths : dict[str, str] | None
        Optional per-model checkpoint file overrides. Keys are adapter names;
        values are absolute or relative paths to specific checkpoint files.
        When a name is present here the adapter receives
        ``checkpoint_path=<value>`` and skips the default
        ``<checkpoint_dir>/<arch>/<file>`` resolution.  Names absent from
        this dict fall back to the standard ``checkpoint_dir`` behaviour.
    **kwargs
        Extra keyword arguments forwarded to each adapter constructor.

    Returns
    -------
    list[ModelAdapter]
        Instantiated adapters in the same order as ``names``.

    Raises
    ------
    KeyError
        If any name is not registered.
    """
    registry = _get_registry()
    per_model = checkpoint_paths or {}
    adapters = []
    for name in names:
        if name not in registry:
            available = ", ".join(sorted(registry.keys()))
            raise KeyError(
                f"Unknown adapter '{name}'. Available: {available}"
            )
        extra = dict(kwargs)
        if name in per_model:
            extra["checkpoint_path"] = per_model[name]
        adapters.append(registry[name](checkpoint_dir, **extra))
    return adapters
