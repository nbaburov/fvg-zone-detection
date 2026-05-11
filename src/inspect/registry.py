"""registry.py — Auto-discovery and instantiation of ModelAdapter subclasses.

Scans ``src/inspect/adapters/`` at import time. No central if/else — adding a new
adapter is: create the file, set ``name``, implement ``predict_proba``. Done.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.inspect.base import ModelAdapter

# Populated by _discover() on first access
_REGISTRY: dict[str, type["ModelAdapter"]] | None = None


def _discover() -> dict[str, type["ModelAdapter"]]:
    """Import all modules in src/inspect/adapters/ and collect ModelAdapter subclasses."""
    from src.inspect.base import ModelAdapter  # local import to avoid circular

    adapters_pkg = "src.inspect.adapters"
    adapters_path = Path(__file__).parent / "adapters"

    for _finder, module_name, _ispkg in pkgutil.iter_modules([str(adapters_path)]):
        full_name = f"{adapters_pkg}.{module_name}"
        importlib.import_module(full_name)

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


def load_adapters(names: list[str], checkpoint_dir: Path, **kwargs) -> list["ModelAdapter"]:
    """Instantiate adapters by name.

    Parameters
    ----------
    names : list[str]
        Adapter names to load. Each must be registered.
    checkpoint_dir : Path
        Passed to each adapter constructor as the first positional argument.
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
    adapters = []
    for name in names:
        if name not in registry:
            available = ", ".join(sorted(registry.keys()))
            raise KeyError(
                f"Unknown adapter '{name}'. Available: {available}"
            )
        adapters.append(registry[name](checkpoint_dir, **kwargs))
    return adapters
