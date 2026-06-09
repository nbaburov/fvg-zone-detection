"""test_registry.py — Unit tests for adapter auto-discovery and registry."""

from __future__ import annotations

import importlib
import logging

import pytest

from src.inspect.registry import list_available, load_adapters


def test_list_available_returns_sorted_list():
    names = list_available()
    assert isinstance(names, list)
    assert names == sorted(names), "list_available() should return a sorted list"


def test_list_available_includes_known_adapters():
    names = list_available()
    assert "lstm" in names
    assert "xgboost" in names


def test_list_available_has_no_duplicates():
    names = list_available()
    assert len(names) == len(set(names))


def test_load_adapters_raises_key_error_for_unknown():
    from pathlib import Path
    with pytest.raises(KeyError, match="Unknown adapter 'nonexistent'"):
        load_adapters(["nonexistent"], checkpoint_dir=Path("/tmp"))


def test_load_adapters_error_message_lists_available():
    from pathlib import Path
    try:
        load_adapters(["not_a_real_model"], checkpoint_dir=Path("/tmp"))
        pytest.fail("Expected KeyError")
    except KeyError as exc:
        msg = str(exc)
        # Available adapters should be mentioned
        assert "lstm" in msg or "xgboost" in msg


def test_load_adapters_missing_checkpoint_raises_file_not_found(tmp_path):
    """Loading a known adapter with a missing checkpoint raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_adapters(["lstm"], checkpoint_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        load_adapters(["xgboost"], checkpoint_dir=tmp_path)


def test_discover_resilient_to_single_adapter_import_failure(monkeypatch, caplog):
    """A broken adapter import must not prevent other adapters from registering.

    Monkeypatches importlib.import_module so that exactly one adapter module
    (xlstm_adapter) raises ImportError on import, then re-runs discovery from
    scratch.  The other adapters must still appear in the registry and a WARNING
    naming the broken module must be emitted.
    """
    import src.inspect.registry as registry_mod

    BROKEN_MODULE = "src.inspect.adapters.xlstm_adapter"
    _real_import = importlib.import_module

    def _patched_import(name: str, *args, **kwargs):
        if name == BROKEN_MODULE:
            raise ImportError(f"Simulated missing optional dep in {name}")
        return _real_import(name, *args, **kwargs)

    # Force fresh discovery by resetting the module-level cache.
    original_registry = registry_mod._REGISTRY
    monkeypatch.setattr(registry_mod, "_REGISTRY", None)
    monkeypatch.setattr(importlib, "import_module", _patched_import)

    try:
        with caplog.at_level(logging.WARNING, logger="src.inspect.registry"):
            names = list_available()
    finally:
        # Always restore cache so subsequent tests are unaffected.
        monkeypatch.setattr(registry_mod, "_REGISTRY", original_registry)

    # Core adapters must still be present — discovery did not abort on the error.
    assert "lstm" in names, f"lstm missing after broken xlstm import; got {names}"
    assert "xgboost" in names, f"xgboost missing after broken xlstm import; got {names}"
    assert "cnn_lstm" in names, f"cnn_lstm missing after broken xlstm import; got {names}"

    # A warning naming the broken module must have been logged.
    # (We do NOT assert xlstm absent: once a module is imported in a prior test the
    # class is already registered in ModelAdapter.__subclasses__ and re-discovery picks
    # it up regardless.  The important invariant is that the bad import path was hit,
    # warned about, and discovery continued rather than raising.)
    warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(BROKEN_MODULE in str(w) for w in warning_texts), (
        f"Expected a WARNING mentioning {BROKEN_MODULE!r}; got: {warning_texts}"
    )
