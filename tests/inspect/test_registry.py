"""test_registry.py — Unit tests for adapter auto-discovery and registry."""

from __future__ import annotations

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
