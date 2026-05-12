"""Tests for scripts/eval/dual_fvg_compare.py output.

These tests run AFTER Part A training and dual_fvg_compare.py have been executed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPORT_DIR = Path("reports/rigor/2026-05-13/baselines")
JSON_PATH = REPORT_DIR / "dual_fvg_compare.json"
MD_PATH   = REPORT_DIR / "dual_fvg_compare.md"


@pytest.mark.skipif(
    not JSON_PATH.exists(),
    reason="dual_fvg_compare.json not yet generated — run scripts/eval/dual_fvg_compare.py first",
)
class TestDualFVGCompareReport:
    def test_json_exists(self):
        assert JSON_PATH.exists(), f"Missing: {JSON_PATH}"

    def test_md_exists(self):
        assert MD_PATH.exists(), f"Missing: {MD_PATH}"

    def test_json_has_all_model_label_keys(self):
        with open(JSON_PATH) as fh:
            data = json.load(fh)
        expected_keys = {
            "lstm_rawfvg", "lstm_validfvg",
            "xgb_rawfvg", "xgb_validfvg",
            "naive_rawfvg", "naive_validfvg",
            "gate_a5_passed",
        }
        assert expected_keys.issubset(data.keys()), (
            f"Missing keys: {expected_keys - set(data.keys())}"
        )

    def test_gate_a5_passed(self):
        with open(JSON_PATH) as fh:
            data = json.load(fh)
        assert data["gate_a5_passed"] is True, "Gate A5 not marked as passed in report"

    def test_f1_values_in_unit_range(self):
        with open(JSON_PATH) as fh:
            data = json.load(fh)
        for model_key in ["lstm_rawfvg", "lstm_validfvg", "xgb_rawfvg", "xgb_validfvg",
                           "naive_rawfvg", "naive_validfvg"]:
            f1 = data[model_key]["test_macro_f1"]
            assert 0.0 <= f1 <= 1.0, f"{model_key} F1={f1} outside [0,1]"

    def test_rawfvg_f1_higher_than_validfvg(self):
        """Raw FVG F1 should be higher than ValidFVG F1 — easier problem (~25% vs ~3% pos)."""
        with open(JSON_PATH) as fh:
            data = json.load(fh)
        lstm_raw   = data["lstm_rawfvg"]["test_macro_f1"]
        lstm_valid = data["lstm_validfvg"]["test_macro_f1"]
        assert lstm_raw > lstm_valid, (
            f"Expected raw-FVG LSTM F1 ({lstm_raw:.4f}) > ValidFVG ({lstm_valid:.4f}). "
            "If not, inspect the rawfvg splits for labelling errors."
        )
