"""test_cell_matrix.py — Unit tests for the pure cell-matrix builder.

Tests §4b legality rules (§4.1 empty-set, §4.2 out-of-matrix, §4.3 duplicate
real ticker) plus arbitrary subset cartesian counts, real-XOR-sim invariant,
--list-cells output shape, and non-example matrix shapes.

All tests run without Alpaca, torch, or any adapter — build_cell_matrix is
purely functional (no I/O, no executor instantiation).
"""

from __future__ import annotations

import io

import pytest

from src.live.fleet import build_cell_matrix, format_cell_list, make_cell_key


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_simple(models=("lstm",), tickers=("SPY",), strategies=("fixed_2r",), live=()):
    return build_cell_matrix(
        models=list(models),
        tickers=list(tickers),
        strategies=list(strategies),
        live_subset=list(live),
    )


# -----------------------------------------------------------------------
# Cartesian product sizes (arbitrary subsets — not from the plan example table)
# -----------------------------------------------------------------------


class TestCartesianCounts:
    def test_1x1x1(self):
        specs = _make_simple(("lstm",), ("SPY",), ("fixed_2r",))
        assert len(specs) == 1

    def test_2x1x1(self):
        specs = _make_simple(("lstm", "cnn_lstm"), ("SPY",), ("fixed_2r",))
        assert len(specs) == 2

    def test_1x2x1(self):
        specs = _make_simple(("lstm",), ("SPY", "QQQ"), ("fixed_2r",))
        assert len(specs) == 2

    def test_1x1x4(self):
        specs = _make_simple(("lstm",), ("SPY",), ("fixed_2r", "ict_iofed", "ce_50pct", "tradinglab"))
        assert len(specs) == 4

    def test_2x4x4(self):
        specs = _make_simple(
            ("lstm", "cnn_lstm"),
            ("SPY", "QQQ", "IWM", "DIA"),
            ("fixed_2r", "ict_iofed", "ce_50pct", "tradinglab"),
        )
        assert len(specs) == 32

    def test_full_matrix_4x4x4(self):
        specs = _make_simple(
            ("lstm", "cnn_lstm", "transformer", "xgboost"),
            ("SPY", "QQQ", "IWM", "DIA"),
            ("fixed_2r", "ict_iofed", "ce_50pct", "tradinglab"),
        )
        assert len(specs) == 64

    def test_3x2x3_non_example_shape(self):
        specs = _make_simple(
            ("lstm", "cnn_lstm", "xgboost"),
            ("SPY", "IWM"),
            ("fixed_2r", "ce_50pct", "tradinglab"),
        )
        assert len(specs) == 18

    def test_4x1x2(self):
        specs = _make_simple(
            ("lstm", "cnn_lstm", "transformer", "xgboost"),
            ("QQQ",),
            ("fixed_2r", "tradinglab"),
        )
        assert len(specs) == 8


# -----------------------------------------------------------------------
# Cell key format and uniqueness
# -----------------------------------------------------------------------


class TestCellKeys:
    def test_key_format(self):
        key = make_cell_key("SPY", "lstm", "fixed_2r")
        assert key == "SPY:lstm:fixed_2r"

    def test_all_keys_unique(self):
        specs = _make_simple(
            ("lstm", "cnn_lstm"),
            ("SPY", "QQQ"),
            ("fixed_2r", "tradinglab"),
        )
        keys = [s.key for s in specs]
        assert len(keys) == len(set(keys))

    def test_spec_key_matches_make_cell_key(self):
        specs = _make_simple(("lstm",), ("SPY",), ("ce_50pct",))
        spec = specs[0]
        assert spec.key == make_cell_key(spec.ticker, spec.model, spec.strategy)


# -----------------------------------------------------------------------
# Real XOR sim invariant
# -----------------------------------------------------------------------


class TestRealXorSim:
    def test_all_sim_when_no_live_subset(self):
        specs = _make_simple(("lstm", "cnn_lstm"), ("SPY", "QQQ"), ("fixed_2r",))
        assert all(s.executor_type == "sim" for s in specs)

    def test_single_real_cell(self):
        specs = _make_simple(
            ("lstm",), ("SPY",), ("fixed_2r",),
            live=("lstm:SPY:fixed_2r",),
        )
        assert len(specs) == 1
        assert specs[0].executor_type == "real"

    def test_one_real_rest_sim(self):
        specs = _make_simple(
            ("lstm", "cnn_lstm"),
            ("SPY", "QQQ"),
            ("fixed_2r", "tradinglab"),
            live=("lstm:SPY:fixed_2r",),
        )
        real = [s for s in specs if s.executor_type == "real"]
        sim = [s for s in specs if s.executor_type == "sim"]
        assert len(real) == 1
        assert real[0].key == "SPY:lstm:fixed_2r"
        assert len(sim) == 7  # 8 total - 1 real

    def test_two_real_different_tickers(self):
        specs = _make_simple(
            ("lstm", "cnn_lstm"),
            ("SPY", "QQQ"),
            ("fixed_2r", "tradinglab"),
            live=("lstm:SPY:fixed_2r", "cnn_lstm:QQQ:tradinglab"),
        )
        real_keys = {s.key for s in specs if s.executor_type == "real"}
        assert "SPY:lstm:fixed_2r" in real_keys
        assert "QQQ:cnn_lstm:tradinglab" in real_keys
        assert len(real_keys) == 2

    def test_no_cell_is_both_real_and_sim(self):
        specs = _make_simple(
            ("lstm", "cnn_lstm"),
            ("SPY", "QQQ"),
            ("fixed_2r", "tradinglab"),
            live=("lstm:SPY:fixed_2r",),
        )
        # Every cell appears exactly once — no duplicates
        keys = [s.key for s in specs]
        assert len(keys) == len(set(keys))


# -----------------------------------------------------------------------
# Legality: empty set → ValueError
# -----------------------------------------------------------------------


class TestEmptySetReject:
    def test_empty_models(self):
        with pytest.raises(ValueError, match="models"):
            build_cell_matrix([], ["SPY"], ["fixed_2r"], [])

    def test_empty_tickers(self):
        with pytest.raises(ValueError, match="tickers"):
            build_cell_matrix(["lstm"], [], ["fixed_2r"], [])

    def test_empty_strategies(self):
        with pytest.raises(ValueError, match="strategies"):
            build_cell_matrix(["lstm"], ["SPY"], [], [])


# -----------------------------------------------------------------------
# Legality: out-of-matrix live-subset triple → ValueError
# -----------------------------------------------------------------------


class TestOutOfMatrixReject:
    def test_model_not_in_matrix(self):
        with pytest.raises(ValueError, match="not in the resolved cell matrix"):
            _make_simple(
                ("lstm",), ("SPY",), ("fixed_2r",),
                live=("xgboost:SPY:fixed_2r",),  # xgboost not in models
            )

    def test_ticker_not_in_matrix(self):
        with pytest.raises(ValueError, match="not in the resolved cell matrix"):
            _make_simple(
                ("lstm",), ("SPY",), ("fixed_2r",),
                live=("lstm:QQQ:fixed_2r",),  # QQQ not in tickers
            )

    def test_strategy_not_in_matrix(self):
        with pytest.raises(ValueError, match="not in the resolved cell matrix"):
            _make_simple(
                ("lstm",), ("SPY",), ("fixed_2r",),
                live=("lstm:SPY:tradinglab",),  # tradinglab not in strategies
            )

    def test_malformed_triple(self):
        with pytest.raises(ValueError, match="model:ticker:strategy"):
            build_cell_matrix(["lstm"], ["SPY"], ["fixed_2r"], ["lstm:SPY"])


# -----------------------------------------------------------------------
# Legality: duplicate real ticker → ValueError
# -----------------------------------------------------------------------


class TestDuplicateRealTickerReject:
    def test_two_real_cells_same_ticker(self):
        with pytest.raises(ValueError, match="Duplicate real ticker 'SPY'"):
            _make_simple(
                ("lstm", "cnn_lstm"),
                ("SPY",),
                ("fixed_2r", "tradinglab"),
                live=("lstm:SPY:fixed_2r", "cnn_lstm:SPY:tradinglab"),
            )

    def test_two_real_cells_same_ticker_different_model(self):
        with pytest.raises(ValueError, match="Duplicate real ticker"):
            build_cell_matrix(
                models=["lstm", "cnn_lstm"],
                tickers=["SPY", "QQQ"],
                strategies=["fixed_2r"],
                live_subset=["lstm:SPY:fixed_2r", "cnn_lstm:SPY:fixed_2r"],
            )

    def test_duplicate_qqq(self):
        with pytest.raises(ValueError, match="Duplicate real ticker 'QQQ'"):
            build_cell_matrix(
                models=["lstm"],
                tickers=["QQQ"],
                strategies=["fixed_2r", "tradinglab"],
                live_subset=["lstm:QQQ:fixed_2r", "lstm:QQQ:tradinglab"],
            )


# -----------------------------------------------------------------------
# --list-cells output
# -----------------------------------------------------------------------


class TestListCells:
    def test_output_contains_all_cells(self):
        specs = _make_simple(
            ("lstm", "cnn_lstm"), ("SPY",), ("fixed_2r",),
        )
        output = format_cell_list(specs)
        assert "lstm" in output
        assert "cnn_lstm" in output
        assert "SPY" in output
        assert "fixed_2r" in output

    def test_output_shows_real_and_sim(self):
        specs = _make_simple(
            ("lstm", "cnn_lstm"),
            ("SPY", "QQQ"),
            ("fixed_2r",),
            live=("lstm:SPY:fixed_2r",),
        )
        output = format_cell_list(specs)
        assert "real" in output
        assert "sim" in output

    def test_output_total_count(self):
        specs = _make_simple(("lstm",), ("SPY", "QQQ", "IWM"), ("fixed_2r",))
        output = format_cell_list(specs)
        assert "Total cells: 3" in output

    def test_output_real_sim_counts(self):
        specs = _make_simple(
            ("lstm", "cnn_lstm"),
            ("SPY", "QQQ"),
            ("fixed_2r",),
            live=("lstm:SPY:fixed_2r",),
        )
        output = format_cell_list(specs)
        # 4 total, 1 real, 3 sim
        assert "1" in output  # real count
        assert "3" in output  # sim count

    def test_output_header_row(self):
        specs = _make_simple(("lstm",), ("SPY",), ("fixed_2r",))
        output = format_cell_list(specs)
        assert "TICKER" in output
        assert "MODEL" in output
        assert "STRATEGY" in output
        assert "EXECUTOR" in output

    def test_all_sim_output(self):
        specs = _make_simple(("lstm",), ("SPY",), ("fixed_2r",))
        output = format_cell_list(specs)
        assert "real (PaperExecutor):   0" in output
        assert "sim  (SimFillExecutor): 1" in output


# -----------------------------------------------------------------------
# model_tfs propagation — regression for TF-guard fix (Jun-12-26)
#
# Before the fix, build_cell_matrix had no model_tfs param; all cells
# defaulted to tf="h1".  The FleetRouter TF-guard then rejected 5m/15m
# checkpoints with "model declares tf='h1' but checkpoint meta says tf='5m'".
# The fix: resolve_tf_from_meta(path) → passed as model_tfs kwarg.
# -----------------------------------------------------------------------


class TestModelTfsPropagation:
    """resolve_tf_from_meta + build_cell_matrix(model_tfs=...) contract."""

    def test_resolve_tf_from_meta_reads_5m(self, tmp_path):
        """Meta JSON with timeframe:'5m' → resolve returns '5m'."""
        from src.live.fleet import resolve_tf_from_meta

        ckpt = tmp_path / "cnn_lstm_5m.pt"
        ckpt.write_bytes(b"")  # stub checkpoint
        meta = tmp_path / "cnn_lstm_5m.meta.json"
        meta.write_text('{"timeframe": "5m"}')

        assert resolve_tf_from_meta(str(ckpt)) == "5m"

    def test_resolve_tf_from_meta_reads_15m(self, tmp_path):
        from src.live.fleet import resolve_tf_from_meta

        ckpt = tmp_path / "lstm_15m.pt"
        ckpt.write_bytes(b"")
        (tmp_path / "lstm_15m.meta.json").write_text('{"timeframe": "15m"}')

        assert resolve_tf_from_meta(str(ckpt)) == "15m"

    def test_resolve_tf_from_meta_no_meta_defaults_h1(self, tmp_path):
        """No .meta.json → default 'h1' (back-compat)."""
        from src.live.fleet import resolve_tf_from_meta

        ckpt = tmp_path / "lstm_h1.pt"
        ckpt.write_bytes(b"")
        # deliberately no .meta.json

        assert resolve_tf_from_meta(str(ckpt)) == "h1"

    def test_resolve_tf_from_meta_missing_key_defaults_h1(self, tmp_path):
        """Meta JSON exists but has no 'timeframe' key → default 'h1'."""
        from src.live.fleet import resolve_tf_from_meta

        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"")
        (tmp_path / "model.meta.json").write_text('{"arch": "cnn_lstm"}')

        assert resolve_tf_from_meta(str(ckpt)) == "h1"

    def test_resolve_tf_from_meta_none_defaults_h1(self):
        """None checkpoint path → 'h1' (no-override code path)."""
        from src.live.fleet import resolve_tf_from_meta

        assert resolve_tf_from_meta(None) == "h1"

    def test_build_cell_matrix_model_tfs_5m_propagates(self, tmp_path):
        """build_cell_matrix(model_tfs={'cnn_lstm':'5m'}) → spec.tf == '5m'."""
        specs = build_cell_matrix(
            models=["cnn_lstm"],
            tickers=["SPY"],
            strategies=["fixed_2r"],
            live_subset=[],
            model_tfs={"cnn_lstm": "5m"},
        )
        assert len(specs) == 1
        assert specs[0].tf == "5m"

    def test_build_cell_matrix_model_tfs_15m_propagates(self, tmp_path):
        specs = build_cell_matrix(
            models=["lstm"],
            tickers=["SPY"],
            strategies=["fixed_2r"],
            live_subset=[],
            model_tfs={"lstm": "15m"},
        )
        assert specs[0].tf == "15m"

    def test_build_cell_matrix_no_model_tfs_defaults_h1(self):
        """Omitting model_tfs (back-compat) → tf defaults to 'h1'."""
        specs = build_cell_matrix(
            models=["lstm"],
            tickers=["SPY"],
            strategies=["fixed_2r"],
            live_subset=[],
        )
        assert specs[0].tf == "h1"

    def test_build_cell_matrix_partial_model_tfs_mixed(self):
        """Only one model in model_tfs → that model gets 5m, other gets h1."""
        specs = build_cell_matrix(
            models=["lstm", "cnn_lstm"],
            tickers=["SPY"],
            strategies=["fixed_2r"],
            live_subset=[],
            model_tfs={"cnn_lstm": "5m"},
        )
        by_model = {s.model: s.tf for s in specs}
        assert by_model["cnn_lstm"] == "5m"
        assert by_model["lstm"] == "h1"

    def test_end_to_end_meta_to_cell_tf(self, tmp_path):
        """Full fix path: write meta.json → resolve_tf_from_meta → build_cell_matrix → spec.tf."""
        from src.live.fleet import resolve_tf_from_meta

        ckpt = tmp_path / "cnn_lstm_5m.pt"
        ckpt.write_bytes(b"")
        (tmp_path / "cnn_lstm_5m.meta.json").write_text('{"timeframe": "5m"}')

        tf = resolve_tf_from_meta(str(ckpt))
        specs = build_cell_matrix(
            models=["cnn_lstm"],
            tickers=["SPY"],
            strategies=["fixed_2r"],
            live_subset=[],
            model_tfs={"cnn_lstm": tf},
        )
        assert specs[0].tf == "5m"
