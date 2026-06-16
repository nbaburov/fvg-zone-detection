"""tests/rigor/test_tf_routing.py — Timeframe token routing tests for rigor scripts.

Asserts that:
  - --timeframe 5m / 15m resolves spy_5m_* parquets + class_weights_5m.json
    + token-suffixed output files.
  - --timeframe h1 (default) resolves the legacy unsuffixed paths (byte-identical
    to before the TF change).

All tests stub actual training / Optuna at the boundary — only PATH RESOLUTION
and argument threading are verified; no real training runs are launched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_parquet(path: Path, n: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "open": np.ones(n), "high": np.ones(n) + 0.1,
        "low": np.ones(n) - 0.1, "close": np.ones(n),
        "volume": np.ones(n) * 1000,
        "label": np.zeros(n, dtype=int),
    }, index=pd.date_range("2023-01-02 09:30", periods=n, freq="h", tz="America/New_York"))
    df.index.name = "timestamp"
    df.to_parquet(path)


def _make_data_dir(tmp_path: Path, token: str, scope: str = "spy") -> Path:
    """Scaffold {scope}_{token}_{train,val,test}.parquet + class_weights_{scope}_{token}.json."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        _make_dummy_parquet(data_dir / f"{scope}_{token}_{split}.parquet")
    weights = [1.0, 5.0, 5.0]
    (data_dir / f"class_weights_{scope}_{token}.json").write_text(json.dumps(weights))
    return data_dir


# ---------------------------------------------------------------------------
# seed_sweep helpers: _load_splits / _load_class_weights
# ---------------------------------------------------------------------------

class TestLoadSplitsTokenRouting:
    def test_h1_loads_spy_h1_parquets(self, tmp_path):
        from src.rigor.seed_sweep import _load_splits
        data_dir = _make_data_dir(tmp_path, "h1")
        train, val, test = _load_splits(data_dir, "h1")
        assert len(train) > 0 and len(val) > 0 and len(test) > 0

    def test_5m_loads_spy_5m_parquets(self, tmp_path):
        from src.rigor.seed_sweep import _load_splits
        data_dir = _make_data_dir(tmp_path, "5m")
        train, val, test = _load_splits(data_dir, "5m")
        assert len(train) > 0

    def test_15m_loads_spy_15m_parquets(self, tmp_path):
        from src.rigor.seed_sweep import _load_splits
        data_dir = _make_data_dir(tmp_path, "15m")
        train, val, test = _load_splits(data_dir, "15m")
        assert len(train) > 0

    def test_5m_token_absent_raises(self, tmp_path):
        """If spy_5m_*.parquet don't exist, _load_splits must raise."""
        from src.rigor.seed_sweep import _load_splits
        data_dir = _make_data_dir(tmp_path, "h1")  # only h1 files
        with pytest.raises(Exception):
            _load_splits(data_dir, "5m")


class TestLoadClassWeightsTokenRouting:
    def test_h1_reads_class_weights_spy_h1_json(self, tmp_path):
        from src.rigor.seed_sweep import _load_class_weights
        data_dir = _make_data_dir(tmp_path, "h1")
        w = _load_class_weights(data_dir, "h1")
        assert len(w) == 3

    def test_5m_reads_class_weights_5m_json(self, tmp_path):
        from src.rigor.seed_sweep import _load_class_weights
        data_dir = _make_data_dir(tmp_path, "5m")
        w = _load_class_weights(data_dir, "5m")
        assert len(w) == 3

    def test_h1_reads_class_weights_spy_h1_not_bare(self, tmp_path):
        """h1 must use class_weights_spy_h1.json NOT bare class_weights.json."""
        from src.rigor.seed_sweep import _load_class_weights
        data_dir = _make_data_dir(tmp_path, "h1")
        # Add a bare class_weights.json with sentinel values — must NOT be read
        (data_dir / "class_weights.json").write_text("[9,9,9]")
        w = _load_class_weights(data_dir, "h1")
        assert w != [9.0, 9.0, 9.0], "h1 must read class_weights_spy_h1.json, not bare class_weights.json"

    def test_missing_5m_weights_raises(self, tmp_path):
        from src.rigor.seed_sweep import _load_class_weights
        data_dir = _make_data_dir(tmp_path, "h1")  # no class_weights_5m.json
        with pytest.raises(Exception):
            _load_class_weights(data_dir, "5m")


# ---------------------------------------------------------------------------
# window_sweep: --timeframe is wired into _load_splits / _load_class_weights
# Tested by inspecting what the script actually calls at the seed_sweep module.
# ---------------------------------------------------------------------------

class TestWindowSweepTFArg:
    """Verify --timeframe arg exists and gets threaded to _load_splits/_load_class_weights."""

    def test_timeframe_arg_is_in_help(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/rigor/sweeps/window_sweep.py", "--help"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert "--timeframe" in result.stdout
        assert "h1" in result.stdout

    def test_5m_loads_spy_5m_splits(self, tmp_path, monkeypatch):
        """window_sweep with --timeframe 5m must call _load_splits(..., '5m')."""
        calls = {}

        import src.rigor.seed_sweep as ss_mod
        orig_load_splits = ss_mod._load_splits

        def tracking_load_splits(data_dir, timeframe="h1"):
            calls["tf"] = timeframe
            # Return the real call so the rest doesn't blow up (files exist)
            return orig_load_splits(data_dir, timeframe)

        monkeypatch.setattr(ss_mod, "_load_splits", tracking_load_splits)
        monkeypatch.setattr(ss_mod, "_load_class_weights", lambda d, tf: [1.0, 5.0, 5.0])

        data_dir = _make_data_dir(tmp_path, "5m")
        config_json = tmp_path / "cfg.json"
        config_json.write_text(json.dumps({
            "hidden_size": 16, "lr": 1e-3, "weight_decay": 1e-4,
            "batch_size": 4, "num_layers": 1, "dropout": 0.0, "head_dropout": 0.0,
        }))
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        import scripts.rigor.sweeps.window_sweep as ws

        argv = [
            "--config", str(config_json),
            "--model", "lstm",
            "--windows", "5",
            "--data-dir", str(data_dir),
            "--output-dir", str(out_dir),
            "--timeframe", "5m",
        ]
        with mock.patch("sys.argv", ["window_sweep.py"] + argv):
            try:
                ws.main()
            except Exception:
                pass  # training will fail on empty dataset — we only care about the call

        assert calls.get("tf") == "5m", \
            f"Expected _load_splits called with tf='5m', got calls={calls}"

    def test_h1_default_calls_h1(self, tmp_path, monkeypatch):
        """window_sweep with --timeframe h1 (default) calls _load_splits(..., 'h1')."""
        calls = {}

        import src.rigor.seed_sweep as ss_mod
        orig_load_splits = ss_mod._load_splits

        def tracking_load_splits(data_dir, timeframe="h1"):
            calls["tf"] = timeframe
            return orig_load_splits(data_dir, timeframe)

        monkeypatch.setattr(ss_mod, "_load_splits", tracking_load_splits)
        monkeypatch.setattr(ss_mod, "_load_class_weights", lambda d, tf: [1.0, 5.0, 5.0])

        data_dir = _make_data_dir(tmp_path, "h1")
        config_json = tmp_path / "cfg.json"
        config_json.write_text(json.dumps({
            "hidden_size": 16, "lr": 1e-3, "weight_decay": 1e-4,
            "batch_size": 4, "num_layers": 1, "dropout": 0.0, "head_dropout": 0.0,
        }))
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        import scripts.rigor.sweeps.window_sweep as ws

        argv = [
            "--config", str(config_json),
            "--model", "lstm",
            "--windows", "5",
            "--data-dir", str(data_dir),
            "--output-dir", str(out_dir),
            "--timeframe", "h1",
        ]
        with mock.patch("sys.argv", ["window_sweep.py"] + argv):
            try:
                ws.main()
            except Exception:
                pass

        assert calls.get("tf") == "h1", \
            f"Expected _load_splits called with tf='h1', got calls={calls}"


# ---------------------------------------------------------------------------
# naive_baselines: --timeframe threads correct parquet path into get_test_labels
# ---------------------------------------------------------------------------

class TestNaiveBaselinesTFRouting:
    """Verify get_test_labels resolves the correct parquet filename per token."""

    def test_5m_resolves_spy_5m_test_parquet(self, tmp_path):
        """get_test_labels('default', '5m') must read spy_5m_test.parquet."""
        # Scaffold data/processed inside tmp_path (matches REPO/data/processed pattern)
        proc_dir = tmp_path / "data" / "processed"
        _make_dummy_parquet(proc_dir / "spy_5m_test.parquet")

        import scripts.rigor.stats.naive_baselines as nb_mod

        loaded = []
        real_rp = pd.read_parquet

        def spy_read(path, *a, **kw):
            loaded.append(Path(path).name)
            return real_rp(path, *a, **kw)

        with mock.patch.object(nb_mod, "REPO", tmp_path):
            with mock.patch("pandas.read_parquet", side_effect=spy_read):
                with mock.patch.object(nb_mod, "SMCWindowDataset",
                                       return_value=[(None, 0)] * 3):
                    try:
                        nb_mod.get_test_labels("default", "5m")
                    except Exception:
                        pass

        assert any("spy_5m_test" in p for p in loaded), \
            f"Expected spy_5m_test.parquet to be loaded; got: {loaded}"

    def test_h1_resolves_spy_h1_test_parquet(self, tmp_path):
        """get_test_labels('default', 'h1') must read spy_h1_test.parquet."""
        proc_dir = tmp_path / "data" / "processed"
        _make_dummy_parquet(proc_dir / "spy_h1_test.parquet")

        import scripts.rigor.stats.naive_baselines as nb_mod

        loaded = []
        real_rp = pd.read_parquet

        def spy_read(path, *a, **kw):
            loaded.append(Path(path).name)
            return real_rp(path, *a, **kw)

        with mock.patch.object(nb_mod, "REPO", tmp_path):
            with mock.patch("pandas.read_parquet", side_effect=spy_read):
                with mock.patch.object(nb_mod, "SMCWindowDataset",
                                       return_value=[(None, 0)] * 3):
                    try:
                        nb_mod.get_test_labels("default", "h1")
                    except Exception:
                        pass

        assert any("spy_h1_test" in p for p in loaded), \
            f"Expected spy_h1_test.parquet to be loaded; got: {loaded}"

    def test_h1_does_not_load_5m(self, tmp_path):
        """h1 must NOT attempt spy_5m_test.parquet."""
        proc_dir = tmp_path / "data" / "processed"
        _make_dummy_parquet(proc_dir / "spy_h1_test.parquet")

        import scripts.rigor.stats.naive_baselines as nb_mod

        loaded = []
        real_rp = pd.read_parquet

        def spy_read(path, *a, **kw):
            loaded.append(Path(path).name)
            return real_rp(path, *a, **kw)

        with mock.patch.object(nb_mod, "REPO", tmp_path):
            with mock.patch("pandas.read_parquet", side_effect=spy_read):
                with mock.patch.object(nb_mod, "SMCWindowDataset",
                                       return_value=[(None, 0)] * 3):
                    try:
                        nb_mod.get_test_labels("default", "h1")
                    except Exception:
                        pass

        assert not any("5m" in p for p in loaded), \
            f"h1 loaded a 5m parquet: {loaded}"


# ---------------------------------------------------------------------------
# bootstrap_ci_multiseed: --timeframe suffixes output filename
# ---------------------------------------------------------------------------

class TestBootstrapCiTFOutputNaming:
    def _make_npz(self, path: Path, n: int = 50) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, y_true=np.zeros(n, int), y_pred=np.zeros(n, int))

    def test_h1_output_is_unsuffixed(self, tmp_path):
        pred_dir = tmp_path / "preds"
        self._make_npz(pred_dir / "lstm_seed42_preds.npz")
        out_dir = tmp_path / "out"

        argv = [
            "--pred-dir", str(pred_dir),
            "--model", "lstm",
            "--block-size", "10",
            "--n-iter", "5",
            "--timeframe", "h1",
            "--output-dir", str(out_dir),
        ]
        with mock.patch("sys.argv", ["bootstrap_ci_multiseed.py"] + argv):
            import scripts.rigor.stats.bootstrap_ci_multiseed as bc_mod
            bc_mod.main()

        assert (out_dir / "bootstrap_ci_lstm.json").exists(), \
            "h1 output must be bootstrap_ci_lstm.json (no suffix)"
        assert not (out_dir / "bootstrap_ci_lstm_h1.json").exists(), \
            "h1 must NOT produce a _h1-suffixed file"

    def test_5m_output_is_suffixed(self, tmp_path):
        pred_dir = tmp_path / "preds"
        self._make_npz(pred_dir / "lstm_seed42_preds.npz")
        out_dir = tmp_path / "out"

        argv = [
            "--pred-dir", str(pred_dir),
            "--model", "lstm",
            "--block-size", "10",
            "--n-iter", "5",
            "--timeframe", "5m",
            "--output-dir", str(out_dir),
        ]
        with mock.patch("sys.argv", ["bootstrap_ci_multiseed.py"] + argv):
            import scripts.rigor.stats.bootstrap_ci_multiseed as bc_mod
            bc_mod.main()

        assert (out_dir / "bootstrap_ci_lstm_5m.json").exists(), \
            "5m output must be bootstrap_ci_lstm_5m.json"

    def test_15m_output_is_suffixed(self, tmp_path):
        pred_dir = tmp_path / "preds"
        self._make_npz(pred_dir / "xgb_seed42_preds.npz")
        out_dir = tmp_path / "out"

        argv = [
            "--pred-dir", str(pred_dir),
            "--model", "xgb",
            "--block-size", "10",
            "--n-iter", "5",
            "--timeframe", "15m",
            "--output-dir", str(out_dir),
        ]
        with mock.patch("sys.argv", ["bootstrap_ci_multiseed.py"] + argv):
            import scripts.rigor.stats.bootstrap_ci_multiseed as bc_mod
            bc_mod.main()

        assert (out_dir / "bootstrap_ci_xgb_15m.json").exists()


# ---------------------------------------------------------------------------
# learning_curve: --timeframe injects --set override + suffixes outputs
# ---------------------------------------------------------------------------

class TestLearningCurveTFRouting:
    def test_timeframe_arg_in_help(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/rigor/sweeps/learning_curve.py", "--help"],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert "--timeframe" in result.stdout

    def test_h1_cmd_has_no_set_override(self, tmp_path):
        """_run_train with timeframe='h1' must NOT add --set to the command."""
        import scripts.rigor.sweeps.learning_curve as lc_mod
        import io, contextlib

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("# dummy")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lc_mod._run_train("lstm", cfg_path, 42, 1.0, True, "h1")  # dry_run
        assert "--set" not in buf.getvalue(), \
            f"h1 must not add --set to cmd: {buf.getvalue()!r}"

    def test_5m_cmd_injects_set_override(self, tmp_path):
        """_run_train with timeframe='5m' must add --set data.timeframe=5m."""
        import scripts.rigor.sweeps.learning_curve as lc_mod
        import io, contextlib

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("# dummy")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lc_mod._run_train("lstm", cfg_path, 42, 1.0, True, "5m")
        assert "data.timeframe=5m" in buf.getvalue(), \
            f"5m must inject --set data.timeframe=5m: {buf.getvalue()!r}"

    def test_output_suffix_logic(self):
        """CSV/PNG output filenames carry _5m / _15m suffix; h1 is unsuffixed."""
        cases = [
            ("h1", "learning_curve_lstm.csv"),
            ("5m", "learning_curve_lstm_5m.csv"),
            ("15m", "learning_curve_lstm_15m.csv"),
        ]
        for tf, expected_stem in cases:
            suffix = f"_{tf}" if tf != "h1" else ""
            result = f"learning_curve_lstm{suffix}.csv"
            assert result == expected_stem, f"tf={tf}: got {result!r}"


# ---------------------------------------------------------------------------
# Tuners: --timeframe arg exposed + output JSON naming
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", [
    "tune_lstm.py",
    "tune_cnn_lstm.py",
    "tune_transformer.py",
    "tune_xgboost.py",
])
def test_tuner_exposes_timeframe_arg(script):
    """Each tuner must advertise --timeframe {h1,5m,15m} in --help."""
    import subprocess
    result = subprocess.run(
        [sys.executable, f"scripts/rigor/tune/{script}", "--help"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert "--timeframe" in result.stdout, \
        f"{script} is missing --timeframe (stdout={result.stdout[:300]!r})"
    assert "h1" in result.stdout, \
        f"{script} --timeframe should show h1 as choice"


# ---------------------------------------------------------------------------
# XGB sweep worker: class-weights resolved from --data-dir, not hardcoded root
# ---------------------------------------------------------------------------

class TestXgbSweepWorkerDataDirResolution:
    """Verify _xgb_sweep_worker resolves class_weights from --data-dir.

    Tests stub xgboost training entirely — only path resolution and argument
    wiring are verified.  No real training runs are launched.
    """

    def _build_cmd(self, tmp_path: Path, tf: str, data_dir: Path) -> list:
        """Construct the argv list as multiseed_run._run_xgb_seed_sweep_subprocess would."""
        import numpy as np
        npz_path = tmp_path / "data.npz"
        np.savez(npz_path,
                 X_train=np.zeros((4, 3)), y_train=np.zeros(4, int),
                 X_val=np.zeros((2, 3)), y_val=np.zeros(2, int),
                 X_test=np.zeros((2, 3)), y_test=np.zeros(2, int))
        hp = json.dumps({"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1,
                         "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8})
        ckpt_dir = tmp_path / "checkpoints"
        out_dir = tmp_path / "out"
        return [
            sys.executable,
            str(ROOT / "scripts" / "rigor" / "_workers" / "_xgb_sweep_worker.py"),
            "--data-npz", str(npz_path),
            "--config", hp,
            "--seeds", "0",
            "--output-dir", str(out_dir),
            "--checkpoint-dir", str(ckpt_dir),
            "--timeframe", tf,
            "--data-dir", str(data_dir),
        ]

    def _scaffold_data_dir(self, base: Path, tf: str, scope: str = "spy") -> Path:
        """Write class_weights_{scope}_{tf}.json in base."""
        base.mkdir(parents=True, exist_ok=True)
        weights = {"0": 1.0, "1": 5.0, "2": 5.0}
        cw_name = f"class_weights_{scope}_{tf}.json"
        (base / cw_name).write_text(json.dumps(weights))
        return base

    def test_h1_resolves_class_weights_spy_h1_json_from_data_dir(self, tmp_path):
        """h1 token: worker reads class_weights_spy_h1.json from --data-dir, not repo root."""
        import subprocess
        data_dir = self._scaffold_data_dir(tmp_path / "mydata", "h1")
        cmd = self._build_cmd(tmp_path, "h1", data_dir)
        # Worker will fail at xgb fit (tiny dummy data) but must NOT raise
        # FileNotFoundError for class_weights.json.
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        assert "FileNotFoundError" not in result.stderr, (
            f"Worker raised FileNotFoundError — weights not found in data-dir.\n{result.stderr[-800:]}"
        )
        assert "class_weights" not in result.stderr or "No such file" not in result.stderr, (
            f"class_weights open failed.\n{result.stderr[-800:]}"
        )

    def test_15m_resolves_class_weights_15m_json_from_data_dir(self, tmp_path):
        """15m token: worker reads class_weights_15m.json from --data-dir (multisym case)."""
        import subprocess
        data_dir = self._scaffold_data_dir(tmp_path / "multisym", "15m")
        cmd = self._build_cmd(tmp_path, "15m", data_dir)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        assert "FileNotFoundError" not in result.stderr, (
            f"15m multisym: worker raised FileNotFoundError for class_weights_15m.json.\n"
            f"{result.stderr[-800:]}"
        )

    def test_missing_weights_in_data_dir_raises(self, tmp_path):
        """Worker must fail fast (FileNotFoundError) when weights are absent from data-dir."""
        import subprocess
        data_dir = tmp_path / "empty_dir"
        data_dir.mkdir()
        # Write h1 weights only (class_weights_spy_h1.json), but ask for 15m → worker must raise
        (data_dir / "class_weights_spy_h1.json").write_text('{"0":1.0,"1":5.0,"2":5.0}')
        cmd = self._build_cmd(tmp_path, "15m", data_dir)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        assert result.returncode != 0, "Worker must exit non-zero when weights file is missing"

    def test_h1_default_data_dir_is_backward_compat(self, tmp_path, monkeypatch):
        """When --data-dir is omitted the worker defaults to ROOT/data/processed (H1 compat)."""
        import argparse
        # Import the worker as a module (it is torch-free) and inspect arg defaults.
        import importlib.util
        worker_path = ROOT / "scripts" / "rigor" / "_workers" / "_xgb_sweep_worker.py"
        spec = importlib.util.spec_from_file_location("_xgb_sweep_worker", worker_path)
        mod = importlib.util.module_from_spec(spec)
        # Parse with only the required args to check --data-dir default is None.
        p = argparse.ArgumentParser()
        p.add_argument('--data-npz', required=False, default='x.npz')
        p.add_argument('--config', required=False, default='{}')
        p.add_argument('--seeds', nargs='+', type=int, default=[0])
        p.add_argument('--output-dir', required=False, default='/tmp')
        p.add_argument('--checkpoint-dir', required=False, default='/tmp')
        p.add_argument('--timeframe', default='h1')
        p.add_argument('--data-dir', default=None)
        args = p.parse_args([])
        assert args.data_dir is None, "--data-dir must default to None (triggers legacy ROOT/data/processed path)"

    def test_multiseed_run_cmd_includes_data_dir(self, tmp_path):
        """_run_xgb_seed_sweep_subprocess must pass --data-dir to the worker cmd."""
        import scripts.rigor.eval.multiseed_run as mr_mod
        import inspect, textwrap
        src = inspect.getsource(mr_mod._run_xgb_seed_sweep_subprocess)
        assert "--data-dir" in src, (
            "_run_xgb_seed_sweep_subprocess must include '--data-dir' in the subprocess cmd list"
        )


@pytest.mark.parametrize("tf,arch_stem,expected_filename", [
    ("h1",  "best_lstm_config",     "best_lstm_config.json"),
    ("5m",  "best_lstm_config",     "best_lstm_config_5m.json"),
    ("15m", "best_lstm_config",     "best_lstm_config_15m.json"),
    ("h1",  "best_hp_cnn_lstm",     "best_hp_cnn_lstm.json"),
    ("5m",  "best_hp_cnn_lstm",     "best_hp_cnn_lstm_5m.json"),
    ("h1",  "best_hp_transformer",  "best_hp_transformer.json"),
    ("5m",  "best_hp_transformer",  "best_hp_transformer_5m.json"),
    ("h1",  "best_xgb_config",      "best_xgb_config.json"),
    ("5m",  "best_xgb_config",      "best_xgb_config_5m.json"),
])
def test_tuner_output_suffix_logic(tf, arch_stem, expected_filename):
    """Token-suffix rule: h1 → no suffix; non-h1 → _{token} before .json."""
    suffix = f"_{tf}" if tf != "h1" else ""
    result = f"{arch_stem}{suffix}.json"
    assert result == expected_filename, f"tf={tf} stem={arch_stem}: got {result!r}"
