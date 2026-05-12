"""loader.py — Load experiment configs from YAML or legacy JSON.

load_experiment(path, overrides) — primary entry point.
experiment_from_json(path)       — backwards compat for best_hp_*.json files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.config.schema import ExperimentConfig

_BASE_YAML = Path(__file__).resolve().parents[2] / "experiments" / "_base.yaml"


def load_experiment(
    path: Path | str,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load YAML, merge with _base.yaml defaults, apply dot-key overrides.

    Args:
        path: path to experiment YAML file.
        overrides: flat dotted keys, e.g. {"train.seeds": [17], "model.hidden_size": 256}.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    base = _load_yaml(_BASE_YAML) if _BASE_YAML.exists() else {}
    experiment = _load_yaml(path)
    merged = _deep_merge(base, experiment)

    if overrides:
        for key, val in overrides.items():
            _set_nested(merged, key.split("."), val)

    return ExperimentConfig.model_validate(merged)


def experiment_from_json(path: Path | str) -> ExperimentConfig:
    """Load legacy best_hp_*.json produced by Optuna tuning scripts.

    Strips metadata keys (val_macro_f1, trial_number, study_name, etc.),
    infers arch from filename ("xgb" in name → XGBModelConfig, else LSTM).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON config not found: {path}")

    data = json.loads(path.read_text())
    _STRIP = {
        "val_macro_f1", "best_value", "trial_number", "study_name", "storage",
        "n_complete", "n_pruned", "n_trials_completed",
    }
    hp = {k: v for k, v in data.items() if k not in _STRIP}
    arch = "xgb" if "xgb" in path.name else "lstm"
    return ExperimentConfig.model_validate({"model": {"arch": arch, **hp}})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. override wins on conflict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _set_nested(d: dict, keys: list[str], val: Any) -> None:
    """Set a nested dict value via dotted key path. Creates intermediate dicts."""
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val


def parse_set_args(set_args: list[str]) -> dict[str, Any]:
    """Parse --set key=value strings into an overrides dict.

    Values are parsed with yaml.safe_load for type coercion:
      "train.seeds=[42]"         → {"train.seeds": [42]}
      "model.hidden_size=256"    → {"model.hidden_size": 256}
      "data.labeller=fvg_valid"  → {"data.labeller": "fvg_valid"}

    Args:
        set_args: list of "key=value" strings from argparse nargs.

    Returns:
        Dict mapping dotted key paths to coerced Python values.
    """
    result: dict[str, Any] = {}
    for item in set_args or []:
        if "=" not in item:
            raise ValueError(f"--set argument must be key=value, got: {item!r}")
        key, _, raw_val = item.partition("=")
        result[key.strip()] = yaml.safe_load(raw_val.strip())
    return result
