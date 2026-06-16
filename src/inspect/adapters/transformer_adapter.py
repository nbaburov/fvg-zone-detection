"""transformer_adapter.py — ModelAdapter wrapping FVGTransformerClassifier."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.inspect.base import ModelAdapter
from src.models.transformer import FVGTransformerClassifier

_DEFAULT_CHECKPOINT = "transformer_seed0.pt"

# Parameters accepted by FVGTransformerClassifier.__init__ (excluding self)
_CTOR_PARAMS = set(inspect.signature(FVGTransformerClassifier.__init__).parameters) - {"self"}


class TransformerAdapter(ModelAdapter):
    """Wraps a trained FVGTransformerClassifier for inspection inference.

    Bare-name default: resolves ``checkpoint_dir/"transformer_h1_spy"/<file>``.
    Pass an explicit ``checkpoint_path=`` to load from any path directly.

    Windows are expected to be NORMALISED (N, 60, 5) float32 — the same
    normalisation applied during training via ``normalise_window``.

    Inference runs on CPU unconditionally. ``torch.no_grad()`` is always active.

    Optuna metadata keys stored in meta.json hyperparams (e.g. ``warmup_steps``,
    ``lr``, ``n_trials_completed``) are filtered out before passing to the model
    constructor so they never cause an unexpected-keyword-argument error.
    """

    name = "transformer"

    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_file: str = _DEFAULT_CHECKPOINT,
        checkpoint_path: Path | None = None,
        **_kwargs,
    ) -> None:
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
        else:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_path = checkpoint_dir / "transformer_h1_spy" / checkpoint_file
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Transformer checkpoint not found: {checkpoint_path}. "
                "Pass checkpoint_dir as the root checkpoints/ directory or "
                "supply checkpoint_path= directly."
            )

        # Resolved checkpoint path — exposed for post-load TF validation (H2).
        self.checkpoint_path = checkpoint_path

        # Read HP from meta sidecar so architecture matches checkpoint
        meta_path = checkpoint_path.with_suffix(".meta.json")
        hp: dict = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            hp = meta.get("hyperparams", {})

        # Filter to only recognised ctor kwargs — strips Optuna/training metadata
        model_kwargs = {k: v for k, v in hp.items() if k in _CTOR_PARAMS}

        self._model = FVGTransformerClassifier(
            input_size=model_kwargs.get("input_size", 5),
            d_model=model_kwargs.get("d_model", 64),
            nhead=model_kwargs.get("nhead", 4),
            num_layers=model_kwargs.get("num_layers", 2),
            dim_feedforward=model_kwargs.get("dim_feedforward", 128),
            dropout=model_kwargs.get("dropout", 0.1),
            head_dropout=model_kwargs.get("head_dropout", 0.3),
            num_classes=model_kwargs.get("num_classes", 3),
            pool=model_kwargs.get("pool", "mean"),
        )
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        # Support both bare state_dict and wrapped {"model_state_dict": ...}
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self._model.load_state_dict(state)
        self._model.train(False)

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Run Transformer inference on normalised windows.

        Parameters
        ----------
        windows : np.ndarray, shape (N, 60, 5), dtype float32
            Normalised OHLCV windows.

        Returns
        -------
        np.ndarray, shape (N, 3), dtype float32
            Softmax probabilities [none, bullish, bearish].
        """
        if windows.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        x = torch.from_numpy(windows.astype(np.float32))  # (N, 60, 5)
        with torch.no_grad():
            logits = self._model(x)            # (N, 3)
            probas = F.softmax(logits, dim=-1)  # (N, 3)
        return probas.cpu().numpy().astype(np.float32)
