"""lstm_adapter.py — ModelAdapter wrapping FVGLSTMClassifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.inspect.base import ModelAdapter
from src.models.lstm import FVGLSTMClassifier

_DEFAULT_CHECKPOINT = "lstm_seed42.pt"


class LSTMAdapter(ModelAdapter):
    """Wraps a trained FVGLSTMClassifier for inspection inference.

    Windows are expected to be NORMALISED (N, 60, 5) float32 — the same
    normalisation applied during training via ``normalise_window``.

    Inference runs on CPU unconditionally (MPS is unreliable for batch inference
    per CLAUDE.md). ``torch.no_grad()`` is always active.
    """

    name = "lstm"

    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_file: str = _DEFAULT_CHECKPOINT,
        **_kwargs,
    ) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_path = checkpoint_dir / "lstm" / checkpoint_file
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"LSTM checkpoint not found: {checkpoint_path}"
            )

        # Read HP from meta sidecar so architecture matches checkpoint
        meta_path = checkpoint_path.with_suffix(".meta.json")
        hp: dict = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            hp = meta.get("hyperparams", {})

        self._model = FVGLSTMClassifier(
            hidden_size=hp.get("hidden_size", 64),
            num_layers=hp.get("num_layers", 2),
            dropout=hp.get("dropout", 0.3),
            head_dropout=hp.get("head_dropout", 0.5),
        )
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        # Support both bare state_dict and wrapped {"model_state_dict": ...}
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self._model.load_state_dict(state)
        self._model.train(False)

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Run LSTM inference on normalised windows.

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
