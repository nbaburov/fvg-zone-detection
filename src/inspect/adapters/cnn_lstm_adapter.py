"""cnn_lstm_adapter.py — ModelAdapter wrapping FVGCNNLSTMClassifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.inspect.base import ModelAdapter
from src.models.cnn_lstm import FVGCNNLSTMClassifier

_DEFAULT_CHECKPOINT = "cnn_lstm_seed42.pt"


class CNNLSTMAdapter(ModelAdapter):
    """Wraps a trained FVGCNNLSTMClassifier for inspection inference.

    Windows are expected to be NORMALISED (N, 60, 5) float32 — the same
    normalisation applied during training via ``normalise_window``.

    Inference runs on CPU unconditionally (MPS unreliable for batch inference).
    ``torch.no_grad()`` is always active.
    """

    name = "cnn_lstm"

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
            checkpoint_path = checkpoint_dir / "cnn_lstm_h1_spy" / checkpoint_file
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"CNN-LSTM checkpoint not found: {checkpoint_path}"
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
            # Older meta format stored HP at top level (pre-G1 base runs)
            if not hp:
                hp = {
                    k: meta[k]
                    for k in ("conv_filters", "kernel_size", "n_conv_layers",
                               "use_pool", "lstm_hidden", "lstm_layers",
                               "dropout", "head_dropout")
                    if k in meta
                }

        self._model = FVGCNNLSTMClassifier(
            conv_filters=int(hp.get("conv_filters", 32)),
            kernel_size=int(hp.get("kernel_size", 3)),
            n_conv_layers=int(hp.get("n_conv_layers", 2)),
            use_pool=bool(hp.get("use_pool", False)),
            pool_type=str(hp.get("pool_type", "max")),
            lstm_hidden=int(hp.get("lstm_hidden", 64)),
            lstm_layers=int(hp.get("lstm_layers", 1)),
            dropout=float(hp.get("dropout", 0.318)),
            head_dropout=float(hp.get("head_dropout", 0.526)),
        )
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        # Support both bare state_dict and wrapped {"model_state_dict": ...}
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self._model.load_state_dict(state)
        self._model.train(False)

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Run CNN-LSTM inference on normalised windows.

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
            logits = self._model(x)             # (N, 3)
            probas = F.softmax(logits, dim=-1)  # (N, 3)
        return probas.cpu().numpy().astype(np.float32)
