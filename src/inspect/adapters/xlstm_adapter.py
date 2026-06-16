"""xlstm_adapter.py — ModelAdapter wrapping FVGxLSTMClassifier."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.inspect.base import ModelAdapter
from src.models.xlstm_model import FVGxLSTMClassifier

_DEFAULT_CHECKPOINT = "xlstm_seed42.pt"

# Parameters accepted by FVGxLSTMClassifier.__init__ (excluding self)
_CTOR_PARAMS = set(inspect.signature(FVGxLSTMClassifier.__init__).parameters) - {"self"}


class XLSTMAdapter(ModelAdapter):
    """Wraps a trained FVGxLSTMClassifier for inspection inference.

    Bare-name default: resolves ``checkpoint_dir/"xlstm_h1_spy"/<file>``.
    Pass an explicit ``checkpoint_path=`` to load from any path directly.

    Windows are expected to be NORMALISED (N, 60, 5) float32 — the same
    normalisation applied during training via ``normalise_window``.

    Inference runs on CPU unconditionally. ``torch.no_grad()`` is always active.

    Apple-Silicon constraint (CLAUDE.md): the model is constructed with
    ``sLSTMLayerConfig(backend="vanilla", dtype="float32",
    enable_automatic_mixed_precision=False)`` — this is baked into
    ``FVGxLSTMClassifier`` itself and requires no special handling here.

    Any training/Optuna metadata keys in meta.json hyperparams are filtered
    to the ctor signature before construction.
    """

    name = "xlstm"

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
            checkpoint_path = checkpoint_dir / "xlstm_h1_spy" / checkpoint_file
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"xLSTM checkpoint not found: {checkpoint_path}. "
                "Pass checkpoint_dir as the root checkpoints/ directory or "
                "supply checkpoint_path= directly."
            )

        # Resolved checkpoint path — exposed for post-load TF validation (H2).
        self.checkpoint_path = checkpoint_path

        # Read HP from meta sidecar so architecture matches checkpoint
        meta_path = checkpoint_path.with_suffix(".meta.json")
        hp: dict = {}
        _meta_window_size: int | None = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            hp = meta.get("hyperparams", {})
            # context_length must match the window length the checkpoint was trained on.
            # seed_sweep keeps window_size inside `hyperparams` (it is a _NON_MODEL_HP, so
            # it is stripped from the model ctor kwargs but still recorded in the meta), so
            # read it from there first; fall back to a top-level key, then 60. Resolution
            # order below: explicit ctor context_length -> hyperparams.window_size ->
            # top-level window_size -> 60.
            _ws = hp.get("window_size")
            _meta_window_size = _ws if _ws is not None else meta.get("window_size")

        # Filter to only recognised ctor kwargs — strips training/Optuna metadata
        model_kwargs = {k: v for k, v in hp.items() if k in _CTOR_PARAMS}

        self._model = FVGxLSTMClassifier(
            input_size=model_kwargs.get("input_size", 5),
            embedding_dim=model_kwargs.get("embedding_dim", 64),
            num_blocks=model_kwargs.get("num_blocks", 2),
            num_heads=model_kwargs.get("num_heads", 4),
            num_classes=model_kwargs.get("num_classes", 3),
            dropout=model_kwargs.get("dropout", 0.1),
            head_dropout=model_kwargs.get("head_dropout", 0.3),
            context_length=(
                model_kwargs["context_length"]
                if "context_length" in model_kwargs
                else (_meta_window_size if _meta_window_size is not None else 60)
            ),
        )
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        # Support both bare state_dict and wrapped {"model_state_dict": ...}
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self._model.load_state_dict(state)
        self._model.train(False)

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Run xLSTM inference on normalised windows.

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
