"""base.py — ModelAdapter ABC for the SMC inspection framework."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ModelAdapter(ABC):
    """Abstract base for all model adapters used in inspect_models.

    Subclasses must:
    - Set ``name`` as a class attribute (str, unique, lowercase).
    - Implement ``predict_proba`` accepting (N, 60, 5) float32 and returning (N, 3) float32.
    - Be stateless after ``__init__`` — no internal mutation during inference.

    Windows passed to ``predict_proba`` are NORMALISED (same normalisation as training).
    XGBoost adapters that need raw OHLCV must undo or bypass normalisation internally
    using the raw windows provided by the runner as a separate argument.
    """

    name: str  # must be set as class attribute

    @abstractmethod
    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Run inference on a batch of windows.

        Parameters
        ----------
        windows : np.ndarray, shape (N, 60, 5), dtype float32
            Normalised OHLCV windows.

        Returns
        -------
        np.ndarray, shape (N, 3), dtype float32
            Class probabilities for [none (0), bullish (1), bearish (2)].
            Rows should sum to ~1.0 (or very close).
        """
        ...

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """Return argmax class predictions.

        Parameters
        ----------
        windows : np.ndarray, shape (N, 60, 5), dtype float32
            Normalised OHLCV windows.

        Returns
        -------
        np.ndarray, shape (N,), dtype int
        """
        return np.argmax(self.predict_proba(windows), axis=1).astype(int)
