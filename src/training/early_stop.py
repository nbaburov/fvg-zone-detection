"""early_stop.py — EarlyStop with EMA smoothing for noisy val metrics.

EMA smoothing: smoothed = alpha * metric + (1 - alpha) * prev_smoothed
Patience counted against smoothed metric, not raw metric.
"""

from __future__ import annotations


class EarlyStop:
    """Early stopping with EMA-smoothed metric tracking.

    Args:
        patience: epochs without improvement before stopping
        min_delta: minimum improvement in smoothed metric to count as improvement
        ema_alpha: EMA smoothing factor (0.0 = no update, 1.0 = no smoothing)
        mode: "max" (F1, accuracy) or "min" (loss)
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        ema_alpha: float = 0.3,
        mode: str = "max",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.ema_alpha = ema_alpha
        self.mode = mode

        self._smoothed: float | None = None
        self._best: float | None = None
        self._counter: int = 0
        self._stopped: bool = False

    def update(self, metric: float) -> bool:
        """Update with a new metric value.

        Returns True if training should stop.
        """
        # Initialise smoothed on first call
        if self._smoothed is None:
            self._smoothed = metric
        else:
            self._smoothed = self.ema_alpha * metric + (1.0 - self.ema_alpha) * self._smoothed

        # Check improvement
        if self._best is None:
            self._best = self._smoothed
            self._counter = 0
            return False

        improved = (
            (self._smoothed > self._best + self.min_delta)
            if self.mode == "max"
            else (self._smoothed < self._best - self.min_delta)
        )

        if improved:
            self._best = self._smoothed
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience:
            self._stopped = True
            return True

        return False

    @property
    def smoothed(self) -> float:
        """Current EMA-smoothed metric value."""
        if self._smoothed is None:
            return 0.0
        return self._smoothed

    @property
    def best(self) -> float:
        """Best smoothed metric seen so far."""
        if self._best is None:
            return 0.0
        return self._best

    @property
    def counter(self) -> int:
        """Epochs without improvement."""
        return self._counter
