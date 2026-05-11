"""lstm.py — 2-layer stacked LSTM classifier for FVG ternary prediction.

Architecture: LSTM(5→64, 2 layers) → final hidden state → Dropout(0.5) → Linear(64→3)
Input: (B, 60, 5) float32 — batch of 60-candle OHLCV windows (per-window normalised)
Output: (B, 3) raw logits — CrossEntropyLoss expects logits, not softmax
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FVGLSTMClassifier(nn.Module):
    """2-layer stacked LSTM for FVG 3-class classification.

    Uses nn.LSTM (sequence-batched), NOT nn.LSTMCell — faster on Apple MPS per CLAUDE.md.
    Unidirectional only — bidirectional would leak information at the label position (bar 59).
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        head_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,       # applied between LSTM layers (only active if num_layers > 1)
            batch_first=True,      # expects (B, T, C) — matches DataLoader collation
            bidirectional=False,   # unidirectional: bidirectional leaks at label position 59
        )
        self.head_dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) = (batch, 60, 5) float32

        Returns:
            (B, num_classes) raw logits — no softmax applied
        """
        # _, (h_n, _): h_n shape (num_layers, B, hidden_size)
        _, (h_n, _) = self.lstm(x)

        # Take last layer's final hidden state: (B, hidden_size)
        last_hidden = h_n[-1]

        # Classifier head
        out = self.head_dropout(last_hidden)
        logits = self.classifier(out)  # (B, num_classes)

        return logits
