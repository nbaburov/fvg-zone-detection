"""cnn_lstm.py — CNN-LSTM classifier for FVG ternary prediction.

Architecture: Conv1d(5→F, k) → BN → ReLU [× n_conv_layers] → LSTM(F→H) → FC(H→3)
Input: (B, 60, 5) float32 — batch of 60-candle OHLCV windows (per-window normalised)
Output: (B, 3) raw logits
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FVGCNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        conv_filters: int = 32,
        kernel_size: int = 3,
        n_conv_layers: int = 2,
        use_pool: bool = False,
        pool_type: str = "max",      # "max" | "avg"
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.318,
        head_dropout: float = 0.526,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2   # 'same' padding: preserves L dimension for odd k

        conv_blocks: list[nn.Module] = []
        in_ch = input_size
        for _ in range(n_conv_layers):
            conv_blocks += [
                nn.Conv1d(in_ch, conv_filters, kernel_size, padding=padding),
                nn.BatchNorm1d(conv_filters),
                nn.ReLU(),
            ]
            in_ch = conv_filters
            if use_pool:
                # stride=1 preserves length — does NOT halve T
                Pool = nn.MaxPool1d if pool_type == "max" else nn.AvgPool1d
                conv_blocks.append(Pool(kernel_size=2, stride=1, padding=0))
        self.conv = nn.Sequential(*conv_blocks)

        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.head_dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) — conv expects (B, C, T)
        x = x.permute(0, 2, 1)          # (B, 5, 60)
        x = self.conv(x)                 # (B, conv_filters, T')
        x = x.permute(0, 2, 1)          # (B, T', conv_filters)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]            # (B, lstm_hidden)
        out = self.head_dropout(last_hidden)
        return self.classifier(out)      # (B, 3)
