"""transformer.py — Transformer encoder classifier for FVG ternary prediction.

Architecture: Linear(5→d_model) + LearnedPosEnc(60) → TransformerEncoder(L layers) → pool → FC(d_model→3)
Input: (B, 60, 5) float32 — batch of 60-candle OHLCV windows (per-window normalised)
Output: (B, 3) raw logits — CrossEntropyLoss expects logits, not softmax
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FVGTransformerClassifier(nn.Module):
    """Transformer encoder for FVG 3-class classification.

    Input (B, 60, 5) is linearly projected to d_model, summed with learned
    positional encodings, then passed through a stack of TransformerEncoder layers.
    The sequence is reduced to a single vector via mean-pooling or a prepended CLS
    token (controlled by ``pool``), and a two-stage dropout head maps it to 3 logits.

    Unidirectional by design: standard (non-causal) self-attention is fine here
    because the 60-bar window is a fixed historical context — label leakage is not
    possible; bar 59 is the last bar and no future bars are in the window.
    """

    def __init__(
        self,
        input_size: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        num_classes: int = 3,
        pool: str = "mean",          # "mean" | "cls"
    ) -> None:
        super().__init__()

        if pool not in ("mean", "cls"):
            raise ValueError(f"pool must be 'mean' or 'cls', got {pool!r}")

        self.pool = pool
        self.d_model = d_model

        # --- input projection ---
        self.input_proj = nn.Linear(input_size, d_model)

        # --- CLS token (only used when pool == "cls") ---
        if pool == "cls":
            # shape (1, 1, d_model) — broadcast over batch dimension
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            seq_len = 60 + 1   # CLS + 60 bars
        else:
            seq_len = 60

        # --- learned positional encoding ---
        self.pos_enc = nn.Embedding(seq_len, d_model)
        # register position indices as a buffer so they move with .to(device)
        self.register_buffer(
            "_pos_ids",
            torch.arange(seq_len).unsqueeze(0),   # (1, seq_len)
            persistent=False,
        )

        # --- transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,   # expects (B, T, d_model)
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- classifier head ---
        self.head_dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_enc.weight, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) = (batch, 60, 5) float32

        Returns:
            (B, num_classes) raw logits — no softmax applied
        """
        B = x.size(0)

        # project each bar from input_size to d_model → (B, 60, d_model)
        x = self.input_proj(x)

        if self.pool == "cls":
            # prepend CLS token: (B, 1, d_model) ++ (B, 60, d_model) → (B, 61, d_model)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)   # (B, 61, d_model)

        # add learned positional encodings
        x = x + self.pos_enc(self._pos_ids)  # broadcasts (1, seq_len, d_model)

        # transformer encoder: (B, seq_len, d_model)
        x = self.encoder(x)

        # pool to single vector
        if self.pool == "cls":
            vec = x[:, 0, :]             # CLS position: (B, d_model)
        else:
            vec = x.mean(dim=1)          # mean over T: (B, d_model)

        # classifier head
        out = self.head_dropout(vec)
        return self.classifier(out)      # (B, num_classes)
