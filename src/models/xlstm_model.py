"""xlstm_model.py — xLSTM classifier for FVG ternary prediction.

Architecture: Linear(5→E) → xLSTMBlockStack(E, L blocks, sLSTM-only) → mean-pool → Dropout → FC(E→3)
Input: (B, 60, 5) float32 — batch of 60-candle OHLCV windows (per-window normalised)
Output: (B, 3) raw logits

Apple-Silicon config (CLAUDE.md):
  sLSTMLayerConfig(backend="vanilla") — avoids CUDA kernel path; float32 throughout.
  step_kernel / sequence_kernel were v1 params; xlstm 2.x exposes backend="vanilla" instead.

Installed: xlstm==2.0.5, mlstm_kernels==2.0.2
"""
from __future__ import annotations

try:
    from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
        sLSTMBlockConfig,
        sLSTMLayerConfig,
    )
    _XLSTM_AVAILABLE = True
except ImportError:
    _XLSTM_AVAILABLE = False

import torch
import torch.nn as nn


class FVGxLSTMClassifier(nn.Module):
    """xLSTM-based ternary classifier for FVG detection.

    Parameters
    ----------
    input_size:
        Number of input features per timestep (default 5 = OHLCV).
    embedding_dim:
        Projected dimension fed into the xLSTM stack.
    num_blocks:
        Number of xLSTM blocks stacked.
    num_heads:
        Number of sLSTM heads per block.
    num_classes:
        Output classes (default 3: none / bull-FVG / bear-FVG).
    dropout:
        Dropout applied inside the xLSTM stack.
    head_dropout:
        Dropout applied before the final linear classifier.
    context_length:
        Sequence length seen during training (must match T dimension = 60).
    """

    def __init__(
        self,
        input_size: int = 5,
        embedding_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 4,
        num_classes: int = 3,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        context_length: int = 60,
    ) -> None:
        super().__init__()

        if not _XLSTM_AVAILABLE:
            raise RuntimeError(
                "xlstm is not installed. Run: pip install xlstm\n"
                "FVGxLSTMClassifier cannot be instantiated without it."
            )

        self.input_proj = nn.Linear(input_size, embedding_dim)

        slstm_cfg = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="vanilla",            # Apple-Silicon safe — no CUDA kernel
                dtype="float32",             # full precision; no AMP on MPS
                enable_automatic_mixed_precision=False,
                num_heads=num_heads,
                dropout=dropout,
            )
        )

        stack_cfg = xLSTMBlockStackConfig(
            slstm_block=slstm_cfg,
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at="all",               # all blocks are sLSTM (no mLSTM)
            dropout=dropout,
            add_post_blocks_norm=True,
        )
        self.xlstm_stack = xLSTMBlockStack(stack_cfg)

        self.head_dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_size)
        x = self.input_proj(x)           # (B, T, embedding_dim)
        x = self.xlstm_stack(x)          # (B, T, embedding_dim)
        x = x.mean(dim=1)                # (B, embedding_dim) — mean-pool over time
        x = self.head_dropout(x)
        return self.classifier(x)        # (B, num_classes)
