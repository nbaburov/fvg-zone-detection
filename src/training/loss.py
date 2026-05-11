"""loss.py — Loss functions for FVG classification.

WeightedCE: default for all architectures (CrossEntropyLoss with class weights).
FocalLoss: alternative ablation if minority-F1 is stuck near zero.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCE:
    """Wrapper: nn.CrossEntropyLoss(weight=class_weights).

    Default loss function for all Phase 4 architectures.
    class_weights must already be on the correct device.
    """

    def __init__(self, class_weights: torch.Tensor) -> None:
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) raw logits
            targets: (B,) long tensor of class indices

        Returns:
            scalar loss tensor
        """
        return self.criterion(logits, targets)


class FocalLoss:
    """Focal loss — alternative for LSTM ablation only if minority-F1 is stuck near zero.

    gamma=1.0 is a mild setting; standard is 2.0. Use 1.0 here to reduce aggression
    on a small noisy dataset.
    """

    def __init__(self, class_weights: torch.Tensor, gamma: float = 1.0) -> None:
        self.class_weights = class_weights
        self.gamma = gamma

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) raw logits
            targets: (B,) long tensor

        Returns:
            scalar focal loss tensor
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Gather log-probs + probs for the correct class
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)          # (B,)

        # Focal factor
        focal_factor = (1.0 - p_t) ** self.gamma

        # Per-sample class weight
        sample_weights = self.class_weights[targets]  # (B,)

        loss = -(sample_weights * focal_factor * log_p_t).mean()
        return loss
