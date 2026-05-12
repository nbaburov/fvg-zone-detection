"""_loss_registrations.py — Apply @register_loss to all loss classes.

Leaf module: imports from training/ only. No imports back into config/.
Imported by registry.py to ensure decorators fire on module load.
"""

from __future__ import annotations

from src.config.registry import register_loss
from src.training.loss import FocalLoss, WeightedCE

register_loss("weighted_ce")(WeightedCE)
register_loss("focal")(FocalLoss)
