"""_model_registrations.py — Apply @register_model to all model classes.

Leaf module: imports from models/ only. No imports back into config/.
Imported by registry.py to ensure decorators fire on module load.
"""

from __future__ import annotations

from src.config.registry import register_model
from src.models.cnn_lstm import FVGCNNLSTMClassifier
from src.models.lstm import FVGLSTMClassifier
from src.models.xgboost_baseline import XGBoostFVGClassifier

register_model("lstm")(FVGLSTMClassifier)
register_model("xgb")(XGBoostFVGClassifier)
register_model("cnn_lstm")(FVGCNNLSTMClassifier)
