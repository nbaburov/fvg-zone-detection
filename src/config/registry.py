"""registry.py — Model and loss registries using @register_model / @register_loss decorators.

Pattern mirrors LABELLERS in src/data/labels/__init__.py.

Usage:
    from src.config.registry import MODELS, LOSSES, register_model, register_loss

    @register_model("lstm")
    class FVGLSTMClassifier(nn.Module): ...

    model_cls = MODELS["lstm"]
    instance = model_cls(hidden_size=128, ...)
"""

from __future__ import annotations

MODELS: dict[str, type] = {}
LOSSES: dict[str, type] = {}


def register_model(name: str):
    """Decorator: register a model class under the given arch name."""
    def dec(cls):
        MODELS[name] = cls
        return cls
    return dec


def register_loss(name: str):
    """Decorator: register a loss class under the given name."""
    def dec(cls):
        LOSSES[name] = cls
        return cls
    return dec


# Force-import concrete registrations so decorators fire.
# These are leaf modules with no back-imports into config/.
from src.config import _model_registrations  # noqa: E402, F401
from src.config import _loss_registrations   # noqa: E402, F401
