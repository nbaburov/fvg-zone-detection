"""Label registry — maps labeller name → class."""

from __future__ import annotations

from typing import Callable, TypeVar

LABELLERS: dict[str, type["BaseLabeller"]] = {}

T = TypeVar("T", bound="BaseLabeller")


def register(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator. @register("fvg") on a BaseLabeller subclass adds it to LABELLERS."""

    def decorator(cls: type[T]) -> type[T]:
        LABELLERS[name] = cls
        return cls

    return decorator


# Import concrete labellers so their @register decorators fire.
# Must come after LABELLERS + register are defined to avoid circular import.
from src.data.labels import fvg  # noqa: E402, F401
from src.data.labels import valid_fvg  # noqa: E402, F401
