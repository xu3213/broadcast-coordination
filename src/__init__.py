"""
Broadcast Coordination Framework

Simulation and estimation modules for unidirectional
broadcast coordination of distributed energy storage.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"
__author__ = "Xu, Xie, Zhang"

__all__ = ["signal", "edge", "estimation", "simulation", "analysis"]


def __getattr__(name: str) -> Any:
    """Lazy imports to keep package import lightweight."""
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
