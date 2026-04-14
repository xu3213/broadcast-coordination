"""
Signal Module - 64-bit Broadcast Signal Encoding/Decoding

Signal Structure (64-bit):
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ version  │ region   │ supply   │ intensity│ reserved │ priority │   crc    │
│   _ts    │   _id    │ _demand  │          │          │          │          │
│  8-bit   │  12-bit  │  4-bit   │  12-bit  │  12-bit  │  4-bit   │  12-bit  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
"""

from .encoder import EPSSignal, EPSSignalEncoder
from .decoder import EPSSignalDecoder
from .validator import EPSSignalValidator
from .optimizer import SignalOptimizer, OptimizationTarget, OptimizedSignal

__all__ = [
    "EPSSignal",
    "EPSSignalEncoder",
    "EPSSignalDecoder",
    "EPSSignalValidator",
    "SignalOptimizer",
    "OptimizationTarget",
    "OptimizedSignal",
]
