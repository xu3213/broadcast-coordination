"""
Estimation Module

Two-layer architecture:
1. Dual Quantile NN — charge/discharge aggregate response prediction
2. Conformal Quantile Regression (CQR) — distribution-free coverage
"""

from .estimator import EPSEstimator, EstimationResult, EstimatorConfig

# Conformal Prediction (Layer 2)
from .conformal import (
    ConformalMethod,
    PredictionInterval,
    NonconformityScore,
    CalibrationSet,
    ConformalPredictor,
    CoverageTracker,
)

# Check PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = [
    # Main estimator
    "EPSEstimator",
    "EstimationResult",
    "EstimatorConfig",
    # Conformal Prediction
    "ConformalMethod",
    "PredictionInterval",
    "NonconformityScore",
    "CalibrationSet",
    "ConformalPredictor",
    "CoverageTracker",
    # PyTorch availability flag
    "TORCH_AVAILABLE",
]
