"""
Conformal Quantile Regression (CQR)

Distribution-free prediction intervals with finite-sample coverage
guarantee >= 1 - alpha (Methods, Eq. 5-8).

    Nonconformity score:  e_i = max(q_10(x_i) - y_i, y_i - q_90(x_i))
    Prediction interval:  [q_10(x) - eta, q_90(x) + eta]
    where eta = ceil((n_cal + 1)(1 - alpha))-th smallest score.

Reference: Romano et al. (2019) NeurIPS.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Callable
from enum import Enum
import math
import random
import time
import bisect


class ConformalMethod(Enum):
    """Conformal prediction methods."""
    SPLIT = "split"              # Split conformal (efficient)
    FULL = "full"                # Full conformal (computationally expensive)
    ADAPTIVE = "adaptive"        # Adaptive conformal intervals
    QUANTILE = "quantile"        # Conformalized quantile regression


@dataclass
class ConformalConfig:
    """Configuration for conformal prediction."""

    # Target coverage probability
    target_coverage: float = 0.9  # 90% prediction intervals

    # Coverage margin: set to 0.0 for theoretically correct CQR
    # CQR itself provides finite-sample coverage guarantee via conformal calibration
    coverage_margin: float = 0.0

    # Method selection
    method: ConformalMethod = ConformalMethod.ADAPTIVE

    # Calibration set parameters
    min_calibration_samples: int = 50
    max_calibration_samples: int = 1000

    # Adaptive method parameters
    adaptive_window: int = 100  # Window for local calibration
    use_difficulty_adjustment: bool = True

    # Online update parameters
    enable_online_update: bool = True
    update_batch_size: int = 10


@dataclass
class PredictionInterval:
    """Prediction interval from conformal prediction."""

    # Point estimate (from underlying model)
    point_estimate: float

    # Interval bounds
    lower_bound: float
    upper_bound: float

    # Target and actual coverage
    target_coverage: float = 0.9
    estimated_coverage: float = 0.9

    # Interval quality metrics
    interval_width: float = field(init=False)
    relative_width: float = field(init=False)

    # Calibration information
    n_calibration_samples: int = 0
    quantile_threshold: float = 0.0  # q_{1-alpha}

    def __post_init__(self):
        self.interval_width = self.upper_bound - self.lower_bound
        if abs(self.point_estimate) > 1e-6:
            self.relative_width = self.interval_width / abs(self.point_estimate)
        else:
            self.relative_width = float('inf') if self.interval_width > 0 else 0.0

    def contains(self, value: float) -> bool:
        """Check if value is within the interval."""
        return self.lower_bound <= value <= self.upper_bound


class NonconformityScore:
    """
    Computes nonconformity scores for conformal prediction.

    The nonconformity score measures how "strange" a prediction is relative
    to the calibration data. Common choices:
    - Absolute residual: |y - ŷ|
    - Normalized residual: |y - ŷ| / σ(x)
    - Quantile-based: max(q_low - y, y - q_high)
    """

    def __init__(self, score_type: str = "absolute"):
        """
        Initialize nonconformity scorer.

        Args:
            score_type: One of 'absolute', 'normalized', 'quantile'
        """
        self.score_type = score_type

    def compute(
        self,
        y_true: float,
        y_pred: float,
        uncertainty: float = 1.0,
        quantiles: Optional[Tuple[float, float]] = None,
    ) -> float:
        """
        Compute nonconformity score.

        Args:
            y_true: True value
            y_pred: Predicted value
            uncertainty: Predicted uncertainty (for normalized)
            quantiles: (lower, upper) quantiles (for quantile score)

        Returns:
            Nonconformity score (higher = more strange)
        """
        if self.score_type == "absolute":
            return abs(y_true - y_pred)

        elif self.score_type == "normalized":
            sigma = max(uncertainty, 1e-6)
            return abs(y_true - y_pred) / sigma

        elif self.score_type == "quantile":
            if quantiles is None:
                return abs(y_true - y_pred)
            q_low, q_high = quantiles
            # Standard CQR: allow negative scores (y inside [q_low, q_high])
            return max(q_low - y_true, y_true - q_high)

        else:
            return abs(y_true - y_pred)


class CalibrationSet:
    """
    Manages the calibration set for conformal prediction.

    Supports:
    - Fixed calibration set (split conformal)
    - Sliding window (online updating)
    - Difficulty-stratified calibration
    """

    def __init__(
        self,
        max_size: int = 1000,
        enable_stratification: bool = False,
    ):
        self.max_size = max_size
        self.enable_stratification = enable_stratification

        # Storage for calibration examples (deque for O(1) FIFO eviction)
        self._scores: deque = deque(maxlen=max_size)
        self._timestamps: deque = deque(maxlen=max_size)
        self._features: deque = deque(maxlen=max_size)
        self._difficulties: deque = deque(maxlen=max_size)

        # Sorted scores for efficient quantile computation
        self._sorted_scores: List[float] = []
        self._needs_sort = True

    def add(
        self,
        score: float,
        timestamp: Optional[float] = None,
        features: Optional[List[float]] = None,
        difficulty: Optional[float] = None,
    ):
        """Add a calibration example."""
        self._scores.append(score)
        self._timestamps.append(timestamp or time.time())
        self._features.append(features or [])
        self._difficulties.append(difficulty or 1.0)

        self._needs_sort = True
        # deque(maxlen=max_size) automatically evicts oldest on overflow

    def add_batch(
        self,
        scores: List[float],
        timestamps: Optional[List[float]] = None,
        features: Optional[List[List[float]]] = None,
        difficulties: Optional[List[float]] = None,
    ):
        """Add multiple calibration examples."""
        for i, score in enumerate(scores):
            ts = timestamps[i] if timestamps else None
            feat = features[i] if features else None
            diff = difficulties[i] if difficulties else None
            self.add(score, ts, feat, diff)

    def get_quantile(self, alpha: float) -> float:
        """
        Get the (1-alpha) quantile of calibration scores.

        For 90% coverage, alpha = 0.1, return q_{0.9}

        Args:
            alpha: Miscoverage rate

        Returns:
            Quantile threshold
        """
        if not self._scores:
            return float('inf')

        if self._needs_sort:
            self._sorted_scores = sorted(self._scores)
            self._needs_sort = False

        n = len(self._sorted_scores)

        # Finite sample correction: use ceiling((n+1)(1-alpha))/n quantile
        # This ensures coverage is at least 1-alpha
        idx = math.ceil((n + 1) * (1 - alpha)) - 1
        idx = max(0, min(idx, n - 1))

        return self._sorted_scores[idx]

    def get_local_quantile(
        self,
        alpha: float,
        query_features: List[float],
        k_neighbors: int = 50,
    ) -> float:
        """
        Get locally-weighted quantile for adaptive conformal.

        Uses k-nearest neighbors in feature space.
        """
        if not self._features or not self._scores:
            return self.get_quantile(alpha)

        # Compute distances to query
        distances = []
        for i, feat in enumerate(self._features):
            if feat:
                dist = sum((a - b) ** 2 for a, b in zip(query_features, feat))
                distances.append((math.sqrt(dist), self._scores[i]))

        if not distances:
            return self.get_quantile(alpha)

        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[0])
        k = min(k_neighbors, len(distances))
        local_scores = sorted([d[1] for d in distances[:k]])

        # Compute quantile on local scores
        idx = math.ceil((k + 1) * (1 - alpha)) - 1
        idx = max(0, min(idx, k - 1))

        return local_scores[idx]

    @property
    def size(self) -> int:
        return len(self._scores)

    def clear(self):
        """Clear all calibration data."""
        self._scores.clear()
        self._timestamps.clear()
        self._features.clear()
        self._difficulties.clear()
        self._sorted_scores.clear()
        self._needs_sort = True


class ConformalPredictor:
    """
    Conformal prediction wrapper for any point predictor.

    Provides prediction intervals with guaranteed coverage probability.
    """

    def __init__(
        self,
        config: Optional[ConformalConfig] = None,
        score_type: str = "absolute",
    ):
        """
        Initialize conformal predictor.

        Args:
            config: Conformal configuration
            score_type: Nonconformity score type
        """
        self.config = config or ConformalConfig()
        self.scorer = NonconformityScore(score_type)
        self.calibration = CalibrationSet(
            max_size=self.config.max_calibration_samples,
            enable_stratification=self.config.use_difficulty_adjustment,
        )

        self._is_calibrated = False
        self._coverage_history: List[bool] = []

    def calibrate(
        self,
        y_true: List[float],
        y_pred: List[float],
        uncertainties: Optional[List[float]] = None,
        features: Optional[List[List[float]]] = None,
    ) -> Dict[str, float]:
        """
        Calibrate conformal predictor on held-out data.

        Args:
            y_true: True values
            y_pred: Predicted values
            uncertainties: Predicted uncertainties (optional)
            features: Feature vectors for adaptive method (optional)

        Returns:
            Calibration metrics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        n = len(y_true)
        uncertainties = uncertainties or [1.0] * n
        features = features or [[] for _ in range(n)]

        # Compute nonconformity scores
        scores = []
        for i in range(n):
            score = self.scorer.compute(
                y_true[i], y_pred[i], uncertainties[i]
            )
            scores.append(score)

        # Add to calibration set
        self.calibration.add_batch(scores, features=features)

        self._is_calibrated = self.calibration.size >= self.config.min_calibration_samples

        # Compute calibration metrics
        alpha = 1.0 - self.config.target_coverage
        threshold = self.calibration.get_quantile(alpha)

        return {
            'n_samples': n,
            'calibration_size': self.calibration.size,
            'is_calibrated': self._is_calibrated,
            'threshold': threshold,
            'mean_score': sum(scores) / n,
            'max_score': max(scores),
        }

    def predict(
        self,
        point_estimate: float,
        uncertainty: float = 1.0,
        features: Optional[List[float]] = None,
    ) -> PredictionInterval:
        """
        Generate prediction interval with conformal guarantee.

        Args:
            point_estimate: Point prediction from underlying model
            uncertainty: Predicted uncertainty
            features: Feature vector for adaptive method

        Returns:
            PredictionInterval with guaranteed coverage
        """
        # Use target coverage directly (no artificial margin)
        alpha = 1.0 - self.config.target_coverage

        # Get quantile threshold
        if self.config.method == ConformalMethod.ADAPTIVE and features:
            threshold = self.calibration.get_local_quantile(
                alpha, features, self.config.adaptive_window
            )
        else:
            threshold = self.calibration.get_quantile(alpha)

        # Construct interval based on score type
        if self.scorer.score_type == "normalized":
            # Scale by uncertainty
            margin = threshold * uncertainty
        else:
            margin = threshold

        lower = point_estimate - margin
        upper = point_estimate + margin

        return PredictionInterval(
            point_estimate=point_estimate,
            lower_bound=lower,
            upper_bound=upper,
            target_coverage=self.config.target_coverage,
            estimated_coverage=self._estimate_actual_coverage(),
            n_calibration_samples=self.calibration.size,
            quantile_threshold=threshold,
        )

    def update(
        self,
        y_true: float,
        y_pred: float,
        uncertainty: float = 1.0,
        features: Optional[List[float]] = None,
    ):
        """
        Online update with new observation.

        Args:
            y_true: Observed true value
            y_pred: Predicted value
            uncertainty: Predicted uncertainty
            features: Feature vector
        """
        if not self.config.enable_online_update:
            return

        # Compute and store score
        score = self.scorer.compute(y_true, y_pred, uncertainty)
        self.calibration.add(score, features=features)

        # Track coverage
        alpha = 1.0 - self.config.target_coverage
        threshold = self.calibration.get_quantile(alpha)
        covered = score <= threshold
        self._coverage_history.append(covered)

        # Bound history
        if len(self._coverage_history) > 1000:
            self._coverage_history = self._coverage_history[-1000:]

    def _estimate_actual_coverage(self) -> float:
        """Estimate actual coverage from history."""
        if len(self._coverage_history) < 20:
            return self.config.target_coverage

        recent = self._coverage_history[-100:]
        return sum(recent) / len(recent)

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def get_statistics(self) -> Dict[str, Any]:
        """Get conformal predictor statistics."""
        return {
            'is_calibrated': self._is_calibrated,
            'calibration_size': self.calibration.size,
            'target_coverage': self.config.target_coverage,
            'estimated_coverage': self._estimate_actual_coverage(),
            'coverage_history_size': len(self._coverage_history),
            'method': self.config.method.value,
        }

    def reset(self):
        """Reset calibration data."""
        self.calibration.clear()
        self._coverage_history.clear()
        self._is_calibrated = False


class CQRPredictor:
    """
    Conformalized Quantile Regression (CQR) predictor.

    This is the key innovation that truly utilizes q10/q90 from Quantile NN:
    - Uses [q10, q90] as the base interval (captures conditional uncertainty)
    - Calibrates with conformal adjustment to guarantee coverage

    Reference: Romano et al. (2019) "Conformalized Quantile Regression"

    The nonconformity score for CQR is:
        score = max(q10 - y, y - q90)

    If y is within [q10, q90], score <= 0 (good)
    If y is outside [q10, q90], score > 0 (interval needs expansion)

    Final interval: [q10 - adjustment, q90 + adjustment]
    where adjustment = quantile(scores, 1 - alpha)
    """

    def __init__(
        self,
        config: Optional[ConformalConfig] = None,
    ):
        """
        Initialize CQR predictor.

        Args:
            config: Conformal configuration
        """
        self.config = config or ConformalConfig()
        # Global scores and adjustment (fallback)
        self._cqr_scores: List[float] = []
        self._adjustment: float = 0.0
        # Separate calibration for positive (charge) and negative (discharge) samples
        # This significantly reduces interval width for each direction
        self._pos_cqr_scores: List[float] = []  # Charge samples (y > 0)
        self._neg_cqr_scores: List[float] = []  # Discharge samples (y < 0)
        self._pos_adjustment: float = 0.0
        self._neg_adjustment: float = 0.0
        self._is_calibrated = False
        self._coverage_history: List[bool] = []

        # ACI (Adaptive Conformal Inference) state — Gibbs & Candes (2021)
        # α_{t+1} = α_t + γ × (err_t - α_target)
        # err_t = 1 if not covered, 0 if covered
        self._aci_alpha: float = 1.0 - self.config.target_coverage
        self._aci_gamma: float = 0.005  # Small learning rate per Gibbs & Candes (2021)
        self._aci_min_alpha: float = 0.03  # ≈97% coverage max
        self._aci_max_alpha: float = 0.20  # ≈80% coverage min
        self._aci_warmup: int = 20  # Minimum samples before ACI kicks in

    def calibrate(
        self,
        y_true: List[float],
        q10_pred: List[float],
        q90_pred: List[float],
    ) -> Dict[str, float]:
        """
        Calibrate CQR predictor using quantile predictions.

        Uses separate calibration for positive (charge) and negative (discharge)
        samples to achieve tighter intervals for each direction.

        Args:
            y_true: True values from calibration set
            q10_pred: 10th percentile predictions
            q90_pred: 90th percentile predictions

        Returns:
            Calibration metrics
        """
        if len(y_true) != len(q10_pred) or len(y_true) != len(q90_pred):
            raise ValueError("All inputs must have same length")

        n = len(y_true)
        if n == 0:
            return {'n_samples': 0, 'is_calibrated': False}

        # Compute CQR nonconformity scores with direction separation
        # score = max(q10 - y, y - q90)
        # If y in [q10, q90]: score <= 0
        # If y outside: score > 0 (how much we need to expand)
        self._cqr_scores = []
        self._pos_cqr_scores = []
        self._neg_cqr_scores = []

        for i in range(n):
            # Standard CQR nonconformity score (Romano et al. 2019):
            # score = max(q10 - y, y - q90)
            # Negative scores indicate y is inside [q10, q90], which is essential
            # for CQR's adaptive interval mechanism — negative scores pull the
            # adjustment down, allowing intervals to tighten when the base
            # quantile estimates are already well-calibrated.
            score = max(q10_pred[i] - y_true[i], y_true[i] - q90_pred[i])
            self._cqr_scores.append(score)
            # Separate by response direction
            if y_true[i] >= 0:
                self._pos_cqr_scores.append(score)
            else:
                self._neg_cqr_scores.append(score)

        # Use ACI alpha (starts at effective coverage, adapts online)
        alpha = self._aci_alpha

        # Store sorted scores for dynamic alpha lookup in predict()/update()
        self._sorted_cqr_scores = sorted(self._cqr_scores)
        self._sorted_pos_scores = sorted(self._pos_cqr_scores) if self._pos_cqr_scores else []
        self._sorted_neg_scores = sorted(self._neg_cqr_scores) if self._neg_cqr_scores else []

        def _quantile_adjustment(sorted_scores: List[float], a: float) -> float:
            if not sorted_scores:
                return 0.0
            n_s = len(sorted_scores)
            idx = math.ceil((n_s + 1) * (1 - a)) - 1
            idx = max(0, min(idx, n_s - 1))
            return float(sorted_scores[idx])

        # Compute global adjustment (fallback)
        self._adjustment = _quantile_adjustment(self._sorted_cqr_scores, alpha)

        # Compute separate adjustments for charge/discharge (tighter per direction)
        self._pos_adjustment = _quantile_adjustment(self._sorted_pos_scores, alpha) if self._sorted_pos_scores else self._adjustment
        self._neg_adjustment = _quantile_adjustment(self._sorted_neg_scores, alpha) if self._sorted_neg_scores else self._adjustment

        self._is_calibrated = n >= self.config.min_calibration_samples

        # Compute metrics
        positive_scores = [s for s in self._cqr_scores if s > 0]
        base_coverage = 1.0 - len(positive_scores) / n if n > 0 else 0.0

        return {
            'n_samples': n,
            'n_pos_samples': len(self._pos_cqr_scores),
            'n_neg_samples': len(self._neg_cqr_scores),
            'is_calibrated': self._is_calibrated,
            'adjustment': self._adjustment,
            'pos_adjustment': self._pos_adjustment,
            'neg_adjustment': self._neg_adjustment,
            'base_coverage': base_coverage,  # Coverage of raw [q10, q90]
            'mean_score': sum(self._cqr_scores) / n,
            'max_score': max(self._cqr_scores),
            'min_score': min(self._cqr_scores),
        }

    def predict(
        self,
        q10: float,
        q50: float,
        q90: float,
        apply_boundary_constraint: bool = True,
    ) -> PredictionInterval:
        """
        Generate prediction interval using CQR with direction-specific calibration.

        Uses separate adjustments for charge (positive) and discharge (negative)
        predictions, resulting in tighter, more accurate intervals.

        The interval is [q10 - adjustment, q90 + adjustment]
        where adjustment is direction-specific for better accuracy.

        Args:
            q10: 10th percentile prediction from Quantile NN
            q50: 50th percentile prediction (point estimate)
            q90: 90th percentile prediction from Quantile NN
            apply_boundary_constraint: If True, apply physical boundary constraints
                - Discharge (q50 < 0): upper bound capped at 0
                - Charge (q50 > 0): lower bound floored at 0

        Returns:
            PredictionInterval with guaranteed coverage
        """
        # Direction-specific adjustment requires sufficient calibration samples
        # per pool to be reliable. With small pools (<30), quantile estimates
        # have high variance → over-coverage. Fall back to pooled adjustment.
        MIN_POOL_SIZE = 30
        if q50 >= 0:
            if len(self._pos_cqr_scores) >= MIN_POOL_SIZE:
                adjustment = self._pos_adjustment
            else:
                adjustment = self._adjustment
        else:
            if len(self._neg_cqr_scores) >= MIN_POOL_SIZE:
                adjustment = self._neg_adjustment
            else:
                adjustment = self._adjustment

        # CQR interval: expand [q10, q90] by the calibrated adjustment
        lower = q10 - adjustment
        upper = q90 + adjustment

        # Apply physical boundary constraints
        # This ensures discharge intervals don't cross into positive territory
        # and charge intervals don't cross into negative territory
        if apply_boundary_constraint:
            if q50 < 0:
                # Discharge response: upper bound should not exceed 0
                upper = min(upper, 0.0)
            elif q50 > 0:
                # Charge response: lower bound should not go below 0
                lower = max(lower, 0.0)

        return PredictionInterval(
            point_estimate=q50,
            lower_bound=lower,
            upper_bound=upper,
            target_coverage=self.config.target_coverage,
            estimated_coverage=self._estimate_actual_coverage(),
            n_calibration_samples=len(self._cqr_scores),
            quantile_threshold=adjustment,  # Return direction-specific adjustment
        )

    def update(
        self,
        y_true: float,
        q10_pred: float,
        q50_pred: float,
        q90_pred: float,
    ) -> None:
        """
        Online update for CQR calibration (Gibbs & Candes 2021 protocol).

        Correct ordering to avoid lookahead bias:
        1. Check coverage using CURRENT adjustment (before seeing y_true's score)
        2. Append new nonconformity score and recompute adjustments
        3. Update ACI alpha based on coverage outcome
        """
        # STEP 1: Check coverage using CURRENT (pre-update) adjustment
        # This is critical: the interval must be generated WITHOUT the current
        # sample's nonconformity score to avoid lookahead bias.
        interval = self.predict(q10=q10_pred, q50=q50_pred, q90=q90_pred)
        covered = interval.contains(y_true)
        self._coverage_history.append(covered)
        if len(self._coverage_history) > 1000:
            self._coverage_history = self._coverage_history[-1000:]

        # STEP 2: Append new nonconformity score and update sorted lists
        score = max(q10_pred - y_true, y_true - q90_pred)
        self._cqr_scores.append(score)
        if y_true >= 0:
            self._pos_cqr_scores.append(score)
        else:
            self._neg_cqr_scores.append(score)

        # Maintain bounded window
        max_size = getattr(self.config, 'max_calibration_samples', 1000)
        if len(self._cqr_scores) > max_size:
            self._cqr_scores = self._cqr_scores[-max_size:]
            # Truncate direction-specific lists BEFORE rebuilding sorted lists
            if len(self._pos_cqr_scores) > max_size:
                self._pos_cqr_scores = self._pos_cqr_scores[-max_size:]
            if len(self._neg_cqr_scores) > max_size:
                self._neg_cqr_scores = self._neg_cqr_scores[-max_size:]
            self._sorted_cqr_scores = sorted(self._cqr_scores)
            self._sorted_pos_scores = sorted(self._pos_cqr_scores)
            self._sorted_neg_scores = sorted(self._neg_cqr_scores)
        else:
            # O(n) insert into sorted list (vs O(n log n) full re-sort)
            bisect.insort(self._sorted_cqr_scores, score)
            if y_true >= 0:
                bisect.insort(self._sorted_pos_scores, score)
            else:
                bisect.insort(self._sorted_neg_scores, score)
        if len(self._pos_cqr_scores) > max_size:
            self._pos_cqr_scores = self._pos_cqr_scores[-max_size:]
        if len(self._neg_cqr_scores) > max_size:
            self._neg_cqr_scores = self._neg_cqr_scores[-max_size:]

        alpha = self._aci_alpha

        def _quantile_adjustment(sorted_scores: List[float], a: float) -> float:
            if not sorted_scores:
                return 0.0
            n_s = len(sorted_scores)
            idx = math.ceil((n_s + 1) * (1 - a)) - 1
            idx = max(0, min(idx, n_s - 1))
            return float(sorted_scores[idx])

        self._adjustment = _quantile_adjustment(self._sorted_cqr_scores, alpha)
        self._pos_adjustment = _quantile_adjustment(self._sorted_pos_scores, alpha) if self._sorted_pos_scores else self._adjustment
        self._neg_adjustment = _quantile_adjustment(self._sorted_neg_scores, alpha) if self._sorted_neg_scores else self._adjustment

        self._is_calibrated = len(self._cqr_scores) >= getattr(self.config, 'min_calibration_samples', 50)

        # STEP 3: ACI alpha update — Gibbs & Candes (2021) exact online update
        if len(self._coverage_history) >= self._aci_warmup:
            err_t = 0.0 if covered else 1.0
            target_alpha = 1.0 - self.config.target_coverage
            self._aci_alpha += self._aci_gamma * (err_t - target_alpha)
            self._aci_alpha = max(self._aci_min_alpha, min(self._aci_max_alpha, self._aci_alpha))

    def _estimate_actual_coverage(self) -> float:
        """Estimate actual coverage from history."""
        if len(self._coverage_history) < 20:
            return self.config.target_coverage

        recent = self._coverage_history[-100:]
        return sum(recent) / len(recent)

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def adjustment(self) -> float:
        """Get the current CQR adjustment value."""
        return self._adjustment

    def get_statistics(self) -> Dict[str, Any]:
        """Get CQR predictor statistics with direction-specific and ACI information."""
        return {
            'is_calibrated': self._is_calibrated,
            'n_calibration_samples': len(self._cqr_scores),
            'n_pos_samples': len(self._pos_cqr_scores),
            'n_neg_samples': len(self._neg_cqr_scores),
            'adjustment': self._adjustment,
            'pos_adjustment': self._pos_adjustment,
            'neg_adjustment': self._neg_adjustment,
            'target_coverage': self.config.target_coverage,
            'estimated_coverage': self._estimate_actual_coverage(),
            'aci_alpha': self._aci_alpha,
            'aci_effective_coverage': 1.0 - self._aci_alpha,
            'method': 'cqr_direction_aware_aci',
        }

    def reset(self):
        """Reset calibration data and ACI state."""
        self._cqr_scores.clear()
        self._pos_cqr_scores.clear()
        self._neg_cqr_scores.clear()
        self._sorted_cqr_scores = []
        self._sorted_pos_scores = []
        self._sorted_neg_scores = []
        self._adjustment = 0.0
        self._pos_adjustment = 0.0
        self._neg_adjustment = 0.0
        self._coverage_history.clear()
        self._is_calibrated = False
        # Reset ACI alpha to initial value
        self._aci_alpha = 1.0 - self.config.target_coverage


class CoverageTracker:
    """
    Tracks empirical coverage of prediction intervals.

    Useful for:
    - Validating conformal prediction coverage guarantee
    - Diagnosing calibration issues
    - Comparing interval methods
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._records: List[Tuple[float, bool, float]] = []  # (timestamp, covered, width)

    def record(
        self,
        interval: PredictionInterval,
        true_value: float,
        timestamp: Optional[float] = None,
    ):
        """Record an interval evaluation."""
        covered = interval.contains(true_value)
        width = interval.interval_width
        ts = timestamp or time.time()
        self._records.append((ts, covered, width))

        # Maintain window
        if len(self._records) > self.window_size * 2:
            self._records = self._records[-self.window_size:]

    def get_empirical_coverage(self, n_recent: Optional[int] = None) -> float:
        """Get empirical coverage rate."""
        if not self._records:
            return 0.0

        n = n_recent or self.window_size
        recent = self._records[-n:]
        covered_count = sum(1 for _, c, _ in recent if c)
        return covered_count / len(recent)

    def get_average_width(self, n_recent: Optional[int] = None) -> float:
        """Get average interval width."""
        if not self._records:
            return 0.0

        n = n_recent or self.window_size
        recent = self._records[-n:]
        widths = [w for _, _, w in recent]
        return sum(widths) / len(widths)

    def get_coverage_by_time(
        self,
        bucket_size: float = 3600.0,
    ) -> List[Tuple[float, float]]:
        """Get coverage rate by time bucket."""
        if not self._records:
            return []

        min_ts = min(r[0] for r in self._records)
        max_ts = max(r[0] for r in self._records)

        buckets = []
        current = min_ts

        while current < max_ts:
            bucket_end = current + bucket_size
            bucket_records = [
                r for r in self._records
                if current <= r[0] < bucket_end
            ]
            if bucket_records:
                coverage = sum(1 for _, c, _ in bucket_records if c) / len(bucket_records)
                buckets.append((current, coverage))
            current = bucket_end

        return buckets

    def get_statistics(self) -> Dict[str, float]:
        """Get tracking statistics."""
        n = len(self._records)
        if n == 0:
            return {'n_records': 0}

        return {
            'n_records': n,
            'empirical_coverage': self.get_empirical_coverage(),
            'average_width': self.get_average_width(),
            'coverage_std': self._compute_coverage_std(),
        }

    def _compute_coverage_std(self) -> float:
        """Compute standard deviation of coverage in rolling windows."""
        if len(self._records) < self.window_size * 2:
            return 0.0

        # Compute coverage in sliding windows
        coverages = []
        for i in range(0, len(self._records) - self.window_size, self.window_size // 2):
            window = self._records[i:i + self.window_size]
            cov = sum(1 for _, c, _ in window if c) / len(window)
            coverages.append(cov)

        if len(coverages) < 2:
            return 0.0

        mean_cov = sum(coverages) / len(coverages)
        variance = sum((c - mean_cov) ** 2 for c in coverages) / (len(coverages) - 1)
        return math.sqrt(variance)


def create_conformal_predictor(
    method: str = "adaptive",
    target_coverage: float = 0.9,
    **kwargs,
) -> ConformalPredictor:
    """
    Factory function to create conformal predictor.

    Args:
        method: One of 'split', 'adaptive', 'quantile'
        target_coverage: Target coverage probability
        **kwargs: Additional configuration parameters

    Returns:
        Configured ConformalPredictor
    """
    config = ConformalConfig(
        target_coverage=target_coverage,
        method=ConformalMethod(method),
        **{k: v for k, v in kwargs.items() if hasattr(ConformalConfig, k)},
    )

    return ConformalPredictor(config)
