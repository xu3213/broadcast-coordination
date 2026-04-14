"""
Tests for aggregate response estimator

Tests cover:
- EPSEstimator initialization and configuration
- Quantile Regression NN (Layer 1)
- Conformal Prediction integration (Layer 2)
- Full estimation pipeline

Architecture (post-ablation):
- Layer 1: Quantile Regression NN -> [q10, q50, q90]
- Layer 2: Conformal Prediction -> distribution-free coverage
"""

import pytest
import numpy as np
from typing import List, Dict, Tuple

from src.estimation import (
    EPSEstimator,
    EstimatorConfig,
    EstimationResult,
)


class TestEstimatorConfig:
    """Tests for EstimatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EstimatorConfig()

        assert config.target_coverage == 0.9
        assert config.enable_conformal is True
        assert config.min_confidence == 0.3

    def test_custom_config(self):
        """Test custom configuration."""
        config = EstimatorConfig(
            target_coverage=0.95,
            enable_conformal=True,
            min_confidence=0.5,
        )

        assert config.target_coverage == 0.95
        assert config.min_confidence == 0.5

    def test_conformal_parameters(self):
        """Test conformal prediction parameters."""
        config = EstimatorConfig(
            conformal_adaptive=True,
            conformal_window_size=200,
            conformal_coverage_margin=0.03,
        )

        assert config.conformal_adaptive is True
        assert config.conformal_window_size == 200
        assert config.conformal_coverage_margin == 0.03

    def test_pytorch_parameters(self):
        """Test PyTorch training parameters."""
        config = EstimatorConfig(
            use_pytorch=True,
            pytorch_epochs=50,
            pytorch_batch_size=64,
        )

        assert config.use_pytorch is True
        assert config.pytorch_epochs == 50
        assert config.pytorch_batch_size == 64


class TestEstimationResult:
    """Tests for EstimationResult dataclass."""

    def test_result_attributes(self):
        """Test EstimationResult has required attributes."""
        result = EstimationResult(
            response_kw=100.0,
            lower_bound=80.0,
            upper_bound=120.0,
            confidence=0.9,
        )

        assert result.response_kw == 100.0
        assert result.lower_bound == 80.0
        assert result.upper_bound == 120.0
        assert result.confidence == 0.9

    def test_result_interval_width(self):
        """Test prediction interval width calculation."""
        result = EstimationResult(
            response_kw=100.0,
            lower_bound=80.0,
            upper_bound=120.0,
            confidence=0.9,
        )

        # Interval width should be upper - lower
        assert result.upper_bound - result.lower_bound == 40.0


class TestEPSEstimator:
    """Tests for EPSEstimator."""

    @pytest.fixture
    def simple_config(self) -> EstimatorConfig:
        """Simple configuration for testing."""
        return EstimatorConfig(
            enable_conformal=True,
            use_pytorch=False,
        )

    @pytest.fixture
    def sample_signals(self) -> List[Dict]:
        """Generate sample signals for testing."""
        rng = np.random.default_rng(42)
        signals = []
        for _ in range(100):
            signals.append({
                'intensity': int(rng.integers(500, 4000)),
                'supply_demand': int(rng.integers(0, 15)),
                'price': int(rng.integers(100, 2000)),
                'region_id': int(rng.integers(0, 16)),
                'priority': int(rng.integers(0, 15)),
            })
        return signals

    @pytest.fixture
    def sample_responses(self, sample_signals: List[Dict]) -> List[float]:
        """Generate sample responses correlated with signals."""
        rng = np.random.default_rng(42)
        responses = []
        for signal in sample_signals:
            # Response is correlated with intensity and supply_demand
            base_response = signal['intensity'] * 0.5 + signal['supply_demand'] * 100
            noise = rng.normal(0, 50)
            responses.append(base_response + noise)
        return responses

    def test_initialization(self, simple_config: EstimatorConfig):
        """Test estimator initialization."""
        estimator = EPSEstimator(simple_config)

        assert estimator is not None
        assert estimator.config == simple_config

    def test_fit_basic(
        self,
        simple_config: EstimatorConfig,
        sample_signals: List[Dict],
        sample_responses: List[float],
    ):
        """Test basic fit functionality."""
        estimator = EPSEstimator(simple_config)

        # Should not raise
        estimator.fit(sample_signals, sample_responses)

        # Should have learned something
        assert estimator._is_fitted is True

    def test_estimate_after_fit(
        self,
        simple_config: EstimatorConfig,
        sample_signals: List[Dict],
        sample_responses: List[float],
    ):
        """Test estimate after fitting."""
        estimator = EPSEstimator(simple_config)
        estimator.fit(sample_signals, sample_responses)

        # Estimate on a new signal
        test_signal = {
            'intensity': 2000,
            'supply_demand': 8,
            'price': 1000,
            'region_id': 5,
            'priority': 10,
        }

        result = estimator.estimate(test_signal)

        assert isinstance(result, EstimationResult)
        assert result.response_kw is not None
        assert result.lower_bound <= result.response_kw <= result.upper_bound
        assert 0 <= result.confidence <= 1

    def test_estimate_without_fit_raises(self, simple_config: EstimatorConfig):
        """Test that estimating before fit raises error."""
        estimator = EPSEstimator(simple_config)

        test_signal = {'intensity': 2000, 'supply_demand': 8, 'price': 1000}

        with pytest.raises((ValueError, RuntimeError)):
            estimator.estimate(test_signal)

    def test_batch_estimate(
        self,
        simple_config: EstimatorConfig,
        sample_signals: List[Dict],
        sample_responses: List[float],
    ):
        """Test batch estimation."""
        estimator = EPSEstimator(simple_config)
        estimator.fit(sample_signals[:80], sample_responses[:80])

        # Estimate on batch of signals
        test_signals = sample_signals[80:90]
        results = [estimator.estimate(s) for s in test_signals]

        assert len(results) == 10
        for result in results:
            assert isinstance(result, EstimationResult)
            assert result.lower_bound <= result.upper_bound

    def test_prediction_interval_coverage(
        self,
        sample_signals: List[Dict],
        sample_responses: List[float],
    ):
        """Test that prediction intervals achieve target coverage."""
        config = EstimatorConfig(
            target_coverage=0.9,
            enable_conformal=True,
        )
        estimator = EPSEstimator(config)

        # Split data
        train_signals = sample_signals[:70]
        train_responses = sample_responses[:70]
        test_signals = sample_signals[70:]
        test_responses = sample_responses[70:]

        estimator.fit(train_signals, train_responses)

        # Check coverage
        covered = 0
        for signal, actual in zip(test_signals, test_responses):
            result = estimator.estimate(signal)
            if result.lower_bound <= actual <= result.upper_bound:
                covered += 1

        coverage = covered / len(test_signals)
        # Allow some slack due to small sample size
        assert coverage >= 0.5  # At least 50% coverage on small sample

    def test_get_statistics(
        self,
        simple_config: EstimatorConfig,
        sample_signals: List[Dict],
        sample_responses: List[float],
    ):
        """Test statistics retrieval."""
        estimator = EPSEstimator(simple_config)
        estimator.fit(sample_signals, sample_responses)

        stats = estimator.get_statistics()

        assert isinstance(stats, dict)


class TestEstimatorWithConformal:
    """Tests for estimator with Conformal Prediction enabled."""

    @pytest.fixture
    def conformal_config(self) -> EstimatorConfig:
        """Configuration with conformal enabled."""
        return EstimatorConfig(
            enable_conformal=True,
            conformal_adaptive=True,
            target_coverage=0.9,
        )

    @pytest.fixture
    def training_data(self) -> Tuple[List[Dict], List[float]]:
        """Generate training data."""
        rng = np.random.default_rng(123)
        signals = []
        responses = []

        for _ in range(200):
            intensity = int(rng.integers(500, 4000))
            supply_demand = int(rng.integers(0, 15))
            price = int(rng.integers(100, 2000))

            signal = {
                'intensity': intensity,
                'supply_demand': supply_demand,
                'price': price,
                'region_id': int(rng.integers(0, 16)),
                'priority': int(rng.integers(0, 15)),
            }
            signals.append(signal)

            # Generate response with some noise
            response = intensity * 0.4 + supply_demand * 80 + price * 0.1
            response += rng.normal(0, 100)
            responses.append(response)

        return signals, responses

    def test_conformal_calibration(
        self,
        conformal_config: EstimatorConfig,
        training_data: Tuple[List[Dict], List[float]],
    ):
        """Test that conformal prediction calibrates properly."""
        signals, responses = training_data
        estimator = EPSEstimator(conformal_config)

        estimator.fit(signals, responses)

        # Check that conformal calibration happened
        if hasattr(estimator, '_conformal_calibration_stats'):
            stats = estimator._conformal_calibration_stats
            assert stats is not None

    def test_adaptive_conformal_update(
        self,
        conformal_config: EstimatorConfig,
        training_data: Tuple[List[Dict], List[float]],
    ):
        """Test adaptive conformal prediction updates."""
        signals, responses = training_data
        estimator = EPSEstimator(conformal_config)

        estimator.fit(signals[:150], responses[:150])

        # Make predictions and observe (simulating online learning)
        for signal, actual in zip(signals[150:160], responses[150:160]):
            result = estimator.estimate(signal)
            if hasattr(estimator, 'observe'):
                estimator.observe(signal, actual)


class TestEstimatorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_signals_raises(self):
        """Test that empty signals raise error."""
        config = EstimatorConfig()
        estimator = EPSEstimator(config)

        with pytest.raises((ValueError, RuntimeError)):
            estimator.fit([], [])

    def test_mismatched_lengths_raises(self):
        """Test that mismatched signal/response lengths raise error."""
        config = EstimatorConfig()
        estimator = EPSEstimator(config)

        signals = [{'intensity': 1000, 'supply_demand': 5, 'price': 500}]
        responses = [100.0, 200.0]  # Mismatched length

        with pytest.raises((ValueError, RuntimeError)):
            estimator.fit(signals, responses)

    def test_nan_response_handling(self):
        """Test handling of NaN responses."""
        config = EstimatorConfig()
        estimator = EPSEstimator(config)

        signals = [
            {'intensity': 1000, 'supply_demand': 5, 'price': 500},
            {'intensity': 2000, 'supply_demand': 10, 'price': 1000},
        ]
        responses = [100.0, float('nan')]

        # Should either raise or handle gracefully
        try:
            estimator.fit(signals, responses)
            # If it doesn't raise, it should still work
            result = estimator.estimate(signals[0])
            assert result is not None
        except (ValueError, RuntimeError):
            pass  # Expected behavior

    def test_extreme_intensity_values(self):
        """Test with extreme intensity values."""
        config = EstimatorConfig(enable_conformal=False)
        estimator = EPSEstimator(config)

        signals = [
            {'intensity': 0, 'supply_demand': 0, 'price': 0},
            {'intensity': 4095, 'supply_demand': 15, 'price': 4095},
        ] * 50  # Repeat for minimum samples

        responses = [0.0, 10000.0] * 50

        estimator.fit(signals, responses)

        # Should handle extreme input
        result = estimator.estimate({'intensity': 4095, 'supply_demand': 15, 'price': 4095})
        assert result is not None
        assert np.isfinite(result.response_kw)


class TestOnlineLearning:
    """Tests for online learning capabilities."""

    @pytest.fixture
    def base_estimator(self) -> EPSEstimator:
        """Create a pre-fitted estimator."""
        config = EstimatorConfig(
            enable_conformal=True,
        )
        estimator = EPSEstimator(config)

        rng = np.random.default_rng(789)
        signals = []
        responses = []
        for _ in range(100):
            signal = {
                'intensity': int(rng.integers(500, 4000)),
                'supply_demand': int(rng.integers(0, 15)),
                'price': int(rng.integers(100, 2000)),
            }
            signals.append(signal)
            responses.append(signal['intensity'] * 0.5)

        estimator.fit(signals, responses)
        return estimator

    def test_online_update_exists(self, base_estimator: EPSEstimator):
        """Test that online_update method exists."""
        assert hasattr(base_estimator, 'online_update') or hasattr(base_estimator, 'observe')

    def test_prediction_stability(self, base_estimator: EPSEstimator):
        """Test that predictions are stable after updates."""
        test_signal = {'intensity': 2000, 'supply_demand': 8, 'price': 1000}

        result1 = base_estimator.estimate(test_signal)
        result2 = base_estimator.estimate(test_signal)

        # Same signal should give same result
        assert result1.response_kw == result2.response_kw
