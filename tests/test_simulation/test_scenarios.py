"""Tests for simulation scenarios."""

import pytest
import numpy as np
from src.simulation.scenarios import (
    ScenarioType,
    LoadProfile,
    RenewableProfile,
    ScenarioConfig,
    ScenarioGenerator,
    ScenarioMetrics,
)


class TestScenarioType:
    """Tests for ScenarioType enum."""

    def test_primary_scenario_types(self):
        """Test primary scenario types exist."""
        assert ScenarioType.NORMAL.value == "normal"
        assert ScenarioType.PEAK_SHAVING.value == "peak_shaving"
        assert ScenarioType.VALLEY_FILLING.value == "valley_filling"

    def test_emergency_scenario_types(self):
        """Test emergency scenario subdivisions."""
        assert ScenarioType.EMERGENCY.value == "emergency"
        assert ScenarioType.EMERGENCY_GRID_STABILITY.value == "emergency_grid_stability"
        assert ScenarioType.EMERGENCY_SUPPLY_SHORTAGE.value == "emergency_supply_shortage"
        # FREQUENCY_REGULATION kept for backward compatibility enum

    def test_renewable_integration_removed(self):
        """Verify RENEWABLE_INTEGRATION is no longer a scenario type."""
        # It's the system goal, not a scenario
        assert not hasattr(ScenarioType, 'RENEWABLE_INTEGRATION')


class TestLoadProfile:
    """Tests for LoadProfile."""

    def test_default_profile(self):
        profile = LoadProfile()
        assert profile.base_load_mw == 1000.0
        assert len(profile.hourly_factors) == 24

    def test_get_load(self):
        profile = LoadProfile()
        load = profile.get_load(12.0)
        assert load > 0

    def test_get_load_interpolation(self):
        profile = LoadProfile()
        load1 = profile.get_load(12.0)
        load2 = profile.get_load(12.5)
        load3 = profile.get_load(13.0)
        # Interpolated value should be between
        if load1 != load3:
            assert min(load1, load3) <= load2 <= max(load1, load3)

    def test_typical_residential(self):
        profile = LoadProfile.typical_residential()
        assert profile.peak_hour == 19
        assert profile.valley_hour == 4
        # Evening peak should be higher than early morning
        assert profile.hourly_factors[19] > profile.hourly_factors[4]

    def test_typical_commercial(self):
        profile = LoadProfile.typical_commercial()
        assert profile.peak_hour == 10
        # Business hours should have higher load
        assert profile.hourly_factors[10] > profile.hourly_factors[2]

    def test_high_demand(self):
        """Test high-demand profile for supply shortage scenario."""
        profile = LoadProfile.high_demand()
        assert profile.peak_hour == 15  # Afternoon peak (AC load)
        assert profile.base_load_mw == 1200.0  # Higher base load
        # Peak factor should exceed 1.0 (extreme conditions)
        assert max(profile.hourly_factors) >= 1.0


class TestRenewableProfile:
    """Tests for RenewableProfile."""

    def test_default_profile(self):
        profile = RenewableProfile()
        assert profile.solar_capacity_mw == 500.0
        assert profile.wind_capacity_mw == 200.0

    def test_get_solar(self):
        profile = RenewableProfile.typical_renewable()
        # Solar should be zero at midnight
        assert profile.get_solar(0) == 0
        # Solar should peak around noon
        assert profile.get_solar(12) > profile.get_solar(0)

    def test_get_wind(self):
        profile = RenewableProfile.typical_renewable()
        wind = profile.get_wind(12)
        assert wind >= 0

    def test_get_total(self):
        profile = RenewableProfile.typical_renewable()
        total = profile.get_total(12)
        assert total == profile.get_solar(12) + profile.get_wind(12)

    def test_typical_renewable(self):
        profile = RenewableProfile.typical_renewable()
        assert len(profile.solar_factors) == 24
        assert len(profile.wind_factors) == 24

    def test_high_variability(self):
        """Test high variability profile for grid stability testing."""
        profile = RenewableProfile.high_variability()
        assert len(profile.solar_factors) == 24
        # Check for variability (consecutive hours should differ)
        differences = np.abs(np.diff(profile.solar_factors[6:18]))
        assert np.any(differences > 0.1)  # At least some significant variation


class TestScenarioConfig:
    """Tests for ScenarioConfig."""

    def test_default_config(self):
        config = ScenarioConfig(
            name="Test",
            scenario_type=ScenarioType.PEAK_SHAVING,
            description="Test scenario",
        )
        assert config.duration_hours == 24.0
        assert config.target_response_rate == 0.5
        assert config.perspective == "supply"

    def test_config_with_renewable(self):
        config = ScenarioConfig(
            name="Renewable Test",
            scenario_type=ScenarioType.VALLEY_FILLING,
            description="Test with renewables",
            renewable_profile=RenewableProfile.typical_renewable(),
        )
        assert config.renewable_profile is not None

    def test_demand_side_perspective(self):
        """Test demand-side perspective for supply shortage."""
        config = ScenarioConfig(
            name="Supply Shortage",
            scenario_type=ScenarioType.EMERGENCY_SUPPLY_SHORTAGE,
            description="Urban power deficit",
            perspective="demand",
            supply_deficit_mw=500.0,
        )
        assert config.perspective == "demand"
        assert config.supply_deficit_mw == 500.0


class TestScenarioGenerator:
    """Tests for ScenarioGenerator - publication-grade scenarios."""

    def test_peak_shaving(self):
        """Test peak shaving scenario (supply-side, routine)."""
        config = ScenarioGenerator.peak_shaving()
        assert config.scenario_type == ScenarioType.PEAK_SHAVING
        assert config.target_peak_reduction == 0.20
        assert config.perspective == "supply"
        assert config.paper_description != ""

    def test_valley_filling(self):
        """Test valley filling scenario (supply-side, routine)."""
        config = ScenarioGenerator.valley_filling()
        assert config.scenario_type == ScenarioType.VALLEY_FILLING
        assert config.target_response_rate == 0.6
        assert config.perspective == "supply"

    def test_emergency_grid_stability(self):
        """Test grid stability scenario (supply-side, fast)."""
        config = ScenarioGenerator.emergency_grid_stability()
        assert config.scenario_type == ScenarioType.EMERGENCY_GRID_STABILITY
        assert config.emergency_intensity == 4000
        assert config.max_latency_ms == 100.0  # Critical: <100ms
        assert config.perspective == "supply"
        assert config.emergency_duration_minutes == 15.0  # Short duration

    def test_emergency_supply_shortage(self):
        """Test supply shortage scenario (demand-side, slow)."""
        config = ScenarioGenerator.emergency_supply_shortage()
        assert config.scenario_type == ScenarioType.EMERGENCY_SUPPLY_SHORTAGE
        assert config.perspective == "demand"
        assert config.max_latency_ms > 100.0  # Relaxed latency
        assert config.emergency_duration_minutes > 60.0  # Extended duration
        assert config.supply_deficit_mw > 0
        assert config.shortage_cause != ""

    def test_emergency_response_legacy(self):
        """Test legacy emergency_response() alias."""
        config = ScenarioGenerator.emergency_response()
        # Should be backward compatible
        assert config.scenario_type == ScenarioType.EMERGENCY
        assert config.max_latency_ms == 100.0

    def test_get_all_scenarios(self):
        """Test get_all_scenarios returns 4 scenarios."""
        scenarios = ScenarioGenerator.get_all_scenarios()
        assert len(scenarios) == 4
        names = [s.name for s in scenarios]
        assert "Peak Shaving" in names
        assert "Valley Filling" in names
        assert "Emergency - Grid Stability" in names
        assert "Emergency - Supply Shortage" in names

    def test_get_primary_scenarios(self):
        """Test get_primary_scenarios returns 3 scenarios."""
        scenarios = ScenarioGenerator.get_primary_scenarios()
        assert len(scenarios) == 3
        names = [s.name for s in scenarios]
        assert "Peak Shaving" in names
        assert "Valley Filling" in names
        assert "Emergency Response" in names

    def test_get_scenario_by_name(self):
        """Test scenario lookup by name."""
        # Test various name formats
        assert ScenarioGenerator.get_scenario_by_name("peak_shaving") is not None
        assert ScenarioGenerator.get_scenario_by_name("Peak Shaving") is not None
        assert ScenarioGenerator.get_scenario_by_name("emergency_grid_stability") is not None
        assert ScenarioGenerator.get_scenario_by_name("supply shortage") is not None

        # Test non-existent scenario
        assert ScenarioGenerator.get_scenario_by_name("nonexistent") is None

    def test_no_renewable_integration_scenario(self):
        """Verify renewable_integration is not a standalone scenario."""
        # Should not exist as a method
        assert not hasattr(ScenarioGenerator, 'renewable_integration')


class TestScenarioMetrics:
    """Tests for ScenarioMetrics."""

    def test_default_metrics(self):
        metrics = ScenarioMetrics()
        assert metrics.total_responses == 0
        assert metrics.response_rate == 0.0
        assert metrics.met_response_target is False

    def test_metrics_with_data(self):
        metrics = ScenarioMetrics(
            total_responses=5000,
            response_rate=0.6,
            avg_latency_ms=50.0,
            p99_latency_ms=95.0,
        )
        assert metrics.response_rate == 0.6

    def test_supply_shortage_metrics(self):
        """Test metrics specific to supply shortage scenario."""
        metrics = ScenarioMetrics(
            response_rate=0.7,
            coverage_rate=0.85,
            sustained_duration_minutes=90.0,
        )
        assert metrics.coverage_rate == 0.85
        assert metrics.sustained_duration_minutes == 90.0

    def test_evaluate(self):
        metrics = ScenarioMetrics(
            response_rate=0.6,
            p99_latency_ms=80.0,
            peak_reduction_percent=0.18,
        )

        config = ScenarioConfig(
            name="Test",
            scenario_type=ScenarioType.PEAK_SHAVING,
            description="Test",
            target_response_rate=0.5,
            max_latency_ms=100.0,
            target_peak_reduction=0.15,
        )

        result = metrics.evaluate(config)

        assert result['overall_success'] is True
        assert result['response_rate']['met'] is True
        assert result['latency']['met'] is True
        assert result['perspective'] == "supply"

    def test_evaluate_failure(self):
        metrics = ScenarioMetrics(
            response_rate=0.3,  # Below target
            p99_latency_ms=150.0,  # Above target
        )

        config = ScenarioConfig(
            name="Test",
            scenario_type=ScenarioType.PEAK_SHAVING,
            description="Test",
            target_response_rate=0.5,
            max_latency_ms=100.0,
        )

        result = metrics.evaluate(config)

        assert result['overall_success'] is False
        assert result['response_rate']['met'] is False
        assert result['latency']['met'] is False


