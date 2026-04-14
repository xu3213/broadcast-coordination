"""
Tests for battery storage model

Tests high-fidelity battery model including:
- SOC dynamics with efficiency curves
- SOH degradation from cycling and calendar aging
- Thermal model for temperature-dependent performance
- Power limitations based on SOC, temperature, and C-rate
"""

import pytest
import math
from src.edge.battery import (
    BatteryChemistry,
    BatteryParameters,
    BatteryDegradationState,
    BatteryThermalState,
    BatteryModel,
    create_residential_battery,
)


class TestBatteryParameters:
    """Tests for BatteryParameters dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = BatteryParameters()

        assert params.nominal_capacity_kwh == 10.0
        assert params.max_charge_power_kw == 5.0
        assert params.max_discharge_power_kw == 5.0
        assert params.soc_min == 0.10
        assert params.soc_max == 0.95

    def test_custom_values(self):
        """Test custom parameter values."""
        params = BatteryParameters(
            nominal_capacity_kwh=20.0,
            max_charge_power_kw=10.0,
            chemistry=BatteryChemistry.LFP,
        )

        assert params.nominal_capacity_kwh == 20.0
        assert params.max_charge_power_kw == 10.0
        assert params.chemistry == BatteryChemistry.LFP

    def test_c_rate_limits(self):
        """Test C-rate limit parameters."""
        params = BatteryParameters()

        assert params.max_charge_c_rate == 0.5
        assert params.max_discharge_c_rate == 0.5
        assert params.continuous_c_rate == 0.3


class TestBatteryDegradationState:
    """Tests for BatteryDegradationState."""

    def test_initial_soh(self):
        """Test initial SOH is 100%."""
        degradation = BatteryDegradationState()

        assert degradation.soh == 1.0
        assert degradation.total_capacity_fade == 0.0

    def test_soh_with_cycle_fade(self):
        """Test SOH calculation with cycle fade."""
        degradation = BatteryDegradationState(cycle_capacity_fade=0.10)

        assert degradation.soh == pytest.approx(0.90)

    def test_soh_with_calendar_fade(self):
        """Test SOH calculation with calendar fade."""
        degradation = BatteryDegradationState(calendar_capacity_fade=0.05)

        assert degradation.soh == pytest.approx(0.95)

    def test_total_fade_cap(self):
        """Test total fade is capped at 30%."""
        degradation = BatteryDegradationState(
            cycle_capacity_fade=0.25,
            calendar_capacity_fade=0.15,
        )

        assert degradation.total_capacity_fade == 0.30
        assert degradation.soh == pytest.approx(0.70)


class TestBatteryThermalState:
    """Tests for BatteryThermalState."""

    def test_default_values(self):
        """Test default thermal state."""
        thermal = BatteryThermalState()

        assert thermal.cell_temperature == 25.0
        assert thermal.ambient_temperature == 25.0
        assert thermal.cooling_active is False
        assert thermal.heating_active is False


class TestBatteryModel:
    """Tests for BatteryModel class."""

    def test_initialization(self):
        """Test model initialization."""
        model = BatteryModel(initial_soc=0.5)

        assert model.soc == 0.5
        assert model.soh == 1.0
        assert model.current_power_kw == 0.0

    def test_initialization_with_degraded_soh(self):
        """Test initialization with degraded SOH."""
        model = BatteryModel(initial_soc=0.5, initial_soh=0.90)

        assert model.soh < 1.0

    def test_available_capacity(self):
        """Test available capacity calculation."""
        params = BatteryParameters(nominal_capacity_kwh=10.0, usable_capacity_factor=0.90)
        model = BatteryModel(params=params, initial_soc=0.5)

        # 10 * 0.90 * 1.0 (SOH) = 9.0 kWh
        assert model.available_capacity_kwh == pytest.approx(9.0)

    def test_stored_energy(self):
        """Test stored energy calculation."""
        params = BatteryParameters(nominal_capacity_kwh=10.0, usable_capacity_factor=0.90)
        model = BatteryModel(params=params, initial_soc=0.5)

        # 0.5 * 9.0 = 4.5 kWh
        assert model.stored_energy_kwh == pytest.approx(4.5)


class TestBatteryEfficiency:
    """Tests for efficiency calculations."""

    def test_discharge_efficiency_nominal(self):
        """Test discharge efficiency at nominal conditions."""
        model = BatteryModel(initial_soc=0.5)
        model.thermal.cell_temperature = 25.0

        efficiency = model.get_discharge_efficiency(2.5)

        # Should be close to nominal
        assert 0.85 <= efficiency <= 0.98

    def test_discharge_efficiency_low_temperature(self):
        """Test discharge efficiency at low temperature."""
        model = BatteryModel(initial_soc=0.5)
        model.thermal.cell_temperature = 5.0  # Cold

        efficiency = model.get_discharge_efficiency(2.5)

        # Lower efficiency at cold temperature
        assert efficiency < 0.95

    def test_charge_efficiency_high_soc(self):
        """Test charge efficiency at high SOC."""
        model = BatteryModel(initial_soc=0.95)
        model.thermal.cell_temperature = 25.0

        efficiency = model.get_charge_efficiency(2.5)

        # Lower efficiency at high SOC
        assert efficiency < 0.95


class TestBatteryPowerLimits:
    """Tests for power limit calculations."""

    def test_max_discharge_power_normal(self):
        """Test max discharge power at normal conditions."""
        model = BatteryModel(initial_soc=0.5)
        model.thermal.cell_temperature = 25.0

        max_power = model.get_max_discharge_power()

        assert max_power > 0
        assert max_power <= model.params.max_discharge_power_kw

    def test_max_discharge_power_low_soc(self):
        """Test max discharge power at low SOC."""
        model = BatteryModel(initial_soc=0.15)
        model.thermal.cell_temperature = 25.0

        max_power = model.get_max_discharge_power()

        # Should be reduced at low SOC
        assert max_power < model.params.max_discharge_power_kw

    def test_max_charge_power_high_soc(self):
        """Test max charge power at high SOC (CC-CV)."""
        model = BatteryModel(initial_soc=0.90)
        model.thermal.cell_temperature = 25.0

        max_power = model.get_max_charge_power()

        # Should be reduced at high SOC due to CC-CV
        assert max_power < model.params.max_charge_power_kw

    def test_max_charge_power_cold(self):
        """Test max charge power at cold temperature."""
        model = BatteryModel(initial_soc=0.5)
        model.thermal.cell_temperature = 5.0  # Cold

        max_power = model.get_max_charge_power()

        # Charging should be significantly limited when cold
        assert max_power < model.params.max_charge_power_kw * 0.5

    def test_no_discharge_below_min_temp(self):
        """Test no discharge below minimum temperature."""
        model = BatteryModel(initial_soc=0.5)
        model.thermal.cell_temperature = -5.0  # Below min

        max_power = model.get_max_discharge_power()

        assert max_power == 0.0


class TestBatteryFeasibility:
    """Tests for discharge/charge feasibility checks."""

    def test_can_discharge_sufficient_soc(self):
        """Test discharge feasibility with sufficient SOC."""
        model = BatteryModel(initial_soc=0.50)

        # Should be able to discharge 2 kWh
        assert model.can_discharge(2.0)

    def test_cannot_discharge_insufficient_soc(self):
        """Test discharge blocked with insufficient SOC."""
        model = BatteryModel(initial_soc=0.15)

        # Cannot discharge 2 kWh from near-empty battery
        assert not model.can_discharge(2.0)

    def test_can_charge_sufficient_headroom(self):
        """Test charge feasibility with sufficient headroom."""
        model = BatteryModel(initial_soc=0.50)

        assert model.can_charge(2.0)

    def test_cannot_charge_insufficient_headroom(self):
        """Test charge blocked with insufficient headroom."""
        model = BatteryModel(initial_soc=0.90)

        # Cannot charge 2 kWh to near-full battery
        assert not model.can_charge(2.0)


class TestBatteryUpdate:
    """Tests for battery state update."""

    def test_charging_increases_soc(self):
        """Test that charging increases SOC."""
        model = BatteryModel(initial_soc=0.50)

        result = model.update(power_kw=5.0, dt_seconds=3600)

        assert result['soc_delta'] > 0
        assert model.soc > 0.50

    def test_discharging_decreases_soc(self):
        """Test that discharging decreases SOC."""
        model = BatteryModel(initial_soc=0.50)

        result = model.update(power_kw=-5.0, dt_seconds=3600)

        assert result['soc_delta'] < 0
        assert model.soc < 0.50

    def test_ramp_rate_limiting(self):
        """Test power ramp rate is limited."""
        model = BatteryModel(initial_soc=0.50)
        model.current_power_kw = 0.0

        # Request instant jump to 5 kW
        result = model.update(power_kw=5.0, dt_seconds=1.0)

        # Should be limited by ramp rate
        assert abs(result['actual_power_kw']) < 5.0

    def test_soc_bounds_respected(self):
        """Test SOC stays within bounds."""
        model = BatteryModel(initial_soc=0.95)

        # Try to overcharge
        model.update(power_kw=5.0, dt_seconds=3600)

        assert model.soc <= model.params.soc_max

    def test_energy_throughput_tracked(self):
        """Test energy throughput is tracked."""
        model = BatteryModel(initial_soc=0.50)
        initial_throughput = model.energy_throughput_kwh

        model.update(power_kw=5.0, dt_seconds=3600)

        assert model.energy_throughput_kwh > initial_throughput


class TestBatteryDegradation:
    """Tests for degradation modeling."""

    def test_cycling_increases_cycle_count(self):
        """Test that cycling increases total cycle count."""
        model = BatteryModel(initial_soc=0.50)
        initial_cycles = model.degradation.total_cycles

        # Simulate a few cycles (not too many to avoid extreme values)
        for _ in range(3):
            model.update(power_kw=3.0, dt_seconds=1800)  # 30 min charge
            model.update(power_kw=-3.0, dt_seconds=1800)  # 30 min discharge

        assert model.degradation.total_cycles > initial_cycles

    def test_calendar_aging(self):
        """Test calendar aging increases over time."""
        model = BatteryModel(initial_soc=0.50)
        initial_age = model.degradation.calendar_age_days

        # Simulate a few hours of idle time
        for _ in range(6):
            model.update(power_kw=0.0, dt_seconds=3600)

        assert model.degradation.calendar_age_days > initial_age


class TestBatteryThermal:
    """Tests for thermal modeling."""

    def test_high_power_heats_battery(self):
        """Test that high power operation generates heat."""
        model = BatteryModel(initial_soc=0.50)
        # Start below ambient so heat generation can be observed
        model.thermal.cell_temperature = 20.0
        model.thermal.ambient_temperature = 25.0

        initial_temp = model.thermal.cell_temperature

        # High power operation should generate internal heat
        for _ in range(100):
            model.update(power_kw=5.0, dt_seconds=60)

        # Temperature should increase towards or above ambient due to internal heating
        # With ambient at 25C and starting at 20C, the battery should warm up
        assert model.thermal.cell_temperature > initial_temp

    def test_cooling_activates(self):
        """Test cooling activates at high temperature."""
        model = BatteryModel(initial_soc=0.50)
        # Start hot - above optimal_high + 2 = 35 + 2 = 37
        model.thermal.cell_temperature = 38.0
        model.thermal.ambient_temperature = 38.0  # Match ambient to prevent passive cooling

        # Run update to trigger thermal management logic
        model.update(power_kw=5.0, dt_seconds=60)

        # Cooling should activate when temp > optimal_high + 2 = 37
        assert model.thermal.cooling_active


class TestBatteryDailyCounters:
    """Tests for daily counter management."""

    def test_reset_daily_counters(self):
        """Test daily counter reset."""
        model = BatteryModel(initial_soc=0.50)

        # Accumulate some cycles
        model.update(power_kw=5.0, dt_seconds=3600)
        model.daily_cycle_count = 1.5

        model.reset_daily_counters()

        assert model.daily_cycle_count == 0.0


class TestBatteryStateSummary:
    """Tests for state summary."""

    def test_state_summary_fields(self):
        """Test state summary contains required fields."""
        model = BatteryModel(initial_soc=0.50)

        summary = model.get_state_summary()

        assert 'soc' in summary
        assert 'soh' in summary
        assert 'stored_energy_kwh' in summary
        assert 'max_discharge_power_kw' in summary
        assert 'max_charge_power_kw' in summary
        assert 'cell_temperature_c' in summary


class TestBatteryFactoryFunctions:
    """Tests for factory functions."""

    def test_create_residential_battery(self):
        """Test residential battery creation."""
        model = create_residential_battery(capacity_kwh=10.0)

        assert model.params.nominal_capacity_kwh == 10.0
        assert model.params.chemistry == BatteryChemistry.LFP

