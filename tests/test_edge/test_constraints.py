"""
Tests for physical constraints

Tests the constraint checking logic including:
- Power constraints
- SOC constraints
- Thermal constraints
- Device-specific constraints
- Grid interconnection constraints
"""

import pytest
import time
from src.edge.constraints import (
    ConstraintChecker,
    ConstraintResult,
    ConstraintViolation,
    PhysicalConstraints,
    GridStandard,
    GridConstraints,
    GridState,
    HarmonicConstraints,
    check_discharge_feasibility,
    check_charge_feasibility,
)
from src.edge.state_machine import BatteryState, DeviceState


class TestConstraintResult:
    """Tests for ConstraintResult class."""

    def test_default_not_violated(self):
        """Test default result is not violated."""
        result = ConstraintResult()
        assert not result.violated
        assert len(result.violations) == 0
        assert len(result.messages) == 0

    def test_add_violation(self):
        """Test adding a violation."""
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.MAX_POWER_EXCEEDED, "Test message")

        assert result.violated
        assert ConstraintViolation.MAX_POWER_EXCEEDED in result.violations
        assert "Test message" in result.messages

    def test_add_violation_without_message(self):
        """Test adding a violation without message."""
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.SOC_TOO_LOW)

        assert result.violated
        assert len(result.messages) == 0

    def test_is_safe_with_no_violations(self):
        """Test is_safe with no violations."""
        result = ConstraintResult()
        assert result.is_safe

    def test_is_safe_with_non_critical_violation(self):
        """Test is_safe with non-critical violation."""
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.SOC_TOO_LOW)

        assert result.violated
        assert result.is_safe  # SOC_TOO_LOW is not safety-critical

    def test_is_not_safe_with_critical_violation(self):
        """Test is_safe with critical violation."""
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.OVER_TEMPERATURE)

        assert result.violated
        assert not result.is_safe  # OVER_TEMPERATURE is safety-critical

    def test_is_not_safe_with_device_faulted(self):
        """Test is_safe with device faulted."""
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.DEVICE_FAULTED)

        assert not result.is_safe

    def test_is_not_safe_with_device_offline(self):
        """Test is_safe with device offline."""
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.DEVICE_OFFLINE)

        assert not result.is_safe


class TestPhysicalConstraints:
    """Tests for PhysicalConstraints defaults."""

    def test_default_values(self):
        """Test default constraint values."""
        c = PhysicalConstraints()

        assert c.max_charge_power == 10.0
        assert c.max_discharge_power == 10.0
        assert c.max_ramp_rate == 5.0
        assert c.min_soc == 0.1
        assert c.max_soc == 0.95
        assert c.safe_soc_range == (0.2, 0.8)
        assert c.max_temperature == 45.0
        assert c.min_temperature == 0.0
        assert c.max_daily_cycles == 10


class TestConstraintCheckerBattery:
    """Tests for ConstraintChecker with battery devices."""

    def setup_method(self):
        """Set up test fixtures."""
        self.checker = ConstraintChecker()
        self.state = BatteryState(
            device_id='bat_001',
            device_type='battery',
            soc=0.5,
            temperature=25.0,
        )

    def test_healthy_battery_passes(self):
        """Test healthy battery passes constraints."""
        result = self.checker.check(self.state)
        assert not result.violated

    def test_faulted_device_fails(self):
        """Test faulted device fails."""
        self.state.is_healthy = False
        self.state.fault_code = 'test_fault'

        result = self.checker.check(self.state)
        assert result.violated
        assert ConstraintViolation.DEVICE_FAULTED in result.violations
        assert not result.is_safe

    def test_soc_too_low(self):
        """Test SOC below minimum fails."""
        self.state.soc = 0.05  # Below 10% minimum

        result = self.checker.check(self.state)
        assert result.violated
        assert ConstraintViolation.SOC_TOO_LOW in result.violations

    def test_soc_too_high(self):
        """Test SOC above maximum fails."""
        self.state.soc = 0.98  # Above 95% maximum

        result = self.checker.check(self.state)
        assert result.violated
        assert ConstraintViolation.SOC_TOO_HIGH in result.violations

    def test_over_temperature(self):
        """Test over temperature fails."""
        self.state.temperature = 50.0  # Above 45C maximum

        result = self.checker.check(self.state)
        assert result.violated
        assert ConstraintViolation.OVER_TEMPERATURE in result.violations
        assert not result.is_safe

    def test_under_temperature(self):
        """Test under temperature fails."""
        self.state.temperature = -5.0  # Below 0C minimum

        result = self.checker.check(self.state)
        assert result.violated
        assert ConstraintViolation.UNDER_TEMPERATURE in result.violations

    def test_daily_cycles_exceeded(self):
        """Test daily cycles exceeded fails."""
        self.state.daily_cycles = 15  # Above 10 maximum

        result = self.checker.check(self.state)
        assert result.violated
        assert ConstraintViolation.DAILY_CYCLES_EXCEEDED in result.violations

    def test_discharge_at_low_soc_blocked(self):
        """Test discharge at low SOC is blocked."""
        self.state.soc = 0.2  # At safe minimum

        result = self.checker.check(self.state, target_power=-5.0)
        assert result.violated
        assert ConstraintViolation.SOC_TOO_LOW in result.violations

    def test_charge_at_high_soc_blocked(self):
        """Test charge at high SOC is blocked."""
        self.state.soc = 0.8  # At safe maximum

        result = self.checker.check(self.state, target_power=5.0)
        assert result.violated
        assert ConstraintViolation.SOC_TOO_HIGH in result.violations

    def test_max_charge_power_exceeded(self):
        """Test max charge power exceeded."""
        result = self.checker.check(self.state, target_power=15.0)  # Above 10kW max
        assert result.violated
        assert ConstraintViolation.MAX_POWER_EXCEEDED in result.violations

    def test_max_discharge_power_exceeded(self):
        """Test max discharge power exceeded."""
        result = self.checker.check(self.state, target_power=-15.0)  # Above 10kW max
        assert result.violated
        assert ConstraintViolation.MAX_POWER_EXCEEDED in result.violations

    def test_ramp_rate_exceeded(self):
        """Test ramp rate exceeded."""
        self.state.power_current = 0.0
        result = self.checker.check(self.state, target_power=8.0)  # 8kW/s > 5kW/s max
        assert result.violated
        assert ConstraintViolation.RAMP_RATE_EXCEEDED in result.violations


class TestFeasibilityFunctions:
    """Tests for feasibility check functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.state = BatteryState(
            device_id='bat_001',
            device_type='battery',
            soc=0.5,
            capacity_kwh=10.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
        )

    def test_discharge_feasibility_ok(self):
        """Test discharge feasibility when OK."""
        # 5kW for 1 hour = 5kWh / 0.95 = 5.26kWh
        # SOC drop = 5.26 / 10 = 0.526
        # Final SOC = 0.5 - 0.526 = -0.026 (NOT feasible)
        result = check_discharge_feasibility(self.state, 5.0, 3600)
        assert not result  # Would drop below 20%

        # Try smaller discharge
        result = check_discharge_feasibility(self.state, 1.0, 3600)
        assert result  # Would be above 20%

    def test_discharge_feasibility_blocked(self):
        """Test discharge feasibility when blocked."""
        self.state.soc = 0.3  # Low SOC
        # 5kW for 1 hour would definitely drop below 20%
        result = check_discharge_feasibility(self.state, 5.0, 3600)
        assert not result

    def test_charge_feasibility_ok(self):
        """Test charge feasibility when OK."""
        # 5kW for 1 hour = 5kWh * 0.95 = 4.75kWh
        # SOC increase = 4.75 / 10 = 0.475
        # Final SOC = 0.5 + 0.475 = 0.975 (NOT feasible, > 80%)
        result = check_charge_feasibility(self.state, 5.0, 3600)
        assert not result  # Would exceed 80%

        # Try smaller charge
        result = check_charge_feasibility(self.state, 1.0, 3600)
        assert result  # Would be below 80%

    def test_charge_feasibility_blocked(self):
        """Test charge feasibility when blocked."""
        self.state.soc = 0.7  # Already high
        result = check_charge_feasibility(self.state, 5.0, 3600)
        assert not result


class TestConstraintViolationEnum:
    """Tests for ConstraintViolation enum."""

    def test_all_violations_defined(self):
        """Test all expected violations are defined."""
        violations = list(ConstraintViolation)

        assert ConstraintViolation.MAX_POWER_EXCEEDED in violations
        assert ConstraintViolation.RAMP_RATE_EXCEEDED in violations
        assert ConstraintViolation.SOC_TOO_LOW in violations
        assert ConstraintViolation.SOC_TOO_HIGH in violations
        assert ConstraintViolation.DEPARTURE_SOC_RISK in violations
        assert ConstraintViolation.OVER_TEMPERATURE in violations
        assert ConstraintViolation.UNDER_TEMPERATURE in violations
        assert ConstraintViolation.MIN_RUN_TIME in violations
        assert ConstraintViolation.DEVICE_FAULTED in violations
        assert ConstraintViolation.DEVICE_OFFLINE in violations


class TestCustomConstraints:
    """Tests with custom constraint parameters."""

    def test_custom_power_limits(self):
        """Test custom power limits."""
        constraints = PhysicalConstraints(
            max_charge_power=5.0,
            max_discharge_power=3.0,
        )
        checker = ConstraintChecker(constraints)
        state = BatteryState(device_id='bat_001', device_type='battery', soc=0.5)

        # 6kW charge should fail (max 5kW)
        result = checker.check(state, target_power=6.0)
        assert result.violated
        assert ConstraintViolation.MAX_POWER_EXCEEDED in result.violations

        # 4kW discharge should fail (max 3kW)
        result = checker.check(state, target_power=-4.0)
        assert result.violated
        assert ConstraintViolation.MAX_POWER_EXCEEDED in result.violations

    def test_custom_soc_limits(self):
        """Test custom SOC limits."""
        constraints = PhysicalConstraints(
            min_soc=0.2,
            max_soc=0.9,
        )
        checker = ConstraintChecker(constraints)
        state = BatteryState(device_id='bat_001', device_type='battery', soc=0.15)

        result = checker.check(state)
        assert result.violated
        assert ConstraintViolation.SOC_TOO_LOW in result.violations


# =======================
# Grid Interconnection Constraint Tests
# =======================


class TestGridStandard:
    """Tests for GridStandard enum."""


class TestGridConstraints:
    """Tests for GridConstraints class."""

    def test_default_values_gb_t(self):
        """Test default values are grid standard."""
        gc = GridConstraints()

        assert gc.standard == GridStandard.STANDARD_B
        assert gc.nominal_frequency == 50.0
        # Frequency limits (grid standard)
        assert gc.freq_normal_min == 49.5
        assert gc.freq_normal_max == 50.2
        assert gc.freq_warning_min == 48.5
        assert gc.freq_warning_max == 50.5
        assert gc.freq_critical_min == 47.5
        assert gc.freq_critical_max == 51.5
        # Voltage limits
        assert gc.voltage_normal_min_pu == 0.85
        assert gc.voltage_normal_max_pu == 1.10
        # LVRT/HVRT enabled
        assert gc.lvrt_enabled is True
        assert gc.hvrt_enabled is True
        # Anti-islanding
        assert gc.anti_islanding_enabled is True

    def test_standard_a_category_3(self):
        """Test Standard A Category III factory method."""
        gc = GridConstraints.standard_a(category=3)

        assert gc.standard == GridStandard.STANDARD_A
        assert gc.nominal_frequency == 60.0
        assert gc.freq_normal_min == 59.5
        assert gc.freq_normal_max == 60.1
        assert gc.freq_warning_min == 58.5  # Category II/III
        assert gc.freq_critical_min == 56.5
        assert gc.freq_critical_max == 62.0
        assert gc.voltage_normal_min_pu == 0.88
        assert gc.voltage_normal_max_pu == 1.10
        assert gc.lvrt_threshold_50pct_duration == 2.0  # Category III
        assert gc.rocof_limit == 2.0

    def test_standard_a_category_1(self):
        """Test Standard A Category I factory method."""
        gc = GridConstraints.standard_a(category=1)

        assert gc.freq_warning_min == 59.0  # Less stringent for Category I
        assert gc.lvrt_threshold_20pct_duration == 0.0  # No requirement
        assert gc.rocof_limit == 3.0  # More lenient

    def test_standard_b_factory(self):
        """Test grid standard factory method."""
        gc = GridConstraints.standard_b()

        assert gc.standard == GridStandard.STANDARD_B
        assert gc.nominal_frequency == 50.0
        assert gc.freq_normal_min == 49.5
        assert gc.freq_normal_max == 50.2
        assert gc.voltage_critical_min_pu == 0.20  # LVRT 20% requirement
        assert gc.lvrt_threshold_20pct_duration == 0.625  # 625ms


class TestGridState:
    """Tests for GridState class."""

    def test_default_values(self):
        """Test default grid state is nominal."""
        gs = GridState()

        assert gs.frequency == 50.0
        assert gs.frequency_rate_of_change == 0.0
        assert gs.voltage_pu == 1.0
        assert gs.voltage_sag_duration == 0.0
        assert gs.voltage_swell_duration == 0.0
        assert gs.is_islanded is False
        assert gs.thd_voltage == 0.0
        assert gs.thd_current == 0.0

    def test_custom_state(self):
        """Test custom grid state values."""
        gs = GridState(
            frequency=49.8,
            voltage_pu=0.95,
            frequency_rate_of_change=-0.5,
        )

        assert gs.frequency == 49.8
        assert gs.voltage_pu == 0.95
        assert gs.frequency_rate_of_change == -0.5


class TestConstraintResultGridViolation:
    """Tests for ConstraintResult grid violation properties."""

    def test_has_grid_violation_false_by_default(self):
        """Test has_grid_violation is False with no violations."""
        result = ConstraintResult()
        assert not result.has_grid_violation

    def test_has_grid_violation_frequency(self):
        """Test has_grid_violation for frequency violations."""
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.FREQUENCY_OUT_OF_RANGE)
        assert result.has_grid_violation

    def test_has_grid_violation_voltage(self):
        """Test has_grid_violation for voltage violations."""
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.VOLTAGE_CRITICAL)
        assert result.has_grid_violation

    def test_has_grid_violation_lvrt(self):
        """Test has_grid_violation for LVRT violations."""
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.LVRT_VIOLATION)
        assert result.has_grid_violation

    def test_has_grid_violation_anti_islanding(self):
        """Test has_grid_violation for anti-islanding."""
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.ANTI_ISLANDING_TRIGGERED)
        assert result.has_grid_violation

    def test_is_safe_with_frequency_critical(self):
        """Test is_safe is False with critical frequency violation."""
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.FREQUENCY_CRITICAL)
        assert not result.is_safe

    def test_is_safe_with_voltage_critical(self):
        """Test is_safe is False with critical voltage violation."""
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.VOLTAGE_CRITICAL)
        assert not result.is_safe

    def test_is_safe_with_lvrt_violation(self):
        """Test is_safe is False with LVRT violation."""
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.LVRT_VIOLATION)
        assert not result.is_safe


class TestGridConstraintChecker:
    """Tests for ConstraintChecker grid functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.checker = ConstraintChecker()
        self.battery_state = BatteryState(
            device_id='bat_001',
            device_type='battery',
            soc=0.5,
            temperature=25.0,
        )

    def test_check_with_no_grid_state(self):
        """Test check without grid state skips grid checks."""
        result = self.checker.check(self.battery_state)
        assert not result.has_grid_violation

    def test_check_with_nominal_grid_state(self):
        """Test check with nominal grid state passes."""
        grid_state = GridState()  # Nominal values
        result = self.checker.check(self.battery_state, grid_state=grid_state)
        assert not result.has_grid_violation

    def test_check_grid_only(self):
        """Test check_grid_only method."""
        grid_state = GridState(frequency=47.0)  # Critical low
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.FREQUENCY_CRITICAL in result.violations


class TestGridFrequencyConstraints:
    """Tests for grid frequency constraint checking."""

    def setup_method(self):
        """Set up test fixtures with grid standard defaults."""
        self.checker = ConstraintChecker()

    def test_frequency_normal_passes(self):
        """Test normal frequency passes (49.5-50.2 Hz)."""
        grid_state = GridState(frequency=50.0)
        result = self.checker.check_grid_only(grid_state)
        assert not result.has_grid_violation

        grid_state = GridState(frequency=49.5)
        result = self.checker.check_grid_only(grid_state)
        assert not result.has_grid_violation

        grid_state = GridState(frequency=50.2)
        result = self.checker.check_grid_only(grid_state)
        assert not result.has_grid_violation

    def test_frequency_warning_low(self):
        """Test low warning frequency (48.5-49.5 Hz) triggers warning."""
        grid_state = GridState(frequency=49.0)  # Below 49.5, above 48.5
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.FREQUENCY_OUT_OF_RANGE in result.violations
        # Should be warning, not critical
        assert ConstraintViolation.FREQUENCY_CRITICAL not in result.violations

    def test_frequency_warning_high(self):
        """Test high warning frequency (50.2-50.5 Hz) triggers warning."""
        grid_state = GridState(frequency=50.4)  # Above 50.2, below 50.5
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.FREQUENCY_OUT_OF_RANGE in result.violations

    def test_frequency_critical_low(self):
        """Test critical low frequency (< 47.5 Hz) triggers critical."""
        grid_state = GridState(frequency=47.0)
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.FREQUENCY_CRITICAL in result.violations
        assert not result.is_safe

    def test_frequency_critical_high(self):
        """Test critical high frequency (> 51.5 Hz) triggers critical."""
        grid_state = GridState(frequency=52.0)
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.FREQUENCY_CRITICAL in result.violations
        assert not result.is_safe

    def test_rocof_exceeded(self):
        """Test Rate of Change of Frequency exceeds limit."""
        grid_state = GridState(
            frequency=50.0,
            frequency_rate_of_change=2.5  # > 2.0 Hz/s limit
        )
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.FREQUENCY_CRITICAL in result.violations

    def test_rocof_within_limit(self):
        """Test RoCoF within limit passes."""
        grid_state = GridState(
            frequency=50.0,
            frequency_rate_of_change=1.5  # < 2.0 Hz/s limit
        )
        result = self.checker.check_grid_only(grid_state)
        assert not result.has_grid_violation


class TestGridVoltageConstraints:
    """Tests for grid voltage constraint checking."""

    def setup_method(self):
        """Set up test fixtures with grid standard defaults."""
        self.checker = ConstraintChecker()

    def test_voltage_normal_passes(self):
        """Test normal voltage passes (0.85-1.10 pu)."""
        grid_state = GridState(voltage_pu=1.0)
        result = self.checker.check_grid_only(grid_state)
        assert not result.has_grid_violation

        grid_state = GridState(voltage_pu=0.85)
        result = self.checker.check_grid_only(grid_state)
        assert not result.has_grid_violation

        grid_state = GridState(voltage_pu=1.10)
        result = self.checker.check_grid_only(grid_state)
        assert not result.has_grid_violation

    def test_voltage_warning_low(self):
        """Test low warning voltage (0.70-0.85 pu) triggers warning."""
        grid_state = GridState(voltage_pu=0.80)  # Below 0.85, above 0.70
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.VOLTAGE_OUT_OF_RANGE in result.violations

    def test_voltage_warning_high(self):
        """Test high warning voltage (1.10-1.20 pu) triggers warning."""
        grid_state = GridState(voltage_pu=1.15)  # Above 1.10, below 1.20
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.VOLTAGE_OUT_OF_RANGE in result.violations

    def test_voltage_critical_low(self):
        """Test critical low voltage (< 0.50 pu) triggers critical."""
        grid_state = GridState(voltage_pu=0.40)  # Below 0.50
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.VOLTAGE_CRITICAL in result.violations
        assert not result.is_safe

    def test_voltage_critical_high(self):
        """Test critical high voltage (> 1.30 pu) triggers critical."""
        grid_state = GridState(voltage_pu=1.35)  # Above 1.30
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.VOLTAGE_CRITICAL in result.violations
        assert not result.is_safe


class TestLVRTConstraints:
    """Tests for Low Voltage Ride-Through (LVRT) checking."""

    def setup_method(self):
        """Set up test fixtures with grid standard LVRT curve."""
        self.checker = ConstraintChecker()

    def test_lvrt_20pct_within_time(self):
        """Test LVRT at 20% voltage within 625ms passes."""
        grid_state = GridState(
            voltage_pu=0.20,
            voltage_sag_duration=0.5  # 500ms < 625ms
        )
        result = self.checker.check_grid_only(grid_state)
        # Voltage is critical (< 0.50), but LVRT should not trigger yet
        assert ConstraintViolation.LVRT_VIOLATION not in result.violations

    def test_lvrt_20pct_exceeded(self):
        """Test LVRT at 20% voltage exceeds 625ms triggers violation."""
        grid_state = GridState(
            voltage_pu=0.20,
            voltage_sag_duration=0.7  # 700ms > 625ms
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.LVRT_VIOLATION in result.violations
        assert not result.is_safe

    def test_lvrt_50pct_within_time(self):
        """Test LVRT at 50% voltage within 2s passes."""
        grid_state = GridState(
            voltage_pu=0.50,
            voltage_sag_duration=1.5  # 1.5s < 2s
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.LVRT_VIOLATION not in result.violations

    def test_lvrt_50pct_exceeded(self):
        """Test LVRT at 50% voltage exceeds 2s triggers violation."""
        grid_state = GridState(
            voltage_pu=0.40,  # In 20-50% range
            voltage_sag_duration=2.5  # > 2s
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.LVRT_VIOLATION in result.violations

    def test_lvrt_85pct_within_time(self):
        """Test LVRT at 85% voltage within 10s passes."""
        grid_state = GridState(
            voltage_pu=0.80,  # In 50-85% range
            voltage_sag_duration=8.0  # 8s < 10s
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.LVRT_VIOLATION not in result.violations

    def test_lvrt_85pct_exceeded(self):
        """Test LVRT at 85% voltage exceeds 10s triggers violation."""
        grid_state = GridState(
            voltage_pu=0.70,  # In 50-85% range
            voltage_sag_duration=12.0  # > 10s
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.LVRT_VIOLATION in result.violations

    def test_lvrt_disabled(self):
        """Test LVRT check skipped when disabled."""
        gc = GridConstraints(lvrt_enabled=False)
        checker = ConstraintChecker(grid_constraints=gc)

        grid_state = GridState(
            voltage_pu=0.20,
            voltage_sag_duration=10.0  # Would trigger if enabled
        )
        result = checker.check_grid_only(grid_state)
        assert ConstraintViolation.LVRT_VIOLATION not in result.violations


class TestHVRTConstraints:
    """Tests for High Voltage Ride-Through (HVRT) checking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.checker = ConstraintChecker()

    def test_hvrt_120pct_within_time(self):
        """Test HVRT at 120% voltage within 0.5s passes."""
        grid_state = GridState(
            voltage_pu=1.20,
            voltage_swell_duration=0.3  # 300ms < 500ms
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HVRT_VIOLATION not in result.violations

    def test_hvrt_120pct_exceeded(self):
        """Test HVRT at 120% voltage exceeds 0.5s triggers violation."""
        grid_state = GridState(
            voltage_pu=1.25,  # >= 120%
            voltage_swell_duration=0.7  # > 0.5s
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HVRT_VIOLATION in result.violations
        assert not result.is_safe

    def test_hvrt_110pct_within_time(self):
        """Test HVRT at 110% voltage within 10s passes."""
        grid_state = GridState(
            voltage_pu=1.15,  # 110-120% range
            voltage_swell_duration=8.0  # < 10s
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HVRT_VIOLATION not in result.violations

    def test_hvrt_110pct_exceeded(self):
        """Test HVRT at 110% voltage exceeds 10s triggers violation."""
        grid_state = GridState(
            voltage_pu=1.12,  # In 110-120% range
            voltage_swell_duration=12.0  # > 10s
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HVRT_VIOLATION in result.violations

    def test_hvrt_disabled(self):
        """Test HVRT check skipped when disabled."""
        gc = GridConstraints(hvrt_enabled=False)
        checker = ConstraintChecker(grid_constraints=gc)

        grid_state = GridState(
            voltage_pu=1.25,
            voltage_swell_duration=5.0  # Would trigger if enabled
        )
        result = checker.check_grid_only(grid_state)
        assert ConstraintViolation.HVRT_VIOLATION not in result.violations


class TestAntiIslandingConstraints:
    """Tests for anti-islanding constraint checking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.checker = ConstraintChecker()

    def test_no_islanding_passes(self):
        """Test no islanding condition passes."""
        grid_state = GridState(is_islanded=False)
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.ANTI_ISLANDING_TRIGGERED not in result.violations

    def test_islanding_within_detection_time(self):
        """Test islanding detected within detection time doesn't trigger."""
        grid_state = GridState(
            is_islanded=True,
            island_detection_time=1.5  # < 2.0s limit
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.ANTI_ISLANDING_TRIGGERED not in result.violations

    def test_islanding_exceeded_detection_time(self):
        """Test islanding detected exceeds detection time triggers trip."""
        grid_state = GridState(
            is_islanded=True,
            island_detection_time=2.5  # > 2.0s limit
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.ANTI_ISLANDING_TRIGGERED in result.violations
        assert not result.is_safe

    def test_anti_islanding_disabled(self):
        """Test anti-islanding check skipped when disabled."""
        gc = GridConstraints(anti_islanding_enabled=False)
        checker = ConstraintChecker(grid_constraints=gc)

        grid_state = GridState(
            is_islanded=True,
            island_detection_time=10.0  # Would trigger if enabled
        )
        result = checker.check_grid_only(grid_state)
        assert ConstraintViolation.ANTI_ISLANDING_TRIGGERED not in result.violations


class TestCombinedConstraints:
    """Tests for combined device and grid constraints."""

    def setup_method(self):
        """Set up test fixtures."""
        self.checker = ConstraintChecker()
        self.battery_state = BatteryState(
            device_id='bat_001',
            device_type='battery',
            soc=0.5,
            temperature=25.0,
        )

    def test_both_device_and_grid_violations(self):
        """Test both device and grid violations are detected."""
        self.battery_state.temperature = 50.0  # Over temperature

        grid_state = GridState(frequency=47.0)  # Critical low

        result = self.checker.check(self.battery_state, grid_state=grid_state)

        # Should have both violations
        assert ConstraintViolation.OVER_TEMPERATURE in result.violations
        assert ConstraintViolation.FREQUENCY_CRITICAL in result.violations
        assert not result.is_safe

    def test_device_ok_grid_violation(self):
        """Test device OK but grid violation fails."""
        grid_state = GridState(voltage_pu=1.35)  # Critical high

        result = self.checker.check(self.battery_state, grid_state=grid_state)

        assert ConstraintViolation.VOLTAGE_CRITICAL in result.violations
        assert ConstraintViolation.SOC_TOO_LOW not in result.violations
        assert not result.is_safe

    def test_device_violation_grid_ok(self):
        """Test device violation but grid OK."""
        self.battery_state.soc = 0.05  # Below minimum

        grid_state = GridState()  # Nominal

        result = self.checker.check(self.battery_state, grid_state=grid_state)

        assert ConstraintViolation.SOC_TOO_LOW in result.violations
        assert not result.has_grid_violation
        # SOC_TOO_LOW is not safety-critical
        assert result.is_safe


# ====================
# Harmonic Constraint Tests (harmonic standard)
# ====================


class TestHarmonicConstraints:
    """Tests for HarmonicConstraints dataclass."""

    def test_default_values(self):
        """Test default constraint values (50 Hz standard at 0.38kV)."""
        hc = HarmonicConstraints()

        assert hc.voltage_class_kv == 0.38
        assert hc.use_50hz_standard is True
        assert hc.short_circuit_ratio == 50.0
        assert hc.thd_voltage_limit_038kv == 5.0
        assert hc.tdd_limit_medium_grid == 8.0

    def test_voltage_thd_limit_50hz_038kv(self):
        """Test 50 Hz THD limit at 0.38kV."""
        hc = HarmonicConstraints(voltage_class_kv=0.38)
        assert hc.get_voltage_thd_limit() == 5.0

    def test_voltage_thd_limit_50hz_10kv(self):
        """Test 50 Hz THD limit at 10kV."""
        hc = HarmonicConstraints(voltage_class_kv=10)
        assert hc.get_voltage_thd_limit() == 4.0

    def test_voltage_thd_limit_50hz_35kv(self):
        """Test 50 Hz THD limit at 35kV."""
        hc = HarmonicConstraints(voltage_class_kv=35)
        assert hc.get_voltage_thd_limit() == 3.0

    def test_voltage_thd_limit_50hz_110kv(self):
        """Test 50 Hz THD limit at 110kV."""
        hc = HarmonicConstraints(voltage_class_kv=110)
        assert hc.get_voltage_thd_limit() == 2.0

    def test_voltage_thd_limit_ieee_1kv(self):
        """Test harmonic standard THD limit at ≤1kV."""
        hc = HarmonicConstraints(use_50hz_standard=False, voltage_class_kv=0.4)
        assert hc.get_voltage_thd_limit() == 8.0

    def test_voltage_thd_limit_ieee_69kv(self):
        """Test harmonic standard THD limit at 1-69kV."""
        hc = HarmonicConstraints(use_50hz_standard=False, voltage_class_kv=35)
        assert hc.get_voltage_thd_limit() == 5.0

    def test_voltage_thd_limit_ieee_161kv(self):
        """Test harmonic standard THD limit at 69-161kV."""
        hc = HarmonicConstraints(use_50hz_standard=False, voltage_class_kv=110)
        assert hc.get_voltage_thd_limit() == 2.5

    def test_voltage_thd_limit_ieee_hv(self):
        """Test harmonic standard THD limit at >161kV."""
        hc = HarmonicConstraints(use_50hz_standard=False, voltage_class_kv=220)
        assert hc.get_voltage_thd_limit() == 1.5

    def test_current_tdd_limit_weak_grid(self):
        """Test TDD limit for weak grid (ISC/IL < 20)."""
        hc = HarmonicConstraints(short_circuit_ratio=10)
        assert hc.get_current_tdd_limit() == 5.0

    def test_current_tdd_limit_medium_grid(self):
        """Test TDD limit for medium grid (ISC/IL 20-50)."""
        hc = HarmonicConstraints(short_circuit_ratio=30)
        assert hc.get_current_tdd_limit() == 8.0

    def test_current_tdd_limit_strong_grid(self):
        """Test TDD limit for strong grid (ISC/IL 50-100)."""
        hc = HarmonicConstraints(short_circuit_ratio=80)
        assert hc.get_current_tdd_limit() == 12.0

    def test_current_tdd_limit_very_strong_grid(self):
        """Test TDD limit for very strong grid (ISC/IL 100-1000)."""
        hc = HarmonicConstraints(short_circuit_ratio=500)
        assert hc.get_current_tdd_limit() == 15.0

    def test_current_tdd_limit_extreme_grid(self):
        """Test TDD limit for extremely strong grid (ISC/IL > 1000)."""
        hc = HarmonicConstraints(short_circuit_ratio=2000)
        assert hc.get_current_tdd_limit() == 20.0

    def test_individual_harmonic_limit_by_voltage(self):
        """Test individual harmonic voltage limits."""
        # ≤1kV
        assert HarmonicConstraints(voltage_class_kv=0.4).get_individual_harmonic_limit() == 5.0
        # 1-69kV
        assert HarmonicConstraints(voltage_class_kv=35).get_individual_harmonic_limit() == 3.0
        # 69-161kV
        assert HarmonicConstraints(voltage_class_kv=110).get_individual_harmonic_limit() == 1.5
        # >161kV
        assert HarmonicConstraints(voltage_class_kv=220).get_individual_harmonic_limit() == 1.0


class TestGridStateHarmonics:
    """Tests for GridState harmonic fields."""

    def test_default_harmonic_values(self):
        """Test default harmonic measurements are zero."""
        gs = GridState()
        assert gs.thd_voltage == 0.0
        assert gs.thd_current == 0.0
        assert gs.tdd_current == 0.0
        assert gs.max_individual_voltage_harmonic == 0.0

    def test_harmonic_measurements(self):
        """Test setting harmonic measurements."""
        gs = GridState(
            thd_voltage=3.5,
            thd_current=6.2,
            tdd_current=7.8,
            max_individual_voltage_harmonic=2.1,
            max_harmonic_order=5
        )
        assert gs.thd_voltage == 3.5
        assert gs.thd_current == 6.2
        assert gs.tdd_current == 7.8
        assert gs.max_individual_voltage_harmonic == 2.1
        assert gs.max_harmonic_order == 5

    def test_individual_harmonics_dict(self):
        """Test individual harmonics dictionary."""
        gs = GridState(
            individual_voltage_harmonics={3: 2.5, 5: 4.0, 7: 1.5},
            individual_current_harmonics={5: 3.0, 7: 2.0}
        )
        assert gs.individual_voltage_harmonics[5] == 4.0
        assert len(gs.individual_current_harmonics) == 2


class TestHarmonicConstraintChecker:
    """Tests for harmonic constraint checking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.checker = ConstraintChecker()

    def test_voltage_thd_normal_passes(self):
        """Test normal voltage THD passes."""
        grid_state = GridState(thd_voltage=3.0)  # Below 5% limit
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED not in result.violations

    def test_voltage_thd_exceeds_limit(self):
        """Test voltage THD exceeding limit triggers violation."""
        grid_state = GridState(thd_voltage=6.0)  # Above 5% limit at 0.38kV
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations
        assert result.has_grid_violation
        assert "Voltage THD" in result.messages[0]

    def test_current_tdd_normal_passes(self):
        """Test normal current TDD passes."""
        grid_state = GridState(tdd_current=5.0)  # Below limit
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED not in result.violations

    def test_current_tdd_exceeds_limit(self):
        """Test current TDD exceeding limit triggers violation."""
        # Default ISC/IL=50 gives TDD limit of 12%
        grid_state = GridState(tdd_current=15.0)
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations
        assert "Current TDD" in result.messages[0]

    def test_individual_harmonic_exceeds_limit(self):
        """Test individual harmonic exceeding limit."""
        grid_state = GridState(
            max_individual_voltage_harmonic=6.0,  # Above 5% limit
            max_harmonic_order=5
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations
        assert "Individual voltage harmonic" in result.messages[0]

    def test_multiple_harmonic_violations(self):
        """Test multiple harmonic violations detected."""
        grid_state = GridState(
            thd_voltage=8.0,
            tdd_current=18.0,
            max_individual_voltage_harmonic=7.0,
            max_harmonic_order=3
        )
        result = self.checker.check_grid_only(grid_state)
        # Should have 3 violations (THD, TDD, individual)
        harmonic_violations = [v for v in result.violations
                               if v == ConstraintViolation.HARMONIC_LIMIT_EXCEEDED]
        assert len(harmonic_violations) == 3

    def test_different_voltage_class_limits(self):
        """Test different voltage class has different limits."""
        # At 10kV, limit is 4%
        hc_10kv = HarmonicConstraints(voltage_class_kv=10)
        checker_10kv = ConstraintChecker(harmonic_constraints=hc_10kv)

        grid_state = GridState(thd_voltage=4.5)  # 4.5% > 4% limit

        result = checker_10kv.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations

    def test_ieee_vs_50hz_standard(self):
        """Test alternative standard has different limits than 50 Hz."""
        # Alternative at ≤1kV allows 8% THD (vs 50 Hz 5%)
        hc_ieee = HarmonicConstraints(use_50hz_standard=False)
        checker_ieee = ConstraintChecker(harmonic_constraints=hc_ieee)

        grid_state = GridState(thd_voltage=6.0)  # 6% < 8% standard limit

        result = checker_ieee.check_grid_only(grid_state)
        # Should not have THD violation with alternative standard
        thd_violations = [m for m in result.messages if "Voltage THD" in m]
        assert len(thd_violations) == 0

    def test_weak_grid_tdd_limit(self):
        """Test weak grid has lower TDD limit."""
        hc_weak = HarmonicConstraints(short_circuit_ratio=10)  # < 20
        checker_weak = ConstraintChecker(harmonic_constraints=hc_weak)

        grid_state = GridState(tdd_current=6.0)  # 6% > 5% limit

        result = checker_weak.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations


class TestIndividualHarmonicsCheck:
    """Tests for detailed individual harmonics checking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.checker = ConstraintChecker()

    def test_individual_voltage_harmonic_violation(self):
        """Test individual voltage harmonic exceeding limit."""
        grid_state = GridState(
            individual_voltage_harmonics={
                3: 2.0,  # OK
                5: 6.5,  # Exceeds 5% limit
                7: 1.5   # OK
            }
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations
        assert "Voltage harmonic h5" in str(result.messages)

    def test_individual_current_harmonic_violation(self):
        """Test individual current harmonic exceeding limit."""
        # Default ISC/IL=50 (medium-strong grid)
        # h2-11 limit: 10%
        grid_state = GridState(
            individual_current_harmonics={
                5: 5.0,   # OK
                7: 12.0,  # Exceeds 10% limit for h2-11 at strong grid
                11: 3.0   # OK
            }
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations
        assert "Current harmonic h7" in str(result.messages)

    def test_high_order_current_harmonic_stricter_limit(self):
        """Test higher order current harmonics have stricter limits."""
        # For strong grid (ISC/IL 50-100), h35-50 limit is 0.7%
        hc = HarmonicConstraints(short_circuit_ratio=80)
        checker = ConstraintChecker(harmonic_constraints=hc)

        grid_state = GridState(
            individual_current_harmonics={
                37: 1.0,  # Exceeds 0.7% limit for h35-50
            }
        )
        result = checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations

    def test_all_harmonics_within_limits(self):
        """Test all harmonics within limits passes."""
        grid_state = GridState(
            thd_voltage=3.0,
            tdd_current=5.0,
            individual_voltage_harmonics={3: 2.0, 5: 3.0, 7: 1.5},
            individual_current_harmonics={3: 3.0, 5: 4.0, 7: 2.0}
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED not in result.violations


class TestHarmonicWithOtherGridConstraints:
    """Tests for harmonics combined with other grid constraints."""

    def test_harmonic_and_frequency_violation(self):
        """Test both harmonic and frequency violations detected."""
        checker = ConstraintChecker()
        grid_state = GridState(
            frequency=47.0,      # Critical low frequency
            thd_voltage=7.0      # Exceeds THD limit
        )
        result = checker.check_grid_only(grid_state)

        assert ConstraintViolation.FREQUENCY_CRITICAL in result.violations
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations

    def test_harmonic_and_voltage_violation(self):
        """Test both harmonic and voltage violations detected."""
        checker = ConstraintChecker()
        grid_state = GridState(
            voltage_pu=0.45,     # Critical low voltage
            tdd_current=15.0     # Exceeds TDD limit
        )
        result = checker.check_grid_only(grid_state)

        assert ConstraintViolation.VOLTAGE_CRITICAL in result.violations
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations

    def test_normal_harmonics_with_other_violations(self):
        """Test normal harmonics don't mask other violations."""
        checker = ConstraintChecker()
        grid_state = GridState(
            frequency=47.0,      # Critical
            voltage_pu=1.35,     # Critical high
            thd_voltage=2.0,     # Normal
            tdd_current=3.0      # Normal
        )
        result = checker.check_grid_only(grid_state)

        assert ConstraintViolation.FREQUENCY_CRITICAL in result.violations
        assert ConstraintViolation.VOLTAGE_CRITICAL in result.violations
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED not in result.violations
