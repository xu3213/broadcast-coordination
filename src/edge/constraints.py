"""
Physical Constraints

Battery operational constraint checking:
1. Power limits (SOC bounds, C-rate capacity)
2. Ramp rate limits
3. Grid interconnection compliance
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum, auto

from .state_machine import DeviceState, BatteryState


class ConstraintViolation(Enum):
    """Types of constraint violations."""
    # Power constraints
    MAX_POWER_EXCEEDED = auto()
    RAMP_RATE_EXCEEDED = auto()

    # SOC constraints
    SOC_TOO_LOW = auto()
    SOC_TOO_HIGH = auto()
    DEPARTURE_SOC_RISK = auto()

    # Thermal constraints
    OVER_TEMPERATURE = auto()
    UNDER_TEMPERATURE = auto()

    # Operational constraints
    MIN_RUN_TIME = auto()
    MIN_OFF_TIME = auto()
    DAILY_CYCLES_EXCEEDED = auto()

    # Safety constraints
    DEVICE_FAULTED = auto()
    DEVICE_OFFLINE = auto()

    # Grid interconnection constraints
    FREQUENCY_OUT_OF_RANGE = auto()      # Grid frequency outside normal operating range
    FREQUENCY_CRITICAL = auto()           # Grid frequency in critical/trip zone
    VOLTAGE_OUT_OF_RANGE = auto()         # Grid voltage outside continuous operating range
    VOLTAGE_CRITICAL = auto()             # Grid voltage in trip zone
    LVRT_VIOLATION = auto()               # Low Voltage Ride-Through requirement violated
    HVRT_VIOLATION = auto()               # High Voltage Ride-Through requirement violated
    ANTI_ISLANDING_TRIGGERED = auto()     # Anti-islanding protection activated
    HARMONIC_LIMIT_EXCEEDED = auto()      # THD or individual harmonic exceeded


@dataclass
class ConstraintResult:
    """Result of constraint checking."""
    violated: bool = False
    violations: List[ConstraintViolation] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)

    def add_violation(self, violation: ConstraintViolation, message: str = "") -> None:
        """Add a constraint violation."""
        self.violated = True
        self.violations.append(violation)
        if message:
            self.messages.append(message)

    @property
    def is_safe(self) -> bool:
        """True if no safety-critical violations."""
        safety_critical = {
            ConstraintViolation.OVER_TEMPERATURE,
            ConstraintViolation.DEVICE_FAULTED,
            ConstraintViolation.DEVICE_OFFLINE,
            # Grid safety-critical violations
            ConstraintViolation.FREQUENCY_CRITICAL,
            ConstraintViolation.VOLTAGE_CRITICAL,
            ConstraintViolation.LVRT_VIOLATION,
            ConstraintViolation.HVRT_VIOLATION,
            ConstraintViolation.ANTI_ISLANDING_TRIGGERED,
        }
        return not any(v in safety_critical for v in self.violations)

    @property
    def has_grid_violation(self) -> bool:
        """True if any grid interconnection constraints are violated."""
        grid_violations = {
            ConstraintViolation.FREQUENCY_OUT_OF_RANGE,
            ConstraintViolation.FREQUENCY_CRITICAL,
            ConstraintViolation.VOLTAGE_OUT_OF_RANGE,
            ConstraintViolation.VOLTAGE_CRITICAL,
            ConstraintViolation.LVRT_VIOLATION,
            ConstraintViolation.HVRT_VIOLATION,
            ConstraintViolation.ANTI_ISLANDING_TRIGGERED,
            ConstraintViolation.HARMONIC_LIMIT_EXCEEDED,
        }
        return any(v in grid_violations for v in self.violations)


class GridStandard(Enum):
    """Supported grid interconnection standards."""
    STANDARD_A = auto()
    STANDARD_B = auto()      # 50 Hz grid standard
    CUSTOM = auto()               # Custom parameters


@dataclass
class GridConstraints:
    """
    Grid interconnection constraint parameters.

    Default values target 50 Hz grid deployment.
    """
    standard: GridStandard = GridStandard.STANDARD_B

    # === Frequency Constraints ===
    # Nominal frequency (Hz)
    nominal_frequency: float = 50.0

    # Normal operating range (continuous operation allowed)
    freq_normal_min: float = 49.5       # 49.5 Hz (50 Hz system)
    freq_normal_max: float = 50.2       # 50.2 Hz

    # Warning range (short-duration operation, reduced performance)
    freq_warning_min: float = 48.5      # Warning low
    freq_warning_max: float = 50.5      # Warning high

    # Critical range (must trip/disconnect)
    freq_critical_min: float = 47.5     # Critical low (trip)
    freq_critical_max: float = 51.5     # Critical high (trip)

    # Frequency ride-through time limits (seconds)
    freq_warning_max_duration: float = 300.0   # 300s at warning range
    freq_critical_max_duration: float = 0.16   # 0.16s at critical range

    # === Voltage Constraints ===
    # Nominal voltage (per-unit, 1.0 pu = nominal)
    nominal_voltage_pu: float = 1.0

    # Normal operating range (continuous operation allowed)
    voltage_normal_min_pu: float = 0.85   # 85% nominal
    voltage_normal_max_pu: float = 1.10   # 110% nominal

    # Warning range (limited duration operation)
    voltage_warning_min_pu: float = 0.70   # Warning low
    voltage_warning_max_pu: float = 1.20   # Warning high

    # Critical range (momentary cessation / trip)
    voltage_critical_min_pu: float = 0.50   # Critical low
    voltage_critical_max_pu: float = 1.30   # Critical high

    # === Low Voltage Ride-Through (LVRT) Requirements ===
    lvrt_enabled: bool = True
    lvrt_threshold_20pct_duration: float = 0.625   # Must withstand 20% voltage for 625ms
    lvrt_threshold_50pct_duration: float = 2.0     # Must withstand 50% voltage for 2s
    lvrt_threshold_85pct_duration: float = 10.0    # Must withstand 85% voltage for 10s

    # === High Voltage Ride-Through (HVRT) Requirements ===
    hvrt_enabled: bool = True
    hvrt_threshold_120pct_duration: float = 0.5    # Must withstand 120% voltage for 0.5s
    hvrt_threshold_110pct_duration: float = 10.0   # Must withstand 110% voltage for 10s

    # === Anti-Islanding Protection ===
    anti_islanding_enabled: bool = True
    anti_islanding_detection_time: float = 2.0     # Max 2 seconds to detect and trip

    # === Rate of Change of Frequency (RoCoF) ===
    rocof_limit: float = 2.0    # Hz/s, maximum rate of change

    @classmethod
    def standard_a(cls, category: int = 3) -> "GridConstraints":
        """
        Create constraints for US 60 Hz grid.

        Args:
            category: Performance category (1, 2, or 3)

        Returns:
            GridConstraints configured for 60 Hz system
        """
        constraints = cls(
            standard=GridStandard.STANDARD_A,
            nominal_frequency=60.0,
            # Frequency bounds (60 Hz system)
            freq_normal_min=59.5,
            freq_normal_max=60.1,
            freq_warning_min=58.5 if category >= 2 else 59.0,
            freq_warning_max=61.2 if category >= 2 else 60.5,
            freq_critical_min=56.5,
            freq_critical_max=62.0,
            # Voltage bounds
            voltage_normal_min_pu=0.88,
            voltage_normal_max_pu=1.10,
            voltage_warning_min_pu=0.70,
            voltage_warning_max_pu=1.20,
            voltage_critical_min_pu=0.50,
            voltage_critical_max_pu=1.20,
            # LVRT per category
            lvrt_threshold_20pct_duration=0.0 if category == 1 else 0.16,
            lvrt_threshold_50pct_duration=0.5 if category == 2 else (2.0 if category == 3 else 0.0),
            # RoCoF (more stringent for higher categories)
            rocof_limit=3.0 if category == 1 else 2.0,
        )
        return constraints

    @classmethod
    def standard_b(cls) -> "GridConstraints":
        """
        Create constraints for 50 Hz grid.

        Returns:
            GridConstraints configured for 50 Hz grid requirements
        """
        return cls(
            standard=GridStandard.STANDARD_B,
            nominal_frequency=50.0,
            freq_normal_min=49.5,
            freq_normal_max=50.2,
            freq_warning_min=48.5,
            freq_warning_max=50.5,
            freq_critical_min=47.5,
            freq_critical_max=51.5,
            voltage_normal_min_pu=0.85,
            voltage_normal_max_pu=1.10,
            voltage_warning_min_pu=0.70,
            voltage_warning_max_pu=1.20,
            voltage_critical_min_pu=0.20,   # LVRT requirement
            voltage_critical_max_pu=1.30,
            lvrt_threshold_20pct_duration=0.625,
            lvrt_threshold_50pct_duration=2.0,
            lvrt_threshold_85pct_duration=10.0,
        )


@dataclass
class HarmonicConstraints:
    """
    Harmonic distortion limits.

    Voltage THD and current TDD limits vary by voltage level and
    short-circuit ratio.

    Default values target 0.38 kV residential voltage class.
    """
    # Voltage level for selecting appropriate limits
    voltage_class_kv: float = 0.38   # Nominal voltage (kV)

    # === Voltage THD Limits (%) ===
    # 50 Hz standard limits by voltage class
    thd_voltage_limit_038kv: float = 5.0    # 0.38 kV: 5%
    thd_voltage_limit_10kv: float = 4.0     # 6-10 kV: 4%
    thd_voltage_limit_66kv: float = 3.0     # 35-66 kV: 3%
    thd_voltage_limit_110kv: float = 2.0    # 110 kV: 2%

    # 60 Hz standard limits (alternative)
    ieee_thd_voltage_limit_1kv: float = 8.0     # ≤1 kV: 8%
    ieee_thd_voltage_limit_69kv: float = 5.0    # 1-69 kV: 5%
    ieee_thd_voltage_limit_161kv: float = 2.5   # 69-161 kV: 2.5%
    ieee_thd_voltage_limit_hv: float = 1.5      # >161 kV: 1.5%

    # Individual harmonic voltage limits (% of fundamental)
    individual_harmonic_limit_1kv: float = 5.0
    individual_harmonic_limit_69kv: float = 3.0
    individual_harmonic_limit_161kv: float = 1.5
    individual_harmonic_limit_hv: float = 1.0

    # === Current TDD Limits (%) ===
    # Limits depend on short-circuit ratio ISC/IL

    # ISC/IL < 20 (weak grid)
    tdd_limit_weak_grid: float = 5.0
    h2_11_limit_weak: float = 4.0
    h11_17_limit_weak: float = 2.0
    h17_23_limit_weak: float = 1.5
    h23_35_limit_weak: float = 0.6
    h35_50_limit_weak: float = 0.3

    # ISC/IL 20-50 (medium grid)
    tdd_limit_medium_grid: float = 8.0
    h2_11_limit_medium: float = 7.0
    h11_17_limit_medium: float = 3.5
    h17_23_limit_medium: float = 2.5
    h23_35_limit_medium: float = 1.0
    h35_50_limit_medium: float = 0.5

    # ISC/IL 50-100 (strong grid)
    tdd_limit_strong_grid: float = 12.0
    h2_11_limit_strong: float = 10.0
    h11_17_limit_strong: float = 4.5
    h17_23_limit_strong: float = 4.0
    h23_35_limit_strong: float = 1.5
    h35_50_limit_strong: float = 0.7

    # ISC/IL 100-1000 (very strong grid)
    tdd_limit_very_strong_grid: float = 15.0

    # ISC/IL > 1000 (extremely strong grid)
    tdd_limit_extreme_grid: float = 20.0

    # Short-circuit ratio at PCC (for current limits selection)
    short_circuit_ratio: float = 50.0   # Default: medium grid (ISC/IL = 50)

    # Use 50 Hz standard (True) or 60 Hz standard (False)
    use_50hz_standard: bool = True

    def get_voltage_thd_limit(self) -> float:
        """
        Get voltage THD limit based on voltage class and selected standard.

        Returns:
            Maximum allowed voltage THD (%)
        """
        if self.use_50hz_standard:
            # 50 Hz standard
            if self.voltage_class_kv <= 0.5:
                return self.thd_voltage_limit_038kv
            elif self.voltage_class_kv <= 10:
                return self.thd_voltage_limit_10kv
            elif self.voltage_class_kv <= 66:
                return self.thd_voltage_limit_66kv
            else:
                return self.thd_voltage_limit_110kv
        else:
            # US standard
            if self.voltage_class_kv <= 1:
                return self.ieee_thd_voltage_limit_1kv
            elif self.voltage_class_kv <= 69:
                return self.ieee_thd_voltage_limit_69kv
            elif self.voltage_class_kv <= 161:
                return self.ieee_thd_voltage_limit_161kv
            else:
                return self.ieee_thd_voltage_limit_hv

    def get_current_tdd_limit(self) -> float:
        """
        Get current TDD limit based on short-circuit ratio.

        Returns:
            Maximum allowed current TDD (%)
        """
        isc_il = self.short_circuit_ratio
        if isc_il < 20:
            return self.tdd_limit_weak_grid
        elif isc_il < 50:
            return self.tdd_limit_medium_grid
        elif isc_il < 100:
            return self.tdd_limit_strong_grid
        elif isc_il < 1000:
            return self.tdd_limit_very_strong_grid
        else:
            return self.tdd_limit_extreme_grid

    def get_individual_harmonic_limit(self) -> float:
        """
        Get individual harmonic voltage limit based on voltage class.

        Returns:
            Maximum allowed individual harmonic voltage (%)
        """
        if self.voltage_class_kv <= 1:
            return self.individual_harmonic_limit_1kv
        elif self.voltage_class_kv <= 69:
            return self.individual_harmonic_limit_69kv
        elif self.voltage_class_kv <= 161:
            return self.individual_harmonic_limit_161kv
        else:
            return self.individual_harmonic_limit_hv


@dataclass
class GridState:
    """
    Real-time grid state measurements for constraint checking.

    These values come from grid sensors or power quality analyzers.
    """
    # Frequency measurements
    frequency: float = 50.0              # Current grid frequency (Hz)
    frequency_rate_of_change: float = 0.0  # df/dt (Hz/s)

    # Voltage measurements (per-unit)
    voltage_pu: float = 1.0              # Current voltage as fraction of nominal
    voltage_phase_a_pu: float = 1.0      # Phase A voltage
    voltage_phase_b_pu: float = 1.0      # Phase B voltage
    voltage_phase_c_pu: float = 1.0      # Phase C voltage

    # Voltage event tracking
    voltage_sag_duration: float = 0.0    # Current low voltage event duration (s)
    voltage_swell_duration: float = 0.0  # Current high voltage event duration (s)

    # Anti-islanding detection
    is_islanded: bool = False            # True if islanding detected
    island_detection_time: float = 0.0   # Time since islanding detected

    # === Harmonic measurements ===
    thd_voltage: float = 0.0             # Total Harmonic Distortion - voltage (%)
    thd_current: float = 0.0             # Total Harmonic Distortion - current (%)
    tdd_current: float = 0.0             # Total Demand Distortion - current (%)

    # Individual harmonics (optional, for detailed analysis)
    # Key: harmonic order (2-50), Value: magnitude (% of fundamental)
    individual_voltage_harmonics: dict = field(default_factory=dict)
    individual_current_harmonics: dict = field(default_factory=dict)

    # Maximum individual harmonic values (for quick limit checking)
    max_individual_voltage_harmonic: float = 0.0
    max_individual_current_harmonic: float = 0.0
    max_harmonic_order: int = 0          # Order of the highest individual harmonic

    # Timestamp
    timestamp: float = 0.0               # Measurement timestamp


@dataclass
class PhysicalConstraints:
    """
    Physical constraint parameters for a device.

    Default values are conservative; should be tuned per device.
    """
    # Power constraints
    max_charge_power: float = 10.0  # kW
    max_discharge_power: float = 10.0  # kW
    max_ramp_rate: float = 5.0  # kW/s

    # SOC constraints
    min_soc: float = 0.1  # Don't discharge below 10%
    max_soc: float = 0.95  # Don't charge above 95%
    safe_soc_range: tuple = (0.2, 0.8)  # Preferred operating range

    # Thermal constraints
    max_temperature: float = 45.0  # Celsius
    min_temperature: float = 0.0  # Celsius

    # Operational constraints
    min_run_time: float = 60.0  # Minimum seconds per operation
    min_off_time: float = 60.0  # Minimum seconds between operations
    max_daily_cycles: int = 10  # Maximum charge/discharge cycles per day

class ConstraintChecker:
    """
    Checks device states against physical and grid interconnection constraints.

    Supports battery constraint logic with grid interconnection
    and harmonic distortion requirements.
    """

    def __init__(
        self,
        constraints: Optional[PhysicalConstraints] = None,
        grid_constraints: Optional[GridConstraints] = None,
        harmonic_constraints: Optional[HarmonicConstraints] = None
    ):
        """
        Initialize checker with constraints.

        Args:
            constraints: Physical constraint parameters (uses defaults if None)
            grid_constraints: Grid interconnection constraints (uses 50 Hz defaults if None)
            harmonic_constraints: Harmonic distortion constraints (uses 50 Hz defaults if None)
        """
        self.constraints = constraints or PhysicalConstraints()
        self.grid_constraints = grid_constraints or GridConstraints()
        self.harmonic_constraints = harmonic_constraints or HarmonicConstraints()

    def check(
        self,
        state: DeviceState,
        target_power: Optional[float] = None,
        grid_state: Optional[GridState] = None
    ) -> ConstraintResult:
        """
        Check device state against all relevant constraints.

        Args:
            state: Current device state
            target_power: Proposed power change (kW), if any
            grid_state: Current grid measurements (if None, grid checks are skipped)

        Returns:
            ConstraintResult with any violations found
        """
        result = ConstraintResult()

        # Basic health check
        if not state.is_healthy:
            result.add_violation(
                ConstraintViolation.DEVICE_FAULTED,
                f"Device faulted: {state.fault_code}"
            )

        # Device-specific checks
        if isinstance(state, BatteryState):
            self._check_battery(state, target_power, result)

        # Generic power constraints
        if target_power is not None:
            self._check_power_limits(state, target_power, result)

        # Grid interconnection constraints
        if grid_state is not None:
            self._check_grid_requirements(grid_state, result)

        return result

    def check_grid_only(self, grid_state: GridState) -> ConstraintResult:
        """
        Check only grid interconnection constraints.

        Useful for fast grid state validation without device state.

        Args:
            grid_state: Current grid measurements

        Returns:
            ConstraintResult with any grid-related violations
        """
        result = ConstraintResult()
        self._check_grid_requirements(grid_state, result)
        return result

    def _check_grid_requirements(
        self,
        grid_state: GridState,
        result: ConstraintResult
    ) -> None:
        """
        Check grid interconnection constraints.

        Implements:
        - Frequency range checking (normal, warning, critical zones)
        - Voltage range checking (continuous, limited duration, trip zones)
        - Low Voltage Ride-Through (LVRT) requirements
        - High Voltage Ride-Through (HVRT) requirements
        - Rate of Change of Frequency (RoCoF) limits
        - Anti-islanding protection

        Args:
            grid_state: Current grid measurements
            result: ConstraintResult to add violations to
        """
        gc = self.grid_constraints

        # === Frequency Constraint Checking ===
        freq = grid_state.frequency

        # Critical frequency range (must trip immediately)
        if freq < gc.freq_critical_min:
            result.add_violation(
                ConstraintViolation.FREQUENCY_CRITICAL,
                f"Grid frequency {freq:.2f}Hz critically low (< {gc.freq_critical_min:.1f}Hz), "
                f"must disconnect per {gc.standard.name}"
            )
        elif freq > gc.freq_critical_max:
            result.add_violation(
                ConstraintViolation.FREQUENCY_CRITICAL,
                f"Grid frequency {freq:.2f}Hz critically high (> {gc.freq_critical_max:.1f}Hz), "
                f"must disconnect per {gc.standard.name}"
            )
        # Warning frequency range (limited duration operation)
        elif freq < gc.freq_warning_min:
            result.add_violation(
                ConstraintViolation.FREQUENCY_OUT_OF_RANGE,
                f"Grid frequency {freq:.2f}Hz below normal ({gc.freq_warning_min:.1f}Hz), "
                f"limited duration operation (max {gc.freq_warning_max_duration:.0f}s)"
            )
        elif freq > gc.freq_warning_max:
            result.add_violation(
                ConstraintViolation.FREQUENCY_OUT_OF_RANGE,
                f"Grid frequency {freq:.2f}Hz above normal ({gc.freq_warning_max:.1f}Hz), "
                f"limited duration operation (max {gc.freq_warning_max_duration:.0f}s)"
            )
        # Outside normal operating range but within warning range
        elif freq < gc.freq_normal_min or freq > gc.freq_normal_max:
            result.add_violation(
                ConstraintViolation.FREQUENCY_OUT_OF_RANGE,
                f"Grid frequency {freq:.2f}Hz outside normal range "
                f"({gc.freq_normal_min:.1f}-{gc.freq_normal_max:.1f}Hz)"
            )

        # Rate of Change of Frequency (RoCoF) check
        rocof = abs(grid_state.frequency_rate_of_change)
        if rocof > gc.rocof_limit:
            result.add_violation(
                ConstraintViolation.FREQUENCY_CRITICAL,
                f"RoCoF {rocof:.2f}Hz/s exceeds limit {gc.rocof_limit:.1f}Hz/s, "
                f"possible islanding or major grid event"
            )

        # === Voltage Constraint Checking ===
        voltage = grid_state.voltage_pu

        # Critical voltage range (must enter momentary cessation or trip)
        if voltage < gc.voltage_critical_min_pu:
            result.add_violation(
                ConstraintViolation.VOLTAGE_CRITICAL,
                f"Grid voltage {voltage:.2f}pu critically low (< {gc.voltage_critical_min_pu:.2f}pu), "
                f"momentary cessation required"
            )
        elif voltage > gc.voltage_critical_max_pu:
            result.add_violation(
                ConstraintViolation.VOLTAGE_CRITICAL,
                f"Grid voltage {voltage:.2f}pu critically high (> {gc.voltage_critical_max_pu:.2f}pu), "
                f"must trip"
            )
        # Warning voltage range (limited duration operation)
        elif voltage < gc.voltage_warning_min_pu:
            result.add_violation(
                ConstraintViolation.VOLTAGE_OUT_OF_RANGE,
                f"Grid voltage {voltage:.2f}pu below warning threshold ({gc.voltage_warning_min_pu:.2f}pu)"
            )
        elif voltage > gc.voltage_warning_max_pu:
            result.add_violation(
                ConstraintViolation.VOLTAGE_OUT_OF_RANGE,
                f"Grid voltage {voltage:.2f}pu above warning threshold ({gc.voltage_warning_max_pu:.2f}pu)"
            )
        # Outside normal operating range
        elif voltage < gc.voltage_normal_min_pu or voltage > gc.voltage_normal_max_pu:
            result.add_violation(
                ConstraintViolation.VOLTAGE_OUT_OF_RANGE,
                f"Grid voltage {voltage:.2f}pu outside normal range "
                f"({gc.voltage_normal_min_pu:.2f}-{gc.voltage_normal_max_pu:.2f}pu)"
            )

        # === Low Voltage Ride-Through (LVRT) Check ===
        if gc.lvrt_enabled and grid_state.voltage_sag_duration > 0:
            self._check_lvrt(grid_state, result)

        # === High Voltage Ride-Through (HVRT) Check ===
        if gc.hvrt_enabled and grid_state.voltage_swell_duration > 0:
            self._check_hvrt(grid_state, result)

        # === Anti-Islanding Check ===
        if gc.anti_islanding_enabled and grid_state.is_islanded:
            if grid_state.island_detection_time >= gc.anti_islanding_detection_time:
                result.add_violation(
                    ConstraintViolation.ANTI_ISLANDING_TRIGGERED,
                    f"Islanding detected for {grid_state.island_detection_time:.2f}s, "
                    f"exceeds detection limit {gc.anti_islanding_detection_time:.1f}s, must trip"
                )

        # === Harmonic Distortion Check ===
        self._check_harmonics(grid_state, result)

    def _check_harmonics(self, grid_state: GridState, result: ConstraintResult) -> None:
        """
        Check harmonic distortion limits.

        Implements:
        - Total Harmonic Distortion (THD) voltage limits
        - Total Demand Distortion (TDD) current limits
        - Individual harmonic limits (if detailed measurements available)

        Args:
            grid_state: Current grid measurements including harmonic data
            result: ConstraintResult to add violations to
        """
        hc = self.harmonic_constraints

        # === Voltage THD Check ===
        thd_voltage_limit = hc.get_voltage_thd_limit()
        if grid_state.thd_voltage > thd_voltage_limit:
            standard_name = "50 Hz standard" if hc.use_50hz_standard else "60 Hz standard"
            result.add_violation(
                ConstraintViolation.HARMONIC_LIMIT_EXCEEDED,
                f"Voltage THD {grid_state.thd_voltage:.2f}% exceeds limit {thd_voltage_limit:.1f}% "
                f"({standard_name} at {hc.voltage_class_kv}kV)"
            )

        # === Current TDD Check ===
        tdd_current_limit = hc.get_current_tdd_limit()
        if grid_state.tdd_current > tdd_current_limit:
            result.add_violation(
                ConstraintViolation.HARMONIC_LIMIT_EXCEEDED,
                f"Current TDD {grid_state.tdd_current:.2f}% exceeds limit {tdd_current_limit:.1f}% "
                f"(ISC/IL={hc.short_circuit_ratio:.0f})"
            )

        # === Individual Harmonic Voltage Check ===
        individual_limit = hc.get_individual_harmonic_limit()
        if grid_state.max_individual_voltage_harmonic > individual_limit:
            result.add_violation(
                ConstraintViolation.HARMONIC_LIMIT_EXCEEDED,
                f"Individual voltage harmonic (h{grid_state.max_harmonic_order}) "
                f"{grid_state.max_individual_voltage_harmonic:.2f}% exceeds limit {individual_limit:.1f}%"
            )

        # === Detailed Individual Harmonics Check (optional) ===
        if grid_state.individual_voltage_harmonics:
            self._check_individual_harmonics(
                grid_state.individual_voltage_harmonics,
                "voltage",
                result
            )

        if grid_state.individual_current_harmonics:
            self._check_individual_harmonics(
                grid_state.individual_current_harmonics,
                "current",
                result
            )

    def _check_individual_harmonics(
        self,
        harmonics: dict,
        harmonic_type: str,
        result: ConstraintResult
    ) -> None:
        """
        Check individual harmonic values against limits.

        Limits vary by harmonic group:
        - h2-11: Base limit
        - h11-17: Reduced limit
        - h17-23: Further reduced
        - h23-35: Even lower
        - h35-50: Lowest limit

        Args:
            harmonics: Dict mapping harmonic order to magnitude (%)
            harmonic_type: "voltage" or "current"
            result: ConstraintResult to add violations to
        """
        hc = self.harmonic_constraints

        if harmonic_type == "voltage":
            base_limit = hc.get_individual_harmonic_limit()
            # Same limit for all individual voltage harmonics
            for order, magnitude in harmonics.items():
                if isinstance(order, int) and 2 <= order <= 50:
                    if magnitude > base_limit:
                        result.add_violation(
                            ConstraintViolation.HARMONIC_LIMIT_EXCEEDED,
                            f"Voltage harmonic h{order} = {magnitude:.2f}% exceeds limit {base_limit:.1f}%"
                        )
        else:
            # Current harmonics: limits vary by group and ISC/IL ratio
            isc_il = hc.short_circuit_ratio

            # Get limits based on ISC/IL ratio
            if isc_il < 20:
                limits = {
                    (2, 11): hc.h2_11_limit_weak,
                    (11, 17): hc.h11_17_limit_weak,
                    (17, 23): hc.h17_23_limit_weak,
                    (23, 35): hc.h23_35_limit_weak,
                    (35, 50): hc.h35_50_limit_weak,
                }
            elif isc_il < 50:
                limits = {
                    (2, 11): hc.h2_11_limit_medium,
                    (11, 17): hc.h11_17_limit_medium,
                    (17, 23): hc.h17_23_limit_medium,
                    (23, 35): hc.h23_35_limit_medium,
                    (35, 50): hc.h35_50_limit_medium,
                }
            else:
                limits = {
                    (2, 11): hc.h2_11_limit_strong,
                    (11, 17): hc.h11_17_limit_strong,
                    (17, 23): hc.h17_23_limit_strong,
                    (23, 35): hc.h23_35_limit_strong,
                    (35, 50): hc.h35_50_limit_strong,
                }

            for order, magnitude in harmonics.items():
                if isinstance(order, int) and 2 <= order <= 50:
                    # Find applicable limit based on harmonic group
                    limit = None
                    for (low, high), group_limit in limits.items():
                        if low <= order < high:
                            limit = group_limit
                            break

                    if limit is not None and magnitude > limit:
                        result.add_violation(
                            ConstraintViolation.HARMONIC_LIMIT_EXCEEDED,
                            f"Current harmonic h{order} = {magnitude:.2f}% exceeds limit {limit:.1f}%"
                        )

    def _check_lvrt(self, grid_state: GridState, result: ConstraintResult) -> None:
        """
        Check Low Voltage Ride-Through requirements.

        LVRT curve defines voltage thresholds and required ride-through times:
        - 20% voltage: must ride through for 625ms
        - 50% voltage: must ride through for 2s
        - 85% voltage: must ride through for 10s

        Args:
            grid_state: Current grid measurements
            result: ConstraintResult to add violations to
        """
        gc = self.grid_constraints
        voltage = grid_state.voltage_pu
        duration = grid_state.voltage_sag_duration

        # Determine which LVRT threshold applies based on voltage level
        if voltage <= 0.20:
            # Severe voltage sag (≤20%)
            max_duration = gc.lvrt_threshold_20pct_duration
            threshold_name = "20%"
        elif voltage <= 0.50:
            # Moderate voltage sag (20-50%)
            max_duration = gc.lvrt_threshold_50pct_duration
            threshold_name = "50%"
        elif voltage <= 0.85:
            # Mild voltage sag (50-85%)
            max_duration = gc.lvrt_threshold_85pct_duration
            threshold_name = "85%"
        else:
            # Voltage in normal/warning range, LVRT not triggered
            return

        # Check if ride-through duration exceeded
        if duration > max_duration:
            result.add_violation(
                ConstraintViolation.LVRT_VIOLATION,
                f"LVRT requirement violated: voltage {voltage:.1%} for {duration:.3f}s "
                f"exceeds {threshold_name} threshold limit of {max_duration:.3f}s"
            )

    def _check_hvrt(self, grid_state: GridState, result: ConstraintResult) -> None:
        """
        Check High Voltage Ride-Through requirements.

        HVRT curve defines voltage thresholds and required ride-through times:
        - 120% voltage: must ride through for 0.5s
        - 110% voltage: must ride through for 10s

        Args:
            grid_state: Current grid measurements
            result: ConstraintResult to add violations to
        """
        gc = self.grid_constraints
        voltage = grid_state.voltage_pu
        duration = grid_state.voltage_swell_duration

        # Determine which HVRT threshold applies based on voltage level
        if voltage >= 1.20:
            # Severe voltage swell (≥120%)
            max_duration = gc.hvrt_threshold_120pct_duration
            threshold_name = "120%"
        elif voltage >= 1.10:
            # Moderate voltage swell (110-120%)
            max_duration = gc.hvrt_threshold_110pct_duration
            threshold_name = "110%"
        else:
            # Voltage in normal range, HVRT not triggered
            return

        # Check if ride-through duration exceeded
        if duration > max_duration:
            result.add_violation(
                ConstraintViolation.HVRT_VIOLATION,
                f"HVRT requirement violated: voltage {voltage:.1%} for {duration:.3f}s "
                f"exceeds {threshold_name} threshold limit of {max_duration:.3f}s"
            )

    def _check_power_limits(
        self,
        state: DeviceState,
        target_power: float,
        result: ConstraintResult
    ) -> None:
        """Check power-related constraints."""
        c = self.constraints

        # Maximum power check
        if target_power > 0 and target_power > c.max_charge_power:
            result.add_violation(
                ConstraintViolation.MAX_POWER_EXCEEDED,
                f"Charge power {target_power:.1f}kW exceeds max {c.max_charge_power:.1f}kW"
            )
        elif target_power < 0 and abs(target_power) > c.max_discharge_power:
            result.add_violation(
                ConstraintViolation.MAX_POWER_EXCEEDED,
                f"Discharge power {abs(target_power):.1f}kW exceeds max {c.max_discharge_power:.1f}kW"
            )

        # Ramp rate check
        power_change = abs(target_power - state.power_current)
        # Assume 1 second response time for ramp rate calculation
        if power_change > c.max_ramp_rate:
            result.add_violation(
                ConstraintViolation.RAMP_RATE_EXCEEDED,
                f"Power change {power_change:.1f}kW/s exceeds max ramp {c.max_ramp_rate:.1f}kW/s"
            )

    def _check_battery(
        self,
        state: BatteryState,
        target_power: Optional[float],
        result: ConstraintResult
    ) -> None:
        """Check battery-specific constraints."""
        c = self.constraints

        # SOC limits
        if state.soc < c.min_soc:
            result.add_violation(
                ConstraintViolation.SOC_TOO_LOW,
                f"SOC {state.soc:.1%} below minimum {c.min_soc:.1%}"
            )
        elif state.soc > c.max_soc:
            result.add_violation(
                ConstraintViolation.SOC_TOO_HIGH,
                f"SOC {state.soc:.1%} above maximum {c.max_soc:.1%}"
            )

        # Prevent further discharge if SOC is low
        if target_power is not None and target_power < 0:  # Discharge
            if state.soc <= c.safe_soc_range[0]:
                result.add_violation(
                    ConstraintViolation.SOC_TOO_LOW,
                    f"Cannot discharge: SOC {state.soc:.1%} at/below safe minimum {c.safe_soc_range[0]:.1%}"
                )

        # Prevent further charge if SOC is high
        if target_power is not None and target_power > 0:  # Charge
            if state.soc >= c.safe_soc_range[1]:
                result.add_violation(
                    ConstraintViolation.SOC_TOO_HIGH,
                    f"Cannot charge: SOC {state.soc:.1%} at/above safe maximum {c.safe_soc_range[1]:.1%}"
                )

        # Temperature check
        if state.temperature > c.max_temperature:
            result.add_violation(
                ConstraintViolation.OVER_TEMPERATURE,
                f"Temperature {state.temperature:.1f}°C exceeds max {c.max_temperature:.1f}°C"
            )
        elif state.temperature < c.min_temperature:
            result.add_violation(
                ConstraintViolation.UNDER_TEMPERATURE,
                f"Temperature {state.temperature:.1f}°C below min {c.min_temperature:.1f}°C"
            )

        # Daily cycle limit
        if state.daily_cycles >= c.max_daily_cycles:
            result.add_violation(
                ConstraintViolation.DAILY_CYCLES_EXCEEDED,
                f"Daily cycles {state.daily_cycles} at/above max {c.max_daily_cycles}"
            )

def check_discharge_feasibility(state: BatteryState, power_kw: float, duration_s: float) -> bool:
    """
    Quick check if a discharge operation is feasible.

    Args:
        state: Current battery state
        power_kw: Requested discharge power (positive value)
        duration_s: Duration in seconds

    Returns:
        True if discharge is feasible without violating SOC limits
    """
    energy_kwh = power_kw * (duration_s / 3600) / state.discharge_efficiency
    final_soc = state.soc - (energy_kwh / state.capacity_kwh)
    return final_soc >= 0.2  # Safe minimum


def check_charge_feasibility(state: BatteryState, power_kw: float, duration_s: float) -> bool:
    """
    Quick check if a charge operation is feasible.

    Args:
        state: Current battery state
        power_kw: Requested charge power (positive value)
        duration_s: Duration in seconds

    Returns:
        True if charge is feasible without violating SOC limits
    """
    energy_kwh = power_kw * state.charge_efficiency * (duration_s / 3600)
    final_soc = state.soc + (energy_kwh / state.capacity_kwh)
    return final_soc <= 0.8  # Safe maximum
