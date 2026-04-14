"""
Simulation Scenarios

Four dispatch scenarios defined by supply-demand ratio r and its
rate of change (Methods, Eq. 1):

    r = (P_base + P_RE) / P_load

    Routine absorption  (valley_filling):         r > 1+eps, |dr/dt| < delta
    Routine support     (peak_shaving):           r < 1-eps, |dr/dt| < delta
    Emergency charge    (emergency_grid_stability): r > 1+eps, |dr/dt| >= delta
    Emergency discharge (emergency_supply_shortage): r < 1-eps, |dr/dt| >= delta

    where eps = 0.05 (equilibrium deviation threshold)
          delta = 0.1 per min (rate-of-change threshold)

Each scenario defines r(t) and d(t) profiles that determine the
broadcast signal score s = clip[(r-1)*d, -1, +1].
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import numpy as np
from datetime import datetime, timedelta


class ScenarioType(Enum):
    """
    Types of simulation scenarios.

    Categorization with supply/demand side distinction.
    """
    # Normal operation (baseline)
    NORMAL = "normal"

    # Primary scenarios
    PEAK_SHAVING = "peak_shaving"           # Supply-side, routine
    VALLEY_FILLING = "valley_filling"       # Supply-side, routine

    # Emergency scenarios (subdivided)
    EMERGENCY_GRID_STABILITY = "emergency_grid_stability"    # Supply-side, fast
    EMERGENCY_SUPPLY_SHORTAGE = "emergency_supply_shortage"  # Demand-side, slow

    # Legacy alias for backward compatibility
    EMERGENCY = "emergency"  # Maps to EMERGENCY_GRID_STABILITY

    # Keep for frequency regulation research (subset of grid stability)
    FREQUENCY_REGULATION = "frequency_regulation"


@dataclass
class LoadProfile:
    """
    24-hour load profile definition.

    Based on typical residential/commercial load patterns.
    """
    # Hourly load factors (0-24 hours)
    hourly_factors: np.ndarray = field(
        default_factory=lambda: np.ones(24)
    )

    # Base load in MW
    base_load_mw: float = 1000.0

    # Peak load time
    peak_hour: int = 19

    # Valley load time
    valley_hour: int = 4

    def get_load(self, hour: float) -> float:
        """Get load at a specific hour with interpolation."""
        hour_int = int(hour) % 24
        hour_frac = hour - int(hour)

        # Interpolate between hours
        next_hour = (hour_int + 1) % 24
        factor = (1 - hour_frac) * self.hourly_factors[hour_int] + \
                 hour_frac * self.hourly_factors[next_hour]

        return self.base_load_mw * factor

    @classmethod
    def typical_residential(cls) -> 'LoadProfile':
        """
        Create typical residential load profile.

        Pattern: Morning peak (7-9am), Evening peak (18-21)
        Reference: Representative residential load pattern
        """
        factors = np.array([
            0.6, 0.55, 0.5, 0.5, 0.5, 0.55,    # 0-5
            0.7, 0.85, 0.9, 0.8, 0.75, 0.8,    # 6-11
            0.85, 0.8, 0.75, 0.75, 0.8, 0.9,   # 12-17
            1.0, 1.0, 0.95, 0.85, 0.75, 0.65,  # 18-23
        ])
        return cls(hourly_factors=factors, base_load_mw=1000.0, peak_hour=19, valley_hour=4)

    @classmethod
    def typical_commercial(cls) -> 'LoadProfile':
        """
        Create typical commercial load profile.

        Pattern: Daytime peak (9-18)
        Reference: Representative commercial load pattern
        """
        factors = np.array([
            0.4, 0.35, 0.35, 0.35, 0.35, 0.4,  # 0-5
            0.5, 0.7, 0.85, 1.0, 1.0, 1.0,     # 6-11
            0.95, 1.0, 1.0, 1.0, 0.95, 0.85,   # 12-17
            0.6, 0.5, 0.45, 0.4, 0.4, 0.4,     # 18-23
        ])
        return cls(hourly_factors=factors, base_load_mw=500.0, peak_hour=10, valley_hour=3)

    @classmethod
    def high_demand(cls) -> 'LoadProfile':
        """
        Create high-demand load profile (extended afternoon peak).

        Pattern: Extended afternoon peak due to intensive load
        Reference: Representative high-demand load pattern
        """
        factors = np.array([
            0.7, 0.65, 0.6, 0.6, 0.6, 0.65,    # 0-5
            0.75, 0.85, 0.95, 1.0, 1.0, 1.0,   # 6-11
            1.0, 1.05, 1.1, 1.1, 1.05, 1.0,    # 12-17 (extended peak)
            0.95, 0.9, 0.85, 0.8, 0.75, 0.7,   # 18-23
        ])
        return cls(hourly_factors=factors, base_load_mw=1200.0, peak_hour=15, valley_hour=3)


@dataclass
class RenewableProfile:
    """
    Renewable generation profile (wind/solar).

    Reference: Representative wind/solar generation patterns
    """
    # Solar generation curve (normalized)
    solar_factors: np.ndarray = field(
        default_factory=lambda: np.zeros(24)
    )

    # Wind generation curve (normalized)
    wind_factors: np.ndarray = field(
        default_factory=lambda: np.ones(24) * 0.3
    )

    # Installed capacity (regional scale)
    solar_capacity_mw: float = 500.0
    wind_capacity_mw: float = 200.0

    def get_solar(self, hour: float) -> float:
        """Get solar generation at a specific hour."""
        hour_int = int(hour) % 24
        return self.solar_capacity_mw * self.solar_factors[hour_int]

    def get_wind(self, hour: float) -> float:
        """Get wind generation at a specific hour."""
        hour_int = int(hour) % 24
        return self.wind_capacity_mw * self.wind_factors[hour_int]

    def get_total(self, hour: float) -> float:
        """Get total renewable generation."""
        return self.get_solar(hour) + self.get_wind(hour)

    @classmethod
    def typical_renewable(cls) -> 'RenewableProfile':
        """Create typical renewable generation profile."""
        # Solar peaks at noon
        solar = np.array([
            0, 0, 0, 0, 0, 0.05,               # 0-5
            0.15, 0.35, 0.55, 0.75, 0.9, 0.95, # 6-11
            1.0, 0.95, 0.85, 0.7, 0.5, 0.3,    # 12-17
            0.1, 0.02, 0, 0, 0, 0,             # 18-23
        ])
        # Wind typically stronger at night
        wind = np.array([
            0.4, 0.45, 0.5, 0.5, 0.45, 0.35,   # 0-5
            0.25, 0.2, 0.2, 0.25, 0.3, 0.35,   # 6-11
            0.35, 0.3, 0.25, 0.2, 0.2, 0.25,   # 12-17
            0.3, 0.35, 0.4, 0.45, 0.45, 0.4,   # 18-23
        ])
        return cls(solar_factors=solar, wind_factors=wind)

    @classmethod
    def high_variability(cls) -> 'RenewableProfile':
        """
        Create high variability profile for grid stability testing.

        Simulates rapid renewable fluctuations that trigger frequency events.
        """
        # Simulate cloud passing events with rapid drops
        solar = np.array([
            0, 0, 0, 0, 0, 0.05,
            0.15, 0.35, 0.6, 0.4, 0.9, 0.5,    # High variability
            1.0, 0.6, 0.85, 0.4, 0.5, 0.3,
            0.1, 0.02, 0, 0, 0, 0,
        ])
        wind = np.array([
            0.4, 0.6, 0.3, 0.5, 0.45, 0.25,    # High variability
            0.25, 0.2, 0.2, 0.25, 0.3, 0.35,
            0.35, 0.3, 0.25, 0.2, 0.2, 0.25,
            0.3, 0.5, 0.3, 0.6, 0.4, 0.5,
        ])
        return cls(solar_factors=solar, wind_factors=wind)


@dataclass
class ScenarioConfig:
    """
    Configuration for a simulation scenario.

    All parameters traceable to real-world references.
    """
    name: str
    scenario_type: ScenarioType
    description: str

    # Extended description for Paper
    paper_description: str = ""

    # Perspective
    perspective: str = "supply"  # "supply" or "demand"

    # Duration
    duration_hours: float = 24.0

    # Load and generation profiles
    load_profile: LoadProfile = field(default_factory=LoadProfile.typical_residential)
    renewable_profile: Optional[RenewableProfile] = None

    # Signal parameters
    signal_intensity_range: tuple = (1000, 3000)

    # Success metrics thresholds
    target_response_rate: float = 0.5
    target_peak_reduction: float = 0.15
    max_latency_ms: float = 100.0

    # Emergency parameters
    emergency_duration_minutes: float = 30.0
    emergency_intensity: int = 4000

    # Supply shortage specific
    supply_deficit_mw: float = 0.0
    shortage_cause: str = ""


class ScenarioGenerator:
    """
    Generates predefined scenarios for publication-grade validation.

    Scenario Structure:
    ├── Peak Shaving (supply-side, routine)
    ├── Valley Filling (supply-side, routine)
    └── Emergency Response
        ├── Grid Stability (supply-side, fast)
        └── Supply Shortage (demand-side, slow)
    """

    # ==================
    # PRIMARY SCENARIO 1: PEAK SHAVING
    # ==================

    @staticmethod
    def peak_shaving() -> ScenarioConfig:
        """
        Peak demand reduction scenario (17:00-21:00).

        Supply-side perspective: Grid needs to reduce peak load.

        Context:
        - Evening peak when solar generation drops
        - Air conditioning and EV charging overlap
        - Grid capacity constraints

        EPS Response:
        - Batteries: Discharge to grid
        - EVs: Delay charging
        - HVAC: Pre-cool/reduce setpoint
        """
        return ScenarioConfig(
            name="Peak Shaving",
            scenario_type=ScenarioType.PEAK_SHAVING,
            description="Reduce evening peak demand through coordinated discharge",
            paper_description=(
                "Peak shaving scenario evaluates EPS capability to reduce "
                "evening demand peaks (17:00-21:00) when solar generation drops. "
                "Primary metric: Response magnitude (MW)."
            ),
            perspective="supply",
            load_profile=LoadProfile.typical_residential(),
            renewable_profile=RenewableProfile.typical_renewable(),
            signal_intensity_range=(2000, 4000),

            target_peak_reduction=0.20,
            target_response_rate=0.5,
        )

    # ==================
    # PRIMARY SCENARIO 2: VALLEY FILLING
    # ==================

    @staticmethod
    def valley_filling() -> ScenarioConfig:
        """
        Renewable surplus absorption scenario (23:00-06:00 and 11:00-14:00).

        Supply-side perspective: Absorb excess wind/solar generation.

        Context:
        - Night: Wind generation peak + low demand
        - Noon: Solar generation peak + commercial lunch break
        - Curtailment risk

        EPS Response:
        - Batteries: Charge from grid
        - EVs: Encourage charging
        - HVAC: Pre-condition buildings
        """
        return ScenarioConfig(
            name="Valley Filling",
            scenario_type=ScenarioType.VALLEY_FILLING,
            description="Absorb renewable surplus through flexible charging",
            paper_description=(
                "Valley filling scenario evaluates EPS capability to absorb "
                "excess renewable generation during low-demand periods. "
                "Primary metric: Absorption capacity (MW), Curtailment reduction (%)."
            ),
            perspective="supply",
            load_profile=LoadProfile.typical_residential(),
            renewable_profile=RenewableProfile.typical_renewable(),
            signal_intensity_range=(2000, 4000),

            target_response_rate=0.6,
        )

    # ==================
    # PRIMARY SCENARIO 3: EMERGENCY RESPONSE
    # ==================

    @staticmethod
    def emergency_grid_stability() -> ScenarioConfig:
        """
        Grid stability emergency scenario (frequency/voltage).

        Supply-side perspective: Rapid response to grid events.

        Context:
        - Generator trip
        - Renewable output sudden drop (cloud passing)
        - Transmission line fault

        EPS Response:
        - Sub-second signal broadcast
        - Batteries: Immediate discharge
        - EVs: Suspend charging
        - HVAC: Immediate reduction

        Critical Metric: P99 latency < 100ms
        """
        return ScenarioConfig(
            name="Emergency - Grid Stability",
            scenario_type=ScenarioType.EMERGENCY_GRID_STABILITY,
            description="Rapid frequency/voltage stabilization response",
            paper_description=(
                "Grid stability scenario evaluates EPS emergency response capability "
                "for supply-side contingencies (generator trip, renewable fluctuation). "
                "Critical metric: P99 latency < 100ms for frequency regulation."
            ),
            perspective="supply",
            renewable_profile=RenewableProfile.high_variability(),
            signal_intensity_range=(3500, 4095),

            target_response_rate=0.8,
            max_latency_ms=100.0,  # Critical: <100ms for frequency response
            emergency_duration_minutes=15.0,  # Short duration
            emergency_intensity=4000,
        )

    @staticmethod
    def emergency_supply_shortage() -> ScenarioConfig:
        """
        Supply shortage emergency scenario (urban peak deficit).

        Demand-side perspective: Urban power deficit mitigation.

        Context:
        - Extreme load surge
        - Generator maintenance overlap
        - Cross-region transmission constraint
        - Coal/gas supply shortage

        EPS Response:
        - Sustained demand reduction
        - Batteries: Extended discharge
        - EVs: Defer charging
        - HVAC: Raise setpoint (comfort trade-off)

        Duration: Minutes to hours (longer than grid stability)
        """
        return ScenarioConfig(
            name="Emergency - Supply Shortage",
            scenario_type=ScenarioType.EMERGENCY_SUPPLY_SHORTAGE,
            description="Sustained response to urban power deficit",
            paper_description=(
                "Supply shortage scenario evaluates EPS capability for demand-side "
                "power deficit events (extreme weather, equipment failure). "
                "Demonstrates cross-regional power mutual support. "
                "Primary metric: Coverage rate (%), sustained response duration."
            ),
            perspective="demand",
            load_profile=LoadProfile.high_demand(),
            renewable_profile=RenewableProfile.typical_renewable(),
            signal_intensity_range=(3000, 4095),

            target_response_rate=0.7,
            max_latency_ms=300.0,  # Relaxed: minutes acceptable
            emergency_duration_minutes=120.0,  # Extended duration
            emergency_intensity=3500,
            supply_deficit_mw=500.0,
            shortage_cause="supply_deficit",
        )

    # ==================
    # LEGACY/ALIAS METHODS
    # ==================

    @staticmethod
    def emergency_response() -> ScenarioConfig:
        """
        Legacy alias for emergency_grid_stability().

        For backward compatibility.
        """
        config = ScenarioGenerator.emergency_grid_stability()
        config.name = "Emergency Response"
        config.scenario_type = ScenarioType.EMERGENCY
        return config

    # ==================
    # HELPER METHODS
    # ==================

    @classmethod
    def get_all_scenarios(cls) -> List[ScenarioConfig]:
        """
        Get all primary scenarios (Publication paper).

        Returns the 4 scenarios (3 primary + 1 subdivision):
        1. Peak Shaving
        2. Valley Filling
        3. Emergency - Grid Stability
        4. Emergency - Supply Shortage
        """
        return [
            cls.peak_shaving(),
            cls.valley_filling(),
            cls.emergency_grid_stability(),
            cls.emergency_supply_shortage(),
        ]

    @classmethod
    def get_primary_scenarios(cls) -> List[ScenarioConfig]:
        """
        Get primary 3 scenarios without emergency subdivision.

        For experiments using primary scenario categories.
        """
        return [
            cls.peak_shaving(),
            cls.valley_filling(),
            cls.emergency_response(),
        ]

    @classmethod
    def get_scenario_by_name(cls, name: str) -> Optional[ScenarioConfig]:
        """Get scenario by name (case-insensitive, supports aliases)."""
        name_lower = name.lower().replace("_", " ").replace("-", " ")

        mapping = {
            "peak shaving": cls.peak_shaving,
            "peak_shaving": cls.peak_shaving,
            "valley filling": cls.valley_filling,
            "valley_filling": cls.valley_filling,
            "emergency": cls.emergency_response,
            "emergency response": cls.emergency_response,
            "emergency_response": cls.emergency_response,
            "emergency grid stability": cls.emergency_grid_stability,
            "emergency_grid_stability": cls.emergency_grid_stability,
            "grid stability": cls.emergency_grid_stability,
            "emergency supply shortage": cls.emergency_supply_shortage,
            "emergency_supply_shortage": cls.emergency_supply_shortage,
            "supply shortage": cls.emergency_supply_shortage,
        }

        if name_lower in mapping:
            return mapping[name_lower]()
        return None


@dataclass
class ScenarioMetrics:
    """Metrics for evaluating scenario performance."""
    # Response metrics
    total_responses: int = 0
    response_rate: float = 0.0

    # Energy metrics
    total_energy_kwh: float = 0.0
    peak_reduction_mw: float = 0.0
    peak_reduction_percent: float = 0.0

    # Absorption metrics (for valley filling)
    absorption_mw: float = 0.0
    curtailment_reduction_percent: float = 0.0

    # Latency metrics
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Cost metrics
    total_incentive: float = 0.0  # Reserved (always 0)
    cost_per_kwh: float = 0.0

    # Grid impact
    frequency_deviation_hz: float = 0.0
    voltage_deviation_percent: float = 0.0

    # Supply shortage specific
    coverage_rate: float = 0.0
    sustained_duration_minutes: float = 0.0

    # Success indicators
    met_response_target: bool = False
    met_latency_target: bool = False
    met_peak_reduction_target: bool = False

    def evaluate(self, config: ScenarioConfig) -> Dict[str, Any]:
        """Evaluate metrics against scenario targets."""
        self.met_response_target = self.response_rate >= config.target_response_rate
        self.met_latency_target = self.p99_latency_ms <= config.max_latency_ms
        self.met_peak_reduction_target = self.peak_reduction_percent >= config.target_peak_reduction

        return {
            'overall_success': all([
                self.met_response_target,
                self.met_latency_target,
            ]),
            'response_rate': {
                'actual': self.response_rate,
                'target': config.target_response_rate,
                'met': self.met_response_target,
            },
            'latency': {
                'p99_ms': self.p99_latency_ms,
                'target_ms': config.max_latency_ms,
                'met': self.met_latency_target,
            },
            'peak_reduction': {
                'percent': self.peak_reduction_percent,
                'target': config.target_peak_reduction,
                'met': self.met_peak_reduction_target,
            },
            'perspective': config.perspective,
            'scenario_type': config.scenario_type.value,
        }


