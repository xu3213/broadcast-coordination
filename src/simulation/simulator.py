"""
Agent-Based Battery Fleet Simulator

Simulates N heterogeneous battery storage devices responding independently
to a shared broadcast signal via a three-step Bernoulli protocol:

    Step 1 — Eligibility screening (Methods, Eq. 3):
        alpha_elig_i = 1[SOC_min_i <= SOC_i <= SOC_max_i]
                     * 1[t in T_i]               (all devices schedulable)
                     * 1[device i online]        (offline rate 1.5%)
                     * 1[signal received]        (packet loss 0.1%)

    Step 2 — Available capacity (Methods, Eq. 4):
        E_eff_i   = SOH_i * E_rated_i
        C_avail_i = min(P_rated_i, E_headroom_i / dt)
        where E_headroom = (SOC_max - SOC) * E_eff  [charge]
                         = (SOC - SOC_min) * E_eff  [discharge]

    Step 3 — Probabilistic response (Methods, Eq. 5-8):
        score   = intensity / 4095                    decoded |s|
        w_i     = (SOC_i - SOC_min_i) / (SOC_max_i - SOC_min_i)    (discharge)
                  (SOC_max_i - SOC_i) / (SOC_max_i - SOC_min_i)    (charge)
        p_i     = score * w_i
        alpha_i ~ Bernoulli(p_i)
        P_i     = alpha_i * sign(s) * C_avail_i * eta

    where SOC_min ~ U(0.10, 0.20), SOC_max ~ U(0.80, 0.95) per device,
    SOH ~ U(0.82, 1.0), and eta ~ N(1, 0.039) models delivery noise.

    Aggregate: P_agg = sum_{i=1}^{N} P_i

By the Law of Large Numbers, P_agg converges to a deterministic function
of the signal parameters as N grows, which is the core finding of the paper.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import logging

from ..signal import EPSSignal, EPSSignalEncoder, EPSSignalDecoder
from ..edge import (
    BatteryModel,
    create_residential_battery,
    BatteryState,
)
from .constants import (
    EPS_LATENCY, BATTERY_RESPONSE,
)

logger = logging.getLogger(__name__)


# Correlation Structure

@dataclass
class CorrelationConfig:
    """
    Configuration for inter-device correlation structure.

    Models two sources of correlation among battery DER responses:
    1. Global environmental shock (weather, grid frequency, electricity price)
    2. Regional environmental shock (urban heat island, local weather)

    Mathematical basis:
        For Bernoulli responses B_i with logit-space additive shock Z ~ N(0, σ_z),
        p_i = sigmoid(logit(p) + Z), giving pairwise correlation:
        ρ ≈ p(1-p) × σ_z²
        where p is the marginal response probability.

        Default config: σ_z = sqrt(0.15² + 0.08²) ≈ 0.17, p ≈ 0.52
        → ρ_within ≈ 0.52 × 0.48 × 0.17² ≈ 0.006
        → N_eff = N / [1 + (N/K - 1) × ρ] for K regions

    Parameter sensitivity (N=5000, K=10):
        σ_z=0.0  → ρ=0,     N_eff=5000  (iid)
        σ_z=0.17 → ρ≈0.006, N_eff≈1000  (weak)
        σ_z=0.34 → ρ≈0.023, N_eff≈91    (moderate)
    """
    enable_correlation: bool = False

    # Global environmental shock (weather/price/frequency — affects ALL devices)
    global_shock_std: float = 0.15       # σ_z for logit-space shock; ρ≈0.006 at p≈0.52

    # Regional environmental shock (urban heat island, local weather)
    regional_shock_std: float = 0.08     # Additional per-region perturbation

    # Battery SOC spatial clustering
    battery_regional_soc_std: float = 0.15   # Region-center SOC std
    battery_within_cluster_std: float = 0.08 # Within-cluster SOC std


class SimulationLevel(Enum):
    """Simulation fidelity levels."""
    LEVEL1_AGENT = 1      # Full agent-based simulation


@dataclass
class SimulationConfig:
    """Simulation configuration."""
    level: SimulationLevel = SimulationLevel.LEVEL1_AGENT
    num_devices: int = 5000
    duration_seconds: float = 86400.0  # 1 day
    time_step: float = 60.0  # 1 minute
    random_seed: Optional[int] = 42

    # Device mix (battery-only)
    battery_fraction: float = 1.0

    # Parallelization
    num_workers: int = 4
    chunk_size: int = 1000

    # Region configuration
    num_regions: int = 5

    # Signal parameters
    broadcast_interval: float = 60.0  # seconds

    # LEO satellite packet loss rate (industry benchmark: 0.1%)
    eps_packet_loss_rate: float = 0.001

    # Battery capacity range for heterogeneity sensitivity
    battery_capacity_range: Tuple[float, float] = (5.0, 20.0)  # kWh

    # High-fidelity fleet heterogeneity parameters
    battery_c_rate_mean: float = 0.45       # P/E ratio LogNormal mean (C-rate)
    battery_c_rate_sigma: float = 0.35      # P/E ratio LogNormal sigma
    battery_soh_range: Tuple[float, float] = (0.82, 1.0)  # Fleet age SOH range
    device_offline_rate: float = 0.015      # BMS/inverter random unavailability

    # Correlation structure
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)

    # Model mismatch parameters (robustness analysis for device model errors)
    response_prob_bias: float = 1.0    # Multiplicative bias on response probability (1.0 = no bias)
    soc_noise_std: float = 0.0         # Additive Gaussian noise std on SOC observation

    # Structural mismatch: continuous proportional response (M1 experiment)
    # When True, devices output power proportional to prob instead of Bernoulli sampling.
    # Same E[P_agg] but different variance structure (no binary on/off noise).
    continuous_response: bool = False
    # Sigmoid response: p_i = sigmoid(k * (|s| * w_soc - 0.5)) instead of linear p_i = |s| * w_soc
    # Changes the SHAPE of g(s) from near-linear to S-curve (structural mismatch that alters E[P_agg|s])
    sigmoid_response: bool = False
    sigmoid_steepness: float = 5.0  # k parameter; higher = sharper S-curve

    def __post_init__(self):
        if abs(self.battery_fraction - 1.0) > 1e-6:
            raise ValueError(f"battery_fraction must be 1.0, got {self.battery_fraction}")


@dataclass
class DevicePopulation:
    """Container for device populations (battery-only)."""
    batteries: List[BatteryModel] = field(default_factory=list)

    # State tracking
    battery_states: List[BatteryState] = field(default_factory=list)

    # Region assignments
    region_assignments: Dict[str, int] = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        return len(self.batteries)

    def get_devices_by_region(self, region_id: int) -> Dict[str, List]:
        """Get devices in a specific region."""
        result = {'batteries': []}
        for i, battery in enumerate(self.batteries):
            if self.region_assignments.get(f'battery_{i}') == region_id:
                result['batteries'].append((i, battery))
        return result


@dataclass
class TimeStep:
    """Represents a single simulation time step."""
    step_index: int
    timestamp: datetime
    elapsed_seconds: float

    # Signal broadcast in this step
    signal: Optional[EPSSignal] = None

    # Aggregated metrics
    total_response_kw: float = 0.0
    responding_devices: int = 0
    latency_samples: List[float] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Result of a simulation run."""
    # Summary statistics
    total_responses: int = 0
    total_energy_kwh: float = 0.0
    average_latency_ms: float = 0.0
    response_rate: float = 0.0

    # Time series data
    time_steps: List[TimeStep] = field(default_factory=list)

    # Per-region metrics
    region_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Device-type breakdown
    battery_response_kwh: float = 0.0

    # Energy accounting
    charge_energy_kwh: float = 0.0
    discharge_energy_kwh: float = 0.0
    net_energy_kwh: float = 0.0
    load_reduction_kwh: float = 0.0
    load_increase_kwh: float = 0.0

    # Statistical measures
    latency_percentiles: Dict[str, float] = field(default_factory=dict)
    response_distribution: Dict[str, float] = field(default_factory=dict)

    # Responding device count (average per step)
    n_responding_devices: int = 0

    # Scalability metrics
    devices_per_second: float = 0.0
    memory_usage_mb: float = 0.0

    # Raw data for further analysis
    metrics: Dict[str, Any] = field(default_factory=dict)

    def compute_latency_percentiles(self):
        """Compute latency percentiles from time steps."""
        all_latencies = []
        for ts in self.time_steps:
            all_latencies.extend(ts.latency_samples)

        if all_latencies:
            arr = np.array(all_latencies)
            self.latency_percentiles = {
                'p50': float(np.percentile(arr, 50)),
                'p90': float(np.percentile(arr, 90)),
                'p95': float(np.percentile(arr, 95)),
                'p99': float(np.percentile(arr, 99)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
            }
            self.average_latency_ms = self.latency_percentiles['mean']


class PopulationGenerator:
    """Generates realistic device populations."""

    def __init__(self, config: SimulationConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    def generate(self) -> DevicePopulation:
        """Generate device population based on config.

        When correlation is enabled, generates spatially clustered initial states:
        - Battery SOC: region-center SOC + within-cluster noise
        """
        pop = DevicePopulation()
        corr = self.config.correlation

        n_batteries = self.config.num_devices

        # Pre-generate region centers for correlated initialization
        if corr.enable_correlation:
            region_soc_centers = {}
            for r in range(self.config.num_regions):
                region_soc_centers[r] = np.clip(
                    self.rng.normal(0.5, corr.battery_regional_soc_std), 0.15, 0.85
                )

        # Generate batteries with varied parameters
        for i in range(n_batteries):
            capacity = self.rng.uniform(*self.config.battery_capacity_range)  # kWh residential

            # Assign to region first (needed for correlated SOC)
            region = int(self.rng.integers(0, self.config.num_regions))
            pop.region_assignments[f'battery_{i}'] = region

            if corr.enable_correlation:
                # Correlated SOC: sample from region center
                initial_soc = np.clip(
                    self.rng.normal(region_soc_centers[region], corr.battery_within_cluster_std),
                    0.1, 0.95
                )
            else:
                initial_soc = self.rng.uniform(0.2, 0.9)  # U(0.2, 0.9) per system description

            # P/E ratio: LogNormal based on residential battery product distribution
            # Typical range 0.22-1.0C across mainstream products
            mu_c = self.config.battery_c_rate_mean
            sigma_c = self.config.battery_c_rate_sigma
            c_rate = float(np.clip(self.rng.lognormal(
                mean=np.log(mu_c) - sigma_c**2 / 2,  # E[X] = mu_c
                sigma=sigma_c,
            ), 0.2, 1.0))

            # Fleet age heterogeneity: SOH ~ U(soh_low, soh_high)
            initial_soh = float(self.rng.uniform(*self.config.battery_soh_range))

            battery = create_residential_battery(
                capacity_kwh=capacity,
                c_rate=c_rate,
                initial_soh=initial_soh,
            )
            # Randomize SOC operating boundaries — per Table 1 (SOC_min,i, SOC_max,i)
            battery.params.soc_reserve = float(self.rng.uniform(0.1, 0.2))  # Discharge lower limit
            battery.params.soc_max = float(self.rng.uniform(0.8, 0.95))     # Charge upper limit
            battery._soc = initial_soc
            pop.batteries.append(battery)

            max_power = capacity * c_rate
            state = BatteryState(
                device_id=f'battery_{i}',
                device_type='battery',
                soc=initial_soc,
                capacity_kwh=capacity,
                max_charge_kw=max_power,
                max_discharge_kw=max_power,
            )
            pop.battery_states.append(state)

        return pop


class SignalGenerator:
    """
    Generates EPS broadcast signals for simulation.

    Signal parameters are derived from the supply-demand balance:
    - Solar generation (sinusoidal daytime profile)
    - Wind generation (stochastic, stronger at night)
    - Electrical load (dual-peak diurnal profile)
    - net_balance = supply - load determines signal direction and intensity
    """

    def __init__(self, config: SimulationConfig, rng: np.random.Generator, start_hour: float = 0.0):
        self.config = config
        self.rng = rng
        self.encoder = EPSSignalEncoder()
        self.start_hour = start_hour

        # Grid-scale generation and load baselines (MW)
        # Sources: NREL SAM (solar), Global Wind Atlas 3.0 (wind), public utility data (load)
        self.base_solar_capacity = 500.0  # PV capacity (MW)
        self.base_wind_capacity = 150.0   # Wind capacity (MW)
        self.base_load = 400.0            # Base load (MW)

        # Override mechanism for external signal injection
        self._override_signal: Optional[EPSSignal] = None

    def set_override_signal(
        self,
        intensity: Optional[int] = None,
        price_value: Optional[float] = None,
        supply_demand: Optional[int] = None,
    ) -> None:
        """
        Set override signal parameters for external control.

        Args:
            intensity: Override intensity (0-4095), or None to clear.
            price_value: Accepted for API compatibility, ignored.
            supply_demand: Override supply_demand (0-15), or None.
        """
        if intensity is not None or supply_demand is not None:
            self._override_signal = {
                'intensity': intensity,
                'supply_demand': supply_demand,
            }
        else:
            self._override_signal = None

    def clear_override_signal(self) -> None:
        """Clear any override signal, returning to normal generation."""
        self._override_signal = None

    def set_start_hour(self, hour: float) -> None:
        """
        Set the simulation start hour.

        This is crucial for generating balanced training data:
        - start_hour=10-14: Solar surplus period → charge (positive) responses
        - start_hour=18-22: Evening peak period → discharge (negative) responses
        - start_hour=0: Default, starts at midnight

        Args:
            hour: Start hour (0-24)
        """
        self.start_hour = hour % 24

    def get_solar_generation(self, hour: float) -> float:
        """Solar generation profile (MW). Sinusoidal 06:00-18:00 with noise."""
        if hour < 6 or hour > 18:
            return 0.0

        # Sinusoidal profile: 6-18h mapped to 0-pi
        phase = (hour - 6) / 12 * np.pi
        generation = self.base_solar_capacity * np.sin(phase)

        noise = self.rng.normal(1.0, 0.1)  # +/-10% random variation
        return max(0, generation * noise)

    def get_wind_generation(self, hour: float) -> float:
        """Wind generation profile (MW). Stronger at night, weaker midday."""
        if 22 <= hour or hour <= 6:
            base_factor = 0.7  # Night (strong wind)
        elif 10 <= hour <= 16:
            base_factor = 0.3  # Midday (calm)
        else:
            base_factor = 0.5

        noise = self.rng.normal(1.0, 0.3)  # +/-30% random variation
        return max(0, self.base_wind_capacity * base_factor * noise)

    def get_base_load(self, hour: float) -> float:
        """Base electrical load profile (MW). Dual-peak: morning and evening."""
        if 0 <= hour < 5:
            factor = 0.6   # Night valley
        elif 5 <= hour < 7:
            factor = 0.7   # Early morning ramp
        elif 7 <= hour < 9:
            factor = 1.0   # Morning peak
        elif 9 <= hour < 11:
            factor = 0.9   # Late morning
        elif 11 <= hour < 14:
            factor = 0.95  # Midday plateau
        elif 14 <= hour < 17:
            factor = 0.85  # Afternoon
        elif 17 <= hour < 21:
            factor = 1.2   # Evening peak (highest)
        elif 21 <= hour < 23:
            factor = 0.9   # Evening decline
        else:
            factor = 0.7   # Late night

        noise = self.rng.normal(1.0, 0.05)  # +/-5% random variation
        return self.base_load * factor * noise

    def compute_supply_demand_state(self, hour: float) -> dict:
        """Compute supply-demand state for signal generation.

        Returns:
            Dict with supply_demand (0-15), scenario, net_balance (MW).
            0-3: surplus (emergency charge), 4-6: surplus (normal),
            7: balanced, 8-11: shortage (normal), 12-15: shortage (emergency).
        """
        solar = self.get_solar_generation(hour)
        wind = self.get_wind_generation(hour)
        load = self.get_base_load(hour)

        total_supply = solar + wind
        net_balance = total_supply - load

        if net_balance > 200:
            supply_demand = 0  # Surplus
            scenario = 'valley_filling'
            is_surplus = True
        elif net_balance > 100:
            supply_demand = 2
            scenario = 'valley_filling'
            is_surplus = True
        elif net_balance > 50:
            supply_demand = 4
            scenario = 'valley_filling'
            is_surplus = True
        elif net_balance > 0:
            supply_demand = 6
            scenario = 'normal'
            is_surplus = True
        elif net_balance > -50:
            supply_demand = 7  # Near-balanced
            scenario = 'normal'
            is_surplus = False
        elif net_balance > -100:
            supply_demand = 9
            scenario = 'peak_shaving'
            is_surplus = False
        elif net_balance > -200:
            supply_demand = 11
            scenario = 'peak_shaving'
            is_surplus = False
        elif net_balance > -300:
            supply_demand = 13
            scenario = 'peak_shaving'
            is_surplus = False
        else:
            supply_demand = 15  # Shortage
            scenario = 'emergency'
            is_surplus = False

        # Supply-demand ratio r = total_supply / load (system description alignment)
        r = total_supply / max(load, 1e-6)

        return {
            'supply_demand': supply_demand,
            'supply_demand_ratio': r,
            'scenario': scenario,
            'is_surplus': is_surplus,
            'net_balance': net_balance,
            'solar_mw': solar,
            'wind_mw': wind,
            'load_mw': load,
            'total_supply_mw': total_supply,
        }

    def generate_signal(
        self,
        region_id: int,
        supply_demand: int,
        intensity: int,
        priority: int = 8,
    ) -> EPSSignal:
        """Generate a broadcast signal (price field reserved=0)."""
        return EPSSignal(
            version=1,
            timestamp_seq=int(datetime.now().timestamp()) % 16,
            region_id=region_id,
            supply_demand=supply_demand,
            intensity=intensity,
            price=0,  # Reserved field
            priority=priority,
        )

    def generate_scenario_signals(
        self,
        scenario: str,
        time_step: int,
        num_regions: int,
    ) -> List[EPSSignal]:
        """
        Generate signals for a scenario.

        Signal = (supply_demand, intensity, priority). Price field is reserved.

        
        -  net_balance = (solar + wind) - load
        - Surplus (net_balance > 0): Charge direction, supply_demand < 8
        - Shortage (net_balance < 0): Discharge direction, supply_demand >= 8
        """
        signals = []
        hour = (self.start_hour + time_step * self.config.time_step / 3600) % 24

        state = self.compute_supply_demand_state(hour)
        net_balance = state['net_balance']
        imbalance = abs(net_balance)

        if scenario == 'peak_shaving':
            if net_balance < -100:
                supply_demand = min(15, 10 + int(imbalance / 50))
                intensity = int(self.rng.uniform(3000, 4095))
                priority = 12
            elif net_balance < -50:
                supply_demand = 10
                intensity = int(self.rng.uniform(2000, 3000))
                priority = 10
            elif net_balance < 0:
                supply_demand = 8
                intensity = int(self.rng.uniform(1500, 2000))
                priority = 8
            else:
                supply_demand = max(3, 7 - int(net_balance / 30))
                intensity = int(self.rng.uniform(1000, 1500))
                priority = 6

        elif scenario == 'valley_filling':
            if net_balance > 100:
                supply_demand = max(0, 3 - int(net_balance / 50))
                intensity = int(self.rng.uniform(3000, 4095))
                priority = 12
            elif net_balance > 50:
                supply_demand = 4
                intensity = int(self.rng.uniform(2000, 3000))
                priority = 10
            elif net_balance > 0:
                supply_demand = 6
                intensity = int(self.rng.uniform(1500, 2000))
                priority = 8
            else:
                supply_demand = min(12, 8 + int(imbalance / 30))
                intensity = int(self.rng.uniform(1000, 1500))
                priority = 6

        elif scenario in ['emergency', 'emergency_grid_stability']:
            intensity = int(self.rng.uniform(3500, 4095))
            supply_demand = 15
            priority = 15

        elif scenario == 'emergency_supply_shortage':
            if imbalance > 200:
                intensity = int(self.rng.uniform(3000, 4095))
                supply_demand = 14
                priority = 14
            elif imbalance > 100:
                intensity = int(self.rng.uniform(2500, 3500))
                supply_demand = 12
                priority = 12
            else:
                intensity = int(self.rng.uniform(2000, 3000))
                supply_demand = 10
                priority = 10

        else:  # 'normal' or any other
            supply_demand = state['supply_demand']

            if imbalance > 200:
                intensity = int(self.rng.uniform(3000, 4095))
                priority = 12
            elif imbalance > 100:
                intensity = int(self.rng.uniform(2000, 3000))
                priority = 10
            elif imbalance > 50:
                intensity = int(self.rng.uniform(1500, 2000))
                priority = 8
            else:
                intensity = int(self.rng.uniform(800, 1500))
                priority = 6

        # Apply override if set (for external closed-loop control)
        if self._override_signal is not None:
            if self._override_signal.get('intensity') is not None:
                intensity = self._override_signal['intensity']
            if self._override_signal.get('supply_demand') is not None:
                supply_demand = self._override_signal['supply_demand']

        # Generate signal for each region
        for region_id in range(num_regions):
            regional_intensity = max(0, min(4095,
                intensity + int(self.rng.normal(0, 200))))
            regional_sd = max(0, min(15,
                supply_demand + int(self.rng.normal(0, 1))))

            signal = self.generate_signal(
                region_id=region_id,
                supply_demand=regional_sd,
                intensity=regional_intensity,
                priority=priority,
            )
            signals.append(signal)

        return signals


class EPSSimulator:
    """
    Agent-based simulation framework for broadcast-coordinated energy storage.

    Implements full agent-based simulation (Level 1) where each device
    independently executes eligibility screening, response probability
    computation, and Bernoulli sampling based on the broadcast signal
    and local state.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

        self._population: Optional[DevicePopulation] = None
        self._signal_generator: Optional[SignalGenerator] = None
        self._is_initialized = False

        # Correlation state (per-step, set in _run_level1_agent)
        self._region_perturbation: Optional[Dict[int, float]] = None

        # Metrics collection
        self._step_metrics: List[Dict] = []

    def initialize(self) -> 'EPSSimulator':
        """Initialize simulation environment."""
        logger.info(f"Initializing simulator with {self.config.num_devices} devices")

        # Generate device population
        pop_gen = PopulationGenerator(self.config, self.rng)
        self._population = pop_gen.generate()

        # Initialize signal generator
        self._signal_generator = SignalGenerator(self.config, self.rng)

        self._is_initialized = True
        logger.info(f"Initialization complete: {self._population.total_count} devices")
        return self

    def set_override_signal(
        self,
        intensity: Optional[int] = None,
        price_value: Optional[float] = None,
        supply_demand: Optional[int] = None,
    ) -> None:
        """Set override signal. price_value reserved for future use."""
        if not self._is_initialized:
            self.initialize()
        self._signal_generator.set_override_signal(intensity, price_value, supply_demand)

    def clear_override_signal(self) -> None:
        """Clear any override signal."""
        if self._signal_generator:
            self._signal_generator.clear_override_signal()

    def set_start_hour(self, hour: float) -> None:
        """
        Set the simulation start hour.

        Crucial for generating balanced training data for dual-NN architecture:
        - hour=10-14: Solar surplus → charge (positive) responses
        - hour=18-22: Evening peak → discharge (negative) responses

        Args:
            hour: Start hour (0-24)
        """
        if not self._is_initialized:
            self.initialize()
        self._signal_generator.set_start_hour(hour)

    def run(
        self,
        num_steps: Optional[int] = None,
        scenario: str = 'normal',
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SimulationResult:
        """
        Run simulation.

        Args:
            num_steps: Number of time steps (default: based on duration/time_step)
            scenario: Scenario name ('normal', 'peak_shaving', 'valley_filling', 'emergency')
            progress_callback: Optional callback(current_step, total_steps)

        Returns:
            SimulationResult with all metrics
        """
        if not self._is_initialized:
            self.initialize()

        if num_steps is None:
            num_steps = int(self.config.duration_seconds / self.config.time_step)

        result = SimulationResult()
        start_time = datetime.now()

        # Run agent-based simulation (only Level 1 is supported)
        result = self._run_level1_agent(num_steps, scenario, progress_callback)

        # Compute final statistics
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        result.devices_per_second = (self.config.num_devices * num_steps) / max(elapsed, 0.001)
        result.compute_latency_percentiles()

        return result

    def _run_level1_agent(
        self,
        num_steps: int,
        scenario: str,
        progress_callback: Optional[Callable],
    ) -> SimulationResult:
        """Level 1: Full agent-based simulation."""
        result = SimulationResult()

        # Pre-compute PERSISTENT correlated perturbations for the entire run.
        # Environmental conditions (weather, electricity price regime, grid frequency)
        # are persistent over the simulation horizon (hours), not per-step transients.
        # This ensures cross-run CV properly reflects inter-device correlation.
        corr = self.config.correlation
        if corr.enable_correlation:
            persistent_global_shock = self.rng.normal(0, corr.global_shock_std)
            self._region_perturbation = {}
            for r in range(self.config.num_regions):
                self._region_perturbation[r] = (
                    persistent_global_shock
                    + self.rng.normal(0, corr.regional_shock_std)
                )
        else:
            self._region_perturbation = None

        for step in range(num_steps):
            ts = TimeStep(
                step_index=step,
                timestamp=datetime.now() + timedelta(seconds=step * self.config.time_step),
                elapsed_seconds=step * self.config.time_step,
            )

            # Generate signals for this step
            signals = self._signal_generator.generate_scenario_signals(
                scenario, step, self.config.num_regions
            )

            # Store the first signal as representative for this time step
            if signals:
                ts.signal = signals[0]

            # Process each region
            for region_id, signal in enumerate(signals):
                # Pass simulation time for time-based constraints
                region_response = self._process_region_level1(
                    region_id, signal, step, scenario, sim_time=ts.timestamp
                )
                ts.total_response_kw += region_response['total_kw']
                ts.responding_devices += region_response['responding']
                ts.latency_samples.extend(region_response['latencies'])

                # Track by device type
                result.battery_response_kwh += region_response.get('battery_kwh', 0)

            result.time_steps.append(ts)
            result.total_responses += ts.responding_devices

            # Energy accounting
            energy_delta_kwh = ts.total_response_kw * (self.config.time_step / 3600)

            if energy_delta_kwh > 0:  # Charging (valley filling)
                result.charge_energy_kwh += energy_delta_kwh
                result.load_increase_kwh += energy_delta_kwh
            else:  # Discharging (peak shaving)
                result.discharge_energy_kwh += abs(energy_delta_kwh)
                result.load_reduction_kwh += abs(energy_delta_kwh)

            result.net_energy_kwh += energy_delta_kwh
            result.total_energy_kwh = result.net_energy_kwh

            if progress_callback:
                progress_callback(step + 1, num_steps)

        # Compute response rate and responding device count
        total_possible = self.config.num_devices * num_steps
        result.response_rate = result.total_responses / max(total_possible, 1)
        result.n_responding_devices = result.total_responses // max(num_steps, 1)

        return result

    def _process_region_level1(
        self,
        region_id: int,
        signal: EPSSignal,
        step: int,
        scenario: str = 'normal',
        sim_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Process all devices in a region for Level 1 simulation."""
        result = {
            'total_kw': 0.0,
            'responding': 0,
            'latencies': [],
            'battery_kwh': 0.0,
        }

        devices = self._population.get_devices_by_region(region_id)

        # Process batteries
        for idx, battery in devices['batteries']:
            response = self._simulate_battery_response(
                battery, signal, step, scenario, device_id=f'battery_{idx}')
            if response['responded']:
                result['responding'] += 1
                result['total_kw'] += response['power_kw']
                result['latencies'].append(response['latency_ms'])
                result['battery_kwh'] += abs(response['power_kw']) * (self.config.time_step / 3600)

        return result

    # Device response probability: p = |s| * w(SOC), where s = intensity/4095

    def _simulate_battery_response(
        self,
        battery: BatteryModel,
        signal: EPSSignal,
        step: int,
        scenario: str = 'normal',
        device_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Simulate individual battery response — Scheme A: Bernoulli + full commitment.

        Three-step decision model (aligned with system description document):
          Step 1: α_i ∈ {0,1} — Safety eligibility screening
          Step 2: C_i^avail = min(P_rated_eff, E_headroom/Δt)
          Step 3a: p_i = |s| × w(SOC_i) — Bernoulli response probability
          Step 3b: P_i = sign(s) × C_avail × (1+ξ) — Full power commitment

        E[P_agg] = s × Σ w(SOC_i) × C_avail,i  (≈ linear in s)
        Var comes from Bernoulli → CV ∝ 1/√N → N* threshold exists
        """
        # ─── Communication layer: LEO satellite packet loss ───
        if self.rng.random() < self.config.eps_packet_loss_rate:
            return {'responded': False, 'power_kw': 0, 'latency_ms': 0,
                    'reason': 'packet_loss'}

        # ─── BMS/inverter random unavailability (firmware update/maintenance/fault) ───
        # Industrial measurement: 1-2% unavailability
        if self.rng.random() < self.config.device_offline_rate:
            return {'responded': False, 'power_kw': 0, 'latency_ms': 0,
                    'reason': 'device_offline'}

        # ─── Step 1: Eligibility screening α_i ∈ {0, 1} ───
        # BMS real-time safety check
        # SOC boundaries are per-device parameters (randomized at creation):
        #   soc_reserve ~ U(0.1, 0.2), soc_max ~ U(0.8, 0.95)
        soc = battery.soc
        # Model mismatch: SOC observation noise
        # True SOC used for state update, noisy SOC for decision-making
        soc_observed = soc
        if self.config.soc_noise_std > 0:
            soc_observed = float(np.clip(
                soc + self.rng.normal(0, self.config.soc_noise_std), 0.0, 1.0))
        is_discharge = signal.supply_demand >= 8

        soc_reserve = getattr(battery.params, 'soc_reserve', 0.2)
        soc_max = getattr(battery.params, 'soc_max', 0.95)

        if is_discharge:
            qualified = soc_observed > soc_reserve
        else:
            qualified = soc_observed < soc_max

        if not qualified:
            return {'responded': False, 'power_kw': 0, 'latency_ms': 0,
                    'reason': 'eligibility_fail'}

        # ─── Step 2: Available capacity C_i^avail ───
        # C_avail = min(P_rated_effective, E_headroom / Δt)
        # P_rated_effective from BatteryModel (includes C-rate/SOC derating/temp derating)
        dt_hours = self.config.time_step / 3600.0
        e_available = battery.available_capacity_kwh  # SOH-adjusted

        if is_discharge:
            p_rated_eff = battery.get_max_discharge_power()
            e_headroom_kw = max(0, (soc_observed - soc_reserve) * e_available / dt_hours)
        else:
            p_rated_eff = battery.get_max_charge_power()
            e_headroom_kw = max(0, (soc_max - soc_observed) * e_available / dt_hours)

        c_avail = min(p_rated_eff, e_headroom_kw)

        if c_avail < 0.01:  # Minimum controllable power 10W (inverter deadband)
            return {'responded': False, 'power_kw': 0, 'latency_ms': 0,
                    'reason': 'insufficient_capacity'}

        # ─── Step 3a: Response probability p_i = |s| × w(SOC) ───
        # s = intensity/4095 ∈ [0,1]: signal score magnitude
        # w(SOC): SOC willingness weight, computed using per-device boundaries
        score = signal.intensity / 4095.0

        soc_range = soc_max - soc_reserve
        if is_discharge:
            w_soc = max(0, min(1, (soc_observed - soc_reserve) / max(soc_range, 0.1)))
        else:
            w_soc = max(0, min(1, (soc_max - soc_observed) / max(soc_range, 0.1)))

        prob = score * w_soc  # p ∈ [0, 1]

        # ─── Sigmoid response (M1 structural mismatch — changes g(s) shape) ───
        # Replaces linear p = |s|*w with sigmoid, altering E[P_agg|s] shape
        if self.config.sigmoid_response:
            k = self.config.sigmoid_steepness
            # sigmoid centered at 0.5: p_sigmoid = 1/(1+exp(-k*(p_linear - 0.5)))
            prob = float(1.0 / (1.0 + np.exp(-k * (prob - 0.5))))

        # Model mismatch: response probability bias
        if self.config.response_prob_bias != 1.0:
            prob = float(np.clip(prob * self.config.response_prob_bias, 0.0, 1.0))

        # ─── Continuous proportional response (structural mismatch M1) ───
        # When continuous_response=True, skip Bernoulli and output prob-scaled power.
        # Same E[P_agg] as Bernoulli (E[Bernoulli(p)*C] = p*C), but lower variance
        # because there is no binary on/off noise — only delivery noise remains.
        if self.config.continuous_response:
            inverter_err = self.rng.normal(0, 0.025)
            voltage_err = self.rng.normal(0, 0.030)
            eta_delivery = 1.0 + inverter_err + voltage_err
            power_kw = prob * c_avail * eta_delivery * (-1 if is_discharge else 1)

            # Update battery state
            battery.update(power_kw=power_kw, dt_seconds=self.config.time_step)

            latency_ms = EPS_LATENCY.base_ms + self.rng.exponential(
                EPS_LATENCY.jitter_scale_ms)
            return {'responded': True, 'power_kw': power_kw, 'latency_ms': latency_ms}

        # Bernoulli sampling — the core randomness source for LLN
        responded = self.rng.random() < prob

        if not responded:
            return {'responded': False, 'power_kw': 0, 'latency_ms': 0,
                    'reason': 'bernoulli_reject'}

        # ─── Step 3b: Response power P_i = sign(s) × C_avail × η_delivery ───
        # Full commitment: once responding, output full C_avail
        # Composite power delivery noise (dual-source, traceable):
        #   Source 1: Inverter tracking accuracy ±2.5%
        #   Source 2: Grid voltage fluctuation ±3.0%
        #   σ_total = √(0.025² + 0.030²) ≈ 3.9%
        inverter_err = self.rng.normal(0, 0.025)
        voltage_err = self.rng.normal(0, 0.030)
        eta_delivery = 1.0 + inverter_err + voltage_err

        if is_discharge:
            power_kw = -c_avail * eta_delivery  # Discharge is negative
        else:
            power_kw = c_avail * eta_delivery   # Charge is positive

        # Regional environmental perturbation (power amplitude modulation)
        if self._region_perturbation is not None and device_id is not None:
            region = self._population.region_assignments.get(device_id)
            if region is not None:
                perturbation = self._region_perturbation.get(region, 0.0)
                power_kw *= (1.0 + perturbation * 0.15)

        # Update battery state (includes efficiency, thermal model, SOH degradation)
        battery.update(power_kw=power_kw, dt_seconds=self.config.time_step)

        # Communication latency (LEO satellite base 33ms + exponential jitter)
        latency_ms = EPS_LATENCY.base_ms + self.rng.exponential(
            EPS_LATENCY.jitter_scale_ms)

        return {'responded': True, 'power_kw': power_kw, 'latency_ms': latency_ms}

    # Level 2/3 simulation methods removed (mean-field and extrapolation).
    # Only Level 1 agent-based simulation is used.

    def run_with_mask(
        self,
        signal_override: Dict[str, Any],
        device_mask: Dict[str, bool],
        num_steps: Optional[int] = None,
        scenario: str = 'normal',
    ) -> SimulationResult:
        """
        Run simulation with a device coverage mask.

        Only devices where device_mask[device_id] == True will receive the signal.
        This enables unified baseline comparison: all methods use the same physics
        engine, with differences arising solely from communication coverage.

        Args:
            signal_override: Signal parameters (intensity, price_value, supply_demand)
            device_mask: Dict mapping device_id → bool (True = receives signal)
            num_steps: Number of time steps
            scenario: Scenario name

        Returns:
            SimulationResult for covered devices only
        """
        if not self._is_initialized:
            self.initialize()

        if num_steps is None:
            num_steps = int(self.config.duration_seconds / self.config.time_step)

        # Set override signal
        self.set_override_signal(
            intensity=signal_override.get('intensity'),
            price_value=signal_override.get('price_value') or signal_override.get('price'),
            supply_demand=signal_override.get('supply_demand'),
        )

        result = SimulationResult()

        # Pre-compute PERSISTENT correlated perturbations for entire run
        corr = self.config.correlation
        if corr.enable_correlation:
            persistent_global_shock = self.rng.normal(0, corr.global_shock_std)
            self._region_perturbation = {}
            for r in range(self.config.num_regions):
                self._region_perturbation[r] = (
                    persistent_global_shock
                    + self.rng.normal(0, corr.regional_shock_std)
                )
        else:
            self._region_perturbation = None

        for step in range(num_steps):
            ts = TimeStep(
                step_index=step,
                timestamp=datetime.now() + timedelta(seconds=step * self.config.time_step),
                elapsed_seconds=step * self.config.time_step,
            )

            signals = self._signal_generator.generate_scenario_signals(
                scenario, step, self.config.num_regions
            )

            if signals:
                ts.signal = signals[0]

            # Process each region, but only masked devices receive EPS signal.
            # Uncovered devices execute self-interested default behavior.
            for region_id, signal in enumerate(signals):
                devices = self._population.get_devices_by_region(region_id)

                for idx, battery in devices['batteries']:
                    did = f'battery_{idx}'
                    if device_mask.get(did, False):
                        response = self._simulate_battery_response(
                            battery, signal, step, scenario, device_id=did)
                        if response['responded']:
                            ts.responding_devices += 1
                            ts.total_response_kw += response['power_kw']
                            ts.latency_samples.append(response['latency_ms'])
                            result.battery_response_kwh += abs(response['power_kw']) * (self.config.time_step / 3600)
                    else:
                        # Default behavior: self-interested charging when SOC low and cheap
                        if battery._soc < 0.5 and self.rng.random() < 0.3:
                            charge_kw = battery.get_max_charge_power() * 0.3
                            battery.update(power_kw=charge_kw, dt_seconds=self.config.time_step)

            result.time_steps.append(ts)
            result.total_responses += ts.responding_devices

            energy_delta_kwh = ts.total_response_kw * (self.config.time_step / 3600)
            if energy_delta_kwh > 0:
                result.charge_energy_kwh += energy_delta_kwh
                result.load_increase_kwh += energy_delta_kwh
            else:
                result.discharge_energy_kwh += abs(energy_delta_kwh)
                result.load_reduction_kwh += abs(energy_delta_kwh)
            result.net_energy_kwh += energy_delta_kwh
            result.total_energy_kwh = result.net_energy_kwh

        self.clear_override_signal()

        # Compute response rate (based on masked devices only)
        n_masked = sum(1 for v in device_mask.values() if v)
        total_possible = n_masked * num_steps
        result.response_rate = result.total_responses / max(total_possible, 1)
        result.compute_latency_percentiles()

        return result

    @property
    def population(self) -> Optional[DevicePopulation]:
        """Access population for external use (e.g., coverage mask generation)."""
        return self._population

    def reset(self) -> None:
        """Reset simulation state."""
        self._population = None
        self._signal_generator = None
        self._step_metrics = []
        self._region_perturbation = None
        self._is_initialized = False

    def get_population_summary(self) -> Dict[str, Any]:
        """Get summary of current device population."""
        if not self._population:
            return {}

        return {
            'total_devices': self._population.total_count,
            'batteries': len(self._population.batteries),
            'regions': self.config.num_regions,
        }
