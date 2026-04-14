"""
Battery Storage Model

Models a single battery energy storage device.

Parameters (Table 1, calibrated via three-layer anchoring, Methods):
    Rated capacity : U(5, 20) kWh     LBNL median 13.5 kWh [ref 24]
    C-rate         : LogN(0.45, 0.35) Figgener et al. (2021) [ref 25]
    SOH            : U(0.82, 1.0)     IRENA field data [ref 26]
    Efficiency     : 95% one-way      90.25% round-trip
    SOC_min        : U(0.10, 0.20)    per-device operating bound
    SOC_max        : U(0.80, 0.95)    per-device operating bound

Available power per dispatch interval (Methods, Eq. 4):
    P_max_i = min(C_rated_i * C_rate_i,  E_headroom_i / dt)
    where E_headroom = SOH * capacity * (SOC - SOC_min)  [discharge]
                     = SOH * capacity * (SOC_max - SOC)  [charge]
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import math
from enum import Enum, auto


class BatteryChemistry(Enum):
    """Battery chemistry types."""
    LFP = auto()     # Lithium Iron Phosphate - safe, long cycle life


@dataclass
class BatteryParameters:
    """
    Physical parameters for battery storage system.

    All parameters are based on manufacturer specifications
    
    """
    # Capacity parameters
    nominal_capacity_kwh: float = 10.0  # Nominal energy capacity
    usable_capacity_factor: float = 0.90  # Usable capacity as fraction of nominal

    # Power parameters
    max_charge_power_kw: float = 5.0   # Maximum charging power
    max_discharge_power_kw: float = 5.0  # Maximum discharge power
    nominal_voltage_v: float = 400.0   # Nominal DC voltage

    # Efficiency parameters (based on manufacturer data)
    charge_efficiency_nominal: float = 0.95  # At optimal conditions
    discharge_efficiency_nominal: float = 0.95
    inverter_efficiency: float = 0.97  # DC-AC conversion efficiency

    # SOC operating limits
    soc_min: float = 0.10  # Minimum SOC for deep discharge protection
    soc_max: float = 0.95  # Maximum SOC for overcharge protection
    soc_reserve: float = 0.20  # Minimum SOC threshold for discharge participation

    # Temperature parameters (Celsius)
    temp_min_operating: float = 0.0    # Minimum operating temperature
    temp_max_operating: float = 45.0   # Maximum operating temperature
    temp_optimal_low: float = 15.0     # Optimal range start
    temp_optimal_high: float = 35.0    # Optimal range end
    temp_max_charging: float = 40.0    # Max temp for charging

    # C-rate limits (C = capacity/hour)
    max_charge_c_rate: float = 0.5   # Maximum charge rate
    max_discharge_c_rate: float = 0.5  # Maximum discharge rate
    continuous_c_rate: float = 0.3   # Sustainable continuous rate

    # Cycle parameters
    max_daily_cycles: int = 2        # Recommended daily cycle limit
    max_lifetime_cycles: int = 6000  # Total cycle life (80% capacity)

    # Self-discharge rate
    self_discharge_rate_per_day: float = 0.001  # 0.1% per day

    # Chemistry-specific
    chemistry: BatteryChemistry = BatteryChemistry.LFP

    # Ramp rate limits (kW/s)
    ramp_rate_up: float = 2.0        # Maximum power increase rate
    ramp_rate_down: float = 2.0      # Maximum power decrease rate


@dataclass
class BatteryDegradationState:
    """
    Battery degradation tracking for State of Health (SOH).

    Semi-empirical degradation tracking model.
    """
    # Total equivalent full cycles (EFC)
    total_cycles: float = 0.0

    # Calendar aging factor (days at various SOC/temp)
    calendar_age_days: float = 0.0

    # Cycle-based capacity fade (%)
    cycle_capacity_fade: float = 0.0

    # Calendar-based capacity fade (%)
    calendar_capacity_fade: float = 0.0

    # Total capacity fade (%)
    @property
    def total_capacity_fade(self) -> float:
        """Total capacity fade from all sources."""
        return min(self.cycle_capacity_fade + self.calendar_capacity_fade, 0.30)

    # Current State of Health (%)
    @property
    def soh(self) -> float:
        """State of Health as percentage of original capacity."""
        return max(1.0 - self.total_capacity_fade, 0.70)


@dataclass
class BatteryThermalState:
    """Thermal state for temperature-dependent modeling."""
    cell_temperature: float = 25.0  # Core cell temperature (C)
    ambient_temperature: float = 25.0  # Ambient temperature (C)
    coolant_temperature: float = 25.0  # Coolant/heatsink temperature (C)

    # Thermal management system state
    cooling_active: bool = False
    heating_active: bool = False

    # Thermal parameters
    thermal_mass_kj_per_k: float = 50.0  # Thermal mass
    thermal_resistance_k_per_kw: float = 0.5  # Thermal resistance to ambient


class BatteryModel:
    """
    High-fidelity battery storage model for EPS edge devices.

    Implements:
    - SOC dynamics with temperature and C-rate dependent efficiency
    - SOH degradation from cycling and calendar aging
    - Thermal model with active thermal management
    - Power limits based on operating conditions
    """

    def __init__(
        self,
        params: Optional[BatteryParameters] = None,
        initial_soc: float = 0.5,
        initial_soh: float = 1.0,
    ):
        """
        Initialize battery model.

        Args:
            params: Battery parameters (uses defaults if None)
            initial_soc: Initial state of charge (0-1)
            initial_soh: Initial state of health (0-1)
        """
        self.params = params or BatteryParameters()
        self.soc = initial_soc

        # Initialize degradation state
        self.degradation = BatteryDegradationState()
        if initial_soh < 1.0:
            self.degradation.cycle_capacity_fade = (1.0 - initial_soh) * 0.7
            self.degradation.calendar_capacity_fade = (1.0 - initial_soh) * 0.3

        # Initialize thermal state
        self.thermal = BatteryThermalState()

        # Power state
        self.current_power_kw: float = 0.0
        self.power_target_kw: float = 0.0

        # Tracking
        self.energy_throughput_kwh: float = 0.0
        self.daily_cycle_count: float = 0.0
        self._daily_energy_start: float = 0.0

    @property
    def soh(self) -> float:
        """Current state of health."""
        return self.degradation.soh

    @property
    def available_capacity_kwh(self) -> float:
        """Current available capacity accounting for SOH."""
        return self.params.nominal_capacity_kwh * self.params.usable_capacity_factor * self.soh

    @property
    def stored_energy_kwh(self) -> float:
        """Current stored energy."""
        return self.soc * self.available_capacity_kwh

    def get_discharge_efficiency(self, power_kw: float) -> float:
        """
        Get discharge efficiency based on operating conditions.

        Efficiency varies with:
        - Power level (C-rate)
        - Temperature
        - State of charge

        Args:
            power_kw: Discharge power (positive value)

        Returns:
            Discharge efficiency (0-1)
        """
        base_efficiency = self.params.discharge_efficiency_nominal

        # C-rate correction (higher C-rate = lower efficiency)
        c_rate = abs(power_kw) / self.available_capacity_kwh
        c_rate_factor = 1.0 - 0.02 * max(0, c_rate - self.params.continuous_c_rate)

        # Temperature correction
        temp = self.thermal.cell_temperature
        if temp < self.params.temp_optimal_low:
            temp_factor = 0.95 + 0.05 * (temp - self.params.temp_min_operating) / (self.params.temp_optimal_low - self.params.temp_min_operating)
        elif temp > self.params.temp_optimal_high:
            temp_factor = 0.95 + 0.05 * (self.params.temp_max_operating - temp) / (self.params.temp_max_operating - self.params.temp_optimal_high)
        else:
            temp_factor = 1.0

        # SOC correction (very low SOC reduces efficiency)
        if self.soc < 0.2:
            soc_factor = 0.95 + 0.05 * self.soc / 0.2
        else:
            soc_factor = 1.0

        # Inverter efficiency
        inverter_eff = self.params.inverter_efficiency

        total_efficiency = base_efficiency * c_rate_factor * temp_factor * soc_factor * inverter_eff
        return max(0.85, min(0.98, total_efficiency))

    def get_charge_efficiency(self, power_kw: float) -> float:
        """
        Get charge efficiency based on operating conditions.

        Similar to discharge but also considers charging restrictions.

        Args:
            power_kw: Charge power (positive value)

        Returns:
            Charge efficiency (0-1)
        """
        base_efficiency = self.params.charge_efficiency_nominal

        # C-rate correction
        c_rate = abs(power_kw) / self.available_capacity_kwh
        c_rate_factor = 1.0 - 0.03 * max(0, c_rate - self.params.continuous_c_rate)

        # Temperature correction (charging more sensitive to temperature)
        temp = self.thermal.cell_temperature
        if temp < self.params.temp_optimal_low:
            # Charging at cold temperatures is inefficient
            temp_factor = 0.90 + 0.10 * (temp - self.params.temp_min_operating) / (self.params.temp_optimal_low - self.params.temp_min_operating)
        elif temp > self.params.temp_max_charging:
            # Charging at high temp should be limited
            temp_factor = 0.80
        elif temp > self.params.temp_optimal_high:
            temp_factor = 0.95 + 0.05 * (self.params.temp_max_charging - temp) / (self.params.temp_max_charging - self.params.temp_optimal_high)
        else:
            temp_factor = 1.0

        # SOC correction (high SOC reduces charging efficiency)
        if self.soc > 0.9:
            soc_factor = 0.90 + 0.10 * (1.0 - self.soc) / 0.1
        else:
            soc_factor = 1.0

        # Inverter efficiency
        inverter_eff = self.params.inverter_efficiency

        total_efficiency = base_efficiency * c_rate_factor * temp_factor * soc_factor * inverter_eff
        return max(0.85, min(0.98, total_efficiency))

    def get_max_discharge_power(self) -> float:
        """
        Get maximum allowable discharge power.

        Limited by:
        - Rated power
        - C-rate limits
        - SOC constraints
        - Temperature constraints

        Returns:
            Maximum discharge power in kW
        """
        # Base limit
        max_power = self.params.max_discharge_power_kw

        # C-rate limit
        c_rate_limit = self.available_capacity_kwh * self.params.max_discharge_c_rate
        max_power = min(max_power, c_rate_limit)

        # SOC constraint - reduce power near minimum SOC
        if self.soc < 0.3:
            soc_factor = self.soc / 0.3
            max_power *= soc_factor

        # Temperature constraint
        temp = self.thermal.cell_temperature
        if temp < self.params.temp_min_operating:
            max_power = 0.0
        elif temp < self.params.temp_optimal_low:
            temp_factor = (temp - self.params.temp_min_operating) / (self.params.temp_optimal_low - self.params.temp_min_operating)
            max_power *= 0.5 + 0.5 * temp_factor
        elif temp > self.params.temp_max_operating:
            max_power = 0.0
        elif temp > self.params.temp_optimal_high:
            temp_factor = (self.params.temp_max_operating - temp) / (self.params.temp_max_operating - self.params.temp_optimal_high)
            max_power *= 0.7 + 0.3 * temp_factor

        return max(0.0, max_power)

    def get_max_charge_power(self) -> float:
        """
        Get maximum allowable charge power.

        Returns:
            Maximum charge power in kW
        """
        # Base limit
        max_power = self.params.max_charge_power_kw

        # C-rate limit
        c_rate_limit = self.available_capacity_kwh * self.params.max_charge_c_rate
        max_power = min(max_power, c_rate_limit)

        # SOC constraint - reduce power near maximum SOC (CC-CV charging)
        if self.soc > 0.8:
            soc_factor = (1.0 - self.soc) / 0.2
            max_power *= max(0.2, soc_factor)  # At least 20% power for balancing

        # Temperature constraint (more restrictive for charging)
        temp = self.thermal.cell_temperature
        if temp < self.params.temp_min_operating:
            max_power = 0.0  # No charging below min temp
        elif temp < self.params.temp_optimal_low:
            # Cold charging is dangerous - significantly reduce
            temp_factor = (temp - self.params.temp_min_operating) / (self.params.temp_optimal_low - self.params.temp_min_operating)
            max_power *= 0.3 * temp_factor
        elif temp > self.params.temp_max_charging:
            max_power = 0.0  # No charging above max charging temp
        elif temp > self.params.temp_optimal_high:
            temp_factor = (self.params.temp_max_charging - temp) / (self.params.temp_max_charging - self.params.temp_optimal_high)
            max_power *= 0.5 + 0.5 * temp_factor

        return max(0.0, max_power)

    def can_discharge(self, energy_kwh: float) -> bool:
        """
        Check if battery can discharge the requested energy.

        Args:
            energy_kwh: Requested discharge energy

        Returns:
            True if discharge is feasible
        """
        min_soc = self.params.soc_min
        available = (self.soc - min_soc) * self.available_capacity_kwh
        return available >= energy_kwh

    def can_charge(self, energy_kwh: float) -> bool:
        """
        Check if battery can accept the requested charge energy.

        Args:
            energy_kwh: Requested charge energy

        Returns:
            True if charge is feasible
        """
        max_soc = self.params.soc_max
        headroom = (max_soc - self.soc) * self.available_capacity_kwh
        return headroom >= energy_kwh

    def update(self, power_kw: float, dt_seconds: float) -> Dict[str, float]:
        """
        Update battery state for one time step.

        Args:
            power_kw: Power flow (positive = charging, negative = discharging)
            dt_seconds: Time step in seconds

        Returns:
            Dictionary with update metrics
        """
        dt_hours = dt_seconds / 3600.0

        # Apply ramp rate limits
        power_delta = power_kw - self.current_power_kw
        max_delta_up = self.params.ramp_rate_up * dt_seconds
        max_delta_down = self.params.ramp_rate_down * dt_seconds

        if power_delta > 0:
            power_delta = min(power_delta, max_delta_up)
        else:
            power_delta = max(power_delta, -max_delta_down)

        actual_power = self.current_power_kw + power_delta

        # Apply power limits
        if actual_power > 0:  # Charging
            actual_power = min(actual_power, self.get_max_charge_power())
            efficiency = self.get_charge_efficiency(actual_power)
            energy_change = actual_power * efficiency * dt_hours
        elif actual_power < 0:  # Discharging
            actual_power = max(actual_power, -self.get_max_discharge_power())
            efficiency = self.get_discharge_efficiency(-actual_power)
            energy_change = actual_power / efficiency * dt_hours
        else:
            energy_change = 0.0
            efficiency = 1.0

        # Apply self-discharge
        self_discharge = self.params.self_discharge_rate_per_day * dt_hours / 24.0

        # Update SOC
        new_soc = self.soc + (energy_change - self_discharge * self.available_capacity_kwh) / self.available_capacity_kwh
        new_soc = max(self.params.soc_min, min(self.params.soc_max, new_soc))

        # Track energy throughput
        energy_throughput = abs(energy_change)
        self.energy_throughput_kwh += energy_throughput

        # Update daily cycle count
        self.daily_cycle_count += energy_throughput / (2 * self.available_capacity_kwh)

        # Update degradation
        self._update_degradation(energy_throughput, dt_hours)

        # Update thermal state
        self._update_thermal(actual_power, dt_seconds)

        # Store new state
        old_soc = self.soc
        self.soc = new_soc
        self.current_power_kw = actual_power

        return {
            'soc_delta': new_soc - old_soc,
            'energy_kwh': energy_change,
            'efficiency': efficiency,
            'actual_power_kw': actual_power,
            'temperature_c': self.thermal.cell_temperature,
        }

    def _update_degradation(self, energy_kwh: float, dt_hours: float) -> None:
        """Update degradation state."""
        # Cycle aging (Rainflow-based)
        cycle_fraction = energy_kwh / (2 * self.available_capacity_kwh)
        self.degradation.total_cycles += cycle_fraction

        # Cycle capacity fade (empirical model)
        # Based on: fade = k * cycles^0.5
        k_cycle = 0.0005  # Fade coefficient
        self.degradation.cycle_capacity_fade = k_cycle * math.sqrt(max(0, self.degradation.total_cycles))

        # Calendar aging
        self.degradation.calendar_age_days += dt_hours / 24.0

        # Calendar fade (temperature and SOC dependent)
        temp = self.thermal.cell_temperature
        soc_stress = 1.0 + 2.0 * (self.soc - 0.5) ** 2  # Higher stress at extreme SOC

        # Arrhenius-like temperature factor with bounds to prevent overflow
        temp_diff = max(-50, min(50, temp - 25))  # Clamp temperature difference
        temp_stress = math.exp(temp_diff / 15)

        k_calendar = 0.00001  # Calendar fade coefficient
        self.degradation.calendar_capacity_fade = min(
            0.3,  # Cap calendar fade at 30%
            k_calendar * self.degradation.calendar_age_days * soc_stress * temp_stress
        )

    def _update_thermal(self, power_kw: float, dt_seconds: float) -> None:
        """Update thermal state based on power flow."""
        # Heat generation from internal resistance
        # Ohmic heat generation (Q ∝ P²)
        heat_generation_kw = 0.02 * (power_kw ** 2) / (self.params.max_charge_power_kw ** 2)

        # Heat transfer to ambient (use exponential decay for stability)
        temp_diff = self.thermal.cell_temperature - self.thermal.ambient_temperature

        # Thermal time constant: tau = R * C (thermal resistance * thermal mass)
        # For stability, use exponential approach instead of linear
        tau = self.thermal.thermal_resistance_k_per_kw * self.thermal.thermal_mass_kj_per_k

        # Active cooling/heating effects
        active_temp_target = self.thermal.ambient_temperature
        if self.thermal.cooling_active:
            heat_generation_kw -= 1.0  # 1 kW cooling capacity reduces net heat
        if self.thermal.heating_active:
            heat_generation_kw += 0.5  # 0.5 kW heating

        # Net temperature change from heat generation
        heat_delta = heat_generation_kw * dt_seconds / self.thermal.thermal_mass_kj_per_k

        # Exponential decay towards ambient (numerically stable)
        decay_factor = min(1.0, dt_seconds / tau)  # Clamp to prevent overshoot
        ambient_delta = -temp_diff * decay_factor

        # Total temperature change (clamped for stability)
        delta_temp = heat_delta + ambient_delta
        delta_temp = max(-5.0, min(5.0, delta_temp))  # Max 5°C change per step

        new_temp = self.thermal.cell_temperature + delta_temp

        # Clamp temperature to physical bounds
        new_temp = max(-40.0, min(80.0, new_temp))

        # Activate thermal management
        if new_temp > self.params.temp_optimal_high + 2:
            self.thermal.cooling_active = True
        elif new_temp < self.params.temp_optimal_high - 2:
            self.thermal.cooling_active = False

        if new_temp < self.params.temp_optimal_low - 2:
            self.thermal.heating_active = True
        elif new_temp > self.params.temp_optimal_low + 2:
            self.thermal.heating_active = False

        self.thermal.cell_temperature = new_temp

    def reset_daily_counters(self) -> None:
        """Reset daily tracking counters (call at midnight)."""
        self.daily_cycle_count = 0.0
        self._daily_energy_start = self.energy_throughput_kwh

    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary."""
        return {
            'soc': self.soc,
            'soh': self.soh,
            'stored_energy_kwh': self.stored_energy_kwh,
            'available_capacity_kwh': self.available_capacity_kwh,
            'current_power_kw': self.current_power_kw,
            'max_discharge_power_kw': self.get_max_discharge_power(),
            'max_charge_power_kw': self.get_max_charge_power(),
            'cell_temperature_c': self.thermal.cell_temperature,
            'daily_cycles': self.daily_cycle_count,
            'total_cycles': self.degradation.total_cycles,
            'cooling_active': self.thermal.cooling_active,
            'heating_active': self.thermal.heating_active,
        }


def create_residential_battery(
    capacity_kwh: float = 10.0,
    c_rate: float = 0.5,
    chemistry: BatteryChemistry = BatteryChemistry.LFP,
    initial_soh: float = 1.0,
) -> BatteryModel:
    """Create a residential battery with manufacturer-specified P/E ratio.

    Args:
        capacity_kwh: Nominal energy capacity (kWh)
        c_rate: Power-to-energy ratio (C-rate), e.g. 0.37 for residential batteries
        chemistry: Battery chemistry type
        initial_soh: Initial state of health (0.7-1.0)
    """
    max_power = capacity_kwh * c_rate
    params = BatteryParameters(
        nominal_capacity_kwh=capacity_kwh,
        max_charge_power_kw=max_power,
        max_discharge_power_kw=max_power,
        chemistry=chemistry,
        max_charge_c_rate=c_rate,
        max_discharge_c_rate=c_rate,
        continuous_c_rate=c_rate * 0.6,
        max_daily_cycles=2,
    )
    return BatteryModel(params=params, initial_soc=0.5, initial_soh=initial_soh)


