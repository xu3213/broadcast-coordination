"""
Simulation Constants

LEO satellite latency model and battery response parameters.
All values traceable to published sources (see inline references).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EPSLatencyParams:
    """
    LEO satellite end-to-end broadcast latency.

    Path: Dispatch Centre -> Ground Station -> LEO (550 km) -> Edge Device

    Component breakdown (one-way):
        Dispatch to ground station:   ~2 ms   (fibre backhaul)
        Ground station processing:    ~3 ms   (encoding, scheduling)
        Ground-to-LEO uplink:       ~12 ms   (550 km alt, ~800 km slant)
        LEO on-board processing:      ~2 ms   (relay, beam forming)
        LEO-to-edge downlink:        ~12 ms   (slant range)
        Edge device processing:       ~2 ms   (decode, parse)
        Total base:                  ~33 ms
        + Exponential jitter:         scale = 8 ms
        => P50 ~ 38 ms,  P99 ~ 70 ms

    References:
        Starlink FCC filings; Michel et al. (2022) IMC; SpaceX specifications.
    """
    base_ms: float = 33.0
    jitter_scale_ms: float = 8.0
    min_ms: float = 20.0
    max_ms: float = 100.0
    target_p99_ms: float = 70.0


@dataclass(frozen=True)
class BatteryResponseParams:
    """Battery SOC boundaries and response probability bounds."""
    soc_charge_max: float = 0.95
    soc_charge_range: float = 0.70
    soc_discharge_min: float = 0.20
    soc_discharge_range: float = 0.60
    prob_min: float = 0.05
    prob_max: float = 0.95


# Singleton instances (imported by simulator.py)
EPS_LATENCY = EPSLatencyParams()
BATTERY_RESPONSE = BatteryResponseParams()
