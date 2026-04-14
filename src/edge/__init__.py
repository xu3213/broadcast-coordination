"""
Edge Module — Battery Model and State Machine

Battery-only mode: all devices are electrochemical storage.
Each device makes an autonomous Bernoulli response decision
based on SOC and signal intensity.
"""

from .state_machine import (
    DeviceStateMachine,
    DeviceState,
    BatteryState,
    Action,
)
from .constraints import (
    ConstraintViolation,
    ConstraintResult,
    ConstraintChecker,
    PhysicalConstraints,
    GridStandard,
    GridConstraints,
    GridState,
    HarmonicConstraints,
    check_discharge_feasibility,
    check_charge_feasibility,
)
from .battery import (
    BatteryChemistry,
    BatteryParameters,
    BatteryDegradationState,
    BatteryThermalState,
    BatteryModel,
    create_residential_battery,
)

__all__ = [
    "DeviceStateMachine",
    "DeviceState",
    "BatteryState",
    "Action",
    "ConstraintViolation",
    "ConstraintResult",
    "ConstraintChecker",
    "PhysicalConstraints",
    "GridStandard",
    "GridConstraints",
    "GridState",
    "HarmonicConstraints",
    "check_discharge_feasibility",
    "check_charge_feasibility",
    "BatteryChemistry",
    "BatteryParameters",
    "BatteryDegradationState",
    "BatteryThermalState",
    "BatteryModel",
    "create_residential_battery",
]
