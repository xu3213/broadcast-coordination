"""
Device State Definitions

Defines BatteryState (SOC, capacity, C-rate, SOH) used by the
simulator to track per-device physical state across time steps.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Callable, Dict, List, Any
import time


class DeviceStateType(Enum):
    """Device state types."""
    IDLE = auto()       # Waiting for broadcast signal
    DECIDING = auto()   # Evaluating whether to respond
    RESPONDING = auto() # Actively responding (charging/discharging/etc.)
    OFFLINE = auto()    # Disconnected or faulted


class Action(Enum):
    """Possible actions a device can take."""
    HOLD = auto()           # Do nothing, maintain current state
    CHARGE = auto()         # Start/continue charging
    DISCHARGE = auto()      # Start/continue discharging
    STOP = auto()           # Stop current response


@dataclass
class DeviceState:
    """
    Complete state representation of an edge device.

    This is a base class - specific device types extend with additional fields.
    """
    # Core identification
    device_id: str
    device_type: str  # 'battery'

    # Current state machine state
    state_type: DeviceStateType = DeviceStateType.IDLE

    # Power state
    power_current: float = 0.0  # Current power flow (kW), positive=consuming
    power_target: float = 0.0   # Target power after response

    # Timing
    state_entered_at: float = field(default_factory=time.time)
    last_signal_at: Optional[float] = None
    last_response_at: Optional[float] = None

    # Response tracking
    current_action: Action = Action.HOLD
    response_duration: float = 0.0  # Seconds in current response

    # Health status
    is_healthy: bool = True
    fault_code: Optional[str] = None

    @property
    def time_in_state(self) -> float:
        """Seconds since entering current state."""
        return time.time() - self.state_entered_at

    @property
    def time_since_signal(self) -> Optional[float]:
        """Seconds since last signal received, or None if never."""
        if self.last_signal_at is None:
            return None
        return time.time() - self.last_signal_at


@dataclass
class BatteryState(DeviceState):
    """State for battery storage devices."""
    soc: float = 0.5  # State of charge (0.0-1.0)
    capacity_kwh: float = 10.0  # Total capacity in kWh
    max_charge_kw: float = 5.0  # Maximum charge power
    max_discharge_kw: float = 5.0  # Maximum discharge power
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    temperature: float = 25.0  # Celsius
    daily_cycles: int = 0  # Cycles completed today

    def __post_init__(self):
        self.device_type = 'battery'


class StateTransition:
    """Represents a state transition event."""

    def __init__(
        self,
        from_state: DeviceStateType,
        to_state: DeviceStateType,
        trigger: str,
        timestamp: Optional[float] = None,
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.trigger = trigger
        self.timestamp = timestamp or time.time()


class DeviceStateMachine(ABC):
    """
    Abstract base class for device state machines.

    Manages state transitions and enforces valid transition rules.
    Subclasses implement device-specific transition logic.
    """

    # Valid state transitions: (from_state, to_state) -> trigger
    VALID_TRANSITIONS = {
        (DeviceStateType.IDLE, DeviceStateType.DECIDING): 'signal_received',
        (DeviceStateType.IDLE, DeviceStateType.OFFLINE): 'fault',
        (DeviceStateType.DECIDING, DeviceStateType.RESPONDING): 'decision_respond',
        (DeviceStateType.DECIDING, DeviceStateType.IDLE): 'decision_hold',
        (DeviceStateType.DECIDING, DeviceStateType.OFFLINE): 'fault',
        (DeviceStateType.RESPONDING, DeviceStateType.IDLE): 'response_complete',
        (DeviceStateType.RESPONDING, DeviceStateType.OFFLINE): 'fault',
        (DeviceStateType.OFFLINE, DeviceStateType.IDLE): 'recovery',
    }

    def __init__(self, state: DeviceState):
        """
        Initialize state machine with device state.

        Args:
            state: Initial device state
        """
        self.state = state
        self.transition_history: List[StateTransition] = []
        self._callbacks: Dict[str, List[Callable]] = {
            'on_enter': [],
            'on_exit': [],
            'on_transition': [],
        }

    @property
    def current_state(self) -> DeviceStateType:
        """Current state type."""
        return self.state.state_type

    def can_transition(self, to_state: DeviceStateType) -> bool:
        """Check if transition to target state is valid."""
        return (self.current_state, to_state) in self.VALID_TRANSITIONS

    def transition_to(self, to_state: DeviceStateType, trigger: str) -> bool:
        """
        Attempt state transition.

        Args:
            to_state: Target state
            trigger: Trigger event name

        Returns:
            True if transition was successful
        """
        if not self.can_transition(to_state):
            return False

        # Create transition record
        transition = StateTransition(
            from_state=self.current_state,
            to_state=to_state,
            trigger=trigger,
        )

        # Call exit callbacks
        for callback in self._callbacks['on_exit']:
            callback(self.current_state)

        # Update state
        old_state = self.current_state
        self.state.state_type = to_state
        self.state.state_entered_at = time.time()
        self.transition_history.append(transition)

        # Call enter callbacks
        for callback in self._callbacks['on_enter']:
            callback(to_state)

        # Call transition callbacks
        for callback in self._callbacks['on_transition']:
            callback(transition)

        return True

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register callback for state machine events.

        Args:
            event: One of 'on_enter', 'on_exit', 'on_transition'
            callback: Function to call on event
        """
        if event not in self._callbacks:
            raise ValueError(f"Unknown event: {event}")
        self._callbacks[event].append(callback)

    def on_signal_received(self) -> None:
        """Handle incoming broadcast signal."""
        self.state.last_signal_at = time.time()

        if self.current_state == DeviceStateType.IDLE:
            self.transition_to(DeviceStateType.DECIDING, 'signal_received')
        elif self.current_state == DeviceStateType.OFFLINE:
            # Signal received while offline - attempt recovery
            self.transition_to(DeviceStateType.IDLE, 'recovery')
            self.transition_to(DeviceStateType.DECIDING, 'signal_received')

    def on_decision_made(self, action: Action) -> None:
        """
        Handle decision result from controller.

        Args:
            action: The decided action
        """
        if self.current_state != DeviceStateType.DECIDING:
            return

        self.state.current_action = action

        if action == Action.HOLD:
            self.transition_to(DeviceStateType.IDLE, 'decision_hold')
        else:
            self.transition_to(DeviceStateType.RESPONDING, 'decision_respond')
            self.state.last_response_at = time.time()

    def on_response_complete(self) -> None:
        """Handle response completion."""
        if self.current_state == DeviceStateType.RESPONDING:
            self.state.response_duration = time.time() - (self.state.last_response_at or time.time())
            self.state.current_action = Action.HOLD
            self.transition_to(DeviceStateType.IDLE, 'response_complete')

    def on_fault(self, fault_code: str) -> None:
        """
        Handle device fault.

        Args:
            fault_code: Identifier for the fault condition
        """
        self.state.is_healthy = False
        self.state.fault_code = fault_code
        self.transition_to(DeviceStateType.OFFLINE, 'fault')

    def on_recovery(self) -> None:
        """Handle recovery from fault state."""
        if self.current_state == DeviceStateType.OFFLINE:
            self.state.is_healthy = True
            self.state.fault_code = None
            self.transition_to(DeviceStateType.IDLE, 'recovery')

    @abstractmethod
    def update(self, dt: float) -> None:
        """
        Update device state for time step.

        Args:
            dt: Time step in seconds
        """
        pass

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state for logging/monitoring."""
        return {
            'device_id': self.state.device_id,
            'device_type': self.state.device_type,
            'state': self.current_state.name,
            'action': self.state.current_action.name,
            'power_kw': self.state.power_current,
            'time_in_state': self.state.time_in_state,
            'is_healthy': self.state.is_healthy,
        }



