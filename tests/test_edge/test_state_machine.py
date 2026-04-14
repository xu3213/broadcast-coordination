"""
Tests for device state machine

Tests the state machine logic including:
- State transitions
- Callback invocation
- Device-specific state machines
"""

import pytest
import time
from unittest.mock import MagicMock

from src.edge.state_machine import (
    DeviceStateType,
    Action,
    DeviceState,
    BatteryState,
    DeviceStateMachine,
    StateTransition,
)


class TestDeviceState:
    """Tests for DeviceState base class."""

    def test_default_values(self):
        """Test default state values."""
        state = DeviceState(device_id='dev_001', device_type='generic')

        assert state.device_id == 'dev_001'
        assert state.device_type == 'generic'
        assert state.state_type == DeviceStateType.IDLE
        assert state.power_current == 0.0
        assert state.is_healthy is True

    def test_time_in_state(self):
        """Test time_in_state calculation."""
        state = DeviceState(device_id='dev_001', device_type='generic')

        # Should be very small initially
        assert state.time_in_state < 1.0

        # After a short sleep
        time.sleep(0.1)
        assert state.time_in_state >= 0.1

    def test_time_since_signal(self):
        """Test time_since_signal calculation."""
        state = DeviceState(device_id='dev_001', device_type='generic')

        # Initially None (no signal received)
        assert state.time_since_signal is None

        # After setting last_signal_at
        state.last_signal_at = time.time()
        time.sleep(0.1)
        assert state.time_since_signal is not None
        assert state.time_since_signal >= 0.1


class TestBatteryState:
    """Tests for BatteryState class."""

    def test_default_values(self):
        """Test battery default values."""
        state = BatteryState(device_id='bat_001', device_type='battery')

        assert state.soc == 0.5
        assert state.capacity_kwh == 10.0
        assert state.max_charge_kw == 5.0
        assert state.charge_efficiency == 0.95

    def test_post_init_sets_device_type(self):
        """Test __post_init__ sets device_type."""
        state = BatteryState(device_id='bat_001', device_type='wrong')

        # Should be corrected by __post_init__
        assert state.device_type == 'battery'


class TestStateTransition:
    """Tests for StateTransition class."""

    def test_creation(self):
        """Test transition creation."""
        transition = StateTransition(
            from_state=DeviceStateType.IDLE,
            to_state=DeviceStateType.DECIDING,
            trigger='signal_received',
        )

        assert transition.from_state == DeviceStateType.IDLE
        assert transition.to_state == DeviceStateType.DECIDING
        assert transition.trigger == 'signal_received'
        assert transition.timestamp > 0

    def test_explicit_timestamp(self):
        """Test transition with explicit timestamp."""
        transition = StateTransition(
            from_state=DeviceStateType.IDLE,
            to_state=DeviceStateType.DECIDING,
            trigger='test',
            timestamp=12345.0,
        )

        assert transition.timestamp == 12345.0


class TestDeviceStateTypeEnum:
    """Tests for DeviceStateType enum."""

    def test_all_states_defined(self):
        """Test all expected states are defined."""
        states = list(DeviceStateType)

        assert DeviceStateType.IDLE in states
        assert DeviceStateType.DECIDING in states
        assert DeviceStateType.RESPONDING in states
        assert DeviceStateType.OFFLINE in states


class TestActionEnum:
    """Tests for Action enum."""

    def test_all_actions_defined(self):
        """Test all expected actions are defined."""
        actions = list(Action)

        assert Action.HOLD in actions
        assert Action.CHARGE in actions
        assert Action.DISCHARGE in actions
        assert Action.STOP in actions
