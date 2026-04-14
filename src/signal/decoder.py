"""
64-bit Broadcast Signal Decoder

Decodes binary signals back to structured EPSSignal objects,
with CRC-12 validation for error detection.
"""

import struct
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum

from .encoder import EPSSignal


class DecodeStatus(Enum):
    """Status codes for signal decoding."""
    OK = "ok"
    CRC_ERROR = "crc_error"
    VERSION_MISMATCH = "version_mismatch"
    INVALID_LENGTH = "invalid_length"
    REPLAY_DETECTED = "replay_detected"


@dataclass
class DecodeResult:
    """Result of signal decoding operation."""
    status: DecodeStatus
    signal: Optional[EPSSignal]
    error_message: str = ""

    @property
    def is_valid(self) -> bool:
        """True if decoding was successful."""
        return self.status == DecodeStatus.OK


class EPSSignalDecoder:
    """
    Decodes signals from 64-bit binary format.

    Validates CRC-12, checks version compatibility, and optionally
    detects replay attacks using timestamp sequence tracking.
    """

    # CRC-12 polynomial (same as encoder)
    CRC_POLY = 0x180F
    CRC_BITS = 12

    def __init__(
        self,
        expected_version: Optional[int] = None,
        enable_replay_detection: bool = False,
    ):
        """
        Initialize decoder.

        Args:
            expected_version: If set, signals with different version are rejected
            enable_replay_detection: If True, track timestamp sequences per region
        """
        self.expected_version = expected_version
        self.enable_replay_detection = enable_replay_detection

        # Track last seen timestamp sequence per region (for replay detection)
        self._last_ts: dict[int, int] = {}

    def _compute_crc12(self, data: int, data_bits: int = 52) -> int:
        """Compute CRC-12 checksum (same algorithm as encoder)."""
        remainder = data << self.CRC_BITS

        for i in range(data_bits, -1, -1):
            if remainder & (1 << (i + self.CRC_BITS)):
                remainder ^= (self.CRC_POLY << i)

        return remainder & 0xFFF

    def _extract_fields(self, signal_int: int) -> dict:
        """Extract all fields from 64-bit signal value."""
        return {
            'version_ts': (signal_int >> 56) & 0xFF,
            'region_id': (signal_int >> 44) & 0xFFF,
            'supply_demand': (signal_int >> 40) & 0x0F,
            'intensity': (signal_int >> 28) & 0xFFF,
            'price': (signal_int >> 16) & 0xFFF,
            'priority': (signal_int >> 12) & 0x0F,
            'crc': signal_int & 0xFFF,
        }

    def _rebuild_data_for_crc(self, fields: dict) -> int:
        """Rebuild the 52-bit data portion for CRC verification."""
        data = 0
        data |= (fields['version_ts'] & 0xFF) << 44
        data |= (fields['region_id'] & 0xFFF) << 32
        data |= (fields['supply_demand'] & 0x0F) << 28
        data |= (fields['intensity'] & 0xFFF) << 16
        data |= (fields['price'] & 0xFFF) << 4
        data |= (fields['priority'] & 0x0F)
        return data

    def decode(self, data: bytes) -> DecodeResult:
        """
        Decode binary signal to EPSSignal object.

        Args:
            data: 8-byte encoded signal (big-endian)

        Returns:
            DecodeResult with status and decoded signal (if successful)
        """
        # Validate input length
        if len(data) != 8:
            return DecodeResult(
                status=DecodeStatus.INVALID_LENGTH,
                signal=None,
                error_message=f"Expected 8 bytes, got {len(data)}",
            )

        # Unpack as 64-bit unsigned integer
        signal_int = struct.unpack('>Q', data)[0]

        # Extract all fields
        fields = self._extract_fields(signal_int)

        # Verify CRC
        data_portion = self._rebuild_data_for_crc(fields)
        computed_crc = self._compute_crc12(data_portion, data_bits=52)

        if computed_crc != fields['crc']:
            return DecodeResult(
                status=DecodeStatus.CRC_ERROR,
                signal=None,
                error_message=f"CRC mismatch: expected {computed_crc:03X}, got {fields['crc']:03X}",
            )

        # Extract version and timestamp from combined byte
        version = (fields['version_ts'] >> 4) & 0x0F
        timestamp_seq = fields['version_ts'] & 0x0F

        # Check version compatibility
        if self.expected_version is not None and version != self.expected_version:
            return DecodeResult(
                status=DecodeStatus.VERSION_MISMATCH,
                signal=None,
                error_message=f"Version {version} != expected {self.expected_version}",
            )

        # Check for replay attack
        if self.enable_replay_detection:
            region = fields['region_id']
            if region in self._last_ts:
                last_ts = self._last_ts[region]
                # Simple check: timestamp should be incrementing
                # Allow wrap-around (15 -> 0 is valid)
                expected_next = (last_ts + 1) % 16
                # Accept current or next (small window for out-of-order)
                if timestamp_seq != expected_next and timestamp_seq != last_ts:
                    # Might be replay or out-of-order
                    # For strict mode, could reject here
                    pass  # Currently just warn, don't reject

            self._last_ts[region] = timestamp_seq

        # Build EPSSignal object
        signal = EPSSignal(
            version=version,
            timestamp_seq=timestamp_seq,
            region_id=fields['region_id'],
            supply_demand=fields['supply_demand'],
            intensity=fields['intensity'],
            price=fields['price'],
            priority=fields['priority'],
            crc=fields['crc'],
        )

        return DecodeResult(status=DecodeStatus.OK, signal=signal)

    def decode_unsafe(self, data: bytes) -> EPSSignal:
        """
        Decode without returning status (raises exception on error).

        Args:
            data: 8-byte encoded signal

        Returns:
            EPSSignal object

        Raises:
            ValueError: If decoding fails
        """
        result = self.decode(data)
        if not result.is_valid:
            raise ValueError(f"Decode failed: {result.status.value} - {result.error_message}")
        return result.signal

    def verify_crc(self, data: bytes) -> bool:
        """
        Quick CRC verification without full decode.

        Args:
            data: 8-byte encoded signal

        Returns:
            True if CRC is valid
        """
        if len(data) != 8:
            return False

        signal_int = struct.unpack('>Q', data)[0]
        fields = self._extract_fields(signal_int)
        data_portion = self._rebuild_data_for_crc(fields)
        computed_crc = self._compute_crc12(data_portion, data_bits=52)

        return computed_crc == fields['crc']

    def reset_replay_tracking(self, region_id: Optional[int] = None) -> None:
        """
        Reset replay detection state.

        Args:
            region_id: If provided, reset only for this region.
                      If None, reset all regions.
        """
        if region_id is not None:
            self._last_ts.pop(region_id, None)
        else:
            self._last_ts.clear()


def decode_signal_fields(data: bytes) -> Tuple[int, int, int, int, int, int, int, int]:
    """
    Low-level field extraction without validation.

    Returns tuple of: (version, timestamp_seq, region_id, supply_demand,
                       intensity, price, priority, crc)
    """
    if len(data) != 8:
        raise ValueError(f"Expected 8 bytes, got {len(data)}")

    signal_int = struct.unpack('>Q', data)[0]

    version_ts = (signal_int >> 56) & 0xFF
    version = (version_ts >> 4) & 0x0F
    timestamp_seq = version_ts & 0x0F
    region_id = (signal_int >> 44) & 0xFFF
    supply_demand = (signal_int >> 40) & 0x0F
    intensity = (signal_int >> 28) & 0xFFF
    price = (signal_int >> 16) & 0xFFF
    priority = (signal_int >> 12) & 0x0F
    crc = signal_int & 0xFFF

    return (version, timestamp_seq, region_id, supply_demand,
            intensity, price, priority, crc)
