"""
64-bit Broadcast Signal Encoder

Encodes the dispatch signal into a compact 64-bit frame for satellite broadcast.
The two operationally critical fields are:

    intensity     = round(|s| * 4095)     12-bit, encodes signal score magnitude
    supply_demand:
        s >= 0 (charge):   round(7 * (1 - |s|))    in [0, 7]
        s <  0 (discharge): round(8 + 7 * |s|)      in [8, 15]

    where s = clip[(r - 1) * d,  -1, +1]  is the signal score from the
    supply-demand ratio r and dispatch intensity d.

Bit layout (64 bits, MSB to LSB):
    63-56  version_ts     (8 bits)   protocol version + sequence
    55-44  region_id      (12 bits)  feeder/region identifier
    43-40  supply_demand  (4 bits)   charge/discharge direction + level
    39-28  intensity      (12 bits)  response magnitude |s| * 4095
    27-16  reserved       (12 bits)  set to 0
    15-12  priority       (4 bits)   device-type priority
    11-0   crc            (12 bits)  CRC-12 error detection
"""

from dataclasses import dataclass
from typing import Optional
import struct


@dataclass
class EPSSignal:
    """Structured representation of a broadcast signal."""

    version: int  # Signal version (0-15)
    timestamp_seq: int  # Timestamp sequence number (0-15)
    region_id: int  # Region identifier (0-4095)
    supply_demand: int  # Supply-demand state: 0=surplus, 7=balance, 15=shortage
    intensity: int  # Response intensity (0-4095, 12-bit encoding of 0%-100%)
    price: int  # Reserved field (always 0 in paper experiments; retained for 64-bit format)
    priority: int  # Device type priority (0-15)
    crc: int = 0  # CRC-12 checksum (computed automatically)

    @property
    def version_ts(self) -> int:
        """Combined version and timestamp byte."""
        return ((self.version & 0x0F) << 4) | (self.timestamp_seq & 0x0F)

    @property
    def intensity_percent(self) -> float:
        """Response intensity as percentage (0.0-100.0%)."""
        return self.intensity / 40.95

    @property
    def price_value(self) -> float:
        """Reserved field (always 0)."""
        return self.price / 100.0

    @property
    def is_shortage(self) -> bool:
        """True if supply_demand indicates shortage (>7)."""
        return self.supply_demand > 7

    @property
    def is_surplus(self) -> bool:
        """True if supply_demand indicates surplus (<7)."""
        return self.supply_demand < 7


class EPSSignalEncoder:
    """
    Encodes EPS signals into 64-bit binary format.

    Uses CRC-12 polynomial: x^12 + x^11 + x^3 + x^2 + x + 1 (0x180F)
    This is suitable for short messages and provides good error detection.
    """

    # CRC-12 polynomial (0x180F = 110000001111 in binary)
    CRC_POLY = 0x180F
    CRC_BITS = 12

    # Field bit positions and masks
    FIELD_SPECS = {
        'version_ts': {'bits': 8, 'shift': 56, 'max': 255},
        'region_id': {'bits': 12, 'shift': 44, 'max': 4095},
        'supply_demand': {'bits': 4, 'shift': 40, 'max': 15},
        'intensity': {'bits': 12, 'shift': 28, 'max': 4095},
        'price': {'bits': 12, 'shift': 16, 'max': 4095},
        'priority': {'bits': 4, 'shift': 12, 'max': 15},
        'crc': {'bits': 12, 'shift': 0, 'max': 4095},
    }

    def __init__(self, version: int = 1):
        """
        Initialize encoder with protocol version.

        Args:
            version: Protocol version (0-15), default 1
        """
        if not 0 <= version <= 15:
            raise ValueError(f"Version must be 0-15, got {version}")
        self.version = version
        self._timestamp_seq = 0

    def _compute_crc12(self, data: int, data_bits: int = 52) -> int:
        """
        Compute CRC-12 checksum for the data portion.

        Uses bit-by-bit polynomial division.

        Args:
            data: The 52-bit data value (signal without CRC field)
            data_bits: Number of data bits

        Returns:
            12-bit CRC value
        """
        # Shift data left by CRC bits to make room for remainder
        remainder = data << self.CRC_BITS

        # Polynomial with leading 1 bit for CRC-12
        divisor = self.CRC_POLY << data_bits

        # Perform polynomial division
        for i in range(data_bits, -1, -1):
            if remainder & (1 << (i + self.CRC_BITS)):
                remainder ^= (self.CRC_POLY << i)

        return remainder & 0xFFF  # Return 12-bit remainder

    def _validate_field(self, name: str, value: int) -> None:
        """Validate a field value is within range."""
        spec = self.FIELD_SPECS[name]
        if not 0 <= value <= spec['max']:
            raise ValueError(
                f"{name} must be 0-{spec['max']}, got {value}"
            )

    def encode(
        self,
        region_id: int,
        supply_demand: int,
        intensity: int,
        price: int,
        priority: int,
        timestamp_seq: Optional[int] = None,
    ) -> bytes:
        """
        Encode signal parameters into 64-bit (8 bytes) binary format.

        Args:
            region_id: Region identifier (0-4095)
            supply_demand: Supply-demand state (0-15)
                0 = extreme surplus
                7 = balanced
                15 = extreme shortage
            intensity: Response intensity (0-4095)
                Maps to 0.0%-100.0% (12-bit, step ≈ 0.024%)
            price: Incentive price (0-4095)
                Reserved field (always 0 in current implementation)
            priority: Device type priority (0-15)
                Higher = higher priority for response
            timestamp_seq: Optional timestamp sequence (0-15)
                Auto-increments if not provided

        Returns:
            8-byte encoded signal

        Raises:
            ValueError: If any parameter is out of range
        """
        # Validate all fields
        self._validate_field('region_id', region_id)
        self._validate_field('supply_demand', supply_demand)
        self._validate_field('intensity', intensity)
        self._validate_field('price', price)
        self._validate_field('priority', priority)

        # Handle timestamp sequence
        if timestamp_seq is None:
            ts = self._timestamp_seq
            self._timestamp_seq = (self._timestamp_seq + 1) % 16
        else:
            if not 0 <= timestamp_seq <= 15:
                raise ValueError(f"timestamp_seq must be 0-15, got {timestamp_seq}")
            ts = timestamp_seq

        # Combine version and timestamp
        version_ts = ((self.version & 0x0F) << 4) | (ts & 0x0F)

        # Build the 52-bit data portion (without CRC)
        data = 0
        data |= (version_ts & 0xFF) << 44  # 8 bits at position 44
        data |= (region_id & 0xFFF) << 32  # 12 bits at position 32
        data |= (supply_demand & 0x0F) << 28  # 4 bits at position 28
        data |= (intensity & 0xFFF) << 16  # 12 bits at position 16
        data |= (price & 0xFFF) << 4  # 12 bits at position 4
        data |= (priority & 0x0F)  # 4 bits at position 0

        # Compute CRC-12
        crc = self._compute_crc12(data, data_bits=52)

        # Build final 64-bit value
        signal = 0
        signal |= (version_ts & 0xFF) << 56
        signal |= (region_id & 0xFFF) << 44
        signal |= (supply_demand & 0x0F) << 40
        signal |= (intensity & 0xFFF) << 28
        signal |= (price & 0xFFF) << 16
        signal |= (priority & 0x0F) << 12
        signal |= (crc & 0xFFF)

        # Convert to 8 bytes (big-endian)
        return struct.pack('>Q', signal)

    def encode_signal(self, signal: EPSSignal) -> bytes:
        """
        Encode an EPSSignal object to binary format.

        Args:
            signal: EPSSignal object with all fields set

        Returns:
            8-byte encoded signal
        """
        return self.encode(
            region_id=signal.region_id,
            supply_demand=signal.supply_demand,
            intensity=signal.intensity,
            price=signal.price,
            priority=signal.priority,
            timestamp_seq=signal.timestamp_seq,
        )

    @classmethod
    def encode_from_physical(
        cls,
        region_id: int,
        supply_demand: int,
        intensity_percent: float,
        price_value: float,
        priority: int,
        version: int = 1,
    ) -> bytes:
        """
        Convenience method to encode using physical units.

        Args:
            region_id: Region identifier (0-4095)
            supply_demand: Supply-demand state (0-15)
            intensity_percent: Response intensity (0.0-100.0%)
            price_value: Reserved field (always 0)
            priority: Device type priority (0-15)
            version: Protocol version (default 1)

        Returns:
            8-byte encoded signal
        """
        encoder = cls(version=version)

        # Convert physical units to encoded values
        intensity = int(round(intensity_percent * 40.95))
        intensity = max(0, min(4095, intensity))

        price = int(round(price_value * 100))
        price = max(0, min(4095, price))

        return encoder.encode(
            region_id=region_id,
            supply_demand=supply_demand,
            intensity=intensity,
            price=price,
            priority=priority,
        )
