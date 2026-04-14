"""
Tests for signal decoder

Tests the 64-bit signal decoding including:
- CRC-12 validation
- Field extraction
- Version checking
- Replay detection
"""

import pytest
import struct
from src.signal.decoder import EPSSignalDecoder, DecodeStatus, DecodeResult, decode_signal_fields
from src.signal.encoder import EPSSignalEncoder


class TestEPSSignalDecoder:
    """Tests for EPSSignalDecoder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = EPSSignalEncoder(version=1)
        self.decoder = EPSSignalDecoder()

    def test_basic_decode(self):
        """Test basic signal decoding."""
        data = self.encoder.encode(
            region_id=100,
            supply_demand=12,
            intensity=2048,
            price=1000,
            priority=5,
        )

        result = self.decoder.decode(data)

        assert result.is_valid
        assert result.status == DecodeStatus.OK
        assert result.signal is not None
        assert result.signal.region_id == 100

    def test_decode_all_fields(self):
        """Test all fields are decoded correctly."""
        data = self.encoder.encode(
            region_id=2048,
            supply_demand=8,
            intensity=3000,
            price=2000,
            priority=10,
            timestamp_seq=7,
        )

        result = self.decoder.decode(data)

        assert result.is_valid
        assert result.signal.region_id == 2048
        assert result.signal.supply_demand == 8
        assert result.signal.intensity == 3000
        assert result.signal.price == 2000
        assert result.signal.priority == 10
        assert result.signal.timestamp_seq == 7
        assert result.signal.version == 1

    def test_invalid_length(self):
        """Test decoding rejects wrong-length data."""
        result = self.decoder.decode(b'\x00' * 7)  # 7 bytes instead of 8
        assert not result.is_valid
        assert result.status == DecodeStatus.INVALID_LENGTH

        result = self.decoder.decode(b'\x00' * 9)  # 9 bytes instead of 8
        assert not result.is_valid
        assert result.status == DecodeStatus.INVALID_LENGTH

    def test_crc_error_detection(self):
        """Test CRC errors are detected."""
        data = self.encoder.encode(
            region_id=100,
            supply_demand=12,
            intensity=2048,
            price=1000,
            priority=5,
        )

        # Flip some bits in the data portion (not CRC)
        corrupted = bytearray(data)
        corrupted[2] ^= 0x55  # Corrupt a middle byte
        corrupted = bytes(corrupted)

        result = self.decoder.decode(corrupted)

        assert not result.is_valid
        assert result.status == DecodeStatus.CRC_ERROR
        assert "CRC" in result.error_message

    def test_crc_error_in_crc_field(self):
        """Test CRC errors when CRC field itself is corrupted."""
        data = self.encoder.encode(
            region_id=100,
            supply_demand=12,
            intensity=2048,
            price=1000,
            priority=5,
        )

        # Corrupt the CRC field (last 12 bits = last 1.5 bytes)
        corrupted = bytearray(data)
        corrupted[7] ^= 0x0F  # Flip bits in last byte
        corrupted = bytes(corrupted)

        result = self.decoder.decode(corrupted)

        assert not result.is_valid
        assert result.status == DecodeStatus.CRC_ERROR

    def test_version_checking(self):
        """Test version mismatch detection."""
        encoder_v2 = EPSSignalEncoder(version=2)
        decoder_v1 = EPSSignalDecoder(expected_version=1)

        data = encoder_v2.encode(
            region_id=100,
            supply_demand=12,
            intensity=2048,
            price=1000,
            priority=5,
        )

        result = decoder_v1.decode(data)

        assert not result.is_valid
        assert result.status == DecodeStatus.VERSION_MISMATCH
        assert "Version" in result.error_message

    def test_version_checking_disabled(self):
        """Test version checking can be disabled."""
        encoder_v2 = EPSSignalEncoder(version=2)
        decoder_any = EPSSignalDecoder(expected_version=None)

        data = encoder_v2.encode(
            region_id=100,
            supply_demand=12,
            intensity=2048,
            price=1000,
            priority=5,
        )

        result = decoder_any.decode(data)
        assert result.is_valid

    def test_decode_unsafe(self):
        """Test decode_unsafe raises on errors."""
        data = self.encoder.encode(
            region_id=100,
            supply_demand=12,
            intensity=2048,
            price=1000,
            priority=5,
        )

        # Valid data should work
        signal = self.decoder.decode_unsafe(data)
        assert signal.region_id == 100

        # Invalid data should raise
        with pytest.raises(ValueError, match="Decode failed"):
            self.decoder.decode_unsafe(b'\x00' * 7)

    def test_verify_crc_quick(self):
        """Test quick CRC verification."""
        data = self.encoder.encode(
            region_id=100,
            supply_demand=12,
            intensity=2048,
            price=1000,
            priority=5,
        )

        assert self.decoder.verify_crc(data) is True

        # Corrupt the data
        corrupted = bytearray(data)
        corrupted[3] ^= 0xFF
        assert self.decoder.verify_crc(bytes(corrupted)) is False

        # Wrong length
        assert self.decoder.verify_crc(b'\x00' * 7) is False

    def test_replay_detection_enabled(self):
        """Test replay detection when enabled."""
        decoder = EPSSignalDecoder(enable_replay_detection=True)

        # First signal with ts_seq=0
        data0 = self.encoder.encode(
            region_id=100,
            supply_demand=12,
            intensity=2048,
            price=1000,
            priority=5,
            timestamp_seq=0,
        )

        # Second signal with ts_seq=1
        self.encoder._timestamp_seq = 0  # Reset
        data1 = self.encoder.encode(
            region_id=100,
            supply_demand=12,
            intensity=2048,
            price=1000,
            priority=5,
            timestamp_seq=1,
        )

        result0 = decoder.decode(data0)
        assert result0.is_valid

        result1 = decoder.decode(data1)
        assert result1.is_valid

        # Currently replay detection doesn't reject, just tracks
        # Future: could add strict mode that rejects

    def test_replay_tracking_reset(self):
        """Test replay tracking can be reset."""
        decoder = EPSSignalDecoder(enable_replay_detection=True)

        data = self.encoder.encode(
            region_id=100,
            supply_demand=12,
            intensity=2048,
            price=1000,
            priority=5,
            timestamp_seq=5,
        )

        decoder.decode(data)
        assert 100 in decoder._last_ts

        # Reset specific region
        decoder.reset_replay_tracking(region_id=100)
        assert 100 not in decoder._last_ts

        # Test reset all
        decoder.decode(data)
        decoder.reset_replay_tracking()
        assert len(decoder._last_ts) == 0


class TestDecodeResult:
    """Tests for DecodeResult class."""

    def test_is_valid_property(self):
        """Test is_valid reflects status."""
        result_ok = DecodeResult(status=DecodeStatus.OK, signal=None)
        assert result_ok.is_valid is True

        result_crc = DecodeResult(status=DecodeStatus.CRC_ERROR, signal=None)
        assert result_crc.is_valid is False

        result_ver = DecodeResult(status=DecodeStatus.VERSION_MISMATCH, signal=None)
        assert result_ver.is_valid is False


class TestDecodeSignalFields:
    """Tests for low-level decode_signal_fields function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = EPSSignalEncoder(version=1)

    def test_field_extraction(self):
        """Test all fields are extracted correctly."""
        data = self.encoder.encode(
            region_id=1234,
            supply_demand=11,
            intensity=2500,
            price=1500,
            priority=6,
            timestamp_seq=9,
        )

        version, ts_seq, region, sd, intensity, price, priority, crc = decode_signal_fields(data)

        assert version == 1
        assert ts_seq == 9
        assert region == 1234
        assert sd == 11
        assert intensity == 2500
        assert price == 1500
        assert priority == 6
        # CRC is computed, just check it's a valid 12-bit value
        assert 0 <= crc <= 4095

    def test_invalid_length_raises(self):
        """Test wrong length raises ValueError."""
        with pytest.raises(ValueError, match="Expected 8 bytes"):
            decode_signal_fields(b'\x00' * 6)

    def test_boundary_values(self):
        """Test decoding boundary values."""
        # All zeros
        data_min = self.encoder.encode(
            region_id=0,
            supply_demand=0,
            intensity=0,
            price=0,
            priority=0,
            timestamp_seq=0,
        )

        fields_min = decode_signal_fields(data_min)
        assert fields_min[2] == 0  # region
        assert fields_min[3] == 0  # supply_demand
        assert fields_min[4] == 0  # intensity
        assert fields_min[5] == 0  # price
        assert fields_min[6] == 0  # priority

        # All max
        encoder_max = EPSSignalEncoder(version=15)
        data_max = encoder_max.encode(
            region_id=4095,
            supply_demand=15,
            intensity=4095,
            price=4095,
            priority=15,
            timestamp_seq=15,
        )

        fields_max = decode_signal_fields(data_max)
        assert fields_max[0] == 15  # version
        assert fields_max[1] == 15  # timestamp_seq
        assert fields_max[2] == 4095  # region
        assert fields_max[3] == 15  # supply_demand
        assert fields_max[4] == 4095  # intensity
        assert fields_max[5] == 4095  # price
        assert fields_max[6] == 15  # priority


class TestRoundTrip:
    """Tests for encoding/decoding round trips."""

    def test_full_range_round_trip(self):
        """Test round trip with various values across full range."""
        encoder = EPSSignalEncoder(version=1)
        decoder = EPSSignalDecoder()

        # Sample values across range
        test_values = [
            (0, 0, 0, 0, 0),
            (4095, 15, 4095, 4095, 15),
            (100, 7, 2048, 1000, 8),
            (2000, 10, 3500, 2500, 5),
            (1, 1, 1, 1, 1),
        ]

        for region, sd, intensity, price, priority in test_values:
            original = encoder.encode(
                region_id=region,
                supply_demand=sd,
                intensity=intensity,
                price=price,
                priority=priority,
            )

            result = decoder.decode(original)

            assert result.is_valid
            assert result.signal.region_id == region
            assert result.signal.supply_demand == sd
            assert result.signal.intensity == intensity
            assert result.signal.price == price
            assert result.signal.priority == priority

    def test_multiple_encoders_same_decoder(self):
        """Test signals from different encoders decode correctly."""
        encoder_v1_a = EPSSignalEncoder(version=1)
        encoder_v1_b = EPSSignalEncoder(version=1)
        decoder = EPSSignalDecoder()

        data_a = encoder_v1_a.encode(region_id=100, supply_demand=5, intensity=1000, price=500, priority=3)
        data_b = encoder_v1_b.encode(region_id=200, supply_demand=10, intensity=2000, price=1500, priority=7)

        result_a = decoder.decode(data_a)
        result_b = decoder.decode(data_b)

        assert result_a.is_valid and result_a.signal.region_id == 100
        assert result_b.is_valid and result_b.signal.region_id == 200
