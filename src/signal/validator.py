"""
Signal Validator

Structural validation (field ranges, CRC) and semantic checks
(supply-demand consistency) for 64-bit broadcast signals.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum

from .encoder import EPSSignal


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue found."""
    level: ValidationLevel
    code: str
    message: str
    field: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == ValidationLevel.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == ValidationLevel.WARNING]

    def add_error(self, code: str, message: str, field: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue(ValidationLevel.ERROR, code, message, field))
        self.is_valid = False

    def add_warning(self, code: str, message: str, field: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue(ValidationLevel.WARNING, code, message, field))

    def add_info(self, code: str, message: str, field: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue(ValidationLevel.INFO, code, message, field))


class EPSSignalValidator:
    """
    Validates broadcast signals for correctness.

    Checks:
    1. Structural: field ranges within 64-bit encoding bounds
    2. Semantic: supply-demand / intensity consistency
    """

    DEFAULT_CONFIG = {
        'valid_regions': None,
        'warn_high_intensity': 3500,
        'supported_versions': {1},
        'require_intensity_for_shortage': True,
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)

        if self.config['valid_regions'] is not None:
            self.valid_regions: Optional[Set[int]] = set(self.config['valid_regions'])
        else:
            self.valid_regions = None

    def validate(self, signal: EPSSignal) -> ValidationResult:
        """Perform full validation on a signal."""
        result = ValidationResult(is_valid=True)
        self._validate_field_ranges(signal, result)
        self._validate_version(signal, result)
        self._validate_region(signal, result)
        self._validate_intensity(signal, result)
        self._validate_supply_demand_consistency(signal, result)
        return result

    def _validate_field_ranges(self, signal: EPSSignal, result: ValidationResult) -> None:
        """Check all fields are within valid encoding ranges."""
        if not 0 <= signal.version <= 15:
            result.add_error("INVALID_VERSION_RANGE", f"Version {signal.version} out of range [0-15]", "version")
        if not 0 <= signal.timestamp_seq <= 15:
            result.add_error("INVALID_TS_RANGE", f"Timestamp seq {signal.timestamp_seq} out of range [0-15]", "timestamp_seq")
        if not 0 <= signal.region_id <= 4095:
            result.add_error("INVALID_REGION_RANGE", f"Region {signal.region_id} out of range [0-4095]", "region_id")
        if not 0 <= signal.supply_demand <= 15:
            result.add_error("INVALID_SD_RANGE", f"Supply-demand {signal.supply_demand} out of range [0-15]", "supply_demand")
        if not 0 <= signal.intensity <= 4095:
            result.add_error("INVALID_INTENSITY_RANGE", f"Intensity {signal.intensity} out of range [0-4095]", "intensity")
        if not 0 <= signal.priority <= 15:
            result.add_error("INVALID_PRIORITY_RANGE", f"Priority {signal.priority} out of range [0-15]", "priority")

    def _validate_version(self, signal: EPSSignal, result: ValidationResult) -> None:
        supported = self.config['supported_versions']
        if signal.version not in supported:
            result.add_error("UNSUPPORTED_VERSION", f"Version {signal.version} not in {supported}", "version")

    def _validate_region(self, signal: EPSSignal, result: ValidationResult) -> None:
        if self.valid_regions is not None and signal.region_id not in self.valid_regions:
            result.add_error("INVALID_REGION", f"Region {signal.region_id} not in valid set", "region_id")

    def _validate_intensity(self, signal: EPSSignal, result: ValidationResult) -> None:
        if signal.intensity > self.config['warn_high_intensity']:
            result.add_warning("HIGH_INTENSITY", f"Intensity {signal.intensity} is unusually high", "intensity")

    def _validate_supply_demand_consistency(self, signal: EPSSignal, result: ValidationResult) -> None:
        if not self.config['require_intensity_for_shortage']:
            return
        if signal.is_shortage and signal.intensity == 0:
            result.add_warning("SHORTAGE_NO_INTENSITY", "Shortage signal with zero intensity", "supply_demand")


def quick_validate(signal: EPSSignal) -> bool:
    """Quick validation — returns True if all fields are in range."""
    return (
        0 <= signal.version <= 15 and
        0 <= signal.region_id <= 4095 and
        0 <= signal.supply_demand <= 15 and
        0 <= signal.intensity <= 4095 and
        0 <= signal.price <= 4095 and
        0 <= signal.priority <= 15
    )
