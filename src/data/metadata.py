"""Metadata record schema and serialization for MacroRates processed datasets."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MissingnessReport:
    """Statistical summary of dataset observation coverage and gap breakdown."""

    total_calendar_days: int
    business_days: int
    observed_trading_days: int
    expected_holidays: int
    series_not_started_days: int
    known_discontinuity_days: int
    unexpected_gaps: int
    coverage_ratio_trading_days: float
    maturity_coverage: Dict[str, float] = field(default_factory=dict)
    unexpected_gap_dates: List[str] = field(default_factory=list)


@dataclass
class DatasetMetadata:
    """Formal metadata record describing every processed MacroRates dataset."""

    source: str
    retrieval_timestamp: str
    series: List[str]
    units: str
    frequency: str
    first_observation: str
    last_observation: str
    missingness: MissingnessReport
    transformation: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata record to nested dictionary."""
        return asdict(self)

    def save_json(self, path: Path) -> None:
        """Serialize metadata to a formatted JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "DatasetMetadata":
        """Load metadata from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        miss_data = data["missingness"]
        data["missingness"] = MissingnessReport(**miss_data)
        return cls(**data)


def create_metadata_record(
    source: str,
    series: List[str],
    units: str,
    frequency: str,
    first_observation: str,
    last_observation: str,
    missingness_report: MissingnessReport,
    transformation: str,
    extra: Optional[Dict[str, Any]] = None,
) -> DatasetMetadata:
    """Factory helper to construct an immutable metadata record with current UTC timestamp."""
    retrieval_timestamp = datetime.now(timezone.utc).isoformat()
    return DatasetMetadata(
        source=source,
        retrieval_timestamp=retrieval_timestamp,
        series=series,
        units=units,
        frequency=frequency,
        first_observation=first_observation,
        last_observation=last_observation,
        missingness=missingness_report,
        transformation=transformation,
        extra=extra or {},
    )
