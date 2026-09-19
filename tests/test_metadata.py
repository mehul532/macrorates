"""Tests for dataset metadata creation, validation, and JSON serialization."""

from pathlib import Path
import tempfile
from src.data.metadata import DatasetMetadata, MissingnessReport, create_metadata_record


def test_metadata_record_creation_and_roundtrip():
    """Verify metadata records can be created, saved to JSON, and reloaded identically."""
    missingness = MissingnessReport(
        total_calendar_days=1000,
        business_days=714,
        observed_trading_days=680,
        expected_holidays=34,
        series_not_started_days=0,
        known_discontinuity_days=0,
        unexpected_gaps=0,
        coverage_ratio_trading_days=1.0,
        maturity_coverage={"DGS10": 1.0, "DGS2": 1.0},
        unexpected_gap_dates=[],
    )

    metadata = create_metadata_record(
        source="FRED",
        series=["DGS2", "DGS10"],
        units="Percent per annum",
        frequency="Business daily",
        first_observation="2020-01-02",
        last_observation="2022-12-30",
        missingness_report=missingness,
        transformation="None",
        extra={"model_class": "CMT"},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_metadata.json"
        metadata.save_json(json_path)

        assert json_path.exists()
        loaded = DatasetMetadata.load_json(json_path)

        assert loaded.source == "FRED"
        assert loaded.series == ["DGS2", "DGS10"]
        assert loaded.missingness.unexpected_gaps == 0
        assert loaded.missingness.observed_trading_days == 680
        assert loaded.missingness.coverage_ratio_trading_days == 1.0
        assert loaded.extra["model_class"] == "CMT"
