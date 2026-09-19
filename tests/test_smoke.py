"""
Smoke test suite for MacroRates data pipeline.

Ensures pipeline strictly enforces data integrity:
- Fails on any unexpected gap.
- Fails on any silent forward-fill.
"""

from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest

from src.data.gap_detector import GapDetector, is_us_bond_market_holiday


def generate_synthetic_clean_panel() -> pd.DataFrame:
    """Generate a clean synthetic Treasury yield panel with realistic holidays and gaps."""
    # Build 2 years of daily calendar dates
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    df = pd.DataFrame({"date": dates})

    # Add realistic yields on trading days, NaN on holidays/weekends
    for sid, base_yield in [("DGS2", 4.50), ("DGS10", 4.20), ("DGS30", 4.35)]:
        yields = []
        for d in dates:
            d_date = d.date()
            if d_date.weekday() >= 5 or is_us_bond_market_holiday(d_date):
                yields.append(np.nan)
            else:
                yields.append(base_yield + 0.1 * np.sin(d.dayofyear / 20.0))
        df[sid] = yields

    return df


def test_smoke_clean_panel_passes():
    """Smoke test: A clean yield panel must pass all integrity checks with zero unexpected gaps."""
    df = generate_synthetic_clean_panel()
    detector = GapDetector(df, date_col="date")

    # Check for silent fills on all tenors
    for tenor in ["DGS2", "DGS10", "DGS30"]:
        detector.check_for_silent_forward_fill(tenor)

    report = detector.generate_missingness_report(["DGS2", "DGS10", "DGS30"])
    assert report.unexpected_gaps == 0, f"Clean panel should have 0 unexpected gaps, found {report.unexpected_gaps}"
    assert report.coverage_ratio_trading_days == 1.0


def test_smoke_fails_on_unexpected_gap():
    """Smoke test: The test suite MUST FAIL if an unexpected gap exists in the yield panel."""
    df = generate_synthetic_clean_panel()

    # Inject an unexpected gap: missing yield on a normal trading Wednesday (e.g. 2024-05-15)
    target_date = pd.to_datetime("2024-05-15")
    assert df.loc[df["date"] == target_date, "DGS10"].notna().item() is True

    # Drop the observation
    df_with_gap = df.copy()
    df_with_gap.loc[df_with_gap["date"] == target_date, "DGS10"] = np.nan

    detector = GapDetector(df_with_gap, date_col="date")
    report = detector.generate_missingness_report(["DGS10"])

    # Strict Smoke Check Assertion
    with pytest.raises(AssertionError, match="Unexpected gap detected"):
        if report.unexpected_gaps > 0:
            raise AssertionError(
                f"Unexpected gap detected! Series contains {report.unexpected_gaps} unclassified gaps on dates: "
                f"{report.unexpected_gap_dates}"
            )


def test_smoke_fails_on_silent_forward_fill():
    """Smoke test: The test suite MUST FAIL if unobserved dates were silently forward-filled."""
    # Test on historical 30-year Treasury suspension (Feb 2002 to Feb 2006)
    dates = pd.date_range("2002-01-01", "2002-06-30", freq="D")
    df = pd.DataFrame({"date": dates})

    # Properly construct clean series where DGS30 is NaN after 2002-02-18
    yields = []
    for d in dates:
        d_date = d.date()
        if d_date.weekday() >= 5 or is_us_bond_market_holiday(d_date) or d_date >= date(2002, 2, 18):
            yields.append(np.nan)
        else:
            yields.append(5.40)
    df["DGS30"] = yields

    # Clean check passes
    clean_detector = GapDetector(df, date_col="date")
    clean_detector.check_for_silent_forward_fill("DGS30")

    # Now silently forward-fill across the suspension
    df_polluted = df.copy()
    df_polluted["DGS30"] = df_polluted["DGS30"].ffill()

    # Smoke test assertion: must raise ValueError and fail on the silent fill
    polluted_detector = GapDetector(df_polluted, date_col="date")
    with pytest.raises(ValueError, match="Silent forward-fill detected"):
        polluted_detector.check_for_silent_forward_fill("DGS30")
