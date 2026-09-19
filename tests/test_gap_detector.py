"""Tests for yield curve calendar gap detection and integrity auditing."""

from datetime import date
import numpy as np
import pandas as pd
import pytest

from src.data.gap_detector import (
    DateClassification,
    GapDetector,
    classify_date_observation,
    is_in_discontinuity,
    is_us_bond_market_holiday,
)


def test_holiday_detection():
    """Verify federal holidays and SIFMA market closures are correctly identified."""
    # Independence Day
    assert is_us_bond_market_holiday(date(2024, 7, 4)) is True
    # Christmas Day
    assert is_us_bond_market_holiday(date(2024, 12, 25)) is True
    # Good Friday (Easter Friday)
    assert is_us_bond_market_holiday(date(2024, 3, 29)) is True
    # Normal trading day
    assert is_us_bond_market_holiday(date(2024, 7, 2)) is False


def test_known_discontinuity_classification():
    """Verify known issuance suspensions are recognized."""
    # 30Y Treasury suspended in 2003
    d_30y_suspended = date(2003, 6, 15)
    assert is_in_discontinuity("DGS30", d_30y_suspended) is True

    # 30Y Treasury active in 2024
    d_30y_active = date(2024, 6, 15)
    assert is_in_discontinuity("DGS30", d_30y_active) is False


def test_classify_date_observation():
    """Verify date classification logic for trading day, holiday, discontinuity, and unexpected gap."""
    # 1. Trading day with observed value
    cls_obs = classify_date_observation("DGS10", date(2024, 5, 1), 4.60)
    assert cls_obs == DateClassification.TRADING_DAY

    # 2. Holiday with NaN
    cls_holiday = classify_date_observation("DGS10", date(2024, 7, 4), np.nan)
    assert cls_holiday == DateClassification.EXPECTED_HOLIDAY

    # 3. Known discontinuity for 30Y in 2004 with NaN
    cls_disc = classify_date_observation("DGS30", date(2004, 3, 10), np.nan)
    assert cls_disc == DateClassification.KNOWN_DISCONTINUITY

    # 4. Pre-inception for 1M prior to 2001
    cls_pre = classify_date_observation("DGS1MO", date(1998, 3, 10), np.nan)
    assert cls_pre == DateClassification.SERIES_NOT_STARTED

    # 5. Missing value on a normal trading Wednesday -> UNEXPECTED_GAP
    cls_gap = classify_date_observation("DGS10", date(2024, 5, 8), np.nan)
    assert cls_gap == DateClassification.UNEXPECTED_GAP


def test_silent_forward_fill_raises_error():
    """Verify that forward-filling across a known discontinuity raises ValueError."""
    # Create sample dataframe with 30Y yields
    dates = pd.date_range("2002-02-01", "2002-03-15", freq="D")
    df = pd.DataFrame({"date": dates, "DGS30": np.nan})

    # Legitimate state: during suspension, DGS30 is NaN
    detector = GapDetector(df, date_col="date")
    detector.check_for_silent_forward_fill("DGS30")  # Should not raise

    # Now simulate silent forward-fill: fill the suspension period with a stale quote
    df_filled = df.copy()
    df_filled.loc[df_filled["date"] == "2002-02-25", "DGS30"] = 5.40

    detector_filled = GapDetector(df_filled, date_col="date")
    with pytest.raises(ValueError, match="Silent forward-fill detected"):
        detector_filled.check_for_silent_forward_fill("DGS30")
