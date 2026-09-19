"""Data integrity and gap detection routine for U.S. Treasury yield curves."""

from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.data.metadata import MissingnessReport


class DateClassification(str, Enum):
    """Classification of a single date in the yield curve calendar."""

    TRADING_DAY = "TRADING_DAY"
    EXPECTED_HOLIDAY = "EXPECTED_HOLIDAY"
    SERIES_NOT_STARTED = "SERIES_NOT_STARTED"
    KNOWN_DISCONTINUITY = "KNOWN_DISCONTINUITY"
    UNEXPECTED_GAP = "UNEXPECTED_GAP"


# Inception dates for standard constant-maturity Treasury series
SERIES_INCEPTION_DATES: Dict[str, str] = {
    "DGS1MO": "2001-07-31",
    "DGS3MO": "1981-09-01",
    "DGS6MO": "1981-09-01",
    "DGS1": "1962-01-02",
    "DGS2": "1976-06-01",
    "DGS3": "1962-01-02",
    "DGS5": "1962-01-02",
    "DGS7": "1969-07-01",
    "DGS10": "1962-01-02",
    "DGS20": "1962-01-02",
    "DGS30": "1977-02-15",
}

# Known statutory / Treasury issuance suspensions: (start_date_inclusive, end_date_inclusive)
KNOWN_DISCONTINUITIES: Dict[str, List[Tuple[str, str]]] = {
    "DGS30": [
        # Treasury suspended 30-year bond issuance during federal budget surplus expectations
        ("2002-02-18", "2006-02-08"),
    ],
    "DGS20": [
        # Treasury suspended 20-year bond issuance at end of 1986, reintroduced Oct 1993
        ("1987-01-01", "1993-09-30"),
    ],
}

# Historical market holidays & extraordinary closures
EXTRAORDINARY_MARKET_CLOSURES: Set[str] = {
    # Funerals & National Mourning
    "1963-11-25",  # JFK Funeral
    "1968-04-09",  # MLK Day of Mourning
    "1969-03-31",  # Eisenhower Funeral
    "1972-12-28",  # Truman Funeral
    "1973-01-25",  # LBJ Funeral
    "1973-12-24",  # Special Christmas Eve closure
    "1994-04-27",  # Nixon Funeral
    "2004-06-11",  # Reagan Funeral
    "2007-01-02",  # Ford Funeral
    "2018-12-05",  # George H.W. Bush Funeral
    # Historic events & Weather
    "1969-07-21",  # Apollo 11 Lunar Landing Day of Participation
    "1977-07-14",  # NYC Blackout
    "1981-12-09",  # Exceptional Treasury holiday / blizzard
    "1985-01-21",  # Reagan 2nd Inauguration extreme cold closure
    "1985-09-27",  # Hurricane Gloria
    "2001-09-11",  # 9/11 Attacks
    "2001-09-12",  # 9/11 Attacks
    "2001-09-13",  # 9/11 Attacks
    "2012-10-29",  # Hurricane Sandy
    "2012-10-30",  # Hurricane Sandy
}


def _get_easter_date(year: int) -> date:
    """Compute Easter Sunday using Butcher's algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def is_us_bond_market_holiday(d: date) -> bool:
    """Determine if a weekday is an official Federal Reserve / SIFMA bond market holiday."""
    iso = d.isoformat()
    if iso in EXTRAORDINARY_MARKET_CLOSURES:
        return True

    year = d.year
    month = d.month
    day = d.day
    weekday = d.weekday()  # Monday=0, Sunday=6

    # 1. New Year's Day (Jan 1; observed on Mon if Sun, or Fri Dec 31 if Sat)
    if month == 1 and day == 1:
        return True
    if month == 1 and day == 2 and weekday == 0:
        return True
    if month == 12 and day == 31 and weekday == 4:
        return True

    # 2. Lincoln's Birthday (Feb 12; observed as market holiday prior to 1971)
    if year <= 1971 and month == 2 and day == 12:
        return True
    if year <= 1971 and month == 2 and day == 13 and weekday == 0:
        return True

    # 3. Washington's Birthday / Presidents' Day
    # Pre-1971: Fixed on Feb 22. Post-1971: Third Monday of February.
    if year < 1971:
        if month == 2 and day == 22:
            return True
        if month == 2 and day == 23 and weekday == 0:
            return True
        if month == 2 and day == 21 and weekday == 4:
            return True
    else:
        if month == 2 and weekday == 0 and 15 <= day <= 21:
            return True

    # 4. Martin Luther King Jr. Day (Third Monday of January, starting 1986)
    if year >= 1986 and month == 1 and weekday == 0 and 15 <= day <= 21:
        return True

    # 5. Good Friday (SIFMA recommended bond market close)
    easter = _get_easter_date(year)
    good_friday = easter - timedelta(days=2)
    if d == good_friday:
        return True

    # 6. Memorial Day (Pre-1971: May 30; Post-1971: Last Monday of May)
    if year < 1971:
        if month == 5 and day == 30:
            return True
        if month == 5 and day == 31 and weekday == 0:
            return True
        if month == 5 and day == 29 and weekday == 4:
            return True
    else:
        if month == 5 and weekday == 0 and day >= 25:
            return True

    # 7. Juneteenth (June 19, observed from 2021 onwards)
    if year >= 2021:
        if month == 6 and day == 19:
            return True
        if month == 6 and day == 20 and weekday == 0:
            return True
        if month == 6 and day == 18 and weekday == 4:
            return True

    # 8. Independence Day (July 4)
    if month == 7 and day == 4:
        return True
    if month == 7 and day == 5 and weekday == 0:
        return True
    if month == 7 and day == 3 and weekday == 4:
        return True

    # 9. Labor Day (First Monday of September)
    if month == 9 and weekday == 0 and 1 <= day <= 7:
        return True

    # 10. Columbus Day (Pre-1971: Oct 12; Post-1971: Second Monday of October)
    if year < 1971:
        if month == 10 and day == 12:
            return True
        if month == 10 and day == 13 and weekday == 0:
            return True
    elif year in (1971, 1972, 1973):
        # 1971-1973 fourth Monday of October
        if month == 10 and weekday == 0 and 22 <= day <= 28:
            return True
    else:
        if month == 10 and weekday == 0 and 8 <= day <= 14:
            return True

    # 11. Election Day (Financial markets closed on presidential and mid-term elections prior to 1980)
    if year <= 1980 and month == 11 and weekday == 1 and 2 <= day <= 8:
        return True

    # 12. Veterans Day (Nov 11)
    if month == 11 and day == 11:
        return True
    if month == 11 and day == 12 and weekday == 0:
        return True
    if month == 11 and day == 10 and weekday == 4:
        return True

    # 13. Thanksgiving Day (Fourth Thursday of November)
    if month == 11 and weekday == 3 and 22 <= day <= 28:
        return True

    # 14. Christmas Day (December 25)
    if month == 12 and day == 25:
        return True
    if month == 12 and day == 26 and weekday == 0:
        return True
    if month == 12 and day == 24 and weekday == 4:
        return True

    return False


def is_in_discontinuity(series_name: str, d: date) -> bool:
    """Check whether a date falls inside a documented statutory suspension window."""
    d_str = d.isoformat()
    for start, end in KNOWN_DISCONTINUITIES.get(series_name, []):
        if start <= d_str <= end:
            return True
    return False


def is_before_inception(series_name: str, d: date) -> bool:
    """Check whether a date is prior to series inception."""
    inception_str = SERIES_INCEPTION_DATES.get(series_name)
    if not inception_str:
        return False
    return d.isoformat() < inception_str


def classify_date_observation(
    series_name: str,
    target_date: date,
    value: Optional[float],
) -> DateClassification:
    """Classify the observation state of a specific series on a given calendar date."""
    if not pd.isna(value):
        return DateClassification.TRADING_DAY

    # It's missing (NaN or None). Determine the expected cause:
    if target_date.weekday() >= 5:  # Saturday or Sunday
        return DateClassification.EXPECTED_HOLIDAY

    if is_before_inception(series_name, target_date):
        return DateClassification.SERIES_NOT_STARTED

    if is_in_discontinuity(series_name, target_date):
        return DateClassification.KNOWN_DISCONTINUITY

    if is_us_bond_market_holiday(target_date):
        return DateClassification.EXPECTED_HOLIDAY

    return DateClassification.UNEXPECTED_GAP


class GapDetector:
    """Comprehensive gap detection and integrity auditing engine for yield panels."""

    def __init__(self, df: pd.DataFrame, date_col: str = "date"):
        """
        Initialize detector with yield panel.
        
        Args:
            df: DataFrame containing date and yield columns.
            date_col: Name of column containing datetime or ISO date strings.
        """
        self.df = df.copy()
        if date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            self.df = self.df.sort_values(date_col).reset_index(drop=True)
            self.date_col = date_col
        else:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame.")

    def audit_series(self, series_name: str) -> Dict[str, Any]:
        """
        Audit a specific yield tenor across the full calendar span.
        
        Returns:
            Dict containing detailed counts, classifications, and unexpected gap dates.
        """
        if series_name not in self.df.columns:
            raise ValueError(f"Series '{series_name}' not found in DataFrame.")

        start_date = self.df[self.date_col].min().date()
        end_date = self.df[self.date_col].max().date()

        # Map actual observations by date
        obs_map = {
            row[self.date_col].date(): row[series_name]
            for _, row in self.df.iterrows()
        }

        current = start_date
        total_days = 0
        biz_days = 0
        counts = {c: 0 for c in DateClassification}
        unexpected_dates: List[str] = []

        while current <= end_date:
            total_days += 1
            if current.weekday() < 5:
                biz_days += 1

            val = obs_map.get(current, np.nan)
            cls = classify_date_observation(series_name, current, val)
            counts[cls] += 1
            if cls == DateClassification.UNEXPECTED_GAP:
                unexpected_dates.append(current.isoformat())

            current += timedelta(days=1)

        total_trading = counts[DateClassification.TRADING_DAY]
        valid_eligible_days = (
            total_trading + counts[DateClassification.UNEXPECTED_GAP]
        )
        coverage_ratio = (
            (total_trading / valid_eligible_days) if valid_eligible_days > 0 else 1.0
        )

        return {
            "series": series_name,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_calendar_days": total_days,
            "business_days": biz_days,
            "observed_trading_days": total_trading,
            "expected_holidays": counts[DateClassification.EXPECTED_HOLIDAY],
            "series_not_started_days": counts[DateClassification.SERIES_NOT_STARTED],
            "known_discontinuity_days": counts[DateClassification.KNOWN_DISCONTINUITY],
            "unexpected_gaps": counts[DateClassification.UNEXPECTED_GAP],
            "unexpected_gap_dates": unexpected_dates,
            "coverage_ratio": coverage_ratio,
        }

    def generate_missingness_report(
        self, series_columns: List[str]
    ) -> MissingnessReport:
        """Generate an aggregated MissingnessReport across all tenors."""
        audits = [self.audit_series(s) for s in series_columns]
        mat_coverage = {a["series"]: round(a["coverage_ratio"], 4) for a in audits}
        all_unexpected: Set[str] = set()
        for a in audits:
            all_unexpected.update(a["unexpected_gap_dates"])

        # Reference tenor: DGS10 or first column
        ref = audits[0]
        for a in audits:
            if a["series"] == "DGS10":
                ref = a
                break

        return MissingnessReport(
            total_calendar_days=ref["total_calendar_days"],
            business_days=ref["business_days"],
            observed_trading_days=ref["observed_trading_days"],
            expected_holidays=ref["expected_holidays"],
            series_not_started_days=ref["series_not_started_days"],
            known_discontinuity_days=ref["known_discontinuity_days"],
            unexpected_gaps=len(all_unexpected),
            coverage_ratio_trading_days=round(
                sum(mat_coverage.values()) / len(mat_coverage), 4
            ),
            maturity_coverage=mat_coverage,
            unexpected_gap_dates=sorted(list(all_unexpected)),
        )

    def check_for_silent_forward_fill(self, series_name: str) -> None:
        """
        Detect if unavailable dates were silently forward-filled.
        
        Raises:
            ValueError: If a known discontinued period or market holiday contains non-NaN data,
                        or if an identical static quote spans across known non-trading periods.
        """
        if series_name not in self.df.columns:
            return

        # Check known discontinuities: should strictly be NaN
        for start, end in KNOWN_DISCONTINUITIES.get(series_name, []):
            mask = (self.df[self.date_col] >= start) & (self.df[self.date_col] <= end)
            filled_count = self.df.loc[mask, series_name].dropna().count()
            if filled_count > 0:
                raise ValueError(
                    f"Silent forward-fill detected! Series '{series_name}' has {filled_count} "
                    f"non-null values in statutory suspension window {start} to {end}."
                )

        # Check before inception: should strictly be NaN
        inception = SERIES_INCEPTION_DATES.get(series_name)
        if inception:
            mask_pre = self.df[self.date_col] < inception
            pre_filled = self.df.loc[mask_pre, series_name].dropna().count()
            if pre_filled > 0:
                raise ValueError(
                    f"Silent forward-fill detected! Series '{series_name}' has {pre_filled} "
                    f"non-null values prior to inception date {inception}."
                )
