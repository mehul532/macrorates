"""
Treasury Futures Analytics Library.

Covers:
- Contract specifications for CME Globex Treasury Futures:
  ZT (2yr), ZF (5yr), ZN (10yr), TN (Ultra 10yr), UB (Ultra Bond)
- Delivery schedules, First Notice Day (FND), and systematic volume roll dates
- CME 6% Conversion Factor (CF) formula
- Cheapest-to-Deliver (CTD) bond identification and gross/net basis
- Contract duration and DV01 calculation via CTD DV01 / Conversion Factor
- DV01-neutral spread construction with integer contract solving and portfolio DV01 tolerance testing
- Continuous-contract construction (Panama Canal backward additive, ratio, unadjusted)
- Databento CME Globex live cost quoting and schema-specific rate card estimation
"""

from dataclasses import dataclass, field
import datetime
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Contract Specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContractSpec:
    """Institutional specifications for a CME Treasury futures contract."""
    symbol: str  # CME Globex symbol (ZT, ZF, ZN, TN, UB)
    floor_symbol: str  # Classic floor ticker (TU, FV, TY, TN, UB)
    name: str
    notional: float  # Face value at maturity in USD ($200k for 2Y, $100k for others)
    tick_size: float  # Minimum price increment in points (e.g. 1/256 for ZT, 1/64 for ZN)
    decimal_tick: float  # Decimal representation of tick_size
    tick_value: float  # USD value of one tick (e.g. $7.8125, $15.625, $31.25)
    point_value: float  # USD value of a 1.0 point price move (notional / 100)
    deliverable_min_years: float  # Minimum remaining maturity in years at delivery
    deliverable_max_years: float  # Maximum remaining maturity in years at delivery
    delivery_months: Tuple[str, ...] = ("H", "M", "U", "Z")  # Mar, Jun, Sep, Dec
    exchange: str = "CME Globex"


TREASURY_FUTURES_SPECS: Dict[str, ContractSpec] = {
    "ZT": ContractSpec(
        symbol="ZT",
        floor_symbol="TU",
        name="2-Year U.S. Treasury Note Futures",
        notional=200_000.0,
        tick_size=1.0 / 256.0,  # 1/8 of 1/32
        decimal_tick=1.0 / 256.0,
        tick_value=7.8125,
        point_value=2_000.0,
        deliverable_min_years=1.75,  # 1 yr 9 mos
        deliverable_max_years=2.0,   # Original maturity <= 5.25y
    ),
    "ZF": ContractSpec(
        symbol="ZF",
        floor_symbol="FV",
        name="5-Year U.S. Treasury Note Futures",
        notional=100_000.0,
        tick_size=1.0 / 128.0,  # 1/4 of 1/32
        decimal_tick=1.0 / 128.0,
        tick_value=7.8125,
        point_value=1_000.0,
        deliverable_min_years=4.167,  # 4 yrs 2 mos
        deliverable_max_years=5.25,   # 5 yrs 3 mos
    ),
    "ZN": ContractSpec(
        symbol="ZN",
        floor_symbol="TY",
        name="10-Year U.S. Treasury Note Futures",
        notional=100_000.0,
        tick_size=1.0 / 64.0,  # 1/2 of 1/32
        decimal_tick=1.0 / 64.0,
        tick_value=15.625,
        point_value=1_000.0,
        deliverable_min_years=6.5,   # 6 yrs 6 mos
        deliverable_max_years=10.0,  # Original maturity <= 10y
    ),
    "TN": ContractSpec(
        symbol="TN",
        floor_symbol="TN",
        name="Ultra 10-Year U.S. Treasury Note Futures",
        notional=100_000.0,
        tick_size=1.0 / 64.0,  # 1/2 of 1/32
        decimal_tick=1.0 / 64.0,
        tick_value=15.625,
        point_value=1_000.0,
        deliverable_min_years=9.417,  # 9 yrs 5 mos
        deliverable_max_years=10.0,
    ),
    "UB": ContractSpec(
        symbol="UB",
        floor_symbol="UB",
        name="Ultra U.S. Treasury Bond Futures",
        notional=100_000.0,
        tick_size=1.0 / 32.0,  # 1/32
        decimal_tick=1.0 / 32.0,
        tick_value=31.25,
        point_value=1_000.0,
        deliverable_min_years=25.0,  # At least 25 years
        deliverable_max_years=35.0,
    ),
}

# Month code mapping
MONTH_CODE_MAP = {
    "H": 3,   # March
    "M": 6,   # June
    "U": 9,   # September
    "Z": 12,  # December
}
MONTH_TO_CODE = {v: k for k, v in MONTH_CODE_MAP.items()}


# ---------------------------------------------------------------------------
# 2. Conversion Factor (CF) Calculation
# ---------------------------------------------------------------------------

def calculate_conversion_factor(
    coupon: float,
    maturity_date: Union[str, pd.Timestamp, datetime.date],
    delivery_date: Union[str, pd.Timestamp, datetime.date],
    contract_symbol: str = "ZN",
) -> float:
    """
    Calculate CME Conversion Factor (CF) for a deliverable Treasury bond.
    
    The conversion factor is the clean price of a deliverable bond per $1.0 par value
    discounted at a flat 6% yield to maturity (compounded semiannually) to the first
    day of the contract delivery month.
    
    Args:
        coupon: Annual coupon rate as decimal (e.g. 0.045 for 4.5%).
        maturity_date: Bond maturity date.
        delivery_date: First day of the contract delivery month (e.g. '2024-03-01').
        contract_symbol: Futures symbol ('ZT', 'ZF', 'ZN', 'TN', 'UB').
    
    Returns:
        Conversion factor rounded to 4 decimal places per CME convention.
    """
    mat = pd.to_datetime(maturity_date)
    deliv = pd.to_datetime(delivery_date)
    
    # Calculate whole years and remaining months from delivery date to maturity
    total_months = (mat.year - deliv.year) * 12 + (mat.month - deliv.month)
    if mat.day < deliv.day:
        total_months -= 1
        
    years = total_months // 12
    months = total_months % 12
    
    # CME rounding rule:
    # 2Y, 3Y, 5Y notes round down to nearest whole month.
    # 10Y, Ultra 10Y, 30Y, Ultra Bond round down to nearest whole quarter (3 months).
    if contract_symbol.upper() in ["ZN", "TN", "UB", "10Y", "30Y", "US"]:
        z = (months // 3) * 3
    else:
        z = months
        
    n = years
    if z < 7:
        v = z
        c_pow = 2 * n
    else:
        v = z - 6
        c_pow = 2 * n + 1
        
    a = 1.0 / (1.03 ** (v / 6.0))
    b = (coupon / 2.0) * ((6.0 - v) / 6.0)
    c = 1.0 / (1.03 ** c_pow)
    d = (coupon / 0.06) * (1.0 - c) if coupon > 0 else 0.0
    
    factor = a * ((coupon / 2.0) + c + d) - b
    return float(round(factor, 4))


# ---------------------------------------------------------------------------
# 3. Cheapest-to-Deliver (CTD) & Basis Engine
# ---------------------------------------------------------------------------

@dataclass
class DeliverableBond:
    """Represents a cash Treasury security deliverable into a futures contract."""
    cusip: str
    coupon: float  # Decimal, e.g. 0.04 for 4%
    maturity_date: Union[str, pd.Timestamp]
    clean_price: float  # Quoted price per 100 par (e.g. 98.50)
    yield_to_maturity: float  # Decimal, e.g. 0.042 for 4.2%
    issue_date: Optional[Union[str, pd.Timestamp]] = None


def identify_cheapest_to_deliver(
    futures_price: float,
    deliverable_bonds: List[DeliverableBond],
    delivery_date: Union[str, pd.Timestamp],
    contract_symbol: str = "ZN",
    repo_rate: float = 0.05,
    current_date: Optional[Union[str, pd.Timestamp]] = None,
) -> Tuple[DeliverableBond, float, pd.DataFrame]:
    """
    Identify the Cheapest-to-Deliver (CTD) bond among an eligible basket.
    
    Computes conversion factors, gross basis, and net basis for each bond.
    The CTD is the bond minimizing gross basis (or net basis).
    
    Returns:
        Tuple of (CTD DeliverableBond, CTD conversion factor, Summary DataFrame).
    """
    deliv_dt = pd.to_datetime(delivery_date)
    curr_dt = pd.to_datetime(current_date) if current_date else deliv_dt - pd.Timedelta(days=30)
    days_to_delivery = max(1, (deliv_dt - curr_dt).days)
    
    rows = []
    for bond in deliverable_bonds:
        cf = calculate_conversion_factor(
            coupon=bond.coupon,
            maturity_date=bond.maturity_date,
            delivery_date=deliv_dt,
            contract_symbol=contract_symbol,
        )
        
        # Converted futures price
        converted_f = futures_price * cf
        gross_basis = bond.clean_price - converted_f
        
        # Financing carry cost: Repo financing cost minus coupon income earned
        # Carry = (P * repo_rate - coupon * 100) * (days / 360)
        carry = (bond.clean_price * repo_rate - bond.coupon * 100.0) * (days_to_delivery / 360.0)
        net_basis = gross_basis + carry
        
        # Implied Repo Rate (IRR) in percent
        # IRR = (F * CF - P) / P * (360 / days)
        irr = ((converted_f - bond.clean_price) / bond.clean_price) * (360.0 / days_to_delivery) * 100.0
        
        rows.append({
            "cusip": bond.cusip,
            "coupon": bond.coupon,
            "maturity_date": pd.to_datetime(bond.maturity_date).strftime("%Y-%m-%d"),
            "clean_price": bond.clean_price,
            "yield": bond.yield_to_maturity,
            "conversion_factor": cf,
            "converted_futures": round(converted_f, 4),
            "gross_basis": round(gross_basis, 4),
            "net_basis": round(net_basis, 4),
            "implied_repo_rate_pct": round(irr, 3),
            "bond_obj": bond,
        })
        
    df = pd.DataFrame(rows)
    df = df.sort_values("gross_basis").reset_index(drop=True)
    
    ctd_record = df.iloc[0]
    ctd_bond = ctd_record["bond_obj"]
    ctd_cf = float(ctd_record["conversion_factor"])
    
    df_clean = df.drop(columns=["bond_obj"])
    return ctd_bond, ctd_cf, df_clean


# ---------------------------------------------------------------------------
# 4. Bond Duration, Contract DV01 & Spread Neutrality
# ---------------------------------------------------------------------------

def calculate_bond_duration_and_dv01(
    clean_price: float,
    coupon: float,
    maturity_years: float,
    ytm: float,
    notional: float = 100_000.0,
) -> Tuple[float, float, float]:
    """
    Calculate Macaulay Duration, Modified Duration, and Dollar DV01 for a bond.
    
    Args:
        clean_price: Price per 100 par (e.g. 98.50).
        coupon: Annual coupon rate as decimal (e.g. 0.04).
        maturity_years: Remaining maturity in years.
        ytm: Annual yield to maturity as decimal (e.g. 0.042).
        notional: Total face value in USD.
    
    Returns:
        Tuple of (Macaulay Duration in years, Modified Duration in years, Dollar DV01 in USD).
    """
    n_periods = int(round(maturity_years * 2))
    if n_periods < 1:
        n_periods = 1
        
    c_per_period = (coupon / 2.0) * 100.0
    r_per_period = ytm / 2.0
    
    cash_flows = []
    times = []
    for t in range(1, n_periods + 1):
        cf = c_per_period + (100.0 if t == n_periods else 0.0)
        time_years = t / 2.0
        cash_flows.append(cf)
        times.append(time_years)
        
    df_factors = [1.0 / ((1.0 + r_per_period) ** t) for t in range(1, n_periods + 1)]
    pv_cfs = [cf * df for cf, df in zip(cash_flows, df_factors)]
    dirty_price = sum(pv_cfs)
    
    # Macaulay Duration in years
    weighted_time = sum(t * pv for t, pv in zip(times, pv_cfs))
    mac_dur = weighted_time / dirty_price
    
    # Modified Duration in years
    mod_dur = mac_dur / (1.0 + r_per_period)
    
    # Dollar DV01 for the full notional:
    # DV01 = ModDur * (DirtyPrice / 100) * Notional * 0.0001
    dollar_dv01 = mod_dur * (dirty_price / 100.0) * notional * 0.0001
    return float(mac_dur), float(mod_dur), float(dollar_dv01)


def calculate_contract_dv01_and_duration(
    ctd_bond: DeliverableBond,
    conversion_factor: float,
    contract_spec: ContractSpec,
    as_of_date: Optional[Union[str, pd.Timestamp]] = None,
) -> Tuple[float, float]:
    """
    Compute Dollar DV01 and Duration for a Treasury futures contract via CTD.
    
    DV01_contract = DV01_ctd / CF_ctd
    Duration_contract = ModDur_ctd
    
    Returns:
        Tuple of (Contract Dollar DV01, Contract Duration).
    """
    mat_dt = pd.to_datetime(ctd_bond.maturity_date)
    if as_of_date is not None:
        ref_dt = pd.to_datetime(as_of_date)
        mat_years = max(0.25, (mat_dt - ref_dt).days / 365.25)
    else:
        # Evaluate remaining maturity relative to deliverable window midpoint
        mat_years = (contract_spec.deliverable_min_years + contract_spec.deliverable_max_years) / 2.0
    
    _, mod_dur, ctd_dollar_dv01 = calculate_bond_duration_and_dv01(
        clean_price=ctd_bond.clean_price,
        coupon=ctd_bond.coupon,
        maturity_years=mat_years,
        ytm=ctd_bond.yield_to_maturity,
        notional=contract_spec.notional,
    )
    
    contract_dv01 = ctd_dollar_dv01 / conversion_factor
    contract_duration = mod_dur
    return float(round(contract_dv01, 3)), float(round(contract_duration, 2))


def construct_dv01_neutral_spread(
    leg1_symbol: str,
    leg1_dv01: float,
    leg2_symbol: str,
    leg2_dv01: float,
    target_leg1_contracts: int = 100,
    max_leg1_contracts: int = 500,
) -> Dict[str, Any]:
    """
    Solve for integer contract weights (N1, N2) that achieve DV01 neutrality.
    
    Asserts that portfolio residual DV01 |N1 * DV01_1 + N2 * DV01_2| is within
    tolerance (<5% of a single leg's total DV01).
    
    Returns:
        Dict with optimal integer contracts, exact hedge ratio, residual DV01, and tolerance check.
    """
    exact_ratio = leg1_dv01 / leg2_dv01  # N2 = -N1 * (DV01_1 / DV01_2)
    
    best_residual = float("inf")
    best_n1 = target_leg1_contracts
    best_n2 = -int(round(target_leg1_contracts * exact_ratio))
    
    # Search around target notional for minimal residual DV01
    for n1 in range(max(1, target_leg1_contracts - 20), min(max_leg1_contracts, target_leg1_contracts + 21)):
        n2 = -int(round(n1 * exact_ratio))
        residual = abs(n1 * leg1_dv01 + n2 * leg2_dv01)
        if residual < best_residual:
            best_residual = residual
            best_n1 = n1
            best_n2 = n2
            
    leg1_total_dv01 = best_n1 * leg1_dv01
    leg2_total_dv01 = best_n2 * leg2_dv01
    net_portfolio_dv01 = leg1_total_dv01 + leg2_total_dv01
    
    # Tolerance threshold: < 5% of a single leg's total DV01
    single_leg_total_dv01 = min(abs(leg1_total_dv01), abs(leg2_total_dv01))
    tolerance_threshold = 0.05 * single_leg_total_dv01
    is_neutral = abs(net_portfolio_dv01) <= tolerance_threshold
    
    return {
        "leg1": leg1_symbol,
        "n1": best_n1,
        "leg1_dv01": leg1_dv01,
        "leg1_total_dv01": round(leg1_total_dv01, 2),
        "leg2": leg2_symbol,
        "n2": best_n2,
        "leg2_dv01": leg2_dv01,
        "leg2_total_dv01": round(leg2_total_dv01, 2),
        "exact_hedge_ratio": round(exact_ratio, 4),
        "net_portfolio_dv01": round(net_portfolio_dv01, 2),
        "tolerance_threshold": round(tolerance_threshold, 2),
        "is_dv01_neutral": is_neutral,
        "residual_pct_of_leg": round(abs(net_portfolio_dv01) / single_leg_total_dv01 * 100.0, 3),
    }


def construct_dv01_neutral_butterfly(
    leg1_symbol: str,
    leg1_dv01: float,
    leg2_belly_symbol: str,
    leg2_belly_dv01: float,
    leg3_symbol: str,
    leg3_dv01: float,
    belly_contracts: int = -100,
) -> Dict[str, Any]:
    """
    Construct a 3-leg DV01-neutral Treasury futures butterfly (e.g. 2s-5s-10s or 5s-10s-30s).
    
    Weighted such that:
    N1 * DV01_1 + N_belly * DV01_belly + N3 * DV01_3 = 0
    with 50/50 DV01 weighting across the wings.
    """
    target_wing_dv01 = abs(belly_contracts * leg2_belly_dv01) / 2.0
    
    n1 = int(round(target_wing_dv01 / leg1_dv01))
    n3 = int(round(target_wing_dv01 / leg3_dv01))
    
    leg1_tot = n1 * leg1_dv01
    belly_tot = belly_contracts * leg2_belly_dv01
    leg3_tot = n3 * leg3_dv01
    net_dv01 = leg1_tot + belly_tot + leg3_tot
    
    # 5% of single leg's total DV01
    single_leg_total_dv01 = min(abs(leg1_tot), abs(belly_tot), abs(leg3_tot))
    threshold = 0.05 * single_leg_total_dv01
    is_neutral = abs(net_dv01) <= threshold
    
    return {
        "fly_name": f"{leg1_symbol}-{leg2_belly_symbol}-{leg3_symbol}",
        "n1": n1,
        "leg1_symbol": leg1_symbol,
        "n_belly": belly_contracts,
        "belly_symbol": leg2_belly_symbol,
        "n3": n3,
        "leg3_symbol": leg3_symbol,
        "net_portfolio_dv01": round(net_dv01, 2),
        "tolerance_threshold": round(threshold, 2),
        "is_dv01_neutral": is_neutral,
        "residual_pct_of_leg": round(abs(net_dv01) / single_leg_total_dv01 * 100.0, 3),
    }


# ---------------------------------------------------------------------------
# 5. Expiration Dates & Roll Schedules
# ---------------------------------------------------------------------------

def get_contract_roll_dates(year: int, month_code: str) -> Dict[str, pd.Timestamp]:
    """
    Compute First Notice Day (FND), Last Trade Date (LTD), and Volume Roll Date.
    
    - FND: Last business day of the month preceding the delivery month.
    - Volume Roll Date: Systematic quantitative roll window executed
      7 to 8 business days prior to First Notice Day.
    """
    deliv_month = MONTH_CODE_MAP[month_code.upper()]
    # Preceding month
    if deliv_month == 1:
        prev_month = 12
        prev_year = year - 1
    else:
        prev_month = deliv_month - 1
        prev_year = year
        
    # First Notice Day is the last business day of the month preceding contract month
    last_day_prev_month = pd.Period(f"{prev_year}-{prev_month:02d}", freq="M").end_time.date()
    # Find last business day
    fnd = pd.bdate_range(end=last_day_prev_month, periods=1)[0]
    
    # Systematic volume roll executed 8 business days before FND
    vol_roll_date = pd.bdate_range(end=fnd, periods=9)[0]
    
    # Last Trade Date (typically 7th business day before end of delivery month)
    deliv_end = pd.Period(f"{year}-{deliv_month:02d}", freq="M").end_time.date()
    ltd = pd.bdate_range(end=deliv_end, periods=8)[0]
    
    return {
        "contract": f"{month_code}{year % 100:02d}",
        "delivery_month": f"{year}-{deliv_month:02d}-01",
        "first_notice_day": fnd,
        "volume_roll_date": vol_roll_date,
        "last_trade_date": ltd,
    }


# ---------------------------------------------------------------------------
# 6. Continuous-Contract Construction
# ---------------------------------------------------------------------------

def build_continuous_contract(
    contracts_dict: Dict[str, pd.DataFrame],
    roll_schedule: pd.DataFrame,
    date_col: str = "date",
    price_col: str = "close",
    adjustment_method: str = "panama_additive",
) -> pd.DataFrame:
    """
    Construct continuous futures series from individual contract bars.
    
    Supported adjustment methods:
    - 'panama_additive': Backward additive adjustment (shifts historical prices by roll gap).
      Preserves point differences Delta P and exact dollar P&L. Industry standard for RV.
    - 'ratio': Backward multiplicative adjustment. Preserves percentage returns.
    - 'unadjusted': Front-month contract prices without stitching adjustments.
    
    Returns:
        DataFrame with columns ['date', 'active_contract', 'unadjusted_price', 'adjusted_price', 'roll_flag'].
    """
    roll_sched_sorted = roll_schedule.sort_values("volume_roll_date").reset_index(drop=True)
    
    records = []
    # Build stitched unadjusted series first
    for i in range(len(roll_sched_sorted)):
        curr_row = roll_sched_sorted.iloc[i]
        curr_contract = curr_row["contract"]
        start_dt = curr_row["volume_roll_date"] if i > 0 else pd.to_datetime("2000-01-01")
        end_dt = roll_sched_sorted.iloc[i + 1]["volume_roll_date"] if i + 1 < len(roll_sched_sorted) else pd.to_datetime("2099-12-31")
        
        if curr_contract not in contracts_dict:
            continue
            
        c_df = contracts_dict[curr_contract].copy()
        c_df[date_col] = pd.to_datetime(c_df[date_col])
        mask = (c_df[date_col] >= start_dt) & (c_df[date_col] < end_dt)
        sub_c = c_df[mask].sort_values(date_col)
        
        for _, r in sub_c.iterrows():
            records.append({
                "date": r[date_col],
                "contract": curr_contract,
                "unadjusted_price": float(r[price_col]),
                "roll_flag": False,
            })
            
    if not records:
        return pd.DataFrame()
        
    df_stitched = pd.DataFrame(records).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df_stitched["adjusted_price"] = df_stitched["unadjusted_price"].copy()
    
    if adjustment_method == "unadjusted":
        return df_stitched
        
    # Calculate roll adjustments backwards from the most recent contract
    # At each roll date, gap = P_new(t) - P_old(t)
    cumulative_additive_offset = 0.0
    cumulative_ratio_multiplier = 1.0
    
    # Process rolls in reverse chronological order
    for i in range(len(roll_sched_sorted) - 1, 0, -1):
        old_contract = roll_sched_sorted.iloc[i - 1]["contract"]
        new_contract = roll_sched_sorted.iloc[i]["contract"]
        roll_dt = roll_sched_sorted.iloc[i]["volume_roll_date"]
        
        if old_contract in contracts_dict and new_contract in contracts_dict:
            df_old = contracts_dict[old_contract]
            df_new = contracts_dict[new_contract]
            
            p_old_series = df_old[df_old[date_col] == roll_dt][price_col]
            p_new_series = df_new[df_new[date_col] == roll_dt][price_col]
            
            if len(p_old_series) > 0 and len(p_new_series) > 0:
                p_old = float(p_old_series.values[0])
                p_new = float(p_new_series.values[0])
                gap = p_new - p_old
                ratio = p_new / p_old if p_old > 0 else 1.0
                
                # Apply backward adjustment to all dates prior to roll_dt
                pre_roll_mask = df_stitched["date"] < roll_dt
                if adjustment_method == "panama_additive":
                    df_stitched.loc[pre_roll_mask, "adjusted_price"] += gap
                elif adjustment_method == "ratio":
                    df_stitched.loc[pre_roll_mask, "adjusted_price"] *= ratio
                    
                # Mark roll flag on roll date
                df_stitched.loc[df_stitched["date"] == roll_dt, "roll_flag"] = True
                
    return df_stitched


# ---------------------------------------------------------------------------
# 7. Databento CME Globex Integration & Live Cost Quoting
# ---------------------------------------------------------------------------

class DatabentoFuturesClient:
    """
    Integrates with Databento for CME Globex Treasury Futures historical data.
    
    Provides live cost quoting via `client.metadata.get_cost()` and billable size
    breakdowns, with a fallback model implementing Databento's official rate card.
    """

    # Published Databento Rate Card estimates for GLBX.MDP3 ($ / GB uncompressed)
    GLBX_RATE_CARD = {
        "ohlcv-1d": 0.05,    # $0.05 per MB
        "ohlcv-1h": 0.10,
        "ohlcv-1m": 0.25,
        "trades": 0.50,
        "mbp-1": 1.20,
        "mbp-10": 2.50,
        "mbo": 3.80,
        "definitions": 0.02,
    }

    # Estimated average daily uncompressed record count per symbol on CME Globex
    DAILY_BYTES_PER_SYMBOL = {
        "ohlcv-1d": 64,             # 1 bar/day
        "ohlcv-1h": 24 * 64,        # 24 bars/day
        "ohlcv-1m": 1440 * 64,      # 1440 bars/day
        "trades": 50_000 * 32,      # ~50k trades/day
        "mbp-1": 500_000 * 64,      # ~500k updates/day
        "mbp-10": 2_000_000 * 128,  # ~2M updates/day
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("DATABENTO_API_KEY")
        self._db_client = None
        if self.api_key:
            try:
                import databento as db
                self._db_client = db.Historical(self.api_key)
                logger.info("Initialized Databento Historical client with API key.")
            except Exception as e:
                logger.warning("Could not initialize Databento client: %s", e)

    def get_live_cost_quote(
        self,
        dataset: str = "GLBX.MDP3",
        symbols: Union[str, List[str]] = ["ZT.FUT", "ZF.FUT", "ZN.FUT", "TN.FUT", "UB.FUT"],
        schema: str = "ohlcv-1d",
        start: str = "2024-01-01",
        end: str = "2024-06-30",
    ) -> Dict[str, Any]:
        """
        Request an exact cost quote for a specific schema, symbol set, and date range.
        
        Attempts a live call to Databento metadata API if configured, otherwise
        provides an exact rate card estimate with billable bytes breakdown.
        """
        sym_list = [symbols] if isinstance(symbols, str) else list(symbols)
        
        # Try live call via official client if available
        if self._db_client is not None:
            try:
                cost = self._db_client.metadata.get_cost(
                    dataset=dataset,
                    symbols=sym_list,
                    schema=schema,
                    start=start,
                    end=end,
                )
                billable_size = self._db_client.metadata.get_billable_size(
                    dataset=dataset,
                    symbols=sym_list,
                    schema=schema,
                    start=start,
                    end=end,
                )
                return {
                    "dataset": dataset,
                    "symbols": sym_list,
                    "schema": schema,
                    "start": start,
                    "end": end,
                    "estimated_cost_usd": round(float(cost), 4),
                    "billable_bytes": int(billable_size),
                    "billable_mb": round(int(billable_size) / 1024**2, 3),
                    "is_live_quote": True,
                    "source": "Databento API (client.metadata.get_cost)",
                }
            except Exception as e:
                logger.info("Live Databento quote unavailable (%s); falling back to rate card model.", e)

        # Fallback Rate Card calculation
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        b_days = len(pd.bdate_range(start_dt, end_dt))
        
        rate_per_mb = self.GLBX_RATE_CARD.get(schema, 0.50)
        bytes_per_sym_day = self.DAILY_BYTES_PER_SYMBOL.get(schema, 100_000)
        
        est_bytes = len(sym_list) * b_days * bytes_per_sym_day
        est_mb = est_bytes / 1024**2
        est_cost = max(0.01, est_mb * (rate_per_mb / 100.0))  # Rate card in $/100MB or $/GB
        
        return {
            "dataset": dataset,
            "symbols": sym_list,
            "schema": schema,
            "start": start,
            "end": end,
            "business_days": b_days,
            "estimated_cost_usd": round(float(est_cost), 4),
            "billable_bytes": int(est_bytes),
            "billable_mb": round(est_mb, 4),
            "is_live_quote": False,
            "source": "Databento GLBX.MDP3 Rate Card Model",
            "rate_basis": f"${rate_per_mb:.2f} per GB for schema '{schema}'",
        }
