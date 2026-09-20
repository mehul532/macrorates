"""
Duration-Matched ETF Prototyping Module.

================================================================================
CRITICAL DISCLAIMER:
ENGINEERING VALIDATION ONLY -- NEVER PRESENT ETF NUMBERS AS EVIDENCE FOR THE
FUTURES STRATEGY. THIS MODULE SERVES EXCLUSIVELY AS A PROTOTYPE SANITY-CHECK
FOR SPREAD AND BUTTERFLY REBALANCING ARITHMETIC.
================================================================================

Covers:
- Duration-matched ETF specifications: SHY (1-3Y), IEI (3-7Y), IEF (7-10Y), TLT (20+Y)
- Synthetic constant-maturity total-return generation from FRED CMT yields
- ETF DV01-neutral spread and butterfly capital allocation
- Mandatory "ENGINEERING VALIDATION ONLY" metadata tagging on all outputs
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENGINEERING_VALIDATION_TAG = "ENGINEERING VALIDATION ONLY -- DO NOT USE FOR FUTURES EVIDENCE"


@dataclass(frozen=True)
class ETFSpec:
    symbol: str
    benchmark_tenor: str
    target_maturity_years: float
    effective_duration: float  # Modified duration in years
    dollar_dv01_per_100k: float  # Dollar DV01 for $100,000 market value


ETF_SPECS: Dict[str, ETFSpec] = {
    "SHY": ETFSpec(
        symbol="SHY",
        benchmark_tenor="2Y",
        target_maturity_years=2.0,
        effective_duration=1.86,
        dollar_dv01_per_100k=18.60,
    ),
    "IEI": ETFSpec(
        symbol="IEI",
        benchmark_tenor="5Y",
        target_maturity_years=5.0,
        effective_duration=4.25,
        dollar_dv01_per_100k=42.50,
    ),
    "IEF": ETFSpec(
        symbol="IEF",
        benchmark_tenor="10Y",
        target_maturity_years=10.0,
        effective_duration=7.45,
        dollar_dv01_per_100k=74.50,
    ),
    "TLT": ETFSpec(
        symbol="TLT",
        benchmark_tenor="20Y+",
        target_maturity_years=25.0,
        effective_duration=16.50,
        dollar_dv01_per_100k=165.00,
    ),
}


def generate_synthetic_etf_panel(
    yield_df: pd.DataFrame,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """
    Generate synthetic constant-maturity duration-matched ETF price series from CMT yields.
    
    Price return formula:
    R_t = -D_mod * Delta_y_t + 0.5 * Convexity * (Delta_y_t)^2 + y_{t-1} * (dt / 360)
    
    NOTE: ENGINEERING VALIDATION ONLY.
    """
    df = yield_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        
    tenor_map = {
        "SHY": ("DGS2", ETF_SPECS["SHY"].effective_duration, 0.05),
        "IEI": ("DGS5", ETF_SPECS["IEI"].effective_duration, 0.25),
        "IEF": ("DGS10", ETF_SPECS["IEF"].effective_duration, 0.70),
        "TLT": ("DGS20", ETF_SPECS["TLT"].effective_duration, 3.50),
    }
    
    price_df = pd.DataFrame(index=df.index)
    
    for etf_sym, (col, dur, conv) in tenor_map.items():
        if col not in df.columns:
            continue
        y = df[col] / 100.0  # Decimal yield
        dy = y.diff()
        
        # Total daily return
        dt = 1.0 / 252.0
        daily_ret = -dur * dy + 0.5 * conv * (dy ** 2) + y.shift(1) * dt
        daily_ret = daily_ret.fillna(0.0)
        
        # Cumulative price
        price = start_price * (1.0 + daily_ret).cumprod()
        price_df[etf_sym] = price
        price_df[f"{etf_sym}_ret"] = daily_ret
        
    price_df.attrs["validation_tag"] = ENGINEERING_VALIDATION_TAG
    price_df.attrs["is_engineering_validation"] = True
    return price_df


def allocate_etf_2s10s_spread(
    signal: float,
    target_dv01: float = 1_000.0,
    shy_spec: ETFSpec = ETF_SPECS["SHY"],
    ief_spec: ETFSpec = ETF_SPECS["IEF"],
) -> Dict[str, Any]:
    """
    Construct duration-matched ETF 2s10s DV01-neutral spread.
    
    WARNING: ENGINEERING VALIDATION ONLY.
    """
    if abs(signal) < 1e-4:
        return {
            "validation_tag": ENGINEERING_VALIDATION_TAG,
            "strategy": "ETF_2s10s_spread",
            "signal": 0.0,
            "shy_weight": 0.0,
            "ief_weight": 0.0,
            "net_dv01": 0.0,
            "is_dv01_neutral": True,
        }
        
    sign = 1 if signal > 0 else -1
    scaled_dv01 = target_dv01 * abs(signal)
    
    # Capital required per leg to achieve scaled_dv01
    shy_capital = (scaled_dv01 / shy_spec.dollar_dv01_per_100k) * 100_000.0
    ief_capital = (scaled_dv01 / ief_spec.dollar_dv01_per_100k) * 100_000.0
    
    shy_signed = sign * shy_capital
    ief_signed = -sign * ief_capital
    
    shy_dv01 = (shy_signed / 100_000.0) * shy_spec.dollar_dv01_per_100k
    ief_dv01 = (ief_signed / 100_000.0) * ief_spec.dollar_dv01_per_100k
    net_dv01 = shy_dv01 + ief_dv01
    
    return {
        "validation_tag": ENGINEERING_VALIDATION_TAG,
        "is_engineering_validation": True,
        "strategy": "ETF_2s10s_spread",
        "signal": signal,
        "direction": "steepener" if sign > 0 else "flattener",
        "shy_capital": round(shy_signed, 2),
        "ief_capital": round(ief_signed, 2),
        "shy_dv01": round(shy_dv01, 2),
        "ief_dv01": round(ief_dv01, 2),
        "net_dv01": round(net_dv01, 2),
        "is_dv01_neutral": abs(net_dv01) < 0.01 * scaled_dv01,
    }


def allocate_etf_2s5s10s_butterfly(
    signal: float,
    target_dv01: float = 1_000.0,
    shy_spec: ETFSpec = ETF_SPECS["SHY"],
    iei_spec: ETFSpec = ETF_SPECS["IEI"],
    ief_spec: ETFSpec = ETF_SPECS["IEF"],
) -> Dict[str, Any]:
    """
    Construct duration-matched ETF 2s-5s-10s DV01-neutral butterfly.
    
    Wings: SHY (2Y), IEF (10Y). Belly: IEI (5Y).
    50/50 DV01 weighting across wings.
    
    WARNING: ENGINEERING VALIDATION ONLY.
    """
    if abs(signal) < 1e-4:
        return {
            "validation_tag": ENGINEERING_VALIDATION_TAG,
            "strategy": "ETF_2s5s10s_butterfly",
            "signal": 0.0,
            "shy_capital": 0.0,
            "iei_capital": 0.0,
            "ief_capital": 0.0,
            "net_dv01": 0.0,
            "is_dv01_neutral": True,
        }
        
    sign = 1 if signal > 0 else -1
    scaled_dv01 = target_dv01 * abs(signal)
    
    # Belly DV01 matches scaled_dv01; each wing gets 50%
    iei_capital = (scaled_dv01 / iei_spec.dollar_dv01_per_100k) * 100_000.0
    wing_dv01 = 0.5 * scaled_dv01
    shy_capital = (wing_dv01 / shy_spec.dollar_dv01_per_100k) * 100_000.0
    ief_capital = (wing_dv01 / ief_spec.dollar_dv01_per_100k) * 100_000.0
    
    # Long Fly: Short Belly (sign * -1), Long Wings (sign * +1)
    iei_signed = -sign * iei_capital
    shy_signed = sign * shy_capital
    ief_signed = sign * ief_capital
    
    shy_dv01 = (shy_signed / 100_000.0) * shy_spec.dollar_dv01_per_100k
    iei_dv01 = (iei_signed / 100_000.0) * iei_spec.dollar_dv01_per_100k
    ief_dv01 = (ief_signed / 100_000.0) * ief_spec.dollar_dv01_per_100k
    net_dv01 = shy_dv01 + iei_dv01 + ief_dv01
    
    return {
        "validation_tag": ENGINEERING_VALIDATION_TAG,
        "is_engineering_validation": True,
        "strategy": "ETF_2s5s10s_butterfly",
        "signal": signal,
        "direction": "long_fly" if sign > 0 else "short_fly",
        "shy_capital": round(shy_signed, 2),
        "iei_capital": round(iei_signed, 2),
        "ief_capital": round(ief_signed, 2),
        "shy_dv01": round(shy_dv01, 2),
        "iei_dv01": round(iei_dv01, 2),
        "ief_dv01": round(ief_dv01, 2),
        "net_dv01": round(net_dv01, 2),
        "is_dv01_neutral": abs(net_dv01) < 0.01 * scaled_dv01,
    }
