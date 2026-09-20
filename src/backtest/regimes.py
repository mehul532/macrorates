"""
Regime Tagging and Empirical Threshold-Rule Robustness Cuts.

Covers:
1. Strict dual-column regime classification:
   - fed_regime_ex_ante: Knowable ONLY from backward-looking data at t (usable for conditioning)
   - fed_regime_ex_post: Descriptive attribution only, never for trading signals
2. Named economic episodes loader (config/episodes.yaml)
3. Non-clustering threshold rules:
   - Rate Level: Low Rate (<2.5%) vs High Rate (>=2.5%)
   - Realized Volatility: Low Vol vs High Vol (30-day trailing realized yield vol)
   - Curve Shape: Inverted (Slope < 0) vs Normal (Slope >= 0)
   - Fed Stance (Ex-Ante): Hiking, Easing, ZLB, Pause/Hold
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/episodes.yaml")


def load_episodes_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load named economic episodes and calendar sub-periods from YAML configuration."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        if not config_path.exists():
            config_path = Path("../config/episodes.yaml")
            
    if not config_path.exists():
        raise FileNotFoundError(f"Episodes config file not found at: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def tag_fed_regimes(
    yield_df: pd.DataFrame,
    config_path: Optional[Path] = None,
    lookback_days: int = 60,
    hiking_threshold_pct: float = 0.25,
) -> pd.DataFrame:
    """
    Tag historical dates with two strictly separated regime columns:
    
    1. fed_regime_ex_ante:
       - Knowable strictly at t using backward-looking data (trailing lookback_days).
       - Uses 3-month T-Bill (DGS3MO) or 1M CMT as short-rate policy proxy:
         * Policy rate <= 0.25% -> "ZLB"
         * Delta policy rate over lookback >= +0.25% -> "Hiking"
         * Delta policy rate over lookback <= -0.25% -> "Easing"
         * Otherwise -> "Pause/Hold"
       - Strictly usable for strategy conditioning without lookahead.
       
    2. fed_regime_ex_post:
       - Defined based on historical economic episodes from config/episodes.yaml.
       - STRICTLY FOR ATTRIBUTION ONLY -- NEVER TO BE USED FOR SIGNALS.
    """
    df = yield_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        
    # Short rate policy proxy
    policy_col = "DGS3MO" if "DGS3MO" in df.columns else ("DGS1MO" if "DGS1MO" in df.columns else "DGS2")
    short_rate = df[policy_col]
    
    # Trailing change over lookback_days
    delta_rate = short_rate - short_rate.shift(lookback_days)
    
    # 1. EX-ANTE REGIME (Strictly knowable at t)
    ex_ante = pd.Series("Pause/Hold", index=df.index, dtype=object)
    
    is_zlb = short_rate <= 0.25
    is_hiking = (delta_rate >= hiking_threshold_pct) & (~is_zlb)
    is_easing = (delta_rate <= -hiking_threshold_pct)
    
    ex_ante[is_zlb] = "ZLB"
    ex_ante[is_hiking] = "Hiking"
    ex_ante[is_easing] = "Easing"
    
    # 2. EX-POST REGIME (Descriptive attribution only)
    config = load_episodes_config(config_path)
    ex_post = pd.Series("Other/Unclassified", index=df.index, dtype=object)
    
    for ep_key, ep_info in config.get("episodes", {}).items():
        s_date = pd.to_datetime(ep_info["start_date"])
        e_date = pd.to_datetime(ep_info["end_date"])
        mask = (df.index >= s_date) & (df.index <= e_date)
        ex_post[mask] = ep_info["name"]
        
    # Combine into regime panel
    res = pd.DataFrame(index=df.index)
    res["fed_regime_ex_ante"] = ex_ante
    res["fed_regime_ex_post"] = ex_post
    
    # Add Observable Threshold Partitions (Empirical threshold rules)
    # Rate Level: Low (<2.5%) vs High (>=2.5%)
    y10 = df["DGS10"] if "DGS10" in df.columns else df.iloc[:, min(8, len(df.columns)-1)]
    res["threshold_rate_level"] = np.where(y10 < 2.5, "Low Rate (<2.5%)", "High Rate (>=2.5%)")
    
    # Curve Slope: Inverted (10Y - 2Y < 0) vs Normal (10Y - 2Y >= 0)
    y2 = df["DGS2"] if "DGS2" in df.columns else df.iloc[:, min(4, len(df.columns)-1)]
    slope_2s10s = (y10 - y2) * 100.0  # in bp
    res["threshold_curve_shape"] = np.where(slope_2s10s < 0.0, "Inverted Curve (<0 bp)", "Normal Curve (>=0 bp)")
    
    # Realized Volatility: Trailing 30-day annualized vol of 10Y yield changes
    dy10 = y10.diff()
    vol_30d = dy10.rolling(window=30, min_periods=15).std() * np.sqrt(252.0)
    median_vol = float(vol_30d.median()) if not np.isnan(vol_30d.median()) else 0.80
    res["threshold_realized_vol"] = np.where(vol_30d < median_vol, f"Low Vol (<{median_vol:.2f}%)", f"High Vol (>={median_vol:.2f}%)")
    
    res.attrs["disclaimer"] = "fed_regime_ex_post is strictly for descriptive attribution; use only fed_regime_ex_ante for conditioning."
    return res


def slice_metrics_by_regime(
    returns_series: pd.Series,
    regime_series: pd.Series,
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    """
    Compute risk and performance metrics partitioned across regime categories.
    """
    common_idx = returns_series.index.intersection(regime_series.index)
    rets = returns_series.loc[common_idx]
    regs = regime_series.loc[common_idx]
    
    records = []
    for regime_name, sub_rets in rets.groupby(regs):
        n_days = len(sub_rets)
        if n_days < 5:
            continue
            
        ann_vol = float(sub_rets.std() * np.sqrt(252.0))
        mean_ret = float(sub_rets.mean() * 252.0)
        sharpe = (mean_ret - rf_annual) / (ann_vol + 1e-8) if ann_vol > 0 else 0.0
        
        downside = sub_rets[sub_rets < 0.0]
        downside_vol = float(downside.std() * np.sqrt(252.0)) if len(downside) > 0 else 1e-8
        sortino = (mean_ret - rf_annual) / (downside_vol + 1e-8)
        
        cum_ret = (1.0 + sub_rets).cumprod()
        running_max = cum_ret.cummax()
        dd = (cum_ret - running_max) / running_max
        max_dd = float(dd.min() * 100.0)
        win_rate = float((sub_rets > 0.0).sum() / max(1, (sub_rets != 0.0).sum()) * 100.0)
        
        records.append({
            "Regime": regime_name,
            "Days": n_days,
            "Ann. Return (%)": round(mean_ret * 100.0, 2),
            "Ann. Vol (%)": round(ann_vol * 100.0, 2),
            "Sharpe Ratio": round(sharpe, 3),
            "Sortino Ratio": round(sortino, 3),
            "Max Drawdown (%)": round(max_dd, 2),
            "Win Rate (%)": round(win_rate, 2),
        })
        
    return pd.DataFrame(records).set_index("Regime")
