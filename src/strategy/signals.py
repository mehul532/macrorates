"""
Systematic Relative-Value Signal Generation Engine.

Implements:
1. Baseline 0: Cash / No-trade benchmark
2. Baseline 1: Simple Rolling Slope / Fly Z-Score mean-reversion rule
3. Baseline 2: Dynamic Nelson-Siegel (DNS) Kalman factor residual signal
4. Main Signal: Macro-Conditioned DNS signal combining latent factor mispricing
   with contemporaneous macro announcement surprise impulses (CPI, NFP, FOMC)
   from Milestone 4.

Strict Zero-Lookahead Guarantee:
All rolling means, standard deviations, and event shocks use data strictly <= t.
"""

import logging
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Milestone 4 Empirical Betas (from contemporaneous HAC regressions)
# Response of Slope and Curvature factors in basis points per +1.0 sigma surprise
MACRO_EVENT_BETAS = {
    "CPI": {
        "slope_beta": -0.26,       # Mild flattening (p=0.64)
        "curvature_beta": 4.72,    # Strong belly cheapening (p=0.008)
    },
    "CORE_CPI": {
        "slope_beta": -0.45,
        "curvature_beta": 3.85,
    },
    "NFP": {
        "slope_beta": -1.39,       # Strong bear flattening (p=0.002)
        "curvature_beta": 4.38,    # Belly cheapens (p=0.014)
    },
    "FOMC": {
        "slope_beta": -1.63,       # Strong bear flattening (p=0.002)
        "curvature_beta": 3.73,
    },
}


def compute_slope_zscore_signals(
    yield_df: pd.DataFrame,
    tenor_short: str = "DGS2",
    tenor_belly: str = "DGS5",
    tenor_long: str = "DGS10",
    window: int = 60,
    min_periods: int = 20,
    z_clip: float = 2.0,
) -> pd.DataFrame:
    """
    Compute Baseline 1: Simple rolling Slope and Curvature Z-Score signals.
    
    Slope = y_long - y_short (e.g. 10Y - 2Y).
    Z_slope = (Slope - MA_window(Slope)) / Std_window(Slope).
    Signal_slope = -clip(Z_slope / z_clip, -1.0, 1.0)
      -> When slope is unusually flat (Z < 0), signal is positive (Steepener: Long 2Y, Short 10Y).
      -> When slope is unusually steep (Z > 0), signal is negative (Flattener: Short 2Y, Long 10Y).
      
    Butterfly Curvature = 2 * y_belly - y_short - y_long (e.g. 2*5Y - 2Y - 10Y).
    Z_curv = (Curv - MA_window(Curv)) / Std_window(Curv).
    Signal_fly = -clip(Z_curv / z_clip, -1.0, 1.0)
      -> When belly is rich (yield low, Curv < 0), signal is positive (Long fly: Short belly, Long wings).
      -> When belly is cheap (yield high, Curv > 0), signal is negative (Short fly: Long belly, Short wings).
    """
    df = yield_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        
    slope = df[tenor_long] - df[tenor_short]
    slope_ma = slope.rolling(window=window, min_periods=min_periods).mean()
    slope_std = slope.rolling(window=window, min_periods=min_periods).std()
    z_slope = (slope - slope_ma) / (slope_std + 1e-6)
    signal_slope = -np.clip(z_slope / z_clip, -1.0, 1.0)
    
    # Curvature: 2 * 5Y - 2Y - 10Y
    curvature = 2.0 * df[tenor_belly] - df[tenor_short] - df[tenor_long]
    curv_ma = curvature.rolling(window=window, min_periods=min_periods).mean()
    curv_std = curvature.rolling(window=window, min_periods=min_periods).std()
    z_curv = (curvature - curv_ma) / (curv_std + 1e-6)
    signal_fly = -np.clip(z_curv / z_clip, -1.0, 1.0)
    
    res = pd.DataFrame(index=df.index)
    res["yield_slope"] = slope
    res["z_slope"] = z_slope
    res["signal_2s10s_zscore"] = signal_slope.fillna(0.0)
    
    res["yield_curvature"] = curvature
    res["z_curvature"] = z_curv
    res["signal_fly_zscore"] = signal_fly.fillna(0.0)
    
    return res


def compute_dns_factor_signals(
    factor_df: pd.DataFrame,
    slope_col: str = "kf_slope",
    curvature_col: str = "kf_curvature",
    window: int = 60,
    min_periods: int = 20,
    z_clip: float = 2.0,
) -> pd.DataFrame:
    """
    Compute Baseline 2: Dynamic Nelson-Siegel (DNS) latent factor residual signals.
    
    Uses Kalman filtered/smoothed Slope and Curvature latent factors.
    In Nelson-Siegel, beta_slope loading is (1 - e^(-lambda*tau)) / (lambda*tau).
    A decrease in beta_slope corresponds to curve flattening / inversion.
    Mean-reversion towards the rolling factor mean generates steepener/flattener
    and butterfly signals.
    """
    df = factor_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        
    s = df[slope_col]
    s_ma = s.rolling(window=window, min_periods=min_periods).mean()
    s_std = s.rolling(window=window, min_periods=min_periods).std()
    z_slope = (s - s_ma) / (s_std + 1e-6)
    # When slope factor is low/negative (flattened curve), expect mean reversion to steepen (+1)
    signal_slope = -np.clip(z_slope / z_clip, -1.0, 1.0)
    
    c = df[curvature_col]
    c_ma = c.rolling(window=window, min_periods=min_periods).mean()
    c_std = c.rolling(window=window, min_periods=min_periods).std()
    z_curv = (c - c_ma) / (c_std + 1e-6)
    # When curvature factor is low (belly rich), expect mean reversion higher (+1: Long fly)
    signal_fly = -np.clip(z_curv / z_clip, -1.0, 1.0)
    
    res = pd.DataFrame(index=df.index)
    res["dns_slope"] = s
    res["z_dns_slope"] = z_slope
    res["signal_2s10s_dns"] = signal_slope.fillna(0.0)
    
    res["dns_curvature"] = c
    res["z_dns_curv"] = z_curv
    res["signal_fly_dns"] = signal_fly.fillna(0.0)
    
    return res


def compute_macro_surprise_signals(
    dates: pd.DatetimeIndex,
    macro_df: pd.DataFrame,
    surprise_col: str = "surprise_ann",
    decay_halflife: int = 5,
    min_surprise_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compute Macro Surprise Signals from announcement surprises (S_ann).
    
    Transforms point-in-time announcement surprises into continuous daily
    curve positioning impulses using empirical betas:
    - NFP and FOMC hawkish surprises -> bear flattening impulse -> Flattener (-1)
    - CPI hot surprises -> belly cheapening impulse -> Long fly (+1)
    
    Shocks decay with exponential half-life over subsequent business days,
    consistent with the Jordà local projection results from Milestone 4.
    """
    m_df = macro_df.copy()
    if "date" in m_df.columns:
        m_df["date"] = pd.to_datetime(m_df["date"])
        
    res = pd.DataFrame(index=dates)
    res["macro_slope_impulse"] = 0.0
    res["macro_curvature_impulse"] = 0.0
    
    # Aggregate announcements per date
    for dt, group in m_df.groupby("date"):
        if dt not in res.index:
            continue
        slope_impulse = 0.0
        curv_impulse = 0.0
        for _, row in group.iterrows():
            ind = row["indicator"]
            surp = row[surprise_col]
            if pd.isna(surp) or abs(surp) < min_surprise_threshold:
                continue
            betas = MACRO_EVENT_BETAS.get(ind, {"slope_beta": 0.0, "curvature_beta": 0.0})
            slope_impulse += betas["slope_beta"] * surp
            curv_impulse += betas["curvature_beta"] * surp
            
        res.loc[dt, "macro_slope_impulse"] = slope_impulse
        res.loc[dt, "macro_curvature_impulse"] = curv_impulse
        
    # Apply exponential decay to simulate persistent market impulse (EWMA)
    alpha = 1.0 - np.exp(-np.log(2.0) / decay_halflife)
    res["macro_slope_decay"] = res["macro_slope_impulse"].ewm(alpha=alpha, adjust=False).mean()
    res["macro_curv_decay"] = res["macro_curvature_impulse"].ewm(alpha=alpha, adjust=False).mean()
    
    # Scale into [-1.0, 1.0] signal
    # Negative slope impulse -> flattening -> signal -1
    res["signal_2s10s_macro"] = np.clip(res["macro_slope_decay"] / 2.0, -1.0, 1.0)
    # Positive curvature impulse -> belly cheapens -> Long fly (short belly, long wings) -> signal +1
    res["signal_fly_macro"] = np.clip(res["macro_curv_decay"] / 4.0, -1.0, 1.0)
    
    return res


def generate_all_signals(
    yield_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    window: int = 60,
    macro_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Generate unified signals dataframe containing:
    1. Baseline 0: No-trade (zero position)
    2. Baseline 1: Simple Slope / Fly Z-score
    3. Baseline 2: DNS Factor Residual
    4. Main Signal: Macro-Conditioned DNS Signal (50% DNS residual + 50% Macro shock)
    """
    # 1. Slope Z-Score
    df_zscore = compute_slope_zscore_signals(yield_df, window=window)
    
    # 2. DNS Factor
    df_dns = compute_dns_factor_signals(factor_df, window=window)
    
    # Combine on common dates
    common_idx = df_zscore.index.intersection(df_dns.index)
    df_zscore = df_zscore.loc[common_idx]
    df_dns = df_dns.loc[common_idx]
    
    # 3. Macro surprise impulses
    df_macro = compute_macro_surprise_signals(common_idx, macro_df)
    
    signals = pd.DataFrame(index=common_idx)
    signals.index.name = "date"
    
    # Baseline 0: No-Trade
    signals["signal_2s10s_cash"] = 0.0
    signals["signal_fly_cash"] = 0.0
    
    # Baseline 1: Simple Z-Score
    signals["signal_2s10s_zscore"] = df_zscore["signal_2s10s_zscore"]
    signals["signal_fly_zscore"] = df_zscore["signal_fly_zscore"]
    
    # Baseline 2: DNS Factor Residual
    signals["signal_2s10s_dns"] = df_dns["signal_2s10s_dns"]
    signals["signal_fly_dns"] = df_dns["signal_fly_dns"]
    
    # Macro Component
    signals["signal_2s10s_macro_only"] = df_macro["signal_2s10s_macro"]
    signals["signal_fly_macro_only"] = df_macro["signal_fly_macro"]
    
    # Main Signal: Macro-Conditioned DNS
    # Weighted combination of DNS factor residual and macro shock impulse
    dns_weight = 1.0 - macro_weight
    signals["signal_2s10s_macro_dns"] = np.clip(
        dns_weight * df_dns["signal_2s10s_dns"] + macro_weight * df_macro["signal_2s10s_macro"],
        -1.0, 1.0
    )
    signals["signal_fly_macro_dns"] = np.clip(
        dns_weight * df_dns["signal_fly_dns"] + macro_weight * df_macro["signal_fly_macro"],
        -1.0, 1.0
    )
    
    return signals
