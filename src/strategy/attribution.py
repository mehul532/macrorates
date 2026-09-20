"""
Return Attribution & Risk Decomposition Engine.

Decomposes relative-value portfolio returns across:
- Level factor innovations (Delta L): Asserts near-zero sensitivity (|beta_L| ~ 0, t-stat insignificant)
- Slope factor innovations (Delta S): Drives 2s10s spread P&L
- Curvature factor innovations (Delta C): Drives 2s5s10s butterfly P&L
- Cost & roll drag: Quantifies transaction friction and roll degradation
"""

import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_factor_attribution(
    pnl_series: pd.Series,
    factor_df: pd.DataFrame,
    level_col: str = "kf_level",
    slope_col: str = "kf_slope",
    curvature_col: str = "kf_curvature",
) -> Dict[str, Any]:
    """
    Regress portfolio returns/PnL on latent factor changes:
    PnL_t = alpha + beta_L * Delta L_t + beta_S * Delta S_t + beta_C * Delta C_t + eps_t
    
    Verifies that Level beta is statistically negligible (confirming DV01 neutrality).
    """
    df_f = factor_df.copy()
    if "date" in df_f.columns:
        df_f["date"] = pd.to_datetime(df_f["date"])
        df_f = df_f.sort_values("date").set_index("date")
        
    common_idx = pnl_series.index.intersection(df_f.index)
    pnl = pnl_series.loc[common_idx]
    
    # Factor daily changes
    dL = df_f.loc[common_idx, level_col].diff().fillna(0.0)
    dS = df_f.loc[common_idx, slope_col].diff().fillna(0.0)
    dC = df_f.loc[common_idx, curvature_col].diff().fillna(0.0)
    
    X = pd.DataFrame({
        "const": 1.0,
        "delta_level": dL,
        "delta_slope": dS,
        "delta_curvature": dC,
    }, index=common_idx)
    
    model = sm.OLS(pnl, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    
    # Calculate variance decomposition
    pred_level = model.params["delta_level"] * dL
    pred_slope = model.params["delta_slope"] * dS
    pred_curv = model.params["delta_curvature"] * dC
    
    tot_var = pnl.var()
    r2 = model.rsquared
    
    return {
        "alpha": float(model.params["const"]),
        "alpha_pvalue": float(model.pvalues["const"]),
        "beta_level": float(model.params["delta_level"]),
        "beta_level_tstat": float(model.tvalues["delta_level"]),
        "beta_level_pvalue": float(model.pvalues["delta_level"]),
        "beta_slope": float(model.params["delta_slope"]),
        "beta_slope_tstat": float(model.tvalues["delta_slope"]),
        "beta_slope_pvalue": float(model.pvalues["delta_slope"]),
        "beta_curvature": float(model.params["delta_curvature"]),
        "beta_curvature_tstat": float(model.tvalues["delta_curvature"]),
        "beta_curvature_pvalue": float(model.pvalues["delta_curvature"]),
        "rsquared": float(r2),
        "is_level_neutral": abs(model.tvalues["delta_level"]) < 2.0,  # Insignificant at 5%
    }
