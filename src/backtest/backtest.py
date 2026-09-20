"""
Institutional Backtesting & Attribution Engine.

Re-exports core V1 backtest engine and provides:
- Economic episode performance attribution (config/episodes.yaml)
- Regime robustness slicing across empirical threshold rules
- Comprehensive risk metrics (Sharpe, Sortino, MaxDD, Turnover, PnL/DV01, PnL/Turnover)
- Full unbundled attribution chain:
  Gross P&L -> Transaction Costs -> Roll Costs -> Cash Interest -> Net P&L
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.strategy.backtest import (
    CostModelV1Config,
    BacktestResult,
    RelativeValueBacktestEngine,
)
from src.backtest.regimes import load_episodes_config, tag_fed_regimes, slice_metrics_by_regime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_episode_performance(
    backtest_result: BacktestResult,
    config_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Decompose strategy performance across named economic episodes from config/episodes.yaml.
    """
    config = load_episodes_config(config_path)
    returns = backtest_result.daily_returns
    pnl_df = backtest_result.pnl_components
    
    records = []
    
    # 1. Named Economic Episodes
    for ep_id, ep in config.get("episodes", {}).items():
        s_date = pd.to_datetime(ep["start_date"])
        e_date = pd.to_datetime(ep["end_date"])
        
        mask = (returns.index >= s_date) & (returns.index <= e_date)
        sub_rets = returns.loc[mask]
        sub_pnl = pnl_df.loc[mask]
        
        if len(sub_rets) < 5:
            continue
            
        n_days = len(sub_rets)
        mean_ret = float(sub_rets.mean() * 252.0)
        ann_vol = float(sub_rets.std() * np.sqrt(252.0))
        sharpe = mean_ret / (ann_vol + 1e-8) if ann_vol > 0 else 0.0
        
        cum = (1.0 + sub_rets).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = float(dd.min() * 100.0)
        
        records.append({
            "Period Type": "Named Episode",
            "Episode": ep["name"],
            "Start": ep["start_date"],
            "End": ep["end_date"],
            "Days": n_days,
            "Total PnL ($)": round(float(sub_pnl["net_pnl"].sum()), 2),
            "Ann. Return (%)": round(mean_ret * 100.0, 2),
            "Ann. Vol (%)": round(ann_vol * 100.0, 2),
            "Sharpe Ratio": round(sharpe, 3),
            "Max Drawdown (%)": round(max_dd, 2),
            "Gross PnL ($)": round(float(sub_pnl["gross_pnl"].sum()), 2),
            "Trade Costs ($)": round(float(sub_pnl["trade_cost"].sum()), 2),
            "Roll Costs ($)": round(float(sub_pnl["roll_cost"].sum()), 2),
        })
        
    # 2. Calendar Sub-Periods
    for cal_id, cal in config.get("calendar_subperiods", {}).items():
        s_date = pd.to_datetime(cal["start_date"])
        e_date = pd.to_datetime(cal["end_date"])
        
        mask = (returns.index >= s_date) & (returns.index <= e_date)
        sub_rets = returns.loc[mask]
        sub_pnl = pnl_df.loc[mask]
        
        if len(sub_rets) < 5:
            continue
            
        n_days = len(sub_rets)
        mean_ret = float(sub_rets.mean() * 252.0)
        ann_vol = float(sub_rets.std() * np.sqrt(252.0))
        sharpe = mean_ret / (ann_vol + 1e-8) if ann_vol > 0 else 0.0
        
        cum = (1.0 + sub_rets).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = float(dd.min() * 100.0)
        
        records.append({
            "Period Type": "Calendar Period",
            "Episode": cal["name"],
            "Start": cal["start_date"],
            "End": cal["end_date"],
            "Days": n_days,
            "Total PnL ($)": round(float(sub_pnl["net_pnl"].sum()), 2),
            "Ann. Return (%)": round(mean_ret * 100.0, 2),
            "Ann. Vol (%)": round(ann_vol * 100.0, 2),
            "Sharpe Ratio": round(sharpe, 3),
            "Max Drawdown (%)": round(max_dd, 2),
            "Gross PnL ($)": round(float(sub_pnl["gross_pnl"].sum()), 2),
            "Trade Costs ($)": round(float(sub_pnl["trade_cost"].sum()), 2),
            "Roll Costs ($)": round(float(sub_pnl["roll_cost"].sum()), 2),
        })
        
    df_res = pd.DataFrame(records)
    return df_res


def compute_regime_robustness_table(
    backtest_result: BacktestResult,
    yield_df: pd.DataFrame,
    config_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Partition strategy returns across empirical threshold rules:
    - Rate Level (Low vs High)
    - Realized Volatility (Low vs High)
    - Curve Slope (Inverted vs Normal)
    - Fed Policy Stance (Ex-Ante Hiking, Easing, ZLB, Pause/Hold)
    """
    regimes = tag_fed_regimes(yield_df, config_path=config_path)
    common_idx = backtest_result.daily_returns.index.intersection(regimes.index)
    
    rets = backtest_result.daily_returns.loc[common_idx]
    reg_df = regimes.loc[common_idx]
    
    categories = [
        ("Rate Level Threshold", "threshold_rate_level"),
        ("Realized Volatility Threshold", "threshold_realized_vol"),
        ("Curve Shape Threshold", "threshold_curve_shape"),
        ("Fed Policy Stance (Ex-Ante)", "fed_regime_ex_ante"),
    ]
    
    rows = []
    for cat_label, col in categories:
        sub_metrics = slice_metrics_by_regime(rets, reg_df[col])
        for reg_val, row in sub_metrics.iterrows():
            rows.append({
                "Threshold Dimension": cat_label,
                "Regime Partition": reg_val,
                "Days": row["Days"],
                "Ann. Return (%)": row["Ann. Return (%)"],
                "Ann. Vol (%)": row["Ann. Vol (%)"],
                "Sharpe Ratio": row["Sharpe Ratio"],
                "Sortino Ratio": row["Sortino Ratio"],
                "Max Drawdown (%)": row["Max Drawdown (%)"],
                "Win Rate (%)": row["Win Rate (%)"],
            })
            
    return pd.DataFrame(rows).set_index(["Threshold Dimension", "Regime Partition"])
