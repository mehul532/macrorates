"""
Systematic Treasury Relative-Value Strategy Pipeline.

Provides high-level orchestration for:
- 2s10s DV01-neutral steepener/flattener
- 2s5s10s DV01-neutral butterfly
- Multi-baseline scorecard execution:
  1. Baseline 0: Cash (No-trade)
  2. Baseline 1: Simple Slope / Fly Z-score
  3. Baseline 2: Dynamic Nelson-Siegel (DNS) latent factor residual
  4. Main Signal: Macro-Conditioned DNS signal
- Institutional V1 cost model accounting (fees, slippage, rolls, margin buffer)
- Return attribution across Level, Slope, and Curvature factors
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.strategy.signals import generate_all_signals
from src.strategy.portfolio import compute_continuous_positions, DEFAULT_FUTURES_DV01
from src.strategy.backtest import RelativeValueBacktestEngine, CostModelV1Config, BacktestResult
from src.strategy.attribution import run_factor_attribution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystematicRelativeValuePipeline:
    """
    End-to-end systematic Treasury curve relative-value pipeline.
    """
    
    def __init__(
        self,
        yield_panel_path: Path = Path("data/processed/yield_panel.parquet"),
        factor_panel_path: Path = Path("data/processed/factor_panel.parquet"),
        macro_surprises_path: Path = Path("data/processed/macro_surprises.parquet"),
        cost_config: Optional[CostModelV1Config] = None,
        initial_capital: float = 10_000_000.0,
        target_dv01: float = 10_000.0,
    ):
        self.yield_panel_path = Path(yield_panel_path)
        self.factor_panel_path = Path(factor_panel_path)
        self.macro_surprises_path = Path(macro_surprises_path)
        self.cost_config = cost_config or CostModelV1Config()
        self.initial_capital = initial_capital
        self.target_dv01 = target_dv01
        
        self.engine = RelativeValueBacktestEngine(
            cost_config=self.cost_config,
            initial_capital=self.initial_capital,
        )
        
        # Loaded datasets
        self.yield_df: Optional[pd.DataFrame] = None
        self.factor_df: Optional[pd.DataFrame] = None
        self.macro_df: Optional[pd.DataFrame] = None
        self.signals_df: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """Load and harmonize processed datasets."""
        logger.info("Loading processed term structure and macro datasets...")
        self.yield_df = pd.read_parquet(self.yield_panel_path)
        if "date" in self.yield_df.columns:
            self.yield_df["date"] = pd.to_datetime(self.yield_df["date"])
            self.yield_df = self.yield_df.sort_values("date").set_index("date")
            
        self.factor_df = pd.read_parquet(self.factor_panel_path)
        if "date" in self.factor_df.columns:
            self.factor_df["date"] = pd.to_datetime(self.factor_df["date"])
            self.factor_df = self.factor_df.sort_values("date").set_index("date")
            
        self.macro_df = pd.read_parquet(self.macro_surprises_path)
        if "date" in self.macro_df.columns:
            self.macro_df["date"] = pd.to_datetime(self.macro_df["date"])
            
        logger.info(
            "Data loaded successfully: %d yield days, %d factor days, %d macro events.",
            len(self.yield_df), len(self.factor_df), len(self.macro_df)
        )

    def generate_signals(self, window: int = 60, macro_weight: float = 0.5) -> pd.DataFrame:
        """Generate signals across all baselines and main models."""
        if self.yield_df is None or self.factor_df is None or self.macro_df is None:
            self.load_data()
            
        self.signals_df = generate_all_signals(
            yield_df=self.yield_df,
            factor_df=self.factor_df,
            macro_df=self.macro_df,
            window=window,
            macro_weight=macro_weight,
        )
        return self.signals_df

    def run_all_backtests(
        self,
        target_dv01: Optional[float] = None,
    ) -> Dict[str, BacktestResult]:
        """
        Run side-by-side backtests for 2s10s and 2s5s10s across all models:
        1. 2s10s Cash (Baseline 0)
        2. 2s10s Slope Z-Score (Baseline 1)
        3. 2s10s DNS Factor (Baseline 2)
        4. 2s10s Macro-Conditioned DNS (Main Model)
        5. Butterfly Cash (Baseline 0)
        6. Butterfly Curvature Z-Score (Baseline 1)
        7. Butterfly DNS Factor (Baseline 2)
        8. Butterfly Macro-Conditioned DNS (Main Model)
        """
        if self.signals_df is None:
            self.generate_signals()
            
        t_dv01 = target_dv01 or self.target_dv01
        results = {}
        
        strategies_to_run = [
            # Strategy Name, signal_column, curve_type
            ("2s10s_Cash_Baseline", "signal_2s10s_cash", "2s10s"),
            ("2s10s_Slope_ZScore", "signal_2s10s_zscore", "2s10s"),
            ("2s10s_DNS_Factor", "signal_2s10s_dns", "2s10s"),
            ("2s10s_Macro_DNS", "signal_2s10s_macro_dns", "2s10s"),
            ("Fly_Cash_Baseline", "signal_fly_cash", "2s5s10s"),
            ("Fly_Curvature_ZScore", "signal_fly_zscore", "2s5s10s"),
            ("Fly_DNS_Factor", "signal_fly_dns", "2s5s10s"),
            ("Fly_Macro_DNS", "signal_fly_macro_dns", "2s5s10s"),
        ]
        
        for name, sig_col, str_type in strategies_to_run:
            logger.info("Executing backtest for %s...", name)
            sig_series = self.signals_df[sig_col]
            pos_df = compute_continuous_positions(
                signals_series=sig_series,
                strategy_type=str_type,
                target_dv01=t_dv01,
            )
            
            res = self.engine.run_strategy(
                strategy_name=name,
                positions_df=pos_df,
                yield_df=self.yield_df,
            )
            results[name] = res
            
        return results

    def build_scorecard(self, results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """
        Build comparative scorecard across all strategies.
        """
        rows = []
        for name, res in results.items():
            m = res.metrics
            rows.append({
                "Strategy": name,
                "Total Return (%)": m.get("total_return_pct", 0.0),
                "CAGR (%)": m.get("cagr_pct", 0.0),
                "Annual Vol (%)": m.get("annualized_vol_pct", 0.0),
                "Sharpe Ratio": m.get("sharpe_ratio", 0.0),
                "Sortino Ratio": m.get("sortino_ratio", 0.0),
                "Max Drawdown (%)": m.get("max_drawdown_pct", 0.0),
                "Calmar Ratio": m.get("calmar_ratio", 0.0),
                "Cost Drag (bp/yr)": m.get("cost_drag_bp_annual", 0.0),
                "Net PnL ($)": m.get("total_net_pnl_usd", 0.0),
            })
        return pd.DataFrame(rows).set_index("Strategy")
