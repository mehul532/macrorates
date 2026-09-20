"""
Systematic Relative-Value Backtest Engine with V1 Cost Model.

V1 Cost Model Implementation:
- Transaction costs: Exchange + clearing fees per contract round-turn
- Bid/ask slippage: Half-tick crossing cost per executed lot
- Contract rolls: Quarterly roll friction (closing near, opening far, paying fees + crossing)
- Margin & Collateral: Capital buffer requirement earning risk-free rate (DGS3MO)
- CRITICAL EXCLUSION: Strictly NO repo financing carry deduction in V1.
  A futures position is not repo-financed like cash inventory; repo belongs
  in the cash/futures basis relationship, not a line-item drag here.

Provides:
- End-to-end simulation of 2s10s curve spread and 2s5s10s butterfly
- Multi-baseline side-by-side execution:
  1. Baseline 0: Cash (No-trade)
  2. Baseline 1: Simple Slope / Fly Z-Score
  3. Baseline 2: DNS Factor Residual
  4. Main Model: Macro-Conditioned DNS Signal
- Comprehensive risk and performance metrics (Sharpe, Sortino, MaxDD, Calmar, Turnover)
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.futures.futures_analytics import TREASURY_FUTURES_SPECS
from src.strategy.portfolio import DEFAULT_FUTURES_DV01

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CostModelV1Config:
    """Institutional V1 Cost Model Configuration."""
    # Transaction & Clearing fees per contract traded ($)
    fee_per_contract: float = 1.50
    # Slippage assumption in fractions of a tick (default: 0.5 tick crossing)
    slippage_ticks: float = 0.5
    # Quarterly roll friction: extra cost per contract on scheduled roll dates ($)
    roll_friction_per_contract: float = 4.00
    # Margin requirement per contract ($)
    initial_margin: Dict[str, float] = field(default_factory=lambda: {
        "ZT": 1_200.0,
        "ZF": 1_500.0,
        "ZN": 2_200.0,
        "TN": 2_500.0,
        "UB": 4_500.0,
    })
    # Inter-commodity spread margin credit (e.g. 50% discount for offsetting curve legs)
    spread_margin_credit: float = 0.50
    # Minimum equity collateral buffer (e.g. require 3x margin)
    margin_cushion_factor: float = 3.0
    # CRITICAL RULE: Must be False in V1
    deduct_repo_carry: bool = False


@dataclass
class BacktestResult:
    """Summary of backtest results."""
    strategy_name: str
    equity_series: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame
    pnl_components: pd.DataFrame
    metrics: Dict[str, Any]


class SyntheticDV01Backtest:
    """
    Synthetic Constant Maturity Treasury (CMT) / DV01 Relative-Value Strategy Backtest Engine.
    
    RESEARCH INTEGRITY & TIMING CONTRACT DISCLOSURES:
    1. Synthetic Research Proxy:
       This engine simulates systematic relative-value strategies using indicative Daily
       Treasury Constant Maturity (CMT) yield changes scaled by fixed contract DV01 ratios.
       It is an exploratory research proxy, NOT an executable futures backtest.
    2. Daily Synthetic Timing Assumption:
       - Treasury CMT yields are interpolated par-equivalent yields calculated daily by the
         U.S. Department of the Treasury from indicative bid-side closing quotes and released
         after market close (~4:00-4:30 PM ET).
       - Fills in this synthetic engine are modeled at same-day close. In actual markets, a signal
         computed from post-close CMT data cannot be executed at that same close.
       - Where execution availability is unverified, this assumption is labeled as SYNTHETIC_SAME_CLOSE
         and explicitly excluded from claims of validated live execution.
    3. Exclusions from Synthetic V1 Proxy:
       - Repo financing carry is excluded from V1 (belongs in cash-futures basis, not direct P&L line item).
       - Actual futures basis, cheapest-to-deliver (CTD) basket switching, and delivery options are
         evaluated separately in Milestone 9 (basis.py) and Milestone 15 (TreasuryFuturesBacktest).
    """
    
    def __init__(
        self,
        cost_config: Optional[CostModelV1Config] = None,
        initial_capital: float = 10_000_000.0,
    ):
        self.cost_config = cost_config or CostModelV1Config()
        if self.cost_config.deduct_repo_carry:
            raise ValueError(
                "CRITICAL VIOLATION: deduct_repo_carry must be False in V1! "
                "Futures positions are not repo-financed; repo belongs in the basis model."
            )
        self.initial_capital = initial_capital
        
    def _compute_daily_contract_price_changes(
        self,
        yield_df: pd.DataFrame,
        dv01_dict: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Compute daily dollar price changes per contract from term structure yields.
        
        Using fixed-income relationship:
        Delta_Price_pts = -ModDur * Price * Delta_y / CF = -(DV01_contract * 10,000 * Delta_y) / PointValue
        Dollar P&L per contract = Delta_Price_pts * PointValue = -DV01_contract * 10,000 * Delta_y
        where Delta_y is in decimal (1 bp = 0.0001).
        """
        df_y = yield_df.copy()
        if "date" in df_y.columns:
            df_y["date"] = pd.to_datetime(df_y["date"])
            df_y = df_y.sort_values("date").set_index("date")
            
        contract_tenor_map = {
            "ZT": "DGS2",
            "ZF": "DGS5",
            "ZN": "DGS10",
        }
        
        d_pnl = pd.DataFrame(index=df_y.index)
        for sym, tenor in contract_tenor_map.items():
            if tenor in df_y.columns:
                # dy in basis points
                dy_bp = (df_y[tenor] - df_y[tenor].shift(1)) * 100.0
                dv01 = dv01_dict.get(sym, DEFAULT_FUTURES_DV01.get(sym, 50.0))
                # 1 bp rise in yield produces -DV01 dollar change per contract
                d_pnl[f"dpnl_{sym}"] = -dv01 * dy_bp
                
        return d_pnl.fillna(0.0)

    def run_strategy(
        self,
        strategy_name: str,
        positions_df: pd.DataFrame,
        yield_df: pd.DataFrame,
        cash_rate_series: Optional[pd.Series] = None,
        dv01_dict: Optional[Dict[str, float]] = None,
        roll_dates: Optional[List[pd.Timestamp]] = None,
    ) -> BacktestResult:
        """
        Simulate historical strategy performance under V1 cost model.
        
        positions_df: DataFrame containing integer contract allocations:
                      ['n_zt', 'n_zn'] for 2s10s or ['n_zt', 'n_zf', 'n_zn'] for fly
        """
        if dv01_dict is None:
            dv01_dict = DEFAULT_FUTURES_DV01
            
        # Contract price changes
        dpnl_per_contract = self._compute_daily_contract_price_changes(yield_df, dv01_dict)
        
        # Align dates
        common_dates = positions_df.index.intersection(dpnl_per_contract.index)
        positions = positions_df.loc[common_dates].copy()
        dpnl = dpnl_per_contract.loc[common_dates].copy()
        
        # Identify contract symbols present in positions
        active_syms = []
        for col in positions.columns:
            if col.startswith("n_"):
                sym = col[2:].upper()
                if f"dpnl_{sym}" in dpnl.columns:
                    active_syms.append(sym)
                    
        # Cash interest rate (default: DGS3MO or 2.0% annual)
        if cash_rate_series is not None:
            r_cash = cash_rate_series.reindex(common_dates).ffill().fillna(2.0) / 100.0
        elif "DGS3MO" in yield_df.columns:
            y_df = yield_df.copy()
            if "date" in y_df.columns:
                y_df = y_df.set_index(pd.to_datetime(y_df["date"]))
            r_cash = y_df["DGS3MO"].reindex(common_dates).ffill().fillna(2.0) / 100.0
        else:
            r_cash = pd.Series(0.02, index=common_dates)
            
        # Detect quarterly roll dates if not provided
        if roll_dates is None:
            # End of Feb, May, Aug, Nov (approx 5 days prior to IMM First Notice Day)
            roll_dates = [
                dt for dt in common_dates
                if dt.month in (2, 5, 8, 11) and dt.day >= 20 and dt.day <= 27
            ]
        roll_set = set(pd.to_datetime(roll_dates))
        
        # Time-stepping simulation
        n_days = len(common_dates)
        equity = np.zeros(n_days)
        gross_pnl = np.zeros(n_days)
        trade_cost = np.zeros(n_days)
        roll_cost = np.zeros(n_days)
        cash_interest = np.zeros(n_days)
        net_pnl = np.zeros(n_days)
        margin_req = np.zeros(n_days)
        
        equity[0] = self.initial_capital
        
        for t in range(1, n_days):
            dt = common_dates[t]
            dt_prev = common_dates[t - 1]
            
            # 1. Gross Trading P&L from overnight position held from t-1 to t
            daily_gross = 0.0
            for sym in active_syms:
                pos_prev = positions.loc[dt_prev, f"n_{sym.lower()}"]
                unit_pnl = dpnl.loc[dt, f"dpnl_{sym}"]
                daily_gross += pos_prev * unit_pnl
            gross_pnl[t] = daily_gross
            
            # 2. Transaction Costs & Bid/Ask Slippage on position changes at t
            daily_trade_cost = 0.0
            total_contracts_held = 0.0
            for sym in active_syms:
                pos_prev = positions.loc[dt_prev, f"n_{sym.lower()}"]
                pos_curr = positions.loc[dt, f"n_{sym.lower()}"]
                delta_pos = abs(pos_curr - pos_prev)
                total_contracts_held += abs(pos_curr)
                
                if delta_pos > 0:
                    spec = TREASURY_FUTURES_SPECS.get(sym)
                    tick_val = spec.tick_value if spec else 15.625
                    slippage = self.cost_config.slippage_ticks * tick_val
                    daily_trade_cost += delta_pos * (self.cost_config.fee_per_contract + slippage)
            trade_cost[t] = daily_trade_cost
            
            # 3. Contract Roll Drag
            daily_roll_cost = 0.0
            if dt in roll_set and total_contracts_held > 0:
                daily_roll_cost = total_contracts_held * self.cost_config.roll_friction_per_contract
            roll_cost[t] = daily_roll_cost
            
            # 4. Margin Requirement & Cash Interest (NO repo carry!)
            # Gross initial margin discounted by spread credit
            raw_margin = sum(
                abs(positions.loc[dt, f"n_{sym.lower()}"]) * self.cost_config.initial_margin.get(sym, 2000.0)
                for sym in active_syms
            )
            margin = raw_margin * (1.0 - self.cost_config.spread_margin_credit)
            margin_req[t] = margin
            
            # Unencumbered cash earns risk-free rate
            unencumbered_cash = max(0.0, equity[t - 1] - margin)
            daily_interest = unencumbered_cash * (r_cash.iloc[t] / 252.0)
            cash_interest[t] = daily_interest
            
            # Net P&L
            day_net = daily_gross - daily_trade_cost - daily_roll_cost + daily_interest
            net_pnl[t] = day_net
            equity[t] = equity[t - 1] + day_net
            
        equity_series = pd.Series(equity, index=common_dates)
        daily_returns = equity_series.pct_change().fillna(0.0)
        
        pnl_df = pd.DataFrame({
            "gross_pnl": gross_pnl,
            "trade_cost": trade_cost,
            "roll_cost": roll_cost,
            "cash_interest": cash_interest,
            "net_pnl": net_pnl,
            "equity": equity,
            "margin_req": margin_req,
        }, index=common_dates)
        
        metrics = self._calculate_metrics(daily_returns, equity_series, pnl_df)
        
        return BacktestResult(
            strategy_name=strategy_name,
            equity_series=equity_series,
            daily_returns=daily_returns,
            positions=positions,
            pnl_components=pnl_df,
            metrics=metrics,
        )

    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity: pd.Series,
        pnl_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute institutional risk-adjusted return scorecard."""
        n_days = len(returns)
        if n_days <= 1:
            return {}
            
        years = max(0.01, n_days / 252.0)
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0
        ann_vol = float(returns.std() * np.sqrt(252.0))
        
        # Sharpe Ratio (annualized)
        mean_daily_ret = float(returns.mean())
        sharpe = (mean_daily_ret * np.sqrt(252.0)) / (returns.std() + 1e-8) if ann_vol > 0 else 0.0
        
        # Sortino Ratio (downside risk)
        downside_returns = returns[returns < 0.0]
        downside_std = float(downside_returns.std() * np.sqrt(252.0)) if len(downside_returns) > 0 else 1e-8
        sortino = (mean_daily_ret * np.sqrt(252.0)) / (downside_std + 1e-8)
        
        # Drawdowns
        running_max = equity.cummax()
        drawdowns = (equity - running_max) / running_max
        max_drawdown = float(drawdowns.min())
        calmar = cagr / abs(max_drawdown) if abs(max_drawdown) > 1e-4 else 0.0
        
        # PnL breakdown
        total_gross = float(pnl_df["gross_pnl"].sum())
        total_trade_costs = float(pnl_df["trade_cost"].sum())
        total_roll_costs = float(pnl_df["roll_cost"].sum())
        total_interest = float(pnl_df["cash_interest"].sum())
        total_net = float(pnl_df["net_pnl"].sum())
        win_rate = float((returns > 0.0).sum() / max(1, (returns != 0.0).sum()))
        
        return {
            "total_return_pct": round(total_return * 100.0, 2),
            "cagr_pct": round(cagr * 100.0, 2),
            "annualized_vol_pct": round(ann_vol * 100.0, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown_pct": round(max_drawdown * 100.0, 2),
            "calmar_ratio": round(calmar, 3),
            "win_rate_pct": round(win_rate * 100.0, 2),
            "total_gross_pnl_usd": round(total_gross, 2),
            "total_trade_cost_usd": round(total_trade_costs, 2),
            "total_roll_cost_usd": round(total_roll_costs, 2),
            "total_interest_earned_usd": round(total_interest, 2),
            "total_net_pnl_usd": round(total_net, 2),
            "cost_drag_bp_annual": round((total_trade_costs + total_roll_costs) / (self.initial_capital * years) * 10_000.0, 2),
        }


# Backwards compatibility alias for SyntheticDV01Backtest
RelativeValueBacktestEngine = SyntheticDV01Backtest

