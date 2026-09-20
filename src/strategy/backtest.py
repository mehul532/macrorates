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
class TradeRecord:
    """Detailed record of an executed trade for audit and ledger accounting."""
    timestamp: pd.Timestamp
    symbol: str
    direction: str       # "BUY" or "SELL"
    quantity: int        # Number of contracts traded (positive int)
    price_points: float  # Price in points / ticks
    fee: float           # Exchange and clearing fees ($)
    slippage: float      # Half-tick crossing slippage ($)
    trade_type: str      # "ENTRY", "REBALANCE", "ROLL", "LIQUIDATION"

    @property
    def total_cost(self) -> float:
        return self.fee + self.slippage


class TradeLedger:
    """
    Formal trade accounting ledger tracking all executed fills, fees, slippage, and roll events.
    
    RESEARCH INTEGRITY CONTRACT (Prompt 5):
    - Turnover is derived strictly from contract quantities only (excluding signal/net_dv01 columns).
    - Initial entry costs from zero inventory are explicitly recorded.
    - Contract rolls are recorded as explicit single events per quarterly cycle.
    - Cash-ledger identity: final equity - initial capital == total collateral P&L.
    """
    def __init__(self):
        self.trades: List[TradeRecord] = []
        self.roll_events: List[Dict[str, Any]] = []

    def add_trade(self, trade: TradeRecord):
        self.trades.append(trade)

    def add_roll_event(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        contracts: int,
        friction_per_contract: float,
        total_cost: float,
    ):
        self.roll_events.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "contracts": contracts,
            "friction_per_contract": friction_per_contract,
            "total_cost": total_cost,
        })

    def total_contract_turnover(self) -> int:
        """Derive total contracts traded strictly from contract trades."""
        return sum(t.quantity for t in self.trades)

    def total_trade_fees(self) -> float:
        return sum(t.fee for t in self.trades)

    def total_slippage(self) -> float:
        return sum(t.slippage for t in self.trades)

    def total_trade_costs(self) -> float:
        return sum(t.total_cost for t in self.trades)

    def total_roll_costs(self) -> float:
        return sum(r["total_cost"] for r in self.roll_events)


def compute_proxy_roll_dates(dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """
    Compute exactly one scheduled roll date per quarterly contract cycle.
    
    DOCUMENTED PROXY SCHEDULE (Prompt 5):
    Quarterly Treasury futures (ZT, ZF, ZN, TN, UB) follow IMM cycles:
    March (H), June (M), September (U), December (Z).
    Roll friction is charged once per cycle in February, May, August, November.
    For each cycle month, the roll occurs on the FIRST trading day on or after day 20.
    This replaces repeated day-20-27 charges with exactly ONE roll event per cycle.
    """
    roll_dates = []
    dt_series = pd.Series(dates, index=dates)
    for (year, month), grp in dt_series.groupby([dates.year, dates.month]):
        if month in (2, 5, 8, 11):
            candidates = grp[grp.dt.day >= 20]
            if not candidates.empty:
                roll_dates.append(candidates.iloc[0])
    return roll_dates


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
    # Day-count convention for unencumbered cash interest ("act_360" or "bus_252")
    day_count_convention: str = "act_360"
    # Configured final liquidation at end of backtest
    liquidate_at_end: bool = False


@dataclass
class BacktestResult:
    """Summary of backtest results."""
    strategy_name: str
    equity_series: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame
    pnl_components: pd.DataFrame
    metrics: Dict[str, Any]
    trade_ledger: Optional[TradeLedger] = None


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
        Simulate historical strategy performance under V1 cost model with formal trade-ledger accounting.
        
        RESEARCH INTEGRITY & CASH ACCOUNTING (Prompt 5):
        - Fills and initial entry from zero inventory are charged explicitly.
        - Quarterly rolls occur once per cycle on documented proxy schedule.
        - Unencumbered cash earns short rate with documented day-count convention.
        - Trade ledger derives costs and turnover counting contract quantities only.
        - Cash ledger identity: final equity - initial capital == total collateral P&L.
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
            
        # One scheduled roll date per contract cycle under documented proxy schedule
        if roll_dates is None:
            roll_dates = compute_proxy_roll_dates(common_dates)
        roll_set = set(pd.to_datetime(roll_dates))
        
        trade_ledger = TradeLedger()
        n_days = len(common_dates)
        equity = np.zeros(n_days)
        gross_pnl = np.zeros(n_days)
        trade_cost = np.zeros(n_days)
        roll_cost = np.zeros(n_days)
        cash_interest = np.zeros(n_days)
        net_trading_pnl = np.zeros(n_days)
        net_pnl = np.zeros(n_days)
        margin_req = np.zeros(n_days)
        
        # -------------------------------------------------------------------
        # Day 0: Initial entry from zero inventory
        # -------------------------------------------------------------------
        t0 = common_dates[0]
        initial_entry_cost = 0.0
        for sym in active_syms:
            pos_0 = int(positions.loc[t0, f"n_{sym.lower()}"])
            q0 = abs(pos_0)
            if q0 > 0:
                spec = TREASURY_FUTURES_SPECS.get(sym)
                tick_val = spec.tick_value if spec else 15.625
                slip = self.cost_config.slippage_ticks * tick_val * q0
                fee = self.cost_config.fee_per_contract * q0
                tot = fee + slip
                initial_entry_cost += tot
                trade_ledger.add_trade(TradeRecord(
                    timestamp=t0,
                    symbol=sym,
                    direction="BUY" if pos_0 > 0 else "SELL",
                    quantity=q0,
                    price_points=0.0,
                    fee=fee,
                    slippage=slip,
                    trade_type="ENTRY",
                ))

        trade_cost[0] = initial_entry_cost
        equity[0] = self.initial_capital - initial_entry_cost
        net_trading_pnl[0] = -initial_entry_cost
        net_pnl[0] = -initial_entry_cost

        raw_margin_0 = sum(
            abs(positions.loc[t0, f"n_{sym.lower()}"]) * self.cost_config.initial_margin.get(sym, 2000.0)
            for sym in active_syms
        )
        margin_req[0] = raw_margin_0 * (1.0 - self.cost_config.spread_margin_credit)

        # -------------------------------------------------------------------
        # Days 1 .. n_days - 1: Sequential simulation
        # -------------------------------------------------------------------
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
            
            # 2. Transaction Costs & Bid/Ask Slippage on rebalances at t
            daily_trade_cost = 0.0
            total_contracts_held = 0
            for sym in active_syms:
                pos_prev = int(positions.loc[dt_prev, f"n_{sym.lower()}"])
                pos_curr = int(positions.loc[dt, f"n_{sym.lower()}"])
                delta_pos = abs(pos_curr - pos_prev)
                total_contracts_held += abs(pos_curr)
                
                if delta_pos > 0:
                    spec = TREASURY_FUTURES_SPECS.get(sym)
                    tick_val = spec.tick_value if spec else 15.625
                    slip = self.cost_config.slippage_ticks * tick_val * delta_pos
                    fee = self.cost_config.fee_per_contract * delta_pos
                    tot = fee + slip
                    daily_trade_cost += tot
                    trade_ledger.add_trade(TradeRecord(
                        timestamp=dt,
                        symbol=sym,
                        direction="BUY" if pos_curr > pos_prev else "SELL",
                        quantity=delta_pos,
                        price_points=0.0,
                        fee=fee,
                        slippage=slip,
                        trade_type="REBALANCE",
                    ))
            trade_cost[t] = daily_trade_cost
            
            # 3. Contract Roll Drag (Single roll event per cycle on documented proxy schedule)
            daily_roll_cost = 0.0
            if dt in roll_set and total_contracts_held > 0:
                for sym in active_syms:
                    n_held = abs(int(positions.loc[dt, f"n_{sym.lower()}"]))
                    if n_held > 0:
                        c_roll = n_held * self.cost_config.roll_friction_per_contract
                        daily_roll_cost += c_roll
                        trade_ledger.add_roll_event(
                            timestamp=dt,
                            symbol=sym,
                            contracts=n_held,
                            friction_per_contract=self.cost_config.roll_friction_per_contract,
                            total_cost=c_roll,
                        )
            roll_cost[t] = daily_roll_cost
            
            # 4. Margin Requirement & Cash Interest (NO repo carry!)
            raw_margin = sum(
                abs(positions.loc[dt, f"n_{sym.lower()}"]) * self.cost_config.initial_margin.get(sym, 2000.0)
                for sym in active_syms
            )
            margin = raw_margin * (1.0 - self.cost_config.spread_margin_credit)
            margin_req[t] = margin
            
            # Unencumbered cash earns risk-free rate
            unencumbered_cash = max(0.0, equity[t - 1] - margin_req[t - 1])
            days_elapsed = max(1, (dt - dt_prev).days)
            if self.cost_config.day_count_convention == "act_360":
                daily_interest = unencumbered_cash * r_cash.iloc[t - 1] * (days_elapsed / 360.0)
            else:
                daily_interest = unencumbered_cash * (r_cash.iloc[t - 1] / 252.0)
            cash_interest[t] = daily_interest
            
            # Net Trading P&L (strictly excluding cash interest)
            day_net_trading = daily_gross - daily_trade_cost - daily_roll_cost
            net_trading_pnl[t] = day_net_trading

            # Net P&L (Total Collateral P&L)
            day_net = day_net_trading + daily_interest
            net_pnl[t] = day_net
            equity[t] = equity[t - 1] + day_net

        # Final liquidation if configured
        if self.cost_config.liquidate_at_end:
            final_dt = common_dates[-1]
            liq_cost = 0.0
            for sym in active_syms:
                pos_end = abs(int(positions.loc[final_dt, f"n_{sym.lower()}"]))
                if pos_end > 0:
                    spec = TREASURY_FUTURES_SPECS.get(sym)
                    tick_val = spec.tick_value if spec else 15.625
                    slip = self.cost_config.slippage_ticks * tick_val * pos_end
                    fee = self.cost_config.fee_per_contract * pos_end
                    tot = fee + slip
                    liq_cost += tot
                    trade_ledger.add_trade(TradeRecord(
                        timestamp=final_dt,
                        symbol=sym,
                        direction="SELL" if positions.loc[final_dt, f"n_{sym.lower()}"] > 0 else "BUY",
                        quantity=pos_end,
                        price_points=0.0,
                        fee=fee,
                        slippage=slip,
                        trade_type="LIQUIDATION",
                    ))
            trade_cost[-1] += liq_cost
            net_trading_pnl[-1] -= liq_cost
            net_pnl[-1] -= liq_cost
            equity[-1] -= liq_cost
            
        equity_series = pd.Series(equity, index=common_dates)
        daily_returns = equity_series.pct_change().fillna(0.0)
        
        pnl_df = pd.DataFrame({
            "gross_pnl": gross_pnl,
            "trade_cost": trade_cost,
            "roll_cost": roll_cost,
            "net_trading_pnl": net_trading_pnl,
            "cash_interest": cash_interest,
            "net_pnl": net_pnl,
            "equity": equity,
            "margin_req": margin_req,
        }, index=common_dates)
        
        metrics = self._calculate_metrics(daily_returns, equity_series, pnl_df, trade_ledger)
        
        return BacktestResult(
            strategy_name=strategy_name,
            equity_series=equity_series,
            daily_returns=daily_returns,
            positions=positions,
            pnl_components=pnl_df,
            metrics=metrics,
            trade_ledger=trade_ledger,
        )

    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity: pd.Series,
        pnl_df: pd.DataFrame,
        trade_ledger: Optional[TradeLedger] = None,
    ) -> Dict[str, Any]:
        """Compute institutional risk-adjusted return scorecard with unbundled components."""
        n_days = len(returns)
        if n_days <= 1:
            return {}
            
        years = max(0.01, n_days / 252.0)
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1.0 else -1.0
        
        # PnL unbundled breakdowns
        total_gross = float(pnl_df["gross_pnl"].sum())
        total_trade_costs = float(pnl_df["trade_cost"].sum())
        total_roll_costs = float(pnl_df["roll_cost"].sum())
        total_net_trading = float(pnl_df["net_trading_pnl"].sum())
        total_interest = float(pnl_df["cash_interest"].sum())
        total_collateral_pnl = float(pnl_df["net_pnl"].sum())
        final_equity = float(equity.iloc[-1])

        # Cash ledger identity verification
        ledger_diff = abs((final_equity - self.initial_capital) - total_collateral_pnl)
        assert ledger_diff < 1e-4, f"Cash ledger identity broken: equity diff != net PnL (diff={ledger_diff})"

        # Primary strategy return series: daily net trading P&L / initial_capital
        daily_trading_ret = pnl_df["net_trading_pnl"] / self.initial_capital
        trading_vol = float(daily_trading_ret.std() * np.sqrt(252.0))
        mean_trading_ret = float(daily_trading_ret.mean())

        # Sharpe ratio on net trading return (undefined / NaN when risk is zero)
        if trading_vol < 1e-10:
            sharpe = np.nan
        else:
            sharpe = round(float((mean_trading_ret * np.sqrt(252.0)) / trading_vol), 3)
            
        # Sortino Ratio (coherent downside deviation on net trading return)
        downside_diff = np.minimum(0.0, daily_trading_ret.values)
        downside_var = float(np.mean(downside_diff ** 2))
        downside_std = float(np.sqrt(downside_var) * np.sqrt(252.0))
        if downside_std < 1e-10:
            sortino = np.nan
        else:
            sortino = round(float((mean_trading_ret * np.sqrt(252.0)) / downside_std), 3)
            
        # Drawdowns
        running_max = equity.cummax()
        drawdowns = (equity - running_max) / running_max
        max_drawdown = float(drawdowns.min())
        calmar = round(cagr / abs(max_drawdown), 3) if abs(max_drawdown) > 1e-4 else np.nan
        
        # Contract turnover derived strictly from trade ledger contract quantities
        total_contracts = trade_ledger.total_contract_turnover() if trade_ledger else 0
        pnl_turnover = round(total_net_trading / total_contracts, 2) if total_contracts > 0 else np.nan

        win_rate = float((returns > 0.0).sum() / max(1, (returns != 0.0).sum()))
        
        return {
            "total_return_pct": round(total_return * 100.0, 2),
            "cagr_pct": round(cagr * 100.0, 2),
            "annualized_vol_pct": round(trading_vol * 100.0, 2),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": round(max_drawdown * 100.0, 2),
            "calmar_ratio": calmar,
            "win_rate_pct": round(win_rate * 100.0, 2),
            "contract_turnover_lots": int(total_contracts),
            "pnl_turnover_usd_per_lot": pnl_turnover,
            "total_gross_pnl_usd": round(total_gross, 2),
            "total_trade_cost_usd": round(total_trade_costs, 2),
            "total_roll_cost_usd": round(total_roll_costs, 2),
            "total_net_trading_pnl_usd": round(total_net_trading, 2),
            "total_interest_earned_usd": round(total_interest, 2),
            "total_net_pnl_usd": round(total_collateral_pnl, 2),
            "total_collateral_pnl_usd": round(total_collateral_pnl, 2),
            "final_equity_usd": round(final_equity, 2),
            "cash_ledger_discrepancy_usd": round(ledger_diff, 4),
            "cost_drag_bp_annual": round((total_trade_costs + total_roll_costs) / (self.initial_capital * years) * 10_000.0, 2),
        }


# Backwards compatibility alias for SyntheticDV01Backtest
RelativeValueBacktestEngine = SyntheticDV01Backtest


