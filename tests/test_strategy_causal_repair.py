"""
Unit tests for Prompt 5: Strategy Causal Repair, Trade-Ledger Accounting, and Execution Timing.

Acceptance Tests:
1. Independently calculated initial entry costs are charged at t=0.
2. Constant position incurs exactly one configured roll charge per cycle.
3. Zero trades earn zero trading P&L despite possible collateral interest.
4. Final equity minus initial capital equals the complete cash ledger including entry costs.
5. Hand-worked long/short and spread/fly scenarios verify allocations profit from intended yield moves.
6. No event-day decision captures price movement before permitted execution.
7. Sharpe, Sortino, turnover ratios handle zero risk/zero turnover without epsilon explosion (np.nan).
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.backtest import (
    SyntheticDV01Backtest,
    CostModelV1Config,
    TradeLedger,
    compute_proxy_roll_dates,
)
from src.strategy.portfolio import (
    allocate_2s10s_spread,
    allocate_2s5s10s_butterfly,
    compute_continuous_positions,
    DEFAULT_FUTURES_DV01,
)
from src.strategy.signals import map_curve_forecast_to_spread_signal
from src.futures.futures_analytics import TREASURY_FUTURES_SPECS


def _make_daily_dates(start: str = "2023-01-03", n_days: int = 260) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n_days)


def _make_flat_yield_df(dates: pd.DatetimeIndex, base_yields: dict = None) -> pd.DataFrame:
    if base_yields is None:
        base_yields = {
            "DGS3MO": 4.50,
            "DGS1": 4.60,
            "DGS2": 4.40,
            "DGS5": 4.00,
            "DGS10": 3.80,
            "DGS30": 4.10,
        }
    df = pd.DataFrame(index=dates)
    for col, y in base_yields.items():
        df[col] = y
    df["date"] = dates
    return df


def test_initial_entry_costs_charged_at_t0():
    """Acceptance test 1: Independently calculated initial entry costs are charged at t=0."""
    dates = _make_daily_dates("2023-01-03", 10)
    yield_df = _make_flat_yield_df(dates)

    # Strategy enters at t0: 10 ZT contracts, -5 ZN contracts
    positions_df = pd.DataFrame(index=dates)
    positions_df["n_zt"] = 10
    positions_df["n_zn"] = -5

    cost_cfg = CostModelV1Config(fee_per_contract=1.50, slippage_ticks=0.5)
    engine = SyntheticDV01Backtest(cost_config=cost_cfg, initial_capital=10_000_000.0)
    res = engine.run_strategy("Test_Entry", positions_df, yield_df)

    # Independent calculation:
    # ZT: tick_val = 7.8125. slip = 0.5 * 7.8125 * 10 = 39.0625. fee = 1.50 * 10 = 15.0. total = 54.0625
    # ZN: tick_val = 15.625. slip = 0.5 * 15.625 * 5 = 39.0625. fee = 1.50 * 5 = 7.50. total = 46.5625
    expected_entry_cost = 54.0625 + 46.5625  # 100.625

    assert res.trade_ledger is not None
    assert res.trade_ledger.total_trade_costs() == pytest.approx(expected_entry_cost)
    assert res.pnl_components.loc[dates[0], "trade_cost"] == pytest.approx(expected_entry_cost)
    assert res.trade_ledger.trades[0].trade_type == "ENTRY"
    assert res.trade_ledger.trades[1].trade_type == "ENTRY"
    assert res.trade_ledger.total_contract_turnover() == 15


def test_constant_position_single_roll_charge_per_cycle():
    """Acceptance test 2: Constant position incurs exactly one configured roll charge per cycle."""
    # Full year 2023 covering 4 IMM cycles (Feb, May, Aug, Nov)
    dates = pd.bdate_range("2023-01-03", "2023-12-29")
    yield_df = _make_flat_yield_df(dates)

    # Constant position of 8 contracts of ZN, 0 trades after t0
    positions_df = pd.DataFrame(index=dates)
    positions_df["n_zn"] = 8

    cost_cfg = CostModelV1Config(roll_friction_per_contract=4.00)
    engine = SyntheticDV01Backtest(cost_config=cost_cfg, initial_capital=10_000_000.0)
    res = engine.run_strategy("Test_Roll", positions_df, yield_df)

    # Assert exactly 4 roll events (one in Feb, May, Aug, Nov)
    assert len(res.trade_ledger.roll_events) == 4
    roll_months = [r["timestamp"].month for r in res.trade_ledger.roll_events]
    assert roll_months == [2, 5, 8, 11]

    for r in res.trade_ledger.roll_events:
        assert r["timestamp"].day >= 20
        assert r["contracts"] == 8
        assert r["total_cost"] == pytest.approx(32.00)  # 8 * 4.00

    # Total roll cost across year = 4 * 32.00 = 128.00
    assert res.trade_ledger.total_roll_costs() == pytest.approx(128.00)
    assert res.metrics["total_roll_cost_usd"] == pytest.approx(128.00)


def test_zero_trades_earn_zero_trading_pnl_despite_interest():
    """Acceptance test 3: Zero trades earn zero trading P&L despite possible collateral interest."""
    dates = _make_daily_dates("2023-01-03", 50)
    yield_df = _make_flat_yield_df(dates)

    # All positions 0.0
    positions_df = pd.DataFrame(index=dates)
    positions_df["n_zt"] = 0
    positions_df["n_zn"] = 0

    engine = SyntheticDV01Backtest(initial_capital=10_000_000.0)
    res = engine.run_strategy("Test_Zero", positions_df, yield_df)

    met = res.metrics
    assert met["total_gross_pnl_usd"] == 0.0
    assert met["total_trade_cost_usd"] == 0.0
    assert met["total_roll_cost_usd"] == 0.0
    assert met["total_net_trading_pnl_usd"] == 0.0
    assert met["contract_turnover_lots"] == 0
    assert met["total_interest_earned_usd"] > 0.0
    assert met["total_collateral_pnl_usd"] == pytest.approx(met["total_interest_earned_usd"])

    # Sharpe and Sortino must be np.nan when trading volatility is zero
    assert np.isnan(met["sharpe_ratio"])
    assert np.isnan(met["sortino_ratio"])
    assert np.isnan(met["pnl_turnover_usd_per_lot"])


def test_cash_ledger_identity_with_entry_and_rolls():
    """Acceptance test 4: Final equity - initial capital == total collateral P&L (< 1e-4)."""
    dates = pd.bdate_range("2023-01-03", "2023-08-31")
    yield_df = _make_flat_yield_df(dates)

    # Varying positions to incur entries, rebalances, and rolls
    positions_df = pd.DataFrame(index=dates)
    positions_df["n_zt"] = np.where(np.arange(len(dates)) % 2 == 0, 15, -10)
    positions_df["n_zn"] = np.where(np.arange(len(dates)) % 3 == 0, -8, 5)

    engine = SyntheticDV01Backtest(initial_capital=10_000_000.0)
    res = engine.run_strategy("Test_Ledger", positions_df, yield_df)

    final_equity = res.metrics["final_equity_usd"]
    collateral_pnl = res.metrics["total_collateral_pnl_usd"]
    diff = abs((final_equity - 10_000_000.0) - collateral_pnl)

    assert diff < 1e-4
    assert res.metrics["cash_ledger_discrepancy_usd"] < 1e-4


def test_hand_worked_spread_and_fly_profit_mechanisms():
    """
    Acceptance test 5: Hand-worked long/short and spread/fly scenarios verify
    that allocations profit from intended yield moves.
    """
    dates = pd.date_range("2023-01-03", periods=2, freq="B")

    # (a) Steepener (Signal > 0: Long ZT, Short ZN)
    # Target: 10Y yield rises +10 bp, 2Y yield falls -10 bp (Curve steepens by 20 bp)
    y_steepen = pd.DataFrame({
        "DGS2": [4.00, 3.90],   # -10 bp
        "DGS10": [4.00, 4.10],  # +10 bp
        "DGS3MO": [4.00, 4.00],
    }, index=dates)

    pos_steep = allocate_2s10s_spread(signal=1.0, target_dv01=10_000.0)
    assert pos_steep["n_zt"] > 0   # Long ZT
    assert pos_steep["n_zn"] < 0   # Short ZN

    pos_df_steep = pd.DataFrame([pos_steep, pos_steep], index=dates)
    engine = SyntheticDV01Backtest(cost_config=CostModelV1Config(fee_per_contract=0.0, slippage_ticks=0.0))
    res_steep = engine.run_strategy("Steepener", pos_df_steep, y_steepen)

    # Gross P&L on day 1 must be strictly positive
    day1_gross_steep = res_steep.pnl_components.loc[dates[1], "gross_pnl"]
    assert day1_gross_steep > 0.0

    # (b) Flattener (Signal < 0: Short ZT, Long ZN)
    # Target: 2Y yield rises +10 bp, 10Y yield falls -10 bp (Curve flattens by 20 bp)
    y_flatten = pd.DataFrame({
        "DGS2": [4.00, 4.10],   # +10 bp
        "DGS10": [4.00, 3.90],  # -10 bp
        "DGS3MO": [4.00, 4.00],
    }, index=dates)

    pos_flat = allocate_2s10s_spread(signal=-1.0, target_dv01=10_000.0)
    assert pos_flat["n_zt"] < 0   # Short ZT
    assert pos_flat["n_zn"] > 0   # Long ZN

    pos_df_flat = pd.DataFrame([pos_flat, pos_flat], index=dates)
    res_flat = engine.run_strategy("Flattener", pos_df_flat, y_flatten)
    day1_gross_flat = res_flat.pnl_components.loc[dates[1], "gross_pnl"]
    assert day1_gross_flat > 0.0

    # (c) Long Fly (Signal > 0: Short Belly ZF, Long Wings ZT & ZN)
    # Belly yield rises +15 bp, wings flat (Belly cheapens, Curvature increases)
    y_fly_cheap = pd.DataFrame({
        "DGS2": [4.00, 4.00],
        "DGS5": [4.00, 4.15],   # +15 bp
        "DGS10": [4.00, 4.00],
        "DGS3MO": [4.00, 4.00],
    }, index=dates)

    pos_long_fly = allocate_2s5s10s_butterfly(signal=1.0, target_dv01=10_000.0)
    assert pos_long_fly["n_zf"] < 0  # Short Belly ZF
    assert pos_long_fly["n_zt"] > 0  # Long Wing ZT
    assert pos_long_fly["n_zn"] > 0  # Long Wing ZN

    pos_df_long_fly = pd.DataFrame([pos_long_fly, pos_long_fly], index=dates)
    res_lfly = engine.run_strategy("LongFly", pos_df_long_fly, y_fly_cheap)
    day1_gross_lfly = res_lfly.pnl_components.loc[dates[1], "gross_pnl"]
    assert day1_gross_lfly > 0.0

    # (d) Short Fly (Signal < 0: Long Belly ZF, Short Wings ZT & ZN)
    # Belly yield falls -15 bp, wings flat (Belly richens, Curvature decreases)
    y_fly_rich = pd.DataFrame({
        "DGS2": [4.00, 4.00],
        "DGS5": [4.00, 3.85],   # -15 bp
        "DGS10": [4.00, 4.00],
        "DGS3MO": [4.00, 4.00],
    }, index=dates)

    pos_short_fly = allocate_2s5s10s_butterfly(signal=-1.0, target_dv01=10_000.0)
    assert pos_short_fly["n_zf"] > 0  # Long Belly ZF
    assert pos_short_fly["n_zt"] < 0  # Short Wing ZT
    assert pos_short_fly["n_zn"] < 0  # Short Wing ZN

    pos_df_short_fly = pd.DataFrame([pos_short_fly, pos_short_fly], index=dates)
    res_sfly = engine.run_strategy("ShortFly", pos_df_short_fly, y_fly_rich)
    day1_gross_sfly = res_sfly.pnl_components.loc[dates[1], "gross_pnl"]
    assert day1_gross_sfly > 0.0


def test_forecast_spread_signal_mapping_directions():
    """Verify map_curve_forecast_to_spread_signal produces economically correct signals."""
    tenors = ["DGS2", "DGS5", "DGS10"]
    y_orig = np.array([4.00, 4.00, 4.00])

    # 1. 2s10s Steepener forecast: y(10) rises to 4.20, y(2) falls to 3.90
    y_pred_steep = np.array([3.90, 4.00, 4.20])
    sig_steep = map_curve_forecast_to_spread_signal(y_pred_steep, y_orig, tenors, train_spread_std=0.10, strategy_type="2s10s")
    assert sig_steep > 0.0  # Steepener signal

    # 2. 2s10s Flattener forecast: y(10) falls to 3.90, y(2) rises to 4.20
    y_pred_flat = np.array([4.20, 4.00, 3.90])
    sig_flat = map_curve_forecast_to_spread_signal(y_pred_flat, y_orig, tenors, train_spread_std=0.10, strategy_type="2s10s")
    assert sig_flat < 0.0  # Flattener signal

    # 3. 2s5s10s Belly cheapening forecast: 5Y yield rises to 4.20 while 2Y and 10Y stay at 4.00
    y_pred_belly_cheap = np.array([4.00, 4.20, 4.00])
    sig_bcheap = map_curve_forecast_to_spread_signal(y_pred_belly_cheap, y_orig, tenors, train_spread_std=0.10, strategy_type="2s5s10s")
    assert sig_bcheap > 0.0  # Long fly (short belly)

    # 4. 2s5s10s Belly richening forecast: 5Y yield falls to 3.80 while 2Y and 10Y stay at 4.00
    y_pred_belly_rich = np.array([4.00, 3.80, 4.00])
    sig_brich = map_curve_forecast_to_spread_signal(y_pred_belly_rich, y_orig, tenors, train_spread_std=0.10, strategy_type="2s5s10s")
    assert sig_brich < 0.0  # Short fly (long belly)


def test_no_event_day_decision_captures_price_before_permitted_execution():
    """
    Acceptance test 6: No event-day decision captures price movement before permitted execution.
    An event occurring on date t produces a position executed to earn (t -> t+1) price change.
    It does not earn (t-1 -> t) yield change.
    """
    dates = pd.date_range("2023-01-03", periods=3, freq="B")
    t0, t1, t2 = dates[0], dates[1], dates[2]

    # Large price shock occurs between t0 and t1: 2Y falls 50 bp
    y_df = pd.DataFrame({
        "DGS2": [4.50, 4.00, 4.00],  # 50 bp drop on date t1
        "DGS10": [4.00, 4.00, 4.00],
        "DGS3MO": [4.00, 4.00, 4.00],
    }, index=dates)

    # Event shock on date t1 triggers a Steepener allocation for forward execution
    # Position on t0 is 0; position set at t1 close is Long ZT (e.g. 10 contracts)
    positions_df = pd.DataFrame({
        "n_zt": [0, 10, 10],
        "n_zn": [0, -5, -5],
    }, index=dates)

    engine = SyntheticDV01Backtest(cost_config=CostModelV1Config(fee_per_contract=0.0, slippage_ticks=0.0))
    res = engine.run_strategy("Forward_Timing", positions_df, y_df)

    # Date t1 gross P&L must be 0.0 because position held overnight into t1 was 0!
    assert res.pnl_components.loc[t1, "gross_pnl"] == 0.0
    # The 50 bp drop on t1 is NOT captured
    assert res.metrics["total_gross_pnl_usd"] == 0.0
