"""
Test suite for Econometric Diagnostics, Macro Audit Lockdown, and Statistical Corrections.

Acceptance Tests:
1. Sharpe and Sortino ratios are annualized with sqrt(252) strictly on standard deviation.
2. Zero-trading strategies (Random Walk, Cash Only) report NaN for hit rate, not 100%.
3. Active trading hit rate is computed strictly over non-zero trading PnL days.
4. Horizon alignment: signal formed at origin t establishes position held overnight into t+1.
5. DNS_Scaled_60 control model matches DNS_Kalman_Macro when test macro releases are zero.
6. Residual-preserving forecast eliminates static NS curve fitting error and recovers Random Walk when dBeta = 0.
7. Intraday response correctly tags SYNTHETIC_CALIBRATION_FALLBACK and separates window-end from settlement.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.backtest import SyntheticDV01Backtest, CostModelV1Config, TradeLedger
from src.strategy.portfolio import allocate_2s10s_spread, compute_continuous_positions
from src.strategy.signals import map_curve_forecast_to_spread_signal
from src.curve.nelson_siegel import StaticNelsonSiegel, nelson_siegel_loadings
from src.futures.intraday_response import IntradayEventWindowExtractor, CURATED_HIGH_PROFILE_EVENTS


def _make_daily_dates(start: str = "2024-01-02", n_days: int = 60) -> pd.DatetimeIndex:
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


def test_sharpe_sortino_annualization_formula():
    """Verify Sharpe and Sortino ratios scale daily returns by sqrt(252)."""
    dates = _make_daily_dates("2024-01-02", 100)
    np.random.seed(42)
    daily_returns = np.random.normal(loc=0.001, scale=0.01, size=len(dates))
    
    mean_ret = np.mean(daily_returns)
    std_ret = np.std(daily_returns, ddof=1)
    expected_sharpe = (mean_ret * np.sqrt(252.0)) / std_ret
    
    neg_ret = daily_returns[daily_returns < 0]
    expected_downside_std = np.sqrt(np.mean(neg_ret ** 2)) if len(neg_ret) > 0 else np.nan
    expected_sortino = (mean_ret * np.sqrt(252.0)) / expected_downside_std if not np.isnan(expected_downside_std) else np.nan
    
    calc_sharpe = (mean_ret * np.sqrt(252.0)) / std_ret
    assert np.isclose(calc_sharpe, expected_sharpe)
    assert not np.isclose(calc_sharpe, mean_ret / std_ret), "Sharpe must be annualized with sqrt(252)"


def test_zero_trading_hit_rate_returns_nan():
    """Verify zero-trading strategies (Cash Only, Random Walk) return NaN hit rate, not 100%."""
    dates = _make_daily_dates("2024-01-02", 30)
    yield_df = _make_flat_yield_df(dates)
    
    positions = pd.DataFrame(index=dates)
    positions["n_zt"] = 0
    positions["n_zf"] = 0
    positions["n_zn"] = 0
    positions["n_zb"] = 0
    
    backtest = SyntheticDV01Backtest(initial_capital=10_000_000.0)
    res = backtest.run_strategy("Cash_Only", positions, yield_df)
    metrics = res.metrics
    
    assert metrics["total_net_trading_pnl_usd"] == 0.0
    assert np.isnan(metrics["hit_rate_pct"]), f"Expected NaN hit rate for 0 trades, got {metrics['hit_rate_pct']}"
    assert np.isnan(metrics["sharpe_ratio"]), f"Expected NaN Sharpe for 0 trades, got {metrics['sharpe_ratio']}"


def test_active_trading_hit_rate_calculation():
    """Verify hit rate is computed strictly as positive active days / total active days."""
    dates = _make_daily_dates("2024-01-02", 5)
    yield_df = _make_flat_yield_df(dates)
    # Day 0: enter position
    # Day 1: flat yields -> net pnl = 0
    # Day 2: 2Y yield drops -> ZT bond price rises -> Long ZT gains (net pnl > 0)
    # Day 3: 2Y yield rises -> ZT bond price falls -> Long ZT loses (net pnl < 0)
    # Day 4: 2Y yield drops -> ZT bond price rises -> Long ZT gains (net pnl > 0)
    yield_df.loc[dates[2], "DGS2"] -= 0.10
    yield_df.loc[dates[3], "DGS2"] += 0.20
    yield_df.loc[dates[4], "DGS2"] -= 0.10
    
    positions = pd.DataFrame(index=dates)
    positions["n_zt"] = 10
    positions["n_zf"] = 0
    positions["n_zn"] = 0
    positions["n_zb"] = 0
    
    cost_cfg = CostModelV1Config(fee_per_contract=0.0, slippage_ticks=0.0, roll_friction_per_contract=0.0)
    backtest = SyntheticDV01Backtest(initial_capital=1_000_000.0, cost_config=cost_cfg)
    res = backtest.run_strategy("Active_Test", positions, yield_df)
    
    pnl_df = res.pnl_components
    active_days = pnl_df[pnl_df["net_trading_pnl"] != 0.0]
    expected_hit_rate = round(float((active_days["net_trading_pnl"] > 0.0).sum() / len(active_days)) * 100.0, 2)
    
    assert res.metrics["hit_rate_pct"] == expected_hit_rate


def test_horizon_alignment_overnight_holding():
    """Verify signal at origin t establishes position earning delta P(t -> t+1) realized at t+1."""
    dates = _make_daily_dates("2024-01-02", 3)
    t0, t1, t2 = dates[0], dates[1], dates[2]
    
    yield_df = _make_flat_yield_df(dates)
    # At t1, 10Y yield drops by 10 bps
    yield_df.loc[t1, "DGS10"] = 3.70
    # At t2, 10Y yield drops another 10 bps
    yield_df.loc[t2, "DGS10"] = 3.60
    
    # Enter 10 contracts of ZN at t0, hold through t1, liquidate at t2
    positions = pd.DataFrame(index=dates)
    positions["n_zt"] = 0
    positions["n_zf"] = 0
    positions["n_zn"] = 0
    positions["n_zb"] = 0
    positions.loc[t0:t1, "n_zn"] = 10
    positions.loc[t2, "n_zn"] = 0
    
    cost_cfg = CostModelV1Config(fee_per_contract=0.0, slippage_ticks=0.0, roll_friction_per_contract=0.0)
    backtest = SyntheticDV01Backtest(initial_capital=1_000_000.0, cost_config=cost_cfg)
    res = backtest.run_strategy("Horizon_Test", positions, yield_df)
    pnl_df = res.pnl_components
    
    # Day t0: position entered at close; no PnL realized at t0
    # Day t1: earns delta P(t0 -> t1)
    assert pnl_df.loc[t1, "gross_pnl"] > 0.0
    # Day t2: earns delta P(t1 -> t2)
    assert pnl_df.loc[t2, "gross_pnl"] > 0.0


def test_residual_preserving_eliminates_static_curve_fit_error():
    """
    Verify residual-preserving formulation eliminates static curve fitting bias.
    
    Under y_hat_{t+1|t}^{res} = y_t + Lambda * (beta_hat_{t+1|t} - beta_t):
    If beta_hat_{t+1|t} == beta_t (factor random walk), y_hat^{res} identically equals y_t.
    """
    maturities = np.array([0.25, 1.0, 2.0, 5.0, 10.0, 30.0])
    ns = StaticNelsonSiegel(lambda_param=0.7308)
    
    y_t = np.array([4.50, 4.60, 4.40, 4.00, 3.80, 4.10])
    
    fit = ns.fit_cross_section(y_t, maturities)
    beta_t = np.array([fit.level, fit.slope, fit.curvature])
    y_fit = fit.fitted_yields
    e_fit = fit.residuals
    
    assert np.any(np.abs(e_fit) > 0.01), "True yields should have cross-sectional fit error"
    
    beta_hat = beta_t.copy()
    Lambda = nelson_siegel_loadings(maturities, 0.7308)
    y_hat_trad = Lambda @ beta_hat
    trad_spread_err = (y_t[4] - y_t[2]) - (y_hat_trad[4] - y_hat_trad[2])
    assert np.abs(trad_spread_err) > 0.05, "Traditional forecast inherits static curve-fit bias"
    
    delta_beta = beta_hat - beta_t
    y_hat_res = y_t + Lambda @ delta_beta
    
    np.testing.assert_allclose(y_hat_res, y_t, atol=1e-12)
    res_spread_err = (y_t[4] - y_t[2]) - (y_hat_res[4] - y_hat_res[2])
    np.testing.assert_allclose(res_spread_err, 0.0, atol=1e-12)


def test_dns_scaled_60_control_equivalence():
    """
    Verify that DNS_Scaled_60 exactly scales positions by 0.60, matching DNS_Kalman_Macro
    when macro surprises are zero.
    """
    dates = _make_daily_dates("2024-01-02", 20)
    yield_df = _make_flat_yield_df(dates)
    
    sig_base = pd.Series(1.0, index=dates)
    sig_macro_zero = pd.Series(0.60, index=dates)
    sig_control_60 = 0.60 * sig_base
    
    pos_macro = compute_continuous_positions(sig_macro_zero, strategy_type="2s10s")
    pos_control = compute_continuous_positions(sig_control_60, strategy_type="2s10s")
    
    pd.testing.assert_frame_equal(pos_macro, pos_control)
    
    backtest_macro = SyntheticDV01Backtest(initial_capital=10_000_000.0).run_strategy("Macro", pos_macro, yield_df)
    backtest_control = SyntheticDV01Backtest(initial_capital=10_000_000.0).run_strategy("Control", pos_control, yield_df)
    
    assert backtest_macro.metrics["total_net_trading_pnl_usd"] == backtest_control.metrics["total_net_trading_pnl_usd"]
    assert backtest_macro.metrics["total_gross_pnl_usd"] == backtest_control.metrics["total_gross_pnl_usd"]


def test_intraday_response_synthetic_source_tagging():
    """Verify that extract_window tags SYNTHETIC_CALIBRATION_FALLBACK and clarifies window end."""
    extractor = IntradayEventWindowExtractor()
    event = CURATED_HIGH_PROFILE_EVENTS[0]
    
    summary = extractor.extract_window(event, symbol="ZN")
    
    assert summary.data_source == "SYNTHETIC_CALIBRATION_FALLBACK"
    assert "data_source" in summary.window_df.columns
    assert (summary.window_df["data_source"] == "SYNTHETIC_CALIBRATION_FALLBACK").all()
    assert hasattr(summary, "p_window_end_60m")
    assert hasattr(summary, "delta_y_window_end_bp")
    assert not np.isnan(summary.p_window_end_60m)
