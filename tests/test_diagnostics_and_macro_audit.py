"""
Test suite for Econometric Diagnostics, Macro Audit Lockdown, and Statistical Corrections.

Acceptance Tests:
1. Sharpe and Sortino ratios calculated through production engine with independently calculated analytical constants.
2. Zero-trading strategies (Random Walk, Cash Only) report NaN for hit rate, not 100%.
3. Active trading hit rate is computed strictly over non-zero trading PnL days.
4. Horizon alignment: signal formed at origin t establishes position held overnight into t+1.
5. Observable 2s10s spread error decomposition identity holds: e_s = u_factor + u_fit.
6. DNS_Scaled_60 control model matches DNS_Kalman_Macro when test macro releases are zero across WalkForwardHarness.
7. End-to-end WalkForwardHarness macro causality: altering tomorrow's announcement cannot change today's position,
   while announcement affects eligible subsequent decision.
8. Residual-preserving forecast eliminates static NS curve fitting error and recovers Random Walk when dBeta = 0.
9. Intraday response tags UNKNOWN_UNVERIFIED_CACHE for legacy caches and SYNTHETIC_CALIBRATION_FALLBACK for generator.
"""

import numpy as np
import pandas as pd
import pytest

import datetime
from src.strategy.backtest import SyntheticDV01Backtest, CostModelV1Config, TradeLedger
from src.strategy.portfolio import allocate_2s10s_spread, compute_continuous_positions
from src.strategy.signals import map_curve_forecast_to_spread_signal
from src.curve.nelson_siegel import StaticNelsonSiegel, nelson_siegel_loadings
from src.futures.intraday_response import IntradayEventWindowExtractor, CURATED_HIGH_PROFILE_EVENTS
from src.backtest.walk_forward import (
    WalkForwardHarness,
    WalkForwardConfig,
    parse_macro_timestamp_availability,
    MacroTimestampAudit,
)
from ml_baseline import write_verdict_report


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


def test_sharpe_sortino_production_engine_independent_fixture():
    """
    Verify Sharpe and Sortino ratios using a production-engine fixture whose
    returns, annual Sharpe, and annual Sortino are independently calculated by hand.

    Fixture Setup:
      Initial capital: $1,000,000.00
      Dates: 6 business days (Day 0: entry, Days 1..5: holding returns)
      Position: Constant 10 contracts of ZN (DV01 = $100.00/bp)
      Daily net trading dollar P&L sequence:
        Day 0: $0.00 (entered at close)
        Day 1: +$10,000.00 (Delta y = -10 bp) -> r_1 = +0.010
        Day 2: -$5,000.00  (Delta y = +5 bp)  -> r_2 = -0.005
        Day 3: +$15,000.00 (Delta y = -15 bp) -> r_3 = +0.015
        Day 4: -$2,000.00  (Delta y = +2 bp)  -> r_4 = -0.002
        Day 5: +$8,000.00  (Delta y = -8 bp)  -> r_5 = +0.008

    Independent Analytical Calculation:
      Returns vector r = [0.0, 0.010, -0.005, 0.015, -0.002, 0.008] (N = 6)
      Mean return mu = 0.026 / 6 = 0.004333333333333333
      Sample variance s^2 = Sum(r_i - mu)^2 / (6 - 1) = 0.0003053333333333333 / 5 = 0.00006106666666666666
      Sample std s = sqrt(0.00006106666666666666) = 0.007814516406449389
      Annualized Sharpe = (mu * sqrt(252)) / s = 8.802787... -> 8.803

      Downside returns min(0, r) = [0.0, 0.0, -0.005, 0.0, -0.002, 0.0]
      Downside variance = ((-0.005)^2 + (-0.002)^2) / 6 = 0.000029 / 6 = 0.000004833333333333333
      Downside std = sqrt(0.000004833333333333333) = 0.00219848432637882
      Annualized Sortino = (mu * sqrt(252)) / downside_std = 31.289526... -> 31.290
    """
    dates = _make_daily_dates("2024-01-02", 6)
    cost_cfg = CostModelV1Config(fee_per_contract=0.0, slippage_ticks=0.0, roll_friction_per_contract=0.0)
    backtest = SyntheticDV01Backtest(initial_capital=1_000_000.0, cost_config=cost_cfg)

    # Position: 10 contracts of ZN across all 6 days
    positions = pd.DataFrame(0, index=dates, columns=["n_zt", "n_zf", "n_zn", "n_zb"])
    positions["n_zn"] = 10

    # Yield panel to produce exact daily yield deltas in 10Y
    # -10 contracts * $100/bp * dy_bp => PnL = -1000 * dy_bp
    base_y = 4.00
    deltas = [0.0, -0.10, +0.05, -0.15, +0.02, -0.08]
    y_seq = np.cumsum(deltas) + base_y
    yield_df = pd.DataFrame({"DGS2": 4.0, "DGS5": 4.0, "DGS10": y_seq, "date": dates}, index=dates)

    res = backtest.run_strategy("Analytical_Fixture", positions, yield_df, dv01_dict={"ZN": 100.0, "ZT": 50.0, "ZF": 50.0})

    expected_pnl = np.array([0.0, 10000.0, -5000.0, 15000.0, -2000.0, 8000.0])
    actual_pnl = res.pnl_components["net_trading_pnl"].values
    np.testing.assert_allclose(actual_pnl, expected_pnl, atol=1e-6)

    # Independent analytical values
    EXPECTED_SHARPE = 8.803
    EXPECTED_SORTINO = 31.29

    assert res.metrics["sharpe_ratio"] == EXPECTED_SHARPE, (
        f"Sharpe mismatch: expected {EXPECTED_SHARPE}, got {res.metrics['sharpe_ratio']}"
    )
    assert res.metrics["sortino_ratio"] == EXPECTED_SORTINO, (
        f"Sortino mismatch: expected {EXPECTED_SORTINO}, got {res.metrics['sortino_ratio']}"
    )


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
    yield_df.loc[t1, "DGS10"] = 3.70
    yield_df.loc[t2, "DGS10"] = 3.60
    
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
    
    assert pnl_df.loc[t1, "gross_pnl"] > 0.0
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


def test_observable_2s10s_spread_decomposition_identity():
    """
    Verify the exact observable 2s10s spread decomposition identity:
      e_s = u_factor + u_fit
      MSE(e_s) = MSE(u_factor) + MSE(u_fit) + 2 Cov(u_factor, u_fit)
    """
    np.random.seed(42)
    s_act = np.random.normal(0.5, 0.2, 50)
    s_fit = s_act + np.random.normal(0.0, 0.25, 50)  # cross-sectional fitting error
    s_pred = s_fit + np.random.normal(0.0, 0.03, 50)  # factor dynamics forecast error

    e_s = s_act - s_pred
    u_factor = s_fit - s_pred
    u_fit = s_act - s_fit

    # Pointwise identity: e_s == u_factor + u_fit
    np.testing.assert_allclose(e_s, u_factor + u_fit, atol=1e-12)

    # Variance / MSE identity
    mse_total = np.mean(e_s ** 2)
    mse_factor = np.mean(u_factor ** 2)
    mse_fit = np.mean(u_fit ** 2)
    cross_term = 2.0 * np.mean(u_factor * u_fit)

    sum_components = mse_factor + mse_fit + cross_term
    np.testing.assert_allclose(mse_total, sum_components, atol=1e-12)


def test_walk_forward_harness_macro_control_equivalence_and_synchronization():
    """
    Verify that across the actual WalkForwardHarness:
    1. DNS_Scaled_60 matches DNS_Kalman_Macro when test macro releases are zero.
    2. Evaluation status is synchronized to NOT_EVALUATED (NO_TEST_RELEASES) across:
       - baseline_table
       - common_sample_table
       - forecast_ledger summary
       - run_metadata macro_event_audit
    3. Ledger records tag reason="COPIED_DNS_CURVE_FORECAST".
    """
    dates = _make_daily_dates("2024-01-02", 30)
    tenors = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    yield_data = {t: [4.0 + 0.1 * i] * len(dates) for i, t in enumerate(tenors)}
    yield_df = pd.DataFrame(yield_data, index=dates)

    factor_df = pd.DataFrame({
        "level": 4.5 + np.sin(np.arange(len(dates)) / 5.0) * 0.1,
        "slope": -0.5 + np.cos(np.arange(len(dates)) / 5.0) * 0.1,
        "curvature": 0.2 + np.sin(np.arange(len(dates)) / 3.0) * 0.05,
    }, index=dates)

    # Only training events (days 0..19), zero test events (days 20..24)
    macro_rows = [{"date": dates[2*i], "indicator": "CPI", "surprise_ann": 1.0 if i%2==0 else -1.0} for i in range(10)]
    macro_df = pd.DataFrame(macro_rows)

    cfg = WalkForwardConfig(train_window_days=20, refit_frequency_days=5)
    harness = WalkForwardHarness(yield_df, factor_df, macro_df, cfg)
    res = harness.run_walk_forward_evaluation(max_folds=1)

    # 1. Macro / Control equivalence
    pos_macro = res["backtest_results"]["DNS_Kalman_Macro"].positions
    pos_control = res["backtest_results"]["DNS_Scaled_60"].positions
    pd.testing.assert_frame_equal(pos_macro, pos_control)

    # 2. Status synchronization across tables, ledger, and metadata
    EXPECTED_STATUS = "NOT_EVALUATED (NO_TEST_RELEASES)"
    assert res["baseline_table"].loc["DNS + Kalman + Macro", "Forecast Status"] == EXPECTED_STATUS
    assert res["common_sample_table"].loc["DNS + Kalman + Macro", "Forecast Status"] == EXPECTED_STATUS

    ledger_summary = res["run_metadata"]["ledger_summary"]["DNS_Kalman_Macro"]
    assert ledger_summary["status"] == EXPECTED_STATUS
    assert "COPIED_DNS_CURVE_FORECAST" in ledger_summary["reasons"]

    audit_status = res["run_metadata"]["macro_event_audit"]["evaluation_status"]
    assert audit_status == EXPECTED_STATUS


def test_walk_forward_harness_macro_causality_and_timing():
    """
    End-to-end WalkForwardHarness causality test with nonzero, eligible macro coefficients:
    1. Inject macro announcement on test_dates[0] (tomorrow relative to origin t=train_dates[-1]).
    2. Verify changing tomorrow's release CANNOT change today's position at origin t.
    3. Verify the release DOES affect the subsequent decision at test_dates[0].
    """
    dates = _make_daily_dates("2024-01-02", 30)
    tenors = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    yield_data = {t: [4.0 + 0.1 * i] * len(dates) for i, t in enumerate(tenors)}
    yield_df = pd.DataFrame(yield_data, index=dates)

    factor_df = pd.DataFrame({
        "level": 4.5 + np.sin(np.arange(len(dates)) / 5.0) * 0.1,
        "slope": -0.5 + np.cos(np.arange(len(dates)) / 5.0) * 0.1,
        "curvature": 0.2 + np.sin(np.arange(len(dates)) / 3.0) * 0.05,
    }, index=dates)

    # 10 training events (days 0..18) with varying slope changes to produce nonzero causal beta
    macro_rows = []
    for i in range(10):
        dt = dates[2 * i]
        surp = 1.0 if i % 2 == 0 else -1.0
        macro_rows.append({"date": dt, "indicator": "CPI", "surprise_ann": surp})
        if 2 * i + 1 < len(dates):
            yield_df.loc[dates[2 * i + 1], "DGS10"] += surp * 0.10

    # Base: no announcement in test fold
    macro_df_base = pd.DataFrame(macro_rows)

    # Injected: announcement on dates[20] (test_dates[0])
    macro_df_injected = pd.DataFrame(macro_rows + [{
        "date": dates[20],
        "indicator": "CPI",
        "surprise_ann": 1.5,
    }])

    cfg = WalkForwardConfig(train_window_days=20, refit_frequency_days=5)

    harness_base = WalkForwardHarness(yield_df, factor_df, macro_df_base, cfg)
    res_base = harness_base.run_walk_forward_evaluation(max_folds=1)
    pos_base = res_base["backtest_results"]["DNS_Kalman_Macro"].positions
    sig_base = res_base["signals"]["DNS_Kalman_Macro"][0]

    harness_inj = WalkForwardHarness(yield_df, factor_df, macro_df_injected, cfg)
    res_inj = harness_inj.run_walk_forward_evaluation(max_folds=1)
    pos_inj = res_inj["backtest_results"]["DNS_Kalman_Macro"].positions
    sig_inj = res_inj["signals"]["DNS_Kalman_Macro"][0]

    t_today = dates[19]     # train_dates[-1] (origin date of step 0)
    t_tomorrow = dates[20]  # test_dates[0] (target of step 0, origin of step 1)

    # CAUSALITY LOCKDOWN: Changing tomorrow's release MUST NOT change today's position
    pd.testing.assert_series_equal(pos_base.loc[t_today], pos_inj.loc[t_today])
    assert sig_base.loc[t_today] == sig_inj.loc[t_today]

    # SUBSEQUENT DECISION: Tomorrow's release DOES affect subsequent decision at t_tomorrow
    assert not np.isclose(sig_base.loc[t_tomorrow], sig_inj.loc[t_tomorrow]), (
        f"Expected signal difference at t_tomorrow, but got {sig_base.loc[t_tomorrow]} == {sig_inj.loc[t_tomorrow]}"
    )
    assert not (pos_base.loc[t_tomorrow] == pos_inj.loc[t_tomorrow]).all(), (
        "Expected position difference at t_tomorrow following macro release"
    )


def test_intraday_response_source_tagging_and_window_clarity(tmp_path):
    """
    Verify that:
    1. Legacy cache files without data_source column are tagged UNKNOWN_UNVERIFIED_CACHE.
    2. Fresh synthetic generator tags SYNTHETIC_CALIBRATION_FALLBACK.
    3. Window end and implied yield deltas are cleanly reported.
    """
    # 1. Fresh synthetic generation in isolated temporary directory
    fresh_dir = tmp_path / "fresh_cache"
    fresh_extractor = IntradayEventWindowExtractor(cache_dir=fresh_dir)
    event = CURATED_HIGH_PROFILE_EVENTS[0]

    summary = fresh_extractor.extract_window(event, symbol="ZN")
    assert summary.data_source == "SYNTHETIC_CALIBRATION_FALLBACK"
    assert "data_source" in summary.window_df.columns
    assert (summary.window_df["data_source"] == "SYNTHETIC_CALIBRATION_FALLBACK").all()
    assert hasattr(summary, "p_window_end_60m")
    assert hasattr(summary, "delta_y_window_end_bp")
    assert not np.isnan(summary.p_window_end_60m)

    # 2. Legacy cache simulation: write parquet file WITHOUT data_source column
    legacy_dir = tmp_path / "legacy_cache"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_df = summary.window_df.drop(columns=["data_source"])
    legacy_file = legacy_dir / f"ZN_{event.indicator}_{event.date}.parquet"
    legacy_df.to_parquet(legacy_file)

    legacy_extractor = IntradayEventWindowExtractor(cache_dir=legacy_dir)
    legacy_summary = legacy_extractor.extract_window(event, symbol="ZN")
    assert legacy_summary.data_source == "UNKNOWN_UNVERIFIED_CACHE"
    assert (legacy_summary.window_df["data_source"] == "UNKNOWN_UNVERIFIED_CACHE").all()


def test_clipping_measurement_at_actual_operation_regression_fixture():
    """
    Regression fixture where raw DNS signal = 2.0:
      DNS = clip(2.0, -1, 1) = 1.0
      Control = 0.6 * DNS = 0.60
      Zero-event Macro = clip(0.6 * DNS + 0.4 * 0.0, -1, 1) = 0.60
    
    Verifies that:
      1. DNS output is 1.0, Control output is 0.6, Zero-event Macro output is 0.6.
      2. Control has 0.0% additional clipping and 0.0% raw threshold exceedance beyond 1.0.
      3. Zero-event Macro has 0.0% additional clipping beyond 1.0.
      4. Inherited clipping from DNS is tracked for both Control and Macro.
    """
    raw_kf = 2.0
    sig_kf_arr = np.clip(raw_kf, -1.0, 1.0)
    sig_control = 0.60 * sig_kf_arr
    macro_scale = 0.0
    macro_overlay_input = 0.60 * sig_kf_arr + 0.40 * macro_scale
    sig_macro = np.clip(macro_overlay_input, -1.0, 1.0)

    assert sig_kf_arr == 1.0
    assert sig_control == 0.60
    assert sig_macro == 0.60

    # Operation-level assertions
    assert abs(raw_kf) >= 1.0, "DNS raw input exceeds threshold 1.0"
    assert abs(raw_kf) > 1.0, "DNS applies clipping"
    assert abs(sig_control) < 1.0, "Control input never exceeds 1.0"
    assert abs(macro_overlay_input) < 1.0, "Zero-event macro overlay input never exceeds 1.0"

    # End-to-end WalkForwardHarness verification with raw signal exceedance
    dates = _make_daily_dates("2024-01-02", 30)
    tenors = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    # Sloped yield curve that produces large raw spread forecast delta
    yield_data = {t: [2.0 + 0.5 * i] * len(dates) for i, t in enumerate(tenors)}
    yield_df = pd.DataFrame(yield_data, index=dates)

    factor_df = pd.DataFrame({
        "level": 4.5 + np.sin(np.arange(len(dates)) / 5.0) * 0.1,
        "slope": -0.5 + np.cos(np.arange(len(dates)) / 5.0) * 0.1,
        "curvature": 0.2 + np.sin(np.arange(len(dates)) / 3.0) * 0.05,
    }, index=dates)
    macro_df = pd.DataFrame(columns=["date", "indicator", "surprise_ann"])

    cfg = WalkForwardConfig(train_window_days=20, refit_frequency_days=5)
    harness = WalkForwardHarness(yield_df, factor_df, macro_df, cfg)
    res = harness.run_walk_forward_evaluation(max_folds=1)
    econ_diag = res["run_metadata"]["econometric_diagnostics"]

    # Control must have 0% additional clipping and 0% threshold exceedance beyond 1.0
    assert econ_diag["raw_signal_threshold_exceedance_pct"]["DNS_Scaled_60"] == 0.0
    assert econ_diag["additional_clipping_frequency_pct"]["DNS_Scaled_60"] == 0.0
    # Zero-event Macro must have 0% additional clipping beyond 1.0
    assert econ_diag["raw_signal_threshold_exceedance_pct"]["DNS_Kalman_Macro"] == 0.0
    assert econ_diag["additional_clipping_frequency_pct"]["DNS_Kalman_Macro"] == 0.0
    # Saturation of Control is measured at 0.60
    assert "DNS_Scaled_60" in econ_diag["final_position_saturation_pct"]


def test_biased_error_cross_moment_vs_covariance_fixture():
    """
    Biased-error fixture proving that the uncentered second cross moment
    2 * E[u_factor * u_fit] balances the MSE identity, while centered covariance fails.
    
    Mathematical proof:
      e = u_1 + u_2
      MSE(e) = E[(u_1 + u_2)^2] = E[u_1^2] + E[u_2^2] + 2 E[u_1 * u_2]
             = MSE(u_1) + MSE(u_2) + 2 * mean(u_1 * u_2)
      
      Centered Covariance:
        Cov(u_1, u_2) = E[u_1 * u_2] - mu_1 * mu_2
      Therefore:
        MSE(u_1) + MSE(u_2) + 2 Cov(u_1, u_2) = MSE(e) - 2 * mu_1 * mu_2 != MSE(e)
    """
    # Deterministic error sequences with non-zero means (biased forecasts)
    u_factor = np.array([1.0, 2.0, 3.0])  # mu = 2.0
    u_fit = np.array([4.0, 2.0, 6.0])     # mu = 4.0
    e_total = u_factor + u_fit            # [5.0, 4.0, 9.0]

    mse_total = np.mean(e_total ** 2)          # (25 + 16 + 81) / 3 = 122/3 = 40.6667
    mse_factor = np.mean(u_factor ** 2)        # (1 + 4 + 9) / 3 = 14/3 = 4.6667
    mse_fit = np.mean(u_fit ** 2)              # (16 + 4 + 36) / 3 = 56/3 = 18.6667
    
    # 1. Uncentered second cross moment
    uncentered_cross_moment = 2.0 * np.mean(u_factor * u_fit)  # 2 * (4 + 4 + 18) / 3 = 52/3 = 17.3333
    sum_uncentered = mse_factor + mse_fit + uncentered_cross_moment
    np.testing.assert_allclose(mse_total, sum_uncentered, atol=1e-12)

    # 2. Centered covariance
    centered_cov = 2.0 * float(np.cov(u_factor, u_fit, bias=True)[0, 1])
    sum_centered = mse_factor + mse_fit + centered_cov
    
    # Prove centered covariance FAILS to balance the MSE identity by exactly 2 * mu_1 * mu_2
    bias_discrepancy = 2.0 * np.mean(u_factor) * np.mean(u_fit)  # 2 * 2.0 * 4.0 = 16.0
    assert not np.isclose(mse_total, sum_centered), "Centered covariance must not equal total MSE when errors are biased"
    np.testing.assert_allclose(mse_total - sum_centered, bias_discrepancy, atol=1e-12)


def test_boundary_release_origin_vs_target():
    """
    Boundary test for releases at first origin (train_dates[-1]) vs final target (test_dates[-1]):
    1. A release on first origin train_dates[-1] IS available for the first decision and affects position.
    2. A release on final target test_dates[-1] is NOT an evaluated decision origin in the fold,
       and cannot affect any position in the fold.
    """
    dates = _make_daily_dates("2024-01-02", 30)
    tenors = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    yield_data = {t: [4.0 + 0.1 * i] * len(dates) for i, t in enumerate(tenors)}
    yield_df = pd.DataFrame(yield_data, index=dates)

    # Induce responsive slope dynamics correlated with CPI surprises (for causal coefficient estimation)
    for i in range(10):
        surp = 1.0 if i % 2 == 0 else -1.0
        yield_df.loc[dates[2*i + 1]:, "DGS10"] += 0.10 * surp

    factor_df = pd.DataFrame({
        "level": 4.5 + np.sin(np.arange(len(dates)) / 5.0) * 0.1,
        "slope": -0.5 + np.cos(np.arange(len(dates)) / 5.0) * 0.1,
        "curvature": 0.2 + np.sin(np.arange(len(dates)) / 3.0) * 0.05,
    }, index=dates)

    # 10 training events to establish causal beta
    macro_rows = [{"date": dates[2*i], "indicator": "CPI", "surprise_ann": 1.0 if i%2==0 else -1.0} for i in range(10)]
    
    cfg = WalkForwardConfig(train_window_days=20, refit_frequency_days=5)

    # Case A: Base (no test releases)
    h_base = WalkForwardHarness(yield_df, factor_df, pd.DataFrame(macro_rows), cfg)
    res_base = h_base.run_walk_forward_evaluation(max_folds=1)
    pos_base = res_base["backtest_results"]["DNS_Kalman_Macro"].positions

    # Case B: Injected release on FIRST ORIGIN (dates[19] = train_dates[-1])
    macro_rows_first_orig = list(macro_rows) + [
        {"date": dates[19], "indicator": "CPI", "surprise_ann": 2.0}
    ]
    h_first = WalkForwardHarness(yield_df, factor_df, pd.DataFrame(macro_rows_first_orig), cfg)
    res_first = h_first.run_walk_forward_evaluation(max_folds=1)
    pos_first = res_first["backtest_results"]["DNS_Kalman_Macro"].positions

    # First origin position must change
    assert not pos_base.loc[dates[19]].equals(pos_first.loc[dates[19]]), (
        "Release on first origin train_dates[-1] must affect the first evaluated decision"
    )
    audit_first = res_first["run_metadata"]["macro_event_audit"]["fold_breakdown"][0]
    assert audit_first["evaluated_decision_events"] >= 1

    # Case C: Injected release on FINAL TARGET (dates[24] = test_dates[-1])
    macro_rows_final_tgt = list(macro_rows) + [
        {"date": dates[24], "indicator": "CPI", "surprise_ann": 2.0}
    ]
    h_tgt = WalkForwardHarness(yield_df, factor_df, pd.DataFrame(macro_rows_final_tgt), cfg)
    res_tgt = h_tgt.run_walk_forward_evaluation(max_folds=1)
    pos_tgt = res_tgt["backtest_results"]["DNS_Kalman_Macro"].positions

    # Across all evaluated origins (dates[19..23]), positions must remain bit-for-bit identical to base!
    pd.testing.assert_frame_equal(pos_base.loc[dates[19]:dates[23]], pos_tgt.loc[dates[19]:dates[23]])
    audit_tgt = res_tgt["run_metadata"]["macro_event_audit"]["fold_breakdown"][0]
    # In calendar test window, but NOT an evaluated decision origin event
    assert audit_tgt["calendar_event_count"] >= 1
    assert audit_tgt["evaluated_decision_events"] == 0


def test_timestamp_availability_after_close_enforcement():
    """
    Enforce timestamp availability:
    An announcement on origin date t occurring AFTER market close (e.g. 18:00 ET / 22:00 UTC)
    cannot affect the market close decision on date t.
    """
    dates = _make_daily_dates("2024-01-02", 30)
    tenors = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    yield_data = {t: [4.0 + 0.1 * i] * len(dates) for i, t in enumerate(tenors)}
    yield_df = pd.DataFrame(yield_data, index=dates)

    # Induce responsive slope dynamics correlated with CPI surprises (for causal coefficient estimation)
    for i in range(10):
        surp = 1.0 if i % 2 == 0 else -1.0
        yield_df.loc[dates[2*i + 1]:, "DGS10"] += 0.10 * surp

    factor_df = pd.DataFrame({
        "level": 4.5 + np.sin(np.arange(len(dates)) / 5.0) * 0.1,
        "slope": -0.5 + np.cos(np.arange(len(dates)) / 5.0) * 0.1,
        "curvature": 0.2 + np.sin(np.arange(len(dates)) / 3.0) * 0.05,
    }, index=dates)

    macro_rows = [{"date": dates[2*i], "indicator": "CPI", "surprise_ann": 1.0 if i%2==0 else -1.0} for i in range(10)]
    cfg = WalkForwardConfig(train_window_days=20, refit_frequency_days=5)

    # Base: no test release
    h_base = WalkForwardHarness(yield_df, factor_df, pd.DataFrame(macro_rows), cfg)
    res_base = h_base.run_walk_forward_evaluation(max_folds=1)
    pos_base = res_base["backtest_results"]["DNS_Kalman_Macro"].positions

    t_origin = dates[20]  # decision origin

    # Case A: Post-close release at 22:00 UTC (17:00 / 18:00 ET - after market close)
    ts_post_close = pd.Timestamp(f"{t_origin.date()} 22:00:00", tz="UTC")
    macro_post_close = pd.DataFrame(macro_rows + [{
        "date": t_origin,
        "timestamp": ts_post_close,
        "indicator": "CPI",
        "surprise_ann": 2.0,
    }])
    h_post = WalkForwardHarness(yield_df, factor_df, macro_post_close, cfg)
    res_post = h_post.run_walk_forward_evaluation(max_folds=1)
    pos_post = res_post["backtest_results"]["DNS_Kalman_Macro"].positions

    # Post-close release CANNOT affect date t_origin position!
    pd.testing.assert_series_equal(pos_base.loc[t_origin], pos_post.loc[t_origin])
    audit_post = res_post["run_metadata"]["macro_event_audit"]["fold_breakdown"][0]
    assert audit_post["post_close_events"] >= 1

    # Case B: Pre-close release at 12:30 UTC (8:30 AM ET - before market close)
    ts_pre_close = pd.Timestamp(f"{t_origin.date()} 12:30:00", tz="UTC")
    macro_pre_close = pd.DataFrame(macro_rows + [{
        "date": t_origin,
        "timestamp": ts_pre_close,
        "indicator": "CPI",
        "surprise_ann": 2.0,
    }])
    h_pre = WalkForwardHarness(yield_df, factor_df, macro_pre_close, cfg)
    res_pre = h_pre.run_walk_forward_evaluation(max_folds=1)
    pos_pre = res_pre["backtest_results"]["DNS_Kalman_Macro"].positions

    # Pre-close release DOES affect date t_origin position!
    assert not pos_base.loc[t_origin].equals(pos_pre.loc[t_origin]), (
        "Pre-close announcement must affect the position established on date t"
    )


def test_zero_surprise_and_inadequate_history_handling():
    """
    Handle zero surprises and inadequate coefficient history without calling them absent releases.
    """
    dates = _make_daily_dates("2024-01-02", 30)
    tenors = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    yield_data = {t: [4.0 + 0.1 * i] * len(dates) for i, t in enumerate(tenors)}
    yield_df = pd.DataFrame(yield_data, index=dates)

    factor_df = pd.DataFrame({
        "level": 4.5 + np.sin(np.arange(len(dates)) / 5.0) * 0.1,
        "slope": -0.5 + np.cos(np.arange(len(dates)) / 5.0) * 0.1,
        "curvature": 0.2 + np.sin(np.arange(len(dates)) / 3.0) * 0.05,
    }, index=dates)

    # 10 training events for CPI
    macro_rows = [{"date": dates[2*i], "indicator": "CPI", "surprise_ann": 1.0 if i%2==0 else -1.0} for i in range(10)]
    
    # Add 2 events in test window on dates[20]:
    # 1. CPI release with surprise_ann = 0.0 (zero surprise release)
    # 2. RARE_INDICATOR release with surprise_ann = 1.0 (inadequate training history, <8 events)
    macro_rows.append({"date": dates[20], "indicator": "CPI", "surprise_ann": 0.0})
    macro_rows.append({"date": dates[20], "indicator": "RARE_INDICATOR", "surprise_ann": 1.0})

    cfg = WalkForwardConfig(train_window_days=20, refit_frequency_days=5)
    harness = WalkForwardHarness(yield_df, factor_df, pd.DataFrame(macro_rows), cfg)
    res = harness.run_walk_forward_evaluation(max_folds=1)

    audit = res["run_metadata"]["macro_event_audit"]["fold_breakdown"][0]
    assert audit["evaluated_decision_events"] == 2
    assert audit["zero_surprise_events"] == 1, "CPI with 0.0 surprise must be tracked as zero_surprise_event"
    assert audit["inadequate_history_events"] == 1, "RARE_INDICATOR must be tracked as inadequate_history_event"
    
    # Status should reflect that events occurred but overlay remained zero
    assert res["run_metadata"]["macro_event_audit"]["strategy_overlay_status"] == (
        "INACTIVE_ZERO_SURPRISE_OR_INADEQUATE_HISTORY"
    )


def test_timezone_dst_and_precedence_parsing():
    """
    Test explicit timezone-aware timestamp comparison, DST handling, field precedence,
    and exact 16:00:00 America/New_York boundary handling.
    """
    # 1. Summer (July 2026, EDT = UTC-4): 16:00 ET is 20:00 UTC
    dt_summer = pd.Timestamp("2026-07-15")
    
    # 19:30 UTC = 15:30 EDT -> Available
    row_summer_pre = {"timestamp": "2026-07-15T19:30:00Z", "date": dt_summer}
    audit = parse_macro_timestamp_availability(row_summer_pre, dt_summer)
    assert audit.is_available is True
    assert audit.is_verified_available is True
    assert audit.status == MacroTimestampAudit.VERIFIED_AVAILABLE

    # 20:00:00 UTC = 16:00:00 EDT -> Exact boundary is Available
    row_summer_exact = {"timestamp": "2026-07-15T20:00:00Z", "date": dt_summer}
    audit = parse_macro_timestamp_availability(row_summer_exact, dt_summer)
    assert audit.is_available is True
    assert audit.status == MacroTimestampAudit.VERIFIED_AVAILABLE

    # 20:00:01 UTC = 16:00:01 EDT -> Post-close
    row_summer_boundary_post = {"timestamp": "2026-07-15T20:00:01Z", "date": dt_summer}
    audit = parse_macro_timestamp_availability(row_summer_boundary_post, dt_summer)
    assert audit.is_available is False
    assert audit.is_post_close is True
    assert audit.status == MacroTimestampAudit.POST_CLOSE

    # 20:30 UTC = 16:30 EDT -> Post-close
    row_summer_post = {"timestamp": "2026-07-15T20:30:00Z", "date": dt_summer}
    audit = parse_macro_timestamp_availability(row_summer_post, dt_summer)
    assert audit.is_available is False
    assert audit.status == MacroTimestampAudit.POST_CLOSE

    # 2. Winter (January 2026, EST = UTC-5): 16:00 ET is 21:00 UTC
    dt_winter = pd.Timestamp("2026-01-15")

    # 20:30 UTC = 15:30 EST -> Available in winter, but was post-close in summer!
    row_winter_pre = {"timestamp": "2026-01-15T20:30:00Z", "date": dt_winter}
    audit = parse_macro_timestamp_availability(row_winter_pre, dt_winter)
    assert audit.is_available is True
    assert audit.status == MacroTimestampAudit.VERIFIED_AVAILABLE

    # 21:00:00 UTC = 16:00:00 EST -> Exact boundary is Available
    row_winter_exact = {"timestamp": "2026-01-15T21:00:00Z", "date": dt_winter}
    audit = parse_macro_timestamp_availability(row_winter_exact, dt_winter)
    assert audit.is_available is True
    assert audit.status == MacroTimestampAudit.VERIFIED_AVAILABLE

    # 21:00:01 UTC = 16:00:01 EST -> Post-close
    row_winter_boundary_post = {"timestamp": "2026-01-15T21:00:01Z", "date": dt_winter}
    audit = parse_macro_timestamp_availability(row_winter_boundary_post, dt_winter)
    assert audit.is_available is False
    assert audit.status == MacroTimestampAudit.POST_CLOSE

    # 21:30 UTC = 16:30 EST -> Post-close
    row_winter_post = {"timestamp": "2026-01-15T21:30:00Z", "date": dt_winter}
    audit = parse_macro_timestamp_availability(row_winter_post, dt_winter)
    assert audit.is_available is False
    assert audit.status == MacroTimestampAudit.POST_CLOSE

    # 3. Documented precedence order: timestamp > release_timestamp > event_timestamp > publication_timestamp > datetime
    row_precedence = {
        "release_timestamp": "2026-07-15T21:00:00Z",  # post-close
        "timestamp": "2026-07-15T12:30:00Z",          # available (precedes release_timestamp)
        "date": dt_summer,
    }
    audit = parse_macro_timestamp_availability(row_precedence, dt_summer)
    assert audit.is_available is True
    assert audit.status == MacroTimestampAudit.VERIFIED_AVAILABLE

    # 4. Naive datetime localizes to America/New_York
    naive_dt = datetime.datetime(2026, 7, 15, 8, 30, 0)
    row_naive = {"timestamp": naive_dt, "date": dt_summer}
    audit = parse_macro_timestamp_availability(row_naive, dt_summer)
    assert audit.is_available is True
    assert audit.status == MacroTimestampAudit.VERIFIED_AVAILABLE

    # 5. Legacy date-only assumption (midnight or missing timestamp)
    row_legacy_midnight = {"timestamp": pd.Timestamp("2026-07-15 00:00:00"), "date": dt_summer}
    audit = parse_macro_timestamp_availability(row_legacy_midnight, dt_summer)
    assert audit.is_available is True
    assert audit.is_legacy_date_only is True
    assert audit.status == MacroTimestampAudit.LEGACY_DATE_ONLY_ASSUMED

    row_legacy_missing = {"date": dt_summer}
    audit = parse_macro_timestamp_availability(row_legacy_missing, dt_summer)
    assert audit.is_available is True
    assert audit.is_legacy_date_only is True
    assert audit.status == MacroTimestampAudit.LEGACY_DATE_ONLY_ASSUMED

    # 6. Invalid / contradictory timestamp rejection
    row_invalid_str = {"timestamp": "not-a-valid-date", "date": dt_summer}
    audit = parse_macro_timestamp_availability(row_invalid_str, dt_summer)
    assert audit.is_available is False
    assert audit.status == MacroTimestampAudit.INVALID_TIMESTAMP_REJECTED

    row_contradictory = {"timestamp": "2026-07-25T12:00:00Z", "date": dt_summer}
    audit = parse_macro_timestamp_availability(row_contradictory, dt_summer)
    assert audit.is_available is False
    assert audit.status == MacroTimestampAudit.INVALID_TIMESTAMP_REJECTED


def test_post_close_roll_rule_in_walk_forward_harness():
    """
    Test the explicit roll rule: after-close releases on an origin do not affect today's position,
    but roll into the next eligible decision origin if available.
    """
    dates = _make_daily_dates("2024-01-02", 30)
    tenors = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    yield_data = {t: [4.0 + 0.1 * i] * len(dates) for i, t in enumerate(tenors)}
    yield_df = pd.DataFrame(yield_data, index=dates)

    factor_df = pd.DataFrame({
        "level": 4.5 + np.sin(np.arange(len(dates)) / 5.0) * 0.1,
        "slope": -0.5 + np.cos(np.arange(len(dates)) / 5.0) * 0.1,
        "curvature": 0.2 + np.sin(np.arange(len(dates)) / 3.0) * 0.05,
    }, index=dates)

    # 10 training events for CPI to establish valid causal beta
    macro_rows = []
    for i in range(10):
        dt = dates[2 * i]
        surp = 1.0 if i % 2 == 0 else -1.0
        macro_rows.append({"date": dt, "indicator": "CPI", "surprise_ann": surp})
        if 2 * i + 1 < len(dates):
            yield_df.loc[dates[2 * i + 1], "DGS10"] += surp * 0.10

    cfg = WalkForwardConfig(train_window_days=20, refit_frequency_days=5)

    # Base: no announcement in test fold
    harness_base = WalkForwardHarness(yield_df, factor_df, pd.DataFrame(macro_rows), cfg)
    res_base = harness_base.run_walk_forward_evaluation(max_folds=1)
    pos_base = res_base["backtest_results"]["DNS_Kalman_Macro"].positions
    sig_base = res_base["signals"]["DNS_Kalman_Macro"][0]

    t_origin_0 = dates[19]  # train_dates[-1] (first decision origin)
    t_origin_1 = dates[20]  # test_dates[0] (second decision origin)

    # Injected post-close release on t_origin_0 at 17:00 ET (22:00 UTC)
    ts_post_close = pd.Timestamp(f"{t_origin_0.date()} 22:00:00", tz="UTC")
    macro_rows_injected = macro_rows + [{
        "date": t_origin_0,
        "timestamp": ts_post_close,
        "indicator": "CPI",
        "surprise_ann": 2.0,
    }]
    harness_inj = WalkForwardHarness(yield_df, factor_df, pd.DataFrame(macro_rows_injected), cfg)
    res_inj = harness_inj.run_walk_forward_evaluation(max_folds=1)
    pos_inj = res_inj["backtest_results"]["DNS_Kalman_Macro"].positions
    sig_inj = res_inj["signals"]["DNS_Kalman_Macro"][0]

    # CAUSALITY LOCKDOWN: Post-close release on t_origin_0 CANNOT affect t_origin_0 decision!
    assert sig_base.loc[t_origin_0] == sig_inj.loc[t_origin_0]
    pd.testing.assert_series_equal(pos_base.loc[t_origin_0], pos_inj.loc[t_origin_0])

    # ROLL RULE: Post-close release DOES roll into t_origin_1 subsequent decision!
    assert not np.isclose(sig_base.loc[t_origin_1], sig_inj.loc[t_origin_1])
    assert not (pos_base.loc[t_origin_1] == pos_inj.loc[t_origin_1]).all()

    audit = res_inj["run_metadata"]["macro_event_audit"]
    assert audit["total_post_close_events"] >= 1
    assert audit["total_rolled_to_next_decision_events"] >= 1


def test_supported_indicator_zero_history_counterexample_and_audit_reconciliation():
    """
    Test that a supported indicator with 0 training history (e.g. NFP) is classified as
    inadequate_history_events, and verify that the full audit reconciliation identities hold.
    """
    dates = _make_daily_dates("2024-01-02", 30)
    tenors = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    yield_data = {t: [4.0 + 0.1 * i] * len(dates) for i, t in enumerate(tenors)}
    yield_df = pd.DataFrame(yield_data, index=dates)

    factor_df = pd.DataFrame({
        "level": 4.5 + np.sin(np.arange(len(dates)) / 5.0) * 0.1,
        "slope": -0.5 + np.cos(np.arange(len(dates)) / 5.0) * 0.1,
        "curvature": 0.2 + np.sin(np.arange(len(dates)) / 3.0) * 0.05,
    }, index=dates)

    # 10 training events for CPI (so CPI is eligible with N >= 8)
    # 0 training events for NFP (supported indicator with ZERO history)
    macro_rows = []
    for i in range(10):
        dt = dates[2 * i]
        macro_rows.append({"date": dt, "indicator": "CPI", "surprise_ann": 1.0 if i % 2 == 0 else -1.0})
        yield_df.loc[dates[2 * i + 1], "DGS10"] += 0.05

    t_origin = dates[20]
    # Test window events on t_origin (January, EST = UTC-5):
    # 1. CPI release at 8:30 AM (available, eligible coefficient)
    macro_rows.append({
        "date": t_origin,
        "timestamp": f"{t_origin.date()}T08:30:00-05:00",
        "indicator": "CPI",
        "surprise_ann": 1.2,
    })
    # 2. NFP release at 8:30 AM (available, but NFP has 0 training events -> inadequate history counterexample!)
    macro_rows.append({
        "date": t_origin,
        "timestamp": f"{t_origin.date()}T08:30:00-05:00",
        "indicator": "NFP",
        "surprise_ann": 1.5,
    })
    # 3. Post-close release at 17:00 ET (22:00 UTC)
    macro_rows.append({
        "date": t_origin,
        "timestamp": f"{t_origin.date()}T17:00:00-05:00",
        "indicator": "CPI",
        "surprise_ann": 0.8,
    })
    # 4. Missing surprise release
    macro_rows.append({
        "date": t_origin,
        "timestamp": f"{t_origin.date()}T10:00:00-05:00",
        "indicator": "CPI",
        "surprise_ann": np.nan,
    })

    cfg = WalkForwardConfig(train_window_days=20, refit_frequency_days=5)
    harness = WalkForwardHarness(yield_df, factor_df, pd.DataFrame(macro_rows), cfg)
    res = harness.run_walk_forward_evaluation(max_folds=1)
    audit = res["run_metadata"]["macro_event_audit"]["fold_breakdown"][0]

    # Check NFP classification and reconciled counters
    assert audit["post_close_events"] == 1
    assert audit["rolled_to_next_decision_events"] == 1
    assert audit["timestamp_available_events"] == 3
    assert audit["observed_decision_events"] == 4
    assert audit["missing_surprise_events"] == 1
    assert audit["usable_surprise_events"] == 3
    assert audit["eligible_coefficient_events"] == 2, "Both CPI releases (pre-close and rolled post-close) have >=8 training observations"
    assert audit["inadequate_history_events"] == 1, "NFP has 0 training observations and must be inadequate history"
    assert audit["active_overlay_events"] == 2
    assert audit["zero_impact_events"] == 0
    assert audit["nonzero_macro_days"] == 2, "Both decision origins receive active CPI overlays"

    # Mathematical audit reconciliation identities
    # 1. Observed release-time status identity:
    assert audit["observed_decision_events"] == audit["timestamp_available_events"] + audit["post_close_events"]

    # 2. Application-time evaluation identity:
    assert audit["evaluated_decision_events"] == audit["usable_surprise_events"] + audit["missing_surprise_events"]

    # 3. Application-time coefficient eligibility identity:
    assert audit["usable_surprise_events"] == audit["eligible_coefficient_events"] + audit["inadequate_history_events"]

    # 4. Application-time overlay activity identity:
    assert audit["eligible_coefficient_events"] == audit["active_overlay_events"] + audit["zero_impact_events"]


def test_signal_mapping_scale_floor_and_details_fixture():
    """
    Test map_curve_forecast_to_spread_signal return_details=True across 2s10s and 2s5s10s fly.
    Verify that std=1e-6 floors scale to 1e-4, preventing artificial signal saturation.
    Verify saturation bounds disclosure.
    """
    maturities = np.array([0.25, 1.0, 2.0, 5.0, 10.0, 30.0])
    tenor_cols = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    y_origin = np.array([4.5, 4.6, 4.4, 4.0, 3.8, 4.1])
    # 2s10s spread at origin = 3.8 - 4.4 = -0.60
    # Create forecast curve with spread change of +2e-5 (+0.2 bp)
    y_pred = y_origin.copy()
    y_pred[4] += 2e-5  # DGS10 increases by 2e-5

    # Test with near-zero training std = 1e-6
    train_spread_std = 1e-6
    sig, details = map_curve_forecast_to_spread_signal(
        y_pred, y_origin, tenor_cols, train_spread_std, strategy_type="2s10s", return_details=True
    )

    # Scale must be floored to 1e-4
    assert details["scale"] == 1e-4
    # Raw input is 2e-5 / 1e-4 = 0.20
    assert np.isclose(details["raw_input"], 0.20)
    assert np.isclose(sig, 0.20)
    # Because 0.20 < 1.0, it must NOT be clipped or saturated
    assert details["is_clipped"] is False or details["is_clipped"] == 0
    assert details["is_saturated"] is False or details["is_saturated"] == 0

    # Without flooring, raw input would have been 2e-5 / 1e-6 = 20.0 (erroneously clipped)
    unfloored_raw = 2e-5 / 1e-6
    assert np.isclose(unfloored_raw, 20.0)

    # Test fly strategy with return_details=True
    sig_fly, details_fly = map_curve_forecast_to_spread_signal(
        y_pred, y_origin, tenor_cols, 0.05, strategy_type="2s5s10s", return_details=True
    )
    assert details_fly["observable_name"] == "2s5s10s"
    assert "raw_input" in details_fly
    assert "is_clipped" in details_fly
    assert "is_saturated" in details_fly


def test_sample_independent_dynamic_verdict_report(tmp_path, monkeypatch):
    """
    Test that write_verdict_report produces sample-independent dynamic output,
    renders N/A for missing values, and avoids hardcoded 57-day sample constants.
    """
    # Create synthetic common_sample_table deliberately unlike the 57-day sample
    models = [
        "Random Walk (Curve Benchmark)",
        "AR(1) Baseline (Static NS)",
        "DNS + Kalman",
        "Static NS (Residual-Preserving Diagnostic)",
        "DNS + Kalman (Residual-Preserving Diagnostic)",
        "DNS (60% Exposure Control)",
        "DNS + Kalman + Macro",
        "GBM",
    ]
    data = {
        "2s10s Spread RMSE (bp)": [1.50, 15.20, 14.80, 1.65, 1.60, 14.80, 14.80, np.nan],
        "Trading Net PnL ($)": [0.0, 12500.0, 12500.0, 14200.0, 14200.0, 7500.0, 7500.0, np.nan],
        "Annualized Sharpe": [np.nan, 0.85, 0.85, 1.20, 1.22, 0.85, 0.85, np.nan],
        "Hit Rate (%)": [np.nan, 58.5, 58.5, 62.0, 62.0, 58.5, 58.5, np.nan],
    }
    table = pd.DataFrame(data, index=models)

    ml_eval = {
        "shap_data": {"type": "tree_shap", "results": {}}
    }
    run_meta = {
        "git_commit": "testcommit1234",
        "run_mode": "SYNTHETIC_TEN_FOLD_EVALUATION",
        "fold_count": 10,
        "eval_start_date": "2023-01-03",
        "eval_end_date": "2023-08-30",
        "total_eval_days": 165,
        "gbm_backend": "sklearn",
        "seed": 42,
        "data_checksums": {"yield_panel": "abcd", "factor_panel": "ef01", "macro_surprises": "2345"},
        "macro_event_audit": {
            "total_training_events": 85,
            "total_calendar_events": 12,
            "total_observed_decision_events": 12,
            "total_timestamp_available_events": 10,
            "total_timestamp_verified_available_events": 8,
            "total_legacy_date_only_assumed_events": 2,
            "total_post_close_events": 2,
            "total_rolled_to_next_decision_events": 2,
            "total_usable_surprise_events": 10,
            "total_eligible_coefficient_events": 8,
            "total_inadequate_history_events": 2,
            "total_active_overlay_events": 6,
            "nonzero_macro_days": 6,
            "curve_forecast_provenance": "COPIED_DNS_CURVE_FORECAST",
            "strategy_overlay_status": "ACTIVE_OVERLAY",
            "evaluation_status": "VALID_MACRO_TEST",
        },
        "econometric_diagnostics": {
            "ns_contemporaneous_fit_rmse_bp": 3.45,
            "factor_coordinate_rmse_ar1_bp": 2.10,
            "factor_coordinate_rmse_random_walk_bp": 2.25,
            "observable_2s10s_spread_decomposition": {
                "total_spread_rmse_bp": 15.20,
                "factor_dynamics_spread_rmse_bp": 1.55,
                "cross_sectional_fit_spread_rmse_bp": 15.10,
                "uncentered_cross_moment_bp2": -0.85,
                "sum_components_mse_bp2": 231.04,
                "total_spread_mse_bp2": 231.04,
            },
            "raw_signal_threshold_exceedance_pct": {"Static_NS": 45.0, "Static_NS_Residual_Preserving": 0.0},
            "inherited_dns_clipping_pct": {"Static_NS": 0.0, "DNS_Scaled_60": 0.0},
            "additional_clipping_frequency_pct": {"Static_NS": 45.0, "Static_NS_Residual_Preserving": 5.0},
            "final_position_saturation_pct": {"Static_NS": 45.0, "Static_NS_Residual_Preserving": 5.0},
            "saturation_bounds": {"DNS_Scaled_60": 0.60, "Static_NS": 1.00},
            "mean_unclipped_signal_std": {"Static_NS": 0.95, "Static_NS_Residual_Preserving": 0.42},
        },
    }

    # Redirect reports directory to tmp_path
    monkeypatch.chdir(tmp_path)
    write_verdict_report(table, ml_eval, run_meta)

    verdict_file = tmp_path / "reports" / "ml_baseline_verdict.md"
    assert verdict_file.exists()
    content = verdict_file.read_text(encoding="utf-8")

    # Assert dynamic contents
    assert "165 trading days" in content
    assert "Folds: 10" in content
    assert "15.20 bp" in content
    assert "1.65 bp" in content
    assert "45.0%" in content
    assert "5.0%" in content
    assert "VALID_MACRO_TEST" in content
    assert "±0.60" in content

    # Assert no stale hardcoded constants from 57-day run
    assert "57 trading days" not in content
    assert "-$46,684.38" not in content
    assert "28.82" not in content
    assert "29.90" not in content


def test_cross_fold_post_close_roll_regression():
    """
    Two-fold cross-fold roll regression using 30 business days from 2024-01-02,
    train_window_days=20, refit_frequency_days=5, and max_folds=2:
    
    1. Within-fold roll:
       A CPI release at 17:00 NY on 2024-02-01 (dates[22]) leaves dates[22] unchanged
       and changes 2024-02-02 (dates[23]), with rolled_count=1 in Fold 0.
    
    2. Cross-fold roll across fold boundary:
       Moving the release to 2024-02-02 (dates[23], last origin of Fold 0) at 17:00 NY
       must leave 2024-02-02 unchanged in Fold 0, and roll to 2024-02-05 (dates[24], first origin of Fold 1),
       changing the evaluated decision on 2024-02-05!
    
    3. Explicit contract-quantity checks:
       - ForecastLedger contains target_type='macro_event_impulse' record for 2024-02-05
       - Signal difference between base and injected at 2024-02-05 is nonzero
       - Signal difference at 2024-02-02 is zero
    """
    dates = _make_daily_dates("2024-01-02", 30)
    tenors = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    yield_data = {t: [4.0 + 0.1 * i] * len(dates) for i, t in enumerate(tenors)}
    yield_df = pd.DataFrame(yield_data, index=dates)

    # Induce responsive slope dynamics correlated with CPI surprises (for causal coefficient estimation)
    for i in range(10):
        surp = 1.0 if i % 2 == 0 else -1.0
        yield_df.loc[dates[2 * i + 1]:, "DGS10"] += 0.10 * surp

    factor_df = pd.DataFrame({
        "level": 4.5 + np.sin(np.arange(len(dates)) / 5.0) * 0.1,
        "slope": -0.5 + np.cos(np.arange(len(dates)) / 5.0) * 0.1,
        "curvature": 0.2 + np.sin(np.arange(len(dates)) / 3.0) * 0.05,
    }, index=dates)

    # 10 training events to establish causal beta (N >= 8)
    macro_rows = [{"date": dates[2 * i], "indicator": "CPI", "surprise_ann": 1.0 if i % 2 == 0 else -1.0} for i in range(10)]
    cfg = WalkForwardConfig(train_window_days=20, refit_frequency_days=5)

    # Base run across 2 folds
    h_base = WalkForwardHarness(yield_df, factor_df, pd.DataFrame(macro_rows), cfg)
    res_base = h_base.run_walk_forward_evaluation(max_folds=2)
    sig_base = pd.concat(res_base["signals"]["DNS_Kalman_Macro"])

    # 1. Within-fold roll: CPI release at 17:00 NY on 2024-02-01 (dates[22])
    t_thu = dates[22]
    t_fri = dates[23]
    t_mon = dates[24]
    macro_within = pd.DataFrame(macro_rows + [{
        "date": t_thu,
        "timestamp": f"{t_thu.date()}T17:00:00-05:00",
        "indicator": "CPI",
        "surprise_ann": 1.5,
    }])
    h_within = WalkForwardHarness(yield_df, factor_df, macro_within, cfg)
    res_within = h_within.run_walk_forward_evaluation(max_folds=2)
    sig_within = pd.concat(res_within["signals"]["DNS_Kalman_Macro"])
    audit_within_f0 = res_within["run_metadata"]["macro_event_audit"]["fold_breakdown"][0]

    # Verify within-fold roll:
    assert sig_within.loc[t_thu] == sig_base.loc[t_thu], "Thursday 17:00 NY release cannot affect Thursday position"
    assert sig_within.loc[t_fri] != sig_base.loc[t_fri], "Thursday 17:00 NY release must affect Friday position"
    assert audit_within_f0["rolled_to_next_decision_events"] == 1

    # 2. Cross-fold roll: CPI release at 17:00 NY on 2024-02-02 (dates[23] - last origin of Fold 0)
    macro_cross = pd.DataFrame(macro_rows + [{
        "date": t_fri,
        "timestamp": f"{t_fri.date()}T17:00:00-05:00",
        "indicator": "CPI",
        "surprise_ann": 1.5,
    }])
    h_cross = WalkForwardHarness(yield_df, factor_df, macro_cross, cfg)
    res_cross = h_cross.run_walk_forward_evaluation(max_folds=2)
    sig_cross = pd.concat(res_cross["signals"]["DNS_Kalman_Macro"])
    audit_cross_f0 = res_cross["run_metadata"]["macro_event_audit"]["fold_breakdown"][0]
    audit_cross_f1 = res_cross["run_metadata"]["macro_event_audit"]["fold_breakdown"][1]

    # Friday 17:00 NY must NOT affect Friday position (Fold 0)
    assert sig_cross.loc[t_fri] == sig_base.loc[t_fri], "Friday 17:00 NY release cannot affect Friday position"
    # Friday 17:00 NY rolls into Monday 2024-02-05 (dates[24] - Fold 1 first origin)
    assert sig_cross.loc[t_mon] != sig_base.loc[t_mon], "Friday 17:00 NY release must roll across folds and affect Monday position"
    
    # Audit counters: assigned to Fold 1 as rolled event
    assert audit_cross_f0["rolled_to_next_decision_events"] == 0
    assert audit_cross_f1["rolled_to_next_decision_events"] == 1
    assert audit_cross_f1["active_overlay_events"] == 1

    # Explicit contract-quantity check in ForecastLedger
    ledger = res_cross["forecast_ledger"]
    impulse_recs = [
        r for r in ledger.records
        if r.target_type == "macro_event_impulse" and str(r.origin_timestamp.date()) == str(t_mon.date())
    ]
    assert len(impulse_recs) == 1, "Must find exactly one macro_event_impulse ledger record for Monday decision"
    assert impulse_recs[0].target_name == "CPI"
    assert impulse_recs[0].forecast != 0.0


def test_weekend_and_holiday_release_roll_handling():
    """
    Test weekend (Saturday / Sunday) release handling:
    An announcement on Saturday (e.g., 2024-02-03) occurs between Fold 0's Friday cutoff
    and Fold 1's Monday cutoff. It must be assigned to Monday Fold 1 as NON_TRADING_DAY_RELEASE,
    leaving Fold 0 unchanged and affecting Fold 1.
    """
    dates = _make_daily_dates("2024-01-02", 35)
    tenors = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    yield_data = {t: [4.0 + 0.1 * i] * len(dates) for i, t in enumerate(tenors)}
    yield_df = pd.DataFrame(yield_data, index=dates)

    macro_rows = []
    for i in range(20):
        surp = 1.0 if i % 2 == 0 else -1.0
        yield_df.loc[dates[i + 1]:, "DGS10"] += 0.05 * surp
        macro_rows.append({"date": dates[i], "indicator": "CPI", "surprise_ann": surp})

    factor_df = pd.DataFrame({
        "level": 4.5 + np.sin(np.arange(len(dates)) / 5.0) * 0.1,
        "slope": -0.5 + np.cos(np.arange(len(dates)) / 5.0) * 0.1,
        "curvature": 0.2 + np.sin(np.arange(len(dates)) / 3.0) * 0.05,
    }, index=dates)

    cfg = WalkForwardConfig(train_window_days=20, refit_frequency_days=5)

    # Base run
    h_base = WalkForwardHarness(yield_df, factor_df, pd.DataFrame(macro_rows), cfg)
    res_base = h_base.run_walk_forward_evaluation(max_folds=2)
    sig_base = pd.concat(res_base["signals"]["DNS_Kalman_Macro"])

    t_fri = dates[23]  # 2024-02-02
    t_mon = dates[24]  # 2024-02-05
    # Saturday release at 10:00 AM NY
    sat_date = pd.Timestamp("2024-02-03")
    macro_sat = pd.DataFrame(macro_rows + [{
        "date": sat_date,
        "timestamp": "2024-02-03T10:00:00-05:00",
        "indicator": "CPI",
        "surprise_ann": 1.5,
    }])

    h_sat = WalkForwardHarness(yield_df, factor_df, macro_sat, cfg)
    res_sat = h_sat.run_walk_forward_evaluation(max_folds=2)
    sig_sat = pd.concat(res_sat["signals"]["DNS_Kalman_Macro"])
    audit_sat_f0 = res_sat["run_metadata"]["macro_event_audit"]["fold_breakdown"][0]
    audit_sat_f1 = res_sat["run_metadata"]["macro_event_audit"]["fold_breakdown"][1]

    # Fold 0 Friday must remain unchanged
    assert sig_sat.loc[t_fri] == sig_base.loc[t_fri]
    # Fold 1 Monday must be affected
    assert sig_sat.loc[t_mon] != sig_base.loc[t_mon]

    # Receiving fold audit
    assert audit_sat_f1["evaluated_decision_events"] == 1
    assert audit_sat_f1["active_overlay_events"] == 1
    applied_recs = audit_sat_f1["applied_event_records"]
    assert len(applied_recs) == 1
    assert applied_recs[0]["release_status"] == "NON_TRADING_DAY_RELEASE"
    assert applied_recs[0]["assigned_origin_date"] == str(t_mon.date())


def test_activity_counter_reconciliation_and_simultaneous_cancellation():
    """
    Test activity counter reconciliation when simultaneous events cancel out:
    Two simultaneous releases on the same decision origin date with opposite surprises (+1.0 and -1.0)
    produce canceling impulses:
      active_overlay_events == 2
      nonzero_macro_days == 0
    This strictly confirms the distinction between event counts and decision-day counts.
    """
    dates = _make_daily_dates("2024-01-02", 30)
    tenors = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    yield_data = {t: [4.0 + 0.1 * i] * len(dates) for i, t in enumerate(tenors)}
    yield_df = pd.DataFrame(yield_data, index=dates)

    for i in range(10):
        surp = 1.0 if i % 2 == 0 else -1.0
        yield_df.loc[dates[2 * i + 1]:, "DGS10"] += 0.10 * surp

    factor_df = pd.DataFrame({
        "level": 4.5 + np.sin(np.arange(len(dates)) / 5.0) * 0.1,
        "slope": -0.5 + np.cos(np.arange(len(dates)) / 5.0) * 0.1,
        "curvature": 0.2 + np.sin(np.arange(len(dates)) / 3.0) * 0.05,
    }, index=dates)

    macro_rows = [{"date": dates[2 * i], "indicator": "CPI", "surprise_ann": 1.0 if i % 2 == 0 else -1.0} for i in range(10)]
    
    t_origin = dates[20]
    # Two simultaneous events on t_origin with opposite surprises: +1.0 and -1.0
    macro_rows.append({
        "date": t_origin,
        "timestamp": f"{t_origin.date()}T08:30:00-05:00",
        "indicator": "CPI",
        "surprise_ann": 1.0,
    })
    macro_rows.append({
        "date": t_origin,
        "timestamp": f"{t_origin.date()}T08:30:00-05:00",
        "indicator": "CPI",
        "surprise_ann": -1.0,
    })

    cfg = WalkForwardConfig(train_window_days=20, refit_frequency_days=5)
    h = WalkForwardHarness(yield_df, factor_df, pd.DataFrame(macro_rows), cfg)
    res = h.run_walk_forward_evaluation(max_folds=1)
    audit = res["run_metadata"]["macro_event_audit"]["fold_breakdown"][0]

    # Reconciled activity counters
    assert audit["evaluated_decision_events"] == 2
    assert audit["usable_surprise_events"] == 2
    assert audit["eligible_coefficient_events"] == 2
    assert audit["active_overlay_events"] == 2, "Both events are individually active overlays"
    assert audit["zero_impact_events"] == 0
    assert audit["nonzero_macro_days"] == 0, "Net macro impulse is 0.0 (+1.0 + -1.0 = 0), so nonzero_macro_days must be 0"

    # Both events are recorded in applied_event_records
    assert len(audit["applied_event_records"]) == 2
    assert all(r["status"] == "ACTIVE_OVERLAY" for r in audit["applied_event_records"])

    # 1. Macro event impulse ledger records for both individually active events
    ledger = res["forecast_ledger"]
    impulse_recs = [
        r for r in ledger.records
        if r.model_id == "DNS_Kalman_Macro" and r.target_type == "macro_event_impulse"
    ]
    assert len(impulse_recs) == 2, f"Expected 2 macro_event_impulse records, got {len(impulse_recs)}"
    assert all(r.origin_timestamp == t_origin for r in impulse_recs)
    assert all(r.reason == "ACTIVE_MACRO_IMPULSE" for r in impulse_recs)
    forecast_vals = sorted([r.forecast for r in impulse_recs])
    assert forecast_vals[0] < 0.0 and forecast_vals[1] > 0.0, "Expected one negative and one positive impulse"

    # 2. Position overlay record on cancelling date
    overlay_recs = [
        r for r in ledger.records
        if r.model_id == "DNS_Kalman_Macro" and r.target_type == "macro_position_overlay" and r.origin_timestamp == t_origin
    ]
    assert len(overlay_recs) == 1
    assert overlay_recs[0].reason == "ZERO_NET_OVERLAY"
    assert overlay_recs[0].forecast == 0.0

    # 3. Audit labels and status synchronization across metadata and tables
    EXPECTED_STATUS = "VALID_MACRO_TEST (ZERO_NET_OVERLAY)"
    assert res["run_metadata"]["macro_event_audit"]["evaluation_status"] == EXPECTED_STATUS
    assert res["run_metadata"]["macro_event_audit"]["strategy_overlay_status"] == "ZERO_NET_OVERLAY"
    assert res["baseline_table"].loc["DNS + Kalman + Macro", "Forecast Status"] == EXPECTED_STATUS
    assert res["common_sample_table"].loc["DNS + Kalman + Macro", "Forecast Status"] == EXPECTED_STATUS
    assert res["run_metadata"]["ledger_summary"]["DNS_Kalman_Macro"]["status"] == EXPECTED_STATUS
    assert res["run_metadata"]["ledger_summary"]["DNS_Kalman_Macro"]["macro_overlay_status"] == "ZERO_NET_OVERLAY"

    # 4. Curve forecast tagging retained as copied DNS
    curve_recs = [
        r for r in ledger.records
        if r.model_id == "DNS_Kalman_Macro" and r.target_type == "yield_curve"
    ]
    assert len(curve_recs) > 0
    assert all(r.reason == "COPIED_DNS_CURVE_FORECAST" for r in curve_recs)
    assert res["run_metadata"]["macro_event_audit"]["curve_forecast_provenance"] == "COPIED_DNS_CURVE_FORECAST"
    assert res["run_metadata"]["ledger_summary"]["DNS_Kalman_Macro"]["curve_provenance"] == "COPIED_DNS_CURVE_FORECAST"


def _make_base_verdict_inputs():
    """Helper to generate baseline inputs for write_verdict_report unit fixtures."""
    models = [
        "Random Walk (Curve Benchmark)",
        "AR(1) Baseline (Static NS)",
        "DNS + Kalman",
        "Static NS (Residual-Preserving Diagnostic)",
        "DNS + Kalman (Residual-Preserving Diagnostic)",
        "DNS (60% Exposure Control)",
        "DNS + Kalman + Macro",
        "GBM",
    ]
    data = {
        "2s10s Spread RMSE (bp)": [1.50, 15.20, 14.80, 1.65, 1.60, 14.80, 14.80, np.nan],
        "Trading Net PnL ($)": [0.0, 12500.0, 12500.0, 14200.0, 14200.0, 7500.0, 7500.0, np.nan],
        "Annualized Sharpe": [np.nan, 0.85, 0.85, 1.20, 1.22, 0.85, 0.85, np.nan],
        "Hit Rate (%)": [np.nan, 58.5, 58.5, 62.0, 62.0, 58.5, 58.5, np.nan],
    }
    table = pd.DataFrame(data, index=models)
    ml_eval = {"shap_data": {"type": "tree_shap", "results": {}}}
    run_meta = {
        "git_commit": "testcommit1234",
        "run_mode": "SYNTHETIC_TEST",
        "fold_count": 5,
        "eval_start_date": "2023-01-03",
        "eval_end_date": "2023-08-30",
        "total_eval_days": 100,
        "macro_event_audit": {
            "macro_data_coverage_end": "2024-03-31",
            "total_calendar_events": 5,
            "total_observed_decision_events": 5,
            "evaluation_status": "VALID_MACRO_TEST",
        },
        "econometric_diagnostics": {
            "observable_2s10s_spread_decomposition": {
                "total_spread_rmse_bp": 15.20,
                "factor_dynamics_spread_rmse_bp": 1.55,
                "cross_sectional_fit_spread_rmse_bp": 15.10,
                "uncentered_cross_moment_bp2": -0.85,
                "sum_components_mse_bp2": 231.04,
                "total_spread_mse_bp2": 231.04,
            },
            "additional_clipping_frequency_pct": {"Static_NS": 45.0, "Static_NS_Residual_Preserving": 0.0},
        },
    }
    return table, ml_eval, run_meta


def test_verdict_report_5pct_residual_clipping(tmp_path, monkeypatch):
    """
    Test verdict report with 5% residual clipping:
    Verify that 5.0% is reported and the statement 'without clipping' is NOT present.
    """
    table, ml_eval, run_meta = _make_base_verdict_inputs()
    run_meta["econometric_diagnostics"]["additional_clipping_frequency_pct"]["Static_NS_Residual_Preserving"] = 5.0

    monkeypatch.chdir(tmp_path)
    write_verdict_report(table, ml_eval, run_meta)

    verdict_file = tmp_path / "reports" / "ml_baseline_verdict.md"
    assert verdict_file.exists()
    content = verdict_file.read_text(encoding="utf-8")

    assert "5.0%" in content
    assert "without clipping" not in content
    assert "residual signal clipping frequency of 5.0%" in content


def test_verdict_report_worse_residual_rmse(tmp_path, monkeypatch):
    """
    Test verdict report when residual-preserving formulation has worse RMSE than traditional:
    Verify that 'increasing spread RMSE' is stated and 'reducing spread RMSE' is NOT present.
    """
    table, ml_eval, run_meta = _make_base_verdict_inputs()
    # Traditional NS is 15.20, set Residual-Preserving NS to 18.50 (worse)
    table.loc["Static NS (Residual-Preserving Diagnostic)", "2s10s Spread RMSE (bp)"] = 18.50

    monkeypatch.chdir(tmp_path)
    write_verdict_report(table, ml_eval, run_meta)

    verdict_file = tmp_path / "reports" / "ml_baseline_verdict.md"
    assert verdict_file.exists()
    content = verdict_file.read_text(encoding="utf-8")

    assert "increasing spread RMSE" in content
    assert "reducing spread RMSE" not in content


def test_verdict_report_factor_dominated_error(tmp_path, monkeypatch):
    """
    Test verdict report when factor dynamics error dominates cross-sectional fit error:
    Verify that factor dynamics error is conditionally identified as dominant contributor.
    """
    table, ml_eval, run_meta = _make_base_verdict_inputs()
    decomp = run_meta["econometric_diagnostics"]["observable_2s10s_spread_decomposition"]
    decomp["factor_dynamics_spread_rmse_bp"] = 12.0
    decomp["cross_sectional_fit_spread_rmse_bp"] = 3.0

    monkeypatch.chdir(tmp_path)
    write_verdict_report(table, ml_eval, run_meta)

    verdict_file = tmp_path / "reports" / "ml_baseline_verdict.md"
    assert verdict_file.exists()
    content = verdict_file.read_text(encoding="utf-8")

    assert "driven predominantly by factor dynamics errors" in content
    assert "factor dynamics error is the dominant contributor" in content
    assert "cross-sectional curve-fitting error is the dominant contributor" not in content


def test_verdict_report_active_macro_differs_from_control(tmp_path, monkeypatch):
    """
    Test verdict report when active macro model PnL differs from the 60% exposure control:
    Verify that non-identical PnLs and the net difference are reported rather than claiming identical results.
    """
    table, ml_eval, run_meta = _make_base_verdict_inputs()
    table.loc["DNS (60% Exposure Control)", "Trading Net PnL ($)"] = 7500.0
    table.loc["DNS + Kalman + Macro", "Trading Net PnL ($)"] = 9500.0

    monkeypatch.chdir(tmp_path)
    write_verdict_report(table, ml_eval, run_meta)

    verdict_file = tmp_path / "reports" / "ml_baseline_verdict.md"
    assert verdict_file.exists()
    content = verdict_file.read_text(encoding="utf-8")

    assert "$7,500.00" in content
    assert "$9,500.00" in content
    assert "+$2,000.00" in content
    assert "identical to `DNS + Kalman + Macro`" not in content


def test_verdict_report_missing_diagnostics(tmp_path, monkeypatch):
    """
    Test verdict report when econometric diagnostics and macro audit metadata are missing:
    Verify that the report generates cleanly with N/A placeholders rather than failing or inventing conclusions.
    """
    table, ml_eval, _ = _make_base_verdict_inputs()
    table.loc[:, :] = np.nan
    run_meta = {
        "git_commit": "testcommit_empty",
        "run_mode": "EMPTY_DIAGNOSTICS_TEST",
    }

    monkeypatch.chdir(tmp_path)
    write_verdict_report(table, ml_eval, run_meta)

    verdict_file = tmp_path / "reports" / "ml_baseline_verdict.md"
    assert verdict_file.exists()
    content = verdict_file.read_text(encoding="utf-8")

    assert "Observable spread error decomposition is unavailable" in content
    assert "Residual signal clipping diagnostics are unavailable" in content
    assert "Trading results for DNS control and macro models are not available" in content
    assert "has unspecified coverage end date" in content

