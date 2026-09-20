"""
Unit tests for Walk-Forward Harness, Regime Tagging, and Attribution Chain.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.backtest.regimes import (
    load_episodes_config,
    tag_fed_regimes,
    slice_metrics_by_regime,
)
from src.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardHarness,
)
from src.backtest.backtest import (
    compute_episode_performance,
    compute_regime_robustness_table,
    CostModelV1Config,
    RelativeValueBacktestEngine,
)
from src.strategy.portfolio import compute_continuous_positions


def test_episodes_config_loading():
    """Verify episodes config loads valid named episodes and calendar sub-periods."""
    config = load_episodes_config()
    assert "episodes" in config
    assert "calendar_subperiods" in config
    
    expected_episodes = [
        "pre_covid_normalization",
        "covid_shock",
        "zlb_qe",
        "inflation_hiking_shock_2022",
        "post_hiking_easing_cycle",
    ]
    for ep in expected_episodes:
        assert ep in config["episodes"], f"Missing episode: {ep}"
        assert "start_date" in config["episodes"][ep]
        assert "end_date" in config["episodes"][ep]
        # Verify valid dates
        s = pd.to_datetime(config["episodes"][ep]["start_date"])
        e = pd.to_datetime(config["episodes"][ep]["end_date"])
        assert s < e


def test_regime_tagging_ex_ante_purity():
    """Verify fed_regime_ex_ante uses zero future information."""
    dates = pd.date_range("2020-01-01", periods=150, freq="B")
    rates = np.linspace(1.5, 3.5, 150)
    
    df1 = pd.DataFrame({
        "DGS3MO": rates,
        "DGS2": rates + 0.2,
        "DGS10": rates + 0.8,
    }, index=dates)
    
    reg1 = tag_fed_regimes(df1, lookback_days=30)
    assert "fed_regime_ex_ante" in reg1.columns
    assert "fed_regime_ex_post" in reg1.columns
    
    # Modify future short rates after day 75
    df2 = df1.copy()
    df2.iloc[75:, 0] += 10.0  # Massive future shock
    reg2 = tag_fed_regimes(df2, lookback_days=30)
    
    # Assert ex-ante regime up to day 75 is identical bit-for-bit
    pd.testing.assert_series_equal(
        reg1.iloc[:75]["fed_regime_ex_ante"],
        reg2.iloc[:75]["fed_regime_ex_ante"],
        check_names=False,
    )


def test_regime_threshold_rules_partitioning():
    """Verify empirical threshold rules partition dates cleanly without missing values."""
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    df = pd.DataFrame({
        "DGS3MO": np.linspace(0.1, 4.0, 100),
        "DGS2": np.linspace(0.5, 4.2, 100),
        "DGS10": np.linspace(1.2, 4.1, 100),  # Inverting at the end
    }, index=dates)
    
    regimes = tag_fed_regimes(df)
    assert not regimes["threshold_rate_level"].isna().any()
    assert not regimes["threshold_curve_shape"].isna().any()
    assert not regimes["threshold_realized_vol"].isna().any()
    assert not regimes["fed_regime_ex_ante"].isna().any()


def test_walk_forward_fold_generation():
    """Verify rolling lookback window generates strictly out-of-sample forward-moving folds."""
    dates = pd.date_range("2010-01-01", periods=500, freq="B")
    yield_df = pd.DataFrame({"DGS2": 2.0, "DGS10": 3.0}, index=dates)
    factor_df = pd.DataFrame({"kf_level": 3.0, "kf_slope": -1.0, "kf_curvature": 0.0}, index=dates)
    macro_df = pd.DataFrame(columns=["date", "indicator", "surprise_ann"])
    
    cfg = WalkForwardConfig(train_window_days=100, refit_frequency_days=20)
    harness = WalkForwardHarness(yield_df, factor_df, macro_df, config=cfg)
    folds = harness.generate_folds()
    
    assert len(folds) > 0
    for train_idx, test_idx in folds:
        assert len(train_idx) == 100
        assert len(test_idx) <= 20
        # Zero lookahead assertion: max train date < min test date
        assert train_idx[-1] < test_idx[0]


def test_attribution_chain_reconciliation():
    """Assert Gross PnL - Trade Costs - Roll Costs + Interest == Net PnL exactly."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    yield_df = pd.DataFrame({
        "DGS2": np.linspace(3.0, 3.5, 10),
        "DGS5": np.linspace(3.2, 3.6, 10),
        "DGS10": np.linspace(3.4, 3.8, 10),
        "DGS3MO": [2.0] * 10,
    }, index=dates)
    
    sig = pd.Series([1.0, 1.0, -1.0, -1.0, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0], index=dates)
    pos_df = compute_continuous_positions(sig, strategy_type="2s10s", target_dv01=10_000.0)
    
    engine = RelativeValueBacktestEngine(initial_capital=1_000_000.0)
    res = engine.run_strategy("Test_Chain", pos_df, yield_df)
    
    pnl = res.pnl_components
    reconciled_net = pnl["gross_pnl"] - pnl["trade_cost"] - pnl["roll_cost"] + pnl["cash_interest"]
    
    # Net PnL must balance penny-for-penny
    np.testing.assert_allclose(
        pnl["net_pnl"].values,
        reconciled_net.values,
        rtol=1e-5,
        atol=1e-2,
        err_msg="Attribution chain does not balance!",
    )


def test_walk_forward_evaluation_run():
    """Run walk-forward evaluation on historical slice and verify baseline table metrics."""
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet")
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet")
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet")
    
    # Fast test on recent sample (last 850 days ~ 3.3 years)
    yield_sub = yield_df.iloc[-850:]
    factor_sub = factor_df.iloc[-850:]
    
    cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=42)
    harness = WalkForwardHarness(yield_sub, factor_sub, macro_df, config=cfg)
    eval_res = harness.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=2)
    
    table = eval_res["baseline_table"]
    assert len(table) == 6
    assert "Random + Walk" in table.index
    assert "PCA + VAR" in table.index
    assert "Static + NS" in table.index
    assert "DNS + Kalman" in table.index
    assert "DNS + Kalman + Macro" in table.index
    assert "GBM" in table.index
    assert "OOS Curve RMSE (bp)" in table.columns
    assert "Factor Forecast RMSE (bp)" in table.columns
    assert "Strategy Sharpe" in table.columns
    assert "Annual Turnover (lots)" in table.columns


def test_episode_and_regime_tables():
    """Verify episode and regime robustness tables compute without errors."""
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet").iloc[-500:]
    sig = pd.Series(1.0, index=pd.to_datetime(yield_df["date"]))
    pos_df = compute_continuous_positions(sig, strategy_type="2s10s", target_dv01=10_000.0)
    
    engine = RelativeValueBacktestEngine(initial_capital=10_000_000.0)
    res = engine.run_strategy("Test_Ep", pos_df, yield_df)
    
    ep_table = compute_episode_performance(res)
    assert isinstance(ep_table, pd.DataFrame)
    
    robust_table = compute_regime_robustness_table(res, yield_df)
    assert isinstance(robust_table, pd.DataFrame)
    assert "Rate Level Threshold" in robust_table.index.get_level_values(0)
    assert "Fed Policy Stance (Ex-Ante)" in robust_table.index.get_level_values(0)
