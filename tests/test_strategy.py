"""
Unit tests for Systematic Relative-Value Strategies, DV01 Neutrality, and V1 Cost Model.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.signals import (
    compute_slope_zscore_signals,
    compute_dns_factor_signals,
    compute_macro_surprise_signals,
    generate_all_signals,
)
from src.strategy.portfolio import (
    allocate_2s10s_spread,
    allocate_2s5s10s_butterfly,
    allocate_outright_duration_trade,
    compute_continuous_positions,
)
from src.strategy.etf_prototype import (
    ETF_SPECS,
    ENGINEERING_VALIDATION_TAG,
    generate_synthetic_etf_panel,
    allocate_etf_2s10s_spread,
    allocate_etf_2s5s10s_butterfly,
)
from src.strategy.backtest import (
    CostModelV1Config,
    RelativeValueBacktestEngine,
)
from src.strategy.strategy import (
    SystematicRelativeValuePipeline,
)


def test_spread_allocation_dv01_neutrality():
    """Verify 2s10s spread positions satisfy <5% residual DV01 tolerance."""
    for target in [5_000.0, 10_000.0, 25_000.0, 50_000.0]:
        for sig in [1.0, -1.0, 0.5, -0.75]:
            alloc = allocate_2s10s_spread(signal=sig, target_dv01=target)
            assert alloc["is_dv01_neutral"] is True
            assert alloc["residual_pct_of_leg"] <= 5.0
            # Check signs match direction
            if sig > 0:
                assert alloc["direction"] == "steepener"
                assert alloc["n_zt"] > 0
                assert alloc["n_zn"] < 0
            elif sig < 0:
                assert alloc["direction"] == "flattener"
                assert alloc["n_zt"] < 0
                assert alloc["n_zn"] > 0


def test_butterfly_allocation_dv01_neutrality_and_wings():
    """Verify 2s5s10s butterfly maintains 50/50 wing DV01 allocation and <5% residual tolerance."""
    for target in [5_000.0, 10_000.0, 30_000.0]:
        for sig in [1.0, -1.0]:
            alloc = allocate_2s5s10s_butterfly(signal=sig, target_dv01=target)
            assert alloc["is_dv01_neutral"] is True
            assert alloc["residual_pct_of_leg"] <= 5.0
            
            # Wing DV01 ratio: each wing should be approximately equal to 0.5 * belly DV01
            belly_dv01 = abs(alloc["zf_total_dv01"])
            w1_dv01 = abs(alloc["zt_total_dv01"])
            w2_dv01 = abs(alloc["zn_total_dv01"])
            assert abs(w1_dv01 - 0.5 * belly_dv01) / (0.5 * belly_dv01) < 0.10
            assert abs(w2_dv01 - 0.5 * belly_dv01) / (0.5 * belly_dv01) < 0.10


def test_outright_duration_trade_explicitly_dropped():
    """Verify outright duration trades raise NotImplementedError as deliberate scope decision."""
    with pytest.raises(NotImplementedError) as exc_info:
        allocate_outright_duration_trade()
    assert "Outright duration trading has been deliberately dropped" in str(exc_info.value)


def test_signals_zero_lookahead():
    """Verify signal generation does not use future data (t > T0)."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(42)
    y2 = np.cumsum(np.random.randn(100) * 0.05) + 2.0
    y5 = np.cumsum(np.random.randn(100) * 0.05) + 2.5
    y10 = np.cumsum(np.random.randn(100) * 0.05) + 3.0
    
    df1 = pd.DataFrame({"DGS2": y2, "DGS5": y5, "DGS10": y10}, index=dates)
    sig1 = compute_slope_zscore_signals(df1, window=30)
    
    # Modify future data after day 50
    df2 = df1.copy()
    df2.iloc[50:, 0] += 5.0  # huge shock to 2Y yield in future
    df2.iloc[50:, 1] += 5.0
    df2.iloc[50:, 2] += 5.0
    sig2 = compute_slope_zscore_signals(df2, window=30)
    
    # Signals up to day 49 must be identical bit-for-bit
    np.testing.assert_allclose(
        sig1.iloc[:50]["signal_2s10s_zscore"].values,
        sig2.iloc[:50]["signal_2s10s_zscore"].values,
        err_msg="Signal leaked future information into past calculations!"
    )


def test_v1_cost_model_deductions():
    """Verify transaction costs and slippage are deducted correctly."""
    cost_cfg = CostModelV1Config(fee_per_contract=2.0, slippage_ticks=1.0)
    engine = RelativeValueBacktestEngine(cost_config=cost_cfg, initial_capital=1_000_000.0)
    
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    pos_df = pd.DataFrame({
        "n_zt": [0, 10, 10, 0, 0],
        "n_zn": [0, -5, -5, 0, 0],
    }, index=dates)
    
    yield_df = pd.DataFrame({
        "DGS2": [2.0, 2.0, 2.0, 2.0, 2.0],
        "DGS10": [3.0, 3.0, 3.0, 3.0, 3.0],
        "DGS3MO": [0.0, 0.0, 0.0, 0.0, 0.0],
    }, index=dates)
    
    res = engine.run_strategy("Test_Cost", pos_df, yield_df)
    # Day 1: bought 10 ZT and sold 5 ZN -> trade cost = 10 * (2 + 1*15.625) + 5 * (2 + 1*31.25)
    zt_cost = 10 * (2.0 + 7.8125)  # ZT tick value is 7.8125
    zn_cost = 5 * (2.0 + 15.625)   # ZN tick value is 15.625
    expected_day1_cost = zt_cost + zn_cost
    
    assert res.pnl_components.loc[dates[1], "trade_cost"] == pytest.approx(expected_day1_cost, rel=1e-2)
    assert res.pnl_components["trade_cost"].sum() > 0


def test_strict_no_repo_financing_carry_rule():
    """Verify that attempting to deduct repo carry raises an immediate ValueError in V1."""
    cost_cfg = CostModelV1Config(deduct_repo_carry=True)
    with pytest.raises(ValueError) as exc_info:
        RelativeValueBacktestEngine(cost_config=cost_cfg)
    assert "CRITICAL VIOLATION: deduct_repo_carry must be False in V1" in str(exc_info.value)


def test_etf_engineering_validation_labeling():
    """Verify that all duration-matched ETF outputs are strictly labeled 'engineering validation only'."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    yield_df = pd.DataFrame({
        "DGS2": np.linspace(4.0, 4.2, 10),
        "DGS5": np.linspace(3.8, 4.0, 10),
        "DGS10": np.linspace(3.7, 3.9, 10),
        "DGS20": np.linspace(4.1, 4.3, 10),
    }, index=dates)
    
    panel = generate_synthetic_etf_panel(yield_df)
    assert panel.attrs["is_engineering_validation"] is True
    assert "ENGINEERING VALIDATION ONLY" in panel.attrs["validation_tag"]
    
    spread_res = allocate_etf_2s10s_spread(signal=1.0)
    assert spread_res["is_engineering_validation"] is True
    assert "ENGINEERING VALIDATION ONLY" in spread_res["validation_tag"]
    
    fly_res = allocate_etf_2s5s10s_butterfly(signal=1.0)
    assert fly_res["is_engineering_validation"] is True
    assert "ENGINEERING VALIDATION ONLY" in fly_res["validation_tag"]


def test_end_to_end_backtest_pipeline():
    """Run full pipeline test and verify comparative scorecard produces valid metrics."""
    pipeline = SystematicRelativeValuePipeline()
    pipeline.load_data()
    signals = pipeline.generate_signals(window=60)
    assert "signal_2s10s_zscore" in signals.columns
    assert "signal_2s10s_dns" in signals.columns
    assert "signal_2s10s_macro_dns" in signals.columns
    
    results = pipeline.run_all_backtests(target_dv01=5_000.0)
    assert "2s10s_Cash_Baseline" in results
    assert "2s10s_Slope_ZScore" in results
    assert "2s10s_Macro_DNS" in results
    
    scorecard = pipeline.build_scorecard(results)
    assert len(scorecard) == 8
    assert "Sharpe Ratio" in scorecard.columns
    assert "Max Drawdown (%)" in scorecard.columns
    # Cash baseline must have zero drawdown and zero cost drag
    assert scorecard.loc["2s10s_Cash_Baseline", "Max Drawdown (%)"] == 0.0
    assert scorecard.loc["2s10s_Cash_Baseline", "Cost Drag (bp/yr)"] == 0.0
