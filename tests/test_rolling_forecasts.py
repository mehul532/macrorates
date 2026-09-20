"""
Unit tests for Genuine Rolling One-Step Forecasts (Prompt 2).

Verifies:
1. Canonical tenor mapping integrity (order-invariance, exact dictionary lookup).
2. Pure rolling 1-step forecasts for RW, PCA/VAR, NS/AR, and DNS.
3. Causal prefix invariance: future data perturbations do not leak into earlier forecasts.
4. Stationary bounded parameterization of DNS transition matrix.
5. Exact reconciliation between ForecastLedger and baseline table RMSE.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.curve.canonical import (
    CANONICAL_TENORS,
    CORE_BENCHMARK_TENORS,
    get_canonical_maturities,
    compute_observable_spreads,
)
from src.curve.nelson_siegel import NelsonSiegelAR1Forecaster, nelson_siegel_loadings
from src.curve.pca import PCAVARForecaster
from src.state_space.state_space import (
    KalmanFilterSmoother,
    estimate_and_filter_state_space,
    evaluate_ols_vs_kalman,
)
from src.backtest.contracts import ForecastStatus
from src.backtest.walk_forward import WalkForwardConfig, WalkForwardHarness


def test_canonical_tenor_mapping():
    """Verify that tenor lookups are order-invariant and reject invalid column names."""
    cols = ["DGS10", "DGS2", "DGS3MO", "DGS30"]
    mats = get_canonical_maturities(cols)
    expected = np.array([10.0, 2.0, 0.25, 30.0])
    np.testing.assert_allclose(mats, expected)

    # Reordered columns
    cols_rev = list(reversed(cols))
    mats_rev = get_canonical_maturities(cols_rev)
    np.testing.assert_allclose(mats_rev, list(reversed(expected)))

    # Reject invalid column
    with pytest.raises(KeyError, match="Tenors not recognized"):
        get_canonical_maturities(["DGS10", "UNKNOWN_TENOR"])


def test_observable_spreads_computation():
    """Verify observable 2s10s spread and 2s5s10s butterfly formulas."""
    cols = ["DGS2", "DGS5", "DGS10"]
    yields = np.array([
        [2.0, 2.5, 3.0],   # 2s10s = 1.0, 2s5s10s = 2*2.5 - 2.0 - 3.0 = 0.0
        [3.0, 4.0, 4.5],   # 2s10s = 1.5, 2s5s10s = 2*4.0 - 3.0 - 4.5 = 0.5
    ])
    spreads = compute_observable_spreads(yields, cols)
    assert "2s10s" in spreads
    assert "2s5s10s" in spreads
    np.testing.assert_allclose(spreads["2s10s"], [1.0, 1.5])
    np.testing.assert_allclose(spreads["2s5s10s"], [0.0, 0.5])


def test_nelson_siegel_ar1_forecaster_causal_prefix_invariance():
    """Verify that perturbing future test data does not affect prior rolling predictions."""
    np.random.seed(42)
    mats = np.array([0.25, 1.0, 2.0, 5.0, 10.0, 30.0])
    T_tr, T_te, N = 100, 20, len(mats)
    
    loadings = nelson_siegel_loadings(mats, 0.7308)
    factors_tr = np.column_stack([
        np.linspace(3.0, 4.0, T_tr),
        np.linspace(-1.5, -0.5, T_tr),
        np.linspace(0.5, 1.0, T_tr),
    ])
    y_tr = factors_tr @ loadings.T + np.random.normal(0, 0.02, (T_tr, N))
    
    factors_te = np.column_stack([
        np.linspace(4.0, 4.2, T_te),
        np.linspace(-0.5, -0.2, T_te),
        np.linspace(1.0, 1.2, T_te),
    ])
    y_te1 = factors_te @ loadings.T + np.random.normal(0, 0.02, (T_te, N))
    y_te2 = y_te1.copy()
    # Shock the last 5 days
    y_te2[15:] += 5.0

    forecaster1 = NelsonSiegelAR1Forecaster(lambda_param=0.7308).fit(y_tr, mats)
    pred1, f_pred1, f_obs1 = forecaster1.sequential_predict_and_update(y_te1)

    forecaster2 = NelsonSiegelAR1Forecaster(lambda_param=0.7308).fit(y_tr, mats)
    pred2, f_pred2, f_obs2 = forecaster2.sequential_predict_and_update(y_te2)

    # Forecasts up to day 15 must be identical bit-for-bit
    np.testing.assert_allclose(pred1[:15], pred2[:15], err_msg="Causal leak in NS AR(1) forecaster!")
    np.testing.assert_allclose(f_pred1[:15], f_pred2[:15], err_msg="Causal leak in NS factor predictions!")


def test_pca_var_forecaster_causal_prefix_invariance():
    """Verify that perturbing future test data does not affect prior PCA/VAR predictions."""
    np.random.seed(42)
    cols = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    mat_dict = {c: CANONICAL_TENORS[c] for c in cols}
    dates_tr = pd.date_range("2020-01-01", periods=100, freq="B")
    dates_te = pd.date_range("2020-06-01", periods=20, freq="B")
    
    y_tr_df = pd.DataFrame(np.random.normal(3.0, 0.5, (100, len(cols))), index=dates_tr, columns=cols)
    y_tr_df["date"] = dates_tr
    
    y_te1 = np.random.normal(3.5, 0.5, (20, len(cols)))
    y_te2 = y_te1.copy()
    y_te2[15:] += 10.0  # Massive future shock

    forecaster1 = PCAVARForecaster(n_components=3).fit(y_tr_df, maturities_dict=mat_dict)
    pred1, s_pred1, s_obs1 = forecaster1.sequential_predict_and_update(y_te1)

    forecaster2 = PCAVARForecaster(n_components=3).fit(y_tr_df, maturities_dict=mat_dict)
    pred2, s_pred2, s_obs2 = forecaster2.sequential_predict_and_update(y_te2)

    # Day 0 to 15 predictions must be identical
    np.testing.assert_allclose(pred1[:15], pred2[:15], err_msg="Causal leak in PCA/VAR forecaster!")


def test_kalman_sequential_predict_and_update():
    """Verify KalmanFilterSmoother sequential 1-step prediction and update."""
    mats = np.array([2.0, 5.0, 10.0])
    kf = KalmanFilterSmoother(
        maturities=mats,
        lambda_param=0.7308,
        mu=np.array([4.0, -1.0, 1.0]),
        transition_matrix=np.diag([0.98, 0.96, 0.92]),
    )
    
    init_state = np.array([4.0, -1.0, 1.0])
    init_cov = np.eye(3) * 0.01
    
    y_test = np.array([
        [3.0, 3.5, 4.0],
        [3.1, 3.6, 4.1],
        [3.2, 3.7, 4.2],
    ])
    
    y_pred, b_pred, b_filt, p_filt = kf.sequential_predict_and_update(
        y_test, initial_state=init_state, initial_cov=init_cov
    )
    
    assert y_pred.shape == (3, 3)
    assert b_pred.shape == (3, 3)
    assert b_filt.shape == (3, 3)
    assert p_filt.shape == (3, 3, 3)
    # 1-step forecast for step 0 must strictly use init_state
    expected_b0 = (np.eye(3) - kf.A) @ kf.mu + kf.A @ init_state
    np.testing.assert_allclose(b_pred[0], expected_b0)
    np.testing.assert_allclose(y_pred[0], kf.Lambda @ expected_b0)


def test_dns_bounded_stationary_diagonal_stability():
    """Verify that estimate_and_filter_state_space enforces diagonal stationary parameters [0.0, 0.999]."""
    cols = ["DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    mat_dict = {c: CANONICAL_TENORS[c] for c in cols}
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    
    # Generate explosive yield sequence
    df = pd.DataFrame(np.outer(np.linspace(1.0, 10.0, 100), np.ones(len(cols))), index=dates, columns=cols)
    df["date"] = dates

    res = estimate_and_filter_state_space(df, maturities_dict=mat_dict, use_mle_optimization=False)
    
    A_diag = np.diag(res.transition_matrix)
    assert np.all(A_diag >= 0.0), f"Negative transition eigenvalue: {A_diag}"
    assert np.all(A_diag <= 0.999), f"Explosive transition eigenvalue: {A_diag}"
    # Off-diagonals must be zero
    off_diag = res.transition_matrix - np.diag(A_diag)
    np.testing.assert_allclose(off_diag, np.zeros((3, 3)))


def test_ledger_reconciliation_with_baseline_table():
    """Verify that baseline table RMSE matches independent calculation from ForecastLedger."""
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet").iloc[-850:]
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet").iloc[-850:]
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet")

    cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=42)
    harness = WalkForwardHarness(yield_df, factor_df, macro_df, config=cfg)
    harness.X_ml = pd.DataFrame()
    eval_res = harness.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=1)

    table = eval_res["baseline_table"]
    ledger = eval_res["forecast_ledger"]

    # Reconcile Random Walk
    rw_rmse_ledger = ledger.compute_rmse("Random_Walk") * 100.0  # bp
    rw_rmse_table = table.loc["Random + Walk", "OOS Curve RMSE (bp)"]
    assert np.isclose(rw_rmse_ledger, rw_rmse_table, atol=0.05)

    # Reconcile Static NS
    ns_rmse_ledger = ledger.compute_rmse("Static_NS") * 100.0
    ns_rmse_table = table.loc["Static + NS", "OOS Curve RMSE (bp)"]
    assert np.isclose(ns_rmse_ledger, ns_rmse_table, atol=0.05)

    # Reconcile DNS Kalman
    dns_rmse_ledger = ledger.compute_rmse("DNS_Kalman") * 100.0
    dns_rmse_table = table.loc["DNS + Kalman", "OOS Curve RMSE (bp)"]
    assert np.isclose(dns_rmse_ledger, dns_rmse_table, atol=0.05)
