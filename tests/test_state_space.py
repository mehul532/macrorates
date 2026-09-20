"""Unit tests for Linear Gaussian Dynamic Nelson-Siegel state-space models."""

import numpy as np
import pandas as pd
import pytest

from src.curve.nelson_siegel import nelson_siegel_loadings
from src.state_space.state_space import (
    DynamicNelsonSiegelMLE,
    KalmanFilterSmoother,
    estimate_and_filter_state_space,
    evaluate_ols_vs_kalman,
)


def generate_synthetic_dns_data(
    n_days: int = 250, seed: int = 42
) -> tuple:
    """Generate synthetic dynamic Nelson-Siegel yields with known VAR(1) parameters."""
    np.random.seed(seed)
    maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0])
    lambda_param = 0.7308
    Lambda = nelson_siegel_loadings(maturities, lambda_param)

    # True VAR(1) parameters
    mu_true = np.array([5.0, -1.5, 1.2])
    A_true = np.diag([0.98, 0.96, 0.92])
    Q_true = np.diag([0.04**2, 0.06**2, 0.08**2])
    H_true = np.eye(len(maturities)) * (0.05**2)

    # Simulate states
    states = np.zeros((n_days, 3))
    states[0] = mu_true
    c = (np.eye(3) - A_true) @ mu_true

    for t in range(1, n_days):
        states[t] = c + A_true @ states[t - 1] + np.random.multivariate_normal([0, 0, 0], Q_true)

    # Simulate yields
    yields = states @ Lambda.T + np.random.multivariate_normal(np.zeros(len(maturities)), H_true, size=n_days)

    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    cols = [f"Y_{int(m*12)}M" if m < 1 else f"Y_{int(m)}Y" for m in maturities]
    mat_dict = {cols[i]: maturities[i] for i in range(len(maturities))}

    df = pd.DataFrame(yields, columns=cols)
    df.insert(0, "date", dates)

    return df, mat_dict, maturities, states, (mu_true, A_true, Q_true, H_true)


def test_kalman_filter_smoother_synthetic_recovery():
    """Verify Kalman filter & RTS smoother recover true latent state trajectories."""
    df, mat_dict, mats, true_states, params = generate_synthetic_dns_data(n_days=300, seed=101)
    mu_true, A_true, Q_true, H_true = params

    y = df[[c for c in df.columns if c != "date"]].values
    kf = KalmanFilterSmoother(
        maturities=mats,
        lambda_param=0.7308,
        mu=mu_true,
        transition_matrix=A_true,
        state_cov=Q_true,
        obs_cov=H_true,
    )

    res = kf.filter_and_smooth(y, dates=df["date"])

    # High correlation with true latent states (>0.95 for level/slope, >0.85 for curvature)
    for i, factor in enumerate(["level", "slope", "curvature"]):
        threshold_filt = 0.85 if factor == "curvature" else 0.95
        threshold_smooth = 0.88 if factor == "curvature" else 0.96

        corr_filt = np.corrcoef(res.filtered_states[factor].values, true_states[:, i])[0, 1]
        corr_smooth = np.corrcoef(res.smoothed_states[factor].values, true_states[:, i])[0, 1]
        assert corr_filt > threshold_filt, f"Filtered correlation for {factor} should exceed {threshold_filt}, got {corr_filt}"
        assert corr_smooth > threshold_smooth, f"Smoothed correlation for {factor} should exceed {threshold_smooth}, got {corr_smooth}"

    # RTS Smoother variance reduction: Smoothed uncertainty <= Filtered uncertainty
    P_filt_trace = np.trace(res.filtered_cov, axis1=1, axis2=2)
    P_smooth_trace = np.trace(res.smoothed_cov, axis1=1, axis2=2)
    assert np.mean(P_smooth_trace) < np.mean(P_filt_trace)


def test_kalman_missing_data_resilience():
    """Verify Kalman filter handles missing observations (NaNs) without crashing."""
    df, mat_dict, mats, _, params = generate_synthetic_dns_data(n_days=100, seed=202)
    mu_true, A_true, Q_true, H_true = params

    y_with_holes = df[[c for c in df.columns if c != "date"]].values.copy()
    # Inject holes on every 5th day for 30Y tenor, and whole missing days
    y_with_holes[::5, -1] = np.nan
    y_with_holes[25:28, :] = np.nan

    kf = KalmanFilterSmoother(
        maturities=mats,
        lambda_param=0.7308,
        mu=mu_true,
        transition_matrix=A_true,
        state_cov=Q_true,
        obs_cov=H_true,
    )

    res = kf.filter_and_smooth(y_with_holes, dates=df["date"])
    assert len(res.filtered_states) == 100
    assert not res.filtered_states.isna().any().any()
    assert not res.smoothed_states.isna().any().any()


def test_estimate_and_filter_pipeline():
    """Verify the full estimation and evaluation pipeline on synthetic yield panel."""
    df, mat_dict, mats, _, _ = generate_synthetic_dns_data(n_days=150, seed=303)

    res = estimate_and_filter_state_space(
        yield_df=df,
        maturities_dict=mat_dict,
        lambda_param=0.7308,
        use_mle_optimization=False,  # Test OLS-VAR(1) rapid estimation
    )

    assert not np.isnan(res.log_likelihood)
    assert res.filtered_states.shape == (150, 4)
    assert res.smoothed_states.shape == (150, 4)
