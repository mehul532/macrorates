"""
Unit Tests for Svensson 4-Factor Term Structure Model & Downstream RV Signals.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.curve.svensson import SvenssonCurve, svensson_loadings, compute_aic_bic
from src.strategy.signals import compute_svensson_factor_signals


@pytest.fixture
def sample_maturities():
    return np.array([0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])


def test_svensson_synthetic_exact_recovery(sample_maturities):
    """Assert Svensson model exactly recovers known beta parameters under zero noise."""
    true_b0 = 4.5
    true_b1 = -1.8
    true_b2 = 2.5
    true_b3 = -1.2
    true_l1 = 1.4
    true_l2 = 6.0

    X = svensson_loadings(sample_maturities, true_l1, true_l2, is_scale=True)
    true_betas = np.array([true_b0, true_b1, true_b2, true_b3])
    y_clean = X @ true_betas

    sv = SvenssonCurve(default_lambda1=true_l1, default_lambda2=true_l2, is_scale=True)
    fit = sv.fit_cross_section(y_clean, sample_maturities, optimize_lambdas=False)

    np.testing.assert_allclose(fit.beta0, true_b0, atol=1e-5)
    np.testing.assert_allclose(fit.beta1, true_b1, atol=1e-5)
    np.testing.assert_allclose(fit.beta2, true_b2, atol=1e-5)
    np.testing.assert_allclose(fit.beta3, true_b3, atol=1e-5)
    assert fit.r_squared > 0.9999
    assert fit.rmse < 1e-5


def test_svensson_collinearity_separation_enforced(sample_maturities):
    """Assert joint optimization prevents lambda1 and lambda2 from collapsing into singularity."""
    np.random.seed(42)
    y_noisy = 3.0 + 0.5 * np.log(1 + sample_maturities) + np.random.normal(0, 0.05, len(sample_maturities))

    sv = SvenssonCurve(min_separation=0.40)
    fit = sv.fit_cross_section(y_noisy, sample_maturities, optimize_lambdas=True)

    separation = abs(fit.lambda1 - fit.lambda2)
    assert separation >= 0.38, f"Lambdas collapsed too close: l1={fit.lambda1}, l2={fit.lambda2}"


def test_aic_bic_computation_and_loocv(sample_maturities):
    """Verify information criteria calculation and leave-one-out cross validation."""
    y = np.array([1.5, 1.8, 2.1, 2.5, 3.0, 3.2, 3.5, 3.7, 4.0, 4.2, 4.3])
    sv = SvenssonCurve()
    fit = sv.fit_cross_section(y, sample_maturities, optimize_lambdas=False, compute_loocv=True)

    assert fit.aic is not None and not np.isnan(fit.aic)
    assert fit.bic is not None and not np.isnan(fit.bic)
    # Since n=11 > e^2 (~7.39), BIC penalty k*ln(n) > AIC penalty 2k, so BIC > AIC
    assert fit.bic > fit.aic
    assert fit.loocv_rmse is not None
    assert fit.loocv_rmse > 0.0
    # Out-of-sample LOOCV error should be greater than or equal to in-sample RMSE
    assert fit.loocv_rmse >= fit.rmse * 0.8


def test_svensson_panel_fit_execution():
    """Verify daily panel fitting on sample historical yield data."""
    yield_path = Path("data/processed/yield_panel.parquet")
    if not yield_path.exists():
        pytest.skip("yield_panel.parquet not present")

    df = pd.read_parquet(yield_path).dropna().tail(15)
    tenor_map = {
        "DGS1MO": 1/12, "DGS3MO": 3/12, "DGS6MO": 6/12,
        "DGS1": 1.0, "DGS2": 2.0, "DGS3": 3.0, "DGS5": 5.0,
        "DGS7": 7.0, "DGS10": 10.0, "DGS20": 20.0, "DGS30": 30.0
    }

    sv = SvenssonCurve()
    res_df = sv.fit_panel(df, tenor_map, optimize_lambdas=True, compute_loocv=False)

    assert len(res_df) == len(df)
    required_cols = [
        "date", "sv_level", "sv_slope", "sv_curv1", "sv_curv2",
        "sv_curv_composite", "sv_lambda1", "sv_lambda2", "sv_rmse", "sv_aic", "sv_bic"
    ]
    for c in required_cols:
        assert c in res_df.columns, f"Missing expected column: {c}"
        assert res_df[c].isna().sum() == 0, f"NaNs found in column: {c}"

    assert (res_df["sv_rmse"] > 0.0).all()


def test_svensson_signal_generation_pipeline():
    """Verify compute_svensson_factor_signals produces bounded signals."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    np.random.seed(123)
    factor_df = pd.DataFrame({
        "date": dates,
        "sv_slope": np.cumsum(np.random.normal(0, 0.05, 100)),
        "sv_curv1": np.cumsum(np.random.normal(0, 0.1, 100)),
        "sv_curv2": np.cumsum(np.random.normal(0, 0.1, 100)),
    })

    sig_df = compute_svensson_factor_signals(factor_df, window=20, min_periods=5)
    assert "signal_2s10s_svensson" in sig_df.columns
    assert "signal_fly_svensson" in sig_df.columns
    assert (sig_df["signal_2s10s_svensson"].abs() <= 1.0).all()
    assert (sig_df["signal_fly_svensson"].abs() <= 1.0).all()
