"""Unit tests for yield curve models against synthetic curves with known ground-truth parameters."""

import numpy as np
import pandas as pd
import pytest

from src.curve.curve import fit_pca, fit_static_nelson_siegel
from src.curve.nelson_siegel import (
    StaticNelsonSiegel,
    curvature_peak_maturity,
    nelson_siegel_loadings,
)
from src.curve.pca import YieldCurvePCA
from src.curve.svensson import SvenssonCurve, svensson_loadings


def test_nelson_siegel_synthetic_exact_recovery():
    """Verify Static Nelson-Siegel OLS exactly recovers known parameters with zero noise."""
    true_l = 5.25
    true_s = -1.80
    true_c = 2.40
    true_lambda = 0.7308

    maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    X = nelson_siegel_loadings(maturities, true_lambda)
    true_yields = X @ np.array([true_l, true_s, true_c])

    # Fit with known lambda
    model = StaticNelsonSiegel(lambda_param=true_lambda)
    fit = model.fit_cross_section(true_yields, maturities, optimize_lambda=False)

    # Assert exact numerical recovery
    np.testing.assert_allclose(fit.level, true_l, atol=1e-10)
    np.testing.assert_allclose(fit.slope, true_s, atol=1e-10)
    np.testing.assert_allclose(fit.curvature, true_c, atol=1e-10)
    assert fit.rmse < 1e-10
    assert fit.r_squared > 0.999999


def test_nelson_siegel_noisy_recovery():
    """Verify recovery when mild observation noise (2 bps) is added."""
    np.random.seed(42)
    true_l = 4.50
    true_s = -1.20
    true_c = 1.60
    true_lambda = 0.7308

    maturities = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    X = nelson_siegel_loadings(maturities, true_lambda)
    noisy_yields = X @ np.array([true_l, true_s, true_c]) + np.random.normal(0, 0.02, size=len(maturities))

    model = StaticNelsonSiegel(lambda_param=true_lambda)
    fit = model.fit_cross_section(noisy_yields, maturities, optimize_lambda=False)

    assert abs(fit.level - true_l) < 0.05
    assert abs(fit.slope - true_s) < 0.05
    assert abs(fit.curvature - true_c) < 0.10
    assert fit.rmse < 0.03


def test_curvature_peak():
    """Verify that curvature loading reaches its analytical peak near 2.45 years for lambda=0.7308."""
    tau_star = curvature_peak_maturity(0.7308)
    assert 2.40 < tau_star < 2.50

    # Numerically verify that the loading at tau_star is greater than at nearby points
    mats = np.linspace(1.0, 5.0, 100)
    loadings = nelson_siegel_loadings(mats, 0.7308)[:, 2]
    peak_idx = np.argmax(loadings)
    numerical_peak = mats[peak_idx]
    assert abs(numerical_peak - tau_star) < 0.05


def test_svensson_synthetic_exact_recovery():
    """Verify Svensson 6-parameter model exact recovery on synthetic curve."""
    true_beta = np.array([4.0, -2.0, 1.5, -0.8])
    l1 = 0.73
    l2 = 0.20

    maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    X = svensson_loadings(maturities, l1, l2)
    synthetic_yields = X @ true_beta

    model = SvenssonCurve(init_lambda1=l1, init_lambda2=l2)
    fit = model.fit_cross_section(synthetic_yields, maturities)

    assert fit.rmse < 1e-4
    assert fit.r_squared > 0.9999


def test_pca_variance_and_sign_alignment():
    """Verify PCA variance explained ratio and sign alignment on synthetic yield panels."""
    np.random.seed(123)
    n_days = 200
    maturities = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    mat_dict = {f"Y{int(m)}": m for m in maturities}

    # Generate synthetic yield curves driven by 3 latent factors: Level, Slope, Curvature
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    L = 4.0 + np.cumsum(np.random.normal(0, 0.05, n_days))
    S = -1.0 + np.cumsum(np.random.normal(0, 0.03, n_days))
    C = 1.5 + np.cumsum(np.random.normal(0, 0.02, n_days))

    yield_data = {"date": dates}
    for col, tau in mat_dict.items():
        loadings = nelson_siegel_loadings(np.array([tau]), 0.7308)[0]
        y_tau = L * loadings[0] + S * loadings[1] + C * loadings[2] + np.random.normal(0, 0.005, n_days)
        yield_data[col] = y_tau

    df = pd.DataFrame(yield_data)

    pca_res = fit_pca(df, mat_dict, on_changes=False)

    # 1. Variance explained: Top 3 PCs must explain > 99% of total variance
    summary = pca_res.variance_explained_summary()
    assert summary["R2_3"] > 0.99, f"R2_3 should exceed 0.99, got {summary['R2_3']}"
    assert summary["R2_1"] > 0.70  # Level dominates variance

    # 2. Sign Alignment:
    # PC1 (Level): All loadings strictly positive
    assert np.all(pca_res.loadings[:, 0] > 0), "PC1 loadings should be uniformly positive"

    # PC2 (Slope): Loading at longest maturity > loading at shortest maturity
    assert pca_res.loadings[-1, 1] > pca_res.loadings[0, 1], "PC2 should slope upward"

    # PC3 (Curvature): Belly loading > wings
    belly_idx = np.argmin(np.abs(pca_res.maturities - 5.0))
    wing_avg = 0.5 * (pca_res.loadings[0, 2] + pca_res.loadings[-1, 2])
    assert pca_res.loadings[belly_idx, 2] > wing_avg, "PC3 should peak in the belly"


def test_pca_on_changes():
    """Verify PCA computation on daily yield differences (Delta y) driven by common factors."""
    np.random.seed(456)
    n_days = 200
    maturities = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    mat_dict = {f"Y{int(m)}": m for m in maturities}
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

    # Common factor shocks
    dL = np.random.normal(0, 0.08, n_days)
    dS = np.random.normal(0, 0.04, n_days)
    dC = np.random.normal(0, 0.02, n_days)

    L = 4.0 + np.cumsum(dL)
    S = -1.0 + np.cumsum(dS)
    C = 1.0 + np.cumsum(dC)

    data = {"date": dates}
    for col, tau in mat_dict.items():
        loadings = nelson_siegel_loadings(np.array([tau]), 0.7308)[0]
        data[col] = L * loadings[0] + S * loadings[1] + C * loadings[2] + np.random.normal(0, 0.002, n_days)

    df = pd.DataFrame(data)

    pca_changes = fit_pca(df, mat_dict, on_changes=True)
    assert pca_changes.on_changes is True
    assert len(pca_changes.scores) == n_days - 1
    summary = pca_changes.variance_explained_summary()
    assert summary["R2_3"] > 0.95
    assert pca_changes.loadings[-1, 1] > pca_changes.loadings[0, 1]
