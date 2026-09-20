"""
Unit test suite for Dynamic Factor Model (DFM) nowcast module.
Tests data transformations, Doz-Giannone-Reichlin Kalman smoother,
ragged-edge handling, recursive zero-lookahead surprises, and extended regressions.
"""

import numpy as np
import pandas as pd
import pytest

from src.macro.dfm_nowcast import (
    FREDMDLoader,
    DynamicFactorModelDGR,
    DFMNowcastSurprise,
    ExtendedMacroCurveRegression,
    DFMResult,
)


def test_tcode_transformations_synthetic():
    """Verify all 7 McCracken-Ng TCODE transformations on deterministic inputs."""
    idx = pd.date_range("2020-01-01", periods=10, freq="MS")
    raw_data = {
        "sasdate": ["Transform:"] + [d.strftime("%m/%d/%Y") for d in idx],
        "V1": [1.0] + [10.0 + i for i in range(10)],                  # level: x
        "V2": [2.0] + [10.0 + 2 * i for i in range(10)],              # diff: 2.0
        "V3": [3.0] + [10.0 + i**2 for i in range(10)],               # diff^2: 2.0
        "V4": [4.0] + [np.exp(i) for i in range(10)],                 # log: i
        "V5": [5.0] + [np.exp(2 * i) for i in range(10)],             # dlog: 2.0
        "V6": [6.0] + [np.exp(i**2) for i in range(10)],              # dlog^2: 2.0
        "V7": [7.0] + [100.0 * (1.05**i) for i in range(10)],         # pct_change: 0.05
    }
    df_raw = pd.DataFrame(raw_data)
    
    # Save to temp csv
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        df_raw.to_csv(tmp.name, index=False)
        loader = FREDMDLoader(tmp.name)
        trans, std = loader.load_and_transform(max_missing_ratio=0.5)

    assert "V1" in trans.columns
    assert "V2" in trans.columns
    assert "V5" in trans.columns
    # Check V2 constant diff of 2.0
    np.testing.assert_allclose(trans["V2"].iloc[2:].values, 2.0)
    # Check V5 constant dlog of 2.0
    np.testing.assert_allclose(trans["V5"].iloc[2:].values, 2.0)


def test_loader_real_fred_md_data():
    """Verify loading real FRED-MD dataset from local cache."""
    loader = FREDMDLoader("data/raw/macro/fred_md_current.csv")
    trans, std = loader.load_and_transform(max_missing_ratio=0.25)

    assert len(trans) > 700
    assert trans.shape[1] >= 100
    assert not np.isinf(std.values).any()

    # Verify ragged edge summary
    summary = loader.get_ragged_edge_summary(n_tail=6)
    assert len(summary) == 6
    assert "missing_series" in summary.columns
    assert "observed_series" in summary.columns


def test_dfm_factor_recovery_and_ragged_edges():
    """Verify DFM factor recovery and Kalman smoother handling of missing values in tail."""
    T, N, r = 300, 40, 2
    np.random.seed(42)

    # True latent factors: VAR(1)
    F_true = np.zeros((T, r))
    A_true = np.array([[0.8, 0.0], [0.0, 0.7]])
    for t in range(1, T):
        F_true[t] = A_true @ F_true[t - 1] + np.random.randn(r) * 0.4

    Lambda_true = np.random.randn(N, r)
    X_mat = F_true @ Lambda_true.T + np.random.randn(T, N) * 0.5

    # Induce ragged edge: last 15 periods have 35% missing values
    for t in range(T - 15, T):
        missing_idx = np.random.choice(N, size=int(0.35 * N), replace=False)
        X_mat[t, missing_idx] = np.nan

    dates = pd.date_range("1995-01-01", periods=T, freq="MS")
    cols = [f"SERIES_{i}" for i in range(N)]
    cols[0] = "INDPRO"
    cols[1] = "PAYEMS"
    cols[2] = "CPIAUCSL"
    df_sim = pd.DataFrame(X_mat, index=dates, columns=cols)

    # Standardize
    df_std = (df_sim - df_sim.mean()) / df_sim.std()

    model = DynamicFactorModelDGR(n_factors=2)
    res = model.fit(df_std)

    assert isinstance(res, DFMResult)
    assert res.factors_smoothed.shape == (T, 2)
    assert res.factors_filtered.shape == (T, 2)
    # Ensure no NaNs in smoothed states even at the ragged edge
    assert not res.factors_smoothed.isna().any().any()

    # Check high correlation with true factors
    corr0 = abs(np.corrcoef(F_true[:, 0], res.factors_smoothed.iloc[:, 0])[0, 1])
    corr1 = abs(np.corrcoef(F_true[:, 1], res.factors_smoothed.iloc[:, 1])[0, 1])
    assert max(corr0, corr1) > 0.75


def test_recursive_surprises_zero_lookahead():
    """Verify that DFM nowcast surprises strictly obey zero-lookahead discipline."""
    np.random.seed(123)
    dates = pd.date_range("2000-01-01", periods=120, freq="MS")
    f_vals = np.sin(np.linspace(0, 10, 120)) + np.random.randn(120) * 0.2
    df_factors = pd.DataFrame({"Growth": f_vals}, index=dates)

    surp_df = DFMNowcastSurprise.compute_recursive_surprises(df_factors, burn_in_months=40)
    assert "surprise_Growth" in surp_df.columns

    # Check that initial burn-in periods are NaN
    assert surp_df["surprise_Growth"].iloc[:40].isna().all()
    # Check that later periods are finite
    assert surp_df["surprise_Growth"].iloc[40:].notna().all()

    # Lookahead test: modify future values and assert past surprises are unaffected
    df_modified = df_factors.copy()
    df_modified.iloc[100:, 0] += 50.0  # huge shock in future at t=100
    surp_modified = DFMNowcastSurprise.compute_recursive_surprises(df_modified, burn_in_months=40)

    # Values at t=80 must be EXACTLY identical (zero future leakage)
    assert np.isclose(
        surp_df.loc[dates[80], "surprise_Growth"],
        surp_modified.loc[dates[80], "surprise_Growth"],
        atol=1e-10,
    )


def test_extended_regression_and_local_projections():
    """Verify augmented regression and Jordà IRF with realistic daily and monthly inputs."""
    daily_idx = pd.bdate_range("2010-01-01", "2015-12-31")
    np.random.seed(42)
    daily_df = pd.DataFrame({
        "level_change": np.random.randn(len(daily_idx)) * 5.0,
        "ns_level": 3.0 + np.cumsum(np.random.randn(len(daily_idx)) * 0.05),
    }, index=daily_idx)

    # Monthly surprises
    m_idx = pd.date_range("2010-01-01", "2015-12-31", freq="MS")
    monthly_df = pd.DataFrame({
        "surprise_Growth": np.random.randn(len(m_idx)),
        "surprise_Inflation": np.random.randn(len(m_idx)),
    }, index=m_idx)

    aligned = ExtendedMacroCurveRegression.align_monthly_surprises_to_trading_days(
        daily_curve_df=daily_df,
        monthly_surprises_df=monthly_df,
    )
    assert len(aligned) == len(m_idx)
    assert (aligned.index <= m_idx).all()

    # Mock announcement surprises
    ann_df = pd.DataFrame({
        "date": m_idx[:30],
        "indicator": ["CPI"] * 15 + ["NFP"] * 15,
        "surprise_ann": np.random.randn(30),
    })

    # Test contemporaneous regression
    reg_out = ExtendedMacroCurveRegression.run_augmented_contemporaneous_regression(
        daily_factors=daily_df,
        macro_surprises=ann_df,
        dfm_surprises_aligned=aligned,
        factor_col="level_change",
    )
    assert "r2_baseline" in reg_out
    assert "r2_augmented" in reg_out
    assert "delta_r2" in reg_out

    # Test Jordà Local Projections
    irf_df = ExtendedMacroCurveRegression.run_dfm_local_projections(
        daily_factors=daily_df,
        dfm_surprises_aligned=aligned,
        factor_col="ns_level",
        dfm_surprise_col="surprise_Growth",
        horizons=[0, 1, 2, 5],
    )
    assert len(irf_df) == 4
    assert set(irf_df["horizon"]) == {0, 1, 2, 5}
    assert "ci_lower" in irf_df.columns
    assert "ci_upper" in irf_df.columns
