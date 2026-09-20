"""Unit tests for Regime-Switching Kim filter and Bayesian MCMC Dynamic Nelson-Siegel models."""

import numpy as np
import pandas as pd
import pytest

from src.curve.nelson_siegel import nelson_siegel_loadings
from src.data.fred_ingest import CMT_SERIES_CONFIG
from src.data.pipeline import load_yield_panel
from src.state_space.bayesian_state_space import (
    RegimeSwitchingNelsonSiegel,
    RegimeSwitchingResults,
    BayesianNelsonSiegelSampler,
    BayesianStateSpaceResults,
)

DEFAULT_MATURITIES = {k: v["maturity_years"] for k, v in CMT_SERIES_CONFIG.items()}


def generate_synthetic_regime_dns_data(
    n_days: int = 180, seed: int = 42
) -> tuple:
    """
    Generate synthetic yield curve panel with 3 distinct regimes:
      Regime 0: Cutting (Downward trend in level, steepening slope, higher state noise)
      Regime 1: Hold (Stable rates, moderate noise)
      Regime 2: Hiking (Upward trend in level, flattening/inverting slope, high persistence)
    """
    np.random.seed(seed)
    maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0])
    lambda_param = 0.7308
    Lambda = nelson_siegel_loadings(maturities, lambda_param)

    # 3 regimes
    A_regimes = np.array([
        np.diag([0.94, 0.90, 0.85]),  # Cutting: faster mean-reversion
        np.diag([0.97, 0.95, 0.90]),  # Hold: steady
        np.diag([0.99, 0.98, 0.94]),  # Hiking: high persistence
    ])
    mu_regimes = np.array([
        [2.0, 1.0, -0.5],   # Cutting: low level, steep positive slope
        [3.5, 0.0, 0.5],    # Hold: normal
        [5.0, -1.5, 1.2],   # Hiking: high level, inverted slope
    ])
    Q_regimes = np.array([
        np.diag([0.08**2, 0.10**2, 0.12**2]),  # Cutting: volatile
        np.diag([0.03**2, 0.04**2, 0.05**2]),  # Hold: quiet
        np.diag([0.06**2, 0.07**2, 0.08**2]),  # Hiking: elevated
    ])
    H = np.eye(len(maturities)) * (0.04**2)

    # Markov transition matrix
    Pi = np.array([
        [0.96, 0.04, 0.00],
        [0.02, 0.96, 0.02],
        [0.00, 0.03, 0.97],
    ])

    states = np.zeros((n_days, 3))
    regimes = np.zeros(n_days, dtype=int)

    # Initialize in regime 1
    regimes[0] = 1
    states[0] = mu_regimes[1]

    for t in range(1, n_days):
        # Sample regime transition
        r_prev = regimes[t - 1]
        regimes[t] = np.random.choice(3, p=Pi[r_prev])
        r_curr = regimes[t]

        c = (np.eye(3) - A_regimes[r_curr]) @ mu_regimes[r_curr]
        states[t] = c + A_regimes[r_curr] @ states[t - 1] + np.random.multivariate_normal([0, 0, 0], Q_regimes[r_curr])

    yields = states @ Lambda.T + np.random.multivariate_normal(np.zeros(len(maturities)), H, size=n_days)

    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    cols = [f"Y_{int(m*12)}M" if m < 1 else f"Y_{int(m)}Y" for m in maturities]
    mat_dict = {cols[i]: maturities[i] for i in range(len(maturities))}

    df = pd.DataFrame(yields, columns=cols)
    df.insert(0, "date", dates)

    regime_names_map = {0: "Cutting", 1: "Hold", 2: "Hiking"}
    regime_series = pd.Series([regime_names_map[r] for r in regimes], index=dates)

    return df, mat_dict, maturities, states, regimes, regime_series


def test_kim_filter_probabilities_and_collapse():
    """Verify Kim filter executes, probabilities sum to 1.0, and state recovery is accurate."""
    df, mat_dict, mats, true_states, true_regimes, reg_series = generate_synthetic_regime_dns_data(n_days=150, seed=123)

    rs_model = RegimeSwitchingNelsonSiegel(maturities=mats, lambda_param=0.7308)
    res = rs_model.fit_and_filter(
        yield_df=df,
        maturities_dict=mat_dict,
        regime_series=reg_series,
        date_col="date",
    )

    assert isinstance(res, RegimeSwitchingResults)
    assert len(res.filtered_states) == 150

    # Probabilities must sum to 1.0 across regimes on every single date
    p_cols = ["p_cutting", "p_hold", "p_hiking"]
    p_sums = res.filtered_regime_probs[p_cols].sum(axis=1)
    np.testing.assert_allclose(p_sums.values, 1.0, atol=1e-5)

    # Markov transition matrix rows must sum to 1.0
    Pi = res.transition_matrix_pi
    np.testing.assert_allclose(Pi.sum(axis=1), 1.0, atol=1e-5)

    # Correlation with true states
    for i, col in enumerate(["level", "slope", "curvature"]):
        corr = np.corrcoef(res.filtered_states[col].values, true_states[:, i])[0, 1]
        threshold = 0.75 if col == "curvature" else 0.88
        assert corr > threshold, f"{col} correlation {corr:.3f} below threshold {threshold}"


def test_regime_conditional_dynamics_differ():
    """Verify regime-conditional transition matrices exhibit distinct persistence and stationarity."""
    df, mat_dict, mats, _, _, reg_series = generate_synthetic_regime_dns_data(n_days=200, seed=456)

    rs_model = RegimeSwitchingNelsonSiegel(maturities=mats, lambda_param=0.7308)
    res = rs_model.fit_and_filter(
        yield_df=df,
        maturities_dict=mat_dict,
        regime_series=reg_series,
        date_col="date",
    )

    A_all = res.transition_matrices_a
    assert A_all.shape == (3, 3, 3)

    # All regimes must be stationary: spectral radius < 1.0
    for k in range(3):
        eigs = np.abs(np.linalg.eigvals(A_all[k]))
        assert np.all(eigs < 1.0), f"Regime {k} transition matrix not stationary (eigs: {eigs})"

    # Persistence should not be identical across regimes
    diff_0_1 = np.linalg.norm(A_all[0] - A_all[1])
    diff_1_2 = np.linalg.norm(A_all[1] - A_all[2])
    assert diff_0_1 > 0.01
    assert diff_1_2 > 0.01


def test_gibbs_ffbs_credible_intervals():
    """Verify Carter-Kohn FFBS Gibbs sampler produces valid posterior credible intervals."""
    df, mat_dict, mats, true_states, _, _ = generate_synthetic_regime_dns_data(n_days=100, seed=789)

    sampler = BayesianNelsonSiegelSampler(maturities=mats, lambda_param=0.7308)
    res = sampler.fit_mcmc(
        yield_df=df,
        maturities_dict=mat_dict,
        date_col="date",
        n_draws=150,
        burn_in=30,
        thin=2,
        seed=101,
    )

    assert isinstance(res, BayesianStateSpaceResults)
    assert len(res.state_medians) == 100
    assert res.draws_mu.shape[0] == 120  # 150 - 30

    # 95% Credible interval sanity: Upper >= Lower
    for factor in ["level", "slope", "curvature"]:
        assert (res.ci_upper_95[factor] >= res.ci_lower_95[factor]).all()
        assert (res.ci_width_95[factor] > 0).all()

    # Empirical coverage: true state should be inside 95% CI on >= 80% of days
    coverage_level = np.mean(
        (true_states[:, 0] >= res.ci_lower_95["level"].values) &
        (true_states[:, 0] <= res.ci_upper_95["level"].values)
    )
    assert coverage_level >= 0.80, f"Coverage for level {coverage_level:.2f} too low"


def test_stress_period_uncertainty_widening():
    """
    Empirical research test: Verify that Bayesian 95% credible intervals for Level/Slope
    widen significantly during March 2020 and 2022 hiking shock compared to calm baseline.
    """
    yield_df, _ = load_yield_panel()
    # Filter 2017 to 2024 for testing efficiency
    df_sub = yield_df[(yield_df["date"] >= "2017-01-01") & (yield_df["date"] <= "2023-12-31")].copy().reset_index(drop=True)
    # Stride of 3 days to make test fast yet representative
    df_stride = df_sub.iloc[::3].reset_index(drop=True)

    sampler = BayesianNelsonSiegelSampler(maturities=np.array(list(DEFAULT_MATURITIES.values())), lambda_param=0.7308)
    res = sampler.fit_mcmc(
        yield_df=df_stride,
        maturities_dict=DEFAULT_MATURITIES,
        date_col="date",
        n_draws=120,
        burn_in=30,
        thin=2,
        seed=42,
    )

    dates = pd.to_datetime(res.dates)
    calm_mask = (dates >= "2017-01-01") & (dates <= "2018-12-31")
    covid_mask = (dates >= "2020-03-01") & (dates <= "2020-04-30")
    hiking_mask = (dates >= "2022-03-01") & (dates <= "2022-12-31")

    calm_slope_width = res.ci_width_95.loc[calm_mask, "slope"].mean()
    hiking_slope_width = res.ci_width_95.loc[hiking_mask, "slope"].mean()
    calm_curv_width = res.ci_width_95.loc[calm_mask, "curvature"].mean()
    covid_curv_width = res.ci_width_95.loc[covid_mask, "curvature"].mean()

    # Credible interval width sanity
    assert (res.ci_width_95["level"] > 0).all()
    assert (res.ci_width_95["slope"] > 0).all()
    assert (res.ci_width_95["curvature"] > 0).all()

    # Slope uncertainty expands during aggressive 2022 Fed rate hiking cycle
    assert hiking_slope_width >= calm_slope_width, (
        f"2022 Hiking Slope CI width ({hiking_slope_width:.4f}) should be >= calm ({calm_slope_width:.4f})"
    )
    # Curvature uncertainty expands during March 2020 COVID dislocation
    assert covid_curv_width >= calm_curv_width, (
        f"March 2020 Curvature CI width ({covid_curv_width:.4f}) should be >= calm ({calm_curv_width:.4f})"
    )

    # Kim filter state covariance widening test: 2022 hiking covariance > calm covariance
    rs_model = RegimeSwitchingNelsonSiegel(maturities=np.array(list(DEFAULT_MATURITIES.values())), lambda_param=0.7308)
    rs_res = rs_model.fit_and_filter(df_sub, DEFAULT_MATURITIES, date_col="date")
    trace_p = np.trace(rs_res.filtered_cov, axis1=1, axis2=2)
    dates_rs = pd.to_datetime(rs_res.dates)
    calm_rs_cov = np.mean(trace_p[(dates_rs >= "2017-01-01") & (dates_rs <= "2018-12-31")])
    hiking_rs_cov = np.mean(trace_p[(dates_rs >= "2022-03-01") & (dates_rs <= "2022-12-31")])
    assert hiking_rs_cov > calm_rs_cov * 1.15, (
        f"Kim filter 2022 hiking covariance ({hiking_rs_cov:.4f}) did not widen by >15% vs calm ({calm_rs_cov:.4f})"
    )


def test_pymc_dns_model_instantiation():
    """Verify PyMC Dynamic Nelson-Siegel model constructs successfully without errors."""
    mats = np.array([1.0, 2.0, 5.0, 10.0])
    sampler = BayesianNelsonSiegelSampler(maturities=mats, lambda_param=0.7308)
    dummy_yields = np.random.uniform(2.0, 5.0, size=(20, 4))

    model = sampler.build_pymc_dns_model(yield_data=dummy_yields, subsample_step=2)
    assert model is not None
    assert "mu" in model.named_vars
    assert "a_diag" in model.named_vars
    assert "obs" in model.named_vars
