"""Unit tests for macroeconomic surprise engine, contemporaneous HAC regressions, and local projections."""

import numpy as np
import pandas as pd
import pytest

from src.macro.macro_surprises import (
    MacroEventHarmonizer,
    MacroRegressionEngine,
    SurpriseEngine,
)


def test_announcement_surprise_standardization():
    """Verify announcement surprise calculation and Z-score standardization."""
    np.random.seed(42)
    n = 100
    actual = np.random.normal(loc=2.0, scale=0.5, size=n)
    forecast = actual + np.random.normal(loc=0.0, scale=0.2, size=n)
    # Insert some NaNs in forecast
    forecast[0] = np.nan
    forecast[5] = np.nan

    df = pd.DataFrame({"actual": actual, "forecast": forecast})
    raw, std, sigma = SurpriseEngine.compute_announcement_surprise(df)

    assert len(raw) == n
    assert len(std) == n
    assert np.isnan(std.iloc[0])
    assert np.isnan(std.iloc[5])

    valid_std = std.dropna()
    assert abs(valid_std.mean()) < 0.1
    assert abs(valid_std.std() - 1.0) < 0.10
    assert sigma > 0

    # Retrospective full-sample mode produces exact unit sample variance
    _, std_fs, _ = SurpriseEngine.compute_announcement_surprise(df, use_lagged_expanding_scale=False)
    assert abs(std_fs.dropna().std() - 1.0) < 0.05

    # Causal invariance: appending future extreme shock cannot alter past standardized surprises
    df_ext = pd.concat([df, pd.DataFrame({"actual": [1000.0], "forecast": [0.0]})], ignore_index=True)
    _, std_ext, _ = SurpriseEngine.compute_announcement_surprise(df_ext)
    np.testing.assert_allclose(std.dropna().values, std_ext.iloc[:len(df)].dropna().values)


def test_model_surprise_zero_lookahead():
    """
    CRITICAL TEST: Verify expanding AR(1) model surprise uses STRICTLY data through t-1.
    Altering future observations at t+5 must have ZERO effect on surprise at t.
    """
    np.random.seed(123)
    n = 50
    actual_base = np.cumsum(np.random.normal(0, 1, size=n)) + 10.0

    df1 = pd.DataFrame({"actual": actual_base.copy()})
    _, std1, _ = SurpriseEngine.compute_model_surprise(df1, min_history=10)

    # Modify future observations from index 30 onwards
    actual_modified = actual_base.copy()
    actual_modified[30:] += 1000.0  # massive future shock

    df2 = pd.DataFrame({"actual": actual_modified})
    # Compute raw errors to test unstandardized point predictions
    raw1, _, _ = SurpriseEngine.compute_model_surprise(df1, min_history=10)
    raw2, _, _ = SurpriseEngine.compute_model_surprise(df2, min_history=10)

    # All point predictions up to index 29 MUST be mathematically identical!
    for t in range(1, 30):
        assert np.isclose(raw1.iloc[t], raw2.iloc[t], atol=1e-10), (
            f"Lookahead leak detected at t={t}! Changing future data altered past prediction."
        )


def test_strict_surprise_series_separation():
    """Verify that S_ann and S_model are strictly separated and never merged into a generic column."""
    df_events = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=20, freq="MS"),
        "actual": np.random.normal(2, 0.5, 20),
        "forecast": np.random.normal(2, 0.5, 20),
        "previous": np.random.normal(2, 0.5, 20),
        "unit": ["%"] * 20,
    })
    df_events["date"] = pd.to_datetime(df_events["timestamp"].dt.date)

    engine = SurpriseEngine()
    panel_df, metadata = engine.build_unified_surprises_panel({"CPI": df_events})

    assert "surprise_ann" in panel_df.columns
    assert "surprise_model" in panel_df.columns
    assert "raw_surprise_ann" in panel_df.columns
    assert "raw_surprise_model" in panel_df.columns
    assert "surprise" not in panel_df.columns, "Generic un-suffixed 'surprise' column must NOT exist!"


def test_hac_contemporaneous_regression():
    """Test contemporaneous OLS regression with Newey-West HAC standard errors."""
    np.random.seed(42)
    n = 100
    surprise = np.random.normal(0, 1, size=n)
    # Factor change with known true beta = 3.5 bp
    factor_diff = 1.0 + 3.5 * surprise + np.random.normal(0, 0.5, size=n)

    df = pd.DataFrame({"surprise_ann": surprise, "d_factor": factor_diff})
    res = MacroRegressionEngine.run_contemporaneous_regression(
        df,
        factor_diff_col="d_factor",
        surprise_col="surprise_ann",
        maxlags=3,
    )

    assert res["n_obs"] == n
    assert abs(res["beta"] - 3.5) < 0.2
    assert res["hac_se"] > 0
    assert res["t_stat"] > 10.0
    assert res["p_value"] < 0.001
    assert res["r_squared"] > 0.90


def test_local_projections_lagged_controls():
    """Test Jordà local projections across multi-day horizons with strictly lagged controls."""
    np.random.seed(99)
    n_days = 200
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    factors_df = pd.DataFrame({
        "date": dates,
        "kf_level": np.cumsum(np.random.normal(0, 0.05, size=n_days)) + 3.0,
        "kf_slope": np.cumsum(np.random.normal(0, 0.06, size=n_days)) - 0.5,
        "kf_curvature": np.cumsum(np.random.normal(0, 0.1, size=n_days)) + 0.2,
    })

    # Create events matching some dates
    event_indices = list(range(10, n_days - 15, 10))
    event_dates = dates[event_indices]
    surprises_df = pd.DataFrame({
        "date": event_dates,
        "indicator": ["CPI"] * len(event_dates),
        "surprise_ann": np.random.normal(0, 1, size=len(event_dates)),
    })

    lp_res = MacroRegressionEngine.run_local_projections(
        factors_df,
        surprises_df,
        indicator="CPI",
        factor_col="kf_level",
        surprise_col="surprise_ann",
        horizons=[0, 1, 2, 5, 10],
    )

    assert len(lp_res) == 5
    assert set(lp_res["horizon_days"]) == {0, 1, 2, 5, 10}
    assert all(lp_res["hac_se"] > 0)
    assert all(lp_res["ci_95_lower"] < lp_res["ci_95_upper"])
    assert all(lp_res["n_obs"] > 0)
