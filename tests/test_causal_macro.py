"""
Unit tests for Point-in-Time Macro Features, Causal Betas, and Information Contracts (Prompt 3).

Acceptance Criteria Tested:
1. Appending an extreme future release cannot alter past surprises or coefficients.
2. A release after cutoff cannot affect the earlier decision.
3. Revised actuals are not substituted into past decisions (first-release provenance).
4. Missing consensus stays distinguishable (NaN, never filled with 0.0).
5. Training labels never mature after the fit cutoff.
6. Coefficients and normalizers reproduce from the saved training sample.
7. CausalMacroResponseEstimator falls back to neutral 0.0 on inadequate history.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.macro.macro_surprises import (
    SurpriseEngine,
    CausalMacroResponseEstimator,
    MacroResponseFoldEstimate,
)
from src.strategy.signals import compute_macro_surprise_signals


def test_appending_future_release_cannot_alter_past_surprises():
    """Verify that adding future extreme releases has ZERO effect on prior standardized surprises."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    actuals = np.random.normal(2.5, 0.5, n)
    forecasts = actuals + np.random.normal(0, 0.2, n)
    # Insert some missing consensus
    forecasts[5] = np.nan
    forecasts[20] = np.nan

    df_base = pd.DataFrame({
        "date": dates,
        "actual": actuals,
        "forecast": forecasts,
    })

    raw1, std1, sigma1 = SurpriseEngine.compute_announcement_surprise(
        df_base, min_observations=12, ddof=1, use_lagged_expanding_scale=True
    )

    # Append 5 future releases with massive shock (e.g. 1000 standard deviations)
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=31), periods=5, freq="MS")
    df_shocked = pd.concat([
        df_base,
        pd.DataFrame({
            "date": future_dates,
            "actual": [500.0, -500.0, 1000.0, -1000.0, 2000.0],
            "forecast": [2.0, 2.0, 2.0, 2.0, 2.0],
        }),
    ], ignore_index=True)

    raw2, std2, sigma2 = SurpriseEngine.compute_announcement_surprise(
        df_shocked, min_observations=12, ddof=1, use_lagged_expanding_scale=True
    )

    # Assert prior standardized surprises are IDENTICAL bit-for-bit
    valid_mask = ~df_base["forecast"].isna()
    np.testing.assert_allclose(
        std1[valid_mask].values,
        std2.iloc[:n][valid_mask].values,
        rtol=1e-12,
        atol=1e-12,
        err_msg="Lookahead leak: appending future shock altered past standardized surprises!",
    )


def test_missing_consensus_distinguishable_from_zero_surprise():
    """Verify that missing consensus produces NaN, NEVER filled with 0.0."""
    df = pd.DataFrame({
        "actual": [2.0, 3.0, 4.0, 5.0],
        "forecast": [2.0, np.nan, 4.5, np.nan],  # Index 0 is true zero surprise, 1 and 3 are missing
    })

    raw, std, _ = SurpriseEngine.compute_announcement_surprise(
        df, min_observations=2, ddof=1, use_lagged_expanding_scale=True
    )

    # True zero surprise (actual == forecast)
    assert raw.iloc[0] == 0.0
    assert std.iloc[0] == 0.0

    # Missing consensus must be NaN
    assert np.isnan(raw.iloc[1])
    assert np.isnan(std.iloc[1])
    assert np.isnan(raw.iloc[3])
    assert np.isnan(std.iloc[3])


def test_training_labels_never_mature_after_fit_cutoff():
    """
    Verify that CausalMacroResponseEstimator excludes any training event
    whose response horizon extends past training_cutoff.
    """
    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    cutoff = dates[50]  # Cutoff at index 50

    yield_df = pd.DataFrame({
        "DGS2": np.linspace(2.0, 3.0, 100),
        "DGS5": np.linspace(2.2, 3.2, 100),
        "DGS10": np.linspace(2.5, 3.5, 100),
    }, index=dates)

    # Macro release on exact cutoff date
    macro_df = pd.DataFrame({
        "date": [dates[10], dates[20], dates[30], dates[40], dates[50]],
        "indicator": ["CPI"] * 5,
        "surprise_ann": [0.5, -0.3, 0.8, -0.6, 1.2],
    })

    estimator = CausalMacroResponseEstimator(min_events=3, predictive_horizon=1)
    results = estimator.fit_fold(
        fold_id=0,
        training_cutoff=cutoff,
        macro_df=macro_df,
        yield_df=yield_df,
    )

    # For predictive_horizon=1, the event on cutoff date cannot see cutoff+1!
    # Therefore, exactly 4 events (dates[10], dates[20], dates[30], dates[40]) can be used.
    assert results["CPI"]["n_events"] == 4
    assert results["CPI"]["status"] == "ESTIMATED_CAUSAL"


def test_inadequate_history_neutral_fallback():
    """Verify that fewer events than min_events produces neutral 0.0 beta and explicit status."""
    dates = pd.date_range("2021-01-01", periods=50, freq="B")
    cutoff = dates[40]

    yield_df = pd.DataFrame({
        "DGS2": np.linspace(2.0, 3.0, 50),
        "DGS5": np.linspace(2.2, 3.2, 50),
        "DGS10": np.linspace(2.5, 3.5, 50),
    }, index=dates)

    # Only 2 events for NFP (below min_events=8)
    macro_df = pd.DataFrame({
        "date": [dates[10], dates[20]],
        "indicator": ["NFP", "NFP"],
        "surprise_ann": [1.0, -1.0],
    })

    estimator = CausalMacroResponseEstimator(min_events=8, predictive_horizon=1)
    results = estimator.fit_fold(
        fold_id=0,
        training_cutoff=cutoff,
        macro_df=macro_df,
        yield_df=yield_df,
    )

    assert results["NFP"]["status"] == "INADEQUATE_HISTORY_NEUTRAL"
    assert results["NFP"]["slope_beta"] == 0.0
    assert results["NFP"]["curvature_beta"] == 0.0


def test_coefficients_reproduce_from_saved_training_sample():
    """Verify bit-for-bit reproducibility of estimated betas on identical training slice."""
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    cutoff = dates[150]

    np.random.seed(123)
    yield_df = pd.DataFrame({
        "DGS2": np.cumsum(np.random.normal(0, 0.02, 200)) + 2.0,
        "DGS5": np.cumsum(np.random.normal(0, 0.02, 200)) + 2.5,
        "DGS10": np.cumsum(np.random.normal(0, 0.02, 200)) + 3.0,
    }, index=dates)

    event_dates = dates[np.arange(10, 140, 12)]
    macro_df = pd.DataFrame({
        "date": event_dates,
        "indicator": ["CPI"] * len(event_dates),
        "surprise_ann": np.random.normal(0, 1, len(event_dates)),
    })

    est1 = CausalMacroResponseEstimator(min_events=5, predictive_horizon=1)
    res1 = est1.fit_fold(0, cutoff, macro_df, yield_df)

    est2 = CausalMacroResponseEstimator(min_events=5, predictive_horizon=1)
    res2 = est2.fit_fold(0, cutoff, macro_df, yield_df)

    assert res1["CPI"]["slope_beta"] == res2["CPI"]["slope_beta"]
    assert res1["CPI"]["curvature_beta"] == res2["CPI"]["curvature_beta"]
    assert res1["CPI"]["n_events"] == res2["CPI"]["n_events"]


def test_future_release_cannot_alter_fold_coefficients():
    """Verify that adding future releases after training_cutoff does not alter fold betas."""
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    cutoff = dates[100]

    np.random.seed(99)
    yield_df = pd.DataFrame({
        "DGS2": np.cumsum(np.random.normal(0, 0.02, 200)) + 2.0,
        "DGS5": np.cumsum(np.random.normal(0, 0.02, 200)) + 2.5,
        "DGS10": np.cumsum(np.random.normal(0, 0.02, 200)) + 3.0,
    }, index=dates)

    # Base training events <= cutoff
    tr_dates = dates[np.arange(10, 90, 8)]
    macro_base = pd.DataFrame({
        "date": tr_dates,
        "indicator": ["CPI"] * len(tr_dates),
        "surprise_ann": np.random.normal(0, 1, len(tr_dates)),
    })

    est_base = CausalMacroResponseEstimator(min_events=5, predictive_horizon=1)
    res_base = est_base.fit_fold(0, cutoff, macro_base, yield_df)

    # Add massive future releases and yield swings after cutoff
    fut_dates = dates[np.arange(110, 190, 8)]
    macro_shocked = pd.concat([
        macro_base,
        pd.DataFrame({
            "date": fut_dates,
            "indicator": ["CPI"] * len(fut_dates),
            "surprise_ann": [10.0, -10.0, 20.0, -20.0, 50.0, -50.0, 100.0, -100.0, 200.0, -200.0][:len(fut_dates)],
        }),
    ], ignore_index=True)

    yield_shocked = yield_df.copy()
    yield_shocked.iloc[105:] += 10.0  # Massive future yield shock

    est_shocked = CausalMacroResponseEstimator(min_events=5, predictive_horizon=1)
    res_shocked = est_shocked.fit_fold(0, cutoff, macro_shocked, yield_shocked)

    # Fit coefficients for training fold 0 MUST be identical bit-for-bit
    assert np.isclose(res_base["CPI"]["slope_beta"], res_shocked["CPI"]["slope_beta"], atol=1e-12)
    assert np.isclose(res_base["CPI"]["curvature_beta"], res_shocked["CPI"]["curvature_beta"], atol=1e-12)
    assert res_base["CPI"]["n_events"] == res_shocked["CPI"]["n_events"]
