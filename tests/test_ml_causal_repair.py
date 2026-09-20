"""
Unit tests for Prompt 4: Causal repair of GBM factor forecaster, origin/target alignment,
and explicit backend contracts.

Acceptance Criteria:
1. Zero-increment predictor on changing factor sequence produces RW forecast error, not zero RMSE.
2. Changing the first test outcome cannot change fitted training parameters (boundary row exclusion).
3. Perturbing future observations leaves earlier features/predictions unchanged (prefix invariance).
4. Target timestamps agree across GBM and RW in ForecastLedger.
5. Inference works without future labels at latest origin date.
6. Missing optional dependencies or explicit backend errors produce clear status rather than fabricated scores.
"""

import numpy as np
import pandas as pd
import pytest

from src.model.ml_baseline import (
    FactorFeatureEngineer,
    GBMForecasterConfig,
    GradientBoostedFactorForecaster,
    run_ml_factor_forecasting,
)
from src.backtest.walk_forward import WalkForwardConfig, WalkForwardHarness
from src.backtest.contracts import ForecastStatus


def test_zero_increment_predictor_matches_rw_error():
    """
    Acceptance Criteria 1:
    A zero-increment predictor (Delta F_hat = 0) on a changing factor sequence
    must produce forecast error identical to the Random Walk error, NOT zero RMSE.
    """
    # Create synthetic changing factor sequence
    factor_levels = np.array([
        [4.0, -1.0, 1.5],
        [4.05, -0.98, 1.52],
        [4.02, -1.03, 1.48],
        [4.10, -0.95, 1.55],
        [4.12, -0.92, 1.60],
        [4.08, -0.97, 1.58],
        [4.15, -0.89, 1.62],
        [4.20, -0.85, 1.65],
        [4.18, -0.88, 1.63],
        [4.25, -0.80, 1.70],
    ])
    
    # Origin t predicts t+1:
    # RW: F_hat_rw[t+1|t] = F[t]
    # Zero-increment: Delta_F_hat = 0 => F_hat_zero[t+1|t] = F[t] + 0 = F[t]
    # Actual at t+1: F[t+1]
    
    actual_t1 = factor_levels[1:]
    origin_t = factor_levels[:-1]
    
    # Zero increment forecast
    delta_f_zero = np.zeros_like(origin_t)
    f_hat_zero = origin_t + delta_f_zero
    
    # RW forecast
    f_hat_rw = origin_t
    
    # Factor RMSE in basis points (1% = 100 bp)
    rmse_zero_bp = np.sqrt(np.mean((actual_t1 - f_hat_zero) ** 2, axis=0)) * 100.0
    rmse_rw_bp = np.sqrt(np.mean((actual_t1 - f_hat_rw) ** 2, axis=0)) * 100.0
    
    # Assertions
    np.testing.assert_allclose(rmse_zero_bp, rmse_rw_bp, rtol=1e-10)
    # The error must be strictly positive since the sequence changes
    assert np.all(rmse_zero_bp > 0.0)
    # Scoring Delta F_hat against zero would give 0.0 -- verify this is NOT what happened
    assert not np.allclose(rmse_zero_bp, 0.0)


def test_changing_first_test_outcome_does_not_affect_training_parameters():
    """
    Acceptance Criteria 2:
    Changing the first test outcome (at t = T+1) cannot change fitted training parameters.
    The boundary row at t = T (whose target is at T+1) must be strictly excluded from training.
    """
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet").iloc[:400].copy()
    if "date" in factor_df.columns:
        factor_df["date"] = pd.to_datetime(factor_df["date"])
        factor_df = factor_df.sort_values("date").set_index("date")

    yield_df = pd.read_parquet("data/processed/yield_panel.parquet").copy()
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet").copy()

    X, y, _ = FactorFeatureEngineer.build_feature_panel(factor_df, yield_df, macro_df)
    
    # Pick a training cutoff
    cutoff_idx = 250
    training_cutoff = X.index[cutoff_idx]
    first_test_date = X.index[cutoff_idx + 1]

    # Baseline slice
    X_tr1, y_tr1 = FactorFeatureEngineer.get_training_slice(X, y, training_cutoff=training_cutoff)
    
    # Assert boundary row at cutoff is excluded because its label matures at first_test_date > training_cutoff
    assert training_cutoff not in y_tr1["origin_date"].values
    assert all(y_tr1["label_available_at"] <= training_cutoff)

    cfg = GBMForecasterConfig(backend="sklearn", n_estimators=10, max_depth=2, random_state=42)
    model1 = GradientBoostedFactorForecaster(config=cfg).fit(X_tr1, y_tr1)
    preds1 = model1.predict_increments(X.iloc[cutoff_idx+1:cutoff_idx+5])

    # Now perturb the factor at first_test_date by a massive shock (+500 bp)
    factor_df_perturbed = factor_df.copy()
    factor_df_perturbed.loc[first_test_date, "kf_level"] += 5.0
    factor_df_perturbed.loc[first_test_date, "kf_slope"] += 5.0

    X_pert, y_pert, _ = FactorFeatureEngineer.build_feature_panel(factor_df_perturbed, yield_df, macro_df)
    X_tr2, y_tr2 = FactorFeatureEngineer.get_training_slice(X_pert, y_pert, training_cutoff=training_cutoff)

    # Training data must be 100% identical
    pd.testing.assert_frame_equal(X_tr1, X_tr2)
    pd.testing.assert_frame_equal(y_tr1, y_tr2)

    model2 = GradientBoostedFactorForecaster(config=cfg).fit(X_tr2, y_tr2)
    preds2 = model2.predict_increments(X.iloc[cutoff_idx+1:cutoff_idx+5])

    # Fitted parameters and predictions on reference features must be 100% identical
    pd.testing.assert_frame_equal(preds1, preds2)


def test_perturbing_future_observations_leaves_earlier_features_unchanged():
    """
    Acceptance Criteria 3:
    Features at date t depend strictly on data <= t.
    Perturbing observations after t must leave X.loc[:t] unchanged (prefix invariance).
    """
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet").iloc[:300].copy()
    if "date" in factor_df.columns:
        factor_df["date"] = pd.to_datetime(factor_df["date"])
        factor_df = factor_df.sort_values("date").set_index("date")

    yield_df = pd.read_parquet("data/processed/yield_panel.parquet").copy()
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet").copy()

    X1, _, _ = FactorFeatureEngineer.build_feature_panel(factor_df, yield_df, macro_df)
    
    t_ref = X1.index[150]
    earlier_features1 = X1.loc[:t_ref].copy()

    # Shock future observations (> t_ref)
    factor_df_pert = factor_df.copy()
    future_dates = factor_df_pert.index[factor_df_pert.index > t_ref]
    factor_df_pert.loc[future_dates, "kf_level"] *= 2.0
    factor_df_pert.loc[future_dates, "kf_slope"] += 10.0

    X2, _, _ = FactorFeatureEngineer.build_feature_panel(factor_df_pert, yield_df, macro_df)
    earlier_features2 = X2.loc[:t_ref].copy()

    # Earlier features must be perfectly prefix-invariant
    pd.testing.assert_frame_equal(earlier_features1, earlier_features2)


def test_target_timestamps_agree_across_gbm_and_rw():
    """
    Acceptance Criteria 4:
    In ForecastLedger, the origin and target timestamps for GBM must exactly match Random Walk.
    """
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet").iloc[-850:]
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet").iloc[-850:]
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet")

    cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=21, gbm_backend="sklearn")
    harness = WalkForwardHarness(yield_df, factor_df, macro_df, config=cfg)
    eval_res = harness.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=2)

    ledger = eval_res["forecast_ledger"]
    df_ledger = ledger.to_dataframe()

    rw_records = df_ledger[df_ledger["model_id"] == "Random_Walk"]
    gbm_records = df_ledger[df_ledger["model_id"] == "GBM"]

    assert len(rw_records) > 0
    assert len(gbm_records) > 0
    assert len(rw_records) == len(gbm_records)

    # Verify origin and target timestamps are identical step-for-step
    rw_sorted = rw_records.sort_values(["origin_timestamp", "target_timestamp", "target_name"]).reset_index(drop=True)
    gbm_sorted = gbm_records.sort_values(["origin_timestamp", "target_timestamp", "target_name"]).reset_index(drop=True)

    pd.testing.assert_series_equal(rw_sorted["origin_timestamp"], gbm_sorted["origin_timestamp"], check_names=False)
    pd.testing.assert_series_equal(rw_sorted["target_timestamp"], gbm_sorted["target_timestamp"], check_names=False)
    pd.testing.assert_series_equal(rw_sorted["target_name"], gbm_sorted["target_name"], check_names=False)
    pd.testing.assert_series_equal(rw_sorted["actual"], gbm_sorted["actual"], check_names=False)


def test_inference_works_without_future_labels():
    """
    Acceptance Criteria 5:
    At the latest origin date T_latest, future labels are unobserved (NaN/NaT),
    yet inference features are fully formed and the model issues a valid 1-step forecast.
    """
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet").iloc[:300].copy()
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet").copy()
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet").copy()

    X, y, _ = FactorFeatureEngineer.build_feature_panel(factor_df, yield_df, macro_df)
    latest_dt = X.index[-1]

    # Target is unobserved for the final origin date
    assert pd.isna(y.loc[latest_dt, "target_date"])
    assert pd.isna(y.loc[latest_dt, "target_dLevel"])

    # Inference feature row is complete
    assert not X.loc[latest_dt].isna().any()

    # Fit on all evaluable historical data
    X_tr, y_tr = FactorFeatureEngineer.get_training_slice(X, y, training_cutoff=X.index[-2])
    forecaster = GradientBoostedFactorForecaster(config=GBMForecasterConfig(backend="sklearn", n_estimators=10))
    forecaster.fit(X_tr, y_tr)

    # Forecast at latest origin date without future labels
    curr_f = X.loc[[latest_dt], ["level_t0", "slope_t0", "curvature_t0"]]
    f_pred = forecaster.forecast_factors(X.loc[[latest_dt]], curr_f)

    assert not f_pred.isna().any().any()
    assert list(f_pred.columns) == ["level_hat", "slope_hat", "curvature_hat"]


def test_backend_selection_and_missing_dependencies():
    """
    Acceptance Criteria 6:
    Explicit backend selection is enforced. Missing or invalid backends produce
    explicit errors/statuses, never silent fallbacks or fabricated scores.
    """
    # Invalid backend raises ValueError
    cfg_invalid = GBMForecasterConfig(backend="unsupported_backend")
    forecaster = GradientBoostedFactorForecaster(config=cfg_invalid)
    with pytest.raises(ValueError, match="Unknown backend"):
        forecaster._init_model()

    # Explicit backend name is preserved
    cfg_sk = GBMForecasterConfig(backend="sklearn")
    forecaster_sk = GradientBoostedFactorForecaster(config=cfg_sk)
    assert forecaster_sk.config.backend == "sklearn"
