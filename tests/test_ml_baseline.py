"""
Unit tests for Milestone 14: Gradient-Boosted Model (LightGBM/GBM) Factor Forecaster & SHAP Attribution.

Covers:
- Leakage-safe feature engineering (lagged factors, both surprise types, ex-ante regime only).
- Strict exclusion of fed_regime_ex_post.
- Multi-output factor increments forecasting (dLevel, dSlope, dCurvature).
- Yield curve reconstruction via Nelson-Siegel loadings.
- TreeSHAP feature attribution and ranking.
- Integration into WalkForwardHarness as the 6th baseline model.
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


def test_feature_engineer_no_leakage():
    """Verify that feature engineering strictly obeys no-leakage rules."""
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet").iloc[:500]
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet")
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet")

    X, y, feature_cols = FactorFeatureEngineer.build_feature_panel(
        factor_df=factor_df,
        yield_df=yield_df,
        macro_df=macro_df,
    )

    # 1. Verification of strict regime purity:
    assert "fed_regime_ex_post" not in X.columns
    assert "regime_ex_ante_ZLB" in X.columns
    assert "regime_ex_ante_Hiking" in X.columns
    assert "regime_ex_ante_Easing" in X.columns
    assert "regime_ex_ante_Pause" in X.columns

    # 2. Both surprise types present and distinct:
    assert "surp_ann_CPI" in X.columns
    assert "surp_model_CPI" in X.columns
    assert "surp_ann_NFP" in X.columns
    assert "surp_model_NFP" in X.columns
    assert "surp_ann_FOMC" in X.columns
    assert "surp_model_FOMC" in X.columns

    # 3. Clean dimensions, fully populated features, and targets observed for all but latest origin
    assert len(X) == len(y)
    assert not X.isna().any().any()
    assert not y.iloc[:-1].isna().any().any()
    assert pd.isna(y.iloc[-1]["target_date"])
    assert {"target_dLevel", "target_dSlope", "target_dCurvature"}.issubset(y.columns)


def test_gbm_forecaster_fit_and_predict():
    """Verify GBM model training and 1-step-ahead factor forecasting."""
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet").iloc[:300]
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet")
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet")

    X, y, _ = FactorFeatureEngineer.build_feature_panel(factor_df, yield_df, macro_df)
    train_n = 200
    X_train, y_train = X.iloc[:train_n], y.iloc[:train_n]
    X_test, y_test = X.iloc[train_n:], y.iloc[train_n:]

    cfg = GBMForecasterConfig(n_estimators=20, learning_rate=0.05, max_depth=3)
    forecaster = GradientBoostedFactorForecaster(config=cfg)
    forecaster.fit(X_train, y_train)

    # Increments prediction
    increments = forecaster.predict_increments(X_test)
    assert len(increments) == len(X_test)
    assert list(increments.columns) == ["dLevel_hat", "dSlope_hat", "dCurvature_hat"]

    # Level forecasting
    curr_f = X_test[["level_t0", "slope_t0", "curvature_t0"]]
    f_hat = forecaster.forecast_factors(X_test, curr_f)
    assert len(f_hat) == len(X_test)
    assert list(f_hat.columns) == ["level_hat", "slope_hat", "curvature_hat"]

    # Consistency: F_hat = F_t + dF_hat
    np.testing.assert_allclose(
        f_hat["slope_hat"].values,
        (curr_f["slope_t0"] + increments["dSlope_hat"]).values,
        rtol=1e-5,
    )


def test_yield_curve_reconstruction():
    """Verify reconstruction of yield curve from predicted factors via Nelson-Siegel loadings."""
    maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
    pred_factors = pd.DataFrame({
        "level_hat": [4.5, 4.6],
        "slope_hat": [-1.0, -0.8],
        "curvature_hat": [2.0, 1.5],
    }, index=pd.date_range("2024-01-01", periods=2))

    curve_df = GradientBoostedFactorForecaster.reconstruct_yield_curve(
        predicted_factors=pred_factors,
        maturities=maturities,
        lambda_param=0.7308,
    )

    assert curve_df.shape == (2, len(maturities))
    # At very long maturity (30Y), yield approaches Level
    assert abs(curve_df.iloc[0, -1] - pred_factors.iloc[0]["level_hat"]) < 0.5


def test_shap_attribution():
    """Verify TreeSHAP explanation extraction across all 3 factor models."""
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet").iloc[:250]
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet")
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet")

    X, y, _ = FactorFeatureEngineer.build_feature_panel(factor_df, yield_df, macro_df)
    cfg = GBMForecasterConfig(n_estimators=15, learning_rate=0.05, max_depth=3)
    forecaster = GradientBoostedFactorForecaster(config=cfg).fit(X.iloc[:180], y.iloc[:180])

    shap_data = forecaster.explain_shap(X.iloc[180:], sample_size=30)
    assert shap_data["type"] in ["tree_shap", "gini"]

    if shap_data["type"] == "tree_shap":
        results = shap_data["results"]
        for target in ["target_dLevel", "target_dSlope", "target_dCurvature"]:
            assert target in results
            assert "mean_abs_shap" in results[target]
            top_features = results[target]["mean_abs_shap"]
            assert len(top_features) == len(X.columns)
            assert top_features.iloc[0] > 0.0


def test_walk_forward_with_gbm_integration():
    """Verify walk-forward harness executes with 6 models including GBM."""
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet").iloc[-850:]
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet").iloc[-850:]
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet")

    cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=42)
    harness = WalkForwardHarness(yield_df, factor_df, macro_df, config=cfg)
    eval_res = harness.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=2)

    table = eval_res["baseline_table"]
    assert len(table) == 6
    assert "GBM" in table.index
    assert table.loc["GBM", "OOS Curve RMSE (bp)"] > 0.0
    assert table.loc["GBM", "Factor Forecast RMSE (bp)"] > 0.0
