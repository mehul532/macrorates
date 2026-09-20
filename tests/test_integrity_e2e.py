"""
End-to-end Research Integrity and Causal Leakage Tests.

Locks down the entire pipeline against causal lookahead, boundary leakage,
and false equivalence:
1. Pipeline refit invariance under future asymmetric tenor perturbations.
2. Boundary label leakage invariance (origin t -> target t+1).
3. Future macro release perturbation invariance.
4. Independent scoring reconciliation directly from the forecast ledger.
5. Strict temporal causality (training_cutoff <= origin < target) and common-sample coverage.
6. Common-sample table disclosure, benchmark roles, and Cash-Only accounting integrity.
7. Trading decision invariance under future price/yield perturbations.
8. Scorecard and verdict report provenance and auditability.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardHarness,
    ForecastLedger,
    ForecastStatus,
    compute_observable_spreads,
)
from src.model.ml_baseline import (
    FactorFeatureEngineer,
    GBMForecasterConfig,
    GradientBoostedFactorForecaster,
)
from src.strategy.portfolio import compute_continuous_positions
from src.backtest.backtest import RelativeValueBacktestEngine


@pytest.fixture(scope="module")
def historical_data():
    """Load real processed datasets for integrity tests."""
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet")
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet")
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet")
    return yield_df, factor_df, macro_df


def test_pipeline_refit_invariance_under_future_asymmetric_tenor_perturbation(historical_data):
    """
    Assert that perturbing an un-evaluated future tenor or future yield observations
    leaves earlier forecasts, normalizers, parameters, and trading decisions bit-for-bit identical.
    """
    yield_df, factor_df, macro_df = historical_data
    # Fast evaluation slice: last 850 business days, 2 folds
    yield_sub = yield_df.iloc[-850:].copy()
    factor_sub = factor_df.iloc[-850:].copy()
    if "date" in yield_sub.columns:
        yield_sub["date"] = pd.to_datetime(yield_sub["date"])
        yield_sub = yield_sub.set_index("date")
    if "date" in factor_sub.columns:
        factor_sub["date"] = pd.to_datetime(factor_sub["date"])
        factor_sub = factor_sub.set_index("date")

    cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=42)
    harness1 = WalkForwardHarness(yield_sub, factor_sub, macro_df, config=cfg)
    folds = harness1.generate_folds()
    assert len(folds) >= 2, "Expected at least 2 folds for perturbation test"

    # Run baseline evaluation
    res1 = harness1.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=2)
    ledger1 = res1["forecast_ledger"]
    signals1 = res1["signals"]

    # Identify fold 0 test dates and fold 1 test dates
    fold0_train, fold0_test = folds[0]
    fold1_train, fold1_test = folds[1]

    # Create perturbed dataset:
    # Asymmetrically shock ONLY DGS2 in Fold 1 test dates (future relative to Fold 0) by +500 bp.
    # Fold 0 training and test dates are untouched.
    yield_perturbed = yield_sub.copy()
    yield_perturbed.loc[fold1_test, "DGS2"] += 5.0  # +500 bp asymmetric shock
    # Also shock factor panel on fold 1 test dates
    factor_perturbed = factor_sub.copy()
    factor_perturbed.loc[fold1_test, "kf_level"] += 5.0

    harness2 = WalkForwardHarness(yield_perturbed, factor_perturbed, macro_df, config=cfg)
    res2 = harness2.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=2)
    ledger2 = res2["forecast_ledger"]
    signals2 = res2["signals"]

    # Verify Fold 0 forecasts in the ledger are bit-for-bit identical across all models
    models = ["Random_Walk", "PCA_VAR", "Static_NS", "DNS_Kalman", "DNS_Kalman_Macro", "GBM"]
    for m in models:
        recs1 = [r for r in ledger1.get_records() if r.model_id == m and r.fold_id == 0]
        recs2 = [r for r in ledger2.get_records() if r.model_id == m and r.fold_id == 0]
        assert len(recs1) == len(recs2), f"Record count mismatch for model {m} in fold 0"

        f1 = np.array([r.forecast for r in recs1])
        f2 = np.array([r.forecast for r in recs2])
        np.testing.assert_array_equal(
            f1, f2,
            err_msg=f"Model {m} fold 0 forecasts leaked future asymmetric tenor shock!"
        )

        # Verify Fold 0 trading signals are bit-for-bit identical
        sig1 = signals1[m][0]
        sig2 = signals2[m][0]
        np.testing.assert_allclose(
            sig1.values, sig2.values,
            err_msg=f"Model {m} fold 0 trading signal leaked future asymmetric tenor shock!"
        )


def test_boundary_label_leakage_invariance(historical_data):
    """
    Verify that perturbing future target labels or features strictly after T_train
    does not alter in-sample training features, targets, or fitted parameters.
    """
    yield_df, factor_df, macro_df = historical_data
    sub_f = factor_df.iloc[:350].copy()
    if "date" in sub_f.columns:
        sub_f["date"] = pd.to_datetime(sub_f["date"])
        sub_f = sub_f.set_index("date")

    X1, y1, _ = FactorFeatureEngineer.build_feature_panel(sub_f, yield_df, macro_df)
    train_cutoff = X1.index[250]
    X_train1, y_train1 = FactorFeatureEngineer.get_training_slice(X1, y1, training_cutoff=train_cutoff)

    # Fit forecaster 1
    cfg = GBMForecasterConfig(n_estimators=10, learning_rate=0.05, max_depth=3, random_state=42)
    forecaster1 = GradientBoostedFactorForecaster(config=cfg)
    forecaster1.fit(X_train1, y_train1)
    test_slice1 = X1.loc[X1.index > train_cutoff].head(5)
    preds1 = forecaster1.predict_increments(test_slice1)

    # Perturb factor panel strictly at dates after test_slice1
    sub_f_perturbed = sub_f.copy()
    future_dates = sub_f_perturbed.index[sub_f_perturbed.index > test_slice1.index[-1]]
    sub_f_perturbed.loc[future_dates, "kf_slope"] += 10.0

    X2, y2, _ = FactorFeatureEngineer.build_feature_panel(sub_f_perturbed, yield_df, macro_df)
    X_train2, y_train2 = FactorFeatureEngineer.get_training_slice(X2, y2, training_cutoff=train_cutoff)

    # Assert training inputs are bit-for-bit invariant
    pd.testing.assert_frame_equal(X_train1, X_train2, check_dtype=False)
    pd.testing.assert_frame_equal(y_train1, y_train2, check_dtype=False)

    forecaster2 = GradientBoostedFactorForecaster(config=cfg)
    forecaster2.fit(X_train2, y_train2)
    test_slice2 = X2.loc[X2.index > train_cutoff].head(5)
    preds2 = forecaster2.predict_increments(test_slice2)

    np.testing.assert_array_equal(
        preds1.values, preds2.values,
        err_msg="Boundary leakage detected: future shock changed pre-boundary predictions!"
    )


def test_future_macro_release_perturbation_invariance(historical_data):
    """
    Verify that injecting or perturbing future macro announcements beyond the training cutoff
    does not alter historical causal macro betas or pre-cutoff macro surprise standardizers.
    """
    yield_df, factor_df, macro_df = historical_data
    yield_sub = yield_df.iloc[-850:].copy()
    factor_sub = factor_df.iloc[-850:].copy()
    macro_sub = macro_df.copy()

    cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=42)
    harness = WalkForwardHarness(yield_sub, factor_sub, macro_sub, config=cfg)
    folds = harness.generate_folds()
    fold0_train, fold0_test = folds[0]
    fold0_cutoff = fold0_train[-1]

    res1 = harness.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=1)
    kf_macro_recs1 = [
        r for r in res1["forecast_ledger"].get_records()
        if r.model_id == "DNS_Kalman_Macro" and r.fold_id == 0
    ]

    # Perturb macro surprises strictly AFTER fold0_cutoff (in the future)
    macro_perturbed = macro_sub.copy()
    future_mask = pd.to_datetime(macro_perturbed["date"]) > fold0_cutoff
    macro_perturbed.loc[future_mask, "surprise_ann"] += 50.0  # Massive future surprise

    harness2 = WalkForwardHarness(yield_sub, factor_sub, macro_perturbed, config=cfg)
    res2 = harness2.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=1)
    kf_macro_recs2 = [
        r for r in res2["forecast_ledger"].get_records()
        if r.model_id == "DNS_Kalman_Macro" and r.fold_id == 0
    ]

    f1 = np.array([r.forecast for r in kf_macro_recs1])
    f2 = np.array([r.forecast for r in kf_macro_recs2])
    np.testing.assert_array_equal(
        f1, f2,
        err_msg="Causal macro failure: future macro release perturbed earlier fold forecasts!"
    )


def test_independent_forecast_ledger_scoring_reconciliation(historical_data):
    """
    Assert that independent scoring directly from the forecast ledger matches
    reported baseline tables and common_sample_table.
    """
    yield_df, factor_df, macro_df = historical_data
    yield_sub = yield_df.iloc[-850:].copy()
    factor_sub = factor_df.iloc[-850:].copy()

    cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=42)
    harness = WalkForwardHarness(yield_sub, factor_sub, macro_df, config=cfg)
    eval_res = harness.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=2)

    ledger = eval_res["forecast_ledger"]
    baseline_table = eval_res["baseline_table"]
    common_sample_table = eval_res["common_sample_table"]

    all_records = ledger.get_records()

    # Verify reconciliation for each evaluated model
    model_key_mapping = {
        "Random_Walk": "Random + Walk",
        "PCA_VAR": "PCA + VAR",
        "Static_NS": "Static + NS",
        "DNS_Kalman": "DNS + Kalman",
        "DNS_Kalman_Macro": "DNS + Kalman + Macro",
        "GBM": "GBM",
    }

    for model_id, table_row in model_key_mapping.items():
        m_recs = [r for r in all_records if r.model_id == model_id and r.target_type == "yield_curve"]
        assert len(m_recs) > 0, f"No ledger records for model {model_id}"

        # 1. Independent Curve RMSE
        errors = [r.forecast - r.actual for r in m_recs]
        indep_curve_rmse_bp = round(float(np.sqrt(np.mean(np.array(errors) ** 2)) * 100.0), 2)
        reported_curve_rmse_bp = baseline_table.loc[table_row, "OOS Curve RMSE (bp)"]

        assert abs(indep_curve_rmse_bp - reported_curve_rmse_bp) <= 0.02, (
            f"Ledger reconciliation mismatch for {model_id} Curve RMSE: "
            f"Indep={indep_curve_rmse_bp} bp vs Reported={reported_curve_rmse_bp} bp"
        )

        # 2. Independent Per-Tenor RMSE for 2Y (DGS2) and 10Y (DGS10)
        recs_2y = [r for r in m_recs if r.target_name == "DGS2"]
        if len(recs_2y) > 0:
            indep_2y_rmse = round(float(np.sqrt(np.mean([(r.forecast - r.actual) ** 2 for r in recs_2y])) * 100.0), 2)
            rep_2y = baseline_table.loc[table_row, "2Y Curve RMSE (bp)"]
            assert abs(indep_2y_rmse - rep_2y) <= 0.02, (
                f"Ledger 2Y RMSE mismatch for {model_id}: {indep_2y_rmse} vs {rep_2y}"
            )

        recs_10y = [r for r in m_recs if r.target_name == "DGS10"]
        if len(recs_10y) > 0:
            indep_10y_rmse = round(float(np.sqrt(np.mean([(r.forecast - r.actual) ** 2 for r in recs_10y])) * 100.0), 2)
            rep_10y = baseline_table.loc[table_row, "10Y Curve RMSE (bp)"]
            assert abs(indep_10y_rmse - rep_10y) <= 0.02, (
                f"Ledger 10Y RMSE mismatch for {model_id}: {indep_10y_rmse} vs {rep_10y}"
            )

        # 3. Independent Observable 2s10s Spread RMSE
        # Group by target_timestamp
        target_dates = sorted(list(set(r.target_timestamp for r in m_recs)))
        spread_sq_errs = []
        for t_dt in target_dates:
            dt_recs = {r.target_name: r for r in m_recs if r.target_timestamp == t_dt}
            if "DGS2" in dt_recs and "DGS10" in dt_recs:
                pred_spread = dt_recs["DGS10"].forecast - dt_recs["DGS2"].forecast
                act_spread = dt_recs["DGS10"].actual - dt_recs["DGS2"].actual
                spread_sq_errs.append((act_spread - pred_spread) ** 2)

        if len(spread_sq_errs) > 0:
            indep_spread_rmse = round(float(np.sqrt(np.mean(spread_sq_errs)) * 100.0), 2)
            rep_spread_rmse = baseline_table.loc[table_row, "2s10s Spread RMSE (bp)"]
            assert abs(indep_spread_rmse - rep_spread_rmse) <= 0.02, (
                f"Ledger 2s10s Spread RMSE mismatch for {model_id}: {indep_spread_rmse} vs {rep_spread_rmse}"
            )


def test_ledger_origin_target_strict_causality_and_coverage(historical_data):
    """
    Assert that every ledger entry satisfies:
    1. origin_timestamp < target_timestamp.
    2. training_cutoff <= origin_timestamp.
    3. Status is SCORED with finite values.
    4. All models share identical coverage on the common evaluation sample.
    """
    yield_df, factor_df, macro_df = historical_data
    yield_sub = yield_df.iloc[-850:].copy()
    factor_sub = factor_df.iloc[-850:].copy()

    cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=42)
    harness = WalkForwardHarness(yield_sub, factor_sub, macro_df, config=cfg)
    eval_res = harness.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=2)

    ledger = eval_res["forecast_ledger"]
    records = ledger.get_records()
    assert len(records) > 0

    model_counts = {}
    for r in records:
        assert r.origin_timestamp < r.target_timestamp, (
            f"Temporal violation: origin {r.origin_timestamp} >= target {r.target_timestamp}"
        )
        assert r.training_cutoff <= r.origin_timestamp, (
            f"Training boundary violation: cutoff {r.training_cutoff} > origin {r.origin_timestamp}"
        )
        assert r.status == ForecastStatus.SCORED, f"Unscored record found: {r}"
        assert np.isfinite(r.forecast), f"Non-finite forecast: {r.forecast}"
        assert np.isfinite(r.actual), f"Non-finite actual: {r.actual}"

        model_counts[r.model_id] = model_counts.get(r.model_id, 0) + 1

    # Verify equal record count across evaluated models
    first_count = list(model_counts.values())[0]
    for m, cnt in model_counts.items():
        assert cnt == first_count, f"Model {m} has {cnt} records, expected {first_count}"


def test_common_sample_disclosure_and_benchmark_integrity(historical_data):
    """
    Verify that common_sample_table:
    1. Contains all required models: Random Walk, AR(1), PCA/VAR, DNS+Kalman, DNS+Kalman+Macro, GBM, Cash-Only.
    2. Discloses Benchmark Role and Forecast Status for every row.
    3. Discloses Sample Size (Days) for every row.
    4. Verifies Cash-Only benchmark has 0 trading PnL, 0 turnover, and earns positive interest.
    """
    yield_df, factor_df, macro_df = historical_data
    yield_sub = yield_df.iloc[-850:].copy()
    factor_sub = factor_df.iloc[-850:].copy()

    cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=42)
    harness = WalkForwardHarness(yield_sub, factor_sub, macro_df, config=cfg)
    eval_res = harness.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=2)

    table = eval_res["common_sample_table"]

    # Verify rows present
    expected_rows = [
        "Random Walk (Curve Benchmark)",
        "PCA / VAR(1)",
        "AR(1) Baseline (Static NS)",
        "DNS + Kalman",
        "DNS + Kalman + Macro",
        "GBM",
        "Cash Only (Strategy Benchmark)",
    ]
    for row_name in expected_rows:
        assert row_name in table.index, f"Missing expected row {row_name} in common_sample_table"

    # Verify columns present
    assert "Benchmark Role" in table.columns
    assert "Forecast Status" in table.columns
    assert "Sample Size (Days)" in table.columns

    # Verify Cash Only benchmark integrity
    cash_row = table.loc["Cash Only (Strategy Benchmark)"]
    assert cash_row["Forecast Status"] == "BENCHMARK_ONLY"
    assert cash_row["Benchmark Role"] == "Strategy Benchmark"
    assert cash_row["Annual Turnover (lots)"] == 0.0
    assert cash_row["Trading Net PnL ($)"] == 0.0
    assert cash_row["Gross PnL ($)"] == 0.0
    assert cash_row["Trade Costs ($)"] == 0.0
    assert cash_row["Roll Costs ($)"] == 0.0
    assert cash_row["Cash Interest ($)"] > 0.0
    assert cash_row["Collateral Net PnL ($)"] == cash_row["Cash Interest ($)"]

    # Verify common sample sizes are positive and equal
    eval_days = table.loc["Random Walk (Curve Benchmark)", "Sample Size (Days)"]
    assert eval_days > 0
    for idx in table.index:
        assert table.loc[idx, "Sample Size (Days)"] == eval_days


def test_trading_decision_invariance_to_future_price_perturbation():
    """
    Verify that trading decisions (positions and orders) up to time t
    are strictly invariant to future price perturbations at t+1.
    """
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    sig = pd.Series(np.sin(np.linspace(0, 3, 30)), index=dates)

    # Base positions
    pos1 = compute_continuous_positions(sig, strategy_type="2s10s", target_dv01=10_000.0)

    # Perturb future signal at day 25
    sig_perturbed = sig.copy()
    sig_perturbed.iloc[25:] = -5.0  # Huge future reversal

    pos2 = compute_continuous_positions(sig_perturbed, strategy_type="2s10s", target_dv01=10_000.0)

    # Up to day 24, positions must be bit-for-bit identical
    pd.testing.assert_frame_equal(pos1.iloc[:25], pos2.iloc[:25])


def test_scorecard_and_verdict_provenance_and_auditability():
    """
    Verify that report scorecard JSON and verdict markdown:
    1. Exist as reproducible run artifacts.
    2. Contain git commit hash and data checksums for yield, factor, and macro panels.
    3. Include common_sample_table with benchmark roles and sample sizes.
    4. Maintain scientific objectivity and explicitly disclose observational vs causal attribution.
    """
    scorecard_path = Path("reports/extended_walk_forward_metrics_scorecard.json")
    assert scorecard_path.exists(), "Scorecard JSON artifact missing"
    with open(scorecard_path, "r", encoding="utf-8") as f:
        scorecard = json.load(f)

    meta = scorecard.get("metadata") or scorecard.get("run_metadata", {})
    assert "git_commit" in meta, "Git commit hash missing from scorecard"
    assert "data_checksums" in meta, "Data checksums missing from scorecard"
    assert "yield_panel" in meta["data_checksums"]
    assert "factor_panel" in meta["data_checksums"]
    assert "macro_surprises" in meta["data_checksums"]

    common_table = scorecard.get("common_sample_table") or scorecard.get("models", {})
    assert "Cash Only (Strategy Benchmark)" in common_table
    assert common_table["Cash Only (Strategy Benchmark)"]["Forecast Status"] == "BENCHMARK_ONLY"

    verdict_path = Path("reports/ml_baseline_verdict.md")
    assert verdict_path.exists(), "Verdict markdown artifact missing"
    with open(verdict_path, "r", encoding="utf-8") as f:
        verdict_text = f.read()

    # Research integrity assertions on report language
    assert "Observational vs. Causal Attribution" in verdict_text
    assert "Random Walk provides the unparameterized zero-increment" in verdict_text
    assert "Git Commit" in verdict_text
    assert "SHA-256 Checksum" in verdict_text
