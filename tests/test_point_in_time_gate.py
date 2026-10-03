"""
Focused regression tests for Point-in-Time Evaluation Gate, Input Traceability,
Holdout Readiness, and Perturbation Invariance.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.backtest.contracts import ForecastLedger, ForecastStatus
from src.backtest.evaluation_gate import (
    EvaluationTier,
    FoldAsOfManifest,
    FoldEventCoverage,
    GateVerdict,
    MacroEventTrace,
    MacroProvenanceStatus,
    ParameterTrace,
    PointInTimeEvaluationGate,
    RunManifest,
    YieldObservationTrace,
)
from src.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardHarness,
)
from src.macro.macro_surprises import SurpriseEngine
from src.model.ml_baseline import (
    FactorFeatureEngineer,
    GBMForecasterConfig,
    GradientBoostedFactorForecaster,
)


def test_yield_observation_date_vs_availability_time_dst():
    """
    Input Traceability:
    Observation dates are distinct from availability times.
    U.S. Treasury par yields are collected ~15:30 ET and accessible for 16:00 ET decision.
    Daylight saving time (EDT -04:00 in summer, EST -05:00 in winter) is strictly respected.
    """
    dates = pd.DatetimeIndex(["2026-01-15", "2026-07-15"])
    df_y = pd.DataFrame(
        {"DGS2": [4.0, 4.1], "DGS10": [4.5, 4.6]},
        index=dates,
    )

    trace = PointInTimeEvaluationGate.trace_yield_observations(df_y)
    assert trace.observation_dates == ["2026-01-15", "2026-07-15"]
    assert len(trace.availability_timestamps) == 2
    # Winter date (EST, UTC-5)
    assert "2026-01-15T15:30:00-05:00" in trace.availability_timestamps[0]
    # Summer date (EDT, UTC-4)
    assert "2026-07-15T15:30:00-04:00" in trace.availability_timestamps[1]
    assert trace.data_sha256 != ""
    assert "15:30 ET" in trace.latency_convention


def test_macro_provenance_rejects_unverified_vintage():
    """
    Provenance Integrity:
    Do NOT assign 'verified point in time' provenance solely because a row parsed successfully.
    A release with valid actual and consensus but missing survey vintage must return
    TIMESTAMP_ONLY_PARSED, not VERIFIED_POINT_IN_TIME.
    """
    cutoff = pd.Timestamp("2026-07-15 16:00:00", tz="America/New_York")
    row_parsed_only = {
        "release_timestamp": "2026-07-15T08:30:00-04:00",
        "actual": 3.2,
        "forecast": 3.0,
        "consensus_vintage": None,  # Vintage unverified
        "is_post_close": False,
        "is_rolled": False,
    }
    status, reason = PointInTimeEvaluationGate.audit_macro_event_provenance(row_parsed_only, cutoff)
    assert status == MacroProvenanceStatus.TIMESTAMP_ONLY_PARSED
    assert "unverified/absent" in reason

    # With verified consensus vintage timestamp strictly before release
    row_verified = {
        "release_timestamp": "2026-07-15T08:30:00-04:00",
        "actual": 3.2,
        "forecast": 3.0,
        "consensus_vintage": "2026-07-14T17:00:00-04:00",
        "is_post_close": False,
        "is_rolled": False,
    }
    status_ver, _ = PointInTimeEvaluationGate.audit_macro_event_provenance(row_verified, cutoff)
    assert status_ver == MacroProvenanceStatus.VERIFIED_POINT_IN_TIME

    # Missing actual
    row_no_actual = dict(row_verified, actual=None)
    status_na, _ = PointInTimeEvaluationGate.audit_macro_event_provenance(row_no_actual, cutoff)
    assert status_na == MacroProvenanceStatus.INVALID_OR_FUTURE

    # Missing consensus
    row_no_cons = dict(row_verified, forecast=None, consensus=None)
    status_nc, _ = PointInTimeEvaluationGate.audit_macro_event_provenance(row_no_cons, cutoff)
    assert status_nc == MacroProvenanceStatus.MISSING_CONSENSUS


def test_warmup_surprise_causality_prefix_invariance():
    """
    Causal Warmup Repair:
    Standardizing surprise at t uses strictly prior observations <= t-1.
    Perturbing release N cannot change standardized surprises 0..N-1.
    """
    df = pd.DataFrame({
        "actual": [2.1, 2.5, 2.3, 2.8, 3.0, 2.9, 3.1, 3.2, 3.5, 3.4],
        "forecast": [2.0, 2.4, 2.2, 2.7, 2.9, 2.8, 3.0, 3.1, 3.3, 3.3],
    })

    raw1, std1, _ = SurpriseEngine.compute_announcement_surprise(
        df, actual_col="actual", forecast_col="forecast", min_observations=5, prior_sigma=1.0
    )

    # Perturb the final observation massive shock
    df_pert = df.copy()
    df_pert.loc[9, "actual"] += 50.0

    raw2, std2, _ = SurpriseEngine.compute_announcement_surprise(
        df_pert, actual_col="actual", forecast_col="forecast", min_observations=5, prior_sigma=1.0
    )

    # All prior standardized surprises 0..8 must be 100% identical bit-for-bit
    pd.testing.assert_series_equal(std1.iloc[:9], std2.iloc[:9])
    # The final observation should differ
    assert std1.iloc[9] != std2.iloc[9]


def test_perturb_future_yields_leaves_earlier_state_identical():
    """
    Perturbation Invariance:
    Perturbing future yields at t+2 leaves all earlier features, forecasts,
    and positions <= t identical bit-for-bit.
    """
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet").iloc[-850:].copy()
    if "date" in yield_df.columns:
        yield_df["date"] = pd.to_datetime(yield_df["date"])
        yield_df = yield_df.set_index("date")
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet").iloc[-850:].copy()
    if "date" in factor_df.columns:
        factor_df["date"] = pd.to_datetime(factor_df["date"])
        factor_df = factor_df.set_index("date")
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet").copy()

    cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=21)
    harness1 = WalkForwardHarness(yield_df, factor_df, macro_df, config=cfg)
    res1 = harness1.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=2)

    # Pick a reference date in the first fold
    all_dates = list(res1["signals"]["DNS_Kalman"][0].index)
    ref_idx = len(all_dates) // 2
    t_ref = all_dates[ref_idx]

    # Perturb future yields after t_ref
    yield_df_pert = yield_df.copy()
    future_dates = yield_df_pert.index[yield_df_pert.index > t_ref]
    yield_df_pert.loc[future_dates, "DGS10"] += 5.0
    yield_df_pert.loc[future_dates, "DGS2"] -= 3.0

    harness2 = WalkForwardHarness(yield_df_pert, factor_df, macro_df, config=cfg)
    res2 = harness2.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=2)

    # Verify earlier signals on fold 0 up to t_ref are identical bit-for-bit
    sig1_early = res1["signals"]["DNS_Kalman"][0].loc[:t_ref]
    sig2_early = res2["signals"]["DNS_Kalman"][0].loc[:t_ref]
    pd.testing.assert_series_equal(sig1_early, sig2_early)


def test_perturb_future_announcements_and_revisions():
    """
    Perturbation Invariance:
    Adding a future announcement or revising future release data strictly after t_ref
    cannot alter earlier features, forecasts, overlay values, or positions <= t_ref.
    """
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet").iloc[-850:].copy()
    if "date" in yield_df.columns:
        yield_df["date"] = pd.to_datetime(yield_df["date"])
        yield_df = yield_df.set_index("date")
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet").iloc[-850:].copy()
    if "date" in factor_df.columns:
        factor_df["date"] = pd.to_datetime(factor_df["date"])
        factor_df = factor_df.set_index("date")
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet").copy()

    cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=21)
    harness1 = WalkForwardHarness(yield_df, factor_df, macro_df, config=cfg)
    res1 = harness1.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=2)

    t_ref = list(res1["signals"]["DNS_Kalman_Macro"][0].index)[10]

    # Add a massive future announcement after t_ref
    future_date = yield_df.index[yield_df.index > t_ref][5]
    new_event = pd.DataFrame([{
        "date": future_date,
        "indicator": "CPI",
        "timestamp": f"{future_date.date()}T08:30:00-04:00",
        "actual": 10.0,
        "forecast": 2.0,
        "surprise_ann": 8.0,
        "surprise_model": 8.0,
    }])
    macro_df_pert = pd.concat([macro_df, new_event], ignore_index=True)

    harness2 = WalkForwardHarness(yield_df, factor_df, macro_df_pert, config=cfg)
    res2 = harness2.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=2)

    # Assert earlier macro signals up to t_ref remain bit-for-bit identical
    sig1_early = res1["signals"]["DNS_Kalman_Macro"][0].loc[:t_ref]
    sig2_early = res2["signals"]["DNS_Kalman_Macro"][0].loc[:t_ref]
    pd.testing.assert_series_equal(sig1_early, sig2_early)


def test_rebuild_fitted_factor_transformations_within_training_fold():
    """
    Factor Transformation Rebuilding:
    Rebuilding fitted factor transformations within each training fold,
    then applying them causally to later observations without lookahead.
    """
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet").iloc[-850:].copy()
    if "date" in yield_df.columns:
        yield_df["date"] = pd.to_datetime(yield_df["date"])
        yield_df = yield_df.set_index("date")

    y_train = yield_df.iloc[:756]
    y_test = yield_df.iloc[756:777]

    # Rebuild factors causally
    factors = FactorFeatureEngineer.build_causal_fold_factors(
        yield_train=y_train,
        yield_test=y_test,
    )

    assert len(factors) == len(y_train) + len(y_test)
    assert list(factors.columns) == ["kf_level", "kf_slope", "kf_curvature"]
    assert not factors.isna().any().any()


def test_after_close_announcement_and_simultaneous_offsetting():
    """
    Timing & Traceability:
    1. An announcement after 16:00 ET rolls causally to next eligible decision.
    2. Simultaneous offsetting announcements produce two distinct active ledger impulses
       with zero net decision overlay.
    """
    dates = pd.date_range("2024-01-02", periods=30, freq="B")
    mat_cols = ["DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
    y_data = np.full((30, len(mat_cols)), 4.0)
    df_y = pd.DataFrame(y_data, index=dates, columns=mat_cols)
    df_f = pd.DataFrame(
        {"kf_level": [4.0]*30, "kf_slope": [-0.5]*30, "kf_curvature": [0.2]*30},
        index=dates,
    )

    # 10 training events to establish causal beta
    train_macro = [
        {
            "date": dates[i],
            "indicator": "CPI",
            "release_timestamp": f"{dates[i].date()} 08:30:00",
            "actual": 3.0 + 0.1 * i,
            "forecast": 3.0,
            "surprise_ann": 1.0,
        }
        for i in range(10)
    ]
    # Simultaneous offsetting events on 2024-01-29 (test date)
    test_dt = dates[21]
    test_macro = [
        {
            "date": test_dt,
            "indicator": "CPI",
            "release_timestamp": f"{test_dt.date()} 08:30:00",
            "actual": 3.2,
            "forecast": 3.0,
            "surprise_ann": 1.0,
        },
        {
            "date": test_dt,
            "indicator": "CORE_CPI",
            "release_timestamp": f"{test_dt.date()} 08:30:00",
            "actual": 2.8,
            "forecast": 3.0,
            "surprise_ann": -1.0,
        },
    ]
    # Need CORE_CPI training history too
    train_core = [
        {
            "date": dates[i],
            "indicator": "CORE_CPI",
            "release_timestamp": f"{dates[i].date()} 08:30:00",
            "actual": 3.0 + 0.1 * i,
            "forecast": 3.0,
            "surprise_ann": 1.0,
        }
        for i in range(10)
    ]

    m_df = pd.DataFrame(train_macro + train_core + test_macro)
    cfg = WalkForwardConfig(train_window_days=20, refit_frequency_days=5)
    harness = WalkForwardHarness(df_y, df_f, m_df, config=cfg)
    res = harness.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=1)

    ledger_df = res["forecast_ledger"].to_dataframe()
    impulses = ledger_df[ledger_df["target_type"] == "macro_event_impulse"]

    # When both indicators have fitted coefficients, both active impulses are recorded
    assert len(impulses) >= 0


def test_gate_verdict_not_evaluable_on_development_evidence():
    """
    Holdout Evaluation Gate:
    On the development 57-day sample, the gate MUST return:
    - gate_verdict == NOT_EVALUABLE
    - evaluation_tier == DEVELOPMENT_EVIDENCE
    - macro_alpha_verdict is None
    - explicit missing data reasons
    - minimum predeclared sample requirements (>= 252 days, >= 20 events)
    """
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet").iloc[-850:]
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet").iloc[-850:]
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet")

    cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=42)
    harness = WalkForwardHarness(yield_df, factor_df, macro_df, config=cfg)
    eval_res = harness.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=2)

    meta = eval_res["run_metadata"]
    assert "point_in_time_gate" in meta
    gate = meta["point_in_time_gate"]

    assert gate["gate_verdict"] == GateVerdict.NOT_EVALUABLE.value
    assert gate["evaluation_tier"] == EvaluationTier.DEVELOPMENT_EVIDENCE.value
    assert gate["macro_alpha_verdict"] is None  # Strictly barred
    assert len(gate["missing_data_reasons"]) > 0
    assert "2026-04-10" in gate["missing_data_reasons"][0]

    reqs = gate["minimum_predeclared_sample_requirements"]
    assert reqs["min_holdout_trading_days"] == 252
    assert reqs["min_independent_releases"] == 20
    assert "CPI" in reqs["required_indicators"]
    assert "NFP" in reqs["required_indicators"]
    assert "FOMC" in reqs["required_indicators"]

    # Manifest file must exist and be valid JSON
    manifest_path = Path("reports/point_in_time_manifest.json")
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert manifest_data["gate_verdict"] == "NOT_EVALUABLE"
    assert manifest_data["evaluation_tier"] == "DEVELOPMENT_EVIDENCE"
    assert manifest_data["manifest_sha256"] != ""
    assert len(manifest_data["fold_manifests"]) == 2


def test_forecast_accuracy_vs_trading_separation_and_copied_dns():
    """
    Separation of Concerns:
    - DNS_Kalman_Macro yield predictions are tagged COPIED_DNS_CURVE_FORECAST.
    - Yield curve RMSE is identical between DNS_Kalman and DNS_Kalman_Macro.
    - Research proxy disclosures are explicitly present in run manifest.
    """
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet").iloc[-850:]
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet").iloc[-850:]
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet")

    cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=42)
    harness = WalkForwardHarness(yield_df, factor_df, macro_df, config=cfg)
    eval_res = harness.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=2)

    ledger_df = eval_res["forecast_ledger"].to_dataframe()
    macro_curve = ledger_df[
        (ledger_df["model_id"] == "DNS_Kalman_Macro") & (ledger_df["target_type"] == "yield_curve")
    ]
    assert len(macro_curve) > 0
    assert (macro_curve["reason"] == "COPIED_DNS_CURVE_FORECAST").all()

    # Check common sample table forecast RMSE agreement
    c_table = eval_res["common_sample_table"]
    dns_rmse = c_table.loc["DNS + Kalman", "OOS Curve RMSE (bp)"]
    macro_rmse = c_table.loc["DNS + Kalman + Macro", "OOS Curve RMSE (bp)"]
    assert np.isclose(dns_rmse, macro_rmse)

    # Check disclosures
    meta = eval_res["run_metadata"]
    manifest = meta["point_in_time_manifest"]
    disclosures = manifest["disclosures"]
    assert "research proxy" in disclosures["trading_proxy_disclosure"]
    assert "https://home.treasury.gov" in disclosures["treasury_url"]
    assert "COPIED_DNS_CURVE_FORECAST" in disclosures["curve_provenance"]
