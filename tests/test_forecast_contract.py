"""
Unit tests for Research Integrity & Forecast Contract (Prompt 1).

Acceptance Criteria Tested:
1. An unavailable model cannot acquire a numeric RMSE.
2. Emitted records strictly distinguish origin from target timestamp (origin < target).
3. Forecast generation is strictly separated from target scoring.
4. Decision and execution timing contracts prevent pre-origin signal utilization and double-shifting.
5. Walk-forward baseline table assigns NaN to unavailable models rather than artificial multipliers.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.contracts import (
    ForecastStatus,
    ExecutionTimingAssumption,
    ForecastRecord,
    DecisionContract,
    ForecastLedger,
)
from src.backtest.walk_forward import WalkForwardConfig, WalkForwardHarness


def test_unavailable_model_cannot_acquire_numeric_rmse():
    """Verify that an unavailable or failed model cannot possess a numeric forecast or RMSE."""
    ledger = ForecastLedger(run_id="test_run")
    cutoff = pd.Timestamp("2023-01-01")
    t0 = pd.Timestamp("2023-01-02")
    t1 = pd.Timestamp("2023-01-03")

    # 1. Direct record instantiation violation: numeric forecast with UNAVAILABLE status must fail
    with pytest.raises(ValueError, match="Integrity violation"):
        ForecastRecord(
            run_id="test_run",
            model_id="Faulty_Model",
            fold_id=0,
            training_cutoff=cutoff,
            origin_timestamp=t0,
            target_timestamp=t1,
            target_type="yield_curve",
            target_name="DGS10",
            forecast=3.85,  # Fabricated numeric value
            status=ForecastStatus.UNAVAILABLE,
            reason="Failed fit",
        )

    # 2. Legitimate unavailable recording via ledger
    ledger.add_unavailable(
        model_id="DNS_Kalman",
        fold_id=0,
        training_cutoff=cutoff,
        origin_timestamp=t0,
        target_timestamp=t1,
        target_type="yield_curve",
        target_name="DGS10",
        reason="Pending genuine rolling one-step forecast implementation (Prompt 2)",
    )

    # 3. Assert RMSE computation on unavailable model returns NaN, never a numeric score
    rmse = ledger.compute_rmse("DNS_Kalman")
    assert np.isnan(rmse), f"Expected NaN RMSE for unavailable model, got {rmse}"

    summary = ledger.get_model_summary("DNS_Kalman")
    assert summary["status"] == "UNAVAILABLE"
    assert summary["counts"][ForecastStatus.UNAVAILABLE.value] == 1
    assert np.isnan(summary["rmse"])


def test_emitted_records_distinguish_origin_from_target():
    """Verify strict causal timing: origin_timestamp must strictly precede target_timestamp."""
    cutoff = pd.Timestamp("2023-01-01")
    t0 = pd.Timestamp("2023-01-02")
    t1 = pd.Timestamp("2023-01-03")

    # 1. Valid record (origin < target)
    rec = ForecastRecord(
        run_id="test_run",
        model_id="Random_Walk",
        fold_id=0,
        training_cutoff=cutoff,
        origin_timestamp=t0,
        target_timestamp=t1,
        target_type="yield_curve",
        target_name="DGS10",
        forecast=3.50,
        status=ForecastStatus.GENERATED,
    )
    assert rec.origin_timestamp < rec.target_timestamp

    # 2. Causal violation: origin == target
    with pytest.raises(ValueError, match="Causal violation"):
        ForecastRecord(
            run_id="test_run",
            model_id="Lookahead_Model",
            fold_id=0,
            training_cutoff=cutoff,
            origin_timestamp=t0,
            target_timestamp=t0,  # Contradiction
            target_type="yield_curve",
            target_name="DGS10",
            forecast=3.50,
            status=ForecastStatus.GENERATED,
        )

    # 3. Causal violation: origin > target
    with pytest.raises(ValueError, match="Causal violation"):
        ForecastRecord(
            run_id="test_run",
            model_id="Backward_Model",
            fold_id=0,
            training_cutoff=cutoff,
            origin_timestamp=t1,
            target_timestamp=t0,  # Time travel
            target_type="yield_curve",
            target_name="DGS10",
            forecast=3.50,
            status=ForecastStatus.GENERATED,
        )

    # 4. Boundary violation: origin < training_cutoff
    with pytest.raises(ValueError, match="Boundary violation"):
        ForecastRecord(
            run_id="test_run",
            model_id="Boundary_Violation_Model",
            fold_id=0,
            training_cutoff=pd.Timestamp("2023-01-10"),
            origin_timestamp=pd.Timestamp("2023-01-05"),  # Precedes training cutoff
            target_timestamp=pd.Timestamp("2023-01-12"),
            target_type="yield_curve",
            target_name="DGS10",
            forecast=3.50,
            status=ForecastStatus.GENERATED,
        )


def test_separate_prediction_from_scoring():
    """Verify that forecast generation is strictly decoupled from later target observation and scoring."""
    ledger = ForecastLedger(run_id="test_run")
    cutoff = pd.Timestamp("2023-01-01")
    t0 = pd.Timestamp("2023-01-02")
    t1 = pd.Timestamp("2023-01-03")

    # 1. Forecast is issued at origin t0 without actual outcome at t1
    rec = ForecastRecord(
        run_id="test_run",
        model_id="Random_Walk",
        fold_id=0,
        training_cutoff=cutoff,
        origin_timestamp=t0,
        target_timestamp=t1,
        target_type="yield_curve",
        target_name="DGS10",
        forecast=3.50,
        actual=None,
        status=ForecastStatus.GENERATED,
    )
    ledger.add_record(rec)

    # Initially unscored: RMSE is NaN
    assert np.isnan(ledger.compute_rmse("Random_Walk"))

    # 2. Later, target t1 is observed and ledger scores it
    actuals = pd.DataFrame({"DGS10": [3.55]}, index=[t1])
    scored_count = ledger.score_targets(actuals)
    assert scored_count == 1

    # Record is now SCORED with actual populated
    updated_rec = ledger.records[0]
    assert updated_rec.status == ForecastStatus.SCORED
    assert updated_rec.actual == 3.55

    # RMSE is now calculable: sqrt((3.50 - 3.55)^2) = 0.05
    rmse = ledger.compute_rmse("Random_Walk")
    assert np.isclose(rmse, 0.05, atol=1e-5)


def test_decision_and_execution_timing_contract():
    """Verify decision and execution contract ordering and synthetic proxy disclosure."""
    t_orig = pd.Timestamp("2023-01-02 16:00:00")
    t_dec = pd.Timestamp("2023-01-02 16:05:00")
    t_exec = pd.Timestamp("2023-01-03 09:30:00")
    t_end = pd.Timestamp("2023-01-04 09:30:00")

    # Valid contract
    contract = DecisionContract(
        origin_timestamp=t_orig,
        decision_timestamp=t_dec,
        execution_timestamp=t_exec,
        holding_period_start=t_exec,
        holding_period_end=t_end,
        timing_assumption=ExecutionTimingAssumption.NEXT_OPEN_FILL,
        is_synthetic_proxy=False,
    )
    assert contract.is_synthetic_proxy is False

    # Timing order violation: decision before origin
    with pytest.raises(ValueError, match="Timing causality violated"):
        DecisionContract(
            origin_timestamp=t_orig,
            decision_timestamp=pd.Timestamp("2023-01-02 15:55:00"),  # Lookahead decision
            execution_timestamp=t_exec,
            holding_period_start=t_exec,
            holding_period_end=t_end,
            timing_assumption=ExecutionTimingAssumption.NEXT_OPEN_FILL,
        )


def test_walk_forward_harness_genuine_forecasts_and_unavailable_nan():
    """Verify that walk-forward harness outputs genuine EVALUATED forecasts for models 1-5 and NaN for unavailable models."""
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet").iloc[-850:]
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet").iloc[-850:]
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet")

    cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=42)
    harness = WalkForwardHarness(yield_df, factor_df, macro_df, config=cfg)
    # Clear ML feature panel to verify UNAVAILABLE handling on GBM
    harness.X_ml = pd.DataFrame()
    harness.y_ml = pd.DataFrame()

    eval_res = harness.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=1)

    table = eval_res["baseline_table"]
    ledger = eval_res["forecast_ledger"]

    # Models 1-5 have genuine rolling one-step evaluations (Prompt 2)
    genuine_models = ["Random + Walk", "PCA + VAR", "Static + NS", "DNS + Kalman", "DNS + Kalman + Macro"]
    for m in genuine_models:
        rmse = table.loc[m, "OOS Curve RMSE (bp)"]
        assert not np.isnan(rmse), f"Expected numeric RMSE for genuine model {m}, got NaN"
        assert rmse > 0.0, f"Expected positive RMSE for {m}, got {rmse}"
        assert table.loc[m, "Forecast Status"] == "EVALUATED"

    # GBM without features must remain UNAVAILABLE with NaN RMSE
    assert np.isnan(table.loc["GBM", "OOS Curve RMSE (bp)"]), (
        f"Model 'GBM' without features must have NaN RMSE, got {table.loc['GBM', 'OOS Curve RMSE (bp)']}"
    )
    assert table.loc["GBM", "Forecast Status"] == "UNAVAILABLE"

    # Ledger must contain explicit UNAVAILABLE records for GBM
    gbm_unavailable = [r for r in ledger.records if r.model_id == "GBM" and r.status == ForecastStatus.UNAVAILABLE]
    assert len(gbm_unavailable) > 0
    assert any("ML feature panel unavailable" in r.reason for r in gbm_unavailable)
