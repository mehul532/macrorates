"""
Research Integrity & Forecast Contract Module.

Defines the formal contracts governing rolling out-of-sample term structure
forecasting, timing assumptions, and execution:
1. ForecastRecord & ForecastLedger: Common rolling one-step forecast protocol.
   - Separation of forecast generation from target scoring.
   - Strict origin < target timestamp validation.
   - Explicit training cutoff, input availability cutoff, and model status.
   - Invariant: An unavailable or failed model CANNOT acquire a numeric RMSE.
2. ExecutionTimingAssumption & DecisionContract:
   - Distinguishes daily synthetic timing assumptions from executable fills.
   - Eliminates same-close retrospective lookahead assumptions.
   - Prevents double-shifting of signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class ForecastStatus(str, Enum):
    """Lifecycle status of an out-of-sample forecast record."""
    GENERATED = "GENERATED"      # Forecast generated at origin t, target t+1 not yet observed
    SCORED = "SCORED"            # Target t+1 observed and evaluated
    UNAVAILABLE = "UNAVAILABLE"  # Model failed to fit or data unavailable at origin t
    FAILED = "FAILED"            # Model estimation threw an unhandled error
    INVALIDATED = "INVALIDATED"  # Historical record identified as containing lookahead or artificial scaling


class ExecutionTimingAssumption(str, Enum):
    """
    Timing assumptions for trading decision and execution fills.
    """
    SYNTHETIC_SAME_CLOSE = "SYNTHETIC_SAME_CLOSE"
    # Origin at close t, assumed fill at close t.
    # UNVALIDATED FOR LIVE EXECUTION: Treasury CMT yields are published after
    # close (~4:00-4:30 PM ET) and cannot be executed at same-close prices.
    
    NEXT_OPEN_FILL = "NEXT_OPEN_FILL"
    # Origin at close t, order executed at market open on t+1.
    
    NEXT_CLOSE_FILL = "NEXT_CLOSE_FILL"
    # Origin at close t, order executed at market close on t+1.


@dataclass
class ForecastRecord:
    """
    Immutable specification for a single out-of-sample forecast observation.
    
    Guarantees:
    - origin_timestamp < target_timestamp (Strict no-lookahead / causal direction)
    - origin_timestamp >= training_cutoff
    - Unavailable/Failed models cannot possess numeric forecasts.
    """
    run_id: str
    model_id: str
    fold_id: Union[int, str]
    training_cutoff: pd.Timestamp
    origin_timestamp: pd.Timestamp
    target_timestamp: pd.Timestamp
    target_type: str                  # e.g., "yield_curve", "latent_factor", "spread"
    target_name: str                  # e.g., "DGS10", "slope", "2s10s"
    forecast: Optional[float] = None
    actual: Optional[float] = None
    units: str = "basis_points"       # "basis_points" or "percent"
    input_availability_cutoff: Optional[pd.Timestamp] = None
    status: ForecastStatus = ForecastStatus.GENERATED
    reason: Optional[str] = None

    def __post_init__(self):
        # Convert string dates to Timestamps if needed
        if not isinstance(self.training_cutoff, pd.Timestamp):
            self.training_cutoff = pd.Timestamp(self.training_cutoff)
        if not isinstance(self.origin_timestamp, pd.Timestamp):
            self.origin_timestamp = pd.Timestamp(self.origin_timestamp)
        if not isinstance(self.target_timestamp, pd.Timestamp):
            self.target_timestamp = pd.Timestamp(self.target_timestamp)
        if self.input_availability_cutoff and not isinstance(self.input_availability_cutoff, pd.Timestamp):
            self.input_availability_cutoff = pd.Timestamp(self.input_availability_cutoff)

        # Integrity Check 1: Origin strictly precedes Target
        if self.origin_timestamp >= self.target_timestamp:
            raise ValueError(
                f"Causal violation: origin_timestamp ({self.origin_timestamp}) must strictly precede "
                f"target_timestamp ({self.target_timestamp})."
            )

        # Integrity Check 2: Origin is at or after training cutoff
        if self.origin_timestamp < self.training_cutoff:
            raise ValueError(
                f"Boundary violation: origin_timestamp ({self.origin_timestamp}) cannot precede "
                f"training_cutoff ({self.training_cutoff})."
            )

        # Integrity Check 3: Unavailable models cannot acquire numeric forecasts
        if self.status in (ForecastStatus.UNAVAILABLE, ForecastStatus.FAILED, ForecastStatus.INVALIDATED):
            if self.forecast is not None and not np.isnan(self.forecast):
                raise ValueError(
                    f"Integrity violation: Model '{self.model_id}' is marked {self.status.value} "
                    f"but has a numeric forecast ({self.forecast}). Fabricated predictions are barred."
                )


@dataclass
class DecisionContract:
    """
    Contract governing the decision-to-execution-to-holding lifecycle.
    Prevents accidental double-shifting or pre-origin signal utilization.
    """
    origin_timestamp: pd.Timestamp
    decision_timestamp: pd.Timestamp
    execution_timestamp: pd.Timestamp
    holding_period_start: pd.Timestamp
    holding_period_end: pd.Timestamp
    timing_assumption: ExecutionTimingAssumption
    is_synthetic_proxy: bool = True

    def __post_init__(self):
        if not isinstance(self.origin_timestamp, pd.Timestamp):
            self.origin_timestamp = pd.Timestamp(self.origin_timestamp)
        if not isinstance(self.decision_timestamp, pd.Timestamp):
            self.decision_timestamp = pd.Timestamp(self.decision_timestamp)
        if not isinstance(self.execution_timestamp, pd.Timestamp):
            self.execution_timestamp = pd.Timestamp(self.execution_timestamp)
        if not isinstance(self.holding_period_start, pd.Timestamp):
            self.holding_period_start = pd.Timestamp(self.holding_period_start)
        if not isinstance(self.holding_period_end, pd.Timestamp):
            self.holding_period_end = pd.Timestamp(self.holding_period_end)

        # Timing order assertion
        if not (self.origin_timestamp <= self.decision_timestamp <= self.execution_timestamp):
            raise ValueError(
                f"Timing causality violated: origin ({self.origin_timestamp}) <= decision "
                f"({self.decision_timestamp}) <= execution ({self.execution_timestamp})"
            )

        if not (self.execution_timestamp == self.holding_period_start < self.holding_period_end):
            raise ValueError(
                f"Holding lifecycle violated: execution ({self.execution_timestamp}) == "
                f"holding_start ({self.holding_period_start}) < holding_end ({self.holding_period_end})"
            )


class ForecastLedger:
    """
    Ledger for managing, scoring, and independently evaluating out-of-sample forecasts.
    Guarantees that generation is strictly separated from scoring.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._records: List[ForecastRecord] = []

    def add_record(self, record: ForecastRecord) -> None:
        """Add a forecast record to the ledger."""
        if record.run_id != self.run_id:
            raise ValueError(f"Run ID mismatch: expected {self.run_id}, got {record.run_id}")
        self._records.append(record)

    def add_unavailable(
        self,
        model_id: str,
        fold_id: Union[int, str],
        training_cutoff: pd.Timestamp,
        origin_timestamp: pd.Timestamp,
        target_timestamp: pd.Timestamp,
        target_type: str,
        target_name: str,
        reason: str,
    ) -> None:
        """Record an explicit unavailable forecast."""
        rec = ForecastRecord(
            run_id=self.run_id,
            model_id=model_id,
            fold_id=fold_id,
            training_cutoff=training_cutoff,
            origin_timestamp=origin_timestamp,
            target_timestamp=target_timestamp,
            target_type=target_type,
            target_name=target_name,
            forecast=None,
            status=ForecastStatus.UNAVAILABLE,
            reason=reason,
        )
        self.add_record(rec)

    @property
    def records(self) -> List[ForecastRecord]:
        return list(self._records)

    def to_dataframe(self) -> pd.DataFrame:
        """Export ledger to DataFrame."""
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([vars(r) for r in self._records])

    def score_targets(self, actuals_df: pd.DataFrame, date_col: str = "date") -> int:
        """
        Score generated forecasts against observed targets.
        Keeps prediction generation strictly separate from scoring.
        """
        actuals_idx = actuals_df.copy()
        if date_col in actuals_idx.columns:
            actuals_idx[date_col] = pd.to_datetime(actuals_idx[date_col])
            actuals_idx = actuals_idx.set_index(date_col)
        else:
            actuals_idx.index = pd.to_datetime(actuals_idx.index)

        scored_count = 0
        for rec in self._records:
            if rec.status != ForecastStatus.GENERATED:
                continue

            target_dt = rec.target_timestamp
            if target_dt in actuals_idx.index and rec.target_name in actuals_idx.columns:
                val = actuals_idx.loc[target_dt, rec.target_name]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                rec.actual = float(val)
                rec.status = ForecastStatus.SCORED
                scored_count += 1
        return scored_count

    def compute_rmse(
        self,
        model_id: str,
        target_type: Optional[str] = None,
        target_name: Optional[str] = None,
    ) -> Optional[float]:
        """
        Compute RMSE from scored records for a model.
        Returns np.nan if model is unavailable, failed, or has 0 scored forecasts.
        An unavailable model CANNOT acquire a numeric RMSE.
        """
        matching = [
            r for r in self._records
            if r.model_id == model_id
            and (target_type is None or r.target_type == target_type)
            and (target_name is None or r.target_name == target_name)
        ]

        if not matching:
            return np.nan

        # If any record is UNAVAILABLE or FAILED, or no SCORED records exist:
        scored = [r for r in matching if r.status == ForecastStatus.SCORED and r.forecast is not None and r.actual is not None]
        if not scored:
            return np.nan

        sq_errs = [(r.forecast - r.actual) ** 2 for r in scored]
        return float(np.sqrt(np.mean(sq_errs)))

    def get_model_summary(self, model_id: str) -> Dict[str, Any]:
        """Return diagnostic coverage and status summary for a model."""
        matching = [r for r in self._records if r.model_id == model_id]
        if not matching:
            return {
                "model_id": model_id,
                "status": "NOT_IN_LEDGER",
                "total_records": 0,
                "scored": 0,
                "unavailable": 0,
                "failed": 0,
                "rmse": np.nan,
            }

        counts = {s.value: sum(1 for r in matching if r.status == s) for s in ForecastStatus}
        rmse = self.compute_rmse(model_id)

        # Determine primary status
        if counts[ForecastStatus.SCORED.value] > 0 and counts[ForecastStatus.UNAVAILABLE.value] == 0:
            status = "VALID_EVALUATION"
        elif counts[ForecastStatus.UNAVAILABLE.value] > 0:
            status = "UNAVAILABLE"
        elif counts[ForecastStatus.FAILED.value] > 0:
            status = "FAILED"
        else:
            status = "PENDING_SCORING"

        reasons = list(set(r.reason for r in matching if r.reason))

        return {
            "model_id": model_id,
            "status": status,
            "total_records": len(matching),
            "counts": counts,
            "reasons": reasons,
            "rmse": rmse,
        }

    def get_records(self) -> List[ForecastRecord]:
        """Return all forecast records."""
        return list(self._records)

    def summary_by_model(self) -> Dict[str, Any]:
        """Return diagnostic coverage and status summary for all models in ledger."""
        models = sorted(list(set(r.model_id for r in self._records)))
        return {m: self.get_model_summary(m) for m in models}
