"""
Walk-Forward Out-of-Sample Harness & Multi-Model Comparative Evaluator.

Implements:
1. Rolling lookback window (e.g. 3yr lookback = 756 days, monthly refit = 21 days)
   re-estimating curve models and macro regressions without lookahead.
2. Cross-Model Baseline Suite:
   - Random Walk
   - PCA / VAR(1)
   - Static Nelson-Siegel + AR(1)
   - Dynamic Nelson-Siegel (DNS) + Kalman Filter
   - DNS + Kalman + Macro Announcement Surprises
3. Evaluation metrics:
   - Out-of-Sample Curve RMSE (bp)
   - Factor-Forecast RMSE (bp)
   - Strategy Sharpe, Sortino, Max Drawdown, Annualized Turnover, Hit Rate
   - PnL / DV01 and PnL / Turnover
   - Complete unbundled attribution chain:
     Gross P&L -> Transaction Costs -> Roll Costs -> Cash Interest -> Net P&L
"""

from dataclasses import dataclass, field
import datetime
import hashlib
import logging
from pathlib import Path
import subprocess
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash and dirty status for run provenance."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        if status:
            return f"{commit}-dirty"
        return commit
    except Exception:
        return "UNKNOWN_COMMIT"


def get_git_provenance() -> Dict[str, Any]:
    """Retrieve comprehensive git provenance including commit, dirty flag, and diff checksum."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        is_dirty = bool(status)
        diff_hash = "clean"
        if is_dirty:
            diff = subprocess.check_output(
                ["git", "diff", "HEAD"], stderr=subprocess.DEVNULL
            )
            diff_hash = hashlib.sha256(diff).hexdigest()[:16]
        return {
            "commit": commit,
            "is_dirty": is_dirty,
            "commit_or_dirty": f"{commit}-dirty" if is_dirty else commit,
            "dirty_tree_hash": diff_hash,
        }
    except Exception:
        return {
            "commit": "UNKNOWN_COMMIT",
            "is_dirty": False,
            "commit_or_dirty": "UNKNOWN_COMMIT",
            "dirty_tree_hash": "UNKNOWN",
        }


@dataclass
class MacroTimestampAudit:
    VERIFIED_AVAILABLE: ClassVar[str] = "VERIFIED_AVAILABLE"
    AVAILABLE: ClassVar[str] = "VERIFIED_AVAILABLE"
    POST_CLOSE: ClassVar[str] = "POST_CLOSE"
    LEGACY_DATE_ONLY_ASSUMED: ClassVar[str] = "LEGACY_DATE_ONLY_ASSUMED"
    INVALID_TIMESTAMP_REJECTED: ClassVar[str] = "INVALID_TIMESTAMP_REJECTED"

    is_available: bool
    status: str  # "VERIFIED_AVAILABLE", "POST_CLOSE", "LEGACY_DATE_ONLY_ASSUMED", "INVALID_TIMESTAMP_REJECTED"
    release_dt_nyc: Optional[pd.Timestamp]
    decision_cutoff_nyc: pd.Timestamp
    is_legacy_date_only: bool
    is_post_close: bool
    is_verified_available: bool
    reason: str = ""


def parse_macro_timestamp_availability(
    row: Any,
    dt: pd.Timestamp,
) -> MacroTimestampAudit:
    """
    Explicit timezone-aware macro release timestamp validation and availability audit.
    
    RESEARCH INTEGRITY & TIMING CONTRACT:
    - Decision cutoff is constructed in 'America/New_York' at exactly 16:00:00 on date dt.
    - Summer (EDT, UTC-4): 16:00 ET = 20:00 UTC.
    - Winter (EST, UTC-5): 16:00 ET = 21:00 UTC.
    - Release timestamp is normalized to 'America/New_York' before comparison.
    - Supported candidate fields: 'timestamp', 'release_timestamp', 'event_timestamp',
      'publication_timestamp', 'datetime'.
    - Supported inputs: timezone-aware Timestamps, ISO-8601 strings, datetime objects.
    - Explicit audit distinction:
      * VERIFIED_AVAILABLE: Valid timestamp <= 16:00:00 America/New_York on date dt.
      * POST_CLOSE: Valid timestamp > 16:00:00 America/New_York on date dt.
      * LEGACY_DATE_ONLY_ASSUMED: Missing timestamp or 00:00:00 time. Assumed pre-close
        daytime release (e.g. 8:30 AM ET) under legacy convention, but audited separately.
      * INVALID_TIMESTAMP_REJECTED: Corrupted/unparseable string or contradictory date.
    - Latency convention: Exact boundary release <= 16:00:00 ET is available for date dt.
      Releases > 16:00:00 ET enter post-close status and roll to the next eligible decision.
    """
    dt_date = pd.to_datetime(dt).date()
    decision_cutoff_nyc = pd.Timestamp(f"{dt_date} 16:00:00", tz="America/New_York")
    
    # 1. Candidate field extraction with documented precedence
    raw_ts = None
    candidate_fields = ("timestamp", "release_timestamp", "event_timestamp", "publication_timestamp", "datetime")
    for f in candidate_fields:
        if isinstance(row, dict) and f in row:
            v = row[f]
            if v is not None and not pd.isna(v):
                raw_ts = v
                break
        elif hasattr(row, "__getitem__"):
            try:
                if f in row:
                    v = row[f]
                    if v is not None and not pd.isna(v):
                        raw_ts = v
                        break
            except Exception:
                pass

    if raw_ts is None:
        return MacroTimestampAudit(
            is_available=True,
            status="LEGACY_DATE_ONLY_ASSUMED",
            release_dt_nyc=None,
            decision_cutoff_nyc=decision_cutoff_nyc,
            is_legacy_date_only=True,
            is_post_close=False,
            is_verified_available=False,
        )

    # 2. Parse raw_ts to pd.Timestamp
    ts = None
    if isinstance(raw_ts, (pd.Timestamp, datetime.datetime)):
        ts = pd.Timestamp(raw_ts)
    elif isinstance(raw_ts, str):
        try:
            ts = pd.to_datetime(raw_ts)
        except Exception:
            return MacroTimestampAudit(
                is_available=False,
                status="INVALID_TIMESTAMP_REJECTED",
                release_dt_nyc=None,
                decision_cutoff_nyc=decision_cutoff_nyc,
                is_legacy_date_only=False,
                is_post_close=False,
                is_verified_available=False,
            )
    else:
        try:
            ts = pd.to_datetime(raw_ts)
        except Exception:
            return MacroTimestampAudit(
                is_available=False,
                status="INVALID_TIMESTAMP_REJECTED",
                release_dt_nyc=None,
                decision_cutoff_nyc=decision_cutoff_nyc,
                is_legacy_date_only=False,
                is_post_close=False,
                is_verified_available=False,
            )

    if ts is pd.NaT or pd.isna(ts):
        return MacroTimestampAudit(
            is_available=False,
            status="INVALID_TIMESTAMP_REJECTED",
            release_dt_nyc=None,
            decision_cutoff_nyc=decision_cutoff_nyc,
            is_legacy_date_only=False,
            is_post_close=False,
            is_verified_available=False,
        )

    # 3. Check timezone and date-only (00:00:00) condition
    if ts.tz is None:
        if ts.time() == datetime.time(0, 0, 0):
            return MacroTimestampAudit(
                is_available=True,
                status="LEGACY_DATE_ONLY_ASSUMED",
                release_dt_nyc=None,
                decision_cutoff_nyc=decision_cutoff_nyc,
                is_legacy_date_only=True,
                is_post_close=False,
                is_verified_available=False,
            )
        ts_nyc = ts.tz_localize("America/New_York")
    else:
        ts_nyc = ts.tz_convert("America/New_York")

    # 4. Check contradictory dates (timestamp date differs by >1 day from event date)
    row_date = None
    for d_col in ("date", "release_date"):
        if isinstance(row, dict) and d_col in row and not pd.isna(row[d_col]):
            try:
                row_date = pd.to_datetime(row[d_col]).date()
                break
            except Exception:
                pass
        elif hasattr(row, "__getitem__"):
            try:
                if d_col in row and not pd.isna(row[d_col]):
                    row_date = pd.to_datetime(row[d_col]).date()
                    break
            except Exception:
                pass

    if row_date is not None:
        if abs((ts_nyc.date() - row_date).days) > 1:
            return MacroTimestampAudit(
                is_available=False,
                status="INVALID_TIMESTAMP_REJECTED",
                release_dt_nyc=ts_nyc,
                decision_cutoff_nyc=decision_cutoff_nyc,
                is_legacy_date_only=False,
                is_post_close=False,
                is_verified_available=False,
            )

    # 5. Evaluate against decision cutoff
    if ts_nyc <= decision_cutoff_nyc:
        return MacroTimestampAudit(
            is_available=True,
            status="VERIFIED_AVAILABLE",
            release_dt_nyc=ts_nyc,
            decision_cutoff_nyc=decision_cutoff_nyc,
            is_legacy_date_only=False,
            is_post_close=False,
            is_verified_available=True,
        )
    else:
        return MacroTimestampAudit(
            is_available=False,
            status="POST_CLOSE",
            release_dt_nyc=ts_nyc,
            decision_cutoff_nyc=decision_cutoff_nyc,
            is_legacy_date_only=False,
            is_post_close=True,
            is_verified_available=False,
        )


def is_timestamp_available_for_decision(row: Any, dt: pd.Timestamp) -> bool:
    """
    Check if macro release was available prior to the decision at market close on date dt.
    Returns True for verified pre-close releases and legacy date-only assumptions.
    Returns False for post-close and invalid/contradictory releases.
    """
    audit = parse_macro_timestamp_availability(row, dt)
    return audit.is_available


def get_file_checksum(filepath: Union[str, Path]) -> str:
    """Compute 16-character SHA-256 data checksum for audit traceability."""
    p = Path(filepath)
    if not p.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]

from src.curve.canonical import CANONICAL_TENORS, get_canonical_maturities, compute_observable_spreads
from src.curve.nelson_siegel import (
    StaticNelsonSiegel,
    NelsonSiegelFit,
    nelson_siegel_loadings,
    NelsonSiegelAR1Forecaster,
)
from src.curve.pca import YieldCurvePCA, PCAVARForecaster
from src.state_space.state_space import (
    DynamicNelsonSiegelMLE,
    KalmanFilterSmoother,
    StateSpaceResults,
    estimate_and_filter_state_space,
)
from src.strategy.portfolio import allocate_2s10s_spread, allocate_2s5s10s_butterfly, compute_continuous_positions
from src.strategy.backtest import RelativeValueBacktestEngine, CostModelV1Config, BacktestResult
from src.strategy.signals import map_curve_forecast_to_spread_signal
from src.backtest.contracts import ForecastRecord, ForecastLedger, ForecastStatus
from src.macro.macro_surprises import CausalMacroResponseEstimator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration parameters for the walk-forward evaluation harness."""
    train_window_days: int = 756   # 3 years lookback
    refit_frequency_days: int = 21  # Monthly refit (~1 trading month)
    initial_capital: float = 10_000_000.0
    target_dv01: float = 10_000.0
    cost_config: CostModelV1Config = field(default_factory=CostModelV1Config)
    gbm_backend: str = "sklearn"


class WalkForwardHarness:
    """
    Executes walk-forward out-of-sample re-estimation and multi-model comparison.
    """
    
    def __init__(
        self,
        yield_df: pd.DataFrame,
        factor_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        config: Optional[WalkForwardConfig] = None,
    ):
        self.config = config or WalkForwardConfig()
        
        # Prepare and harmonize dataframes
        self.yield_df = yield_df.copy()
        if "date" in self.yield_df.columns:
            self.yield_df["date"] = pd.to_datetime(self.yield_df["date"])
            self.yield_df = self.yield_df.sort_values("date").set_index("date")
            
        self.factor_df = factor_df.copy()
        if "date" in self.factor_df.columns:
            self.factor_df["date"] = pd.to_datetime(self.factor_df["date"])
            self.factor_df = self.factor_df.sort_values("date").set_index("date")
            
        self.macro_df = macro_df.copy()
        if "date" in self.macro_df.columns:
            self.macro_df["date"] = pd.to_datetime(self.macro_df["date"])
            
        # Common historical index
        self.common_dates = self.yield_df.index.intersection(self.factor_df.index)
        self.yield_df = self.yield_df.loc[self.common_dates]
        self.factor_df = self.factor_df.loc[self.common_dates]
        
        self.engine = RelativeValueBacktestEngine(
            cost_config=self.config.cost_config,
            initial_capital=self.config.initial_capital,
        )

        # Precompute ML feature panel for GBM baseline
        try:
            from src.model.ml_baseline import FactorFeatureEngineer
            self.X_ml, self.y_ml, self.ml_cols = FactorFeatureEngineer.build_feature_panel(
                factor_df=self.factor_df,
                yield_df=self.yield_df,
                macro_df=self.macro_df,
            )
        except Exception as e:
            logger.warning("ML feature panel generation failed: %s", e)
            self.X_ml, self.y_ml, self.ml_cols = pd.DataFrame(), pd.DataFrame(), []

    def generate_folds(self) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate (train_idx, test_idx) folds rolling forward through time.
        """
        n_days = len(self.common_dates)
        folds = []
        
        start_idx = 0
        while True:
            train_end = start_idx + self.config.train_window_days
            test_end = train_end + self.config.refit_frequency_days
            
            if train_end >= n_days:
                break
                
            test_end = min(test_end, n_days)
            train_idx = self.common_dates[start_idx:train_end]
            test_idx = self.common_dates[train_end:test_end]
            
            if len(test_idx) > 0:
                folds.append((train_idx, test_idx))
                
            start_idx += self.config.refit_frequency_days
            if test_end >= n_days:
                break
                
        logger.info("Generated %d walk-forward folds.", len(folds))
        return folds

    def run_walk_forward_evaluation(
        self,
        strategy_type: str = "2s10s",
        max_folds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute full walk-forward evaluation across all 5 baseline and main models:
        1. Random Walk
        2. PCA / VAR
        3. Static Nelson-Siegel
        4. DNS + Kalman
        5. DNS + Kalman + Macro
        """
        folds = self.generate_folds()
        if max_folds is not None:
            folds = folds[:max_folds]
            
        models = [
            "Random_Walk",
            "PCA_VAR",
            "Static_NS",
            "DNS_Kalman",
            "DNS_Kalman_Macro",
            "GBM",
        ]
        
        # Diagnostic and control models (Prompt 6 Integrity Audit)
        extra_models = [
            "DNS_Scaled_60",
            "Static_NS_Residual_Preserving",
            "DNS_Kalman_Residual_Preserving",
        ]
        all_eval_models = models + extra_models
        
        # Initialize formal ForecastLedger under research-integrity contract
        ledger = ForecastLedger(run_id="walk_forward_evaluation")
        self.ledger = ledger
        
        tenor_cols = [c for c in CANONICAL_TENORS.keys() if c in self.yield_df.columns]
        maturities = get_canonical_maturities(tenor_cols)
        mat_dict = {c: CANONICAL_TENORS[c] for c in tenor_cols}

        # Cumulative out-of-sample predictions, signals, and raw unclipped signal containers
        oos_signals = {m: [] for m in all_eval_models}
        raw_signals = {m: [] for m in all_eval_models}
        stage_inputs = {m: [] for m in all_eval_models}
        stage_outputs = {m: [] for m in all_eval_models}
        inherited_dns_clipping = {m: [] for m in all_eval_models}
        additional_clipping = {m: [] for m in all_eval_models}
        saturation_flags = {m: [] for m in all_eval_models}
        curve_sq_errors = {m: [] for m in all_eval_models}
        factor_sq_errors = {m: [] for m in all_eval_models}
        spread_2s10s_sq_errors = {m: [] for m in all_eval_models}
        fly_2s5s10s_sq_errors = {m: [] for m in all_eval_models}
        per_tenor_sq_errors = {m: {c: [] for c in tenor_cols} for m in all_eval_models}
        
        # Cross-sectional curve fit & dynamic factor diagnostics
        contemp_fit_sq_errors = []
        factor_rw_sq_errors = []
        factor_ar1_sq_errors = []
        fold_macro_audit = []
        
        # Observable 2s10s spread error decomposition: e_spread = u_factor_spread + u_fit_spread
        spread_decomp_total_sq = []
        spread_decomp_factor_sq = []
        spread_decomp_fit_sq = []
        spread_decomp_cross = []

        # Construct unified evaluated decision calendar across all evaluated folds
        full_decision_calendar = []
        for f_idx_cal, (tr_dates_cal, te_dates_cal) in enumerate(folds):
            pairs_cal = [(tr_dates_cal[-1], te_dates_cal[0])] + [
                (te_dates_cal[i], te_dates_cal[i + 1]) for i in range(len(te_dates_cal) - 1)
            ]
            for k_cal, (o_dt, t_dt) in enumerate(pairs_cal):
                dt_d = pd.to_datetime(o_dt).date()
                c_nyc = pd.Timestamp(f"{dt_d} 16:00:00", tz="America/New_York")
                full_decision_calendar.append({
                    "decision_index": len(full_decision_calendar),
                    "fold_id": f_idx_cal,
                    "fold_k_idx": k_cal,
                    "origin_timestamp": o_dt,
                    "target_timestamp": t_dt,
                    "cutoff_nyc": c_nyc,
                })

        first_orig_dt = full_decision_calendar[0]["origin_timestamp"] if full_decision_calendar else None
        if first_orig_dt is not None and first_orig_dt in self.common_dates:
            first_idx = self.common_dates.get_loc(first_orig_dt)
            if first_idx > 0:
                prior_dt = self.common_dates[first_idx - 1]
                prior_cutoff_nyc = pd.Timestamp(f"{pd.to_datetime(prior_dt).date()} 16:00:00", tz="America/New_York")
            else:
                prior_cutoff_nyc = pd.Timestamp.min.tz_localize("America/New_York")
        else:
            prior_cutoff_nyc = pd.Timestamp.min.tz_localize("America/New_York")

        # Map releases from self.macro_df across the full evaluated decision calendar
        fold_assigned_macro_events = {f_i: [] for f_i in range(len(folds))}
        events_with_no_subsequent_decision = []
        unassigned_invalid_timestamp_events = []
        macro_coverage_end_str = "N/A"

        if self.macro_df is not None and not self.macro_df.empty and "date" in self.macro_df.columns:
            try:
                macro_coverage_end_str = str(pd.to_datetime(self.macro_df["date"]).max().date())
            except Exception:
                macro_coverage_end_str = "N/A"

            for ev_idx, m_row in self.macro_df.iterrows():
                ev_dt = m_row["date"]
                ind = m_row.get("indicator")
                surp = m_row.get("surprise_ann", m_row.get("surprise", np.nan))

                avail_audit = parse_macro_timestamp_availability(m_row, ev_dt)
                if avail_audit.status == MacroTimestampAudit.INVALID_TIMESTAMP_REJECTED:
                    unassigned_invalid_timestamp_events.append({
                        "event_row": ev_idx,
                        "indicator": ind,
                        "date": ev_dt,
                        "reason": avail_audit.reason,
                    })
                    continue

                # Determine normalized release timestamp in America/New_York
                if avail_audit.release_dt_nyc is not None:
                    t_rel = avail_audit.release_dt_nyc
                else:
                    ev_date_obj = pd.to_datetime(ev_dt).date()
                    t_rel = pd.Timestamp(f"{ev_date_obj} 08:30:00", tz="America/New_York")

                # Events occurring on or before prior_cutoff_nyc belong to initial training history
                if t_rel <= prior_cutoff_nyc:
                    continue

                # Find the FIRST decision whose cutoff admits the release timestamp
                assigned_dec = None
                for dec in full_decision_calendar:
                    if t_rel <= dec["cutoff_nyc"]:
                        assigned_dec = dec
                        break

                if assigned_dec is None:
                    # Event occurred after cutoff of the final evaluated decision origin
                    events_with_no_subsequent_decision.append({
                        "event_row": ev_idx,
                        "indicator": ind,
                        "date": str(ev_dt),
                        "release_timestamp": str(t_rel),
                        "surprise": surp,
                    })
                    continue

                ev_date_val = pd.to_datetime(ev_dt).date()
                ev_date_cutoff = pd.Timestamp(f"{ev_date_val} 16:00:00", tz="America/New_York")
                assigned_orig_val = pd.to_datetime(assigned_dec["origin_timestamp"]).date()

                is_trading_date = assigned_dec["origin_timestamp"] in self.common_dates or ev_dt in self.common_dates
                if is_trading_date and t_rel <= ev_date_cutoff and assigned_orig_val == ev_date_val:
                    if avail_audit.status == MacroTimestampAudit.AVAILABLE:
                        release_status = "VERIFIED_AVAILABLE"
                        is_verified = True
                        is_legacy = False
                    else:
                        release_status = "LEGACY_DATE_ONLY_ASSUMED"
                        is_verified = False
                        is_legacy = True
                    is_post_close = False
                    is_rolled = False
                else:
                    is_rolled = True
                    is_verified = False
                    is_legacy = False
                    if t_rel > ev_date_cutoff:
                        release_status = "POST_CLOSE"
                        is_post_close = True
                    else:
                        release_status = "NON_TRADING_DAY_RELEASE"
                        is_post_close = False

                fold_assigned_macro_events[assigned_dec["fold_id"]].append({
                    "event_idx": ev_idx,
                    "indicator": ind,
                    "release_date": ev_dt,
                    "release_timestamp": t_rel,
                    "surprise": surp,
                    "assigned_orig_dt": assigned_dec["origin_timestamp"],
                    "assigned_target_dt": assigned_dec["target_timestamp"],
                    "fold_id": assigned_dec["fold_id"],
                    "fold_k_idx": assigned_dec["fold_k_idx"],
                    "decision_cutoff_nyc": assigned_dec["cutoff_nyc"],
                    "release_status": release_status,
                    "is_verified": is_verified,
                    "is_legacy": is_legacy,
                    "is_post_close": is_post_close,
                    "is_rolled": is_rolled,
                })
        
        for f_idx, (train_dates, test_dates) in enumerate(folds):
            # Define explicit rolling 1-step forecast origin-target pairs:
            # Origin t_0 = train_dates[-1] predicts target test_dates[0].
            # For subsequent steps, origin test_dates[i] predicts target test_dates[i+1].
            # Guarantees origin strictly precedes target with zero lookahead and no lost boundary data.
            origin_target_pairs = [(train_dates[-1], test_dates[0])] + [
                (test_dates[i], test_dates[i + 1]) for i in range(len(test_dates) - 1)
            ]
            orig_dates = [pair[0] for pair in origin_target_pairs]
            tgt_dates = [pair[1] for pair in origin_target_pairs]
            
            # In-sample slices (STRICTLY <= train_dates[-1])
            y_train = self.yield_df.loc[train_dates, tenor_cols]
            y_test = self.yield_df.loc[test_dates, tenor_cols]
            spreads_act = compute_observable_spreads(y_test.values, tenor_cols)
            y_origin_all = np.vstack([y_train.values[-1:], y_test.values[:-1]])

            # Training spread scale for standardizing forecast changes
            train_spreads = compute_observable_spreads(y_train.values, tenor_cols)
            if strategy_type == "2s10s" and "2s10s" in train_spreads:
                train_spread_std = float(np.std(np.diff(train_spreads["2s10s"]))) + 1e-6
            elif strategy_type in ("2s5s10s", "fly") and "2s5s10s" in train_spreads:
                train_spread_std = float(np.std(np.diff(train_spreads["2s5s10s"]))) + 1e-6
            else:
                train_spread_std = 0.05
            
            # Track macroeconomic events in train vs test windows
            if self.macro_df is not None and not self.macro_df.empty and "date" in self.macro_df.columns:
                m_tr = self.macro_df[(self.macro_df["date"] >= train_dates[0]) & (self.macro_df["date"] <= train_dates[-1])]
                m_te_calendar = self.macro_df[(self.macro_df["date"] >= test_dates[0]) & (self.macro_df["date"] <= test_dates[-1])]
            else:
                m_tr = pd.DataFrame()
                m_te_calendar = pd.DataFrame()
            
            fold_macro_audit.append({
                "fold_id": f_idx,
                "train_cutoff": str(train_dates[-1].date()),
                "test_start": str(test_dates[0].date()),
                "test_end": str(test_dates[-1].date()),
                "first_origin": str(orig_dates[0].date()),
                "last_origin": str(orig_dates[-1].date()),
                "training_event_count": len(m_tr),
                "calendar_event_count": len(m_te_calendar),
                "test_event_count": len(m_te_calendar),
                "observed_decision_events": 0,
                "evaluated_decision_events": 0,
                "timestamp_verified_available_events": 0,
                "legacy_date_only_assumed_events": 0,
                "timestamp_available_events": 0,
                "post_close_events": 0,
                "rolled_to_next_decision_events": 0,
                "invalid_timestamp_rejected_events": 0,
                "missing_surprise_events": 0,
                "usable_surprise_events": 0,
                "zero_surprise_events": 0,
                "eligible_coefficient_events": 0,
                "inadequate_history_events": 0,
                "active_overlay_events": 0,
                "zero_impact_events": 0,
                "nonzero_macro_days": 0,
                "applied_event_records": [],
            })

            # --- MODEL 1: RANDOM WALK ---
            # 1-step forecast is previous day's curve
            y_rw_pred = np.vstack([y_train.values[-1:], y_test.values[:-1]])
            curve_sq_errors["Random_Walk"].extend(((y_test.values - y_rw_pred) ** 2).flatten())
            for c_idx, c in enumerate(tenor_cols):
                per_tenor_sq_errors["Random_Walk"][c].extend(((y_test.iloc[:, c_idx].values - y_rw_pred[:, c_idx]) ** 2).tolist())
            
            spreads_rw = compute_observable_spreads(y_rw_pred, tenor_cols)
            if "2s10s" in spreads_rw and "2s10s" in spreads_act:
                spread_2s10s_sq_errors["Random_Walk"].extend(((spreads_act["2s10s"] - spreads_rw["2s10s"]) ** 2).flatten())
                factor_sq_errors["Random_Walk"].extend(((spreads_act["2s10s"] - spreads_rw["2s10s"]) ** 2).flatten())
            if "2s5s10s" in spreads_rw and "2s5s10s" in spreads_act:
                fly_2s5s10s_sq_errors["Random_Walk"].extend(((spreads_act["2s5s10s"] - spreads_rw["2s5s10s"]) ** 2).flatten())
                
            sig_rw = pd.Series(0.0, index=orig_dates)
            oos_signals["Random_Walk"].append(sig_rw)
            raw_signals["Random_Walk"].extend([0.0] * len(orig_dates))
            stage_inputs["Random_Walk"].extend([0.0] * len(orig_dates))
            stage_outputs["Random_Walk"].extend([0.0] * len(orig_dates))
            inherited_dns_clipping["Random_Walk"].extend([False] * len(orig_dates))
            additional_clipping["Random_Walk"].extend([False] * len(orig_dates))
            saturation_flags["Random_Walk"].extend([False] * len(orig_dates))

            # Record Random Walk in ForecastLedger
            for k, (orig_dt, tgt_dt) in enumerate(origin_target_pairs):
                for c_idx, c in enumerate(tenor_cols):
                    ledger.add_record(ForecastRecord(
                        run_id=ledger.run_id,
                        model_id="Random_Walk",
                        fold_id=f_idx,
                        training_cutoff=train_dates[-1],
                        origin_timestamp=orig_dt,
                        target_timestamp=tgt_dt,
                        target_type="yield_curve",
                        target_name=c,
                        forecast=float(y_rw_pred[k, c_idx]),
                        actual=float(y_test.iloc[k, c_idx]),
                        status=ForecastStatus.SCORED,
                    ))
            
            # --- MODEL 2: PCA / VAR(1) ---
            pca_forecaster = PCAVARForecaster(n_components=3)
            y_train_pca_df = y_train.copy()
            y_train_pca_df["date"] = train_dates
            pca_forecaster.fit(y_train_pca_df, maturities_dict=mat_dict, date_col="date")
            y_pred_pca, scores_pred_pca, scores_obs_pca = pca_forecaster.sequential_predict_and_update(y_test.values)
            
            curve_sq_errors["PCA_VAR"].extend(((y_test.values - y_pred_pca) ** 2).flatten())
            for c_idx, c in enumerate(tenor_cols):
                per_tenor_sq_errors["PCA_VAR"][c].extend(((y_test.iloc[:, c_idx].values - y_pred_pca[:, c_idx]) ** 2).tolist())
            spreads_pca = compute_observable_spreads(y_pred_pca, tenor_cols)
            if "2s10s" in spreads_pca and "2s10s" in spreads_act:
                spread_2s10s_sq_errors["PCA_VAR"].extend(((spreads_act["2s10s"] - spreads_pca["2s10s"]) ** 2).flatten())
                factor_sq_errors["PCA_VAR"].extend(((spreads_act["2s10s"] - spreads_pca["2s10s"]) ** 2).flatten())
            if "2s5s10s" in spreads_pca and "2s5s10s" in spreads_act:
                fly_2s5s10s_sq_errors["PCA_VAR"].extend(((spreads_act["2s5s10s"] - spreads_pca["2s5s10s"]) ** 2).flatten())
                
            # Signal: mapped from 1-step predicted yield curve into predicted observable spread change
            sig_pca_arr, diag_pca = map_curve_forecast_to_spread_signal(
                y_pred_pca, y_origin_all, tenor_cols, train_spread_std, strategy_type, return_details=True
            )
            sig_pca = pd.Series(sig_pca_arr, index=orig_dates)
            oos_signals["PCA_VAR"].append(sig_pca)
            raw_pca = diag_pca["raw_input"]
            raw_signals["PCA_VAR"].extend(raw_pca.tolist() if isinstance(raw_pca, np.ndarray) else [raw_pca])
            stage_inputs["PCA_VAR"].extend(raw_pca.tolist() if isinstance(raw_pca, np.ndarray) else [raw_pca])
            stage_outputs["PCA_VAR"].extend(sig_pca_arr.tolist())
            inherited_dns_clipping["PCA_VAR"].extend([False] * len(orig_dates))
            additional_clipping["PCA_VAR"].extend(diag_pca["is_clipped"].tolist() if isinstance(diag_pca["is_clipped"], np.ndarray) else [diag_pca["is_clipped"]])
            saturation_flags["PCA_VAR"].extend(diag_pca["is_saturated"].tolist() if isinstance(diag_pca["is_saturated"], np.ndarray) else [diag_pca["is_saturated"]])
            
            for k, (orig_dt, tgt_dt) in enumerate(origin_target_pairs):
                for c_idx, c in enumerate(tenor_cols):
                    ledger.add_record(ForecastRecord(
                        run_id=ledger.run_id,
                        model_id="PCA_VAR",
                        fold_id=f_idx,
                        training_cutoff=train_dates[-1],
                        origin_timestamp=orig_dt,
                        target_timestamp=tgt_dt,
                        target_type="yield_curve",
                        target_name=c,
                        forecast=float(y_pred_pca[k, c_idx]),
                        actual=float(y_test.iloc[k, c_idx]),
                        status=ForecastStatus.SCORED,
                    ))
            
            # --- MODEL 3: STATIC NELSON-SIEGEL ---
            ns_forecaster = NelsonSiegelAR1Forecaster(lambda_param=0.7308)
            ns_forecaster.fit(y_train.values, maturities=maturities)
            y_pred_ns, factors_pred_ns, factors_obs_ns = ns_forecaster.sequential_predict_and_update(y_test.values)
            
            curve_sq_errors["Static_NS"].extend(((y_test.values - y_pred_ns) ** 2).flatten())
            for c_idx, c in enumerate(tenor_cols):
                per_tenor_sq_errors["Static_NS"][c].extend(((y_test.iloc[:, c_idx].values - y_pred_ns[:, c_idx]) ** 2).tolist())
            spreads_ns = compute_observable_spreads(y_pred_ns, tenor_cols)
            if "2s10s" in spreads_ns and "2s10s" in spreads_act:
                spread_2s10s_sq_errors["Static_NS"].extend(((spreads_act["2s10s"] - spreads_ns["2s10s"]) ** 2).flatten())
                factor_sq_errors["Static_NS"].extend(((spreads_act["2s10s"] - spreads_ns["2s10s"]) ** 2).flatten())
            if "2s5s10s" in spreads_ns and "2s5s10s" in spreads_act:
                fly_2s5s10s_sq_errors["Static_NS"].extend(((spreads_act["2s5s10s"] - spreads_ns["2s5s10s"]) ** 2).flatten())
                
            # Signal: mapped from 1-step predicted yield curve into predicted observable spread change
            sig_ns_arr, diag_ns = map_curve_forecast_to_spread_signal(
                y_pred_ns, y_origin_all, tenor_cols, train_spread_std, strategy_type, return_details=True
            )
            sig_ns = pd.Series(sig_ns_arr, index=orig_dates)
            oos_signals["Static_NS"].append(sig_ns)
            raw_ns = diag_ns["raw_input"]
            raw_signals["Static_NS"].extend(raw_ns.tolist() if isinstance(raw_ns, np.ndarray) else [raw_ns])
            stage_inputs["Static_NS"].extend(raw_ns.tolist() if isinstance(raw_ns, np.ndarray) else [raw_ns])
            stage_outputs["Static_NS"].extend(sig_ns_arr.tolist())
            inherited_dns_clipping["Static_NS"].extend([False] * len(orig_dates))
            additional_clipping["Static_NS"].extend(diag_ns["is_clipped"].tolist() if isinstance(diag_ns["is_clipped"], np.ndarray) else [diag_ns["is_clipped"]])
            saturation_flags["Static_NS"].extend(diag_ns["is_saturated"].tolist() if isinstance(diag_ns["is_saturated"], np.ndarray) else [diag_ns["is_saturated"]])
            
            for k, (orig_dt, tgt_dt) in enumerate(origin_target_pairs):
                for c_idx, c in enumerate(tenor_cols):
                    ledger.add_record(ForecastRecord(
                        run_id=ledger.run_id,
                        model_id="Static_NS",
                        fold_id=f_idx,
                        training_cutoff=train_dates[-1],
                        origin_timestamp=orig_dt,
                        target_timestamp=tgt_dt,
                        target_type="yield_curve",
                        target_name=c,
                        forecast=float(y_pred_ns[k, c_idx]),
                        actual=float(y_test.iloc[k, c_idx]),
                        status=ForecastStatus.SCORED,
                    ))

            # --- Prompt 6 Econometric Diagnostics for Nelson-Siegel ---
            # 1. Contemporaneous curve-fit error: e_t^{fit} = y_t - Lambda @ beta_obs
            Lambda_ns = nelson_siegel_loadings(maturities, lambda_param=0.7308)
            y_fit_ns = (Lambda_ns @ factors_obs_ns.T).T
            contemp_fit_sq_errors.extend(((y_test.values - y_fit_ns) ** 2).flatten())

            # Observable 2s10s spread error decomposition: e_spread = u_factor_spread + u_fit_spread
            s_act_arr = spreads_act["2s10s"]
            s_pred_ns_arr = spreads_ns["2s10s"]
            s_fit_ns_arr = compute_observable_spreads(y_fit_ns, tenor_cols)["2s10s"]
            
            e_total_bp = (s_act_arr - s_pred_ns_arr) * 100.0
            u_factor_bp = (s_fit_ns_arr - s_pred_ns_arr) * 100.0
            u_fit_bp = (s_act_arr - s_fit_ns_arr) * 100.0
            
            spread_decomp_total_sq.extend((e_total_bp ** 2).tolist())
            spread_decomp_factor_sq.extend((u_factor_bp ** 2).tolist())
            spread_decomp_fit_sq.extend((u_fit_bp ** 2).tolist())
            spread_decomp_cross.extend((2.0 * u_factor_bp * u_fit_bp).tolist())
            
            # 2. Factor Random Walk vs AR(1) dynamics
            b_train_last_ns = np.linalg.pinv(Lambda_ns) @ y_train.values[-1]
            b_prev_all_ns = np.vstack([b_train_last_ns.reshape(1, -1), factors_obs_ns[:-1]])
            factor_rw_sq_errors.extend(((factors_obs_ns - b_prev_all_ns) ** 2).flatten())
            factor_ar1_sq_errors.extend(((factors_obs_ns - factors_pred_ns) ** 2).flatten())
            
            # 3. Residual-Preserving Static NS Forecast: y_origin + Lambda @ (beta_pred - beta_orig)
            delta_beta_ns = factors_pred_ns - b_prev_all_ns
            y_pred_ns_res = y_origin_all + (Lambda_ns @ delta_beta_ns.T).T
            curve_sq_errors["Static_NS_Residual_Preserving"].extend(((y_test.values - y_pred_ns_res) ** 2).flatten())
            for c_idx, c in enumerate(tenor_cols):
                per_tenor_sq_errors["Static_NS_Residual_Preserving"][c].extend(
                    ((y_test.iloc[:, c_idx].values - y_pred_ns_res[:, c_idx]) ** 2).tolist()
                )
            spreads_ns_res = compute_observable_spreads(y_pred_ns_res, tenor_cols)
            if "2s10s" in spreads_ns_res and "2s10s" in spreads_act:
                diff_res = ((spreads_act["2s10s"] - spreads_ns_res["2s10s"]) ** 2).flatten()
                spread_2s10s_sq_errors["Static_NS_Residual_Preserving"].extend(diff_res)
                factor_sq_errors["Static_NS_Residual_Preserving"].extend(diff_res)
            if "2s5s10s" in spreads_ns_res and "2s5s10s" in spreads_act:
                fly_2s5s10s_sq_errors["Static_NS_Residual_Preserving"].extend(
                    ((spreads_act["2s5s10s"] - spreads_ns_res["2s5s10s"]) ** 2).flatten()
                )
            sig_ns_res_arr, diag_ns_res = map_curve_forecast_to_spread_signal(
                y_pred_ns_res, y_origin_all, tenor_cols, train_spread_std, strategy_type, return_details=True
            )
            raw_ns_res = diag_ns_res["raw_input"]
            raw_signals["Static_NS_Residual_Preserving"].extend(raw_ns_res.tolist() if isinstance(raw_ns_res, np.ndarray) else [raw_ns_res])
            stage_inputs["Static_NS_Residual_Preserving"].extend(raw_ns_res.tolist() if isinstance(raw_ns_res, np.ndarray) else [raw_ns_res])
            stage_outputs["Static_NS_Residual_Preserving"].extend(sig_ns_res_arr.tolist())
            inherited_dns_clipping["Static_NS_Residual_Preserving"].extend([False] * len(orig_dates))
            additional_clipping["Static_NS_Residual_Preserving"].extend(diag_ns_res["is_clipped"].tolist() if isinstance(diag_ns_res["is_clipped"], np.ndarray) else [diag_ns_res["is_clipped"]])
            saturation_flags["Static_NS_Residual_Preserving"].extend(diag_ns_res["is_saturated"].tolist() if isinstance(diag_ns_res["is_saturated"], np.ndarray) else [diag_ns_res["is_saturated"]])
            sig_ns_res = pd.Series(sig_ns_res_arr, index=orig_dates)
            oos_signals["Static_NS_Residual_Preserving"].append(sig_ns_res)
            
            # --- MODEL 4: DNS + KALMAN ---
            y_train_ss_df = y_train.copy()
            y_train_ss_df["date"] = train_dates
            dns_res = estimate_and_filter_state_space(
                y_train_ss_df,
                maturities_dict=mat_dict,
                date_col="date",
                lambda_param=0.7308,
                use_mle_optimization=False,
            )
            b_train_end = dns_res.filtered_states.iloc[-1][["level", "slope", "curvature"]].values
            P_train_end = dns_res.filtered_cov[-1]
            
            kf_smoother = KalmanFilterSmoother(
                maturities=maturities,
                lambda_param=0.7308,
                mu=dns_res.mu,
                transition_matrix=dns_res.transition_matrix,
                state_cov=dns_res.state_cov,
                obs_cov=dns_res.obs_cov,
            )
            y_pred_kf, beta_pred_kf, beta_filt_kf, P_filt_kf = kf_smoother.sequential_predict_and_update(
                y_test.values,
                initial_state=b_train_end,
                initial_cov=P_train_end,
            )
            
            curve_sq_errors["DNS_Kalman"].extend(((y_test.values - y_pred_kf) ** 2).flatten())
            for c_idx, c in enumerate(tenor_cols):
                sq_kf = ((y_test.iloc[:, c_idx].values - y_pred_kf[:, c_idx]) ** 2).tolist()
                per_tenor_sq_errors["DNS_Kalman"][c].extend(sq_kf)
                per_tenor_sq_errors["DNS_Kalman_Macro"][c].extend(sq_kf)
            spreads_kf = compute_observable_spreads(y_pred_kf, tenor_cols)
            if "2s10s" in spreads_kf and "2s10s" in spreads_act:
                spread_2s10s_sq_errors["DNS_Kalman"].extend(((spreads_act["2s10s"] - spreads_kf["2s10s"]) ** 2).flatten())
                factor_sq_errors["DNS_Kalman"].extend(((spreads_act["2s10s"] - spreads_kf["2s10s"]) ** 2).flatten())
            if "2s5s10s" in spreads_kf and "2s5s10s" in spreads_act:
                fly_2s5s10s_sq_errors["DNS_Kalman"].extend(((spreads_act["2s5s10s"] - spreads_kf["2s5s10s"]) ** 2).flatten())
                
            # Signal: mapped from 1-step predicted yield curve into predicted observable spread change
            sig_kf_arr, diag_kf = map_curve_forecast_to_spread_signal(
                y_pred_kf, y_origin_all, tenor_cols, train_spread_std, strategy_type, return_details=True
            )
            sig_kf = pd.Series(sig_kf_arr, index=orig_dates)
            oos_signals["DNS_Kalman"].append(sig_kf)
            raw_kf = diag_kf["raw_input"]
            raw_signals["DNS_Kalman"].extend(raw_kf.tolist() if isinstance(raw_kf, np.ndarray) else [raw_kf])
            stage_inputs["DNS_Kalman"].extend(raw_kf.tolist() if isinstance(raw_kf, np.ndarray) else [raw_kf])
            stage_outputs["DNS_Kalman"].extend(sig_kf_arr.tolist())
            inherited_dns_clipping["DNS_Kalman"].extend([False] * len(orig_dates))
            additional_clipping["DNS_Kalman"].extend(diag_kf["is_clipped"].tolist() if isinstance(diag_kf["is_clipped"], np.ndarray) else [diag_kf["is_clipped"]])
            saturation_flags["DNS_Kalman"].extend(diag_kf["is_saturated"].tolist() if isinstance(diag_kf["is_saturated"], np.ndarray) else [diag_kf["is_saturated"]])
            
            for k, (orig_dt, tgt_dt) in enumerate(origin_target_pairs):
                for c_idx, c in enumerate(tenor_cols):
                    ledger.add_record(ForecastRecord(
                        run_id=ledger.run_id,
                        model_id="DNS_Kalman",
                        fold_id=f_idx,
                        training_cutoff=train_dates[-1],
                        origin_timestamp=orig_dt,
                        target_timestamp=tgt_dt,
                        target_type="yield_curve",
                        target_name=c,
                        forecast=float(y_pred_kf[k, c_idx]),
                        actual=float(y_test.iloc[k, c_idx]),
                        status=ForecastStatus.SCORED,
                    ))

            # --- Residual-Preserving DNS Forecast Diagnostic ---
            b_prev_all_kf = np.vstack([b_train_end.reshape(1, -1), beta_filt_kf[:-1]])
            delta_beta_kf = beta_pred_kf - b_prev_all_kf
            y_pred_kf_res = y_origin_all + (Lambda_ns @ delta_beta_kf.T).T
            curve_sq_errors["DNS_Kalman_Residual_Preserving"].extend(((y_test.values - y_pred_kf_res) ** 2).flatten())
            for c_idx, c in enumerate(tenor_cols):
                per_tenor_sq_errors["DNS_Kalman_Residual_Preserving"][c].extend(
                    ((y_test.iloc[:, c_idx].values - y_pred_kf_res[:, c_idx]) ** 2).tolist()
                )
            spreads_kf_res = compute_observable_spreads(y_pred_kf_res, tenor_cols)
            if "2s10s" in spreads_kf_res and "2s10s" in spreads_act:
                diff_kf_res = ((spreads_act["2s10s"] - spreads_kf_res["2s10s"]) ** 2).flatten()
                spread_2s10s_sq_errors["DNS_Kalman_Residual_Preserving"].extend(diff_kf_res)
                factor_sq_errors["DNS_Kalman_Residual_Preserving"].extend(diff_kf_res)
            if "2s5s10s" in spreads_kf_res and "2s5s10s" in spreads_act:
                fly_2s5s10s_sq_errors["DNS_Kalman_Residual_Preserving"].extend(
                    ((spreads_act["2s5s10s"] - spreads_kf_res["2s5s10s"]) ** 2).flatten()
                )
            sig_kf_res_arr, diag_kf_res = map_curve_forecast_to_spread_signal(
                y_pred_kf_res, y_origin_all, tenor_cols, train_spread_std, strategy_type, return_details=True
            )
            raw_kf_res = diag_kf_res["raw_input"]
            raw_signals["DNS_Kalman_Residual_Preserving"].extend(raw_kf_res.tolist() if isinstance(raw_kf_res, np.ndarray) else [raw_kf_res])
            stage_inputs["DNS_Kalman_Residual_Preserving"].extend(raw_kf_res.tolist() if isinstance(raw_kf_res, np.ndarray) else [raw_kf_res])
            stage_outputs["DNS_Kalman_Residual_Preserving"].extend(sig_kf_res_arr.tolist())
            inherited_dns_clipping["DNS_Kalman_Residual_Preserving"].extend([False] * len(orig_dates))
            additional_clipping["DNS_Kalman_Residual_Preserving"].extend(diag_kf_res["is_clipped"].tolist() if isinstance(diag_kf_res["is_clipped"], np.ndarray) else [diag_kf_res["is_clipped"]])
            saturation_flags["DNS_Kalman_Residual_Preserving"].extend(diag_kf_res["is_saturated"].tolist() if isinstance(diag_kf_res["is_saturated"], np.ndarray) else [diag_kf_res["is_saturated"]])
            sig_kf_res = pd.Series(sig_kf_res_arr, index=orig_dates)
            oos_signals["DNS_Kalman_Residual_Preserving"].append(sig_kf_res)

            # --- CONTROL MODEL: DNS Scaled at 60% Exposure (Prompt 6 Macro Audit Control) ---
            # Identical forecast curve and spread RMSE to DNS_Kalman, but scaled to 60% risk exposure
            curve_sq_errors["DNS_Scaled_60"].extend(((y_test.values - y_pred_kf) ** 2).flatten())
            for c_idx, c in enumerate(tenor_cols):
                sq_err_c = ((y_test.iloc[:, c_idx].values - y_pred_kf[:, c_idx]) ** 2).tolist()
                per_tenor_sq_errors["DNS_Scaled_60"][c].extend(sq_err_c)
            if "2s10s" in spreads_kf and "2s10s" in spreads_act:
                spread_2s10s_sq_errors["DNS_Scaled_60"].extend(((spreads_act["2s10s"] - spreads_kf["2s10s"]) ** 2).flatten())
                factor_sq_errors["DNS_Scaled_60"].extend(((spreads_act["2s10s"] - spreads_kf["2s10s"]) ** 2).flatten())
            if "2s5s10s" in spreads_kf and "2s5s10s" in spreads_act:
                fly_2s5s10s_sq_errors["DNS_Scaled_60"].extend(((spreads_act["2s5s10s"] - spreads_kf["2s5s10s"]) ** 2).flatten())
            control_input = 0.60 * sig_kf_arr
            sig_scaled_60 = pd.Series(control_input, index=orig_dates)
            oos_signals["DNS_Scaled_60"].append(sig_scaled_60)
            raw_signals["DNS_Scaled_60"].extend(control_input.tolist())
            stage_inputs["DNS_Scaled_60"].extend(control_input.tolist())
            stage_outputs["DNS_Scaled_60"].extend(control_input.tolist())
            inherited_dns_clipping["DNS_Scaled_60"].extend(diag_kf["is_clipped"].tolist() if isinstance(diag_kf["is_clipped"], np.ndarray) else [diag_kf["is_clipped"]])
            additional_clipping["DNS_Scaled_60"].extend([False] * len(orig_dates))
            saturation_flags["DNS_Scaled_60"].extend((np.abs(control_input) >= 0.60 - 1e-6).tolist())
            
            # --- MODEL 5: DNS + KALMAN + MACRO ---
            curve_sq_errors["DNS_Kalman_Macro"].extend(((y_test.values - y_pred_kf) ** 2).flatten())
            if "2s10s" in spreads_kf and "2s10s" in spreads_act:
                spread_2s10s_sq_errors["DNS_Kalman_Macro"].extend(((spreads_act["2s10s"] - spreads_kf["2s10s"]) ** 2).flatten())
                factor_sq_errors["DNS_Kalman_Macro"].extend(((spreads_act["2s10s"] - spreads_kf["2s10s"]) ** 2).flatten())
            if "2s5s10s" in spreads_kf and "2s5s10s" in spreads_act:
                fly_2s5s10s_sq_errors["DNS_Kalman_Macro"].extend(((spreads_act["2s5s10s"] - spreads_kf["2s5s10s"]) ** 2).flatten())
                
            # Prompt 3: Estimate causal macro response coefficients strictly on training slice
            macro_estimator = CausalMacroResponseEstimator(min_events=8, predictive_horizon=1)
            fold_macro_betas = macro_estimator.fit_fold(
                fold_id=f_idx,
                training_cutoff=train_dates[-1],
                macro_df=self.macro_df,
                yield_df=y_train,
            )
            
            # Join macro releases assigned to this fold across the full evaluated decision calendar
            assigned_events = fold_assigned_macro_events[f_idx]
            macro_impulse = pd.Series(0.0, index=orig_dates)
            
            eval_dec_count = 0
            ts_verified_count = 0
            legacy_date_assumed_count = 0
            ts_avail_count = 0
            post_close_count = 0
            rolled_count = 0
            missing_surp_count = 0
            usable_surp_count = 0
            zero_surp_count = 0
            eligible_coeff_count = 0
            inadequate_hist_count = 0
            active_overlay_count = 0
            zero_impact_count = 0
            orig_release_dates = set()
            fold_applied_event_records = []

            for a_ev in assigned_events:
                eval_dec_count += 1
                dt_orig = a_ev["assigned_orig_dt"]
                orig_release_dates.add(dt_orig)
                ind = a_ev["indicator"]
                surp = a_ev["surprise"]

                # Release-time status accounting
                if a_ev["is_post_close"]:
                    post_close_count += 1
                if a_ev["is_rolled"]:
                    rolled_count += 1
                if a_ev["is_verified"]:
                    ts_verified_count += 1
                    ts_avail_count += 1
                elif a_ev["is_legacy"]:
                    legacy_date_assumed_count += 1
                    ts_avail_count += 1

                # Application-time surprise accounting
                if pd.isna(surp):
                    missing_surp_count += 1
                    fold_applied_event_records.append({
                        "event_idx": a_ev["event_idx"],
                        "indicator": ind,
                        "release_date": str(a_ev["release_date"]),
                        "release_timestamp": str(a_ev["release_timestamp"]),
                        "assigned_origin_date": str(dt_orig.date()),
                        "target_date": str(a_ev["assigned_target_dt"].date()) if hasattr(a_ev["assigned_target_dt"], "date") else None,
                        "fold_id": f_idx,
                        "surprise": None,
                        "beta": None,
                        "impulse": None,
                        "status": "MISSING_SURPRISE",
                        "is_rolled": a_ev["is_rolled"],
                        "is_post_close": a_ev["is_post_close"],
                        "release_status": a_ev["release_status"],
                    })
                    continue

                usable_surp_count += 1
                if surp == 0.0:
                    zero_surp_count += 1

                # Application-time coefficient eligibility
                b_info = fold_macro_betas.get(ind, {})
                is_eligible = (
                    b_info.get("status") == "ESTIMATED_CAUSAL" and
                    b_info.get("n_events", 0) >= 8
                )

                if not is_eligible:
                    inadequate_hist_count += 1
                    fold_applied_event_records.append({
                        "event_idx": a_ev["event_idx"],
                        "indicator": ind,
                        "release_date": str(a_ev["release_date"]),
                        "release_timestamp": str(a_ev["release_timestamp"]),
                        "assigned_origin_date": str(dt_orig.date()),
                        "target_date": str(a_ev["assigned_target_dt"].date()) if hasattr(a_ev["assigned_target_dt"], "date") else None,
                        "fold_id": f_idx,
                        "surprise": float(surp),
                        "beta": None,
                        "impulse": None,
                        "status": "INADEQUATE_HISTORY",
                        "is_rolled": a_ev["is_rolled"],
                        "is_post_close": a_ev["is_post_close"],
                        "release_status": a_ev["release_status"],
                    })
                    continue

                eligible_coeff_count += 1
                if strategy_type in ("2s5s10s", "fly"):
                    b_target = b_info.get("curvature_beta", 0.0)
                else:
                    b_target = b_info.get("slope_beta", 0.0)
                impulse_val = b_target * np.clip(surp, -2.0, 2.0)
                macro_impulse.loc[dt_orig] += impulse_val

                if impulse_val != 0.0:
                    active_overlay_count += 1
                    ev_status = "ACTIVE_OVERLAY"
                else:
                    zero_impact_count += 1
                    ev_status = "ZERO_IMPACT"

                fold_applied_event_records.append({
                    "event_idx": a_ev["event_idx"],
                    "indicator": ind,
                    "release_date": str(a_ev["release_date"]),
                    "release_timestamp": str(a_ev["release_timestamp"]),
                    "assigned_origin_date": str(dt_orig.date()),
                    "target_date": str(a_ev["assigned_target_dt"].date()) if hasattr(a_ev["assigned_target_dt"], "date") else None,
                    "fold_id": f_idx,
                    "surprise": float(surp),
                    "beta": float(b_target),
                    "impulse": float(impulse_val),
                    "status": ev_status,
                    "is_rolled": a_ev["is_rolled"],
                    "is_post_close": a_ev["is_post_close"],
                    "release_status": a_ev["release_status"],
                })

            nonzero_macro_days = int((macro_impulse != 0.0).sum())
            fold_macro_audit[-1]["observed_decision_events"] = eval_dec_count
            fold_macro_audit[-1]["evaluated_decision_events"] = eval_dec_count
            fold_macro_audit[-1]["timestamp_verified_available_events"] = ts_verified_count
            fold_macro_audit[-1]["legacy_date_only_assumed_events"] = legacy_date_assumed_count
            fold_macro_audit[-1]["timestamp_available_events"] = ts_avail_count
            fold_macro_audit[-1]["post_close_events"] = post_close_count
            fold_macro_audit[-1]["rolled_to_next_decision_events"] = rolled_count
            fold_macro_audit[-1]["missing_surprise_events"] = missing_surp_count
            fold_macro_audit[-1]["usable_surprise_events"] = usable_surp_count
            fold_macro_audit[-1]["zero_surprise_events"] = zero_surp_count
            fold_macro_audit[-1]["eligible_coefficient_events"] = eligible_coeff_count
            fold_macro_audit[-1]["inadequate_history_events"] = inadequate_hist_count
            fold_macro_audit[-1]["active_overlay_events"] = active_overlay_count
            fold_macro_audit[-1]["zero_impact_events"] = zero_impact_count
            fold_macro_audit[-1]["nonzero_macro_days"] = nonzero_macro_days
            fold_macro_audit[-1]["applied_event_records"] = fold_applied_event_records
            
            macro_scale = macro_impulse.loc[orig_dates].values / train_spread_std
            macro_overlay_input = 0.60 * sig_kf_arr + 0.40 * macro_scale
            sig_macro_arr = np.clip(macro_overlay_input, -1.0, 1.0)
            sig_macro_dns = pd.Series(sig_macro_arr, index=orig_dates)
            oos_signals["DNS_Kalman_Macro"].append(sig_macro_dns)
            raw_signals["DNS_Kalman_Macro"].extend(macro_overlay_input.tolist())
            stage_inputs["DNS_Kalman_Macro"].extend(macro_overlay_input.tolist())
            stage_outputs["DNS_Kalman_Macro"].extend(sig_macro_arr.tolist())
            inherited_dns_clipping["DNS_Kalman_Macro"].extend(diag_kf["is_clipped"].tolist() if isinstance(diag_kf["is_clipped"], np.ndarray) else [diag_kf["is_clipped"]])
            additional_clipping["DNS_Kalman_Macro"].extend((np.abs(macro_overlay_input) > 1.0).tolist())
            saturation_flags["DNS_Kalman_Macro"].extend((np.abs(sig_macro_arr) >= 1.0 - 1e-6).tolist())
            
            for k, (orig_dt, tgt_dt) in enumerate(origin_target_pairs):
                # 1. Unconditionally tag curve forecast as copied from DNS Kalman
                for c_idx, c in enumerate(tenor_cols):
                    ledger.add_record(ForecastRecord(
                        run_id=ledger.run_id,
                        model_id="DNS_Kalman_Macro",
                        fold_id=f_idx,
                        training_cutoff=train_dates[-1],
                        origin_timestamp=orig_dt,
                        target_timestamp=tgt_dt,
                        target_type="yield_curve",
                        target_name=c,
                        forecast=float(y_pred_kf[k, c_idx]),
                        actual=float(y_test.iloc[k, c_idx]),
                        status=ForecastStatus.SCORED,
                        reason="COPIED_DNS_CURVE_FORECAST",
                    ))
                # 2. Record macro position overlay independently
                overlay_val = float(macro_scale[k])
                if overlay_val != 0.0:
                    overlay_reason = "ACTIVE_OVERLAY"
                elif orig_dt in orig_release_dates:
                    overlay_reason = "ZERO_OVERLAY_EVENT"
                else:
                    overlay_reason = "NO_RELEASES"
                ledger.add_record(ForecastRecord(
                    run_id=ledger.run_id,
                    model_id="DNS_Kalman_Macro",
                    fold_id=f_idx,
                    training_cutoff=train_dates[-1],
                    origin_timestamp=orig_dt,
                    target_timestamp=tgt_dt,
                    target_type="macro_position_overlay",
                    target_name="2s10s_overlay",
                    forecast=overlay_val,
                    actual=None,
                    status=ForecastStatus.SCORED,
                    reason=overlay_reason,
                ))
                # 3. For nonzero overlay, record traceable event-level impulse in ledger
                if overlay_val != 0.0:
                    for ev_rec in fold_applied_event_records:
                        if ev_rec["assigned_origin_date"] == str(orig_dt.date()) and ev_rec["status"] == "ACTIVE_OVERLAY":
                            ledger.add_record(ForecastRecord(
                                run_id=ledger.run_id,
                                model_id="DNS_Kalman_Macro",
                                fold_id=f_idx,
                                training_cutoff=train_dates[-1],
                                origin_timestamp=orig_dt,
                                target_timestamp=tgt_dt,
                                target_type="macro_event_impulse",
                                target_name=ev_rec["indicator"],
                                forecast=ev_rec["impulse"],
                                actual=None,
                                status=ForecastStatus.SCORED,
                                reason="ACTIVE_MACRO_IMPULSE",
                            ))

            # --- MODEL 6: GRADIENT BOOSTED MODEL (GBM) ---
            if hasattr(self, "X_ml") and not self.X_ml.empty:
                from src.model.ml_baseline import FactorFeatureEngineer, GradientBoostedFactorForecaster, GBMForecasterConfig
                train_cutoff = train_dates[-1]
                X_tr, y_tr = FactorFeatureEngineer.get_training_slice(self.X_ml, self.y_ml, training_cutoff=train_cutoff)

                if len(X_tr) >= 100:
                    backend = getattr(self.config, "gbm_backend", "sklearn")
                    try:
                        gbm = GradientBoostedFactorForecaster(
                            config=GBMForecasterConfig(backend=backend, n_estimators=30, learning_rate=0.05, max_depth=3)
                        )
                        gbm.fit(X_tr, y_tr)
                        
                        sig_gbm_series = pd.Series(0.0, index=orig_dates)

                        for k, (orig_dt, tgt_dt) in enumerate(origin_target_pairs):
                            if orig_dt in self.X_ml.index:
                                x_row = self.X_ml.loc[[orig_dt]]
                                curr_f = x_row[["level_t0", "slope_t0", "curvature_t0"]]
                                f_hat = gbm.forecast_factors(x_row, curr_f)
                                c_hat = gbm.reconstruct_yield_curve(f_hat, maturities=maturities).values[0]

                                # Actual observation at tgt_dt
                                actual_y = y_test.iloc[k].values
                                curve_sq_errors["GBM"].extend(((actual_y - c_hat) ** 2).tolist())
                                for c_idx, c in enumerate(tenor_cols):
                                    per_tenor_sq_errors["GBM"][c].append(float((actual_y[c_idx] - c_hat[c_idx]) ** 2))

                                # Observable spreads
                                c_hat_2d = c_hat.reshape(1, -1)
                                actual_y_2d = actual_y.reshape(1, -1)
                                sp_pred = compute_observable_spreads(c_hat_2d, tenor_cols)
                                sp_act = compute_observable_spreads(actual_y_2d, tenor_cols)
                                if "2s10s" in sp_pred and "2s10s" in sp_act:
                                    diff_2s10s = float((sp_act["2s10s"] - sp_pred["2s10s"]).item())
                                    spread_2s10s_sq_errors["GBM"].append(diff_2s10s ** 2)
                                    factor_sq_errors["GBM"].append(diff_2s10s ** 2)
                                if "2s5s10s" in sp_pred and "2s5s10s" in sp_act:
                                    diff_fly = float((sp_act["2s5s10s"] - sp_pred["2s5s10s"]).item())
                                    fly_2s5s10s_sq_errors["GBM"].append(diff_fly ** 2)

                                # Signal formed at orig_dt for horizon orig_dt -> tgt_dt
                                sig_gbm_val, diag_gbm = map_curve_forecast_to_spread_signal(
                                    c_hat, y_origin_all[k], tenor_cols, train_spread_std, strategy_type, return_details=True
                                )
                                sig_gbm_series.loc[orig_dt] = float(sig_gbm_val)
                                raw_gbm_val = float(diag_gbm["raw_input"])
                                raw_signals["GBM"].append(raw_gbm_val)
                                stage_inputs["GBM"].append(raw_gbm_val)
                                stage_outputs["GBM"].append(float(sig_gbm_val))
                                inherited_dns_clipping["GBM"].append(False)
                                additional_clipping["GBM"].append(bool(diag_gbm["is_clipped"]))
                                saturation_flags["GBM"].append(bool(diag_gbm["is_saturated"]))

                                for c_idx, c in enumerate(tenor_cols):
                                    ledger.add_record(ForecastRecord(
                                        run_id=ledger.run_id,
                                        model_id="GBM",
                                        fold_id=f_idx,
                                        training_cutoff=train_cutoff,
                                        origin_timestamp=orig_dt,
                                        target_timestamp=tgt_dt,
                                        target_type="yield_curve",
                                        target_name=c,
                                        forecast=float(c_hat[c_idx]),
                                        actual=float(actual_y[c_idx]),
                                        status=ForecastStatus.SCORED,
                                    ))
                            else:
                                raw_signals["GBM"].append(0.0)
                                stage_inputs["GBM"].append(0.0)
                                stage_outputs["GBM"].append(0.0)
                                inherited_dns_clipping["GBM"].append(False)
                                additional_clipping["GBM"].append(False)
                                saturation_flags["GBM"].append(False)
                                for c in tenor_cols:
                                    ledger.add_unavailable(
                                        model_id="GBM",
                                        fold_id=f_idx,
                                        training_cutoff=train_cutoff,
                                        origin_timestamp=orig_dt,
                                        target_timestamp=tgt_dt,
                                        target_type="yield_curve",
                                        target_name=c,
                                        reason=f"Features missing for origin date {orig_dt} (Prompt 4)",
                                    )

                        oos_signals["GBM"].append(sig_gbm_series)
                    except Exception as e:
                        logger.warning("GBM estimation failed on fold %d: %s", f_idx, e)
                        for orig_dt, tgt_dt in origin_target_pairs:
                            for c in tenor_cols:
                                ledger.add_unavailable(
                                    model_id="GBM",
                                    fold_id=f_idx,
                                    training_cutoff=train_cutoff,
                                    origin_timestamp=orig_dt,
                                    target_timestamp=tgt_dt,
                                    target_type="yield_curve",
                                    target_name=c,
                                    reason=f"GBM fitting failed: {e}",
                                )
                        oos_signals["GBM"].append(pd.Series(0.0, index=orig_dates))
                        raw_signals["GBM"].extend([0.0] * len(orig_dates))
                        stage_inputs["GBM"].extend([0.0] * len(orig_dates))
                        stage_outputs["GBM"].extend([0.0] * len(orig_dates))
                        inherited_dns_clipping["GBM"].extend([False] * len(orig_dates))
                        additional_clipping["GBM"].extend([False] * len(orig_dates))
                        saturation_flags["GBM"].extend([False] * len(orig_dates))
                else:
                    for orig_dt, tgt_dt in origin_target_pairs:
                        for c in tenor_cols:
                            ledger.add_unavailable(
                                model_id="GBM",
                                fold_id=f_idx,
                                training_cutoff=train_cutoff,
                                origin_timestamp=orig_dt,
                                target_timestamp=tgt_dt,
                                target_type="yield_curve",
                                target_name=c,
                                reason="Insufficient training observations (<100) for GBM fit (Prompt 4)",
                            )
                    oos_signals["GBM"].append(pd.Series(0.0, index=orig_dates))
                    raw_signals["GBM"].extend([0.0] * len(orig_dates))
                    stage_inputs["GBM"].extend([0.0] * len(orig_dates))
                    stage_outputs["GBM"].extend([0.0] * len(orig_dates))
                    inherited_dns_clipping["GBM"].extend([False] * len(orig_dates))
                    additional_clipping["GBM"].extend([False] * len(orig_dates))
                    saturation_flags["GBM"].extend([False] * len(orig_dates))
            else:
                for orig_dt, tgt_dt in origin_target_pairs:
                    for c in tenor_cols:
                        ledger.add_unavailable(
                            model_id="GBM",
                            fold_id=f_idx,
                            training_cutoff=train_dates[-1],
                            origin_timestamp=orig_dt,
                            target_timestamp=tgt_dt,
                            target_type="yield_curve",
                            target_name=c,
                            reason="ML feature panel unavailable (Prompt 4)",
                        )
                oos_signals["GBM"].append(pd.Series(0.0, index=orig_dates))
                raw_signals["GBM"].extend([0.0] * len(orig_dates))
                stage_inputs["GBM"].extend([0.0] * len(orig_dates))
                stage_outputs["GBM"].extend([0.0] * len(orig_dates))
                inherited_dns_clipping["GBM"].extend([False] * len(orig_dates))
                additional_clipping["GBM"].extend([False] * len(orig_dates))
                saturation_flags["GBM"].extend([False] * len(orig_dates))

        # Concatenate out-of-sample series and execute backtests across all evaluated and diagnostic models
        all_test_dates = [dt for _, test_dates in folds for dt in test_dates]
        last_tgt_date = folds[-1][1][-1]
        backtest_results = {}
        all_model_metrics = {}
        
        for m in all_eval_models:
            full_sig = pd.concat(oos_signals[m])
            pos_df = compute_continuous_positions(
                signals_series=full_sig,
                strategy_type=strategy_type,
                target_dv01=self.config.target_dv01,
            )
            
            # Horizon alignment: position established at origin t is held overnight into target t+1.
            # Append final flat row on last target date to close out the last overnight holding period.
            if last_tgt_date not in pos_df.index:
                flat_row = pd.DataFrame(
                    [{c: 0 for c in pos_df.columns}],
                    index=[last_tgt_date],
                )
                pos_df = pd.concat([pos_df, flat_row])
                
            b_res = self.engine.run_strategy(
                strategy_name=m,
                positions_df=pos_df,
                yield_df=self.yield_df,
            )
            backtest_results[m] = b_res
            met = b_res.metrics
            
            # Compute RMSE in basis points (1 bp = 0.01 percentage point)
            c_rmse_bp = float(np.sqrt(np.mean(curve_sq_errors[m])) * 100.0) if len(curve_sq_errors[m]) > 0 else np.nan
            f_rmse_bp = float(np.sqrt(np.mean(factor_sq_errors[m])) * 100.0) if len(factor_sq_errors[m]) > 0 else np.nan
            s_rmse_bp = float(np.sqrt(np.mean(spread_2s10s_sq_errors[m])) * 100.0) if len(spread_2s10s_sq_errors[m]) > 0 else np.nan
            fly_rmse_bp = float(np.sqrt(np.mean(fly_2s5s10s_sq_errors[m])) * 100.0) if len(fly_2s5s10s_sq_errors[m]) > 0 else np.nan
            
            rmse_2y = float(np.sqrt(np.mean(per_tenor_sq_errors[m]["DGS2"])) * 100.0) if "DGS2" in per_tenor_sq_errors[m] and len(per_tenor_sq_errors[m]["DGS2"]) > 0 else np.nan
            rmse_5y = float(np.sqrt(np.mean(per_tenor_sq_errors[m]["DGS5"])) * 100.0) if "DGS5" in per_tenor_sq_errors[m] and len(per_tenor_sq_errors[m]["DGS5"]) > 0 else np.nan
            rmse_10y = float(np.sqrt(np.mean(per_tenor_sq_errors[m]["DGS10"])) * 100.0) if "DGS10" in per_tenor_sq_errors[m] and len(per_tenor_sq_errors[m]["DGS10"]) > 0 else np.nan

            total_contracts = met.get("contract_turnover_lots", 0)
            pnl_turnover = met.get("pnl_turnover_usd_per_lot", np.nan)
            pnl_dv01 = round(met.get("total_net_trading_pnl_usd", 0.0) / self.config.target_dv01, 2)
            
            all_model_metrics[m] = {
                "Model / Forecast Method": m.replace("_", " + "),
                "Forecast Status": "EVALUATED" if not np.isnan(c_rmse_bp) else "UNAVAILABLE",
                "Sample Size (Days)": len(full_sig),
                "OOS Curve RMSE (bp)": round(c_rmse_bp, 2) if not np.isnan(c_rmse_bp) else np.nan,
                "2Y Curve RMSE (bp)": round(rmse_2y, 2) if not np.isnan(rmse_2y) else np.nan,
                "5Y Curve RMSE (bp)": round(rmse_5y, 2) if not np.isnan(rmse_5y) else np.nan,
                "10Y Curve RMSE (bp)": round(rmse_10y, 2) if not np.isnan(rmse_10y) else np.nan,
                "2s10s Spread RMSE (bp)": round(s_rmse_bp, 2) if not np.isnan(s_rmse_bp) else np.nan,
                "2s5s10s Fly RMSE (bp)": round(fly_rmse_bp, 2) if not np.isnan(fly_rmse_bp) else np.nan,
                "Factor Forecast RMSE (bp)": round(f_rmse_bp, 2) if not np.isnan(f_rmse_bp) else np.nan,
                "Strategy Sharpe": met.get("sharpe_ratio", np.nan),
                "Sortino Ratio": met.get("sortino_ratio", np.nan),
                "Max Drawdown (%)": met.get("max_drawdown_pct", 0.0),
                "Annual Turnover (lots)": round(total_contracts / (len(full_sig)/252.0), 1),
                "Hit Rate (%)": met.get("win_rate_pct", np.nan),
                "PnL / DV01 ($)": pnl_dv01,
                "PnL / Turnover ($/lot)": pnl_turnover,
                "Gross PnL ($)": met.get("total_gross_pnl_usd", 0.0),
                "Trade Costs ($)": met.get("total_trade_cost_usd", 0.0),
                "Roll Costs ($)": met.get("total_roll_cost_usd", 0.0),
                "Trading Net PnL ($)": met.get("total_net_trading_pnl_usd", 0.0),
                "Cash Interest ($)": met.get("total_interest_earned_usd", 0.0),
                "Collateral Net PnL ($)": met.get("total_collateral_pnl_usd", met.get("total_net_pnl_usd", 0.0)),
            }
            
        # Macro audit statistics across all evaluated folds
        total_test_macro_events = sum(f.get("test_event_count", 0) for f in fold_macro_audit)
        total_calendar_events = sum(f.get("calendar_event_count", 0) for f in fold_macro_audit)
        total_train_macro_events = sum(f.get("training_event_count", 0) for f in fold_macro_audit)
        total_observed_decision_events = sum(f.get("observed_decision_events", 0) for f in fold_macro_audit)
        total_evaluated_decision_events = sum(f.get("evaluated_decision_events", 0) for f in fold_macro_audit)
        total_timestamp_verified_available_events = sum(f.get("timestamp_verified_available_events", 0) for f in fold_macro_audit)
        total_legacy_date_only_assumed_events = sum(f.get("legacy_date_only_assumed_events", 0) for f in fold_macro_audit)
        total_timestamp_available_events = sum(f.get("timestamp_available_events", 0) for f in fold_macro_audit)
        total_post_close_events = sum(f.get("post_close_events", 0) for f in fold_macro_audit)
        total_rolled_to_next_decision_events = sum(f.get("rolled_to_next_decision_events", 0) for f in fold_macro_audit)
        total_invalid_timestamp_rejected_events = len(unassigned_invalid_timestamp_events) + sum(f.get("invalid_timestamp_rejected_events", 0) for f in fold_macro_audit)
        total_missing_surprise_events = sum(f.get("missing_surprise_events", 0) for f in fold_macro_audit)
        total_usable_surprise_events = sum(f.get("usable_surprise_events", 0) for f in fold_macro_audit)
        total_zero_surprise_events = sum(f.get("zero_surprise_events", 0) for f in fold_macro_audit)
        total_eligible_coefficient_events = sum(f.get("eligible_coefficient_events", 0) for f in fold_macro_audit)
        total_inadequate_history_events = sum(f.get("inadequate_history_events", 0) for f in fold_macro_audit)
        total_active_overlay_events = sum(f.get("active_overlay_events", 0) for f in fold_macro_audit)
        total_zero_impact_events = sum(f.get("zero_impact_events", 0) for f in fold_macro_audit)
        total_nonzero_macro_days = sum(f.get("nonzero_macro_days", 0) for f in fold_macro_audit)
        total_applied_event_records = []
        for f in fold_macro_audit:
            total_applied_event_records.extend(f.get("applied_event_records", []))

        if total_evaluated_decision_events == 0:
            macro_eval_status = "NOT_EVALUATED (NO_TEST_RELEASES)"
        elif total_nonzero_macro_days == 0:
            macro_eval_status = "INACTIVE_OVERLAY (ZERO_SURPRISE_OR_INADEQUATE_HISTORY)"
        else:
            macro_eval_status = "VALID_MACRO_TEST"

        # Synchronize macro evaluation status across baseline_table, common_sample_table, and ledger
        if "DNS_Kalman_Macro" in all_model_metrics:
            if total_evaluated_decision_events == 0:
                all_model_metrics["DNS_Kalman_Macro"]["Forecast Status"] = "NOT_EVALUATED (NO_TEST_RELEASES)"
            elif total_nonzero_macro_days == 0:
                all_model_metrics["DNS_Kalman_Macro"]["Forecast Status"] = "INACTIVE_OVERLAY (ZERO_SURPRISE_OR_INADEQUATE_HISTORY)"

        # Build baseline_table strictly for the 6 primary evaluated models
        baseline_rows = [all_model_metrics[m] for m in models]
        baseline_table = pd.DataFrame(baseline_rows).set_index("Model / Forecast Method")

        # Build common_sample_table containing evaluated models, risk control, diagnostics, and Cash Only
        ROLE_MAP = {
            "Random_Walk": ("Random Walk (Curve Benchmark)", "Curve Benchmark"),
            "PCA_VAR": ("PCA / VAR(1)", "Term Structure Factor Model"),
            "Static_NS": ("AR(1) Baseline (Static NS)", "AR(1) Baseline"),
            "DNS_Kalman": ("DNS + Kalman", "Dynamic Term Structure Model"),
            "DNS_Kalman_Macro": ("DNS + Kalman + Macro", "Macro-Augmented DTSM"),
            "GBM": ("GBM", "Machine Learning Baseline"),
            "DNS_Scaled_60": ("DNS (60% Exposure Control)", "Risk-Scaling Control (60% Exposure)"),
            "Static_NS_Residual_Preserving": ("Static NS (Residual-Preserving Diagnostic)", "Diagnostic (Static NS Residual-Preserving)"),
            "DNS_Kalman_Residual_Preserving": ("DNS + Kalman (Residual-Preserving Diagnostic)", "Diagnostic (DNS Kalman Residual-Preserving)"),
        }

        common_sample_rows = []
        ordered_models = models + extra_models
        for m in ordered_models:
            orig_row = all_model_metrics[m]
            display_name, role = ROLE_MAP.get(m, (orig_row["Model / Forecast Method"], "Model Baseline"))
            
            # Audit status: separate curve provenance from strategy overlay activity
            f_status = orig_row["Forecast Status"]
            if m == "DNS_Kalman_Macro":
                if total_evaluated_decision_events == 0:
                    f_status = "NOT_EVALUATED (NO_TEST_RELEASES)"
                elif total_nonzero_macro_days == 0:
                    f_status = "INACTIVE_OVERLAY (ZERO_SURPRISE_OR_INADEQUATE_HISTORY)"
            elif m == "DNS_Scaled_60":
                f_status = "CONTROL (SCALED_DNS)"
            elif "Residual_Preserving" in m:
                f_status = "DIAGNOSTIC"

            row_copy = {
                "Model / Forecast Method": display_name,
                "Benchmark Role": role,
                "Forecast Status": f_status,
                "Sample Size (Days)": orig_row["Sample Size (Days)"],
                "OOS Curve RMSE (bp)": orig_row["OOS Curve RMSE (bp)"],
                "2Y Curve RMSE (bp)": orig_row["2Y Curve RMSE (bp)"],
                "5Y Curve RMSE (bp)": orig_row["5Y Curve RMSE (bp)"],
                "10Y Curve RMSE (bp)": orig_row["10Y Curve RMSE (bp)"],
                "2s10s Spread RMSE (bp)": orig_row["2s10s Spread RMSE (bp)"],
                "2s5s10s Fly RMSE (bp)": orig_row["2s5s10s Fly RMSE (bp)"],
                "Factor Forecast RMSE (bp)": orig_row["Factor Forecast RMSE (bp)"],
                "Strategy Sharpe": orig_row["Strategy Sharpe"],
                "Sortino Ratio": orig_row["Sortino Ratio"],
                "Max Drawdown (%)": orig_row["Max Drawdown (%)"],
                "Annual Turnover (lots)": orig_row["Annual Turnover (lots)"],
                "Hit Rate (%)": orig_row["Hit Rate (%)"],
                "PnL / DV01 ($)": orig_row["PnL / DV01 ($)"],
                "PnL / Turnover ($/lot)": orig_row["PnL / Turnover ($/lot)"],
                "Gross PnL ($)": orig_row["Gross PnL ($)"],
                "Trade Costs ($)": orig_row["Trade Costs ($)"],
                "Roll Costs ($)": orig_row["Roll Costs ($)"],
                "Trading Net PnL ($)": orig_row["Trading Net PnL ($)"],
                "Cash Interest ($)": orig_row["Cash Interest ($)"],
                "Collateral Net PnL ($)": orig_row["Collateral Net PnL ($)"],
            }
            common_sample_rows.append(row_copy)

        # Cash-Only benchmark
        first_orig_date = folds[0][0][-1]
        cash_pos_df = pd.DataFrame(
            0,
            index=[first_orig_date] + list(all_test_dates),
            columns=[f"n_{s.lower()}" for s in ["zt", "zf", "zn"]],
        )
        b_cash = self.engine.run_strategy(
            strategy_name="Cash_Only",
            positions_df=cash_pos_df,
            yield_df=self.yield_df,
        )
        backtest_results["Cash_Only"] = b_cash
        met_cash = b_cash.metrics

        common_sample_rows.append({
            "Model / Forecast Method": "Cash Only (Strategy Benchmark)",
            "Benchmark Role": "Strategy Benchmark",
            "Forecast Status": "BENCHMARK_ONLY",
            "Sample Size (Days)": len(all_test_dates),
            "OOS Curve RMSE (bp)": np.nan,
            "2Y Curve RMSE (bp)": np.nan,
            "5Y Curve RMSE (bp)": np.nan,
            "10Y Curve RMSE (bp)": np.nan,
            "2s10s Spread RMSE (bp)": np.nan,
            "2s5s10s Fly RMSE (bp)": np.nan,
            "Factor Forecast RMSE (bp)": np.nan,
            "Strategy Sharpe": np.nan,
            "Sortino Ratio": np.nan,
            "Max Drawdown (%)": 0.0,
            "Annual Turnover (lots)": 0.0,
            "Hit Rate (%)": np.nan,
            "PnL / DV01 ($)": 0.0,
            "PnL / Turnover ($/lot)": np.nan,
            "Gross PnL ($)": 0.0,
            "Trade Costs ($)": 0.0,
            "Roll Costs ($)": 0.0,
            "Trading Net PnL ($)": 0.0,
            "Cash Interest ($)": met_cash.get("total_interest_earned_usd", 0.0),
            "Collateral Net PnL ($)": met_cash.get("total_collateral_pnl_usd", 0.0),
        })
        common_sample_table = pd.DataFrame(common_sample_rows).set_index("Model / Forecast Method")

        # Econometric diagnostics computation
        ns_fit_rmse_bp = float(np.sqrt(np.mean(contemp_fit_sq_errors)) * 100.0) if contemp_fit_sq_errors else np.nan
        factor_rw_rmse_bp = float(np.sqrt(np.mean(factor_rw_sq_errors)) * 100.0) if factor_rw_sq_errors else np.nan
        factor_ar1_rmse_bp = float(np.sqrt(np.mean(factor_ar1_sq_errors)) * 100.0) if factor_ar1_sq_errors else np.nan

        # Exact observable 2s10s spread decomposition: e_s = u_factor + u_fit
        if spread_decomp_total_sq:
            tot_mse = float(np.mean(spread_decomp_total_sq))
            fac_mse = float(np.mean(spread_decomp_factor_sq))
            fit_mse = float(np.mean(spread_decomp_fit_sq))
            cross_term = float(np.mean(spread_decomp_cross))
            tot_rmse = float(np.sqrt(tot_mse))
            fac_rmse = float(np.sqrt(fac_mse))
            fit_rmse = float(np.sqrt(fit_mse))
            decomp_dict = {
                "total_spread_rmse_bp": round(tot_rmse, 2),
                "total_spread_mse_bp2": round(tot_mse, 2),
                "factor_dynamics_spread_rmse_bp": round(fac_rmse, 2),
                "factor_dynamics_spread_mse_bp2": round(fac_mse, 2),
                "cross_sectional_fit_spread_rmse_bp": round(fit_rmse, 2),
                "cross_sectional_fit_spread_mse_bp2": round(fit_mse, 2),
                "uncentered_cross_moment_bp2": round(cross_term, 2),
                "cross_term_cov_bp2": round(cross_term, 2),
                "sum_components_mse_bp2": round(fac_mse + fit_mse + cross_term, 2),
                "identity_holds": abs(tot_mse - (fac_mse + fit_mse + cross_term)) < 1e-4,
            }
        else:
            decomp_dict = {}

        raw_threshold_exceedance_stats = {}
        inherited_clipping_stats = {}
        additional_clipping_stats = {}
        final_saturation_stats = {}
        mean_unclipped_stats = {}

        for m in all_eval_models:
            inp_arr = np.array(stage_inputs[m])
            if len(inp_arr) > 0:
                raw_threshold_exceedance_stats[m] = round(float(np.mean(np.abs(inp_arr) >= 1.0 - 1e-9) * 100.0), 2)
                inherited_clipping_stats[m] = round(float(np.mean(inherited_dns_clipping[m]) * 100.0), 2)
                additional_clipping_stats[m] = round(float(np.mean(additional_clipping[m]) * 100.0), 2)
                final_saturation_stats[m] = round(float(np.mean(saturation_flags[m]) * 100.0), 2)
                mean_unclipped_stats[m] = round(float(np.mean(np.abs(inp_arr))), 3)
            else:
                raw_threshold_exceedance_stats[m] = np.nan
                inherited_clipping_stats[m] = np.nan
                additional_clipping_stats[m] = np.nan
                final_saturation_stats[m] = np.nan
                mean_unclipped_stats[m] = np.nan

        sig_dict = {m: pd.concat(oos_signals[m]).values for m in all_eval_models if len(oos_signals[m]) > 0}
        sig_df = pd.DataFrame(sig_dict)
        sig_corr_matrix = sig_df.corr().round(4).to_dict() if not sig_df.empty else {}

        git_prov = get_git_provenance()
        run_metadata = {
            "git_commit": git_prov["commit_or_dirty"],
            "git_provenance": git_prov,
            "data_checksums": {
                "yield_panel": get_file_checksum("data/processed/yield_panel.parquet"),
                "factor_panel": get_file_checksum("data/processed/factor_panel.parquet"),
                "macro_surprises": get_file_checksum("data/processed/macro_surprises.parquet"),
            },
            "eval_start_date": str(all_test_dates[0].date()) if len(all_test_dates) > 0 else "N/A",
            "eval_end_date": str(all_test_dates[-1].date()) if len(all_test_dates) > 0 else "N/A",
            "total_eval_days": len(all_test_dates),
            "fold_count": len(folds),
            "run_mode": "QUICK_TWO_FOLD_EVALUATION" if len(folds) <= 2 else "FULL_SAMPLE_EVALUATION",
            "tenor_panel": tenor_cols,
            "units": {
                "curve_rmse": "basis points (0.01%)",
                "spread_rmse": "basis points (0.01%)",
                "strategy_returns": "percent per annum",
                "turnover": "contract lots",
                "pnl": "USD ($)",
            },
            "seed": getattr(self.config, "random_state", 42),
            "gbm_backend": getattr(self.config, "gbm_backend", "sklearn"),
            "ledger_summary": ledger.summary_by_model(),
            "macro_event_audit": {
                "macro_data_coverage_end": macro_coverage_end_str,
                "total_training_events": total_train_macro_events,
                "total_calendar_events": total_calendar_events,
                "total_test_events": total_calendar_events,
                "total_observed_decision_events": total_observed_decision_events,
                "total_evaluated_decision_events": total_evaluated_decision_events,
                "total_timestamp_verified_available_events": total_timestamp_verified_available_events,
                "total_legacy_date_only_assumed_events": total_legacy_date_only_assumed_events,
                "total_timestamp_available_events": total_timestamp_available_events,
                "total_post_close_events": total_post_close_events,
                "total_rolled_to_next_decision_events": total_rolled_to_next_decision_events,
                "total_invalid_timestamp_rejected_events": total_invalid_timestamp_rejected_events,
                "total_events_with_no_subsequent_decision": len(events_with_no_subsequent_decision),
                "events_with_no_subsequent_decision": events_with_no_subsequent_decision,
                "total_missing_surprise_events": total_missing_surprise_events,
                "total_usable_surprise_events": total_usable_surprise_events,
                "total_zero_surprise_events": total_zero_surprise_events,
                "total_eligible_coefficient_events": total_eligible_coefficient_events,
                "total_inadequate_history_events": total_inadequate_history_events,
                "total_active_overlay_events": total_active_overlay_events,
                "total_zero_impact_events": total_zero_impact_events,
                "nonzero_macro_days": total_nonzero_macro_days,
                "applied_event_records": total_applied_event_records,
                "curve_forecast_provenance": "COPIED_DNS_CURVE_FORECAST",
                "strategy_overlay_status": "ACTIVE_OVERLAY" if total_nonzero_macro_days > 0 else (
                    "INACTIVE_ZERO_RELEASES" if total_evaluated_decision_events == 0 else "INACTIVE_ZERO_SURPRISE_OR_INADEQUATE_HISTORY"
                ),
                "evaluation_status": macro_eval_status,
                "fold_breakdown": fold_macro_audit,
            },
            "econometric_diagnostics": {
                "ns_contemporaneous_fit_rmse_bp": round(ns_fit_rmse_bp, 2),
                "factor_coordinate_rmse_ar1_bp": round(factor_ar1_rmse_bp, 2),
                "factor_coordinate_rmse_random_walk_bp": round(factor_rw_rmse_bp, 2),
                "factor_rmse_ar1_bp": round(factor_ar1_rmse_bp, 2),
                "factor_rmse_random_walk_bp": round(factor_rw_rmse_bp, 2),
                "observable_2s10s_spread_decomposition": decomp_dict,
                "raw_signal_threshold_exceedance_pct": raw_threshold_exceedance_stats,
                "inherited_dns_clipping_pct": inherited_clipping_stats,
                "additional_clipping_frequency_pct": additional_clipping_stats,
                "actual_clipping_frequency_pct": additional_clipping_stats,
                "signal_clipping_frequency_pct": additional_clipping_stats,
                "final_position_saturation_pct": final_saturation_stats,
                "saturation_bounds": {
                    "DNS_Scaled_60": 0.60,
                    "DNS_Kalman_Macro": 1.00,
                    "Random_Walk": 1.00,
                    "PCA_VAR": 1.00,
                    "Static_NS": 1.00,
                    "DNS_Kalman": 1.00,
                    "Static_NS_Residual_Preserving": 1.00,
                    "DNS_Kalman_Residual_Preserving": 1.00,
                    "GBM": 1.00,
                },
                "mean_unclipped_signal_std": mean_unclipped_stats,
                "signal_correlations": sig_corr_matrix,
                "residual_preserving_comparison": {
                    "traditional_ns_spread_rmse_bp": all_model_metrics["Static_NS"]["2s10s Spread RMSE (bp)"],
                    "residual_preserving_ns_spread_rmse_bp": all_model_metrics["Static_NS_Residual_Preserving"]["2s10s Spread RMSE (bp)"],
                    "traditional_dns_spread_rmse_bp": all_model_metrics["DNS_Kalman"]["2s10s Spread RMSE (bp)"],
                    "residual_preserving_dns_spread_rmse_bp": all_model_metrics["DNS_Kalman_Residual_Preserving"]["2s10s Spread RMSE (bp)"],
                }
            }
        }
        
        return {
            "baseline_table": baseline_table,
            "common_sample_table": common_sample_table,
            "backtest_results": backtest_results,
            "signals": oos_signals,
            "forecast_ledger": ledger,
            "run_metadata": run_metadata,
        }
