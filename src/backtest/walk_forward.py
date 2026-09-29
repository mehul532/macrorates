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
import hashlib
import logging
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
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


def is_timestamp_available_for_decision(row: Any, dt: pd.Timestamp) -> bool:
    """
    Check if macro release was available prior to the decision at market close on date dt.
    Market close cutoff: 16:00 ET / 21:00 UTC (daylight saving) or 22:00 UTC (standard).
    If timestamp is missing, naive date, or has 00:00 time, defaults to standard
    pre-close daytime release assumption (e.g. 8:30 AM / 10:00 AM ET).
    """
    if "timestamp" not in row or pd.isna(row["timestamp"]):
        return True
    ts = row["timestamp"]
    if isinstance(ts, pd.Timestamp):
        if ts.time() == pd.Timestamp("00:00:00").time():
            return True
        if ts.tz is not None:
            # UTC timestamp: after 21:00 UTC (16:00 ET / 17:00 ET) is post-close
            if ts.date() == dt.date():
                return ts.hour < 21 or (ts.hour == 21 and ts.minute == 0 and ts.second == 0)
            elif ts.date() < dt.date():
                return True
            else:
                return False
        else:
            # Naive local timestamp
            if ts.date() == dt.date():
                return ts.hour < 16 or (ts.hour == 16 and ts.minute == 0 and ts.second == 0)
            elif ts.date() < dt.date():
                return True
            else:
                return False
    return True


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
                "evaluated_decision_events": 0,
                "timestamp_available_events": 0,
                "post_close_events": 0,
                "usable_surprise_events": 0,
                "zero_surprise_events": 0,
                "eligible_coefficient_events": 0,
                "inadequate_history_events": 0,
                "nonzero_macro_days": 0,
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
            sig_pca_arr = map_curve_forecast_to_spread_signal(
                y_pred_pca, y_origin_all, tenor_cols, train_spread_std, strategy_type
            )
            sig_pca = pd.Series(sig_pca_arr, index=orig_dates)
            oos_signals["PCA_VAR"].append(sig_pca)
            sp_curr = compute_observable_spreads(y_origin_all, tenor_cols)["2s10s"]
            raw_pca = (spreads_pca["2s10s"] - sp_curr) / train_spread_std
            raw_signals["PCA_VAR"].extend(raw_pca.tolist())
            stage_inputs["PCA_VAR"].extend(raw_pca.tolist())
            stage_outputs["PCA_VAR"].extend(sig_pca_arr.tolist())
            inherited_dns_clipping["PCA_VAR"].extend([False] * len(orig_dates))
            additional_clipping["PCA_VAR"].extend((np.abs(raw_pca) > 1.0).tolist())
            saturation_flags["PCA_VAR"].extend((np.abs(sig_pca_arr) >= 1.0 - 1e-6).tolist())
            
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
            sig_ns_arr = map_curve_forecast_to_spread_signal(
                y_pred_ns, y_origin_all, tenor_cols, train_spread_std, strategy_type
            )
            sig_ns = pd.Series(sig_ns_arr, index=orig_dates)
            oos_signals["Static_NS"].append(sig_ns)
            raw_ns = (spreads_ns["2s10s"] - sp_curr) / train_spread_std
            raw_signals["Static_NS"].extend(raw_ns.tolist())
            stage_inputs["Static_NS"].extend(raw_ns.tolist())
            stage_outputs["Static_NS"].extend(sig_ns_arr.tolist())
            inherited_dns_clipping["Static_NS"].extend([False] * len(orig_dates))
            additional_clipping["Static_NS"].extend((np.abs(raw_ns) > 1.0).tolist())
            saturation_flags["Static_NS"].extend((np.abs(sig_ns_arr) >= 1.0 - 1e-6).tolist())
            
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
            sig_ns_res_arr = map_curve_forecast_to_spread_signal(
                y_pred_ns_res, y_origin_all, tenor_cols, train_spread_std, strategy_type
            )
            raw_ns_res = (spreads_ns_res["2s10s"] - sp_curr) / train_spread_std
            raw_signals["Static_NS_Residual_Preserving"].extend(raw_ns_res.tolist())
            stage_inputs["Static_NS_Residual_Preserving"].extend(raw_ns_res.tolist())
            stage_outputs["Static_NS_Residual_Preserving"].extend(sig_ns_res_arr.tolist())
            inherited_dns_clipping["Static_NS_Residual_Preserving"].extend([False] * len(orig_dates))
            additional_clipping["Static_NS_Residual_Preserving"].extend((np.abs(raw_ns_res) > 1.0).tolist())
            saturation_flags["Static_NS_Residual_Preserving"].extend((np.abs(sig_ns_res_arr) >= 1.0 - 1e-6).tolist())
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
            sig_kf_arr = map_curve_forecast_to_spread_signal(
                y_pred_kf, y_origin_all, tenor_cols, train_spread_std, strategy_type
            )
            sig_kf = pd.Series(sig_kf_arr, index=orig_dates)
            oos_signals["DNS_Kalman"].append(sig_kf)
            raw_kf = (spreads_kf["2s10s"] - sp_curr) / train_spread_std
            raw_signals["DNS_Kalman"].extend(raw_kf.tolist())
            stage_inputs["DNS_Kalman"].extend(raw_kf.tolist())
            stage_outputs["DNS_Kalman"].extend(sig_kf_arr.tolist())
            inherited_dns_clipping["DNS_Kalman"].extend([False] * len(orig_dates))
            additional_clipping["DNS_Kalman"].extend((np.abs(raw_kf) > 1.0).tolist())
            saturation_flags["DNS_Kalman"].extend((np.abs(sig_kf_arr) >= 1.0 - 1e-6).tolist())
            
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
            sig_kf_res_arr = map_curve_forecast_to_spread_signal(
                y_pred_kf_res, y_origin_all, tenor_cols, train_spread_std, strategy_type
            )
            raw_kf_res = (spreads_kf_res["2s10s"] - sp_curr) / train_spread_std
            raw_signals["DNS_Kalman_Residual_Preserving"].extend(raw_kf_res.tolist())
            stage_inputs["DNS_Kalman_Residual_Preserving"].extend(raw_kf_res.tolist())
            stage_outputs["DNS_Kalman_Residual_Preserving"].extend(sig_kf_res_arr.tolist())
            inherited_dns_clipping["DNS_Kalman_Residual_Preserving"].extend([False] * len(orig_dates))
            additional_clipping["DNS_Kalman_Residual_Preserving"].extend((np.abs(raw_kf_res) > 1.0).tolist())
            saturation_flags["DNS_Kalman_Residual_Preserving"].extend((np.abs(sig_kf_res_arr) >= 1.0 - 1e-6).tolist())
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
            inherited_dns_clipping["DNS_Scaled_60"].extend((np.abs(raw_kf) > 1.0).tolist())
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
            
            # Join macro releases by availability at decision origin (orig_dates)
            # Releases on orig_dt (e.g. 8:30 AM / 14:00) are available at orig_dt close; target-date releases are future.
            if self.macro_df is not None and not self.macro_df.empty and "date" in self.macro_df.columns:
                macro_sub = self.macro_df[self.macro_df["date"].isin(orig_dates)]
            else:
                macro_sub = pd.DataFrame()
            macro_impulse = pd.Series(0.0, index=orig_dates)
            
            eval_dec_count = 0
            ts_avail_count = 0
            post_close_count = 0
            usable_surp_count = 0
            zero_surp_count = 0
            eligible_coeff_count = 0
            inadequate_hist_count = 0
            orig_release_dates = set()

            for _, m_row in macro_sub.iterrows():
                dt = m_row["date"]
                ind = m_row.get("indicator")
                surp = m_row.get("surprise_ann", np.nan)
                
                eval_dec_count += 1
                orig_release_dates.add(dt)
                
                if not is_timestamp_available_for_decision(m_row, dt):
                    post_close_count += 1
                    continue
                ts_avail_count += 1
                
                if pd.isna(surp):
                    continue
                usable_surp_count += 1
                
                if surp == 0.0:
                    zero_surp_count += 1
                    
                if ind not in fold_macro_betas:
                    inadequate_hist_count += 1
                    continue
                eligible_coeff_count += 1
                
                b_info = fold_macro_betas[ind]
                if strategy_type in ("2s5s10s", "fly"):
                    b_target = b_info.get("curvature_beta", 0.0)
                else:
                    b_target = b_info.get("slope_beta", 0.0)
                macro_impulse.loc[dt] += b_target * np.clip(surp, -2.0, 2.0)
            
            nonzero_macro_days = int((macro_impulse != 0.0).sum())
            fold_macro_audit[-1]["evaluated_decision_events"] = eval_dec_count
            fold_macro_audit[-1]["timestamp_available_events"] = ts_avail_count
            fold_macro_audit[-1]["post_close_events"] = post_close_count
            fold_macro_audit[-1]["usable_surprise_events"] = usable_surp_count
            fold_macro_audit[-1]["zero_surprise_events"] = zero_surp_count
            fold_macro_audit[-1]["eligible_coefficient_events"] = eligible_coeff_count
            fold_macro_audit[-1]["inadequate_history_events"] = inadequate_hist_count
            fold_macro_audit[-1]["nonzero_macro_days"] = nonzero_macro_days
            
            macro_scale = macro_impulse.loc[orig_dates].values / train_spread_std
            macro_overlay_input = 0.60 * sig_kf_arr + 0.40 * macro_scale
            sig_macro_arr = np.clip(macro_overlay_input, -1.0, 1.0)
            sig_macro_dns = pd.Series(sig_macro_arr, index=orig_dates)
            oos_signals["DNS_Kalman_Macro"].append(sig_macro_dns)
            raw_signals["DNS_Kalman_Macro"].extend(macro_overlay_input.tolist())
            stage_inputs["DNS_Kalman_Macro"].extend(macro_overlay_input.tolist())
            stage_outputs["DNS_Kalman_Macro"].extend(sig_macro_arr.tolist())
            inherited_dns_clipping["DNS_Kalman_Macro"].extend((np.abs(raw_kf) > 1.0).tolist())
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
                                sig_gbm_val = map_curve_forecast_to_spread_signal(
                                    c_hat, y_origin_all[k], tenor_cols, train_spread_std, strategy_type
                                )
                                sig_gbm_series.loc[orig_dt] = float(sig_gbm_val)
                                diff_raw_gbm = (float(sp_pred["2s10s"].item()) - float(sp_curr[k])) / train_spread_std
                                raw_signals["GBM"].append(float(diff_raw_gbm))
                                stage_inputs["GBM"].append(float(diff_raw_gbm))
                                stage_outputs["GBM"].append(float(sig_gbm_val))
                                inherited_dns_clipping["GBM"].append(False)
                                additional_clipping["GBM"].append(bool(abs(diff_raw_gbm) > 1.0))
                                saturation_flags["GBM"].append(bool(abs(sig_gbm_val) >= 1.0 - 1e-6))

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
        total_evaluated_decision_events = sum(f.get("evaluated_decision_events", 0) for f in fold_macro_audit)
        total_timestamp_available_events = sum(f.get("timestamp_available_events", 0) for f in fold_macro_audit)
        total_post_close_events = sum(f.get("post_close_events", 0) for f in fold_macro_audit)
        total_usable_surprise_events = sum(f.get("usable_surprise_events", 0) for f in fold_macro_audit)
        total_zero_surprise_events = sum(f.get("zero_surprise_events", 0) for f in fold_macro_audit)
        total_eligible_coefficient_events = sum(f.get("eligible_coefficient_events", 0) for f in fold_macro_audit)
        total_inadequate_history_events = sum(f.get("inadequate_history_events", 0) for f in fold_macro_audit)
        total_nonzero_macro_days = sum(f.get("nonzero_macro_days", 0) for f in fold_macro_audit)

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
                "total_training_events": total_train_macro_events,
                "total_calendar_events": total_calendar_events,
                "total_test_events": total_calendar_events,
                "total_evaluated_decision_events": total_evaluated_decision_events,
                "total_timestamp_available_events": total_timestamp_available_events,
                "total_post_close_events": total_post_close_events,
                "total_usable_surprise_events": total_usable_surprise_events,
                "total_zero_surprise_events": total_zero_surprise_events,
                "total_eligible_coefficient_events": total_eligible_coefficient_events,
                "total_inadequate_history_events": total_inadequate_history_events,
                "nonzero_macro_days": total_nonzero_macro_days,
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
