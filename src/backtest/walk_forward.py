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
    """Retrieve current git commit hash for run provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "UNKNOWN_COMMIT"


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
        
        # Cumulative out-of-sample predictions and signals containers
        oos_signals = {m: [] for m in models}
        curve_sq_errors = {m: [] for m in models}
        factor_sq_errors = {m: [] for m in models}
        
        # Initialize formal ForecastLedger under research-integrity contract
        ledger = ForecastLedger(run_id="walk_forward_evaluation")
        self.ledger = ledger
        
        tenor_cols = [c for c in CANONICAL_TENORS.keys() if c in self.yield_df.columns]
        maturities = get_canonical_maturities(tenor_cols)
        mat_dict = {c: CANONICAL_TENORS[c] for c in tenor_cols}
        
        spread_2s10s_sq_errors = {m: [] for m in models}
        fly_2s5s10s_sq_errors = {m: [] for m in models}
        per_tenor_sq_errors = {m: {c: [] for c in tenor_cols} for m in models}
        
        for f_idx, (train_dates, test_dates) in enumerate(folds):
            # Define explicit rolling 1-step forecast origin-target pairs:
            # Origin t_0 = train_dates[-1] predicts target test_dates[0].
            # For subsequent steps, origin test_dates[i] predicts target test_dates[i+1].
            # Guarantees origin strictly precedes target with zero lookahead and no lost boundary data.
            origin_target_pairs = [(train_dates[-1], test_dates[0])] + [
                (test_dates[i], test_dates[i + 1]) for i in range(len(test_dates) - 1)
            ]
            
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
                
            sig_rw = pd.Series(0.0, index=test_dates)
            oos_signals["Random_Walk"].append(sig_rw)

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
            sig_pca = pd.Series(sig_pca_arr, index=test_dates)
            oos_signals["PCA_VAR"].append(sig_pca)
            
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
            sig_ns = pd.Series(sig_ns_arr, index=test_dates)
            oos_signals["Static_NS"].append(sig_ns)
            
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
            sig_kf = pd.Series(sig_kf_arr, index=test_dates)
            oos_signals["DNS_Kalman"].append(sig_kf)
            
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
            
            macro_sub = self.macro_df[
                (self.macro_df["date"] >= test_dates[0]) & (self.macro_df["date"] <= test_dates[-1])
            ]
            macro_impulse = pd.Series(0.0, index=test_dates)
            for _, m_row in macro_sub.iterrows():
                dt = m_row["date"]
                ind = m_row.get("indicator")
                surp = m_row.get("surprise_ann", np.nan)
                if pd.notna(surp) and dt in macro_impulse.index:
                    b_info = fold_macro_betas.get(ind, {"slope_beta": 0.0, "curvature_beta": 0.0})
                    if strategy_type in ("2s5s10s", "fly"):
                        b_target = b_info.get("curvature_beta", 0.0)
                    else:
                        b_target = b_info.get("slope_beta", 0.0)
                    macro_impulse.loc[dt] += b_target * np.clip(surp, -2.0, 2.0)
            
            sig_macro_dns = np.clip(0.6 * sig_kf + 0.4 * (macro_impulse / train_spread_std), -1.0, 1.0)
            oos_signals["DNS_Kalman_Macro"].append(sig_macro_dns)
            
            for k, (orig_dt, tgt_dt) in enumerate(origin_target_pairs):
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
                        
                        sig_gbm_series = pd.Series(0.0, index=test_dates)

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

                                # Signal at tgt_dt based on forecast made at orig_dt
                                sig_gbm_val = map_curve_forecast_to_spread_signal(
                                    c_hat, y_origin_all[k], tenor_cols, train_spread_std, strategy_type
                                )
                                sig_gbm_series.loc[tgt_dt] = float(sig_gbm_val)

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
                        oos_signals["GBM"].append(pd.Series(0.0, index=test_dates))
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
                    oos_signals["GBM"].append(pd.Series(0.0, index=test_dates))
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
                oos_signals["GBM"].append(pd.Series(0.0, index=test_dates))

        # Concatenate out-of-sample series and execute backtests
        baseline_rows = []
        backtest_results = {}
        
        for m in models:
            full_sig = pd.concat(oos_signals[m])
            pos_df = compute_continuous_positions(
                signals_series=full_sig,
                strategy_type=strategy_type,
                target_dv01=self.config.target_dv01,
            )
            
            b_res = self.engine.run_strategy(
                strategy_name=m,
                positions_df=pos_df,
                yield_df=self.yield_df,
            )
            backtest_results[m] = b_res
            met = b_res.metrics
            
            # Compute RMSE in basis points (1 bp = 0.01 percentage point)
            # An unavailable model cannot acquire a numeric RMSE
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
            
            baseline_rows.append({
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
                "Hit Rate (%)": met.get("win_rate_pct", 0.0),
                "PnL / DV01 ($)": pnl_dv01,
                "PnL / Turnover ($/lot)": pnl_turnover,
                "Gross PnL ($)": met.get("total_gross_pnl_usd", 0.0),
                "Trade Costs ($)": met.get("total_trade_cost_usd", 0.0),
                "Roll Costs ($)": met.get("total_roll_cost_usd", 0.0),
                "Trading Net PnL ($)": met.get("total_net_trading_pnl_usd", 0.0),
                "Cash Interest ($)": met.get("total_interest_earned_usd", 0.0),
                "Collateral Net PnL ($)": met.get("total_collateral_pnl_usd", met.get("total_net_pnl_usd", 0.0)),
            })
            
        baseline_table = pd.DataFrame(baseline_rows).set_index("Model / Forecast Method")

        # Build common_sample_table containing evaluated models plus Cash Only strategy benchmark
        ROLE_MAP = {
            "Random_Walk": ("Random Walk (Curve Benchmark)", "Curve Benchmark"),
            "PCA_VAR": ("PCA / VAR(1)", "Term Structure Factor Model"),
            "Static_NS": ("AR(1) Baseline (Static NS)", "AR(1) Baseline"),
            "DNS_Kalman": ("DNS + Kalman", "Dynamic Term Structure Model"),
            "DNS_Kalman_Macro": ("DNS + Kalman + Macro", "Macro-Augmented DTSM"),
            "GBM": ("GBM", "Machine Learning Baseline"),
        }

        common_sample_rows = []
        for orig_row, m in zip(baseline_rows, models):
            display_name, role = ROLE_MAP.get(m, (orig_row["Model / Forecast Method"], "Model Baseline"))
            row_copy = {
                "Model / Forecast Method": display_name,
                "Benchmark Role": role,
                "Forecast Status": orig_row["Forecast Status"],
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

        all_eval_dates = pd.concat(oos_signals["Random_Walk"]).index
        cash_pos_df = pd.DataFrame(0, index=all_eval_dates, columns=[f"n_{s.lower()}" for s in ["zt", "zf", "zn"]])
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
            "Sample Size (Days)": len(cash_pos_df),
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

        run_metadata = {
            "git_commit": get_git_commit_hash(),
            "data_checksums": {
                "yield_panel": get_file_checksum("data/processed/yield_panel.parquet"),
                "factor_panel": get_file_checksum("data/processed/factor_panel.parquet"),
                "macro_surprises": get_file_checksum("data/processed/macro_surprises.parquet"),
            },
            "eval_start_date": str(all_eval_dates[0].date()) if len(all_eval_dates) > 0 else "N/A",
            "eval_end_date": str(all_eval_dates[-1].date()) if len(all_eval_dates) > 0 else "N/A",
            "total_eval_days": len(all_eval_dates),
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
        }
        
        return {
            "baseline_table": baseline_table,
            "common_sample_table": common_sample_table,
            "backtest_results": backtest_results,
            "signals": oos_signals,
            "forecast_ledger": ledger,
            "run_metadata": run_metadata,
        }
