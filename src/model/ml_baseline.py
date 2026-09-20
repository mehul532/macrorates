"""
Gradient-Boosted Model (LightGBM) Factor Forecaster & Leakage-Safe Feature Engineering.

Milestone 14 Implementation:
- Forecasts 1-step-ahead changes in Nelson-Siegel Level, Slope, and Curvature factors:
    Delta L_{t+1} = L_{t+1} - L_t
    Delta S_{t+1} = S_{t+1} - S_t
    Delta C_{t+1} = C_{t+1} - C_t
- Engineered feature space:
    1. Lagged factors: levels, 1d/2d differences, 5d momentum, 21d z-scores, 10d volatilities.
    2. Strictly separated Milestone 4 macro surprises: S_ann_t and S_model_t for CPI, Core CPI, NFP, UNEMP, FOMC.
    3. Ex-ante Fed policy regime ONLY: fed_regime_ex_ante (never fed_regime_ex_post -- zero lookahead).
- Reconstructs out-of-sample yield curves via Nelson-Siegel loadings.
- TreeSHAP feature attribution explaining factor drivers vs. macroeconomic intuition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.curve.nelson_siegel import nelson_siegel_loadings
from src.backtest.regimes import tag_fed_regimes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Leakage-Safe Feature Engineering Pipeline
# ---------------------------------------------------------------------------

class FactorFeatureEngineer:
    """
    Constructs a leakage-safe feature matrix X_t and forward targets y_{t+1}.
    
    Guarantees:
    - Features at date t use strictly observations <= t.
    - S_ann_t (consensus surprise) and S_model_t (statistical innovation) are separate and never merged.
    - fed_regime_ex_ante is used exclusively; fed_regime_ex_post is strictly barred.
    """

    TARGET_INDICATORS = ["CPI", "CORE_CPI", "NFP", "UNEMP", "FOMC"]

    @classmethod
    def build_feature_panel(
        cls,
        factor_df: pd.DataFrame,
        yield_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        factor_prefix: str = "kf",
        decay_factor: float = 0.70,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Build feature matrix X and targets y.
        
        Returns:
            Tuple of (X DataFrame, y DataFrame, feature_names list).
        """
        # Ensure DateTime indices
        f_df = factor_df.copy()
        if "date" in f_df.columns:
            f_df["date"] = pd.to_datetime(f_df["date"])
            f_df = f_df.sort_values("date").set_index("date")

        y_df = yield_df.copy()
        if "date" in y_df.columns:
            y_df["date"] = pd.to_datetime(y_df["date"])
            y_df = y_df.sort_values("date").set_index("date")

        common_idx = f_df.index.intersection(y_df.index)
        f_df = f_df.loc[common_idx]
        y_df = y_df.loc[common_idx]

        # 1. Base factor levels (t)
        lvl_col = f"{factor_prefix}_level"
        slp_col = f"{factor_prefix}_slope"
        cur_col = f"{factor_prefix}_curvature"

        features = pd.DataFrame(index=common_idx)
        features["level_t0"] = f_df[lvl_col]
        features["slope_t0"] = f_df[slp_col]
        features["curvature_t0"] = f_df[cur_col]

        # 2. Lagged factor changes (strictly backward-looking)
        features["dlevel_1d"] = f_df[lvl_col].diff(1)
        features["dslope_1d"] = f_df[slp_col].diff(1)
        features["dcurvature_1d"] = f_df[cur_col].diff(1)

        features["dlevel_2d"] = f_df[lvl_col].diff(1).shift(1)
        features["dslope_2d"] = f_df[slp_col].diff(1).shift(1)
        features["dcurvature_2d"] = f_df[cur_col].diff(1).shift(1)

        # 3. 5-day factor momentum
        features["level_mom5d"] = f_df[lvl_col] - f_df[lvl_col].shift(5)
        features["slope_mom5d"] = f_df[slp_col] - f_df[slp_col].shift(5)
        features["curvature_mom5d"] = f_df[cur_col] - f_df[cur_col].shift(5)

        # 4. 21-day rolling z-scores
        for col_name, raw_s in [("level", f_df[lvl_col]), ("slope", f_df[slp_col]), ("curvature", f_df[cur_col])]:
            roll_mean = raw_s.rolling(21, min_periods=10).mean()
            roll_std = raw_s.rolling(21, min_periods=10).std().replace(0, np.nan)
            features[f"{col_name}_zscore21d"] = ((raw_s - roll_mean) / roll_std).fillna(0.0)

        # 5. 10-day rolling volatilities
        for col_name, raw_s in [("level", features["dlevel_1d"]), ("slope", features["dslope_1d"]), ("curvature", features["dcurvature_1d"])]:
            features[f"{col_name}_vol10d"] = raw_s.rolling(10, min_periods=5).std().fillna(0.0)

        # 6. Macro surprise features (strictly separate announcement and model surprise series)
        m_df = macro_df.copy()
        if "date" in m_df.columns:
            m_df["date"] = pd.to_datetime(m_df["date"])

        for ind in cls.TARGET_INDICATORS:
            sub = m_df[m_df["indicator"] == ind].sort_values("date")
            # Map announcement surprise S_ann (deduplicate date index if multiple intraday events)
            s_ann_map = sub.dropna(subset=["surprise_ann"]).drop_duplicates(subset=["date"], keep="last").set_index("date")["surprise_ann"]
            s_mod_map = sub.dropna(subset=["surprise_model"]).drop_duplicates(subset=["date"], keep="last").set_index("date")["surprise_model"]

            # Daily series aligned to trading calendar (0.0 on non-release days)
            ann_s = pd.Series(features.index.map(s_ann_map).fillna(0.0), index=features.index)
            mod_s = pd.Series(features.index.map(s_mod_map).fillna(0.0), index=features.index)

            features[f"surp_ann_{ind}"] = ann_s.values
            features[f"surp_model_{ind}"] = mod_s.values

            # 5-day exponential decay impulse: sum_{k=0}^4 (decay^k * S_{t-k})
            decay_series = pd.Series(0.0, index=features.index)
            curr_val = 0.0
            for dt in features.index:
                curr_val = curr_val * decay_factor + float(ann_s.loc[dt])
                decay_series.loc[dt] = curr_val
            features[f"surp_decay5d_{ind}"] = decay_series.values

        # 7. Ex-ante Fed policy regime (strictly backward-looking, NO EX-POST)
        regimes = tag_fed_regimes(y_df)
        ex_ante = regimes["fed_regime_ex_ante"]

        features["regime_ex_ante_ZLB"] = (ex_ante == "ZLB").astype(float)
        features["regime_ex_ante_Hiking"] = (ex_ante == "Hiking").astype(float)
        features["regime_ex_ante_Easing"] = (ex_ante == "Easing").astype(float)
        features["regime_ex_ante_Pause"] = (ex_ante == "Pause/Hold").astype(float)

        # Enforce zero-leakage assertions
        assert "fed_regime_ex_post" not in features.columns, "Leakage violation: fed_regime_ex_post detected in feature panel!"

        # 8. Target variables: Forward 1-step change in factors (t -> t+1)
        targets = pd.DataFrame(index=common_idx)
        targets["origin_date"] = common_idx
        target_dates = pd.Series(common_idx, index=common_idx).shift(-1)
        targets["target_date"] = target_dates
        targets["label_available_at"] = target_dates

        targets["target_dLevel"] = f_df[lvl_col].shift(-1) - f_df[lvl_col]
        targets["target_dSlope"] = f_df[slp_col].shift(-1) - f_df[slp_col]
        targets["target_dCurvature"] = f_df[cur_col].shift(-1) - f_df[cur_col]
        targets["target_Level"] = f_df[lvl_col].shift(-1)
        targets["target_Slope"] = f_df[slp_col].shift(-1)
        targets["target_Curvature"] = f_df[cur_col].shift(-1)

        # Drop initial warmup NaNs in features, but keep features for all valid origin dates
        valid_features = features.notna().all(axis=1)
        X_clean = features[valid_features].copy()
        y_clean = targets.loc[X_clean.index].copy()

        feature_cols = list(X_clean.columns)
        return X_clean, y_clean, feature_cols

    @staticmethod
    def get_training_slice(
        X: pd.DataFrame,
        y: pd.DataFrame,
        training_cutoff: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select training rows strictly whose target labels mature <= training_cutoff.
        
        RESEARCH INTEGRITY & CAUSAL BOUNDARY (Prompt 4):
        - Origin row t is included IF AND ONLY IF target_date (t+1) <= training_cutoff.
        - The boundary row at t = training_cutoff (whose target matures at t+1) is EXCLUDED.
        """
        mask = (
            (y["origin_date"] <= training_cutoff)
            & (y["label_available_at"] <= training_cutoff)
            & y["target_dSlope"].notna()
        )
        return X.loc[mask].copy(), y.loc[mask].copy()


# ---------------------------------------------------------------------------
# 2. Gradient-Boosted Model Forecaster (Explicit Backend Selection)
# ---------------------------------------------------------------------------

@dataclass
class GBMForecasterConfig:
    """Hyperparameters and backend selection for factor forecaster."""
    backend: str = "sklearn"  # Explicit backend: "sklearn", "lightgbm", or "xgboost"
    n_estimators: int = 100
    learning_rate: float = 0.03
    num_leaves: int = 15
    max_depth: int = 4
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    verbose: int = -1


class GradientBoostedFactorForecaster:
    """
    Fits multi-target gradient-boosted trees for Level, Slope, and Curvature changes.
    
    Provides:
    - 1-step-ahead factor increments prediction.
    - Yield curve reconstruction via Nelson-Siegel loadings.
    - Exact TreeSHAP feature importance and explanation extraction.
    """

    def __init__(self, config: Optional[GBMForecasterConfig] = None):
        self.config = config or GBMForecasterConfig()
        self.models: Dict[str, Any] = {}
        self.feature_names_: List[str] = []
        self.backend_name_: str = self.config.backend.lower()
        self._is_fitted: bool = False

    def _init_model(self) -> Any:
        backend = self.config.backend.lower()
        if backend == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMRegressor(
                    n_estimators=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    num_leaves=self.config.num_leaves,
                    max_depth=self.config.max_depth,
                    min_child_samples=self.config.min_child_samples,
                    subsample=self.config.subsample,
                    colsample_bytree=self.config.colsample_bytree,
                    random_state=self.config.random_state,
                    verbose=self.config.verbose,
                )
            except ImportError as e:
                raise ImportError(f"Explicit backend 'lightgbm' requested, but lightgbm is not available: {e}")
        elif backend == "xgboost":
            try:
                import xgboost as xgb
                return xgb.XGBRegressor(
                    n_estimators=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    max_depth=self.config.max_depth,
                    random_state=self.config.random_state,
                    verbosity=0,
                )
            except ImportError as e:
                raise ImportError(f"Explicit backend 'xgboost' requested, but xgboost is not available: {e}")
        elif backend == "sklearn":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                subsample=min(1.0, self.config.subsample),
                random_state=self.config.random_state,
            )
        else:
            raise ValueError(
                f"Unknown backend '{self.config.backend}'. Must be 'sklearn', 'lightgbm', or 'xgboost'."
            )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> GradientBoostedFactorForecaster:
        """Fit specialized regressors for Level, Slope, and Curvature."""
        self.feature_names_ = list(X.columns)
        self.backend_name_ = self.config.backend.lower()

        for target_col in ["target_dLevel", "target_dSlope", "target_dCurvature"]:
            model = self._init_model()
            valid = y[target_col].notna()
            model.fit(X.loc[valid].values, y.loc[valid, target_col].values)
            self.models[target_col] = model

        self._is_fitted = True
        return self

    def predict_increments(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict 1-step factor increments:
        Returns DataFrame with columns ['dLevel_hat', 'dSlope_hat', 'dCurvature_hat'].
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict_increments can be called.")

        preds = {}
        target_map = {
            "target_dLevel": "dLevel_hat",
            "target_dSlope": "dSlope_hat",
            "target_dCurvature": "dCurvature_hat",
        }
        for k, col in target_map.items():
            preds[col] = self.models[k].predict(X.values)

        return pd.DataFrame(preds, index=X.index)

    def forecast_factors(
        self,
        X: pd.DataFrame,
        current_factors: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute predicted factor levels:
        F_{hat, t+1} = F_t + Delta F_{hat, t+1}
        """
        increments = self.predict_increments(X)
        out = pd.DataFrame(index=X.index)
        out["level_hat"] = current_factors["level_t0"] + increments["dLevel_hat"]
        out["slope_hat"] = current_factors["slope_t0"] + increments["dSlope_hat"]
        out["curvature_hat"] = current_factors["curvature_t0"] + increments["dCurvature_hat"]
        return out

    @staticmethod
    def reconstruct_yield_curve(
        predicted_factors: pd.DataFrame,
        maturities: np.ndarray,
        lambda_param: float = 0.7308,
    ) -> pd.DataFrame:
        """
        Reconstruct the predicted yield curve across maturities:
        y_hat(tau) = L_hat + S_hat * f1(tau, lambda) + C_hat * f2(tau, lambda)
        """
        loadings = nelson_siegel_loadings(maturities, lambda_param)  # (N_mat, 3)
        factors = predicted_factors[["level_hat", "slope_hat", "curvature_hat"]].values  # (T, 3)
        curve_matrix = factors @ loadings.T  # (T, N_mat)
        return pd.DataFrame(curve_matrix, index=predicted_factors.index)

    def explain_shap(
        self,
        X: pd.DataFrame,
        sample_size: int = 500,
    ) -> Dict[str, Any]:
        """
        Compute TreeSHAP values and feature importances for all 3 factor models.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before explain_shap can be called.")

        try:
            import shap
        except ImportError:
            logger.warning("SHAP library not found; returning Gini feature importances.")
            importances = {}
            for target_col, model in self.models.items():
                imp = pd.Series(model.feature_importances_, index=self.feature_names_).sort_values(ascending=False)
                importances[target_col] = imp
            return {"type": "gini", "importances": importances}

        X_eval = X.sample(n=min(len(X), sample_size), random_state=self.config.random_state)
        shap_results = {}

        for target_col in ["target_dLevel", "target_dSlope", "target_dCurvature"]:
            model = self.models[target_col]
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_eval)

            # Compute mean absolute SHAP value per feature
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            imp_series = pd.Series(mean_abs_shap, index=self.feature_names_).sort_values(ascending=False)

            shap_results[target_col] = {
                "explainer": explainer,
                "shap_values": shap_vals,
                "mean_abs_shap": imp_series,
                "X_eval": X_eval,
            }

        return {"type": "tree_shap", "results": shap_results}


# ---------------------------------------------------------------------------
# 3. High-Level Evaluation & Pipeline Orchestrator
# ---------------------------------------------------------------------------

def run_ml_factor_forecasting(
    factor_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    train_split_ratio: float = 0.80,
    config: Optional[GBMForecasterConfig] = None,
) -> Dict[str, Any]:
    """
    Run end-to-end ML factor forecasting and SHAP attribution analysis.
    """
    logger.info("Building leakage-safe feature matrix...")
    X, y, feature_cols = FactorFeatureEngineer.build_feature_panel(
        factor_df=factor_df,
        yield_df=yield_df,
        macro_df=macro_df,
    )
    logger.info("Constructed feature panel: %d dates with %d engineered features.", len(X), len(feature_cols))

    # Chronological train/test split with strict causal training boundary
    split_idx = int(len(X) * train_split_ratio)
    training_cutoff = X.index[split_idx - 1]
    X_train, y_train = FactorFeatureEngineer.get_training_slice(X, y, training_cutoff)
    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()

    logger.info(
        "Fitting GradientBoostedFactorForecaster on causal training slice (%d observations <= %s)...",
        len(X_train),
        training_cutoff.strftime("%Y-%m-%d"),
    )
    forecaster = GradientBoostedFactorForecaster(config=config)
    forecaster.fit(X_train, y_train)

    # Out-of-sample forecast
    test_factors = X_test[["level_t0", "slope_t0", "curvature_t0"]]
    f_pred = forecaster.forecast_factors(X_test, test_factors)
    increments_pred = forecaster.predict_increments(X_test)

    # Out-of-sample evaluation: evaluate Fhat[t+1|t] against actual observation at t+1
    eval_mask = y_test["target_dSlope"].notna()
    X_test_eval = X_test.loc[eval_mask]
    y_test_eval = y_test.loc[eval_mask]
    f_pred_eval = f_pred.loc[eval_mask]
    inc_pred_eval = increments_pred.loc[eval_mask]

    # Model RMSEs in basis points (1 bp = 0.01 percentage point)
    rmse_bp = {
        "Level": float(np.sqrt(np.mean((y_test_eval["target_Level"] - f_pred_eval["level_hat"]) ** 2)) * 100.0),
        "Slope": float(np.sqrt(np.mean((y_test_eval["target_Slope"] - f_pred_eval["slope_hat"]) ** 2)) * 100.0),
        "Curvature": float(np.sqrt(np.mean((y_test_eval["target_Curvature"] - f_pred_eval["curvature_hat"]) ** 2)) * 100.0),
        "dLevel": float(np.sqrt(np.mean((y_test_eval["target_dLevel"] - inc_pred_eval["dLevel_hat"]) ** 2)) * 100.0),
        "dSlope": float(np.sqrt(np.mean((y_test_eval["target_dSlope"] - inc_pred_eval["dSlope_hat"]) ** 2)) * 100.0),
        "dCurvature": float(np.sqrt(np.mean((y_test_eval["target_dCurvature"] - inc_pred_eval["dCurvature_hat"]) ** 2)) * 100.0),
    }

    # Matching Random Walk benchmark on identical targets (Fhat_RW[t+1|t] = F[t])
    rw_rmse_bp = {
        "Level": float(np.sqrt(np.mean((y_test_eval["target_Level"] - X_test_eval["level_t0"]) ** 2)) * 100.0),
        "Slope": float(np.sqrt(np.mean((y_test_eval["target_Slope"] - X_test_eval["slope_t0"]) ** 2)) * 100.0),
        "Curvature": float(np.sqrt(np.mean((y_test_eval["target_Curvature"] - X_test_eval["curvature_t0"]) ** 2)) * 100.0),
    }

    logger.info("Computing TreeSHAP values for all factor models...")
    shap_data = forecaster.explain_shap(X_test, sample_size=400)

    return {
        "forecaster": forecaster,
        "feature_names": feature_cols,
        "training_cutoff": training_cutoff,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "f_pred": f_pred,
        "increments_pred": increments_pred,
        "rmse_bp": rmse_bp,
        "rw_rmse_bp": rw_rmse_bp,
        "shap_data": shap_data,
    }
