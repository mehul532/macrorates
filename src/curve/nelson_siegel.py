"""Static (single-day cross-sectional) Nelson-Siegel term structure model."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


@dataclass
class NelsonSiegelFit:
    """Container for a single-day Nelson-Siegel cross-sectional fit."""

    date: Optional[str]
    level: float        # Beta_0 (asymptotic long rate)
    slope: float        # Beta_1 (short-long spread factor)
    curvature: float    # Beta_2 (intermediate hump factor)
    lambda_param: float # Decay / peak parameter
    rmse: float         # Root mean squared error in percentage points
    r_squared: float    # Coefficient of determination
    fitted_yields: np.ndarray
    residuals: np.ndarray


def nelson_siegel_loadings(maturities: np.ndarray, lambda_param: float) -> np.ndarray:
    """
    Compute Nelson-Siegel factor loadings for given maturities and decay parameter lambda.
    
    Loadings:
      f0(tau) = 1.0                                (Level)
      f1(tau) = (1 - exp(-lambda * tau)) / (lambda * tau)   (Slope)
      f2(tau) = f1(tau) - exp(-lambda * tau)       (Curvature)
      
    Returns:
      Matrix of shape (len(maturities), 3).
    """
    tau = np.asarray(maturities, dtype=float)
    x = lambda_param * tau
    # Handle zero maturity smoothly if present
    x_safe = np.where(x == 0, 1e-8, x)
    exp_neg_x = np.exp(-x)
    f1 = (1.0 - exp_neg_x) / x_safe
    f2 = f1 - exp_neg_x

    f0 = np.ones_like(tau)
    return np.column_stack([f0, f1, f2])


def curvature_peak_maturity(lambda_param: float) -> float:
    """Calculate the maturity tau where the curvature loading reaches its maximum."""
    # Derivative of f2(tau) = 0 yields (x^2 + x + 1) * exp(-x) = 1 => x ~ 1.793282
    return 1.793282 / lambda_param


class StaticNelsonSiegel:
    """
    Static (single-day cross-sectional) Nelson-Siegel term structure model.
    
    Solves Level (L), Slope (S), and Curvature (C) by ordinary least squares (OLS)
    conditional on decay parameter lambda.
    """

    DEFAULT_LAMBDA = 0.7308  # Diebold-Li standard: curvature loading peaks at ~2.45 years

    def __init__(self, lambda_param: float = DEFAULT_LAMBDA):
        self.lambda_param = lambda_param

    def fit_cross_section(
        self,
        yields: np.ndarray,
        maturities: np.ndarray,
        date_label: Optional[str] = None,
        optimize_lambda: bool = False,
        lambda_bounds: Tuple[float, float] = (0.1, 2.5),
    ) -> NelsonSiegelFit:
        """
        Fit Nelson-Siegel factors to a single cross-section of yields by OLS.
        
        Args:
            yields: Array of observed yields (e.g. [4.5, 4.2, ...]).
            maturities: Array of maturities in years corresponding to yields.
            date_label: Optional date string for labeling.
            optimize_lambda: If True, grid-searches / optimizes lambda per cross-section.
            lambda_bounds: Search range for lambda parameter if optimized.
        """
        mask = ~np.isnan(yields) & ~np.isnan(maturities)
        y = yields[mask]
        m = maturities[mask]

        if len(y) < 3:
            raise ValueError(f"At least 3 non-NaN observations required for OLS, got {len(y)}")

        def _fit_given_lambda(l_val: float) -> Tuple[np.ndarray, np.ndarray, float]:
            X = nelson_siegel_loadings(m, l_val)
            # OLS: beta = (X'X)^(-1) X' y
            beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            y_hat = X @ beta
            res = y - y_hat
            ssr = np.sum(res ** 2)
            return beta, y_hat, ssr

        if optimize_lambda:
            res_opt = minimize_scalar(
                lambda l: _fit_given_lambda(l)[2],
                bounds=lambda_bounds,
                method="bounded",
            )
            best_lambda = float(res_opt.x)
        else:
            best_lambda = self.lambda_param

        beta, y_hat, ssr = _fit_given_lambda(best_lambda)
        rmse = np.sqrt(ssr / len(y))

        # Coefficient of determination R^2
        tss = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - (ssr / tss) if tss > 0 else 1.0

        return NelsonSiegelFit(
            date=date_label,
            level=float(beta[0]),
            slope=float(beta[1]),
            curvature=float(beta[2]),
            lambda_param=best_lambda,
            rmse=float(rmse),
            r_squared=float(r_squared),
            fitted_yields=y_hat,
            residuals=y - y_hat,
        )

    def predict(
        self,
        maturities: np.ndarray,
        level: float,
        slope: float,
        curvature: float,
        lambda_param: Optional[float] = None,
    ) -> np.ndarray:
        """Evaluate fitted Nelson-Siegel curve across arbitrary continuous maturities."""
        l_val = lambda_param or self.lambda_param
        X = nelson_siegel_loadings(maturities, l_val)
        beta = np.array([level, slope, curvature])
        return X @ beta

    def fit_panel(
        self,
        yield_df: pd.DataFrame,
        maturities_dict: Dict[str, float],
        date_col: str = "date",
        optimize_lambda: bool = False,
    ) -> pd.DataFrame:
        """
        Fit Static Nelson-Siegel model for each date in yield panel.
        
        Returns:
            DataFrame with columns ['date', 'level', 'slope', 'curvature', 'lambda', 'rmse', 'r_squared'].
        """
        cols = [c for c in yield_df.columns if c in maturities_dict]
        cols = sorted(cols, key=lambda c: maturities_dict[c])
        maturities = np.array([maturities_dict[c] for c in cols])

        dates = yield_df[date_col].values
        Y_all = yield_df[cols].to_numpy(dtype=float, na_value=np.nan)

        records = []
        for i in range(len(yield_df)):
            d = dates[i]
            y = Y_all[i]
            if np.sum(~np.isnan(y)) >= 3:
                fit = self.fit_cross_section(
                    yields=y,
                    maturities=maturities,
                    date_label=str(d),
                    optimize_lambda=optimize_lambda,
                )
                records.append({
                    "date": pd.to_datetime(d),
                    "level": fit.level,
                    "slope": fit.slope,
                    "curvature": fit.curvature,
                    "lambda": fit.lambda_param,
                    "rmse": fit.rmse,
                    "r_squared": fit.r_squared,
                })

        return pd.DataFrame(records)


class NelsonSiegelAR1Forecaster:
    """
    Rolling one-step Nelson-Siegel AR(1) term structure forecaster.
    
    RESEARCH INTEGRITY & INFORMATION CONTRACT:
    - Predeclared or training-only lambda.
    - Estimates OLS cross-sectional factors solely on training slice y_train.
    - Estimates stationary AR(1) dynamics on training factor time series.
    - Sequential OOS propagation: At step k, predicts target k from information at k-1,
      then observes target k to update state for step k+1.
    - Fixed parameters throughout the evaluation fold.
    """
    
    def __init__(self, lambda_param: float = 0.7308):
        self.lambda_param = lambda_param
        self.static_ns = StaticNelsonSiegel(lambda_param=lambda_param)
        self.mu_: Optional[np.ndarray] = None
        self.a_diag_: Optional[np.ndarray] = None
        self.c_: Optional[np.ndarray] = None
        self.last_train_factors_: Optional[np.ndarray] = None
        self.maturities_: Optional[np.ndarray] = None
        self.loadings_: Optional[np.ndarray] = None

    def fit(self, y_train: np.ndarray, maturities: np.ndarray) -> "NelsonSiegelAR1Forecaster":
        """
        Fit factor loadings, historical factor series, and AR(1) dynamics strictly on training yields.
        """
        self.maturities_ = np.asarray(maturities, dtype=float)
        self.loadings_ = nelson_siegel_loadings(self.maturities_, self.lambda_param)
        T, N = y_train.shape
        
        # Fit OLS factor for each training date
        factors_tr = np.zeros((T, 3))
        for t in range(T):
            y_t = y_train[t]
            mask = ~np.isnan(y_t)
            if np.sum(mask) >= 3:
                beta, _, _, _ = np.linalg.lstsq(self.loadings_[mask], y_t[mask], rcond=None)
                factors_tr[t] = beta
            else:
                factors_tr[t] = factors_tr[t - 1] if t > 0 else np.array([4.0, -1.0, 1.0])
                
        self.last_train_factors_ = factors_tr[-1].copy()
        
        # Fit stationary AR(1) on training factor deviations
        mu = np.mean(factors_tr, axis=0)
        factors_dm = factors_tr - mu
        X_lag = factors_dm[:-1]
        Y_lead = factors_dm[1:]
        
        a_diag = []
        for i in range(3):
            denom = float(np.sum(X_lag[:, i] ** 2))
            a_i = float(np.sum(X_lag[:, i] * Y_lead[:, i]) / denom) if denom > 0 else 0.95
            a_diag.append(float(np.clip(a_i, -0.999, 0.999)))
            
        self.mu_ = mu
        self.a_diag_ = np.array(a_diag)
        self.c_ = mu * (1.0 - self.a_diag_)
        self.factors_tr_ = factors_tr
        self.train_slope_std_ = float(np.std(factors_tr[:, 1])) + 1e-6
        return self

    def sequential_predict_and_update(
        self,
        y_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sequentially forecast 1-step curve and factors across test observations.
        
        Returns:
          y_pred: (K, N) predicted yield curves
          factors_pred: (K, 3) 1-step predicted factor state
          factors_obs: (K, 3) observed OLS factors after receiving y_test
        """
        if self.loadings_ is None or self.last_train_factors_ is None or self.c_ is None:
            raise ValueError("Forecaster must be fit before forecasting.")
            
        K, N = y_test.shape
        y_pred = np.zeros((K, N))
        factors_pred = np.zeros((K, 3))
        factors_obs = np.zeros((K, 3))
        
        f_curr = self.last_train_factors_.copy()
        
        for k in range(K):
            # 1. 1-step forecast from information through k-1
            f_p = self.c_ + self.a_diag_ * f_curr
            y_p = self.loadings_ @ f_p
            
            y_pred[k] = y_p
            factors_pred[k] = f_p
            
            # 2. Observe y_test[k] and update factor state
            y_k = y_test[k]
            mask = ~np.isnan(y_k)
            if np.sum(mask) >= 3:
                beta, _, _, _ = np.linalg.lstsq(self.loadings_[mask], y_k[mask], rcond=None)
                f_curr = beta
            else:
                f_curr = f_p
                
            factors_obs[k] = f_curr
            
        return y_pred, factors_pred, factors_obs

