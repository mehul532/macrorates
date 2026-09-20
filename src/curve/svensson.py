"""
Svensson 4-Factor (6-Parameter) Dynamic Term Structure Model.

Promoted from Milestone 2 appendix to a full parallel curve model:
  y(tau) = beta0 + beta1 * [(1 - exp(-tau/lambda1)) / (tau/lambda1)]
                 + beta2 * [(1 - exp(-tau/lambda1)) / (tau/lambda1) - exp(-tau/lambda1)]
                 + beta3 * [(1 - exp(-tau/lambda2)) / (tau/lambda2) - exp(-tau/lambda2)]

Features:
1. Flexible parameterization: lambda1 and lambda2 as scale parameters (years, matching GSW tau1, tau2)
   or decay rates.
2. Cross-sectional estimation via conditional OLS inside 2D bounded optimization (L-BFGS-B / Nelder-Mead)
   with collinearity separation penalty (|lambda1 - lambda2| >= 0.4).
3. Full daily panel fitting across multi-decade yield panels with warm-starting.
4. Model selection diagnostics: SSR, AIC, BIC, and Leave-One-Out Cross-Validation (LOOCV) RMSE.
5. Per-maturity residual tracking and comparison against Nelson-Siegel and GSW.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SvenssonFit:
    """Container for a single-day Svensson 4-factor / 6-parameter fit."""

    date: Optional[str]
    beta0: float           # Long-term asymptotic level
    beta1: float           # Slope (short-to-long spread)
    beta2: float           # First curvature factor (first hump)
    beta3: float           # Second curvature factor (second hump)
    lambda1: float         # Scale parameter 1 in years (tau / lambda1)
    lambda2: float         # Scale parameter 2 in years (tau / lambda2)
    rmse: float            # In-sample Root Mean Squared Error (% points)
    r_squared: float       # In-sample R-squared
    ssr: float             # Sum of squared residuals
    aic: float             # Akaike Information Criterion
    bic: float             # Bayesian Information Criterion
    loocv_rmse: Optional[float]  # Out-of-sample Leave-One-Out Cross-Validation RMSE
    fitted_yields: np.ndarray    # Fitted yield values across evaluated maturities
    residuals: np.ndarray        # y_observed - y_fitted


def svensson_loadings(
    maturities: np.ndarray,
    lambda1: float,
    lambda2: float,
    is_scale: Optional[bool] = None,
) -> np.ndarray:
    """
    Compute Svensson 4-factor loadings for given maturities.

    Formula:
      f0(tau) = 1.0                                                  (Level)
      f1(tau) = (1 - exp(-tau/lambda1)) / (tau/lambda1)              (Slope)
      f2(tau) = f1(tau) - exp(-tau/lambda1)                          (Curvature 1)
      f3(tau) = (1 - exp(-tau/lambda2)) / (tau/lambda2) - exp(-tau/lambda2)  (Curvature 2)

    Args:
        maturities: Array of tenors tau in years (e.g. [0.083, 0.25, ..., 30.0]).
        lambda1: First decay/scale parameter.
        lambda2: Second decay/scale parameter.
        is_scale: If True, x = tau / lambda. If False, x = lambda * tau.
                  If None (auto), treats as rate if both <= 1.0, otherwise scale.

    Returns:
        Array of shape (len(maturities), 4).
    """
    tau = np.asarray(maturities, dtype=float)
    l1 = max(float(lambda1), 1e-5)
    l2 = max(float(lambda2), 1e-5)

    if is_scale is True or (is_scale is None and (l1 > 1.0 or l2 > 1.0)):
        x1 = tau / l1
        x2 = tau / l2
    else:
        x1 = l1 * tau
        x2 = l2 * tau

    x1_safe = np.where(x1 == 0, 1e-8, x1)
    x2_safe = np.where(x2 == 0, 1e-8, x2)

    exp_neg_x1 = np.exp(-x1_safe)
    exp_neg_x2 = np.exp(-x2_safe)

    f0 = np.ones_like(tau)
    f1 = (1.0 - exp_neg_x1) / x1_safe
    f2 = f1 - exp_neg_x1
    f3 = (1.0 - exp_neg_x2) / x2_safe - exp_neg_x2

    return np.column_stack([f0, f1, f2, f3])


def compute_aic_bic(ssr: float, n_obs: int, n_params: int) -> Tuple[float, float]:
    """
    Compute Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).

    For Gaussian errors:
      AIC = n * ln(SSR / n) + 2 * k
      BIC = n * ln(SSR / n) + k * ln(n)
    """
    if ssr <= 0 or n_obs <= 0:
        return np.nan, np.nan
    var_hat = max(ssr / n_obs, 1e-12)
    aic = n_obs * np.log(var_hat) + 2.0 * n_params
    bic = n_obs * np.log(var_hat) + n_params * np.log(n_obs)
    return float(aic), float(bic)


class SvenssonCurve:
    """
    Svensson 4-Factor / 6-Parameter Term Structure Model.

    Fits Level (beta0), Slope (beta1), Primary Curvature (beta2), and Secondary
    Curvature (beta3) conditional on decay scale parameters lambda1 and lambda2.
    """

    # Defaults in years: lambda1 ~ 1.37 yr (peaks at 2.45 yr), lambda2 ~ 5.5 yr (peaks at 9.8 yr)
    DEFAULT_LAMBDA1 = 1.37
    DEFAULT_LAMBDA2 = 5.50

    def __init__(
        self,
        default_lambda1: float = DEFAULT_LAMBDA1,
        default_lambda2: float = DEFAULT_LAMBDA2,
        init_lambda1: Optional[float] = None,
        init_lambda2: Optional[float] = None,
        is_scale: Optional[bool] = None,
        min_separation: Optional[float] = None,
    ):
        self.default_lambda1 = init_lambda1 if init_lambda1 is not None else default_lambda1
        self.default_lambda2 = init_lambda2 if init_lambda2 is not None else default_lambda2
        self.init_lambda1 = self.default_lambda1
        self.init_lambda2 = self.default_lambda2

        if is_scale is None:
            self.is_scale = False if (self.default_lambda1 <= 1.0 and self.default_lambda2 <= 1.0) else True
        else:
            self.is_scale = is_scale

        if min_separation is None:
            self.min_separation = 0.40 if self.is_scale else 0.05
        else:
            self.min_separation = min_separation

    def fit_cross_section(
        self,
        yields: np.ndarray,
        maturities: np.ndarray,
        date_label: Optional[str] = None,
        optimize_lambdas: bool = True,
        init_lambda1: Optional[float] = None,
        init_lambda2: Optional[float] = None,
        bounds_lambda1: Tuple[float, float] = (0.05, 30.0),
        bounds_lambda2: Tuple[float, float] = (0.05, 30.0),
        compute_loocv: bool = False,
    ) -> SvenssonFit:
        """
        Fit Svensson curve to a single cross-section of yields by conditional OLS.

        Args:
            yields: Array of observed yields (e.g. [4.5, 4.2, ...]).
            maturities: Array of maturities in years.
            date_label: Optional date string label.
            optimize_lambdas: If True, jointly optimize lambda1 and lambda2.
            init_lambda1: Starting value for lambda1.
            init_lambda2: Starting value for lambda2.
            bounds_lambda1: (min, max) for lambda1 in years.
            bounds_lambda2: (min, max) for lambda2 in years.
            compute_loocv: If True, computes leave-one-out cross-validation RMSE.

        Returns:
            SvenssonFit container with estimated factors, diagnostics, and fitted yields.
        """
        mask = ~np.isnan(yields) & ~np.isnan(maturities)
        y = yields[mask]
        m = maturities[mask]
        n_obs = len(y)

        if n_obs < 5:
            raise ValueError(f"At least 5 non-NaN observations required for Svensson fit, got {n_obs}")

        def _fit_given_lambdas(l1: float, l2: float) -> Tuple[np.ndarray, np.ndarray, float]:
            X = svensson_loadings(m, l1, l2, is_scale=self.is_scale)
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_hat = X @ beta
            res = y - y_hat
            ssr = float(np.sum(res ** 2))
            return beta, y_hat, ssr

        if optimize_lambdas:
            l1_init = init_lambda1 or self.default_lambda1
            l2_init = init_lambda2 or self.default_lambda2

            def _cost(params: np.ndarray) -> float:
                l1, l2 = params
                if abs(l1 - l2) < self.min_separation:
                    # Penalize near-identical lambdas that cause collinearity explosion
                    return 1e5 + (self.min_separation - abs(l1 - l2)) * 1e6
                X = svensson_loadings(m, l1, l2, is_scale=self.is_scale)
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                res = y - (X @ beta)
                return float(np.sum(res ** 2))

            bounds = [bounds_lambda1, bounds_lambda2]
            res_opt = minimize(
                _cost,
                x0=np.array([l1_init, l2_init]),
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": 100, "ftol": 1e-7},
            )
            best_l1, best_l2 = float(res_opt.x[0]), float(res_opt.x[1])
            k_params = 6  # 4 betas + 2 lambdas
        else:
            best_l1 = init_lambda1 or self.default_lambda1
            best_l2 = init_lambda2 or self.default_lambda2
            k_params = 4  # 4 betas (lambdas fixed)

        beta, y_hat, ssr = _fit_given_lambdas(best_l1, best_l2)
        rmse = float(np.sqrt(ssr / n_obs))

        tss = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = float(1.0 - (ssr / tss)) if tss > 0 else 1.0

        aic, bic = compute_aic_bic(ssr, n_obs, k_params)

        loocv_rmse = None
        if compute_loocv:
            loocv_rmse = self.cross_validate_day(y, m, best_l1, best_l2)

        return SvenssonFit(
            date=date_label,
            beta0=float(beta[0]),
            beta1=float(beta[1]),
            beta2=float(beta[2]),
            beta3=float(beta[3]),
            lambda1=best_l1,
            lambda2=best_l2,
            rmse=rmse,
            r_squared=r_squared,
            ssr=ssr,
            aic=aic,
            bic=bic,
            loocv_rmse=loocv_rmse,
            fitted_yields=y_hat,
            residuals=y - y_hat,
        )

    def cross_validate_day(
        self,
        y: np.ndarray,
        m: np.ndarray,
        lambda1: float,
        lambda2: float,
    ) -> float:
        """
        Compute Leave-One-Out Cross-Validation (LOOCV) RMSE for a single cross-section.
        """
        n = len(y)
        sq_errs = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            y_train, m_train = y[mask], m[mask]
            X_train = svensson_loadings(m_train, lambda1, lambda2, is_scale=self.is_scale)
            beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

            X_val = svensson_loadings(np.array([m[i]]), lambda1, lambda2, is_scale=self.is_scale)
            y_pred = float(np.squeeze(X_val @ beta))
            sq_errs.append((y[i] - y_pred) ** 2)

        return float(np.sqrt(np.mean(sq_errs)))

    def predict(
        self,
        maturities: np.ndarray,
        beta0: float,
        beta1: float,
        beta2: float,
        beta3: float,
        lambda1: Optional[float] = None,
        lambda2: Optional[float] = None,
    ) -> np.ndarray:
        """Evaluate fitted Svensson curve across arbitrary continuous maturities."""
        l1 = lambda1 or self.default_lambda1
        l2 = lambda2 or self.default_lambda2
        X = svensson_loadings(maturities, l1, l2, is_scale=self.is_scale)
        beta = np.array([beta0, beta1, beta2, beta3])
        return X @ beta

    def fit_panel(
        self,
        yield_df: pd.DataFrame,
        maturities_dict: Dict[str, float],
        date_col: str = "date",
        optimize_lambdas: bool = True,
        compute_loocv: bool = False,
    ) -> pd.DataFrame:
        """
        Fit Svensson model across every date in the yield panel.

        Args:
            yield_df: DataFrame containing date and yield tenor columns.
            maturities_dict: Mapping from column name to maturity in years.
            date_col: Date column identifier.
            optimize_lambdas: If True, optimizes lambda1 and lambda2 per date with warm-starting.
            compute_loocv: If True, computes out-of-sample LOOCV RMSE per date.

        Returns:
            DataFrame with fitted parameters, fit metrics (RMSE, R2, AIC, BIC, LOOCV),
            and per-tenor fitted values.
        """
        cols = [c for c in yield_df.columns if c in maturities_dict]
        cols = sorted(cols, key=lambda c: maturities_dict[c])
        maturities = np.array([maturities_dict[c] for c in cols])

        df = yield_df.copy()
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)

        dates = df[date_col].values
        Y_all = df[cols].to_numpy(dtype=float, na_value=np.nan)

        curr_l1 = self.default_lambda1
        curr_l2 = self.default_lambda2

        records = []
        for i in range(len(df)):
            d = dates[i]
            y = Y_all[i]
            valid_mask = ~np.isnan(y)
            if np.sum(valid_mask) >= 5:
                fit = self.fit_cross_section(
                    yields=y,
                    maturities=maturities,
                    date_label=str(d),
                    optimize_lambdas=optimize_lambdas,
                    init_lambda1=curr_l1 if optimize_lambdas else self.default_lambda1,
                    init_lambda2=curr_l2 if optimize_lambdas else self.default_lambda2,
                    compute_loocv=compute_loocv,
                )
                if optimize_lambdas:
                    # Warm-start next day's optimization with smoothed previous lambda
                    curr_l1 = 0.8 * curr_l1 + 0.2 * fit.lambda1
                    curr_l2 = 0.8 * curr_l2 + 0.2 * fit.lambda2

                rec = {
                    "date": pd.to_datetime(d),
                    "sv_level": fit.beta0,
                    "sv_slope": fit.beta1,
                    "sv_curv1": fit.beta2,
                    "sv_curv2": fit.beta3,
                    "sv_curv_composite": fit.beta2 + fit.beta3,
                    "sv_lambda1": fit.lambda1,
                    "sv_lambda2": fit.lambda2,
                    "sv_rmse": fit.rmse,
                    "sv_r_squared": fit.r_squared,
                    "sv_ssr": fit.ssr,
                    "sv_aic": fit.aic,
                    "sv_bic": fit.bic,
                    "sv_loocv_rmse": fit.loocv_rmse,
                }
                # Track per-tenor residuals
                for c, y_obs, y_fit in zip(cols, y, fit.fitted_yields):
                    rec[f"res_{c}"] = float(y_obs - y_fit) if not np.isnan(y_obs) else np.nan

                records.append(rec)

        return pd.DataFrame(records)
