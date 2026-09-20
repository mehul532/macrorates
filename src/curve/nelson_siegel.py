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
