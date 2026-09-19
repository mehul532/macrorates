"""Svensson 6-parameter term structure model (Appendix)."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import minimize


@dataclass
class SvenssonFit:
    """Container for a single-day Svensson 6-parameter fit."""

    date: Optional[str]
    beta0: float  # Long-term level
    beta1: float  # Slope
    beta2: float  # First curvature (first hump)
    beta3: float  # Second curvature (second hump)
    lambda1: float  # Decay 1 (1 / tau1)
    lambda2: float  # Decay 2 (1 / tau2)
    rmse: float
    r_squared: float
    fitted_yields: np.ndarray
    residuals: np.ndarray


def svensson_loadings(
    maturities: np.ndarray, lambda1: float, lambda2: float
) -> np.ndarray:
    """
    Compute Svensson 6-parameter factor loadings.
    
    Loadings:
      f0(tau) = 1.0
      f1(tau) = (1 - exp(-lambda1 * tau)) / (lambda1 * tau)
      f2(tau) = f1(tau) - exp(-lambda1 * tau)
      f3(tau) = (1 - exp(-lambda2 * tau)) / (lambda2 * tau) - exp(-lambda2 * tau)
    """
    tau = np.asarray(maturities, dtype=float)
    x1 = np.where(lambda1 * tau == 0, 1e-8, lambda1 * tau)
    x2 = np.where(lambda2 * tau == 0, 1e-8, lambda2 * tau)

    f0 = np.ones_like(tau)
    f1 = (1.0 - np.exp(-x1)) / x1
    f2 = f1 - np.exp(-x1)
    f3 = (1.0 - np.exp(-x2)) / x2 - np.exp(-x2)

    return np.column_stack([f0, f1, f2, f3])


class SvenssonCurve:
    """
    Svensson 6-parameter model for complex term structures with double-humps.
    
    Treated as a research appendix to demonstrate cases where a second curvature
    factor provides incremental explanatory power over standard Nelson-Siegel.
    """

    def __init__(self, init_lambda1: float = 0.73, init_lambda2: float = 0.20):
        self.init_lambda1 = init_lambda1
        self.init_lambda2 = init_lambda2

    def fit_cross_section(
        self,
        yields: np.ndarray,
        maturities: np.ndarray,
        date_label: Optional[str] = None,
    ) -> SvenssonFit:
        """
        Fit Svensson curve to a single cross-section using conditional OLS inside 2D optimization.
        """
        mask = ~np.isnan(yields) & ~np.isnan(maturities)
        y = yields[mask]
        m = maturities[mask]

        if len(y) < 5:
            raise ValueError(f"At least 5 non-NaN observations required for Svensson fit, got {len(y)}")

        def _cost(params: np.ndarray) -> float:
            l1, l2 = params
            if l1 <= 0.01 or l2 <= 0.01 or abs(l1 - l2) < 0.05:
                return 1e6
            X = svensson_loadings(m, l1, l2)
            beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            res = y - (X @ beta)
            return float(np.sum(res ** 2))

        init_params = np.array([self.init_lambda1, self.init_lambda2])
        bounds = [(0.05, 3.0), (0.05, 3.0)]
        opt_res = minimize(_cost, init_params, bounds=bounds, method="L-BFGS-B")

        best_l1, best_l2 = opt_res.x
        X = svensson_loadings(m, best_l1, best_l2)
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ssr = np.sum((y - y_hat) ** 2)
        rmse = np.sqrt(ssr / len(y))

        tss = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - (ssr / tss) if tss > 0 else 1.0

        return SvenssonFit(
            date=date_label,
            beta0=float(beta[0]),
            beta1=float(beta[1]),
            beta2=float(beta[2]),
            beta3=float(beta[3]),
            lambda1=float(best_l1),
            lambda2=float(best_l2),
            rmse=float(rmse),
            r_squared=float(r_squared),
            fitted_yields=y_hat,
            residuals=y - y_hat,
        )

    def predict(
        self,
        maturities: np.ndarray,
        beta0: float,
        beta1: float,
        beta2: float,
        beta3: float,
        lambda1: float,
        lambda2: float,
    ) -> np.ndarray:
        """Evaluate fitted Svensson curve across arbitrary maturities."""
        X = svensson_loadings(maturities, lambda1, lambda2)
        beta = np.array([beta0, beta1, beta2, beta3])
        return X @ beta
