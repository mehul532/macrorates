"""
Svensson 4-Factor (6-Parameter) Dynamic Term Structure Model.

Top-level module delivering the Svensson curve modeling pipeline:
  y(tau) = beta0 + beta1 * [(1 - exp(-tau/lambda1)) / (tau/lambda1)]
                 + beta2 * [(1 - exp(-tau/lambda1)) / (tau/lambda1) - exp(-tau/lambda1)]
                 + beta3 * [(1 - exp(-tau/lambda2)) / (tau/lambda2) - exp(-tau/lambda2)]

Exposes:
- SvenssonCurve: Cross-sectional and daily panel fitter with joint lambda1/lambda2 optimization.
- SvenssonFit: Container for fitted parameters, diagnostics (SSR, AIC, BIC, LOOCV), and residuals.
- svensson_loadings: Matrix of factor loadings for arbitrary maturities.
- compute_aic_bic: Information criteria computation.
"""

from pathlib import Path
import sys

# Ensure package imports resolve
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.curve.svensson import (
    SvenssonCurve,
    SvenssonFit,
    svensson_loadings,
    compute_aic_bic,
)

__all__ = [
    "SvenssonCurve",
    "SvenssonFit",
    "svensson_loadings",
    "compute_aic_bic",
]

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    print("Running Svensson Term Structure Model validation...")
    yield_path = root_dir / "data" / "processed" / "yield_panel.parquet"
    if yield_path.exists():
        df = pd.read_parquet(yield_path)
        tenor_map = {
            "DGS1MO": 1/12, "DGS3MO": 3/12, "DGS6MO": 6/12,
            "DGS1": 1.0, "DGS2": 2.0, "DGS3": 3.0, "DGS5": 5.0,
            "DGS7": 7.0, "DGS10": 10.0, "DGS20": 20.0, "DGS30": 30.0
        }
        sv = SvenssonCurve()
        print(f"Loaded {len(df)} days of yields. Testing single cross-section fit...")
        latest_row = df.dropna().iloc[-1]
        y_vals = np.array([latest_row[col] for col in tenor_map.keys()])
        m_vals = np.array(list(tenor_map.values()))
        fit = sv.fit_cross_section(y_vals, m_vals, date_label=str(latest_row["date"]), compute_loocv=True)
        print(f"Date: {fit.date}")
        print(f"  Level (beta0): {fit.beta0:.4f}, Slope (beta1): {fit.beta1:.4f}")
        print(f"  Curv1 (beta2): {fit.beta2:.4f}, Curv2 (beta3): {fit.beta3:.4f}")
        print(f"  lambda1: {fit.lambda1:.2f} yr, lambda2: {fit.lambda2:.2f} yr")
        print(f"  In-sample RMSE: {fit.rmse*100:.2f} bp, LOOCV RMSE: {fit.loocv_rmse*100:.2f} bp")
        print(f"  AIC: {fit.aic:.2f}, BIC: {fit.bic:.2f}")
    else:
        print("Yield panel not found at expected path.")
