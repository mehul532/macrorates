"""Unified term structure factor modeling: PCA, Static Nelson-Siegel, and Svensson."""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.curve.nelson_siegel import (
    NelsonSiegelFit,
    StaticNelsonSiegel,
    curvature_peak_maturity,
    nelson_siegel_loadings,
)
from src.curve.pca import PCAResult, YieldCurvePCA
from src.curve.svensson import SvenssonCurve, SvenssonFit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fit_pca(
    yield_df: pd.DataFrame,
    maturities_dict: Dict[str, float],
    on_changes: bool = False,
    n_components: int = 3,
    date_col: str = "date",
) -> PCAResult:
    """
    Extract Level, Slope, and Curvature via Principal Component Analysis.
    
    Args:
        yield_df: Yield panel DataFrame.
        maturities_dict: Mapping from series name to maturity in years.
        on_changes: If True, performs PCA on daily yield differences (Delta y).
        n_components: Number of components to retain (default 3).
        date_col: Date column name.
    """
    pca_model = YieldCurvePCA(n_components=n_components)
    return pca_model.fit(
        yield_df=yield_df,
        maturities_dict=maturities_dict,
        on_changes=on_changes,
        date_col=date_col,
    )


def fit_static_nelson_siegel(
    yield_df: pd.DataFrame,
    maturities_dict: Dict[str, float],
    lambda_param: float = 0.7308,
    optimize_lambda: bool = False,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Fit cross-sectional Static Nelson-Siegel model for each date in panel.
    
    Returns:
        DataFrame with columns ['date', 'level', 'slope', 'curvature', 'lambda', 'rmse', 'r_squared'].
    """
    ns_model = StaticNelsonSiegel(lambda_param=lambda_param)
    return ns_model.fit_panel(
        yield_df=yield_df,
        maturities_dict=maturities_dict,
        date_col=date_col,
        optimize_lambda=optimize_lambda,
    )


def compare_pca_vs_ns(
    pca_levels: PCAResult,
    ns_factors: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Quantitatively compare PCA factors against Nelson-Siegel factors.
    
    Computes correlation matrix between (PC1, PC2, PC3) and (Level, Slope, Curvature),
    merged panel, and statistical summary.
    """
    pca_scores = pca_levels.scores.copy()
    pca_scores["date"] = pd.to_datetime(pca_scores["date"])

    ns_df = ns_factors.copy()
    ns_df["date"] = pd.to_datetime(ns_df["date"])

    merged = pd.merge(pca_scores, ns_df, on="date", how="inner").dropna()

    corr_matrix = merged[["PC1", "PC2", "PC3", "level", "slope", "curvature"]].corr()

    # Economic factor correlations
    correlations = {
        "level_vs_pc1": float(corr_matrix.loc["level", "PC1"]),
        "slope_vs_pc2": float(corr_matrix.loc["slope", "PC2"]),
        "curvature_vs_pc3": float(corr_matrix.loc["curvature", "PC3"]),
        "full_matrix": corr_matrix.to_dict(),
    }

    return {
        "correlations": correlations,
        "merged_df": merged,
        "n_observations": len(merged),
    }


def evaluate_against_gsw(
    yield_df: pd.DataFrame,
    ns_factors: pd.DataFrame,
    gsw_df: pd.DataFrame,
    maturities_dict: Dict[str, float],
    test_tenors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate Static Nelson-Siegel fit quality against the Fed GSW benchmark dataset.
    
    Computes tracking error RMSE, mean absolute error (MAE), and bias across tenors.
    """
    if test_tenors is None:
        test_tenors = ["DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]

    ns_df = ns_factors.copy()
    ns_df["date"] = pd.to_datetime(ns_df["date"])

    gsw_clean = gsw_df.copy()
    gsw_clean["date"] = pd.to_datetime(gsw_clean["date"])

    # Map CMT tenors to corresponding GSW par yields (SVENPYxx)
    tenor_to_gsw_col = {
        "DGS1": "SVENPY01",
        "DGS2": "SVENPY02",
        "DGS3": "SVENPY03",
        "DGS5": "SVENPY05",
        "DGS7": "SVENPY07",
        "DGS10": "SVENPY10",
        "DGS20": "SVENPY20",
        "DGS30": "SVENPY30",
    }

    ns = StaticNelsonSiegel()
    eval_records = []

    merged_panel = pd.merge(ns_df, gsw_clean, on="date", how="inner")

    metrics_by_tenor = {}
    for tenor in test_tenors:
        gsw_col = tenor_to_gsw_col.get(tenor)
        if gsw_col not in merged_panel.columns:
            continue

        tau = maturities_dict[tenor]
        # Reconstruct NS predicted yield for this tenor
        l = merged_panel["level"].values
        s = merged_panel["slope"].values
        c = merged_panel["curvature"].values
        lam = merged_panel["lambda"].values

        # Vectorized NS evaluation
        x = lam * tau
        f1 = (1.0 - np.exp(-x)) / x
        f2 = f1 - np.exp(-x)
        ns_fitted = l + s * f1 + c * f2

        gsw_target = merged_panel[gsw_col].values

        diff = ns_fitted - gsw_target
        valid = ~np.isnan(diff)
        diff_valid = diff[valid]

        rmse = float(np.sqrt(np.mean(diff_valid ** 2)))
        mae = float(np.mean(np.abs(diff_valid)))
        bias = float(np.mean(diff_valid))

        metrics_by_tenor[tenor] = {
            "maturity_years": tau,
            "gsw_benchmark_column": gsw_col,
            "rmse_bp": round(rmse * 100, 2),  # in basis points
            "mae_bp": round(mae * 100, 2),
            "bias_bp": round(bias * 100, 2),
            "sample_count": int(np.sum(valid)),
        }

    overall_rmse_bp = float(np.mean([m["rmse_bp"] for m in metrics_by_tenor.values()]))

    return {
        "overall_rmse_bp": overall_rmse_bp,
        "metrics_by_tenor": metrics_by_tenor,
        "comparison_dates_count": len(merged_panel),
    }


def compare_svensson_on_date(
    date_str: str,
    yield_df: pd.DataFrame,
    maturities_dict: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compare Svensson vs. Nelson-Siegel on a specific date (Appendix demonstration).
    """
    row = yield_df[yield_df["date"] == pd.to_datetime(date_str)]
    if len(row) == 0:
        raise ValueError(f"Date {date_str} not found in yield panel.")

    cols = [c for c in yield_df.columns if c in maturities_dict]
    cols = sorted(cols, key=lambda c: maturities_dict[c])
    maturities = np.array([maturities_dict[c] for c in cols])
    yields = row[cols].values.flatten().astype(float)

    ns_model = StaticNelsonSiegel()
    ns_fit = ns_model.fit_cross_section(yields, maturities, date_label=date_str, optimize_lambda=True)

    sv_model = SvenssonCurve()
    sv_fit = sv_model.fit_cross_section(yields, maturities, date_label=date_str)

    return {
        "date": date_str,
        "observed_yields": yields,
        "maturities": maturities,
        "ns_fit": ns_fit,
        "svensson_fit": sv_fit,
        "ns_rmse_bp": round(ns_fit.rmse * 100, 2),
        "svensson_rmse_bp": round(sv_fit.rmse * 100, 2),
        "improvement_bp": round((ns_fit.rmse - sv_fit.rmse) * 100, 2),
    }
