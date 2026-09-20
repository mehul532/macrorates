"""
Generates Macroeconomic Surprises Report Figures and Metrics.

Produces:
1. data/processed/factor_panel.parquet (Consolidated curve factors 2006-2026)
2. reports/figures/irf_cpi_surprise_level_slope_curvature.png (Standout 3-panel IRF figure)
3. reports/macro_summary_metrics.json (Scorecard, contemporaneous HAC tables, and IRFs)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

# Configure non-interactive matplotlib backend with writable cache
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
Path("/tmp/mpl").mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.curve.curve import fit_pca, fit_static_nelson_siegel
from src.macro.macro_surprises import (
    MacroEventHarmonizer,
    MacroRegressionEngine,
    plot_cpi_impulse_response,
)
from src.state_space.state_space import estimate_and_filter_state_space
from src.backtest.walk_forward import get_git_commit_hash, get_file_checksum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_consolidated_factor_panel(
    yield_path: Path = Path("data/processed/yield_panel.parquet"),
    start_date: str = "2006-01-01",
) -> pd.DataFrame:
    """Build unified panel containing PCA, Static NS, and Kalman factors."""
    yield_df = pd.read_parquet(yield_path)
    yield_df["date"] = pd.to_datetime(yield_df["date"])

    maturities = {
        "DGS1MO": 1 / 12,
        "DGS3MO": 3 / 12,
        "DGS6MO": 6 / 12,
        "DGS1": 1.0,
        "DGS2": 2.0,
        "DGS3": 3.0,
        "DGS5": 5.0,
        "DGS7": 7.0,
        "DGS10": 10.0,
        "DGS20": 20.0,
        "DGS30": 30.0,
    }

    sub_yields = yield_df[yield_df["date"] >= pd.to_datetime(start_date)].copy().reset_index(drop=True)

    logger.info("Extracting PCA factors...")
    pca_res = fit_pca(sub_yields, maturities, date_col="date")
    pca_df = pca_res.scores.copy()
    pca_df["date"] = pd.to_datetime(pca_df["date"])
    pca_df = pca_df.rename(columns={"PC1": "pca_level", "PC2": "pca_slope", "PC3": "pca_curvature"})

    logger.info("Extracting Static Nelson-Siegel factors...")
    ns_df = fit_static_nelson_siegel(sub_yields, maturities, date_col="date")
    ns_df["date"] = pd.to_datetime(ns_df["date"])
    ns_df = ns_df.rename(columns={"level": "ns_level", "slope": "ns_slope", "curvature": "ns_curvature"})

    logger.info("Extracting Kalman Filtered factors...")
    ss_res = estimate_and_filter_state_space(sub_yields, maturities, use_mle_optimization=False)
    kf_df = ss_res.filtered_states.copy()
    kf_df["date"] = pd.to_datetime(kf_df["date"])
    kf_df = kf_df.rename(columns={"level": "kf_level", "slope": "kf_slope", "curvature": "kf_curvature"})

    factors_df = pd.merge(ns_df[["date", "ns_level", "ns_slope", "ns_curvature"]], kf_df[["date", "kf_level", "kf_slope", "kf_curvature"]], on="date")
    factors_df = pd.merge(factors_df, pca_df[["date", "pca_level", "pca_slope", "pca_curvature"]], on="date")
    factors_df = factors_df.sort_values("date").reset_index(drop=True)

    # Compute daily close-to-close differences in basis points (Delta Factor * 100)
    for col in [
        "ns_level", "ns_slope", "ns_curvature",
        "kf_level", "kf_slope", "kf_curvature",
        "pca_level", "pca_slope", "pca_curvature",
    ]:
        factors_df[f"d_{col}"] = factors_df[col].diff() * 100.0

    out_factor_parquet = Path("data/processed/factor_panel.parquet")
    factors_df.to_parquet(out_factor_parquet, index=False)
    logger.info("Saved consolidated factor panel to %s (%d rows)", out_factor_parquet, len(factors_df))
    return factors_df


def run_all_contemporaneous_regressions(
    factors_df: pd.DataFrame,
    surprises_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Run contemporaneous HAC regressions across all 5 macro indicators and factor types."""
    indicators = ["CPI", "CORE_CPI", "NFP", "UNEMP", "FOMC"]
    surprise_types = ["surprise_ann", "surprise_model"]
    factor_diff_cols = [
        "d_kf_level", "d_kf_slope", "d_kf_curvature",
        "d_ns_level", "d_ns_slope", "d_ns_curvature",
        "d_pca_level", "d_pca_slope", "d_pca_curvature",
    ]

    regression_results = {}

    for ind in indicators:
        merged = MacroEventHarmonizer.align_events_with_factors(factors_df, surprises_df, ind)
        regression_results[ind] = {}

        for stype in surprise_types:
            regression_results[ind][stype] = {}
            for fcol in factor_diff_cols:
                res = MacroRegressionEngine.run_contemporaneous_regression(
                    merged,
                    factor_diff_col=fcol,
                    surprise_col=stype,
                    maxlags=5,
                )
                regression_results[ind][stype][fcol] = res

    return regression_results


def run_local_projections_and_plot(
    factors_df: pd.DataFrame,
    surprises_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Run Jordà local projections for CPI surprise and plot standout IRF figure."""
    # Run LP on Kalman factors
    irf_ann_list = []
    irf_mod_list = []

    for factor in ["kf_level", "kf_slope", "kf_curvature"]:
        base_name = factor.replace("kf_", "")
        lp_ann = MacroRegressionEngine.run_local_projections(
            factors_df,
            surprises_df,
            indicator="CPI",
            factor_col=factor,
            surprise_col="surprise_ann",
            horizons=[0, 1, 2, 5, 10],
        )
        lp_ann["factor"] = base_name
        irf_ann_list.append(lp_ann)

        lp_mod = MacroRegressionEngine.run_local_projections(
            factors_df,
            surprises_df,
            indicator="CPI",
            factor_col=factor,
            surprise_col="surprise_model",
            horizons=[0, 1, 2, 5, 10],
        )
        lp_mod["factor"] = base_name
        irf_mod_list.append(lp_mod)

    irf_ann_df = pd.concat(irf_ann_list, ignore_index=True)
    irf_mod_df = pd.concat(irf_mod_list, ignore_index=True)

    fig_path = plot_cpi_impulse_response(
        irf_ann_df,
        irf_mod_df,
        out_path=Path("reports/figures/irf_cpi_surprise_level_slope_curvature.png"),
    )

    return irf_ann_df, irf_mod_df, fig_path


def main():
    logger.info("Starting Macroeconomic Surprises & IRF Generation Pipeline...")

    factors_df = build_consolidated_factor_panel()
    surprises_df = pd.read_parquet("data/processed/macro_surprises.parquet")

    logger.info("Running contemporaneous HAC regressions across all indicators and factors...")
    reg_results = run_all_contemporaneous_regressions(factors_df, surprises_df)

    logger.info("Running Jordà local projections and plotting standout CPI IRF figure...")
    irf_ann_df, irf_mod_df, fig_path = run_local_projections_and_plot(factors_df, surprises_df)

    # Save summary metrics JSON with provenance
    summary_data = {
        "run_metadata": {
            "git_commit": get_git_commit_hash(),
            "data_checksums": {
                "yield_panel": get_file_checksum("data/processed/yield_panel.parquet"),
                "factor_panel": get_file_checksum("data/processed/factor_panel.parquet"),
                "macro_surprises": get_file_checksum("data/processed/macro_surprises.parquet"),
            },
        },
        "contemporaneous_regressions": reg_results,
        "cpi_irf_announcement": irf_ann_df.to_dict(orient="records"),
        "cpi_irf_model_innovation": irf_mod_df.to_dict(orient="records"),
        "irf_figure_path": str(fig_path),
    }

    out_metrics_path = Path("reports/macro_summary_metrics.json")
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("Saved macro summary metrics to %s", out_metrics_path)
    logger.info("Macro report generation complete!")


if __name__ == "__main__":
    main()
