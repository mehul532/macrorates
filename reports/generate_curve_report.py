"""
Research Report & Figure Generator for Milestone 2:
Yield Curve Factor Modeling (PCA, Static Nelson-Siegel, and Svensson Benchmark Evaluation).
"""

import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.curve import (
    StaticNelsonSiegel,
    SvenssonCurve,
    compare_pca_vs_ns,
    compare_svensson_on_date,
    evaluate_against_gsw,
    fit_pca,
    fit_static_nelson_siegel,
    nelson_siegel_loadings,
    svensson_loadings,
)
from src.data.pipeline import load_gsw_panel, load_yield_panel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure matplotlib style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def run_analysis_and_generate_figures():
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed yield and GSW panels...")
    ydf, ymeta = load_yield_panel()
    gdf, gmeta = load_gsw_panel()

    tenors = {
        "DGS1MO": 1.0 / 12.0,
        "DGS3MO": 0.25,
        "DGS6MO": 0.5,
        "DGS1": 1.0,
        "DGS2": 2.0,
        "DGS3": 3.0,
        "DGS5": 5.0,
        "DGS7": 7.0,
        "DGS10": 10.0,
        "DGS20": 20.0,
        "DGS30": 30.0,
    }

    # Filter common continuous sample (2006 to present, all 11 tenors available)
    ydf_sub = ydf[ydf["date"] >= "2006-02-15"].dropna().copy().reset_index(drop=True)
    logger.info("Analyzing %d clean trading days (2006 to 2026)...", len(ydf_sub))

    # 1. Fit PCA on levels and changes
    pca_levels = fit_pca(ydf_sub, tenors, on_changes=False)
    pca_changes = fit_pca(ydf_sub, tenors, on_changes=True)

    summary_levels = pca_levels.variance_explained_summary()
    summary_changes = pca_changes.variance_explained_summary()

    # 2. Fit Static Nelson-Siegel per day
    ns_factors = fit_static_nelson_siegel(ydf_sub, tenors, lambda_param=0.7308)

    # 3. Compare PCA vs. Nelson-Siegel
    comp = compare_pca_vs_ns(pca_levels, ns_factors)

    # 4. Evaluate against GSW Benchmark
    gsw_eval = evaluate_against_gsw(ydf_sub, ns_factors, gdf, tenors)

    # 5. Svensson Appendix Case Study (Debt-ceiling & curve inversion date: 2023-06-01)
    sv_comp = compare_svensson_on_date("2023-06-01", ydf, tenors)

    # -------------------------------------------------------------
    # FIGURE 1: PCA Variance Explained & Factor Loadings
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Scree Plot (Levels vs. Changes)
    components = [1, 2, 3]
    var_levels = [pca_levels.explained_variance_ratio[i] * 100 for i in range(3)]
    var_changes = [pca_changes.explained_variance_ratio[i] * 100 for i in range(3)]
    cum_levels = [pca_levels.cumulative_variance_ratio[i] * 100 for i in range(3)]
    cum_changes = [pca_changes.cumulative_variance_ratio[i] * 100 for i in range(3)]

    x = np.arange(len(components))
    width = 0.35

    rects1 = axes[0].bar(x - width/2, var_levels, width, label="Yield Levels", color="#1f77b4", alpha=0.85)
    rects2 = axes[0].bar(x + width/2, var_changes, width, label="Yield Changes (Δy)", color="#ff7f0e", alpha=0.85)

    # Plot cumulative lines
    axes[0].plot(x, cum_levels, "o--", color="#0d47a1", label="Cum. Levels ($R^2_k$)")
    axes[0].plot(x, cum_changes, "s--", color="#b71c1c", label="Cum. Changes ($R^2_k$)")

    for i, (cl, cc) in enumerate(zip(cum_levels, cum_changes)):
        axes[0].annotate(f"{cl:.1f}%", (x[i] - width/2, var_levels[i] + 2), ha="center", fontsize=9, fontweight="bold")
        axes[0].annotate(f"{cc:.1f}%", (x[i] + width/2, var_changes[i] + 2), ha="center", fontsize=9, fontweight="bold")

    axes[0].set_title("Variance Explained by Factor ($R^2_k$)")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Variance Explained (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)"])
    axes[0].set_ylim(0, 108)
    axes[0].legend(loc="center right")

    # Right: Factor Loadings vs Maturity
    mats = pca_levels.maturities
    axes[1].plot(mats, pca_levels.loadings[:, 0], "o-", color="#1f77b4", linewidth=2.2, label="PC1 (Level)")
    axes[1].plot(mats, pca_levels.loadings[:, 1], "s-", color="#2ca02c", linewidth=2.2, label="PC2 (Slope)")
    axes[1].plot(mats, pca_levels.loadings[:, 2], "^-", color="#d62728", linewidth=2.2, label="PC3 (Curvature)")

    # Overlay analytical Nelson-Siegel loadings for comparison
    dense_mats = np.linspace(0.083, 30.0, 100)
    dense_ns = nelson_siegel_loadings(dense_mats, 0.7308)
    # Scale NS loadings to align visually with unit-norm eigenvectors
    scale_lvl = np.mean(pca_levels.loadings[:, 0])
    scale_slp = np.std(pca_levels.loadings[:, 1]) / np.std(dense_ns[:, 1])
    scale_curv = np.max(pca_levels.loadings[:, 2]) / np.max(dense_ns[:, 2])

    axes[1].plot(dense_mats, dense_ns[:, 0] * scale_lvl, "--", color="#1f77b4", alpha=0.4, label="NS Level (Theo)")
    axes[1].plot(dense_mats, -dense_ns[:, 1] * scale_slp, "--", color="#2ca02c", alpha=0.4, label="NS -Slope (Theo)")
    axes[1].plot(dense_mats, dense_ns[:, 2] * scale_curv, "--", color="#d62728", alpha=0.4, label="NS Curvature (Theo)")

    axes[1].set_title("Empirical PCA Loadings vs. Maturity (Years)")
    axes[1].set_xlabel("Maturity (Years)")
    axes[1].set_ylabel("Factor Loading")
    axes[1].axhline(0, color="gray", linestyle=":", alpha=0.7)
    axes[1].set_xscale("log")
    axes[1].set_xticks([0.25, 0.5, 1, 2, 5, 10, 30])
    axes[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1].legend(loc="best", ncol=2, fontsize=8.5)

    plt.tight_layout()
    fig1_path = figures_dir / "pca_variance_and_loadings.png"
    plt.savefig(fig1_path)
    plt.close()
    logger.info("Saved Figure 1 to %s", fig1_path)

    # -------------------------------------------------------------
    # FIGURE 2: PCA vs. Nelson-Siegel Historical Factors
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    merged = comp["merged_df"]

    # Subplot 1: Level vs PC1
    color = "#1f77b4"
    axes[0].plot(merged["date"], merged["level"], label="Nelson-Siegel Level ($L_t$, %)", color=color, linewidth=1.5)
    axes[0].set_ylabel("NS Level (%)", color=color)
    axes[0].tick_params(axis="y", labelcolor=color)

    ax0_twin = axes[0].twinx()
    color_twin = "#333333"
    ax0_twin.plot(merged["date"], merged["PC1"], label="PCA PC1 Score", color=color_twin, linestyle="--", alpha=0.75)
    ax0_twin.set_ylabel("PC1 Score", color=color_twin)
    ax0_twin.grid(False)
    r_lvl = comp["correlations"]["level_vs_pc1"]
    axes[0].set_title(f"Level Factor: Static Nelson-Siegel vs. Empirical PC1 (Correlation = {r_lvl:.3f})")

    # Subplot 2: Slope vs PC2
    color = "#2ca02c"
    # Note: We plot -Slope (-S_t) because NS defines y(0)=L+S and y(inf)=L (so -S is long minus short)
    axes[1].plot(merged["date"], -merged["slope"], label="Nelson-Siegel Slope ($-S_t$, %)", color=color, linewidth=1.5)
    axes[1].set_ylabel("NS -Slope (%)", color=color)
    axes[1].tick_params(axis="y", labelcolor=color)

    ax1_twin = axes[1].twinx()
    ax1_twin.plot(merged["date"], merged["PC2"], label="PCA PC2 Score (Steepness)", color=color_twin, linestyle="--", alpha=0.75)
    ax1_twin.set_ylabel("PC2 Score", color=color_twin)
    ax1_twin.grid(False)
    r_slp = -comp["correlations"]["slope_vs_pc2"]
    axes[1].set_title(f"Slope Factor: Nelson-Siegel ($-S_t$) vs. Empirical PC2 (Correlation = {r_slp:.3f})")

    # Subplot 3: Curvature vs PC3
    color = "#d62728"
    axes[2].plot(merged["date"], merged["curvature"], label="Nelson-Siegel Curvature ($C_t$, %)", color=color, linewidth=1.5)
    axes[2].set_ylabel("NS Curvature (%)", color=color)
    axes[2].tick_params(axis="y", labelcolor=color)

    ax2_twin = axes[2].twinx()
    ax2_twin.plot(merged["date"], merged["PC3"], label="PCA PC3 Score", color=color_twin, linestyle="--", alpha=0.75)
    ax2_twin.set_ylabel("PC3 Score", color=color_twin)
    ax2_twin.grid(False)
    r_curv = comp["correlations"]["curvature_vs_pc3"]
    axes[2].set_title(f"Curvature Factor: Static Nelson-Siegel ($C_t$) vs. Empirical PC3 (Correlation = {r_curv:.3f})")
    axes[2].set_xlabel("Date")

    plt.tight_layout()
    fig2_path = figures_dir / "pca_vs_nelson_siegel_factors.png"
    plt.savefig(fig2_path)
    plt.close()
    logger.info("Saved Figure 2 to %s", fig2_path)

    # -------------------------------------------------------------
    # FIGURE 3: Nelson-Siegel vs. Federal Reserve GSW Benchmark
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5))

    # Top: Tracking error residuals over time
    ns_merged = pd.merge(ns_factors, gdf, on="date", how="inner")
    tenor_map = {"DGS2": ("SVENPY02", 2.0, "#2ca02c"), "DGS5": ("SVENPY05", 5.0, "#ff7f0e"),
                 "DGS10": ("SVENPY10", 10.0, "#1f77b4"), "DGS30": ("SVENPY30", 30.0, "#9467bd")}

    for tenor, (gsw_col, tau, c) in tenor_map.items():
        l = ns_merged["level"].values
        s = ns_merged["slope"].values
        curv = ns_merged["curvature"].values
        lam = ns_merged["lambda"].values
        x_val = lam * tau
        f1 = (1.0 - np.exp(-x_val)) / x_val
        f2 = f1 - np.exp(-x_val)
        fitted = l + s * f1 + curv * f2
        target = ns_merged[gsw_col].values
        resid_bp = (fitted - target) * 100.0  # in basis points
        axes[0].plot(ns_merged["date"], resid_bp, label=f"{tenor} ({tau}Y)", color=c, alpha=0.6, linewidth=1.0)

    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("Tracking Error Residuals: Static Nelson-Siegel vs. Fed GSW Benchmark ($\hat{y}_{NS} - y_{GSW}$)")
    axes[0].set_ylabel("Residual (Basis Points)")
    axes[0].set_ylim(-60, 60)
    axes[0].legend(loc="upper right", ncol=4)

    # Bottom: Bar chart of RMSE and Bias across maturities
    tenor_order = ["DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    rmse_vals = [gsw_eval["metrics_by_tenor"][t]["rmse_bp"] for t in tenor_order]
    bias_vals = [gsw_eval["metrics_by_tenor"][t]["bias_bp"] for t in tenor_order]

    x_pos = np.arange(len(tenor_order))
    width = 0.35

    axes[1].bar(x_pos - width/2, rmse_vals, width, label="RMSE (bps)", color="#388e3c", alpha=0.85)
    axes[1].bar(x_pos + width/2, bias_vals, width, label="Mean Bias (bps)", color="#d32f2f", alpha=0.85)

    for i in range(len(tenor_order)):
        axes[1].annotate(f"{rmse_vals[i]:.1f} bp", (x_pos[i] - width/2, rmse_vals[i] + 0.8), ha="center", fontsize=9, fontweight="bold")
        axes[1].annotate(f"{bias_vals[i]:.1f} bp", (x_pos[i] + width/2, max(0, bias_vals[i]) + 0.8), ha="center", fontsize=9)

    axes[1].set_title(f"Fit Quality by Maturity (Overall Benchmark Tracking RMSE = {gsw_eval['overall_rmse_bp']:.1f} bps)")
    axes[1].set_xlabel("Maturity Tenor")
    axes[1].set_ylabel("Basis Points (bps)")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"{t}\n({gsw_eval['metrics_by_tenor'][t]['maturity_years']}Y)" for t in tenor_order])
    axes[1].axhline(0, color="gray", linestyle=":", alpha=0.7)
    axes[1].set_ylim(-20, 26)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig3_path = figures_dir / "ns_vs_gsw_benchmark.png"
    plt.savefig(fig3_path)
    plt.close()
    logger.info("Saved Figure 3 to %s", fig3_path)

    # -------------------------------------------------------------
    # FIGURE 4: Svensson vs. Nelson-Siegel (Appendix Case Study)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    obs_mats = sv_comp["maturities"]
    obs_yields = sv_comp["observed_yields"]
    dense_tau = np.linspace(0.083, 30.0, 300)

    # Reconstruct continuous curves
    ns_fit = sv_comp["ns_fit"]
    sv_fit = sv_comp["svensson_fit"]

    ns_pred = StaticNelsonSiegel().predict(dense_tau, ns_fit.level, ns_fit.slope, ns_fit.curvature, ns_fit.lambda_param)
    sv_pred = SvenssonCurve().predict(dense_tau, sv_fit.beta0, sv_fit.beta1, sv_fit.beta2, sv_fit.beta3, sv_fit.lambda1, sv_fit.lambda2)

    # Left: Yield curve fit
    axes[0].scatter(obs_mats, obs_yields, color="black", s=60, zorder=5, label="Observed CMT Yields")
    axes[0].plot(dense_tau, ns_pred, "--", color="#1f77b4", linewidth=2.2,
                 label=f"Nelson-Siegel (RMSE: {sv_comp['ns_rmse_bp']} bps)")
    axes[0].plot(dense_tau, sv_pred, "-", color="#d62728", linewidth=2.5,
                 label=f"Svensson 6-Param (RMSE: {sv_comp['svensson_rmse_bp']} bps)")

    axes[0].set_title(f"Svensson vs. Nelson-Siegel: {sv_comp['date']}\n(Complex Inversion & Debt-Ceiling Front-End Spike)")
    axes[0].set_xlabel("Maturity (Years)")
    axes[0].set_ylabel("Yield (%)")
    axes[0].set_xscale("log")
    axes[0].set_xticks([0.083, 0.25, 0.5, 1, 2, 5, 10, 30])
    axes[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0].legend(loc="lower right")

    # Right: Residuals by maturity
    ns_res_bp = (ns_fit.fitted_yields - obs_yields) * 100
    sv_res_bp = (sv_fit.fitted_yields - obs_yields) * 100

    x_ten = np.arange(len(obs_mats))
    axes[1].bar(x_ten - width/2, ns_res_bp, width, label=f"Nelson-Siegel (RMSE: {sv_comp['ns_rmse_bp']} bp)", color="#1f77b4", alpha=0.85)
    axes[1].bar(x_ten + width/2, sv_res_bp, width, label=f"Svensson (RMSE: {sv_comp['svensson_rmse_bp']} bp)", color="#d62728", alpha=0.85)

    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1.0)
    axes[1].set_title(f"Residual Error Comparison (Svensson Improves by {sv_comp['improvement_bp']} bps)")
    axes[1].set_xlabel("Maturity Tenor")
    axes[1].set_ylabel("Residual Error (Basis Points)")
    axes[1].set_xticks(x_ten)
    axes[1].set_xticklabels([f"{m:.2f}Y" if m < 1 else f"{int(m)}Y" for m in obs_mats], rotation=45)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig4_path = figures_dir / "svensson_vs_ns_appendix.png"
    plt.savefig(fig4_path)
    plt.close()
    logger.info("Saved Figure 4 to %s", fig4_path)

    # -------------------------------------------------------------
    # SAVE METRICS TO JSON
    # -------------------------------------------------------------
    metrics_summary = {
        "analysis_date_range": {
            "start": ydf_sub["date"].min().strftime("%Y-%m-%d"),
            "end": ydf_sub["date"].max().strftime("%Y-%m-%d"),
            "n_trading_days": len(ydf_sub),
        },
        "pca_variance_explained": {
            "levels": summary_levels,
            "changes": summary_changes,
        },
        "pca_vs_nelson_siegel_correlations": comp["correlations"],
        "gsw_benchmark_evaluation": gsw_eval,
        "svensson_case_study": {
            "date": sv_comp["date"],
            "ns_rmse_bp": sv_comp["ns_rmse_bp"],
            "svensson_rmse_bp": sv_comp["svensson_rmse_bp"],
            "improvement_bp": sv_comp["improvement_bp"],
        },
    }

    metrics_path = Path("reports/curve_summary_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info("Saved summary metrics to %s", metrics_path)

    return metrics_summary


if __name__ == "__main__":
    run_analysis_and_generate_figures()
