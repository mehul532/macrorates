"""
Diagnostic Script & Figure Generator for Milestone 3:
Linear Gaussian Dynamic Nelson-Siegel State-Space Estimation vs. Two-Step OLS.
"""

import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.curve import fit_static_nelson_siegel
from src.curve.nelson_siegel import nelson_siegel_loadings
from src.data.pipeline import load_yield_panel
from src.state_space import estimate_and_filter_state_space, evaluate_ols_vs_kalman

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Matplotlib formatting
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


def generate_state_space_figures_and_metrics():
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    ydf, _ = load_yield_panel()
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

    # Filter 2006 to present
    ydf_sub = ydf[ydf["date"] >= "2006-02-15"].dropna().copy().reset_index(drop=True)
    logger.info("Fitting Static Nelson-Siegel OLS and Dynamic State-Space on %d days...", len(ydf_sub))

    ols_factors = fit_static_nelson_siegel(ydf_sub, tenors, lambda_param=0.7308)
    ss_res = estimate_and_filter_state_space(ydf_sub, tenors, lambda_param=0.7308, use_mle_optimization=True)
    eval_res = evaluate_ols_vs_kalman(ydf_sub, tenors, ols_factors, ss_res)

    dates = pd.to_datetime(ydf_sub["date"])

    # -------------------------------------------------------------
    # FIGURE 1: Factor Trajectories: OLS vs. Filtered vs. Smoothed
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    factors = [("level", "Level ($L_t$)", "#1f77b4"),
               ("slope", "Slope ($S_t$)", "#2ca02c"),
               ("curvature", "Curvature ($C_t$)", "#d62728")]

    for i, (col, title, c) in enumerate(factors):
        axes[i].plot(dates, ols_factors[col], label="Two-Step OLS (Unconstrained)", color="gray", alpha=0.45, linewidth=0.9)
        axes[i].plot(dates, ss_res.filtered_states[col], label="Kalman Filtered ($t|t$)", color=c, alpha=0.75, linewidth=1.2)
        axes[i].plot(dates, ss_res.smoothed_states[col], label="Kalman Smoothed ($t|T$)", color="black", linewidth=1.4)
        axes[i].set_title(f"{title}: OLS vs. Filtered vs. Smoothed Factor Path")
        axes[i].set_ylabel(f"{title} (%)")
        axes[i].legend(loc="upper right")

    axes[2].set_xlabel("Date")
    plt.tight_layout()
    fig1_path = figures_dir / "kalman_factor_trajectories.png"
    plt.savefig(fig1_path)
    plt.close()
    logger.info("Saved Figure 1 to %s", fig1_path)

    # -------------------------------------------------------------
    # FIGURE 2: Factor Stability & Noise Reduction (Daily Changes)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for i, (col, title, c) in enumerate(factors):
        d_ols = ols_factors[col].diff().dropna()
        d_filt = ss_res.filtered_states[col].diff().dropna()
        d_smooth = ss_res.smoothed_states[col].diff().dropna()

        axes[i].hist(d_ols, bins=60, range=(-0.35, 0.35), density=True, alpha=0.35, color="gray", label=f"OLS (σ={d_ols.std():.3f})")
        axes[i].hist(d_filt, bins=60, range=(-0.35, 0.35), density=True, alpha=0.5, color=c, label=f"Filtered (σ={d_filt.std():.3f})")
        axes[i].hist(d_smooth, bins=60, range=(-0.35, 0.35), density=True, alpha=0.65, histtype="step", linewidth=1.8, color="black", label=f"Smoothed (σ={d_smooth.std():.3f})")

        axes[i].set_title(f"Δ {title} Distribution")
        axes[i].set_xlabel("Daily Factor Change (pp)")
        axes[i].set_ylabel("Empirical Density")
        axes[i].legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    fig2_path = figures_dir / "kalman_noise_reduction.png"
    plt.savefig(fig2_path)
    plt.close()
    logger.info("Saved Figure 2 to %s", fig2_path)

    # -------------------------------------------------------------
    # FIGURE 3: Residual Whiteness & Autocorrelation Checks
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    tenors_to_plot = ["DGS2", "DGS5", "DGS10", "DGS30"]
    cols_all = [c for c in ydf_sub.columns if c in tenors]
    cols_all = sorted(cols_all, key=lambda c: tenors[c])

    # Plot ACF of residuals for OLS vs Kalman Smoothed
    max_lag = 20
    lags = np.arange(1, max_lag + 1)

    for tenor in tenors_to_plot:
        idx = cols_all.index(tenor)
        ols_res_series = pd.Series(ols_factors["rmse"].values).dropna()  # proxy
        res_ols = pd.Series(ydf_sub[tenor].values - (ols_factors[["level", "slope", "curvature"]].values @ nelson_siegel_loadings(np.array([tenors[tenor]]), 0.7308)[0]))
        res_kf = pd.Series(ss_res.residuals_smoothed[:, idx])

        acf_ols = [res_ols.autocorr(lag=l) for l in lags]
        acf_kf = [res_kf.autocorr(lag=l) for l in lags]

        axes[0].plot(lags, acf_ols, "--o", markersize=3, label=f"{tenor} OLS")
        axes[1].plot(lags, acf_kf, "-s", markersize=3, label=f"{tenor} Kalman Smoothed")

    axes[0].axhline(0, color="gray", linestyle=":")
    axes[0].axhline(1.96 / np.sqrt(len(ydf_sub)), color="red", linestyle="--", alpha=0.6, label="95% CI (White Noise)")
    axes[0].set_title("Autocorrelation of OLS Measurement Residuals")
    axes[0].set_xlabel("Lag (Days)")
    axes[0].set_ylabel("Autocorrelation ρ(k)")
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].legend(loc="upper right", fontsize=8.5)

    axes[1].axhline(0, color="gray", linestyle=":")
    axes[1].axhline(1.96 / np.sqrt(len(ydf_sub)), color="red", linestyle="--", alpha=0.6, label="95% CI (White Noise)")
    axes[1].set_title("Autocorrelation of Kalman Smoothed Residuals")
    axes[1].set_xlabel("Lag (Days)")
    axes[1].set_ylabel("Autocorrelation ρ(k)")
    axes[1].set_ylim(-0.1, 1.05)
    axes[1].legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    fig3_path = figures_dir / "kalman_residual_whiteness.png"
    plt.savefig(fig3_path)
    plt.close()
    logger.info("Saved Figure 3 to %s", fig3_path)

    # -------------------------------------------------------------
    # FIGURE 4: Out-of-Sample Forecasting Horse Race
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))

    horizons = ["1-Day (h=1)", "5-Day (h=5)", "21-Day (h=21)"]
    rw_rmse = [eval_res["forecasting"][f"h_{h}"]["random_walk_rmse_bp"] for h in [1, 5, 21]]
    ols_rmse = [eval_res["forecasting"][f"h_{h}"]["ols_var_rmse_bp"] for h in [1, 5, 21]]
    kf_rmse = [eval_res["forecasting"][f"h_{h}"]["kalman_filter_rmse_bp"] for h in [1, 5, 21]]

    x = np.arange(len(horizons))
    width = 0.26

    r1 = ax.bar(x - width, rw_rmse, width, label="Random Walk Baseline ($y_{t+h} = y_t$)", color="#2ca02c", alpha=0.85)
    r2 = ax.bar(x, ols_rmse, width, label="Two-Step OLS + VAR(1)", color="#1f77b4", alpha=0.85)
    r3 = ax.bar(x + width, kf_rmse, width, label="Dynamic State-Space (Kalman Filter)", color="#d62728", alpha=0.85)

    for rects in [r1, r2, r3]:
        for rect in rects:
            h_val = rect.get_height()
            ax.annotate(f"{h_val:.1f}", (rect.get_x() + rect.get_width() / 2, h_val + 0.5), ha="center", fontsize=9, fontweight="bold")

    ax.set_title("Out-of-Sample Yield Forecasting RMSE by Horizon (Basis Points)\n[Empirical Finding: Random Walk Dominates at Short Horizons]")
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Out-of-Sample RMSE (bps)")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylim(0, max(kf_rmse) + 6)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig4_path = figures_dir / "forecasting_rmse_comparison.png"
    plt.savefig(fig4_path)
    plt.close()
    logger.info("Saved Figure 4 to %s", fig4_path)

    # -------------------------------------------------------------
    # SAVE METRICS TO JSON
    # -------------------------------------------------------------
    summary_path = Path("reports/state_space_summary_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_res, f, indent=2)
    logger.info("Saved state-space evaluation metrics to %s", summary_path)


if __name__ == "__main__":
    generate_state_space_figures_and_metrics()
