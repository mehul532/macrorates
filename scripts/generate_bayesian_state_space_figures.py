"""
Generate publication-quality diagnostic figures for Regime-Switching Kim Filter
and Bayesian MCMC State-Space Nelson-Siegel modeling.
"""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.data.fred_ingest import CMT_SERIES_CONFIG
from src.data.pipeline import load_yield_panel
from src.state_space.bayesian_state_space import (
    RegimeSwitchingNelsonSiegel,
    BayesianNelsonSiegelSampler,
)
from src.state_space.state_space import estimate_and_filter_state_space

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("generate_figures")

DEFAULT_MATURITIES = {k: v["maturity_years"] for k, v in CMT_SERIES_CONFIG.items()}
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("/Users/Patron/.gemini/antigravity/brain/ee02f46c-fd5a-4ed8-84fa-d563ee75ce97")


def generate_all():
    logger.info("Loading yield panel...")
    yield_df, _ = load_yield_panel()
    yield_df["date"] = pd.to_datetime(yield_df["date"])
    yield_df = yield_df.sort_values("date").reset_index(drop=True)

    # 1. Fit Point MLE Kalman
    logger.info("Fitting Point MLE Kalman...")
    mle_res = estimate_and_filter_state_space(
        yield_df=yield_df,
        maturities_dict=DEFAULT_MATURITIES,
        lambda_param=0.7308,
        use_mle_optimization=False,  # OLS-VAR rapid estimation
    )

    # 2. Fit Regime-Switching Kim Filter
    logger.info("Fitting Regime-Switching Kim Filter...")
    rs_model = RegimeSwitchingNelsonSiegel(
        maturities=mle_res.maturities,
        lambda_param=0.7308,
    )
    rs_res = rs_model.fit_and_filter(
        yield_df=yield_df,
        maturities_dict=DEFAULT_MATURITIES,
    )

    # 3. Fit Bayesian MCMC State-Space Sampler
    # Subsample stride 3 for fast yet representative MCMC sampling
    df_mcmc = yield_df.iloc[::3].reset_index(drop=True)
    logger.info("Fitting Bayesian MCMC Sampler (200 draws on %d dates)...", len(df_mcmc))
    bayes_sampler = BayesianNelsonSiegelSampler(
        maturities=mle_res.maturities,
        lambda_param=0.7308,
    )
    bayes_res = bayes_sampler.fit_mcmc(
        yield_df=df_mcmc,
        maturities_dict=DEFAULT_MATURITIES,
        date_col="date",
        n_draws=200,
        burn_in=40,
        thin=2,
        seed=42,
    )

    dates_rs = pd.to_datetime(rs_res.dates)
    dates_bayes = pd.to_datetime(bayes_res.dates)

    # =========================================================================
    # FIGURE 1: REGIME-SWITCHING DYNAMICS (KIM FILTER)
    # =========================================================================
    logger.info("Rendering Figure 1: Regime-Switching Dynamics...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Filtered Regime Probabilities
    p_df = rs_res.filtered_regime_probs
    axes[0].plot(dates_rs, p_df["p_hiking"], color="#d62728", lw=1.5, label="P(Hiking)")
    axes[0].plot(dates_rs, p_df["p_cutting"], color="#1f77b4", lw=1.5, label="P(Cutting/Easing)")
    axes[0].plot(dates_rs, p_df["p_hold"], color="#2ca02c", lw=1.2, alpha=0.8, label="P(Hold/Pause)")
    axes[0].set_title("Kim Filter: Filtered Monetary Policy Regime Probabilities $P(S_t = s \\mid Y_t)$", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Probability", fontsize=10)
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].legend(loc="upper right", frameon=True)

    # Panel 2: Level Factor with Regime Highlights
    y10_idx = rs_res.tenor_names.index("DGS10") if "DGS10" in rs_res.tenor_names else 8
    y10_clean = rs_res.observed_yields[:, y10_idx]
    axes[1].plot(dates_rs, rs_res.filtered_states["level"], color="#2b5c8f", lw=1.5, label="Kim Filter Level Factor ($L_t$)")
    axes[1].plot(dates_rs, y10_clean, color="gray", lw=1.0, ls="--", alpha=0.7, label="Observed 10Y CMT Yield")
    # Highlight Hiking regimes (P(Hiking) > 0.6)
    hiking_periods = p_df["p_hiking"] > 0.6
    axes[1].fill_between(dates_rs, 0, 8, where=hiking_periods, color="#d62728", alpha=0.15, label="Hiking Regime ($P > 0.6$)")
    axes[1].set_title("Level Factor ($L_t$) vs. 10Y Yield with Active Fed Hiking Regimes", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Yield (%)", fontsize=10)
    axes[1].set_ylim(0.0, 7.0)
    axes[1].legend(loc="upper right", frameon=True)

    # Panel 3: Slope Factor (Inversion Dynamics)
    axes[2].plot(dates_rs, rs_res.filtered_states["slope"], color="#ff7f0e", lw=1.5, label="Kim Filter Slope Factor ($S_t$)")
    axes[2].axhline(0.0, color="black", lw=1.0, ls=":")
    axes[2].fill_between(dates_rs, rs_res.filtered_states["slope"], 0, where=(rs_res.filtered_states["slope"] > 0), color="#1f77b4", alpha=0.2, label="Steep / Normal")
    axes[2].fill_between(dates_rs, rs_res.filtered_states["slope"], 0, where=(rs_res.filtered_states["slope"] < 0), color="#d62728", alpha=0.2, label="Inverted")
    axes[2].set_title("Slope Factor ($S_t$): Curve Inversions (Negative Slope)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Slope (pp)", fontsize=10)
    axes[2].legend(loc="upper right", frameon=True)

    # Panel 4: State Covariance Trace (Regime Volatility Expansion)
    trace_rs = np.trace(rs_res.filtered_cov, axis1=1, axis2=2)
    trace_mle = np.trace(mle_res.filtered_cov, axis1=1, axis2=2)
    axes[3].plot(dates_rs, trace_rs, color="#9467bd", lw=1.5, label="Kim Filter State Covariance $\\mathrm{Tr}(P_{t\\mid t})$")
    axes[3].plot(dates_rs, trace_mle, color="black", lw=1.0, ls="--", alpha=0.6, label="Point MLE Constant Steady-State Covariance")
    axes[3].set_title("State Uncertainty Expansion: Kim Filter Adapts Volatility to Policy Regimes", fontsize=11, fontweight="bold")
    axes[3].set_ylabel("Trace($P_{t\\mid t}$)", fontsize=10)
    axes[3].set_xlabel("Date", fontsize=10)
    axes[3].legend(loc="upper right", frameon=True)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path1 = FIGURES_DIR / "regime_switching_dynamics.png"
    fig.savefig(fig_path1, dpi=300)
    plt.close(fig)
    logger.info("Saved Figure 1 to %s", fig_path1)

    # =========================================================================
    # FIGURE 2: BAYESIAN CREDIBLE INTERVALS & STRESS PERIODS
    # =========================================================================
    logger.info("Rendering Figure 2: Bayesian Credible Intervals across Stress Episodes...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), gridspec_kw={"width_ratios": [2, 1]})

    factors = ["level", "slope", "curvature"]
    factor_labels = ["Level ($L_t$)", "Slope ($S_t$)", "Curvature ($C_t$)"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    # Full Sample Credible Intervals (Left Column)
    for i, factor in enumerate(factors):
        ax = axes[i, 0]
        median_s = bayes_res.state_medians[factor]
        low_95 = bayes_res.ci_lower_95[factor]
        upp_95 = bayes_res.ci_upper_95[factor]
        low_90 = bayes_res.ci_lower_90[factor]
        upp_90 = bayes_res.ci_upper_90[factor]

        ax.fill_between(dates_bayes, low_95, upp_95, color=colors[i], alpha=0.18, label="95% Credible Interval")
        ax.fill_between(dates_bayes, low_90, upp_90, color=colors[i], alpha=0.28, label="90% Credible Interval")
        ax.plot(dates_bayes, median_s, color=colors[i], lw=1.4, label=f"Bayesian Posterior Median")
        ax.plot(pd.to_datetime(mle_res.dates), mle_res.filtered_states[factor], color="black", lw=0.9, ls="--", alpha=0.7, label="Point MLE Filter")

        ax.set_title(f"Bayesian Dynamic Nelson-Siegel: {factor_labels[i]} Posterior Distribution", fontsize=11, fontweight="bold")
        ax.set_ylabel("Factor Value", fontsize=10)
        ax.legend(loc="upper right", frameon=True, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Zoomed Inset 1: March 2020 COVID Shock (Right Top)
    ax_covid = axes[0, 1]
    mask_c = ((dates_bayes >= "2020-01-01") & (dates_bayes <= "2020-06-30")).values
    d_c = dates_bayes[mask_c]
    ax_covid.fill_between(d_c, bayes_res.ci_lower_95.loc[mask_c, "level"], bayes_res.ci_upper_95.loc[mask_c, "level"], color="#1f77b4", alpha=0.25, label="95% CI Level")
    ax_covid.plot(d_c, bayes_res.state_medians.loc[mask_c, "level"], color="#1f77b4", lw=1.8, label="Level Median")
    ax_covid.set_title("COVID Shock Zoom: March 2020 (Dash for Cash)", fontsize=10, fontweight="bold")
    ax_covid.set_ylabel("Level (%)", fontsize=9)
    ax_covid.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax_covid.grid(True, alpha=0.3)
    ax_covid.legend(loc="upper right", fontsize=8)

    # Zoomed Inset 2: 2022 Fed Rate Hiking Inversion Shock (Right Middle)
    ax_hike = axes[1, 1]
    mask_h = ((dates_bayes >= "2022-01-01") & (dates_bayes <= "2022-12-31")).values
    d_h = dates_bayes[mask_h]
    ax_hike.fill_between(d_h, bayes_res.ci_lower_95.loc[mask_h, "slope"], bayes_res.ci_upper_95.loc[mask_h, "slope"], color="#d62728", alpha=0.25, label="95% CI Slope")
    ax_hike.plot(d_h, bayes_res.state_medians.loc[mask_h, "slope"], color="#d62728", lw=1.8, label="Slope Median")
    ax_hike.axhline(0.0, color="black", ls=":", lw=1.0)
    ax_hike.set_title("2022 Rate Hiking Shock: Violent Curve Inversion", fontsize=10, fontweight="bold")
    ax_hike.set_ylabel("Slope (pp)", fontsize=9)
    ax_hike.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax_hike.grid(True, alpha=0.3)
    ax_hike.legend(loc="lower left", fontsize=8)

    # Zoomed Inset 3: Credible Interval Width Analysis (Right Bottom)
    ax_w = axes[2, 1]
    w_slope = bayes_res.ci_width_95["slope"] * 100  # in bp
    w_level = bayes_res.ci_width_95["level"] * 100
    ax_w.plot(dates_bayes, w_slope, color="#d62728", lw=1.2, label="Slope 95% CI Width (bp)")
    ax_w.plot(dates_bayes, w_level, color="#1f77b4", lw=1.2, label="Level 95% CI Width (bp)")
    ax_w.axvspan(pd.to_datetime("2020-03-01"), pd.to_datetime("2020-04-30"), color="gray", alpha=0.2, label="March 2020")
    ax_w.axvspan(pd.to_datetime("2022-03-01"), pd.to_datetime("2022-12-31"), color="#d62728", alpha=0.1, label="2022 Hikes")
    ax_w.set_title("Posterior Uncertainty Width ($W_{95\\%}$ in bps)", fontsize=10, fontweight="bold")
    ax_w.set_ylabel("Width (bps)", fontsize=9)
    ax_w.xaxis.set_major_locator(mdates.YearLocator(4))
    ax_w.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig_path2 = FIGURES_DIR / "bayesian_credible_intervals_stress.png"
    fig.savefig(fig_path2, dpi=300)
    plt.close(fig)
    logger.info("Saved Figure 2 to %s", fig_path2)

    # Copy to artifact directory
    if ARTIFACT_DIR.exists():
        import shutil
        shutil.copy(fig_path1, ARTIFACT_DIR / fig_path1.name)
        shutil.copy(fig_path2, ARTIFACT_DIR / fig_path2.name)
        logger.info("Copied figures to artifact directory %s", ARTIFACT_DIR)


if __name__ == "__main__":
    generate_all()
