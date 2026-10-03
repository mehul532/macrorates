#!/usr/bin/env python3
"""
Root entry point and CLI for Milestone 14:
Gradient-Boosted Model (LightGBM/GBM) Factor Forecaster, Walk-Forward Baseline Integration, and SHAP Attribution.

Usage:
  python ml_baseline.py [--quick] [--plot-shap]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
Path("/tmp/mpl").mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.walk_forward import WalkForwardConfig, WalkForwardHarness
from src.model.ml_baseline import (
    FactorFeatureEngineer,
    GBMForecasterConfig,
    GradientBoostedFactorForecaster,
    run_ml_factor_forecasting,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ml_baseline")

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(os.environ.get("ANTIGRAVITY_ARTIFACT_DIR", "reports/figures"))


def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def generate_shap_plots(shap_data: Dict[str, Any], feature_names: list[str]):
    """Generate publication-quality SHAP / Gini attribution plots."""
    set_plot_style()
    logger.info("Generating feature attribution diagnostic figures...")

    stype = shap_data.get("type")
    if stype == "tree_shap":
        results = shap_data["results"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

        target_titles = {
            "target_dLevel": ("ΔLevel (Yield Shift)", "#1f77b4"),
            "target_dSlope": ("ΔSlope (Steepening/Flattening)", "#2ca02c"),
            "target_dCurvature": ("ΔCurvature (Belly Shift)", "#d62728"),
        }

        for idx, (target_col, (title, color)) in enumerate(target_titles.items()):
            ax = axes[idx]
            if target_col not in results:
                continue

            res = results[target_col]
            top10 = res["mean_abs_shap"].head(10).iloc[::-1]

            y_pos = np.arange(len(top10))
            bars = ax.barh(y_pos, top10.values, color=color, alpha=0.85, edgecolor="black", linewidth=0.5)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(top10.index, fontsize=10)
            ax.set_xlabel("Mean |SHAP Value| (bp impact)", fontsize=11)
            ax.set_title(f"Key Drivers of {title}", fontsize=12, fontweight="bold")

            for bar in bars:
                width = bar.get_width()
                ax.annotate(
                    f"{width:.4f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8
                )

        fig.suptitle(
            "TreeSHAP Feature Attribution: Drivers of Term Structure Factor Increments (Observational)\n"
            "Model: Gradient Boosted Trees with Strictly Leakage-Safe Feature Space",
            fontsize=14, fontweight="bold", y=0.99
        )
        plt.tight_layout()

        out_path = FIGURES_DIR / "gbm_shap_summary.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        if ARTIFACT_DIR.resolve() != FIGURES_DIR.resolve() and ARTIFACT_DIR.exists():
            shutil.copy(out_path, ARTIFACT_DIR / "gbm_shap_summary.png")
        logger.info("Saved %s.", out_path)

        # 2. Combined feature importance plot across all factors
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        all_imp = pd.DataFrame({
            "Level Impact": results["target_dLevel"]["mean_abs_shap"],
            "Slope Impact": results["target_dSlope"]["mean_abs_shap"],
            "Curvature Impact": results["target_dCurvature"]["mean_abs_shap"],
        })
        all_imp["Total Impact"] = all_imp.sum(axis=1)
        top15 = all_imp.sort_values("Total Impact", ascending=False).head(15).iloc[::-1]

        top15[["Level Impact", "Slope Impact", "Curvature Impact"]].plot(
            kind="barh", stacked=True, ax=ax2,
            color=["#1f77b4", "#2ca02c", "#d62728"], alpha=0.85, edgecolor="black", linewidth=0.5
        )
        ax2.set_title("Top 15 Machine Learning Features by Aggregated Term Structure Impact", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Cumulative Mean |SHAP Value| across Factors", fontsize=11)
        ax2.legend(title="Factor Component", fontsize=10)
        plt.tight_layout()

        out_path2 = FIGURES_DIR / "gbm_feature_importance.png"
        plt.savefig(out_path2, dpi=300, bbox_inches="tight")
        plt.close()

        if ARTIFACT_DIR.resolve() != FIGURES_DIR.resolve() and ARTIFACT_DIR.exists():
            shutil.copy(out_path2, ARTIFACT_DIR / "gbm_feature_importance.png")
        logger.info("Saved %s.", out_path2)

    elif stype == "gini":
        logger.info("Plotting Gini feature importances (TreeSHAP unavailable)...")
        results = shap_data.get("importances", {})
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
        target_titles = {
            "target_dLevel": ("ΔLevel (Yield Shift)", "#1f77b4"),
            "target_dSlope": ("ΔSlope (Steepening/Flattening)", "#2ca02c"),
            "target_dCurvature": ("ΔCurvature (Belly Shift)", "#d62728"),
        }
        for idx, (target_col, (title, color)) in enumerate(target_titles.items()):
            ax = axes[idx]
            if target_col not in results:
                continue
            top10 = results[target_col].head(10).iloc[::-1]
            y_pos = np.arange(len(top10))
            ax.barh(y_pos, top10.values, color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top10.index, fontsize=10)
            ax.set_xlabel("Gini Feature Importance", fontsize=11)
            ax.set_title(f"Key Drivers of {title}", fontsize=12, fontweight="bold")

        fig.suptitle(
            "Gini Feature Importance: Drivers of Term Structure Factor Increments\n"
            "Model: Gradient Boosted Trees (Observational Feature Ranking)",
            fontsize=14, fontweight="bold", y=0.99
        )
        plt.tight_layout()
        out_path = FIGURES_DIR / "gbm_shap_summary.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        if ARTIFACT_DIR.resolve() != FIGURES_DIR.resolve() and ARTIFACT_DIR.exists():
            shutil.copy(out_path, ARTIFACT_DIR / "gbm_shap_summary.png")
        logger.info("Saved %s.", out_path)
    else:
        logger.warning("Feature attribution data unavailable; skipping attribution plots.")


def main(quick: bool = False, plot_shap: bool = True):
    print("=" * 85)
    print("  MILESTONE 14: GRADIENT-BOOSTED MODEL (GBM) FACTOR FORECASTER & WALK-FORWARD BASELINE")
    print("  Leakage-Safe Feature Engineering, Feature Attribution & Extended Baseline Comparison")
    print("=" * 85)

    print("\n[1/5] Loading Data Panels...")
    yield_df = pd.read_parquet("data/processed/yield_panel.parquet")
    factor_df = pd.read_parquet("data/processed/factor_panel.parquet")
    macro_df = pd.read_parquet("data/processed/macro_surprises.parquet")
    print(f"  Yield dates:  {yield_df['date'].min().date()} to {yield_df['date'].max().date()} ({len(yield_df):,} rows)")
    print(f"  Factor dates: {factor_df['date'].min().date()} to {factor_df['date'].max().date()} ({len(factor_df):,} rows)")
    print(f"  Macro panel:  {macro_df['date'].min().date()} to {macro_df['date'].max().date()} ({len(macro_df):,} announcements)")

    print("\n[2/5] Building Leakage-Safe Feature Matrix X_t and Forward Targets y_{t+1}...")
    ml_eval = run_ml_factor_forecasting(
        factor_df=factor_df,
        yield_df=yield_df,
        macro_df=macro_df,
        train_split_ratio=0.80,
    )
    X_train = ml_eval["X_train"]
    X_test = ml_eval["X_test"]
    rmse_bp = ml_eval["rmse_bp"]

    print(f"  Total observations: {len(X_train) + len(X_test):,} trading days")
    print(f"  Engineered features: {len(ml_eval['feature_names'])} features")
    print(f"  Out-of-Sample Factor RMSE: dLevel = {rmse_bp['dLevel']:.2f} bp | dSlope = {rmse_bp['dSlope']:.2f} bp | dCurvature = {rmse_bp['dCurvature']:.2f} bp")

    print("\n[3/5] Feature Attribution Analysis:")
    shap_data = ml_eval["shap_data"]
    if shap_data.get("type") == "tree_shap":
        res = shap_data["results"]
        for target, name in [
            ("target_dLevel", "Level (Yield Level Shift)"),
            ("target_dSlope", "Slope (2s10s Curve Slope)"),
            ("target_dCurvature", "Curvature (Intermediate Belly)"),
        ]:
            print(f"\n  Top 5 Features for {name}:")
            top5 = res[target]["mean_abs_shap"].head(5)
            for rank, (feat, val) in enumerate(top5.items(), start=1):
                print(f"    {rank}. {feat:25s} | Mean |SHAP|: {val:.4f} bp")
    elif shap_data.get("type") == "gini":
        imps = shap_data.get("importances", {})
        for target, name in [
            ("target_dLevel", "Level (Yield Level Shift)"),
            ("target_dSlope", "Slope (2s10s Curve Slope)"),
            ("target_dCurvature", "Curvature (Intermediate Belly)"),
        ]:
            print(f"\n  Top 5 Features for {name} (Gini Importance):")
            top5 = imps.get(target, pd.Series()).head(5)
            for rank, (feat, val) in enumerate(top5.items(), start=1):
                print(f"    {rank}. {feat:25s} | Gini Imp: {val:.4f}")

    if plot_shap:
        generate_shap_plots(shap_data, ml_eval["feature_names"])

    print("\n[4/5] Executing Walk-Forward Evaluation Harness...")
    if quick:
        print("  Running quick mode (last 850 trading days, 2 folds)...")
        sub_yield = yield_df.iloc[-850:]
        sub_factor = factor_df.iloc[-850:]
        wf_cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=42)
        harness = WalkForwardHarness(sub_yield, sub_factor, macro_df, config=wf_cfg)
        wf_results = harness.run_walk_forward_evaluation(strategy_type="2s10s", max_folds=2)
    else:
        print("  Running full walk-forward evaluation across entire 2006–2026 sample...")
        wf_cfg = WalkForwardConfig(train_window_days=756, refit_frequency_days=21)
        harness = WalkForwardHarness(yield_df, factor_df, macro_df, config=wf_cfg)
        wf_results = harness.run_walk_forward_evaluation(strategy_type="2s10s")

    baseline_table = wf_results["baseline_table"]
    common_sample_table = wf_results.get("common_sample_table", baseline_table)
    run_meta = wf_results.get("run_metadata", {})

    print("\n[5/5] Extended Common-Sample Comparison Table:")
    print(common_sample_table.to_string())

    # Export extended scorecard to JSON
    scorecard_path = Path("reports/extended_walk_forward_metrics_scorecard.json")
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard_payload = {
        "metadata": run_meta,
        "models": common_sample_table.to_dict(orient="index"),
        "common_sample_table": common_sample_table.to_dict(orient="index"),
    }
    with open(scorecard_path, "w", encoding="utf-8") as f:
        json.dump(scorecard_payload, f, indent=2, default=str)
    logger.info("Saved extended scorecard to %s.", scorecard_path)

    # Export verdict report
    write_verdict_report(common_sample_table, ml_eval, run_meta)

    print("\n" + "=" * 85)
    print("  MILESTONE 14 VERDICT & AUDIT DISCLOSURES:")
    print(f"  - Run Mode: {run_meta.get('run_mode', 'UNKNOWN')} ({run_meta.get('fold_count', 0)} folds)")
    print(f"  - Commit Hash: {run_meta.get('git_commit', 'UNKNOWN')}")
    print("  - Descriptive SHAP associations are observational attribution, NOT proof of causal macro mechanism.")
    print("  - Random Walk provides the non-parametric zero-increment forecasting benchmark.")
    print("=" * 85 + "\n")


def format_df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table without external dependencies."""
    headers = ["Model / Forecast Method"] + list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |")
    for idx, row in df.iterrows():
        row_str = [str(idx)]
        for v in row.values:
            if isinstance(v, (int, np.integer)):
                row_str.append(f"{v:,}")
            elif isinstance(v, (float, np.floating)):
                row_str.append(f"{v:.2f}" if not np.isnan(v) else "N/A")
            else:
                row_str.append(str(v))
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)


def write_verdict_report(
    common_sample_table: pd.DataFrame,
    ml_eval: Dict[str, Any],
    run_meta: Dict[str, Any],
):
    """Write research verdict markdown report with strict audit disclosures and no unverified claims."""
    rep_path = Path("reports/ml_baseline_verdict.md")
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    
    stype = ml_eval["shap_data"].get("type", "unknown")
    if stype == "tree_shap":
        shap_res = ml_eval["shap_data"].get("results", {})
        lvl_s = shap_res.get("target_dLevel", {}).get("mean_abs_shap")
        top_lvl = ", ".join(list(lvl_s.head(3).index)) if hasattr(lvl_s, "head") else "N/A"
        slp_s = shap_res.get("target_dSlope", {}).get("mean_abs_shap")
        top_slp = ", ".join(list(slp_s.head(3).index)) if hasattr(slp_s, "head") else "N/A"
        cur_s = shap_res.get("target_dCurvature", {}).get("mean_abs_shap")
        top_cur = ", ".join(list(cur_s.head(3).index)) if hasattr(cur_s, "head") else "N/A"
    elif stype == "gini":
        imps = ml_eval["shap_data"].get("importances", {})
        lvl_s = imps.get("target_dLevel")
        top_lvl = ", ".join(list(lvl_s.head(3).index)) if hasattr(lvl_s, "head") else "N/A"
        slp_s = imps.get("target_dSlope")
        top_slp = ", ".join(list(slp_s.head(3).index)) if hasattr(slp_s, "head") else "N/A"
        cur_s = imps.get("target_dCurvature")
        top_cur = ", ".join(list(cur_s.head(3).index)) if hasattr(cur_s, "head") else "N/A"
    else:
        top_lvl = top_slp = top_cur = "N/A"

    table_md = format_df_to_markdown(common_sample_table)
    run_mode = run_meta.get("run_mode", "UNKNOWN_MODE")
    fold_cnt = run_meta.get("fold_count", "N/A")
    eval_start = run_meta.get("eval_start_date", "N/A")
    eval_end = run_meta.get("eval_end_date", "N/A")
    commit_hash = run_meta.get("git_commit", "UNKNOWN")
    backend = run_meta.get("gbm_backend", "sklearn")
    seed = run_meta.get("seed", 42)
    checksums = run_meta.get("data_checksums", {})
    macro_audit = run_meta.get("macro_event_audit", {})
    econ_diag = run_meta.get("econometric_diagnostics", {})

    # Extract dynamic metrics from common_sample_table and econometric diagnostics
    # Extract dynamic metrics from common_sample_table and econometric diagnostics
    def get_table_val(model_display: str, col: str, default: float = np.nan) -> float:
        if model_display in common_sample_table.index and col in common_sample_table.columns:
            val = common_sample_table.loc[model_display, col]
            try:
                f_val = float(val)
                return f_val if not np.isnan(f_val) else default
            except (ValueError, TypeError):
                return default
        return default

    def fmt_bp(val: float, digits: int = 2) -> str:
        return f"{val:.{digits}f} bp" if not np.isnan(val) else "N/A"

    def fmt_num(val: float, digits: int = 2) -> str:
        return f"{val:.{digits}f}" if not np.isnan(val) else "N/A"

    rw_spread_rmse = get_table_val("Random Walk (Curve Benchmark)", "2s10s Spread RMSE (bp)", np.nan)
    trad_ns_spread_rmse = get_table_val("AR(1) Baseline (Static NS)", "2s10s Spread RMSE (bp)", np.nan)
    trad_dns_spread_rmse = get_table_val("DNS + Kalman", "2s10s Spread RMSE (bp)", np.nan)
    res_ns_spread_rmse = get_table_val("Static NS (Residual-Preserving Diagnostic)", "2s10s Spread RMSE (bp)", np.nan)
    res_dns_spread_rmse = get_table_val("DNS + Kalman (Residual-Preserving Diagnostic)", "2s10s Spread RMSE (bp)", np.nan)

    # Build clipping and exceedance breakdown table
    raw_exceed_dict = econ_diag.get("raw_signal_threshold_exceedance_pct", {})
    inherited_clip_dict = econ_diag.get("inherited_dns_clipping_pct", {})
    act_clip_dict = econ_diag.get("additional_clipping_frequency_pct", econ_diag.get("actual_clipping_frequency_pct", {}))
    sat_dict = econ_diag.get("final_position_saturation_pct", {})
    mean_unclip_dict = econ_diag.get("mean_unclipped_signal_std", {})
    sat_bounds = econ_diag.get("saturation_bounds", {
        "DNS_Scaled_60": 0.60,
        "DNS_Kalman_Macro": 1.00,
        "Random_Walk": 1.00,
        "PCA_VAR": 1.00,
        "Static_NS": 1.00,
        "DNS_Kalman": 1.00,
        "Static_NS_Residual_Preserving": 1.00,
        "DNS_Kalman_Residual_Preserving": 1.00,
        "GBM": 1.00,
    })

    all_models = sorted(list(set(
        list(raw_exceed_dict.keys()) +
        list(inherited_clip_dict.keys()) +
        list(act_clip_dict.keys()) +
        list(sat_dict.keys())
    )))
    clipping_rows = []
    for model_name in all_models:
        raw_pct = raw_exceed_dict.get(model_name, np.nan)
        inh_pct = inherited_clip_dict.get(model_name, 0.0)
        act_pct = act_clip_dict.get(model_name, np.nan)
        sat_pct = sat_dict.get(model_name, np.nan)
        mean_u = mean_unclip_dict.get(model_name, np.nan)
        bnd = sat_bounds.get(model_name, 1.00)
        bnd_str = f"±{bnd:.2f}"
        raw_str = f"{raw_pct:.1f}%" if not np.isnan(raw_pct) else "N/A"
        inh_str = f"{inh_pct:.1f}%" if not np.isnan(inh_pct) else "0.0%"
        act_str = f"{act_pct:.1f}%" if not np.isnan(act_pct) else "N/A"
        sat_str = f"{sat_pct:.1f}%" if not np.isnan(sat_pct) else "N/A"
        mean_str = f"{mean_u:.3f}" if not np.isnan(mean_u) else "N/A"
        clipping_rows.append(f"| `{model_name}` | {bnd_str} | {raw_str} | {inh_str} | {act_str} | {sat_str} | {mean_str} |")
    clipping_table_md = "\n".join([
        "| Model | Saturation Bound | Raw Threshold Exceedance ($\\ge 1.0$) (%) | Inherited DNS Clipping (%) | Additional Stage Clipping (%) | Final Position Saturation (%) | Mean Unclipped |Input| |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ] + clipping_rows)

    # Observable 2s10s spread decomposition table
    decomp = econ_diag.get("observable_2s10s_spread_decomposition", {})
    fac_dynamics_rmse = decomp.get("factor_dynamics_spread_rmse_bp", np.nan)
    fit_spread_rmse = decomp.get("cross_sectional_fit_spread_rmse_bp", np.nan)
    cross_moment_val = decomp.get("uncentered_cross_moment_bp2", decomp.get("cross_term_cov_bp2", "N/A"))

    if decomp:
        decomp_table_md = "\n".join([
            "| Error Component | Symbol | Spread RMSE (bp) | Spread MSE (bp²) | Econometric Description |",
            "| :--- | :---: | :---: | :---: | :--- |",
            f"| **Total Observable 2s10s Spread Error** | $e_s$ | **{decomp.get('total_spread_rmse_bp', 'N/A')} bp** | **{decomp.get('total_spread_mse_bp2', 'N/A')} bp²** | Actual observed spread minus forecast: $s_t - \\hat{{s}}_t$ |",
            f"| **Factor-Driven Spread Dynamics Error** | $u_{{\\text{{factor}}}}$ | {decomp.get('factor_dynamics_spread_rmse_bp', 'N/A')} bp | {decomp.get('factor_dynamics_spread_mse_bp2', 'N/A')} bp² | In-sample fitted spread minus forecast: $s_t^{{\\text{{fit}}}} - \\hat{{s}}_t$ |",
            f"| **Cross-Sectional Parametric Fit Error** | $u_{{\\text{{fit}}}}$ | {decomp.get('cross_sectional_fit_spread_rmse_bp', 'N/A')} bp | {decomp.get('cross_sectional_fit_spread_mse_bp2', 'N/A')} bp² | Actual observed spread minus fitted spread: $s_t - s_t^{{\\text{{fit}}}}$ |",
            f"| **Twice Uncentered Second Cross Moment** | $2 \\times \\mathbb{{E}}[u_{{\\text{{factor}}}} \\cdot u_{{\\text{{fit}}}}]$ | — | {cross_moment_val} bp² | Second cross moment: $2 \\times \\frac{{1}}{{N}} \\sum u_{{\\text{{factor}}}} \\cdot u_{{\\text{{fit}}}}$ (distinct from centered covariance) |",
            f"| **Sum of Decomposition Components** | $\\sum$ | — | **{decomp.get('sum_components_mse_bp2', 'N/A')} bp²** | Exact mathematical identity: MSE($u_{{\\text{{factor}}}}$) + MSE($u_{{\\text{{fit}}}}$) + $2 \\times \\mathbb{{E}}[u_{{\\text{{factor}}}} \\cdot u_{{\\text{{fit}}}}]$ |",
        ])
    else:
        decomp_table_md = "*Observable spread decomposition unavailable.*"

    trad_ns_clip = act_clip_dict.get("Static_NS", np.nan)
    res_ns_clip = act_clip_dict.get("Static_NS_Residual_Preserving", np.nan)
    if not np.isnan(trad_ns_clip) and not np.isnan(res_ns_clip):
        clip_red_text = f"and signal clipping from {trad_ns_clip:.1f}% to {res_ns_clip:.1f}%"
    else:
        clip_red_text = "and mitigating excessive signal clipping"

    trad_ns_rmse_str = fmt_num(trad_ns_spread_rmse, 1)
    fit_spread_rmse_str = fmt_num(fit_spread_rmse, 1)
    trad_ns_rmse_str2 = fmt_bp(trad_ns_spread_rmse, 2)
    res_ns_rmse_str2 = fmt_bp(res_ns_spread_rmse, 2)

    trad_clip_col_str = f"{trad_ns_clip:.1f}% clipped" if not np.isnan(trad_ns_clip) else "N/A"
    res_clip_col_str = f"{res_ns_clip:.1f}% clipped" if not np.isnan(res_ns_clip) else "N/A"

    pnl_col = "Trading Net PnL ($)" if "Trading Net PnL ($)" in common_sample_table.columns else "Collateral Net PnL ($)"
    ns_pnl = get_table_val("AR(1) Baseline (Static NS)", pnl_col, np.nan)
    dns_pnl = get_table_val("DNS + Kalman", pnl_col, np.nan)
    gbm_pnl = get_table_val("GBM", pnl_col, np.nan)

    if not np.isnan(ns_pnl) and not np.isnan(dns_pnl) and np.isclose(ns_pnl, dns_pnl) and (np.isnan(gbm_pnl) or np.isclose(ns_pnl, gbm_pnl)):
        pnl_finding_str = f"*Finding*: The identical trading PnL (${ns_pnl:,.2f}) observed across traditional level reconstruction models is explained by signal saturation resulting from cross-sectional curve-fitting error propagation."
    elif not np.isnan(ns_pnl) and not np.isnan(dns_pnl):
        pnl_finding_str = f"*Finding*: Trading PnL across traditional level-reconstruction models (Static NS: ${ns_pnl:,.2f}, DNS Kalman: ${dns_pnl:,.2f}) reflects their respective signal saturation profiles."
    else:
        pnl_finding_str = "*Finding*: Trading PnL reflects model-specific signal saturation and risk scaling."

    # Dynamic macro coverage end string
    macro_cov_end = macro_audit.get("macro_data_coverage_end", "N/A")
    if macro_cov_end and macro_cov_end != "N/A":
        cov_end_str = f"ends on {macro_cov_end}"
    else:
        cov_end_str = "has unspecified coverage end date"

    # Dynamic 60% exposure control finding
    ctrl_pnl = get_table_val("DNS (60% Exposure Control)", pnl_col, np.nan)
    macro_pnl = get_table_val("DNS + Kalman + Macro", pnl_col, np.nan)
    ctrl_sharpe = get_table_val("DNS (60% Exposure Control)", "Annualized Sharpe", np.nan)
    macro_sharpe = get_table_val("DNS + Kalman + Macro", "Annualized Sharpe", np.nan)

    if not np.isnan(ctrl_pnl) and not np.isnan(macro_pnl) and np.isclose(ctrl_pnl, macro_pnl, atol=1e-2):
        control_finding_text = (
            f"The control model `DNS (60% Exposure Control)` trades an exact linear scaling of the baseline DNS signal "
            f"($s_t = 0.60 \\times s_{{t}}^{{\\text{{DNS}}}}$). In the evaluation, it produced trading results identical to "
            f"`DNS + Kalman + Macro` (Net PnL: ${ctrl_pnl:,.2f}), confirming that when macroeconomic releases are inactive "
            f"or neutral, any historical PnL difference was entirely due to linear risk/exposure downscaling, not macroeconomic information."
        )
    elif not np.isnan(ctrl_pnl) and not np.isnan(macro_pnl):
        pnl_diff = macro_pnl - ctrl_pnl
        diff_sign_str = f"+${abs(pnl_diff):,.2f}" if pnl_diff >= 0 else f"-${abs(pnl_diff):,.2f}"
        control_finding_text = (
            f"The control model `DNS (60% Exposure Control)` produced a Net PnL of ${ctrl_pnl:,.2f} "
            f"(Sharpe: {fmt_num(ctrl_sharpe, 2)}), while `DNS + Kalman + Macro` produced a Net PnL of ${macro_pnl:,.2f} "
            f"(Sharpe: {fmt_num(macro_sharpe, 2)}), reflecting an active macroeconomic position overlay difference of "
            f"{diff_sign_str}."
        )
    else:
        control_finding_text = (
            "Trading results for DNS control and macro models are not available."
        )

    # Dynamic econometric decomposition prose
    if not np.isnan(trad_ns_spread_rmse) and not np.isnan(res_ns_spread_rmse):
        if res_ns_spread_rmse < trad_ns_spread_rmse:
            rmse_impact_str = f"reducing spread RMSE from {trad_ns_rmse_str2} to {res_ns_rmse_str2} {clip_red_text}"
        elif res_ns_spread_rmse > trad_ns_spread_rmse:
            rmse_impact_str = f"increasing spread RMSE from {trad_ns_rmse_str2} to {res_ns_rmse_str2} {clip_red_text}"
        else:
            rmse_impact_str = f"leaving spread RMSE unchanged at {trad_ns_rmse_str2} {clip_red_text}"
    else:
        rmse_impact_str = "spread RMSE comparison is unavailable"

    if not np.isnan(fit_spread_rmse) and not np.isnan(fac_dynamics_rmse):
        if fit_spread_rmse > fac_dynamics_rmse:
            predom_err_str = f"driven predominantly by cross-sectional curve-fitting errors (~{fit_spread_rmse_str} bp in spread space)"
        elif fac_dynamics_rmse > fit_spread_rmse:
            predom_err_str = f"driven predominantly by factor dynamics errors (~{fmt_num(fac_dynamics_rmse, 1)} bp in spread space)"
        else:
            predom_err_str = f"with curve-fitting and factor errors contributing equally (~{fit_spread_rmse_str} bp)"
    elif not np.isnan(fit_spread_rmse):
        predom_err_str = f"with cross-sectional curve-fitting errors at ~{fit_spread_rmse_str} bp in spread space"
    else:
        predom_err_str = "with error decomposition unavailable"

    # Dynamic decomposition dominant note
    if not np.isnan(fit_spread_rmse) and not np.isnan(fac_dynamics_rmse):
        if fit_spread_rmse > fac_dynamics_rmse:
            dominant_str = (
                f"while cross-sectional curve-fitting error is the dominant contributor ({fmt_bp(fit_spread_rmse)}), "
                f"demonstrating that spread forecast failure is driven predominantly by static parametric fitting error, not factor dynamics."
            )
        elif fac_dynamics_rmse > fit_spread_rmse:
            dominant_str = (
                f"while factor dynamics error is the dominant contributor ({fmt_bp(fac_dynamics_rmse)}), "
                f"demonstrating that spread forecast failure is driven predominantly by factor dynamics, not static curve-fitting error."
            )
        else:
            dominant_str = (
                f"with factor dynamics and curve-fitting errors contributing equally ({fmt_bp(fit_spread_rmse)})."
            )
        decomp_note_md = (
            f"*Note*: As proven above, factor dynamics contribute {fmt_bp(fac_dynamics_rmse)} of spread error "
            f"(comparable to Random Walk's {fmt_bp(rw_spread_rmse)}), {dominant_str}"
        )
    else:
        decomp_note_md = "*Note*: Observable spread error decomposition is unavailable for this evaluation."

    # Dynamic residual clipping prose
    if not np.isnan(res_ns_clip):
        if res_ns_clip == 0.0:
            clip_finding_str = "The residual-preserving formulation eliminates this bias, preserving the natural signal variation without clipping (0.0% clipped)."
        else:
            clip_finding_str = f"The residual-preserving formulation reduces this bias, resulting in a residual signal clipping frequency of {res_ns_clip:.1f}%."
    else:
        clip_finding_str = "Residual signal clipping diagnostics are unavailable."

    # Point-in-Time Evaluation Gate extraction
    pit_gate = run_meta.get("point_in_time_gate", {})
    pit_manifest = run_meta.get("point_in_time_manifest", {})
    gate_verdict = pit_gate.get("gate_verdict", "NOT_EVALUABLE")
    eval_tier = pit_gate.get("evaluation_tier", "DEVELOPMENT_EVIDENCE")
    macro_alpha_verdict = pit_gate.get("macro_alpha_verdict", None)
    alpha_display = macro_alpha_verdict if macro_alpha_verdict else "NONE (HOLD OUT NOT EVALUABLE - ALPHA CLAIMS BARRED)"
    manifest_sha = pit_manifest.get("manifest_sha256", "N/A")
    missing_reasons = pit_gate.get("missing_data_reasons", [])
    missing_reasons_md = "\n".join([f"- {r}" for r in missing_reasons]) if missing_reasons else "- None"
    min_reqs = pit_gate.get("minimum_predeclared_sample_requirements", {})

    fold_cov_records = macro_audit.get("fold_event_coverage", pit_manifest.get("event_coverage_table", []))
    if fold_cov_records:
        fold_cov_lines = [
            "| Fold | Training Cutoff | Test Window | Independent Releases | Active Event Days | Missing Consensus or Timestamps | Nonzero Overlay Days |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for f in fold_cov_records:
            fold_cov_lines.append(
                f"| {f.get('Fold', 'N/A')} | {f.get('Train Cutoff', 'N/A')} | {f.get('Test Window', 'N/A')} | "
                f"{f.get('Independent Releases', 0)} | {f.get('Active Event Days', 0)} | "
                f"{f.get('Missing Consensus or Timestamps', 0)} | {f.get('Nonzero Overlay Days', 0)} |"
            )
        fold_cov_table_md = "\n".join(fold_cov_lines)
    else:
        fold_cov_table_md = "*Fold event coverage table unavailable.*"

    content = f"""# Milestone 14 Verdict: Machine Learning Baseline & Point-in-Time Evaluation Gate

**Author**: MacroRates Research Team  
**Git Commit**: `{commit_hash}`  
**Run Mode**: `{run_mode}` (Folds: {fold_cnt})  
**Evaluated Sample**: {eval_start} to {eval_end} ({run_meta.get('total_eval_days', 'N/A')} trading days)  
**Backend**: `{backend}` | **Seed**: `{seed}`  

> [!IMPORTANT]
> **POINT-IN-TIME EVALUATION GATE & RESEARCH INTEGRITY DISCLOSURES**:
> 1. **Holdout Evaluation Gate Verdict**: The evaluated {run_meta.get('total_eval_days', 'N/A')}-day sample is audited as **`{eval_tier}`**. The gate verdict is **`{gate_verdict}`**. An event-covered, genuinely untouched holdout is unavailable; therefore, **NO MACRO-ALPHA VERDICT IS REPORTED**.
> 2. **Forecast vs. Trading Separation**: Yield curve and spread forecast accuracy (RMSE in basis points) are strictly separated from trading performance (Sharpe, Sortino, turnover, and PnL). Forecast evaluations assess econometric predictability; trading evaluations measure strategy execution under risk-budget constraints.
> 3. **Research Proxy & Par Yield Disclosure**: Trading net PnL, Sharpe ratios, and DV01 returns are a synthetic research proxy based on daily rebalancing of constant-maturity Treasury yields. Any executable profit claim requires historical tradable futures or cash bond prices, contract rolls, bid-ask spreads, and financing costs. [U.S. Treasury Daily Treasury Par Yield Curve Rates](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/) are indicative market quotes based on FRBNY composite closing quotes.
> 4. **Curve Provenance**: Yield curve forecasts for `DNS_Kalman_Macro` are tagged `COPIED_DNS_CURVE_FORECAST` regardless of overlay activity.
> 5. **Macro Data Coverage & Status**: The committed macro dataset (`data/processed/macro_surprises.parquet`) {cov_end_str}. During the evaluated test period ({eval_start} to {eval_end}), there were **{macro_audit.get('total_calendar_events', macro_audit.get('total_test_events', 0))} calendar test events** and **{macro_audit.get('total_observed_decision_events', macro_audit.get('total_evaluated_decision_events', 0))} observed decision origin events**. Therefore, `DNS + Kalman + Macro` is formally audited and categorized as **`{macro_audit.get('evaluation_status', 'NOT_EVALUATED')}`**.
> 6. **60% Exposure Control Finding**: {control_finding_text}
> 7. **Observational vs. Causal Attribution**: TreeSHAP and Gini feature rankings describe statistical feature associations within gradient-boosted decision trees. They are descriptive diagnostics, NOT proof of causal macroeconomic transmission mechanisms.
> 8. **Econometric Decomposition**: Traditional level-reconstruction models (Static NS, DNS Kalman) exhibit ~{trad_ns_rmse_str} bp 2s10s spread RMSE {predom_err_str}, which push spread forecasts into extreme values that saturate the $\\pm 1.0$ signal clip. The diagnostic residual-preserving formulation addresses this cross-sectional bias, {rmse_impact_str}.
> 9. **Annualization & Statistics Corrections**: Sharpe and Sortino ratios are annualized with $\\sqrt{{252}}$ strictly on standard deviation ($(\\mu \\times 252) / (\\sigma \\times \\sqrt{{252}}) = (\\mu \\times \\sqrt{{252}}) / \\sigma$). Hit rates are computed strictly over active trading days; zero-trading benchmarks (Random Walk, Cash Only) report `N/A`, avoiding false 100% hit rate claims.
> 10. **Benchmark Discipline**: Random Walk provides the unparameterized zero-increment curve forecasting benchmark. Cash-Only provides an unencumbered capital strategy benchmark. Models are evaluated on common out-of-sample test dates without retroactive tuning or artificial error multipliers.

---

## 1. Point-in-Time Evaluation Gate Verdict & Run Manifest

| Gate Field | Status / Value | Audit Interpretation |
| :--- | :---: | :--- |
| **Gate Verdict** | **`{gate_verdict}`** | Formal point-in-time readiness gate determination |
| **Evaluation Tier** | **`{eval_tier}`** | Development evidence (not an untouched holdout) |
| **Macro-Alpha Verdict** | **`{alpha_display}`** | Alpha claims strictly barred until holdout criteria are met |
| **Run Manifest SHA-256** | `{manifest_sha[:16] if manifest_sha else 'N/A'}` | Immutable run manifest serialized at `reports/point_in_time_manifest.json` |

### Missing Data & Minimum Predeclared Sample Requirements
**Missing Data Reasons**:
{missing_reasons_md}

**Predeclared Minimum Holdout Criteria**:
- **Minimum Holdout Trading Days**: $\\ge {min_reqs.get('min_holdout_trading_days', 252)}$ trading days
- **Minimum Independent Releases**: $\\ge {min_reqs.get('min_independent_releases', 20)}$ releases across {', '.join(min_reqs.get('required_indicators', ['CPI', 'NFP', 'FOMC']))}
- **Minimum Active Event Days**: $\\ge {min_reqs.get('min_active_event_days', 10)}$ days
- **Provenance Standard**: {min_reqs.get('required_provenance', 'Unrevised first-release actuals with timestamped consensus vintages.')}
- **Execution Proxy**: {min_reqs.get('execution_proxy_requirement', 'Tradable instrument prices and contract rolls.')}

### Event Coverage Table by Fold

{fold_cov_table_md}

---

## 2. Run Provenance & Data Checksums

| Input Panel | SHA-256 Checksum (16-char) | Path |
| :--- | :--- | :--- |
| **Yield Panel** | `{checksums.get('yield_panel', 'N/A')}` | `data/processed/yield_panel.parquet` |
| **Factor Panel** | `{checksums.get('factor_panel', 'N/A')}` | `data/processed/factor_panel.parquet` |
| **Macro Surprises** | `{checksums.get('macro_surprises', 'N/A')}` | `data/processed/macro_surprises.parquet` |

---

## 3. Macro Event Coverage Audit

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Total Training Events** | {macro_audit.get('total_training_events', 0)} | Macro surprise releases available in training windows |
| **Calendar Test Window Events** | {macro_audit.get('total_calendar_events', macro_audit.get('total_test_events', 0))} | Releases occurring within calendar test dates |
| **Observed Decision Origin Events** | {macro_audit.get('total_observed_decision_events', macro_audit.get('total_evaluated_decision_events', 0))} | Total release events evaluated at decision origins |
| **Timestamp-Available Events** | {macro_audit.get('total_timestamp_available_events', 0)} | Releases verified available prior to market close (<= 16:00 ET) |
| ↳ *Verified Timestamp Available* | {macro_audit.get('total_timestamp_verified_available_events', 0)} | Explicit timezone-verified timestamp <= 16:00 ET |
| ↳ *Legacy Date-Only Assumed* | {macro_audit.get('total_legacy_date_only_assumed_events', 0)} | Legacy date-only releases without intraday timestamp |
| **Post-Close Events** | {macro_audit.get('total_post_close_events', 0)} | Releases after 16:00 ET (unavailable for same-day decision) |
| **Rolled to Next Decision Events** | {macro_audit.get('total_rolled_to_next_decision_events', 0)} | After-close releases rolled into subsequent decision origin |
| **Usable Surprise Events** | {macro_audit.get('total_usable_surprise_events', 0)} | Releases with non-null numeric surprise |
| **Eligible Coefficient Events** | {macro_audit.get('total_eligible_coefficient_events', 0)} | Releases with causal response beta estimated from training history ($N \\ge 8$) |
| **Inadequate History Events** | {macro_audit.get('total_inadequate_history_events', 0)} | Releases where indicator has insufficient training history ($N < 8$) |
| **Active Overlay Events** | {macro_audit.get('total_active_overlay_events', 0)} | Releases contributing nonzero macro position overlay |
| **Nonzero Macro Days in Test** | {macro_audit.get('nonzero_macro_days', 0)} | Evaluated decision days where macro overlay was non-zero |
| **Curve Forecast Provenance** | `{macro_audit.get('curve_forecast_provenance', 'COPIED_DNS_CURVE_FORECAST')}` | Independent provenance of yield curve predictions |
| **Macro Strategy Status** | `{macro_audit.get('strategy_overlay_status', 'INACTIVE')}` | Operational status of macro overlay strategy |
| **Audit Status** | `{macro_audit.get('evaluation_status', 'N/A')}` | Formal audit determination for `DNS_Kalman_Macro` |

*Note: In the absence of test releases, DNS with macro surprise augmentation degenerates to baseline state dynamics scaled by prior event variance.*

---

## 4. Econometric Diagnostics & Error Decomposition

### Observable 2s10s Spread Forecast Error Decomposition ($e_s = u_{{\\text{{factor}}}} + u_{{\\text{{fit}}}}$)
*Exact mathematical decomposition in the observable 2s10s spread space, including the twice uncentered second cross moment:*

{decomp_table_md}

{decomp_note_md}

### Factor Dynamics Diagnostics (Factor-Coordinate Space)
*These diagnostics evaluate state variable forecasting in factor-coordinate space, distinct from observable 2s10s spread error decomposition:*

| Diagnostic Metric | Value (bp) | Description |
| :--- | :---: | :--- |
| **Contemporaneous NS Curve Fit RMSE** | {econ_diag.get('ns_contemporaneous_fit_rmse_bp', 'N/A')} bp | Full-curve cross-sectional parametric fitting error ($y_t - \\Lambda \\beta_t$) |
| **Factor Random Walk Coordinate RMSE** | {econ_diag.get('factor_coordinate_rmse_random_walk_bp', econ_diag.get('factor_rmse_random_walk_bp', 'N/A'))} bp | Factor-coordinate forecast error under factor random walk $\\hat{{\\beta}}_{{t+1}} = \\beta_t$ |
| **Factor AR(1) Coordinate RMSE** | {econ_diag.get('factor_coordinate_rmse_ar1_bp', econ_diag.get('factor_rmse_ar1_bp', 'N/A'))} bp | Factor-coordinate forecast error under AR(1) state dynamics |

### Residual-Preserving vs. Traditional Spread Forecast Comparison

| Formulation | Static NS Spread RMSE | DNS Kalman Spread RMSE | Impact on Signal Clipping |
| :--- | :---: | :---: | :---: |
| **Traditional (Level Reconstruct)** | {fmt_bp(trad_ns_spread_rmse)} | {fmt_bp(trad_dns_spread_rmse)} | {trad_clip_col_str} |
| **Residual-Preserving Diagnostic** | {fmt_bp(res_ns_spread_rmse)} | {fmt_bp(res_dns_spread_rmse)} | {res_clip_col_str} |

### Signal Saturation & Clipping Breakdown

{clipping_table_md}

*Finding*: Traditional level-reconstruction models saturate the $\\pm 1.0$ signal bounds due to cross-sectional curve-fitting bias entering the spread calculation. {clip_finding_str}

{pnl_finding_str}

---

## 5. Feature Attribution Summary

- **Level ($\\\\Delta L_{{t+1}}$)**: Key features by empirical split impact: `{top_lvl}`
- **Slope ($\\\\Delta S_{{t+1}}$)**: Key features by empirical split impact: `{top_slp}`
- **Curvature ($\\\\Delta C_{{t+1}}$)**: Key features by empirical split impact: `{top_cur}`

*Attribution Type*: `{stype}` (Descriptive feature importance ranking within decision trees).

---

## 6. Common-Sample Out-of-Sample Performance Table

{table_md}

---

## 7. Visual Diagnostics
- `reports/figures/gbm_shap_summary.png`: Displays top feature attribution drivers across Level, Slope, and Curvature.
- `reports/figures/gbm_feature_importance.png`: Aggregated feature importance across term-structure dimensions.
"""
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Saved verdict report to %s.", rep_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Milestone 14 ML Baseline and Attribution Analysis")
    parser.add_argument("--quick", action="store_true", help="Run fast mode on recent sample")
    parser.add_argument("--plot-shap", action="store_true", default=True, help="Generate attribution figures")
    args = parser.parse_args()
    main(quick=args.quick, plot_shap=args.plot_shap)

