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
    
    stype = ml_eval["shap_data"].get("type", "unknown")
    if stype == "tree_shap":
        shap_res = ml_eval["shap_data"].get("results", {})
        top_lvl = ", ".join(list(shap_res.get("target_dLevel", {}).get("mean_abs_shap", {}).head(3).index))
        top_slp = ", ".join(list(shap_res.get("target_dSlope", {}).get("mean_abs_shap", {}).head(3).index))
        top_cur = ", ".join(list(shap_res.get("target_dCurvature", {}).get("mean_abs_shap", {}).head(3).index))
    elif stype == "gini":
        imps = ml_eval["shap_data"].get("importances", {})
        top_lvl = ", ".join(list(imps.get("target_dLevel", pd.Series()).head(3).index))
        top_slp = ", ".join(list(imps.get("target_dSlope", pd.Series()).head(3).index))
        top_cur = ", ".join(list(imps.get("target_dCurvature", pd.Series()).head(3).index))
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

    content = f"""# Milestone 14 Verdict: Machine Learning Baseline & Feature Attribution

**Author**: MacroRates Research Team  
**Git Commit**: `{commit_hash}`  
**Run Mode**: `{run_mode}` (Folds: {fold_cnt})  
**Evaluated Sample**: {eval_start} to {eval_end} ({run_meta.get('total_eval_days', 'N/A')} trading days)  
**Backend**: `{backend}` | **Seed**: `{seed}`  

> [!IMPORTANT]
> **RESEARCH INTEGRITY & CAUSALITY DISCLOSURE (Prompt 6)**:
> 1. **Observational vs. Causal Attribution**: TreeSHAP and Gini feature rankings describe statistical feature associations within the gradient-boosted decision trees. They are descriptive diagnostics, NOT proof of causal macroeconomic transmission mechanisms.
> 2. **Evaluation Scope**: If evaluated under a fast two-fold setting, results represent a recent sample validation, not a full-sample historical evaluation.
> 3. **Benchmark Discipline**: Random Walk provides the unparameterized zero-increment curve forecasting benchmark. Cash-Only provides an unencumbered capital strategy benchmark. Models are evaluated on common out-of-sample test dates without retroactive tuning or artificial error multipliers.

---

## 1. Run Provenance & Data Checksums

| Input Panel | SHA-256 Checksum (16-char) | Path |
| :--- | :--- | :--- |
| **Yield Panel** | `{checksums.get('yield_panel', 'N/A')}` | `data/processed/yield_panel.parquet` |
| **Factor Panel** | `{checksums.get('factor_panel', 'N/A')}` | `data/processed/factor_panel.parquet` |
| **Macro Surprises** | `{checksums.get('macro_surprises', 'N/A')}` | `data/processed/macro_surprises.parquet` |

---

## 2. Feature Attribution Summary

- **Level ($\Delta L_{{t+1}}$)**: Key features by empirical split impact: `{top_lvl}`
- **Slope ($\Delta S_{{t+1}}$)**: Key features by empirical split impact: `{top_slp}`
- **Curvature ($\Delta C_{{t+1}}$)**: Key features by empirical split impact: `{top_cur}`

*Attribution Type*: `{stype}`. (Descriptive feature importance ranking within decision trees).

---

## 3. Common-Sample Out-of-Sample Performance Table

{table_md}

---

## 4. Visual Diagnostics
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

