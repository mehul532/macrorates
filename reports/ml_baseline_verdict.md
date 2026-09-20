# Milestone 14 Verdict: Machine Learning Baseline & Feature Attribution

**Author**: MacroRates Research Team  
**Git Commit**: `e93721bc2f811cbf2855cb4bddab14c2cf910a51`  
**Run Mode**: `QUICK_TWO_FOLD_EVALUATION` (Folds: 2)  
**Evaluated Sample**: 2026-06-29 to 2026-09-17 (57 trading days)  
**Backend**: `sklearn` | **Seed**: `42`  

> [!IMPORTANT]
> **RESEARCH INTEGRITY & CAUSALITY DISCLOSURE (Prompt 6)**:
> 1. **Observational vs. Causal Attribution**: TreeSHAP and Gini feature rankings describe statistical feature associations within the gradient-boosted decision trees. They are descriptive diagnostics, NOT proof of causal macroeconomic transmission mechanisms.
> 2. **Evaluation Scope**: If evaluated under a fast two-fold setting, results represent a recent sample validation, not a full-sample historical evaluation.
> 3. **Benchmark Discipline**: Random Walk provides the unparameterized zero-increment curve forecasting benchmark. Cash-Only provides an unencumbered capital strategy benchmark. Models are evaluated on common out-of-sample test dates without retroactive tuning or artificial error multipliers.

---

## 1. Run Provenance & Data Checksums

| Input Panel | SHA-256 Checksum (16-char) | Path |
| :--- | :--- | :--- |
| **Yield Panel** | `6c5f1abbb87b7697` | `data/processed/yield_panel.parquet` |
| **Factor Panel** | `fa37ee083dfe77ca` | `data/processed/factor_panel.parquet` |
| **Macro Surprises** | `6f7c134c1caf31f6` | `data/processed/macro_surprises.parquet` |

---

## 2. Feature Attribution Summary

- **Level ($\Delta L_{t+1}$)**: Key features by empirical split impact: `dlevel_1d, dcurvature_1d, dslope_1d`
- **Slope ($\Delta S_{t+1}$)**: Key features by empirical split impact: `dslope_1d, dlevel_1d, dslope_2d`
- **Curvature ($\Delta C_{t+1}$)**: Key features by empirical split impact: `dcurvature_1d, dlevel_1d, dlevel_2d`

*Attribution Type*: `tree_shap`. (Descriptive feature importance ranking within decision trees).

---

## 3. Common-Sample Out-of-Sample Performance Table

| Model / Forecast Method | Benchmark Role | Forecast Status | Sample Size (Days) | OOS Curve RMSE (bp) | 2Y Curve RMSE (bp) | 5Y Curve RMSE (bp) | 10Y Curve RMSE (bp) | 2s10s Spread RMSE (bp) | 2s5s10s Fly RMSE (bp) | Factor Forecast RMSE (bp) | Strategy Sharpe | Sortino Ratio | Max Drawdown (%) | Annual Turnover (lots) | Hit Rate (%) | PnL / DV01 ($) | PnL / Turnover ($/lot) | Gross PnL ($) | Trade Costs ($) | Roll Costs ($) | Trading Net PnL ($) | Cash Interest ($) | Collateral Net PnL ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Walk (Curve Benchmark) | Curve Benchmark | EVALUATED | 57 | 4.05 | 4.82 | 4.62 | 4.26 | 2.88 | 1.75 | 2.88 | N/A | N/A | 0.00 | 0.00 | 100.00 | 0.00 | N/A | 0.00 | 0.00 | 0.00 | 0.00 | 86896.71 | 86896.71 |
| PCA / VAR(1) | Term Structure Factor Model | EVALUATED | 57 | 5.14 | 5.23 | 5.27 | 4.47 | 4.64 | 3.86 | 4.64 | 0.04 | 0.06 | -1.03 | 11746.70 | 62.50 | 5.40 | 20.31 | 73068.31 | 17587.06 | 1520.00 | 53961.25 | 85575.97 | 139537.22 |
| AR(1) Baseline (Static NS) | AR(1) Baseline | EVALUATED | 57 | 12.75 | 15.36 | 12.16 | 14.63 | 28.82 | 23.03 | 28.82 | -0.01 | -0.01 | -2.24 | 1680.00 | 60.71 | -1.40 | -36.85 | -9967.08 | 2515.31 | 1520.00 | -14002.39 | 85480.88 | 71478.49 |
| DNS + Kalman | Dynamic Term Structure Model | EVALUATED | 57 | 12.83 | 16.52 | 11.17 | 14.48 | 29.90 | 22.38 | 29.90 | -0.01 | -0.01 | -2.24 | 1680.00 | 60.71 | -1.40 | -36.85 | -9967.08 | 2515.31 | 1520.00 | -14002.39 | 85480.88 | 71478.49 |
| DNS + Kalman + Macro | Macro-Augmented DTSM | EVALUATED | 57 | 12.83 | 16.52 | 11.17 | 14.48 | 29.90 | 22.38 | 29.90 | -0.01 | -0.01 | -1.17 | 968.20 | 60.71 | -0.81 | -37.02 | -5780.82 | 1449.59 | 876.00 | -8106.41 | 86080.62 | 77974.21 |
| GBM | Machine Learning Baseline | EVALUATED | 57 | 13.59 | 12.58 | 11.28 | 9.93 | 20.94 | 23.26 | 20.94 | -0.01 | -0.01 | -2.24 | 1680.00 | 60.71 | -1.40 | -36.85 | -9967.08 | 2515.31 | 1520.00 | -14002.39 | 85480.88 | 71478.49 |
| Cash Only (Strategy Benchmark) | Strategy Benchmark | BENCHMARK_ONLY | 57 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.00 | 0.00 | N/A | 0.00 | N/A | 0.00 | 0.00 | 0.00 | 0.00 | 86896.71 | 86896.71 |

---

## 4. Visual Diagnostics
- `reports/figures/gbm_shap_summary.png`: Displays top feature attribution drivers across Level, Slope, and Curvature.
- `reports/figures/gbm_feature_importance.png`: Aggregated feature importance across term-structure dimensions.
