# Milestone 14 Verdict: Machine Learning Baseline & Feature Attribution

**Author**: MacroRates Research Team  
**Git Commit**: `34c17a2aab19fea81f0cf74dc88dc0a17f5e7928`  
**Run Mode**: `QUICK_TWO_FOLD_EVALUATION` (Folds: 2)  
**Evaluated Sample**: 2026-06-29 to 2026-09-17 (57 trading days)  
**Backend**: `sklearn` | **Seed**: `42`  

> [!IMPORTANT]
> **RESEARCH INTEGRITY & CAUSALITY AUDIT DISCLOSURES**:
> 1. **Macro Data Coverage & Status**: The committed macro dataset (`data/processed/macro_surprises.parquet`) ends on 2026-04-10. During the evaluated test period (2026-06-29 to 2026-09-17), there were **0 calendar test events** and **0 evaluated decision origin events**. Therefore, `DNS + Kalman + Macro` is formally audited and categorized as **`NOT_EVALUATED (NO_TEST_RELEASES)`**. No empirical claims of out-of-sample macro forecasting superiority or trading alpha are supported by this test window.
> 2. **60% Exposure Control Finding**: The control model `DNS (60% Exposure Control)` trades an exact linear scaling of the baseline DNS signal ($s_t = 0.60 \times s_{t}^{\text{DNS}}$). In the evaluation, it produced trading results identical to `DNS + Kalman + Macro`, confirming that any historical PnL difference was entirely due to linear risk/exposure downscaling, not macroeconomic information.
> 3. **Observational vs. Causal Attribution**: TreeSHAP and Gini feature rankings describe statistical feature associations within gradient-boosted decision trees. They are descriptive diagnostics, NOT proof of causal macroeconomic transmission mechanisms.
> 4. **Econometric Decomposition**: Traditional level-reconstruction models (Static NS, DNS Kalman) exhibit ~28.8 bp 2s10s spread RMSE driven predominantly by cross-sectional curve-fitting errors (~28.5 bp in spread space), which push spread forecasts into extreme values that saturate the $\pm 1.0$ signal clip. The diagnostic residual-preserving formulation ($\hat{y}_{t+1|t}^{\text{res}} = y_t + \Lambda(\hat{\beta}_{t+1|t} - \beta_t)$) eliminates this cross-sectional bias, reducing spread RMSE from 28.82 bp to 2.90 bp and signal clipping from 100% to 0%.
> 5. **Annualization & Statistics Corrections**: Sharpe and Sortino ratios are annualized with $\sqrt{252}$ strictly on standard deviation ($(\mu \times 252) / (\sigma \times \sqrt{252}) = (\mu \times \sqrt{252}) / \sigma$). Hit rates are computed strictly over active trading days; zero-trading benchmarks (Random Walk, Cash Only) report `N/A`, avoiding false 100% hit rate claims.
> 6. **Benchmark Discipline**: Random Walk provides the unparameterized zero-increment curve forecasting benchmark. Cash-Only provides an unencumbered capital strategy benchmark. Models are evaluated on common out-of-sample test dates without retroactive tuning or artificial error multipliers.

---

## 1. Run Provenance & Data Checksums

| Input Panel | SHA-256 Checksum (16-char) | Path |
| :--- | :--- | :--- |
| **Yield Panel** | `6c5f1abbb87b7697` | `data/processed/yield_panel.parquet` |
| **Factor Panel** | `fa37ee083dfe77ca` | `data/processed/factor_panel.parquet` |
| **Macro Surprises** | `6f7c134c1caf31f6` | `data/processed/macro_surprises.parquet` |

---

## 2. Macro Event Coverage Audit

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Total Training Events** | 301 | Macro surprise releases available in training windows |
| **Calendar Test Window Events** | 0 | Releases occurring within calendar test dates |
| **Evaluated Decision Origin Events** | 0 | Releases available at evaluated decision origins (orig_dates) |
| **Timestamp-Available Events** | 0 | Releases verified available prior to market close |
| **Usable Surprise Events** | 0 | Releases with non-null numeric surprise |
| **Eligible Coefficient Events** | 0 | Releases with causal response beta estimated from training history |
| **Nonzero Macro Days in Test** | 0 | Evaluated decision days where macro overlay was non-zero |
| **Curve Forecast Provenance** | `COPIED_DNS_CURVE_FORECAST` | Independent provenance of yield curve predictions |
| **Macro Strategy Status** | `INACTIVE_ZERO_RELEASES` | Operational status of macro overlay strategy |
| **Audit Status** | `NOT_EVALUATED (NO_TEST_RELEASES)` | Formal audit determination for `DNS_Kalman_Macro` |

*Note: In the absence of test releases, DNS with macro surprise augmentation degenerates to baseline state dynamics scaled by prior event variance.*

---

## 3. Econometric Diagnostics & Error Decomposition

### Observable 2s10s Spread Forecast Error Decomposition ($e_s = u_{\text{factor}} + u_{\text{fit}}$)
*Exact mathematical decomposition in the observable 2s10s spread space, including the twice uncentered second cross moment:*

| Error Component | Symbol | Spread RMSE (bp) | Spread MSE (bp²) | Econometric Description |
| :--- | :---: | :---: | :---: | :--- |
| **Total Observable 2s10s Spread Error** | $e_s$ | **28.82 bp** | **830.39 bp²** | Actual observed spread minus forecast: $s_t - \hat{s}_t$ |
| **Factor-Driven Spread Dynamics Error** | $u_{\text{factor}}$ | 2.98 bp | 8.9 bp² | In-sample fitted spread minus forecast: $s_t^{\text{fit}} - \hat{s}_t$ |
| **Cross-Sectional Parametric Fit Error** | $u_{\text{fit}}$ | 28.46 bp | 810.21 bp² | Actual observed spread minus fitted spread: $s_t - s_t^{\text{fit}}$ |
| **Twice Uncentered Second Cross Moment** | $2 \times \mathbb{E}[u_{\text{factor}} \cdot u_{\text{fit}}]$ | — | 11.28 bp² | Second cross moment: $2 \times \frac{1}{N} \sum u_{\text{factor}} \cdot u_{\text{fit}}$ (distinct from centered covariance) |
| **Sum of Decomposition Components** | $\sum$ | — | **830.39 bp²** | Exact mathematical identity: MSE($u_{\text{factor}}$) + MSE($u_{\text{fit}}$) + $2 \times \mathbb{E}[u_{\text{factor}} \cdot u_{\text{fit}}]$ |

*Note*: As proven above, factor dynamics contribute 2.98 bp of spread error (comparable to Random Walk's 2.88 bp), while cross-sectional curve-fitting error is the dominant contributor (28.46 bp), demonstrating that spread forecast failure is driven predominantly by static parametric fitting error, not factor dynamics.

### Factor Dynamics Diagnostics (Factor-Coordinate Space)
*These diagnostics evaluate state variable forecasting in factor-coordinate space, distinct from observable 2s10s spread error decomposition:*

| Diagnostic Metric | Value (bp) | Description |
| :--- | :---: | :--- |
| **Contemporaneous NS Curve Fit RMSE** | 12.03 bp | Full-curve cross-sectional parametric fitting error ($y_t - \Lambda \beta_t$) |
| **Factor Random Walk Coordinate RMSE** | 8.67 bp | Factor-coordinate forecast error under factor random walk $\hat{\beta}_{t+1} = \beta_t$ |
| **Factor AR(1) Coordinate RMSE** | 8.89 bp | Factor-coordinate forecast error under AR(1) state dynamics |

### Residual-Preserving vs. Traditional Spread Forecast Comparison

| Formulation | Static NS Spread RMSE | DNS Kalman Spread RMSE | Impact on Signal Clipping |
| :--- | :---: | :---: | :---: |
| **Traditional (Level Reconstruct)** | 28.82 bp | 29.90 bp | 100.0% clipped to $\pm 1.0$ bounds |
| **Residual-Preserving Diagnostic** | 2.90 bp | 2.90 bp | 0.0% clipped (natural dynamic variation) |

### Signal Saturation & Clipping Breakdown

| Model | Raw Threshold Exceedance ($\ge 1.0$) (%) | Inherited DNS Clipping (%) | Additional Stage Clipping (%) | Final Position Saturation (%) | Mean Unclipped |Input| |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `DNS_Kalman` | 100.0% | 0.0% | 100.0% | 100.0% | 8.139 |
| `DNS_Kalman_Macro` | 0.0% | 100.0% | 0.0% | 0.0% | 0.600 |
| `DNS_Kalman_Residual_Preserving` | 0.0% | 0.0% | 0.0% | 0.0% | 0.027 |
| `DNS_Scaled_60` | 0.0% | 100.0% | 0.0% | 100.0% | 0.600 |
| `GBM` | 100.0% | 0.0% | 100.0% | 100.0% | 5.668 |
| `PCA_VAR` | 47.4% | 0.0% | 47.4% | 47.4% | 0.948 |
| `Random_Walk` | 0.0% | 0.0% | 0.0% | 0.0% | 0.000 |
| `Static_NS` | 100.0% | 0.0% | 100.0% | 100.0% | 7.834 |
| `Static_NS_Residual_Preserving` | 0.0% | 0.0% | 0.0% | 0.0% | 0.032 |

*Finding*: Traditional level-reconstruction models saturate the $\pm 1.0$ signal bounds due to cross-sectional curve-fitting bias entering the spread calculation. The residual-preserving formulation eliminates this bias, preserving the natural signal variation without clipping.

*Finding*: The identical trading PnL (-$46,684.38) observed across Static NS, DNS Kalman, and GBM in traditional level reconstruction is explained by 100% signal saturation resulting from cross-sectional curve-fitting error propagation.

---

## 4. Feature Attribution Summary

- **Level ($\Delta L_{t+1}$)**: Key features by empirical split impact: `dlevel_1d, dcurvature_1d, dslope_1d`
- **Slope ($\Delta S_{t+1}$)**: Key features by empirical split impact: `dslope_1d, dlevel_1d, dslope_2d`
- **Curvature ($\Delta C_{t+1}$)**: Key features by empirical split impact: `dcurvature_1d, dlevel_1d, dlevel_2d`

*Attribution Type*: `tree_shap` (Descriptive feature importance ranking within decision trees).

---

## 5. Common-Sample Out-of-Sample Performance Table

| Model / Forecast Method | Benchmark Role | Forecast Status | Sample Size (Days) | OOS Curve RMSE (bp) | 2Y Curve RMSE (bp) | 5Y Curve RMSE (bp) | 10Y Curve RMSE (bp) | 2s10s Spread RMSE (bp) | 2s5s10s Fly RMSE (bp) | Factor Forecast RMSE (bp) | Strategy Sharpe | Sortino Ratio | Max Drawdown (%) | Annual Turnover (lots) | Hit Rate (%) | PnL / DV01 ($) | PnL / Turnover ($/lot) | Gross PnL ($) | Trade Costs ($) | Roll Costs ($) | Trading Net PnL ($) | Cash Interest ($) | Collateral Net PnL ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Walk (Curve Benchmark) | Curve Benchmark | EVALUATED | 57 | 4.05 | 4.82 | 4.62 | 4.26 | 2.88 | 1.75 | 2.88 | N/A | N/A | 0.00 | 0.00 | N/A | 0.00 | N/A | 0.00 | 0.00 | 0.00 | 0.00 | 90116.11 | 90116.11 |
| PCA / VAR(1) | Term Structure Factor Model | EVALUATED | 57 | 5.14 | 5.23 | 5.27 | 4.47 | 4.64 | 3.86 | 4.64 | 0.85 | 1.23 | -0.85 | 12131.40 | 48.28 | 6.44 | 23.48 | 84108.00 | 18162.88 | 1520.00 | 64425.12 | 88484.37 | 152909.50 |
| AR(1) Baseline (Static NS) | AR(1) Baseline | EVALUATED | 57 | 12.75 | 15.36 | 12.16 | 14.63 | 28.82 | 23.03 | 28.82 | -0.44 | -0.61 | -2.26 | 3360.00 | 50.88 | -4.67 | -61.43 | -40133.76 | 5030.62 | 1520.00 | -46684.38 | 88344.94 | 41660.55 |
| DNS + Kalman | Dynamic Term Structure Model | EVALUATED | 57 | 12.83 | 16.52 | 11.17 | 14.48 | 29.90 | 22.38 | 29.90 | -0.44 | -0.61 | -2.26 | 3360.00 | 50.88 | -4.67 | -61.43 | -40133.76 | 5030.62 | 1520.00 | -46684.38 | 88344.94 | 41660.55 |
| DNS + Kalman + Macro | Macro-Augmented DTSM | NOT_EVALUATED (NO_TEST_RELEASES) | 57 | 12.83 | 16.52 | 11.17 | 14.48 | 29.90 | 22.38 | 29.90 | -0.44 | -0.61 | -1.17 | 1936.40 | 50.88 | -2.69 | -61.51 | -23166.96 | 2899.19 | 876.00 | -26942.15 | 89095.23 | 62153.08 |
| GBM | Machine Learning Baseline | EVALUATED | 57 | 13.59 | 12.58 | 11.28 | 9.93 | 20.94 | 23.26 | 20.94 | -0.44 | -0.61 | -2.26 | 3360.00 | 50.88 | -4.67 | -61.43 | -40133.76 | 5030.62 | 1520.00 | -46684.38 | 88344.94 | 41660.55 |
| DNS (60% Exposure Control) | Risk-Scaling Control (60% Exposure) | CONTROL (SCALED_DNS) | 57 | 12.83 | 16.52 | 11.17 | 14.48 | 29.90 | 22.38 | 29.90 | -0.44 | -0.61 | -1.17 | 1936.40 | 50.88 | -2.69 | -61.51 | -23166.96 | 2899.19 | 876.00 | -26942.15 | 89095.23 | 62153.08 |
| Static NS (Residual-Preserving Diagnostic) | Diagnostic (Static NS Residual-Preserving) | DIAGNOSTIC | 57 | 4.36 | 5.20 | 5.01 | 4.57 | 2.90 | 1.77 | 2.90 | -0.26 | -0.37 | -0.05 | 1282.10 | 43.86 | -0.21 | -7.11 | -27.56 | 1919.38 | 116.00 | -2062.93 | 89982.55 | 87919.62 |
| DNS + Kalman (Residual-Preserving Diagnostic) | Diagnostic (DNS Kalman Residual-Preserving) | DIAGNOSTIC | 57 | 4.36 | 5.19 | 5.01 | 4.57 | 2.90 | 1.77 | 2.90 | -1.11 | -1.54 | -0.05 | 1794.90 | 42.11 | -0.90 | -22.06 | -6152.46 | 2687.12 | 116.00 | -8955.58 | 89959.92 | 81004.33 |
| Cash Only (Strategy Benchmark) | Strategy Benchmark | BENCHMARK_ONLY | 57 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.00 | 0.00 | N/A | 0.00 | N/A | 0.00 | 0.00 | 0.00 | 0.00 | 90116.11 | 90116.11 |

---

## 6. Visual Diagnostics
- `reports/figures/gbm_shap_summary.png`: Displays top feature attribution drivers across Level, Slope, and Curvature.
- `reports/figures/gbm_feature_importance.png`: Aggregated feature importance across term-structure dimensions.
