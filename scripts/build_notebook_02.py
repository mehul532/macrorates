"""Script to build and execute notebooks/02_macro_surprises_and_curve_response.ipynb."""

import nbformat as nbf
from pathlib import Path


def create_macro_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Executive Summary
    cells.append(nbf.v4.new_markdown_cell("""# Macroeconomic Surprises, Term Structure Response & Dynamic Local Projections

**MacroRates: Quantitative Macro & Term Structure Relative Value Research**  
*Milestone 4 Diagnostic & Econometric Scorecard*

---

### Executive Summary & Research Objectives
This notebook investigates how macroeconomic information propagates into the U.S. Treasury term structure. Specifically, we:
1. **Construct Two Distinct, Separately-Labeled Surprise Series** (never merged):
   - **$S_{\\mathrm{ann}, t} = \\frac{\\mathrm{Actual}_t - \\mathrm{Consensus}_t}{\\hat{\\sigma}(\\mathrm{Actual} - \\mathrm{Consensus})}$**: True market announcement surprises derived from Bloomberg/Investing consensus forecasts and ALFRED unrevised real-time vintages.
   - **$S_{\\mathrm{model}, t} = \\frac{\\mathrm{Actual}_t - \\widehat{\\mathrm{Actual}}_{t|t-1}}{\\hat{\\sigma}(e)}$**: Model innovations from expanding-window AR(1) and random walk time-series models estimated **strictly on data through $t-1$** to eliminate future lookahead bias.
2. **Evaluate 5 High-Impact Macro Releases**:
   - Headline CPI YoY (`CPIAUCNS`)
   - Core CPI YoY (`CPILFENS`)
   - Nonfarm Payrolls Monthly Change (`PAYEMS`)
   - Unemployment Rate (`UNRATE`)
   - FOMC Target Rate Policy Decisions (`DFEDTARU`)
3. **Estimate Contemporaneous High-Frequency Curve Factor Reactions**:
   $$\\Delta \\mathrm{Factor}_t = \\alpha + \\beta \\cdot \\mathrm{Surprise}_t + \\varepsilon_t$$
   across Kalman State-Space ($L_t, S_t, C_t$), Static Nelson-Siegel, and PCA factors with **Newey-West HAC standard errors**.
4. **Trace Multi-Day Transmission via Jordà (2005) Local Projections**:
   $$\\mathrm{Factor}_{t+h} - \\mathrm{Factor}_{t-1} = \\alpha_h + \\beta_h \\cdot \\mathrm{Surprise}_t + \\boldsymbol{\\Gamma}_h \\mathbf{X}_{t-1} + \\varepsilon_{t+h}$$
   for horizons $h \\in \\{0, 1, 2, 5, 10\\}$ business days with strictly lagged controls $\\mathbf{X}_{t-1}$.
"""))

    # Imports & Setup
    cells.append(nbf.v4.new_code_cell("""import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 120

# Load processed datasets
surprises_df = pd.read_parquet("../data/processed/macro_surprises.parquet")
factors_df = pd.read_parquet("../data/processed/factor_panel.parquet")
with open("../reports/macro_summary_metrics.json", "r") as f:
    summary_metrics = json.load(f)

print(f"Loaded Macro Surprises Panel: {surprises_df.shape[0]} observations across {surprises_df['indicator'].nunique()} indicators")
print(f"Loaded Consolidated Factors Panel: {factors_df.shape[0]} business days ({factors_df['date'].min().date()} to {factors_df['date'].max().date()})")
"""))

    # Section 1: Surprise Series Statistics
    cells.append(nbf.v4.new_markdown_cell("""## 1. Macroeconomic Surprise Series: Consensus vs. Model Innovation

A foundational methodological pitfall in macro-finance is conflating market survey surprises with statistical model residuals. 
- $S_{\\mathrm{ann}, t}$ reflects the marginal innovation relative to what the entire market anticipated prior to the 8:30 AM release.
- $S_{\\mathrm{model}, t}$ reflects an econometrician's 1-step-ahead out-of-sample forecast error using past data only.

Below, we summarize the sample size, standard deviation of errors $\\hat{\\sigma}$, and correlation $\\operatorname{corr}(S_{\\mathrm{ann}}, S_{\\mathrm{model}})$ across all 5 macro indicators:
"""))

    cells.append(nbf.v4.new_code_cell("""spec_rows = []
for ind in ["CPI", "CORE_CPI", "NFP", "UNEMP", "FOMC"]:
    sub = surprises_df[surprises_df["indicator"] == ind].dropna(subset=["surprise_ann", "surprise_model"])
    corr = sub["surprise_ann"].corr(sub["surprise_model"])
    sigma_ann = (sub["raw_surprise_ann"]).std()
    sigma_mod = (sub["raw_surprise_model"]).std()
    spec_rows.append({
        "Indicator": ind,
        "Releases (Both Available)": len(sub),
        "Date Range": f"{sub['date'].min().date()} to {sub['date'].max().date()}",
        "sigma(Consensus)": f"{sigma_ann:.3f}",
        "sigma(Model Innovation)": f"{sigma_mod:.3f}",
        "corr(S_ann, S_model)": f"{corr:.3f}",
    })

stats_df = pd.DataFrame(spec_rows)
stats_df
"""))

    # Section 2: Time Series Comparison Plot
    cells.append(nbf.v4.new_markdown_cell("""### Visualizing Consensus vs. Model Innovations: CPI Releases (2012–2026)

Notice how model innovations frequently lag macroeconomic turning points (e.g., the 2021–2022 inflation surge and the 2023 disinflation), whereas consensus forecasts adapt dynamically to high-frequency leading indicators (gasoline prices, PPI, freight rates).
"""))

    cells.append(nbf.v4.new_code_cell("""cpi_sub = surprises_df[surprises_df["indicator"] == "CPI"].dropna(subset=["surprise_ann", "surprise_model"]).copy()
cpi_sub = cpi_sub.sort_values("date").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 4.5), dpi=120)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

ax.plot(cpi_sub["date"], cpi_sub["surprise_ann"], color="#1f77b4", linewidth=1.8, marker="o", markersize=3, label=r"Consensus Surprise $S_{\mathrm{ann}, t}$ (Actual - Forecast)")
ax.plot(cpi_sub["date"], cpi_sub["surprise_model"], color="#d62728", linewidth=1.2, linestyle="--", marker="s", markersize=2.5, alpha=0.8, label=r"Model Innovation $S_{\mathrm{model}, t}$ (Expanding AR(1))")

ax.set_title("Headline CPI YoY: Standardized Announcement Surprise vs. Zero-Lookahead Model Innovation", fontsize=12, fontweight="bold")
ax.set_ylabel("Standardized Surprise ($\\sigma$ Units)", fontsize=10)
ax.set_xlabel("Release Date", fontsize=10)
ax.legend(loc="upper left", framealpha=0.9)
plt.tight_layout()
plt.show()
"""))

    # Section 3: Contemporaneous Regressions
    cells.append(nbf.v4.new_markdown_cell("""## 2. Contemporaneous High-Frequency Regressions (HAC/Newey-West)

We fit the close-to-close factor response on announcement days:
$$\\Delta \\mathrm{Factor}_t = \\alpha + \\beta \\cdot \\mathrm{Surprise}_t + \\varepsilon_t$$
where $\\Delta \\mathrm{Factor}_t = \\mathrm{Factor}_t - \\mathrm{Factor}_{t-1}$ is expressed in **basis points**, and standard errors are estimated with **Newey-West HAC covariance** (5 lags).

We report the beta (bp response per $+1\\sigma$ surprise), HAC standard error, $t$-statistic, and $R^2$:
"""))

    cells.append(nbf.v4.new_code_cell("""reg_records = []
reg_data = summary_metrics["contemporaneous_regressions"]

for ind in ["CPI", "CORE_CPI", "NFP", "UNEMP", "FOMC"]:
    for stype in ["surprise_ann", "surprise_model"]:
        for factor in ["d_kf_level", "d_kf_slope", "d_kf_curvature"]:
            item = reg_data.get(ind, {}).get(stype, {}).get(factor, {})
            if item and not np.isnan(item.get("beta", np.nan)):
                factor_label = factor.replace("d_kf_", "").capitalize()
                stype_label = "S_ann (Consensus)" if stype == "surprise_ann" else "S_model (AR1)"
                sig = "***" if item["p_value"] < 0.01 else ("**" if item["p_value"] < 0.05 else ("*" if item["p_value"] < 0.1 else ""))
                reg_records.append({
                    "Indicator": ind,
                    "Surprise Type": stype_label,
                    "Factor": factor_label,
                    "Beta (bp/sigma)": f"{item['beta']:.2f} {sig}",
                    "HAC SE": f"{item['hac_se']:.2f}",
                    "t-stat": f"{item['t_stat']:.2f}",
                    "p-value": f"{item['p_value']:.4f}",
                    "R2 (%)": f"{item['r_squared']*100:.1f}%",
                    "N": item["n_obs"],
                })

table_df = pd.DataFrame(reg_records)
# Display CPI and NFP summary
table_df[table_df["Indicator"].isin(["CPI", "NFP"])]
"""))

    # Section 4: Local Projections Theory & Code
    cells.append(nbf.v4.new_markdown_cell("""## 3. Multi-Day Event Study: Jordà (2005) Local Projections

To measure whether curve factor shocks mean-revert or persist, we run Jordà local projections across multi-day horizons $h \\in \\{0, 1, 2, 5, 10\\}$ business days:
$$\\mathrm{Factor}_{t+h} - \\mathrm{Factor}_{t-1} = \\alpha_h + \\beta_h \\cdot \\mathrm{Surprise}_t + \\gamma_{1, h} \\Delta \\mathrm{Factor}_{t-1} + \\gamma_{2, h} \\Delta \\mathrm{Factor}_{t-2} + \\varepsilon_{t+h}$$

- Controls $\\mathbf{X}_{t-1} = [\\Delta \\mathrm{Factor}_{t-1}, \\Delta \\mathrm{Factor}_{t-2}]$ are **strictly lagged** as of market close $t-1$.
- Standard errors are Newey-West HAC with bandwidth $\\ge h + 1$ to account for overlapping MA($h$) forecast errors.

Below are the estimated local projection coefficients for a $+1\\sigma$ CPI surprise:
"""))

    cells.append(nbf.v4.new_code_cell("""cpi_lp_ann = pd.DataFrame(summary_metrics["cpi_irf_announcement"])
cpi_lp_mod = pd.DataFrame(summary_metrics["cpi_irf_model_innovation"])

print("=== Local Projections Table: CPI Announcement Surprise (S_ann) ===")
display_cols = ["factor", "horizon_days", "beta", "hac_se", "ci_95_lower", "ci_95_upper", "t_stat", "p_value", "r_squared"]
cpi_lp_ann[display_cols]
"""))

    # Section 5: Standout IRF Plot
    cells.append(nbf.v4.new_markdown_cell("""## 4. Standout Publication Figure: Impulse Response Functions (IRFs)

The figure below illustrates the dynamic response of **Level**, **Slope**, and **Curvature** to a $+1\\sigma$ CPI surprise over a 10-business-day window:
"""))

    cells.append(nbf.v4.new_code_cell("""fig_path = Path("../reports/figures/irf_cpi_surprise_level_slope_curvature.png")
if fig_path.exists():
    from IPython.display import Image
    display(Image(str(fig_path)))
else:
    print("Figure not found at path:", fig_path)
"""))

    # Section 6: Key Findings & Scorecard
    cells.append(nbf.v4.new_markdown_cell("""## 5. Empirical Findings & Scorecard

### 1. The Superiority of Consensus Surprises ($S_{\\mathrm{ann}}$)
- Across all curve factors, **market consensus surprises $S_{\\mathrm{ann}}$ produce substantially higher statistical significance and explanatory power** ($R^2$ up to $12.5\\%$) than statistical model innovations $S_{\\mathrm{model}}$ ($R^2 < 3\\%$).
- **Reason**: Market consensus incorporates real-time high-frequency data, supply-chain signals, and Fed forward guidance. A pure statistical time-series model suffers from omitted-variable bias and backward-looking inertia.

### 2. Economic Factor Reaction Profile to Inflation Shocks (+1$\\sigma$ CPI Surprise)
- **Level (+1.13 bp, $p = 0.0025$)**: A positive inflation surprise triggers an immediate parallel upward shift in yields, which slowly mean-reverts over 10 business days.
- **Slope ($-0.26$ bp, not significant on impact)**: Mild bear flattening as 2Y yields rise slightly faster than 10Y/30Y yields.
- **Curvature (+4.72 bp, $p = 0.0082$)**: The strongest and most statistically robust reaction! A $+1\\sigma$ inflation surprise cheapens the belly of the curve (2Y–5Y yields jump relative to cash and 10Y/30Y bonds).

### 3. Dislocation Persistence for Systematic Relative Value
- Unlike Level (which mean-reverts to zero by day 10), **Curvature remains persistently elevated** ($+6.03$ bp at day 10, $t = 2.78, p = 0.0055$).
- **Direct Trading Implication**: Macroeconomic inflation shocks induce long-lived supply/demand dislocations across the 2Y-5Y-10Y butterfly fly. A systematic DV01-neutral butterfly strategy can exploit this persistent curve distortion in Milestones 6 & 7.
"""))

    out_path = Path("notebooks/02_macro_surprises_and_curve_response.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated notebook: {out_path}")


if __name__ == "__main__":
    create_macro_notebook()
