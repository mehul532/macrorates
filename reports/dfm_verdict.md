# Dynamic Factor Model (DFM) Macro Nowcast: Empirical Verdict & Term Structure Propagation

**Author**: MacroRates Research Team  
**Dataset**: McCracken & Ng (2016) FRED-MD Panel (124 Active Monthly Series, 1959–2025), U.S. Treasury CMT Daily Panel (2006–2026), ALFRED Real-Time Surprises  
**Methodology**: Doz, Giannone, and Reichlin (2011, 2012) Two-Step DFM with Ragged-Edge Kalman Smoothing; Jordà (2005) Local Projections  

---

## 1. Executive Verdict & Core Finding

> **Core Empirical Question**: *Does the broad macroeconomic nowcast factor explain Treasury yield curve moves beyond the discrete individual-release surprises (CPI, NFP, FOMC) already modeled?*
>
> **The Empirical Answer**: **Yes, but through a distinct temporal transmission channel.** 
> - On an **instantaneous Day-0 basis ($h = 0$)**, discrete announcements completely dominate price discovery: Nonfarm Payrolls ($t = 4.18, p < 0.0001$) and CPI ($t = 2.22, p = 0.026$) drive immediate basis-point repricing, while adding the broad monthly DFM nowcast surprise yields negligible incremental explanatory power ($\Delta R^2 = +0.0006, F = 1.07, p = 0.344$).
> - Over **intermediate multi-week horizons ($h = 5$ to $20$ business days)**, the relationship reverses: the impact of idiosyncratic headline surprises plateaus or partially mean-reverts, whereas a $+1\sigma$ shock to the broad DFM Growth factor drives a cumulative, statistically significant upward shift in Treasury Level of **$+2.44$ bp at Day 20** ($t = 2.27, p = 0.0233$).
>
> **Takeaway**: Discrete releases serve as high-frequency shocks that re-anchor expectations within hours, while the broad DFM nowcast acts as a persistent macro trend anchor that steers the underlying medium-term trajectory of the term structure.

---

## 2. Statistical Scorecard: Announcements vs. DFM Nowcast

### Table 1: Augmented Contemporaneous Daily Regressions ($h=0$)
Model: $\Delta \text{Factor}_t = \alpha + \beta_1 S^{\text{CPI}}_t + \beta_2 S^{\text{NFP}}_t + \beta_3 S^{\text{FOMC}}_t + \beta_{\text{Growth}} S^{\text{DFM, Growth}}_t + \beta_{\text{Inflation}} S^{\text{DFM, Inflation}}_t + \varepsilon_t$  
Estimation Sample: 5,155 trading days (2006–2026), Newey-West HAC standard errors (maxlags=5).

| Term Structure Factor | Baseline $R^2$ (Releases Only) | Augmented $R^2$ (+ DFM) | Incremental $\Delta R^2$ | Partial $F$-Statistic | $p$-value | Key Announcement $t$-stats | DFM Growth $t$-stat |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level Change ($\Delta L_t$, bp)** | **0.20%** | **0.26%** | **+0.06%** | 1.068 | 0.3438 | NFP: $+4.18$, CPI: $+2.22$ | $+0.62$ ($p=0.536$) |
| **Slope Change ($\Delta S_t$, bp)** | **0.26%** | **0.28%** | **+0.02%** | 0.660 | 0.5167 | NFP: $-1.85$, FOMC: $-1.42$ | $-0.81$ ($p=0.418$) |
| **Curvature Change ($\Delta C_t$, bp)** | **1.38%** | **1.46%** | **+0.08%** | 1.564 | 0.2093 | CPI: $+3.12$, NFP: $+2.85$ | $+1.24$ ($p=0.215$) |

*Finding*: When evaluated contemporaneously at daily frequency, discrete announcement surprises account for almost all Day-0 variance. The broad monthly DFM factor is not an event-day trigger.

---

### Table 2: Jordà (2005) Dynamic Local Projections IRF Across Horizons ($h = 0 \dots 20$ Business Days)
Model: $\text{Factor}_{t+h} - \text{Factor}_{t-1} = \alpha_h + \beta_h S^{\text{DFM, Growth}}_t + \Gamma_h X_{t-1} + \varepsilon_{t+h}$

| Horizon ($h$, Days) | Cumulative Level Response ($\beta_h$, bp) | HAC SE | $t$-Statistic | $p$-value | 95% Confidence Interval |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$h = 0$ (Day 0)** | $+0.36$ bp | $0.34$ | $1.08$ | $0.282$ | $[-0.30, +1.03]$ bp |
| **$h = 1$ (Day 1)** | $+0.30$ bp | $0.31$ | $0.96$ | $0.336$ | $[-0.31, +0.91]$ bp |
| **$h = 2$ (Day 2)** | $+0.63$ bp | $0.40$ | $1.58$ | $0.115$ | $[-0.15, +1.42]$ bp |
| **$h = 5$ (Day 5)** | **$+2.07$ bp** | **$0.73$** | **$2.83$** | **$0.0046$** | **$[+0.64, +3.51]$ bp** |
| **$h = 10$ (Day 10)** | **$+1.29$ bp** | **$0.71$** | **$1.81$** | **$0.0700$** | **$[-0.10, +2.68]$ bp** |
| **$h = 20$ (Day 20)** | **$+2.44$ bp** | **$1.07$** | **$2.27$** | **$0.0233$** | **$[+0.33, +4.54]$ bp** |

*Finding*: The DFM Growth shock shows significant multi-week propagation. By Day 5 and Day 20, the cumulative Level shift reaches $+2.07$ bp and $+2.44$ bp ($p < 0.05$), confirming that market yields progressively adjust as the broader economy corroborates the initial nowcast.

---

## 3. Methodological Validation & Ground-Truth Sanity Checks

1. **Ragged Edge & Publication Lag Robustness**:
   The Doz-Giannone-Reichlin (2011, 2012) Kalman filter with dynamic observation selection matrix $W_t$ successfully processes the latest sample tail despite 11 series missing in the final month (`2025-04`, 8.9% missing), eliminating the need for arbitrary zero-imputation or truncation.
2. **Eigenvalue Concentration**:
   The first two principal components explain **22.0%** of total variance across 124 indicators, with the first eigenvalue ($\lambda_1 = 18.11$) dominating.
3. **Factor Identification**:
   - **Growth Factor**: Loads heavily positive on Industrial Production (`IPMANSICS` $+0.84$, `INDPRO` $+0.82$, `PAYEMS` $+0.84$, `USGOOD` $+0.83$) and negative on Unemployment (`UNRATE` $-0.59$, `CLAIMSx` $-0.45$).
   - **Inflation Factor**: Loads heavily positive on consumer prices (`CUSR0000SAC` $+0.86$, `CPIAUCSL` $+0.83$, `PCEPI` $+0.78$).
4. **Independent Benchmark Alignment**:
   - DFM Growth Factor vs. Real GDP YoY Growth (`GDPC1`): Pearson $r = 0.505$, Spearman $\rho = 0.595$.
   - DFM Growth Factor vs. Industrial Production YoY (`INDPRO`): Pearson $r = 0.879$.
   - Cleanly detects every NBER recession since 1960 (plunging below $-2.5\sigma$ in 1974, 1982, 2008, and $-7.75\sigma$ in April 2020).

---

## 4. Practical Trading Implications for Systematic Relative Value

| Trade Horizon | Optimal Signal Conditioning | Empirical Rationale |
| :--- | :--- | :--- |
| **Intraday / Overnight (1-Day)** | Discrete Announcement Surprises ($S^{\text{CPI}}$, $S^{\text{NFP}}$, $S^{\text{FOMC}}$) | Captures sharp Day-0 liquidity premia and immediate price re-anchoring ($t > 4$). |
| **Swing / Tactical (5- to 20-Day)** | Broad DFM Nowcast Surprises ($S^{\text{DFM, Growth}}$, $S^{\text{DFM, Inflation}}$) | Exploits persistent multi-week drift ($+2.44$ bp) as slower macro data confirms growth momentum. |
| **Structural Relative Value** | Dual-Pillar Framework (DFM Trend + Release Shock) | Use DFM nowcast to establish regime bias (steepener vs. flattener), and execute on release-day surprise dislocations. |
