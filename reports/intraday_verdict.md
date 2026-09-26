# Intraday Treasury Futures Event Study: Microstructure Simulation Demonstration & Provenance Audit

**Author**: MacroRates Research Team  
**Dataset Architecture**: CME Globex 1-Minute Futures Pipeline (ZN 10-Year, ZF 5-Year, ZT 2-Year), Databento GLBX.MDP3 Schema (`ohlcv-1m`), Curated High-Profile Macro Surprises (CPI, NFP, FOMC, 2020–2024)  
**Econometric Pipeline**: High-Frequency Event Window Ingestion ($-5\text{m}$ to $+30\text{m}$), Realized Volatility Estimation, Jordà (2005) Intraday Local Projections with Newey-West HAC Covariance  

> [!IMPORTANT]
> **RESEARCH INTEGRITY, SIMULATION STATUS & DATA PROVENANCE AUDIT**:
> 1. **Simulation Demonstration, Not Live Tick Discovery**: In the absence of an authenticated live Databento exchange connection (`DATABENTO_API_KEY`), this study functions strictly as an architectural verification and software demonstration of the intraday event-study engine using the repository's calibrated synthetic jump-diffusion generator (`_generate_realistic_intraday_bars`).
> 2. **Front-Loading & Volatility Decay Are Model Parameters**: In simulation mode, the observed high front-loading (~94.6% within 5 minutes) and rapid realized volatility decay ($\tau \approx 3.3$ min) are **direct mathematical consequences** of the hard-coded jump-diffusion calibration ($50\%$ instantaneous jump at $t=0$ plus exponential decay). They MUST NOT be cited or interpreted as empirical market facts or real-world price discovery findings.
> 3. **Provenance Audit of Legacy Caches (`UNKNOWN_UNVERIFIED_CACHE`)**: Cached event files in `data/processed/intraday_events/` generated without verified provenance tags are formally categorized as **`UNKNOWN_UNVERIFIED_CACHE`**. The repository does not assume or infer that unverified parquet files originate from real exchange data.
> 4. **Window Boundary vs. Daily Settlement**: The horizon labeled "Window End ($+60\text{m}$)" corresponds strictly to the $+60$-minute post-announcement window boundary ($p_{\text{window\_end}}$). It is NOT the official CME 3:00 PM or 5:00 PM ET daily settlement price.
> 5. **Withdrawal of Execution & Alpha Claims**: Claims regarding optimal algorithmic execution, passive liquidity provision, or market-making profit opportunities are unsupported by simulated data and are formally withdrawn. Systematic relative-value models in this repository rely strictly on daily settlement and macro surprise conditioning.

---

## 1. High-Frequency Pipeline Architecture & Demonstration Scope

The intraday event analysis pipeline is designed to evaluate how macroeconomic news releases transmit into Treasury futures contracts (2-Year ZT, 5-Year ZF, 10-Year ZN).

```
Macro Surprise Announcement (t_0)
        │
        ├── Standardized Event Window: [-5m, +30m] (Extended: +60m)
        ├── Implied Yield Conversion: Delta_y = - (Delta_P * PointValue) / DV01
        ├── Microstructure Dynamics: Realized Volatility RV(t_1, t_2) & Volatility Spike Ratio
        └── Response Share Tracking: |Delta_y_h| / |Delta_y_window_end|
```

In demonstration mode, the synthetic jump-diffusion generator evaluates the pipeline's numerical stability, date alignment, and reporting logic under realistic event parameters:

$$\Delta P_t = \mu_t \Delta t + \sigma_t \varepsilon_t + J_t$$

where $J_0 = 0.50 \times \beta_{\text{impact}} \times S_{\text{ann}}$, and volatility $\sigma_t$ decays exponentially toward baseline.

---

## 2. Simulation Demonstration Scorecard: Synthetic Microstructure Response

### Table 1: Synthetic Response & Implied Yield Dynamics (ZN 10-Year Simulation)
*Base Reference Price*: $P_{t_0 - 1\text{m}}$ (1 minute prior to release). Implied yield conversion: $\Delta y = -\frac{\Delta P \times 1,000}{DV01}$.  
*Note*: Horizon $h = +60\text{m}$ denotes the $+60$-minute post-announcement window end ($p_{\text{window\_end}}$).

| Horizon | Mean Response Share ($|\Delta y_h| / |\Delta y_{\text{window\_end}}|$) | Median Response Share | Net Cumulative Drift | Mean Realized Volatility ($\text{bp}$) | Volatility Ratio vs Baseline | Source Tag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pre-Release ($-5\text{m}$ to $-1\text{m}$)** | — | — | Baseline | $0.54\text{ bp}$ | $1.0\times$ | `SYNTHETIC_CALIBRATION_FALLBACK` |
| **$t = +1\text{ minute}$** | **$75.7\%$** | **$78.2\%$** | $+2.57\text{ bp} / \sigma$ | $8.42\text{ bp}$ | $15.6\times$ | `SYNTHETIC_CALIBRATION_FALLBACK` |
| **$t = +5\text{ minutes}$** | **$94.6\%$** | **$93.1\%$** | $+3.66\text{ bp} / \sigma$ | $10.58\text{ bp}$ | **$19.6\times$** | `SYNTHETIC_CALIBRATION_FALLBACK` |
| **$t = +15\text{ minutes}$** | **$96.1\%$** | **$95.4\%$** | $+4.21\text{ bp} / \sigma$ | $4.12\text{ bp}$ | $7.6\times$ | `SYNTHETIC_CALIBRATION_FALLBACK` |
| **$t = +30\text{ minutes}$** | **$96.5\%$** | **$96.0\%$** | $+4.25\text{ bp} / \sigma$ | $2.31\text{ bp}$ | $4.3\times$ | `SYNTHETIC_CALIBRATION_FALLBACK` |
| **Window End ($h = +60\text{m}$)** | **$100.0\%$** | **$100.0\%$** | $+4.44\text{ bp} / \sigma$ | $1.20\text{ bp}$ | $2.2\times$ | `SYNTHETIC_CALIBRATION_FALLBACK` |

---

## 3. Indicator-Specific Simulation Parameters

The simulation pipeline differentiates across release categories via calibrated sensitivity vectors ($\beta_{\text{impact}}$):

### Table 2: Parametric Configurations Across Release Categories

| Indicator | Event Count ($N$) | Calibrated Impact ($\beta_{\text{ZN}}$) | Simulated $1\text{m}$ Share | Simulated $5\text{m}$ Share | Simulated Peak Volatility | Model Dynamics Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **CPI (Consumer Price Index)** | $7$ | $-0.32$ pts / $\sigma$ | $51.2\%$ | $67.7\%$ | $15.9\times$ | Two-stage dispersion modeling initial print and secondary component digestion. |
| **NFP (Nonfarm Payrolls)** | $6$ | $-0.28$ pts / $\sigma$ | $92.4\%$ | $115.4\%$ | $21.9\times$ | Immediate jump with slight mean-reversion simulating wage/revision parsing. |
| **FOMC (Rate Decision)** | $5$ | $-0.40$ pts / $\sigma$ | $86.5\%$ | $107.4\%$ | $22.1\times$ | Large policy rate shift followed by secondary press conference volatility burst. |

---

## 4. Multi-Scale Jordà (2005) Local Projections Demonstration

The pipeline implements Jordà (2005) local projections with Newey-West HAC covariance across both high-frequency minutes and daily panel horizons:

$$\Delta y_{i, t_0+h} = \alpha_h + \beta_h \cdot S_{\text{ann}, i} + \Gamma_h \mathbf{X}_{i, t_0-1\text{m}} + \varepsilon_{i, h}$$

### Table 3: Local Projection Demonstration Across Intraday and Daily Horizons
*Target Variable*: ZN 10-Year Implied Yield Change ($\text{bp}$). Regressor: Standardized Announcement Surprise ($S_{\text{ann}}$).

| Scale | Horizon Label | $\beta_h$ ($\text{bp} / \sigma$) | HAC SE ($\text{bp}$) | $t$-statistic | $p$-value | $95\%$ Confidence Interval | $R^2$ | Regime / Data Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Intraday** | **$1\text{m}$** | $+2.568$ | $0.117$ | $21.88$ | $<0.0001$ | $[+2.338, +2.798]$ | $0.976$ | `SIMULATED_DEMONSTRATION` |
| **Intraday** | **$5\text{m}$** | $+3.658$ | $0.177$ | $20.65$ | $<0.0001$ | $[+3.311, +4.005]$ | $0.974$ | `SIMULATED_DEMONSTRATION` |
| **Intraday** | **$15\text{m}$** | $+4.214$ | $0.217$ | $19.43$ | $<0.0001$ | $[+3.789, +4.639]$ | $0.974$ | `SIMULATED_DEMONSTRATION` |
| **Intraday** | **$30\text{m}$** | $+4.247$ | $0.225$ | $18.89$ | $<0.0001$ | $[+3.807, +4.688]$ | $0.976$ | `SIMULATED_DEMONSTRATION` |
| **Intraday** | **Window End ($+60\text{m}$)** | $+4.444$ | $0.288$ | $15.46$ | $<0.0001$ | $[+3.880, +5.007]$ | $0.971$ | `SIMULATED_DEMONSTRATION` |
| **Daily Panel** | **Day 0** | $+1.139$ | $0.529$ | $2.15$ | $0.0314$ | $[+0.102, +2.176]$ | $0.002$ | `HISTORICAL_CMT_PANEL` |
| **Daily Panel** | **Day 1** | $+1.009$ | $0.646$ | $1.56$ | $0.1180$ | $[-0.257, +2.275]$ | $0.001$ | `HISTORICAL_CMT_PANEL` |
| **Daily Panel** | **Day 2** | $+1.417$ | $0.926$ | $1.53$ | $0.1258$ | $[-0.398, +3.232]$ | $0.002$ | `HISTORICAL_CMT_PANEL` |
| **Daily Panel** | **Day 5** | $+1.317$ | $1.088$ | $1.21$ | $0.2260$ | $[-0.815, +3.449]$ | $0.001$ | `HISTORICAL_CMT_PANEL` |
| **Daily Panel** | **Day 10** | $-0.197$ | $1.391$ | $-0.14$ | $0.8870$ | $[-2.923, +2.529]$ | $0.000$ | `HISTORICAL_CMT_PANEL` |

---

## 5. Frozen Specification for Multi-Regime Empirical Evaluation

To transition this module from an architectural demonstration to an empirical evaluation, the following experimental protocol is frozen:

### 5.1 Required Data Feed & Provenance Verification
1. **Live Exchange Source**: CME Globex MDP 3.0 via Databento (`GLBX.MDP3`), schema `ohlcv-1m` (or tick-level `mbo` for order book depth).
2. **Provenance Integrity**: Each event parquet file must contain certified columns:
   - `data_source`: strictly set to `DATABENTO_LIVE_MDP3`.
   - `dataset_checksum`: SHA-256 hash of raw vendor response.
   - Any unverified legacy cache files without matching checksums must remain labeled `UNKNOWN_UNVERIFIED_CACHE` and excluded from empirical scoring.

### 5.2 Multi-Regime Historical Stratification
The evaluation must cover four distinct macroeconomic regimes to ensure statistical validity across monetary environments:
- **Regime 1: COVID Shock & Zero Lower Bound** (March 2020 – December 2020, $N \ge 8$ events).
- **Regime 2: Post-COVID Inflation Surge & Liftoff Anticipation** (January 2021 – February 2022, $N \ge 12$ events).
- **Regime 3: Aggressive Rate Hiking Shock** (March 2022 – December 2022, $N \ge 12$ events).
- **Regime 4: Plateau, Regional Banking Stress & Easing Cycles** (January 2023 – April 2024, $N \ge 16$ events).

### 5.3 Formal Event Window & Benchmark Settlement Protocol
- **Observation Window**: Exactly $t_0 - 15\text{m}$ to $t_0 + 60\text{m}$.
- **Reference Price ($p_{\text{ref}}$)**: Volume-weighted price or last trade at $t_0 - 1\text{m}$.
- **Benchmark Settlement ($p_{\text{settle}}$)**: The official CME 3:00 PM ET settlement price for the active front-month contract, preventing false convergence claims based on arbitrary window endpoints.
- **Hypothesis Testing**: Formally test $H_0: \text{Share}_{5\text{m}} = 1.0$ against $H_1: \text{Share}_{5\text{m}} < 1.0$ using cluster-robust standard errors across calendar regimes.
- **Pipeline Isolation**: The intraday evaluation must remain strictly isolated from daily term-structure model estimation, preventing any lookahead or parameter leakage into walk-forward trading models.
