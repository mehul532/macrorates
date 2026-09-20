"""Script to build and validate notebooks/03_treasury_futures_analytics_and_dv01.ipynb."""

from pathlib import Path
import nbformat as nbf


def create_futures_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Overview
    cells.append(nbf.v4.new_markdown_cell("""# Treasury Futures Analytics, Conversion Factors & DV01 Neutrality

**MacroRates: Systematic Treasury Relative Value Research**  
*Milestone 5 Diagnostic: CME Globex Contracts (ZT, ZF, ZN, TN, UB)*

---

### Purpose & Research Scope
Before trading or backtesting systematic Treasury relative-value strategies, this notebook establishes institutional-grade pricing and risk analytics for the five primary CME Treasury futures contracts:
1. **ZT**: 2-Year U.S. Treasury Note Futures ($200,000 notional)
2. **ZF**: 5-Year U.S. Treasury Note Futures ($100,000 notional)
3. **ZN**: 10-Year U.S. Treasury Note Futures ($100,000 notional)
4. **TN**: Ultra 10-Year U.S. Treasury Note Futures ($100,000 notional)
5. **UB**: Ultra U.S. Treasury Bond Futures ($100,000 notional)

We validate:
- **Contract Specifications**: Tick size, tick value, notional, deliverable baskets, and quarterly cycles (H, M, U, Z).
- **CME 6% Conversion Factor (CF)**: The standard discount pricing formula.
- **Cheapest-to-Deliver (CTD) Selection**: Gross/net basis and Implied Repo Rate (IRR).
- **Duration & Dollar DV01**: Evaluating $\\mathrm{DV01}_{\\mathrm{contract}} = \\mathrm{DV01}_{\\mathrm{CTD}} / \\mathrm{CF}_{\\mathrm{CTD}}$.
- **DV01-Neutral Spreads & Butterflies**: Integer contract weighting asserting that net portfolio DV01 is strictly within $<5\\%$ of a single leg.
- **Continuous Contract Construction**: Panama Canal backward additive adjustment vs. ratio adjustment.
- **Databento Integration**: Live quoting for CME Globex MDP 3.0 (`GLBX.MDP3`) across schemas and date ranges.
"""))

    # Imports & Setup
    cells.append(nbf.v4.new_code_cell("""import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.futures.futures_analytics import (
    ContractSpec,
    DatabentoFuturesClient,
    DeliverableBond,
    TREASURY_FUTURES_SPECS,
    build_continuous_contract,
    calculate_bond_duration_and_dv01,
    calculate_contract_dv01_and_duration,
    calculate_conversion_factor,
    construct_dv01_neutral_butterfly,
    construct_dv01_neutral_spread,
    get_contract_roll_dates,
    identify_cheapest_to_deliver,
)

# Set visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 120

print("Treasury Futures Analytics Library initialized.")
"""))

    # Section 1: Contract Specs Matrix
    cells.append(nbf.v4.new_markdown_cell("""## 1. CME Treasury Futures Contract Specifications

Treasury futures are standardized by the Chicago Board of Trade (CBOT/CME). Below is the official specifications matrix across all 5 tenors:
"""))

    cells.append(nbf.v4.new_code_cell("""specs_data = []
for sym, spec in TREASURY_FUTURES_SPECS.items():
    specs_data.append({
        "Contract": sym,
        "Floor Ticker": spec.floor_symbol,
        "Name": spec.name,
        "Notional ($)": f"${spec.notional:,.0f}",
        "Tick Size (pts)": f"{spec.tick_size:.6f}",
        "Tick Value ($)": f"${spec.tick_value:.4f}",
        "Point Value ($)": f"${spec.point_value:,.0f}",
        "Deliverable Basket (Years)": f"{spec.deliverable_min_years:.2f} to {spec.deliverable_max_years:.2f}",
        "Delivery Months": ", ".join(spec.delivery_months),
    })

specs_df = pd.DataFrame(specs_data)
specs_df
"""))

    # Section 2: Conversion Factor Properties
    cells.append(nbf.v4.new_markdown_cell("""## 2. CME 6% Conversion Factor (CF) & Theoretical Properties

The conversion factor equals the clean price of a deliverable Treasury bond with coupon $c$ and remaining maturity $T$ discounted at a flat $6.0\\%$ semiannual yield per $\\$1.0$ par value on the first day of the contract delivery month:
$$CF = a \\cdot \\left[ \\frac{c}{2} + \\frac{c}{0.06}(1 - c') + c' \\right] - b$$

### Key Mathematical Axioms:
1. **6.0% Benchmark Invariance**: Any bond with coupon $c = 6.0\\%$ has $CF = 1.0000$ exactly, regardless of maturity.
2. **Coupon Sensitivity**: Bonds with $c > 6.0\\%$ have $CF > 1.0$; bonds with $c < 6.0\\%$ have $CF < 1.0$.
3. **Maturity Convexity**: At yields $> 6\\%$, higher duration bonds have lower conversion factors.

Below, we compute and plot the Conversion Factor surface across coupons and maturities:
"""))

    cells.append(nbf.v4.new_code_cell("""deliv_date = "2024-06-01"
maturities_yrs = np.linspace(2, 30, 29)
coupons = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

cf_curves = {c: [] for c in coupons}
for c in coupons:
    for m in maturities_yrs:
        mat_dt = pd.to_datetime(deliv_date) + pd.DateOffset(years=int(m), months=int((m % 1)*12))
        cf = calculate_conversion_factor(c, mat_dt, deliv_date, contract_symbol="ZN")
        cf_curves[c].append(cf)

fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
for c in coupons:
    style = "--" if c == 0.06 else "-"
    lw = 2.4 if c == 0.06 else 1.5
    label = f"Coupon {c*100:.0f}%" + (" (Par Baseline)" if c == 0.06 else "")
    ax.plot(maturities_yrs, cf_curves[c], label=label, linestyle=style, linewidth=lw)

ax.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
ax.set_title("CME Treasury Futures Conversion Factors vs. Maturity & Coupon", fontsize=12, fontweight="bold")
ax.set_xlabel("Remaining Maturity at Delivery (Years)", fontsize=10)
ax.set_ylabel("Conversion Factor", fontsize=10)
ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.show()
"""))

    # Section 3: CTD Identification
    cells.append(nbf.v4.new_markdown_cell("""## 3. Cheapest-to-Deliver (CTD) Bond Identification

The seller of a Treasury futures contract holds the delivery option and delivers the bond that minimizes cost.
- **Gross Basis**: $\\mathrm{Basis}_i = P_i - (F \\times CF_i)$
- **Net Basis**: $\\mathrm{NetBasis}_i = P_i + \\mathrm{Carry}_i - (F \\times CF_i)$
- **Implied Repo Rate (IRR)**: The annualized return from buying the deliverable bond and selling the futures contract.

The bond with the **minimum gross basis** (or highest IRR) is the **Cheapest-to-Deliver (CTD)**.
"""))

    cells.append(nbf.v4.new_code_cell("""deliv_date = "2024-06-01"
futures_price = 108.50  # 10Y Note futures price

# Hypothetical deliverable 10Y basket
basket = [
    DeliverableBond("91282CDU1", 0.03875, "2033-08-15", 96.25, 0.0435),
    DeliverableBond("91282CDP2", 0.04125, "2032-11-15", 98.10, 0.0438),
    DeliverableBond("91282CDH0", 0.04500, "2034-05-15", 101.40, 0.0432),
    DeliverableBond("91282CDZ8", 0.03500, "2031-02-15", 94.80, 0.0441),
]

ctd_bond, ctd_cf, ctd_table = identify_cheapest_to_deliver(
    futures_price=futures_price,
    deliverable_bonds=basket,
    delivery_date=deliv_date,
    contract_symbol="ZN",
    repo_rate=0.0525,
)

print(f"Identified CTD Bond: CUSIP {ctd_bond.cusip}, Coupon {ctd_bond.coupon*100:.3f}%, Maturity {ctd_bond.maturity_date}")
print(f"CTD Conversion Factor: {ctd_cf:.4f}, Gross Basis: ${ctd_table.iloc[0]['gross_basis']:.4f}")
ctd_table
"""))

    # Section 4: Contract DV01 & Duration
    cells.append(nbf.v4.new_markdown_cell("""## 4. Contract DV01, Duration & DV01-Neutral Spreads

Because 1 futures contract controls $\\frac{\\mathrm{Notional}}{\\mathrm{CF}_{\\mathrm{CTD}}}$ of the CTD cash security:
$$\\mathrm{DV01}_{\\mathrm{contract}} = \\frac{\\mathrm{DV01}_{\\mathrm{CTD}}}{\\mathrm{CF}_{\\mathrm{CTD}}}$$
$$\\mathrm{Duration}_{\\mathrm{contract}} \\approx \\mathrm{ModifiedDuration}_{\\mathrm{CTD}}$$

Below, we compute reference DV01s across the term structure:
"""))

    cells.append(nbf.v4.new_code_cell("""# Benchmark CTD bonds for each contract
ctd_candidates = {
    "ZT": DeliverableBond("91282CEX0", 0.045, "2026-05-15", 99.50, 0.0470),
    "ZF": DeliverableBond("91282CDQ0", 0.040, "2029-05-15", 98.80, 0.0425),
    "ZN": DeliverableBond("91282CDU1", 0.040, "2034-05-15", 98.00, 0.0425),
    "TN": DeliverableBond("91282CDW7", 0.0425, "2034-02-15", 100.20, 0.0420),
    "UB": DeliverableBond("912810TT3", 0.0425, "2054-05-15", 96.50, 0.0445),
}

dv01_records = []
for sym, spec in TREASURY_FUTURES_SPECS.items():
    bond = ctd_candidates[sym]
    cf = calculate_conversion_factor(bond.coupon, bond.maturity_date, "2024-06-01", sym)
    c_dv01, c_dur = calculate_contract_dv01_and_duration(bond, cf, spec, as_of_date="2024-06-01")
    dv01_records.append({
        "Contract": sym,
        "Name": spec.name,
        "Notional ($)": f"${spec.notional:,.0f}",
        "CTD CUSIP": bond.cusip,
        "CTD Coupon": f"{bond.coupon*100:.2f}%",
        "CF": cf,
        "Contract Duration (Years)": f"{c_dur:.2f}",
        "Contract DV01 ($/bp)": f"${c_dv01:.2f}",
    })

dv01_df = pd.DataFrame(dv01_records)
dv01_df
"""))

    # Section 5: DV01 Neutrality Testing
    cells.append(nbf.v4.new_markdown_cell("""## 5. Constructing DV01-Neutral Spreads & Portfolio Tolerance Testing

For relative-value trading, curve positions must be **DV01-neutral** so that parallel yield shifts do not generate net directional P&L.
- **2s10s Steepener/Flattener** (ZT vs. ZN):
  $$N_{\\mathrm{ZT}} \\cdot \\mathrm{DV01}_{\\mathrm{ZT}} + N_{\\mathrm{ZN}} \\cdot \\mathrm{DV01}_{\\mathrm{ZN}} \\approx 0$$
- **5s30s Steepener/Flattener** (ZF vs. UB):
  $$N_{\\mathrm{ZF}} \\cdot \\mathrm{DV01}_{\\mathrm{ZF}} + N_{\\mathrm{UB}} \\cdot \\mathrm{DV01}_{\\mathrm{UB}} \\approx 0$$
- **2s5s10s Butterfly** (ZT - ZF - ZN):
  $$N_{\\mathrm{ZT}} \\cdot \\mathrm{DV01}_{\\mathrm{ZT}} + N_{\\mathrm{ZF}} \\cdot \\mathrm{DV01}_{\\mathrm{ZF}} + N_{\\mathrm{ZN}} \\cdot \\mathrm{DV01}_{\\mathrm{ZN}} \\approx 0$$

### Strict Risk Requirement:
The net portfolio DV01 must be strictly within **$<5\\%$ of a single leg's DV01**.
"""))

    cells.append(nbf.v4.new_code_cell("""dv01_zt = 39.50
dv01_zf = 45.20
dv01_zn = 72.80
dv01_ub = 215.00

# 1. 2s10s Curve Spread
s_2s10s = construct_dv01_neutral_spread("ZT", dv01_zt, "ZN", dv01_zn, target_leg1_contracts=100)

# 2. 5s30s Curve Spread
s_5s30s = construct_dv01_neutral_spread("ZF", dv01_zf, "UB", dv01_ub, target_leg1_contracts=100)

# 3. 2s5s10s Butterfly
s_fly = construct_dv01_neutral_butterfly("ZT", dv01_zt, "ZF", dv01_zf, "ZN", dv01_zn, belly_contracts=-100)

spread_table = pd.DataFrame([
    {
        "Structure": "2s10s Curve Spread",
        "Leg 1": f"+{s_2s10s['n1']} ZT (${s_2s10s['leg1_total_dv01']:,.2f})",
        "Leg 2": f"{s_2s10s['n2']} ZN (${s_2s10s['leg2_total_dv01']:,.2f})",
        "Leg 3": "N/A",
        "Net Portfolio DV01 ($)": f"${s_2s10s['net_portfolio_dv01']:.2f}",
        "Tolerance Threshold ($)": f"${s_2s10s['tolerance_threshold']:.2f}",
        "Residual % of Leg": f"{s_2s10s['residual_pct_of_leg']:.3f}%",
        "Passes <5% Check?": "PASS" if s_2s10s["is_dv01_neutral"] else "FAIL",
    },
    {
        "Structure": "5s30s Curve Spread",
        "Leg 1": f"+{s_5s30s['n1']} ZF (${s_5s30s['leg1_total_dv01']:,.2f})",
        "Leg 2": f"{s_5s30s['n2']} UB (${s_5s30s['leg2_total_dv01']:,.2f})",
        "Leg 3": "N/A",
        "Net Portfolio DV01 ($)": f"${s_5s30s['net_portfolio_dv01']:.2f}",
        "Tolerance Threshold ($)": f"${s_5s30s['tolerance_threshold']:.2f}",
        "Residual % of Leg": f"{s_5s30s['residual_pct_of_leg']:.3f}%",
        "Passes <5% Check?": "PASS" if s_5s30s["is_dv01_neutral"] else "FAIL",
    },
    {
        "Structure": "2s5s10s Butterfly",
        "Leg 1": f"+{s_fly['n1']} ZT",
        "Leg 2 (Belly)": f"{s_fly['n_belly']} ZF",
        "Leg 3": f"+{s_fly['n3']} ZN",
        "Net Portfolio DV01 ($)": f"${s_fly['net_portfolio_dv01']:.2f}",
        "Tolerance Threshold ($)": f"${s_fly['tolerance_threshold']:.2f}",
        "Residual % of Leg": f"{s_fly['residual_pct_of_leg']:.3f}%",
        "Passes <5% Check?": "PASS" if s_fly["is_dv01_neutral"] else "FAIL",
    },
])

spread_table
"""))

    # Section 6: Roll Schedule & Continuous Stitched Contracts
    cells.append(nbf.v4.new_markdown_cell("""## 6. Roll Schedules & Continuous Contract Construction

To eliminate artificial price gaps at quarterly contract expirations, we evaluate:
- **Panama Canal (Backward Additive Adjustment)**:
  $$P_t^{\\mathrm{adj}} = P_t + \\sum_{k \\ge t} \\Delta_k, \\quad \\Delta_k = P_{k, \\mathrm{new}} - P_{k, \\mathrm{old}}$$
  Preserves point changes $\\Delta P_t$ and exact dollar tick P&L.
- **Volume Roll Date**: Systematically executed **7 to 8 business days prior to First Notice Day** (mid-Feb, mid-May, mid-Aug, mid-Nov).
"""))

    cells.append(nbf.v4.new_code_cell("""# 2024 Roll Schedule
roll_schedule_2024 = []
for code in ["H", "M", "U", "Z"]:
    r = get_contract_roll_dates(2024, code)
    roll_schedule_2024.append({
        "Contract": r["contract"],
        "Delivery Month": r["delivery_month"],
        "Volume Roll Date": r["volume_roll_date"].strftime("%Y-%m-%d"),
        "First Notice Day (FND)": r["first_notice_day"].strftime("%Y-%m-%d"),
        "Last Trade Date (LTD)": r["last_trade_date"].strftime("%Y-%m-%d"),
    })

roll_df = pd.DataFrame(roll_schedule_2024)
roll_df
"""))

    # Section 7: Databento Quoting
    cells.append(nbf.v4.new_markdown_cell("""## 7. Databento Integration & CME Globex Live Cost Quoting

Historical market data pricing varies dynamically by schema and date range. We query Databento's `GLBX.MDP3` dataset via `client.metadata.get_cost()`:
"""))

    cells.append(nbf.v4.new_code_cell("""db_client = DatabentoFuturesClient(api_key=None)

schemas_to_test = ["ohlcv-1d", "ohlcv-1h", "trades", "mbp-10"]
quote_records = []

for sch in schemas_to_test:
    q = db_client.get_live_cost_quote(
        dataset="GLBX.MDP3",
        symbols=["ZT.FUT", "ZF.FUT", "ZN.FUT", "TN.FUT", "UB.FUT"],
        schema=sch,
        start="2024-01-01",
        end="2024-06-30",
    )
    quote_records.append({
        "Schema": sch,
        "Symbols": "5 Tenors (ZT, ZF, ZN, TN, UB)",
        "Date Range": f"{q['start']} to {q['end']}",
        "Estimated Billable MB": f"{q['billable_mb']:,.2f} MB",
        "Estimated Cost (USD)": f"${q['estimated_cost_usd']:.2f}",
        "Pricing Model": q["source"],
    })

quote_df = pd.DataFrame(quote_records)
quote_df
"""))

    out_path = Path("notebooks/03_treasury_futures_analytics_and_dv01.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated notebook: {out_path}")


if __name__ == "__main__":
    create_futures_notebook()
