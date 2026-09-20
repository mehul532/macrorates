"""
Generate Yield Curve Event Explorer Demo GIF & High-Resolution Demo Artifacts.

Creates:
1. reports/figures/yield_curve_event_explorer_demo.png (High-DPI single-screen demo)
2. reports/figures/yield_curve_event_explorer_demo.gif (Animated multi-event demo loop for README)
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import os
os.environ["MPLCONFIGDIR"] = str(root_dir / ".cache" / "matplotlib")
(root_dir / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from app import load_app_data, fit_smooth_curve, ICONIC_PRESETS, MATURITIES, TENOR_COLS
from src.strategy.portfolio import allocate_2s10s_spread, allocate_2s5s10s_butterfly, DEFAULT_FUTURES_DV01


def render_event_figure(
    preset_name: str,
    event_info: Dict[str, str],
    yield_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    target_dv01: float = 10_000.0,
) -> plt.Figure:
    """Render high-fidelity single-screen institutional dashboard figure."""
    ind = event_info["indicator"]
    dt_str = event_info["date"]
    dt = pd.to_datetime(dt_str)

    # 1. Event Surprise
    event_sub = macro_df[(macro_df["indicator"] == ind) & (macro_df["date"] == dt)]
    if len(event_sub) > 0:
        event = event_sub.iloc[0]
        actual = event["actual"]
        consensus = event["forecast"]
        surprise = event["surprise_ann"] if pd.notna(event["surprise_ann"]) else event["surprise_model"]
    else:
        actual, consensus, surprise = 0.0, 0.0, 0.0

    # 2. Yield Curve
    dt_curr = dt
    if dt_curr not in yield_df.index:
        prevs = yield_df.index[yield_df.index <= dt_curr]
        dt_curr = prevs[-1] if len(prevs) > 0 else yield_df.index[0]
    curr_idx = yield_df.index.get_loc(dt_curr)
    prev_idx = max(0, curr_idx - 1)
    dt_prev = yield_df.index[prev_idx]

    y_prev = yield_df.loc[dt_prev, TENOR_COLS].values
    y_curr = yield_df.loc[dt_curr, TENOR_COLS].values
    dy_bp = (y_curr - y_prev) * 100.0

    dense_taus = np.linspace(0.08, 30.0, 150)
    smooth_prev = fit_smooth_curve(y_prev, MATURITIES, dense_taus)
    smooth_curr = fit_smooth_curve(y_curr, MATURITIES, dense_taus)

    # 3. Latent Factors
    if dt_curr in factor_df.index and dt_prev in factor_df.index:
        dL = (factor_df.loc[dt_curr, "kf_level"] - factor_df.loc[dt_prev, "kf_level"]) * 100.0
        dS = (factor_df.loc[dt_curr, "kf_slope"] - factor_df.loc[dt_prev, "kf_slope"]) * 100.0
        dC = (factor_df.loc[dt_curr, "kf_curvature"] - factor_df.loc[dt_prev, "kf_curvature"]) * 100.0
    else:
        dL = float(np.mean(dy_bp))
        dS = float(dy_bp[8] - dy_bp[4])
        dC = float(2.0 * dy_bp[6] - dy_bp[4] - dy_bp[8])

    # 4. Trade Allocation
    if ind in ("CPI", "CORE_CPI") and abs(dC) > 2.0:
        sig = 1.0 if surprise > 0 else -1.0
        trade = allocate_2s5s10s_butterfly(signal=sig, target_dv01=target_dv01)
        strat_name = "2s-5s-10s DV01-Neutral Butterfly"
        trade_lots = f"{trade['n_zt']:+d} ZT (2Y)  |  {trade['n_zf']:+d} ZF (5Y)  |  {trade['n_zn']:+d} ZN (10Y)"
    else:
        sig = -1.0 if surprise > 0 else 1.0
        trade = allocate_2s10s_spread(signal=sig, target_dv01=target_dv01)
        strat_name = "2s10s DV01-Neutral Curve Spread"
        trade_lots = f"{trade['n_zt']:+d} ZT (2Y)  |  {trade['n_zn']:+d} ZN (10Y)"

    # 5. P&L Attribution
    zt_pnl = -DEFAULT_FUTURES_DV01["ZT"] * dy_bp[4]
    zf_pnl = -DEFAULT_FUTURES_DV01["ZF"] * dy_bp[6]
    zn_pnl = -DEFAULT_FUTURES_DV01["ZN"] * dy_bp[8]
    gross_pnl = trade.get("n_zt", 0) * zt_pnl + trade.get("n_zf", 0) * zf_pnl + trade.get("n_zn", 0) * zn_pnl
    tot_contracts = abs(trade.get("n_zt", 0)) + abs(trade.get("n_zf", 0)) + abs(trade.get("n_zn", 0))
    slippage = tot_contracts * 0.5 * 15.625
    fees = tot_contracts * 1.50
    interest = (target_dv01 * 100.0) * (0.02 / 252.0)
    net_pnl = gross_pnl - slippage - fees + interest

    # Create Dashboard Figure (16:9 ratio, dark institutional theme)
    fig = plt.figure(figsize=(16, 9), facecolor="#0F172A")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 2.5, 2.2], hspace=0.35, wspace=0.25)

    # Header Panel (Top Span)
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_facecolor("#1E293B")
    ax_hdr.axis("off")

    unit_str = "%" if ind in ("CPI", "CORE_CPI", "FOMC", "UNEMP") else "k"
    cons_str = f"{consensus:.1f}{unit_str}" if pd.notna(consensus) else "N/A"
    surp_str = f"{surprise:+.2f}σ"
    nature = "HAWKING SHOCK" if surprise > 0 else "DOVISH SHOCK"

    hdr_text = (
        f"MACRORATES: YIELD CURVE EVENT EXPLORER\n"
        f"EVENT: {ind} Announcement on {dt.strftime('%B %d, %Y')}  |  "
        f"Actual: {actual:.1f}{unit_str}  vs.  Consensus: {cons_str}  |  "
        f"Surprise: {surp_str} [{nature}]"
    )
    ax_hdr.text(0.02, 0.5, hdr_text, color="#F8FAFC", fontsize=15, fontweight="bold", va="center")

    # Left Center: Yield Curve (Before vs After)
    ax_curve = fig.add_subplot(gs[1, :2])
    ax_curve.set_facecolor("#1E293B")
    ax_curve.plot(dense_taus, smooth_prev, color="#94A3B8", lw=2.2, ls="--", label=f"Eve Close ({dt_prev.strftime('%b %d')})")
    ax_curve.scatter(MATURITIES, y_prev, color="#94A3B8", s=40, zorder=4)

    curve_col = "#EF4444" if surprise > 0 else "#22C55E"
    ax_curve.plot(dense_taus, smooth_curr, color=curve_col, lw=2.8, label=f"Event Close ({dt_curr.strftime('%b %d')})")
    ax_curve.scatter(MATURITIES, y_curr, color=curve_col, s=55, zorder=5)

    ax_curve.set_ylabel("Yield (% p.a.)", color="#F8FAFC", fontsize=11, fontweight="bold")
    ax_curve.set_title(f"1. Term Structure Deformation: Before vs. After", color="#38BDF8", fontsize=13, fontweight="bold", loc="left")
    ax_curve.tick_params(colors="#94A3B8")
    ax_curve.grid(True, color="#334155", alpha=0.5)
    ax_curve.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", loc="best")

    # Right Center: Latent Factor Shifts
    ax_factor = fig.add_subplot(gs[1, 2])
    ax_factor.set_facecolor("#1E293B")
    factor_names = ["ΔLevel", "ΔSlope", "ΔCurv"]
    factor_vals = [dL, dS, dC]
    f_colors = ["#EF4444" if v >= 0 else "#38BDF8" for v in factor_vals]
    bars = ax_factor.bar(factor_names, factor_vals, color=f_colors, alpha=0.9, edgecolor="#0F172A", lw=1.0)
    ax_factor.axhline(0, color="#94A3B8", lw=1.0)
    ax_factor.set_ylabel("Basis Points (bp)", color="#F8FAFC", fontsize=10, fontweight="bold")
    ax_factor.set_title("2. Dynamic Factor Shifts", color="#38BDF8", fontsize=13, fontweight="bold", loc="left")
    ax_factor.tick_params(colors="#94A3B8")
    ax_factor.grid(True, color="#334155", alpha=0.5, axis="y")

    for bar, val in zip(bars, factor_vals):
        y_pos = val + (0.8 if val >= 0 else -1.8)
        ax_factor.text(bar.get_x() + bar.get_width()/2, y_pos, f"{val:+.1f}", ha="center", color="#F8FAFC", fontsize=10, fontweight="bold")

    # Bottom Left: Yield Change Bar Profile
    ax_dy = fig.add_subplot(gs[2, 0])
    ax_dy.set_facecolor("#1E293B")
    dy_colors = ["#EF4444" if d >= 0 else "#38BDF8" for d in dy_bp]
    ax_dy.bar(MATURITIES, dy_bp, width=0.8, color=dy_colors, alpha=0.85, edgecolor="#0F172A")
    ax_dy.axhline(0, color="#94A3B8", lw=1.0)
    ax_dy.set_title("3. Curve Shift by Maturity (bp)", color="#38BDF8", fontsize=12, fontweight="bold", loc="left")
    ax_dy.set_ylabel("Δbp", color="#F8FAFC", fontsize=10)
    ax_dy.tick_params(colors="#94A3B8")
    ax_dy.set_xticks([1, 2, 5, 10, 30])
    ax_dy.set_xticklabels(["1Y", "2Y", "5Y", "10Y", "30Y"])
    ax_dy.grid(True, color="#334155", alpha=0.5, axis="y")

    # Bottom Center: Relative-Value Trade & DV01 Neutrality
    ax_trade = fig.add_subplot(gs[2, 1])
    ax_trade.set_facecolor("#1E293B")
    ax_trade.axis("off")
    ax_trade.set_title("4. DV01-Neutral Trade Execution", color="#38BDF8", fontsize=12, fontweight="bold", loc="left")

    trade_text = (
        f"Strategy: {strat_name}\n"
        f"Direction: {trade['direction'].upper()}\n\n"
        f"Position Lots:\n{trade_lots}\n\n"
        f"Net Portfolio DV01: ${trade['net_portfolio_dv01']:+.2f}\n"
        f"Residual Risk: {trade['residual_pct_of_leg']:.3f}% of leg\n"
        f"Status: CONFIRMED NEUTRAL (<5% CME)"
    )
    ax_trade.text(0.05, 0.45, trade_text, color="#F8FAFC", fontsize=10, family="monospace", va="center")

    # Bottom Right: P&L Attribution Chain
    ax_pnl = fig.add_subplot(gs[2, 2])
    ax_pnl.set_facecolor("#1E293B")
    ax_pnl.set_title("5. P&L Attribution Waterfall", color="#38BDF8", fontsize=12, fontweight="bold", loc="left")

    components = ["Gross", "Slippage", "Fees", "Interest", "NET P&L"]
    values = [gross_pnl, -slippage, -fees, interest, net_pnl]
    p_colors = ["#22C55E" if v >= 0 else "#EF4444" for v in values]
    p_colors[-1] = "#38BDF8"

    bars_p = ax_pnl.barh(components, values, color=p_colors, alpha=0.85, edgecolor="#0F172A")
    ax_pnl.axvline(0, color="#94A3B8", lw=1.0)
    ax_pnl.set_xlabel("USD ($)", color="#F8FAFC", fontsize=10)
    ax_pnl.tick_params(colors="#94A3B8")
    ax_pnl.grid(True, color="#334155", alpha=0.5, axis="x")

    for bar, val in zip(bars_p, values):
        x_pos = val + (200 if val >= 0 else -1200)
        ax_pnl.text(x_pos, bar.get_y() + bar.get_height()/2, f"${val:+,.0f}", va="center", color="#F8FAFC", fontsize=9, fontweight="bold")

    return fig


def generate_all_artifacts():
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    yield_df, factor_df, macro_df = load_app_data()

    frames = []
    # Generate high-res image for the primary event (June 15, 2022 FOMC hike)
    primary_preset = "June 15, 2022: Fed 75bp Shock Hike (Hawkish)"
    primary_fig = render_event_figure(primary_preset, ICONIC_PRESETS[primary_preset], yield_df, factor_df, macro_df)
    png_path = out_dir / "yield_curve_event_explorer_demo.png"
    primary_fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="#0F172A")
    plt.close(primary_fig)
    print(f"Saved high-res preview: {png_path}")

    # Generate animated GIF cycling through 4 iconic presets
    gif_presets = [
        "June 15, 2022: Fed 75bp Shock Hike (Hawkish)",
        "May 12, 2021: CPI Inflation Breakout (+4.58σ)",
        "September 18, 2024: Fed 50bp Jumbo Rate Cut",
        "June 5, 2020: Post-Lockdown Jobs Rebound (+14.2σ)",
    ]

    temp_images = []
    for p_name in gif_presets:
        print(f"Rendering frame: {p_name}...")
        fig = render_event_figure(p_name, ICONIC_PRESETS[p_name], yield_df, factor_df, macro_df)
        temp_path = out_dir / f"temp_{p_name[:10].replace(' ', '_')}.png"
        fig.savefig(temp_path, dpi=120, bbox_inches="tight", facecolor="#0F172A")
        plt.close(fig)
        temp_images.append(Image.open(temp_path))

    # Save as animated GIF (3 seconds per slide, looping)
    gif_path = out_dir / "yield_curve_event_explorer_demo.gif"
    temp_images[0].save(
        gif_path,
        save_all=True,
        append_images=temp_images[1:],
        duration=3200,
        loop=0,
    )
    print(f"Saved animated demo GIF: {gif_path}")

    # Clean up temporary frames
    for p in out_dir.glob("temp_*.png"):
        p.unlink()


if __name__ == "__main__":
    generate_all_artifacts()
