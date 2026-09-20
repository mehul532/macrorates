#!/usr/bin/env python3
"""
Root entry point and CLI for Milestone 13: Intraday Treasury Futures Event Study & Local Projections.

Usage:
  python intraday_response.py [--symbol ZN] [--indicator CPI]
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import pandas as pd

from src.futures.intraday_response import (
    CURATED_HIGH_PROFILE_EVENTS,
    DatabentoIntradayProvider,
    EventWindowSummary,
    FrontloadingVsDriftAnalyzer,
    HighProfileEventRegistry,
    HighProfileMacroEvent,
    IntradayEventWindowExtractor,
    IntradayLocalProjectionEngine,
    MultiHorizonIRFComparator,
    run_intraday_macro_event_study,
)


def main(symbol: str = "ZN", indicator: Optional[str] = None):
    print("=" * 80)
    print("  MILESTONE 13: INTRADAY TREASURY FUTURES EVENT STUDY & LOCAL PROJECTIONS")
    print(f"  Target Contract: {symbol} (10-Year Treasury Futures) | Filter Indicator: {indicator or 'All'}")
    print("=" * 80)

    print("\n[1/5] Initializing High-Profile Event Registry...")
    registry = HighProfileEventRegistry()
    events = registry.filter(indicator=indicator)
    print(f"  Loaded {len(events)} curated historical shock events across CPI, NFP, and FOMC.")

    print("\n[2/5] Extracting [-5m, +30m] Globex Intraday Event Windows...")
    study_results = run_intraday_macro_event_study(indicator=indicator, symbol=symbol)
    summaries = study_results["summaries"]
    print(f"  Successfully extracted {len(summaries)} high-frequency event windows.")

    print("\n[3/5] Empirical Front-Loading vs. Drift Decomposition:")
    frontload = study_results["frontload_analysis"]
    print(f"  Mean 1-Minute Response Share:   {frontload['mean_share_1m']*100:.1f}%")
    print(f"  Mean 5-Minute Response Share:   {frontload['mean_share_5m']*100:.1f}%")
    print(f"  Mean 30-Minute Response Share:  {frontload['mean_share_30m']*100:.1f}%")
    print(f"  Mean Volatility Spike Ratio:    {frontload['mean_vol_spike']:.1f}x baseline")
    print(f"  Events with >=70% move at 5m:   {frontload['pct_frontloaded_at_5m']:.1f}%")
    print(f"  Mean Drift (5m -> 30m):         {frontload['mean_drift_5m_to_30m_bp']:+.2f} bp")
    print(f"  Mean Drift (30m -> Close):      {frontload['mean_drift_30m_to_close_bp']:+.2f} bp")

    print("\n  Summary by Indicator:")
    for ind, stats in frontload.get("by_indicator", {}).items():
        print(f"    - {ind} (N={stats['count']}): 5m Share = {stats['mean_share_5m']*100:.1f}%, Vol Spike = {stats['mean_vol_spike']:.1f}x, Frontloaded = {stats['frontloaded_pct']:.0f}%")

    print("\n[4/5] Jordà (2005) Intraday Local Projections (HAC Standard Errors):")
    lp_df = study_results["local_projections"]
    print(lp_df[["horizon", "beta", "hac_se", "t_stat", "p_value", "r_squared", "ci_lower", "ci_upper"]].to_string(index=False))

    print("\n[5/5] Multi-Scale Impulse Response Integration (Intraday -> Multi-Day):")
    unified_df = study_results["unified_irf"]
    print(unified_df.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"  RESEARCH VERDICT: {frontload['conclusion']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Milestone 13 Intraday Event Study")
    parser.add_argument("--symbol", type=str, default="ZN", help="Futures contract symbol (e.g. ZN, ZF, ZT)")
    parser.add_argument("--indicator", type=str, default=None, help="Macro indicator (CPI, NFP, FOMC)")
    args = parser.parse_args()
    main(symbol=args.symbol, indicator=args.indicator)
