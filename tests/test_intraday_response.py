"""
Unit tests for Milestone 13: Intraday Treasury Futures Event Study & Local Projections.

Covers:
- HighProfileEventRegistry curation and filtering
- Databento provider and 1-minute Globex bar structure
- Event window extraction in [-5m, +30m]
- Implied yield conversion via DV01 and contract specifications
- Realized volatility calculation and post-announcement spike
- Front-loading vs. afternoon drift analyzer
- Jordà (2005) Intraday Local Projections estimation with HAC standard errors
- Multi-horizon IRF panel integration
- End-to-end event study orchestrator
"""

import numpy as np
import pandas as pd
import pytest

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


def test_high_profile_event_registry_curation():
    """Verify curated registry contents, filtering, and schema."""
    registry = HighProfileEventRegistry()
    assert len(registry.events) >= 15

    # Check that all 3 major indicators are represented
    indicators = {e.indicator for e in registry.events}
    assert {"CPI", "NFP", "FOMC"}.issubset(indicators)

    # Filter by indicator
    cpi_events = registry.filter(indicator="CPI")
    assert len(cpi_events) >= 5
    assert all(e.indicator == "CPI" for e in cpi_events)

    # Filter by surprise magnitude
    high_shocks = registry.filter(min_abs_surprise=2.0)
    assert len(high_shocks) >= 5
    assert all(abs(e.surprise_ann) >= 2.0 for e in high_shocks)

    # To DataFrame conversion
    df = registry.to_dataframe()
    expected_cols = {
        "indicator", "date", "timestamp", "actual", "consensus",
        "raw_surprise", "surprise_ann", "headline", "regime"
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) == len(registry.events)


def test_price_to_implied_yield_conversion():
    """Verify mathematical consistency of price delta to implied yield delta conversion."""
    extractor = IntradayEventWindowExtractor()

    # ZN: point_value = $1,000, DV01 = $75
    # If price drops by 0.75 points ($750 drop), yield should RISE by:
    # - (-0.75 * 1000) / 75 = +10.0 bp
    dy_rise = extractor.price_to_implied_yield_delta(delta_price=-0.75, symbol="ZN", contract_dv01=75.0)
    assert abs(dy_rise - 10.0) < 1e-4

    # If price rises by 1.50 points ($1500 gain), yield should DROP by:
    # - (1.50 * 1000) / 75 = -20.0 bp
    dy_drop = extractor.price_to_implied_yield_delta(delta_price=1.50, symbol="ZN", contract_dv01=75.0)
    assert abs(dy_drop - (-20.0)) < 1e-4

    # Zero price change gives zero yield change
    assert extractor.price_to_implied_yield_delta(delta_price=0.0, symbol="ZN") == 0.0


def test_intraday_bar_generation_and_window_extraction(tmp_path):
    """Verify 1-minute Globex bars and [-5m, +30m] window extraction."""
    provider = DatabentoIntradayProvider(cache_dir=tmp_path)
    extractor = IntradayEventWindowExtractor(provider=provider)

    sample_event = HighProfileMacroEvent(
        indicator="CPI",
        date="2022-06-10",
        timestamp="2022-06-10T12:30:00Z",
        actual=8.6,
        consensus=8.3,
        surprise_ann=2.288,
        headline="June 2022 CPI Shock",
        regime="2022 Hiking Shock",
    )

    summary = extractor.extract_window(event=sample_event, symbol="ZN")
    assert isinstance(summary, EventWindowSummary)
    assert summary.event.date == "2022-06-10"
    assert summary.symbol == "ZN"

    # Window DataFrame spans -5m to +30m (36 one-minute bars)
    w_df = summary.window_df
    assert len(w_df) == 36
    assert w_df["minute_offset"].min() == -5
    assert w_df["minute_offset"].max() == 30

    # Hot CPI surprise (positive surprise) should cause price drop and yield spike
    assert summary.delta_p_5m < 0.0
    assert summary.delta_y_5m_bp > 0.0

    # Realized volatility shock at t=0 to 5m must exceed pre-announcement baseline
    assert summary.rv_0_5m_bp > summary.rv_pre_bp
    assert summary.vol_spike_ratio > 1.5


def test_frontloading_vs_drift_analyzer():
    """Verify front-loading evaluation and share metrics."""
    # Synthesize two mock summaries: one heavily frontloaded, one moderate
    reg = HighProfileEventRegistry()
    events = reg.filter(indicator="CPI")[:3]
    extractor = IntradayEventWindowExtractor()

    summaries = [extractor.extract_window(e, symbol="ZN") for e in events]
    result = FrontloadingVsDriftAnalyzer.analyze_event_sample(summaries)

    assert "n_events" in result
    assert result["n_events"] == len(summaries)
    assert "mean_share_1m" in result
    assert "mean_share_5m" in result
    assert "mean_share_30m" in result
    assert "mean_vol_spike" in result
    assert "pct_frontloaded_at_5m" in result
    assert "conclusion" in result

    # 5m share should be substantial (>50%)
    assert result["mean_share_5m"] > 0.50
    assert result["mean_vol_spike"] > 2.0


def test_jorda_intraday_local_projections():
    """Verify estimation of Jordà (2005) intraday local projections with HAC SEs."""
    reg = HighProfileEventRegistry()
    events = reg.events[:12]
    extractor = IntradayEventWindowExtractor()
    summaries = [extractor.extract_window(e, symbol="ZN") for e in events]

    lp_df = IntradayLocalProjectionEngine.estimate_projections(summaries)
    assert not lp_df.empty
    expected_horizons = ["1m", "5m", "15m", "30m", "close"]
    assert list(lp_df["horizon"]) == expected_horizons

    for _, row in lp_df.iterrows():
        # Positive macro surprise raises yields -> beta must be positive
        assert row["beta"] > 0.0
        assert row["hac_se"] > 0.0
        assert row["ci_lower"] < row["beta"] < row["ci_upper"]
        assert row["n_obs"] == len(summaries)
        assert 0.0 <= row["r_squared"] <= 1.0


def test_multi_horizon_irf_comparator():
    """Verify unified multi-scale IRF panel bridging intraday and daily horizons."""
    reg = HighProfileEventRegistry()
    summaries = [IntradayEventWindowExtractor().extract_window(e, symbol="ZN") for e in reg.events[:8]]
    lp_df = IntradayLocalProjectionEngine.estimate_projections(summaries)

    unified = MultiHorizonIRFComparator.create_unified_irf_panel(lp_df)
    assert not unified.empty
    scales = set(unified["scale"])
    assert "intraday" in scales
    assert "daily" in scales

    # Confirm key columns present
    for col in ["scale", "horizon_label", "beta", "hac_se", "t_stat"]:
        assert col in unified.columns


def test_end_to_end_orchestrator():
    """Verify end-to-end execution of run_intraday_macro_event_study."""
    results = run_intraday_macro_event_study(indicator="CPI", symbol="ZN")
    assert results["symbol"] == "ZN"
    assert results["n_events"] >= 5
    assert len(results["summaries"]) == results["n_events"]
    assert not results["local_projections"].empty
    assert not results["unified_irf"].empty
    assert "conclusion" in results["frontload_analysis"]
