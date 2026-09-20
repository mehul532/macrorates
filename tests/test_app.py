"""
Unit tests for Streamlit App (app.py) data loading, curve smoothing, and event logic.
"""

import numpy as np
import pandas as pd
import pytest

from app import (
    load_app_data,
    fit_smooth_curve,
    ICONIC_PRESETS,
    MATURITIES,
    TENOR_COLS,
)
from src.strategy.portfolio import allocate_2s10s_spread, allocate_2s5s10s_butterfly


def test_app_data_loading():
    """Verify load_app_data returns non-empty processed datasets."""
    yield_df, factor_df, macro_df = load_app_data()
    assert len(yield_df) > 1000
    assert len(factor_df) > 1000
    assert len(macro_df) > 500
    assert "DGS10" in yield_df.columns
    assert "kf_slope" in factor_df.columns
    assert "surprise_ann" in macro_df.columns


def test_smooth_curve_fitting():
    """Verify Nelson-Siegel smooth curve interpolates yields smoothly across tenors."""
    sample_yields = np.array([5.2, 5.0, 4.8, 4.5, 4.2, 4.0, 3.9, 3.8, 3.7, 3.9, 4.0])
    dense_taus = np.linspace(0.1, 30.0, 50)
    
    smooth = fit_smooth_curve(sample_yields, MATURITIES, dense_taus)
    assert len(smooth) == 50
    assert not np.isnan(smooth).any()
    # Check bounds reasonable
    assert np.all(smooth > 2.0) and np.all(smooth < 7.0)


def test_iconic_presets_exist_in_data():
    """Verify all curated iconic event dates exist in macro_surprises dataset."""
    _, _, macro_df = load_app_data()
    macro_df["date_str"] = macro_df["date"].dt.strftime("%Y-%m-%d")
    
    for preset_name, info in ICONIC_PRESETS.items():
        sub = macro_df[(macro_df["indicator"] == info["indicator"]) & (macro_df["date_str"] == info["date"])]
        assert len(sub) > 0, f"Preset {preset_name} date {info['date']} not found in data!"


def test_event_trade_dv01_neutrality():
    """Verify event trades triggered by app are confirmed DV01-neutral within 5%."""
    # 2s10s spread
    trade_spread = allocate_2s10s_spread(signal=1.0, target_dv01=10_000.0)
    assert trade_spread["is_dv01_neutral"] is True
    assert trade_spread["residual_pct_of_leg"] < 5.0
    
    # 2s5s10s fly
    trade_fly = allocate_2s5s10s_butterfly(signal=1.0, target_dv01=10_000.0)
    assert trade_fly["is_dv01_neutral"] is True
    assert trade_fly["residual_pct_of_leg"] < 5.0
