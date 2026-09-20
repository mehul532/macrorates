"""Unit tests for Treasury futures analytics, conversion factors, CTD, DV01 neutrality, and Databento quoting."""

import numpy as np
import pandas as pd
import pytest

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


def test_contract_specs_integrity():
    """Verify institutional specifications for all 5 CME Treasury futures contracts."""
    assert set(TREASURY_FUTURES_SPECS.keys()) == {"ZT", "ZF", "ZN", "TN", "UB"}

    # 1. 2-Year (ZT): $200k notional, 1/256 tick size, $7.8125 tick value
    zt = TREASURY_FUTURES_SPECS["ZT"]
    assert zt.notional == 200_000.0
    assert abs(zt.tick_size - 1.0 / 256.0) < 1e-9
    assert abs(zt.tick_value - 7.8125) < 1e-4
    assert zt.point_value == 2_000.0
    assert abs(zt.tick_size * zt.point_value - zt.tick_value) < 1e-4

    # 2. 5-Year (ZF): $100k notional, 1/128 tick size, $7.8125 tick value
    zf = TREASURY_FUTURES_SPECS["ZF"]
    assert zf.notional == 100_000.0
    assert abs(zf.tick_size - 1.0 / 128.0) < 1e-9
    assert abs(zf.tick_value - 7.8125) < 1e-4
    assert zf.point_value == 1_000.0
    assert abs(zf.tick_size * zf.point_value - zf.tick_value) < 1e-4

    # 3. 10-Year (ZN): $100k notional, 1/64 tick size, $15.625 tick value
    zn = TREASURY_FUTURES_SPECS["ZN"]
    assert zn.notional == 100_000.0
    assert abs(zn.tick_size - 1.0 / 64.0) < 1e-9
    assert abs(zn.tick_value - 15.625) < 1e-4
    assert zn.point_value == 1_000.0
    assert abs(zn.tick_size * zn.point_value - zn.tick_value) < 1e-4

    # 4. Ultra 10-Year (TN): $100k notional, 1/64 tick size, $15.625 tick value
    tn = TREASURY_FUTURES_SPECS["TN"]
    assert tn.notional == 100_000.0
    assert abs(tn.tick_size - 1.0 / 64.0) < 1e-9
    assert abs(tn.tick_value - 15.625) < 1e-4
    assert tn.point_value == 1_000.0

    # 5. Ultra Bond (UB): $100k notional, 1/32 tick size, $31.25 tick value
    ub = TREASURY_FUTURES_SPECS["UB"]
    assert ub.notional == 100_000.0
    assert abs(ub.tick_size - 1.0 / 32.0) < 1e-9
    assert abs(ub.tick_value - 31.25) < 1e-4
    assert ub.point_value == 1_000.0
    assert abs(ub.tick_size * ub.point_value - ub.tick_value) < 1e-4


def test_conversion_factor_calculation():
    """Verify CME 6% conversion factor formula against benchmark reference values."""
    deliv_date = "2024-03-01"

    # Property 1: Any 6.0% coupon bond must produce CF = 1.0000 regardless of maturity
    cf_6pct_10y = calculate_conversion_factor(0.06, "2034-03-01", deliv_date, "ZN")
    cf_6pct_7y = calculate_conversion_factor(0.06, "2031-03-01", deliv_date, "ZN")
    cf_6pct_2y = calculate_conversion_factor(0.06, "2026-03-01", deliv_date, "ZT")
    cf_6pct_30y = calculate_conversion_factor(0.06, "2054-03-01", deliv_date, "UB")

    assert abs(cf_6pct_10y - 1.0000) <= 0.0001
    assert abs(cf_6pct_7y - 1.0000) <= 0.0001
    assert abs(cf_6pct_2y - 1.0000) <= 0.0001
    assert abs(cf_6pct_30y - 1.0000) <= 0.0001

    # Property 2: Coupons < 6% must have CF < 1.0, coupons > 6% must have CF > 1.0
    cf_4pct = calculate_conversion_factor(0.04, "2034-03-01", deliv_date, "ZN")
    cf_8pct = calculate_conversion_factor(0.08, "2034-03-01", deliv_date, "ZN")
    assert cf_4pct < 1.0
    assert cf_8pct > 1.0

    # Known reference value check: 4.0% 10-year bond ~ 0.8512
    assert abs(cf_4pct - 0.8512) < 0.001


def test_cheapest_to_deliver_identification():
    """Verify CTD bond selection minimizes gross/net basis."""
    deliv_date = "2024-06-01"
    futures_price = 108.50

    # Basket of deliverable 10Y notes
    b1 = DeliverableBond("91282CDU1", 0.03875, "2033-08-15", 96.25, 0.0435)
    b2 = DeliverableBond("91282CDP2", 0.04125, "2032-11-15", 98.10, 0.0438)
    b3 = DeliverableBond("91282CDH0", 0.04500, "2034-05-15", 101.40, 0.0432)

    basket = [b1, b2, b3]
    ctd_bond, ctd_cf, summary_df = identify_cheapest_to_deliver(
        futures_price=futures_price,
        deliverable_bonds=basket,
        delivery_date=deliv_date,
        contract_symbol="ZN",
    )

    assert len(summary_df) == 3
    assert ctd_bond.cusip == summary_df.iloc[0]["cusip"]
    # Check that basis of CTD is lower than or equal to all others
    assert summary_df.iloc[0]["gross_basis"] <= summary_df.iloc[1]["gross_basis"]
    assert summary_df.iloc[0]["gross_basis"] <= summary_df.iloc[2]["gross_basis"]


def test_contract_dv01_and_duration():
    """Verify DV01 and duration scaling via CTD DV01 / Conversion Factor."""
    deliv_date = "2024-06-01"
    # Benchmark 10Y CTD note
    ctd_bond = DeliverableBond("91282CDU1", 0.040, "2034-05-15", 98.0, 0.0425)
    cf = calculate_conversion_factor(0.040, "2034-05-15", deliv_date, "ZN")

    spec_zn = TREASURY_FUTURES_SPECS["ZN"]
    dv01_zn, dur_zn = calculate_contract_dv01_and_duration(ctd_bond, cf, spec_zn, as_of_date=deliv_date)

    # 10Y duration should be ~6.0 to 8.5 years, DV01 ~ $65 to $98 per contract
    assert 6.0 <= dur_zn <= 9.0
    assert 65.0 <= dv01_zn <= 98.0

    # Test 2Y contract (ZT, $200k notional)
    ctd_2y = DeliverableBond("91282CEX0", 0.045, "2026-05-15", 99.5, 0.047)
    cf_2y = calculate_conversion_factor(0.045, "2026-05-15", deliv_date, "ZT")
    spec_zt = TREASURY_FUTURES_SPECS["ZT"]
    dv01_zt, dur_zt = calculate_contract_dv01_and_duration(ctd_2y, cf_2y, spec_zt, as_of_date=deliv_date)

    # 2Y duration should be ~1.7 to 2.0 years, DV01 ~ $35 to $45 per contract
    assert 1.5 <= dur_zt <= 2.2
    assert 35.0 <= dv01_zt <= 45.0


def test_dv01_neutral_spread_portfolio_tolerance():
    """
    CRITICAL TEST: Assert that constructed DV01-neutral spreads and butterflies
    have net portfolio DV01 strictly within <5% of a single leg's DV01.
    """
    dv01_zt = 39.50   # 2Y
    dv01_zf = 45.20   # 5Y
    dv01_zn = 72.80   # 10Y
    dv01_ub = 215.00  # Ultra Bond

    # 1. 2s10s Spread (ZT vs ZN)
    spread_2s10s = construct_dv01_neutral_spread("ZT", dv01_zt, "ZN", dv01_zn, target_leg1_contracts=100)
    net_dv01_2s10s = spread_2s10s["net_portfolio_dv01"]
    threshold_2s10s = spread_2s10s["tolerance_threshold"]
    pct_leg_2s10s = spread_2s10s["residual_pct_of_leg"]

    assert spread_2s10s["is_dv01_neutral"], f"2s10s spread failed DV01 neutrality: {net_dv01_2s10s} vs {threshold_2s10s}"
    assert abs(net_dv01_2s10s) <= threshold_2s10s
    assert pct_leg_2s10s < 5.0, f"Residual DV01 was {pct_leg_2s10s}% of leg DV01 (exceeds 5%)"

    # 2. 5s30s Spread (ZF vs UB)
    spread_5s30s = construct_dv01_neutral_spread("ZF", dv01_zf, "UB", dv01_ub, target_leg1_contracts=100)
    net_dv01_5s30s = spread_5s30s["net_portfolio_dv01"]
    threshold_5s30s = spread_5s30s["tolerance_threshold"]
    pct_leg_5s30s = spread_5s30s["residual_pct_of_leg"]

    assert spread_5s30s["is_dv01_neutral"]
    assert abs(net_dv01_5s30s) <= threshold_5s30s
    assert pct_leg_5s30s < 5.0

    # 3. 2s-5s-10s Butterfly (ZT - ZF - ZN)
    fly_2s5s10s = construct_dv01_neutral_butterfly("ZT", dv01_zt, "ZF", dv01_zf, "ZN", dv01_zn, belly_contracts=-100)
    net_dv01_fly = fly_2s5s10s["net_portfolio_dv01"]
    threshold_fly = fly_2s5s10s["tolerance_threshold"]
    pct_leg_fly = fly_2s5s10s["residual_pct_of_leg"]

    assert fly_2s5s10s["is_dv01_neutral"]
    assert abs(net_dv01_fly) <= threshold_fly
    assert pct_leg_fly < 5.0


def test_continuous_contract_back_adjustment():
    """Verify Panama additive back-adjustment eliminates roll gaps across contract switches."""
    # Synthetic contract bars around a roll date
    dates_contract1 = pd.date_range("2024-02-01", "2024-02-28", freq="B")
    dates_contract2 = pd.date_range("2024-02-15", "2024-03-31", freq="B")

    # Contract 1 trades around 110.0, Contract 2 trades around 110.75 (0.75 pt roll gap)
    df_h24 = pd.DataFrame({"date": dates_contract1, "close": 110.0 + np.sin(np.linspace(0, 1, len(dates_contract1)))})
    df_m24 = pd.DataFrame({"date": dates_contract2, "close": 110.75 + np.sin(np.linspace(0, 1, len(dates_contract2)))})

    contracts = {"ZNH24": df_h24, "ZNM24": df_m24}

    roll_sched = pd.DataFrame([
        {"contract": "ZNH24", "volume_roll_date": pd.to_datetime("2024-01-15")},
        {"contract": "ZNM24", "volume_roll_date": pd.to_datetime("2024-02-20")},
    ])

    # Unadjusted continuous contract (exhibits roll gap)
    cont_unadj = build_continuous_contract(contracts, roll_sched, adjustment_method="unadjusted")
    # Panama adjusted continuous contract (eliminates roll gap)
    cont_adj = build_continuous_contract(contracts, roll_sched, adjustment_method="panama_additive")

    assert len(cont_adj) == len(cont_unadj)
    # Check roll flag
    roll_rows = cont_adj[cont_adj["roll_flag"]]
    assert len(roll_rows) == 1
    assert roll_rows.iloc[0]["date"] == pd.to_datetime("2024-02-20")

    # Check that price jump at roll date is dampened in adjusted series
    roll_idx = cont_adj[cont_adj["roll_flag"]].index[0]
    unadj_jump = abs(cont_unadj.loc[roll_idx, "unadjusted_price"] - cont_unadj.loc[roll_idx - 1, "unadjusted_price"])
    adj_jump = abs(cont_adj.loc[roll_idx, "adjusted_price"] - cont_adj.loc[roll_idx - 1, "adjusted_price"])
    assert adj_jump < unadj_jump


def test_databento_cost_quote():
    """Verify Databento client live cost quoting and schema-specific rate card estimation."""
    client = DatabentoFuturesClient(api_key=None)  # tests rate card model
    quote_ohlcv = client.get_live_cost_quote(
        dataset="GLBX.MDP3",
        symbols=["ZT.FUT", "ZF.FUT", "ZN.FUT", "TN.FUT", "UB.FUT"],
        schema="ohlcv-1d",
        start="2024-01-01",
        end="2024-06-30",
    )

    assert quote_ohlcv["dataset"] == "GLBX.MDP3"
    assert len(quote_ohlcv["symbols"]) == 5
    assert quote_ohlcv["estimated_cost_usd"] > 0
    assert quote_ohlcv["billable_bytes"] > 0

    quote_trades = client.get_live_cost_quote(
        dataset="GLBX.MDP3",
        symbols=["ZN.FUT"],
        schema="trades",
        start="2024-01-01",
        end="2024-06-30",
    )
    # Trades schema should have higher billable bytes and cost than daily OHLCV
    assert quote_trades["billable_bytes"] > quote_ohlcv["billable_bytes"]
    assert quote_trades["estimated_cost_usd"] > quote_ohlcv["estimated_cost_usd"]
