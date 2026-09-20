"""
Unit test suite for Cash/Futures Basis, Cost-of-Carry Pricing, and Implied Repo Rate.
Tests exact mathematical no-arbitrage identities, rich/cheap sensitivity,
accrued interest conventions, repo rate ingestion, and September 2019 stress behavior.
"""

import numpy as np
import pandas as pd
import pytest

from src.futures.basis import (
    BasisResult,
    RepoRatesIngestor,
    CostOfCarryModel,
    CashFuturesBasisAnalyzer,
)


def test_accrued_interest_exact():
    """Verify ACT/ACT ICMA semiannual accrued interest calculation."""
    # Semi-annual 4.0% bond maturing May 15
    settle = "2024-06-15"
    mat = "2034-05-15"
    coupon = 0.04  # 4%
    ai = CostOfCarryModel.calculate_accrued_interest(settle, mat, coupon)
    
    # Between May 15 and June 15 is 31 days out of 184 days in May-Nov period
    expected = 100.0 * (0.04 / 2.0) * (31.0 / 184.0)
    assert np.isclose(ai, expected, atol=0.01)


def test_cost_of_carry_exact_identity():
    """
    Verify fundamental no-arbitrage theorem:
    When Market Futures Price == Theoretical Cost-of-Carry Price:
    1. Implied Repo Rate (IRR) must identically equal the actual repo financing rate.
    2. Net Basis (basis after carry) must identically equal zero.
    """
    clean_p = 99.25
    coupon = 0.0425
    mat = "2033-08-15"
    settle = "2024-04-10"
    deliv = "2024-06-20"
    cf = 0.8125
    repo_rate = 0.0535  # 5.35%

    res = CostOfCarryModel.evaluate_basis(
        clean_price=clean_p,
        coupon=coupon,
        maturity_date=mat,
        settlement_date=settle,
        delivery_date=deliv,
        conversion_factor=cf,
        repo_rate=repo_rate,
        market_futures_price=None,  # Evaluates at theoretical price
    )

    assert isinstance(res, BasisResult)
    # 1. IRR == repo_rate * 100
    assert np.isclose(res.implied_repo_rate_pct, repo_rate * 100.0, atol=1e-8)
    # 2. Net Basis == 0.0
    assert np.isclose(res.net_basis, 0.0, atol=1e-8)
    # 3. Basis Gap == 0.0
    assert np.isclose(res.basis_gap_bp, 0.0, atol=1e-6)
    assert res.valuation_status == "FAIR"


def test_rich_cheap_valuation_sensitivity():
    """Verify that futures prices above theoretical are rich, and below are cheap."""
    clean_p = 100.0
    coupon = 0.04
    mat = "2034-02-15"
    settle = "2024-04-01"
    deliv = "2024-06-20"
    cf = 0.8000
    repo_rate = 0.0500  # 5.0%

    # Baseline fair value
    res_fair = CostOfCarryModel.evaluate_basis(
        clean_p, coupon, mat, settle, deliv, cf, repo_rate
    )
    f_theo = res_fair.theoretical_futures_price

    # Case 1: Market futures is 0.50 points higher (Rich)
    res_rich = CostOfCarryModel.evaluate_basis(
        clean_p, coupon, mat, settle, deliv, cf, repo_rate,
        market_futures_price=f_theo + 0.50,
    )
    assert res_rich.valuation_status == "RICH"
    assert res_rich.implied_repo_rate_pct > repo_rate * 100.0
    assert res_rich.basis_gap_bp > 0
    assert res_rich.net_basis < 0  # negative net basis indicates cash-and-carry profit

    # Case 2: Market futures is 0.50 points lower (Cheap / Delivery Option Premium)
    res_cheap = CostOfCarryModel.evaluate_basis(
        clean_p, coupon, mat, settle, deliv, cf, repo_rate,
        market_futures_price=f_theo - 0.50,
    )
    assert res_cheap.valuation_status == "CHEAP"
    assert res_cheap.implied_repo_rate_pct < repo_rate * 100.0
    assert res_cheap.basis_gap_bp < 0
    assert res_cheap.net_basis > 0  # positive net basis indicates cheap futures


def test_repo_rates_ingestor():
    """Verify loading repo financing rates from local cache."""
    ingestor = RepoRatesIngestor()
    df = ingestor.load_rates(auto_fetch=False)

    assert len(df) > 5000
    assert "repo_rate" in df.columns
    assert "rate_type" in df.columns

    # Verify SOFR rate on Sept 17, 2019 spike
    r_spike = ingestor.get_rate("2019-09-17")
    assert np.isclose(r_spike, 0.0525, atol=1e-4)  # 5.25%

    # Verify pre-2018 EFFR
    r_2015 = ingestor.get_rate("2015-06-15")
    assert 0.0005 <= r_2015 <= 0.0050  # Near ZLB


def test_september_2019_repo_crisis_behavior():
    """
    Sanity check: During the Sept 17, 2019 repo crisis, the sudden +305 bp spike in actual repo
    financing costs must cause the Basis Gap (IRR - repo) to plummet deeply negative
    and Net Basis to widen sharply.
    """
    analyzer = CashFuturesBasisAnalyzer()
    df_basis = analyzer.generate_daily_basis_history(
        start_date="2019-09-10",
        end_date="2019-09-25",
        contract_symbol="ZN",
    )

    assert not df_basis.empty
    assert "2019-09-17" in df_basis.index.strftime("%Y-%m-%d")

    # On Sept 16, repo was ~2.42%. On Sept 17, repo was 5.25%.
    r_16 = df_basis.loc["2019-09-16", "repo_rate_pct"]
    r_17 = df_basis.loc["2019-09-17", "repo_rate_pct"]
    assert np.isclose(r_17, 5.25, atol=0.01)
    assert r_17 > r_16 + 2.5  # over 250 bp surge

    # Net basis on Sept 17 must be elevated relative to Sept 16
    nb_16 = df_basis.loc["2019-09-16", "net_basis"]
    nb_17 = df_basis.loc["2019-09-17", "net_basis"]
    assert nb_17 > nb_16

    # Extract stress episode scorecard
    ep_metrics = analyzer.extract_stress_episode_metrics(df_basis, "sep_2019_repo_crisis")
    assert ep_metrics["peak_repo_rate_pct"] == 5.25
    assert ep_metrics["peak_basis_gap_bp"] < -200.0  # deeply negative spread
