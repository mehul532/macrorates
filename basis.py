"""
Root convenience export for Cash/Futures Basis & Implied Repo Rate Module.
Re-exports CostOfCarryModel, RepoRatesIngestor, CashFuturesBasisAnalyzer,
and BasisResult from src.futures.basis.
"""

from src.futures.basis import (
    BasisResult,
    RepoRatesIngestor,
    CostOfCarryModel,
    CashFuturesBasisAnalyzer,
)

__all__ = [
    "BasisResult",
    "RepoRatesIngestor",
    "CostOfCarryModel",
    "CashFuturesBasisAnalyzer",
]


if __name__ == "__main__":
    print("MacroRates Cash/Futures Basis & Implied Repo Rate Engine")
    ingestor = RepoRatesIngestor()
    rates = ingestor.load_rates()
    print(f"Loaded {len(rates)} daily financing rates from {rates.index[0].date()} to {rates.index[-1].date()}")
    
    # Sample Cost-of-Carry calculation
    res = CostOfCarryModel.evaluate_basis(
        clean_price=98.50,
        coupon=0.04,
        maturity_date="2034-05-15",
        settlement_date="2024-04-15",
        delivery_date="2024-06-20",
        conversion_factor=0.8250,
        repo_rate=0.0530,
        market_futures_price=118.25,
    )
    print("\nSample Basis Evaluation Result:")
    for k, v in res.to_dict().items():
        print(f"  {k}: {v}")
