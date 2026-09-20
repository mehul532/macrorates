"""
Cash/Futures Basis, Cost-of-Carry Pricing, and Implied Repo Rate Module.

Implements:
1. Ingestion of Federal Reserve Bank of New York overnight financing rates:
   - SOFR (Secured Overnight Financing Rate, 2018–2026)
   - TGCR (Tri-Party General Collateral Rate / GC repo rate, 2018–2026)
   - EFFR (Effective Federal Funds Rate, 2000–2018)
2. Cost-of-carry theoretical pricing for Cheapest-to-Deliver (CTD) Treasury bonds:
   F_theo = [(P_clean + AI_t) * (1 + r_repo * dt) - Coupon_Income - AI_T] / CF_CTD
3. Implied Repo Rate (IRR) backed out from market futures vs. cash prices:
   IRR = [(F_mkt * CF_CTD + AI_T + Coupon_Income) - (P_clean + AI_t)] / [(P_clean + AI_t) * dt]
4. Cash/Futures Basis Analytics:
   - Gross Basis = P_clean - F_mkt * CF_CTD
   - Carry = (Coupon_Income + AI_T - AI_t) - (P_clean + AI_t) * r_repo * dt
   - Net Basis = Gross Basis - Carry = (P_clean + AI_t) * (r_repo - IRR) * dt
   - Basis Gap = (IRR - r_repo) * 10,000 bp (Rich if > 0, Cheap if < 0)
5. Historical Repo Stress Analysis:
   - September 2019 Repo Crisis (overnight rates spiking to 5.25%)
   - Quarter-end turn effects and balance-sheet regulatory reporting
   - March 2020 COVID cash-futures basis trade dislocations
"""

from dataclasses import dataclass
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request
import ssl

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BasisResult:
    """Encapsulates all cost-of-carry and cash/futures basis metrics for a bond-futures pair."""
    settlement_date: str
    delivery_date: str
    contract_symbol: str
    days_to_delivery: int
    dt_years: float  # ACT/360 day count fraction
    clean_price: float
    accrued_interest_settle: float
    dirty_price: float
    accrued_interest_delivery: float
    coupon_income: float
    conversion_factor: float
    repo_rate_pct: float
    financing_cost: float
    theoretical_futures_price: float
    market_futures_price: float
    implied_repo_rate_pct: float
    gross_basis: float
    carry: float
    net_basis: float
    basis_gap_bp: float
    valuation_status: str  # "RICH", "CHEAP", or "FAIR"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "settlement_date": self.settlement_date,
            "delivery_date": self.delivery_date,
            "contract_symbol": self.contract_symbol,
            "days_to_delivery": self.days_to_delivery,
            "dt_years": round(self.dt_years, 4),
            "clean_price": round(self.clean_price, 4),
            "accrued_interest_settle": round(self.accrued_interest_settle, 4),
            "dirty_price": round(self.dirty_price, 4),
            "accrued_interest_delivery": round(self.accrued_interest_delivery, 4),
            "coupon_income": round(self.coupon_income, 4),
            "conversion_factor": round(self.conversion_factor, 4),
            "repo_rate_pct": round(self.repo_rate_pct, 4),
            "financing_cost": round(self.financing_cost, 4),
            "theoretical_futures_price": round(self.theoretical_futures_price, 4),
            "market_futures_price": round(self.market_futures_price, 4),
            "implied_repo_rate_pct": round(self.implied_repo_rate_pct, 4),
            "gross_basis": round(self.gross_basis, 4),
            "carry": round(self.carry, 4),
            "net_basis": round(self.net_basis, 4),
            "basis_gap_bp": round(self.basis_gap_bp, 2),
            "valuation_status": self.valuation_status,
        }


class RepoRatesIngestor:
    """
    Ingests, standardizes, and caches Federal Reserve Bank of New York overnight financing rates:
    - SOFR (Secured Overnight Financing Rate, 2018–2026)
    - TGCR (Tri-Party General Collateral Rate / GC repo rate, 2018–2026)
    - EFFR (Effective Federal Funds Rate, 2000–2018)
    """

    DEFAULT_CACHE_PARQUET = Path("data/raw/macro/repo_financing_rates.parquet")
    DEFAULT_CACHE_CSV = Path("data/raw/macro/repo_financing_rates.csv")

    def __init__(self, cache_path: Optional[Union[str, Path]] = None):
        self.cache_path = Path(cache_path) if cache_path else self.DEFAULT_CACHE_PARQUET
        self._rates_df: Optional[pd.DataFrame] = None

    def load_rates(self, auto_fetch: bool = True) -> pd.DataFrame:
        """
        Load continuous repo financing rates from local cache or fetch from NY Fed API.
        """
        if self._rates_df is not None:
            return self._rates_df

        # 1. Try reading from parquet cache
        if self.cache_path.exists():
            try:
                df = pd.read_parquet(self.cache_path)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                self._rates_df = df
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache from {self.cache_path}: {e}")

        # 2. Try reading from CSV cache
        csv_path = self.cache_path.with_suffix(".csv")
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                self._rates_df = df
                return df
            except Exception as e:
                logger.warning(f"Failed to load CSV cache from {csv_path}: {e}")

        # 3. Fetch from NY Fed API if allowed
        if auto_fetch:
            df = self.fetch_nyfed_rates()
            self._rates_df = df
            return df

        raise FileNotFoundError(f"Repo financing rates not found at {self.cache_path}")

    def fetch_nyfed_rates(self) -> pd.DataFrame:
        """Fetch historical SOFR, TGCR, and EFFR rates directly from the NY Fed Markets API."""
        ctx = ssl._create_unverified_context()

        def _fetch_endpoint(url: str) -> List[Dict[str, Any]]:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("refRates", [])

        # Fetch SOFR (2018–2026)
        sofr_list = _fetch_endpoint(
            "https://markets.newyorkfed.org/api/rates/secured/sofr/search.json?startDate=2018-04-02&endDate=2026-05-01"
        )
        df_sofr = pd.DataFrame(sofr_list)
        if not df_sofr.empty:
            df_sofr["date"] = pd.to_datetime(df_sofr["effectiveDate"])
            df_sofr["sofr_rate"] = pd.to_numeric(df_sofr["percentRate"], errors="coerce")
            df_sofr = df_sofr[["date", "sofr_rate"]].dropna().drop_duplicates("date").set_index("date")

        # Fetch TGCR (2018–2026)
        tgcr_list = _fetch_endpoint(
            "https://markets.newyorkfed.org/api/rates/secured/tgcr/search.json?startDate=2018-04-02&endDate=2026-05-01"
        )
        df_tgcr = pd.DataFrame(tgcr_list)
        if not df_tgcr.empty:
            df_tgcr["date"] = pd.to_datetime(df_tgcr["effectiveDate"])
            df_tgcr["tgcr_rate"] = pd.to_numeric(df_tgcr["percentRate"], errors="coerce")
            df_tgcr = df_tgcr[["date", "tgcr_rate"]].dropna().drop_duplicates("date").set_index("date")

        # Fetch EFFR (2000–2018)
        effr_list = _fetch_endpoint(
            "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json?startDate=2000-01-01&endDate=2018-04-01"
        )
        df_effr = pd.DataFrame(effr_list)
        if not df_effr.empty:
            df_effr["date"] = pd.to_datetime(df_effr["effectiveDate"])
            df_effr["effr_rate"] = pd.to_numeric(df_effr["percentRate"], errors="coerce")
            df_effr = df_effr[["date", "effr_rate"]].dropna().drop_duplicates("date").set_index("date")

        merged = pd.concat([df_effr, df_sofr, df_tgcr], axis=1).sort_index()
        merged["repo_rate"] = merged["tgcr_rate"].combine_first(merged["sofr_rate"]).combine_first(merged["effr_rate"])
        merged["rate_type"] = "EFFR"
        merged.loc[merged.index >= "2018-04-02", "rate_type"] = "SOFR/TGCR"

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        merged.reset_index().to_parquet(self.cache_path, index=False)
        return merged

    def get_rate(self, date: Union[str, pd.Timestamp], default_rate: float = 0.05) -> float:
        """
        Get the annualized financing rate as decimal (e.g. 0.0525 for 5.25%) on date.
        """
        df = self.load_rates()
        dt = pd.to_datetime(date)
        if dt in df.index:
            val = df.loc[dt, "repo_rate"]
            if pd.notna(val):
                return float(val) / 100.0

        # Nearest preceding trading day
        prior_dates = df.index[df.index <= dt]
        if len(prior_dates) > 0:
            val = df.loc[prior_dates[-1], "repo_rate"]
            if pd.notna(val):
                return float(val) / 100.0

        return default_rate


class CostOfCarryModel:
    """
    Cost-of-Carry Theoretical Pricing & Basis Engine for Treasury Futures.
    """

    @staticmethod
    def calculate_accrued_interest(
        settlement_date: Union[str, pd.Timestamp],
        maturity_date: Union[str, pd.Timestamp],
        coupon: float,
    ) -> float:
        """
        Calculate semiannual accrued interest per 100 par under ACT/ACT ICMA convention.

        Args:
            settlement_date: Valuation / settlement date
            maturity_date: Bond maturity date
            coupon: Annual coupon rate as decimal (e.g. 0.04 for 4.0%)

        Returns:
            Accrued interest in price points per 100 par.
        """
        settle = pd.to_datetime(settlement_date)
        mat = pd.to_datetime(maturity_date)

        if coupon <= 0:
            return 0.0

        # Semiannual coupon schedule aligns with maturity month and day
        m_month = mat.month
        c1_month = m_month
        c2_month = (m_month - 6 - 1) % 12 + 1

        year = settle.year
        candidates = []
        for y in [year - 1, year, year + 1]:
            for m in [c1_month, c2_month]:
                try:
                    # Day of month capped to last day of month
                    day = min(mat.day, 28 if m == 2 else (30 if m in [4, 6, 9, 11] else 31))
                    candidates.append(pd.Timestamp(year=y, month=m, day=day))
                except Exception:
                    pass

        candidates = sorted(list(set(c for c in candidates if c <= mat)))
        past_coupons = [c for c in candidates if c <= settle]
        future_coupons = [c for c in candidates if c > settle]

        prev_coupon = past_coupons[-1] if past_coupons else settle
        next_coupon = future_coupons[0] if future_coupons else mat

        days_accrued = (settle - prev_coupon).days
        days_in_period = max(1, (next_coupon - prev_coupon).days)

        ai = 100.0 * (coupon / 2.0) * (days_accrued / days_in_period)
        return float(round(ai, 6))

    @classmethod
    def evaluate_basis(
        cls,
        clean_price: float,
        coupon: float,
        maturity_date: Union[str, pd.Timestamp],
        settlement_date: Union[str, pd.Timestamp],
        delivery_date: Union[str, pd.Timestamp],
        conversion_factor: float,
        repo_rate: float,
        market_futures_price: Optional[float] = None,
        contract_symbol: str = "ZN",
        interim_coupons: float = 0.0,
    ) -> BasisResult:
        """
        Evaluate full cost-of-carry theoretical price, Implied Repo Rate (IRR),
        gross basis, net basis, and rich/cheap basis gap.

        Args:
            clean_price: CTD cash clean price per 100 par (e.g. 98.50)
            coupon: Annual coupon rate as decimal (e.g. 0.04 for 4.0%)
            maturity_date: Bond maturity date
            settlement_date: Current trade/settlement date t
            delivery_date: Futures delivery date T
            conversion_factor: CME conversion factor CF_CTD
            repo_rate: Annualized financing rate as decimal (e.g. 0.0525 for 5.25%)
            market_futures_price: Observed market futures price (defaults to theoretical if None)
            contract_symbol: CME contract ticker (ZN, ZF, ZT, TN, UB)
            interim_coupons: Coupon cash received between settlement and delivery

        Returns:
            BasisResult with all metrics.
        """
        settle_dt = pd.to_datetime(settlement_date)
        deliv_dt = pd.to_datetime(delivery_date)

        days_to_delivery = max(1, (deliv_dt - settle_dt).days)
        dt_years = days_to_delivery / 360.0  # Money-market ACT/360 convention

        # 1. Accrued interest & dirty prices
        ai_settle = cls.calculate_accrued_interest(settle_dt, maturity_date, coupon)
        ai_deliv = cls.calculate_accrued_interest(deliv_dt, maturity_date, coupon)

        dirty_price = clean_price + ai_settle

        # 2. Financing cost and cost-of-carry theoretical futures price
        # Grossed-up cash value: Dirty_Price * (1 + repo_rate * dt)
        financing_cost = dirty_price * repo_rate * dt_years
        grossed_up = dirty_price + financing_cost

        # Invoice price at delivery under no-arbitrage: F_theo * CF + AI_deliv = Grossed_Up - Coupons
        converted_theo = grossed_up - interim_coupons - ai_deliv
        f_theo = converted_theo / conversion_factor

        # 3. Market futures price & Implied Repo Rate (IRR)
        f_mkt = market_futures_price if market_futures_price is not None else f_theo

        # Invoice delivery proceeds: F_mkt * CF + AI_deliv + Coupons
        total_proceeds = f_mkt * conversion_factor + ai_deliv + interim_coupons

        # IRR solves: Dirty_Price * (1 + IRR * dt) = Total_Proceeds
        # IRR = (Total_Proceeds - Dirty_Price) / (Dirty_Price * dt)
        irr_decimal = (total_proceeds - dirty_price) / (dirty_price * dt_years)
        irr_pct = irr_decimal * 100.0
        repo_pct = repo_rate * 100.0

        # 4. Gross Basis, Carry, Net Basis, and Basis Gap
        converted_mkt = f_mkt * conversion_factor
        gross_basis = clean_price - converted_mkt

        # Carry = Coupon income earned minus repo financing cost
        carry = (interim_coupons + ai_deliv - ai_settle) - financing_cost

        # Net Basis (Basis after Carry)
        # Net Basis = Gross Basis - Carry = Dirty_Price * (repo_rate - IRR) * dt
        net_basis = gross_basis - carry

        # Basis Gap in basis points: (IRR - repo_rate) * 10,000
        basis_gap_bp = (irr_pct - repo_pct) * 100.0

        # Rich / Cheap classification
        if basis_gap_bp > 2.0:
            status = "RICH"  # Futures is rich vs cash (IRR > repo rate)
        elif basis_gap_bp < -2.0:
            status = "CHEAP"  # Futures is cheap vs cash (IRR < repo rate, positive net basis)
        else:
            status = "FAIR"

        return BasisResult(
            settlement_date=settle_dt.strftime("%Y-%m-%d"),
            delivery_date=deliv_dt.strftime("%Y-%m-%d"),
            contract_symbol=contract_symbol,
            days_to_delivery=days_to_delivery,
            dt_years=dt_years,
            clean_price=clean_price,
            accrued_interest_settle=ai_settle,
            dirty_price=dirty_price,
            accrued_interest_delivery=ai_deliv,
            coupon_income=interim_coupons,
            conversion_factor=conversion_factor,
            repo_rate_pct=repo_pct,
            financing_cost=financing_cost,
            theoretical_futures_price=f_theo,
            market_futures_price=f_mkt,
            implied_repo_rate_pct=irr_pct,
            gross_basis=gross_basis,
            carry=carry,
            net_basis=net_basis,
            basis_gap_bp=basis_gap_bp,
            valuation_status=status,
        )


class CashFuturesBasisAnalyzer:
    """
    Analyzes historical cash/futures basis, Implied Repo Rates,
    and stress episode dynamics across contracts.
    """

    KNOWN_STRESS_EPISODES = {
        "sep_2019_repo_crisis": {
            "name": "September 2019 Repo Crisis",
            "start_date": "2019-09-10",
            "end_date": "2019-09-25",
            "peak_date": "2019-09-17",
            "description": "Overnight SOFR spiked to 5.25% (up 305 bp) due to corporate tax outflows and Treasury settlement.",
        },
        "dec_2018_year_end": {
            "name": "December 2018 Year-End Turn",
            "start_date": "2018-12-20",
            "end_date": "2019-01-05",
            "peak_date": "2018-12-31",
            "description": "Regulatory balance-sheet window-dressing turn premium.",
        },
        "mar_2020_covid_squeeze": {
            "name": "March 2020 COVID Basis Trade Unwind",
            "start_date": "2020-03-01",
            "end_date": "2020-03-31",
            "peak_date": "2020-03-16",
            "description": "Treasury market liquidity dry-up and hedge fund basis trade deleveraging.",
        },
        "dec_2022_year_end": {
            "name": "December 2022 Year-End Turn",
            "start_date": "2022-12-20",
            "end_date": "2023-01-05",
            "peak_date": "2022-12-30",
            "description": "Rapid Fed hiking cycle and collateral scarcity.",
        },
    }

    def __init__(self, repo_ingestor: Optional[RepoRatesIngestor] = None):
        self.repo_ingestor = repo_ingestor or RepoRatesIngestor()

    def generate_daily_basis_history(
        self,
        yield_panel_path: Union[str, Path] = "data/processed/yield_panel.parquet",
        contract_symbol: str = "ZN",
        start_date: str = "2018-04-02",
        end_date: str = "2026-04-01",
    ) -> pd.DataFrame:
        """
        Generate continuous daily time series of cash/futures basis, IRR, and financing spread.
        """
        repo_df = self.repo_ingestor.load_rates()
        yields_df = pd.read_parquet(yield_panel_path)
        yields_df["date"] = pd.to_datetime(yields_df["date"])
        yields_df = yields_df.set_index("date").sort_index()

        mask = (yields_df.index >= start_date) & (yields_df.index <= end_date)
        sub_yields = yields_df.loc[mask]

        records = []
        # Delivery cycle: Quarterly Mar (H), Jun (M), Sep (U), Dec (Z)
        cycle_months = [3, 6, 9, 12]

        for dt, row in sub_yields.iterrows():
            y_10 = float(row.get("DGS10", np.nan))
            if np.isnan(y_10):
                continue

            # Find next quarterly delivery date
            y_val = dt.year
            deliv_candidates = [
                pd.Timestamp(year=y_val, month=m, day=20)
                for m in cycle_months
                if pd.Timestamp(year=y_val, month=m, day=20) > dt + pd.Timedelta(days=5)
            ]
            if not deliv_candidates:
                deliv_candidates = [pd.Timestamp(year=y_val + 1, month=3, day=20)]
            delivery_date = deliv_candidates[0]

            # CTD benchmark bond: on-the-run Treasury with 10Y maturity at issuance
            coupon = round(y_10 / 0.125) * 0.125 / 100.0  # nearest 1/8%
            maturity_date = dt + pd.Timedelta(days=int(9.75 * 365.25))

            # CTD cash clean price from yield
            # P_clean = sum_{t=1..20} (c/2) / (1 + y/2)^t + 100 / (1 + y/2)^20
            r_semi = y_10 / 200.0
            c_semi = (coupon * 100.0) / 2.0
            t_vec = np.arange(1, 21)
            p_clean = float(np.sum(c_semi / (1.0 + r_semi) ** t_vec) + 100.0 / (1.0 + r_semi) ** 20)

            # Conversion factor
            # Approximate standard 6% conversion factor formula
            cf = round(float(np.sum(c_semi / (1.03) ** t_vec) + 100.0 / (1.03) ** 20) / 100.0, 4)

            # Actual overnight repo rate on date
            repo_rate = self.repo_ingestor.get_rate(dt)

            # Term financing rate expected by the market over the delivery horizon:
            # Futures contracts on CME Globex price off term financing expectations (e.g. rolling 10-day
            # baseline), not a 1-day idiosyncratic overnight squeeze.
            # When overnight repo spikes (e.g. Sept 17, 2019 to 5.25%), actual financing costs surge,
            # blowing out net basis and driving the basis gap deeply negative.
            dt_prior = dt - pd.Timedelta(days=10)
            sub_prior = repo_df.loc[dt_prior:dt, "repo_rate"].dropna()
            term_repo_rate = float(sub_prior.median()) / 100.0 if not sub_prior.empty else repo_rate

            # Baseline theoretical futures price under actual financing rate
            res_theo = CostOfCarryModel.evaluate_basis(
                clean_price=p_clean,
                coupon=coupon,
                maturity_date=maturity_date,
                settlement_date=dt,
                delivery_date=delivery_date,
                conversion_factor=cf,
                repo_rate=repo_rate,
                contract_symbol=contract_symbol,
            )

            # Market futures price trades at term financing minus the delivery option discount (~0.15 - 0.25 pts)
            res_term = CostOfCarryModel.evaluate_basis(
                clean_price=p_clean,
                coupon=coupon,
                maturity_date=maturity_date,
                settlement_date=dt,
                delivery_date=delivery_date,
                conversion_factor=cf,
                repo_rate=term_repo_rate,
                contract_symbol=contract_symbol,
            )
            option_discount = 0.20  # ~20 ticks of delivery option value
            f_market = res_term.theoretical_futures_price - (option_discount / cf)

            # Re-evaluate with observed market futures against actual repo rate
            res_final = CostOfCarryModel.evaluate_basis(
                clean_price=p_clean,
                coupon=coupon,
                maturity_date=maturity_date,
                settlement_date=dt,
                delivery_date=delivery_date,
                conversion_factor=cf,
                repo_rate=repo_rate,
                market_futures_price=f_market,
                contract_symbol=contract_symbol,
            )


            rec = res_final.to_dict()
            rec["date"] = dt
            rec["yield_10y"] = y_10
            records.append(rec)

        df_out = pd.DataFrame(records).set_index("date").sort_index()
        return df_out

    def extract_stress_episode_metrics(
        self,
        basis_df: pd.DataFrame,
        episode_key: str = "sep_2019_repo_crisis",
    ) -> Dict[str, Any]:
        """
        Extract detailed statistical metrics for a specific historical stress episode.
        """
        if episode_key not in self.KNOWN_STRESS_EPISODES:
            raise KeyError(f"Unknown episode: {episode_key}")

        ep = self.KNOWN_STRESS_EPISODES[episode_key]
        sub = basis_df.loc[ep["start_date"]:ep["end_date"]]
        if sub.empty:
            return {"error": "No data in episode range"}

        peak_dt = pd.to_datetime(ep["peak_date"])
        peak_row = sub.loc[peak_dt] if peak_dt in sub.index else sub.iloc[sub["repo_rate_pct"].argmax()]

        return {
            "episode_name": ep["name"],
            "start_date": ep["start_date"],
            "end_date": ep["end_date"],
            "peak_date": ep["peak_date"],
            "description": ep["description"],
            "n_days": len(sub),
            "peak_repo_rate_pct": round(float(peak_row["repo_rate_pct"]), 3),
            "peak_irr_pct": round(float(peak_row["implied_repo_rate_pct"]), 3),
            "peak_basis_gap_bp": round(float(peak_row["basis_gap_bp"]), 2),
            "peak_net_basis": round(float(peak_row["net_basis"]), 4),
            "mean_repo_rate_pct": round(float(sub["repo_rate_pct"].mean()), 3),
            "mean_irr_pct": round(float(sub["implied_repo_rate_pct"].mean()), 3),
            "mean_basis_gap_bp": round(float(sub["basis_gap_bp"].mean()), 2),
            "mean_net_basis": round(float(sub["net_basis"].mean()), 4),
        }
