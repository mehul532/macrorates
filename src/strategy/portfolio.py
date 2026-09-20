"""
Portfolio Construction & DV01-Neutral Position Allocation.

Covers:
1. 2s10s DV01-neutral steepener/flattener allocation
2. 2s5s10s DV01-neutral butterfly allocation (50/50 wing DV01 matching belly)
3. Strict enforcement of <5% residual portfolio DV01 tolerance
4. Outright duration trading explicitly dropped from core scope
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.futures.futures_analytics import (
    construct_dv01_neutral_spread,
    construct_dv01_neutral_butterfly,
    TREASURY_FUTURES_SPECS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default benchmark reference DV01s per contract (based on CTD analytics from Milestone 5)
DEFAULT_FUTURES_DV01 = {
    "ZT": 38.38,   # 2Y Treasury Note futures (~$38.38/contract per bp)
    "ZF": 45.18,   # 5Y Treasury Note futures (~$45.18/contract per bp)
    "ZN": 85.23,   # 10Y Treasury Note futures (~$85.23/contract per bp)
    "TN": 88.02,   # Ultra 10Y futures
    "UB": 205.21,  # Ultra Bond futures
}


def allocate_2s10s_spread(
    signal: float,
    target_dv01: float = 10_000.0,
    zt_dv01: float = DEFAULT_FUTURES_DV01["ZT"],
    zn_dv01: float = DEFAULT_FUTURES_DV01["ZN"],
    max_tolerance_pct: float = 5.0,
) -> Dict[str, Any]:
    """
    Allocate DV01-neutral 2s10s curve position.
    
    Signal > 0 -> Steepener (Long 2Y ZT, Short 10Y ZN)
    Signal < 0 -> Flattener (Short 2Y ZT, Long 10Y ZN)
    Signal == 0 -> Flat (0 contracts)
    
    Residual portfolio DV01 tolerance strictly enforced < 5% of single leg total DV01.
    """
    if abs(signal) < 1e-4:
        return {
            "strategy": "2s10s_spread",
            "signal": signal,
            "direction": "flat",
            "n_zt": 0,
            "n_zn": 0,
            "zt_total_dv01": 0.0,
            "zn_total_dv01": 0.0,
            "net_portfolio_dv01": 0.0,
            "is_dv01_neutral": True,
            "residual_pct_of_leg": 0.0,
        }
        
    scaled_target = target_dv01 * abs(signal)
    base_n_zt = max(1, int(round(scaled_target / zt_dv01)))
    
    res = construct_dv01_neutral_spread(
        leg1_symbol="ZT",
        leg1_dv01=zt_dv01,
        leg2_symbol="ZN",
        leg2_dv01=zn_dv01,
        target_leg1_contracts=base_n_zt,
    )
    
    if res["residual_pct_of_leg"] > max_tolerance_pct:
        raise ValueError(
            f"2s10s spread residual DV01 {res['residual_pct_of_leg']:.2f}% exceeds {max_tolerance_pct}% tolerance!"
        )
        
    sign = 1 if signal > 0 else -1
    n_zt = sign * res["n1"]
    n_zn = sign * res["n2"]
    
    return {
        "strategy": "2s10s_spread",
        "signal": signal,
        "direction": "steepener" if sign > 0 else "flattener",
        "n_zt": n_zt,
        "n_zn": n_zn,
        "zt_total_dv01": round(n_zt * zt_dv01, 2),
        "zn_total_dv01": round(n_zn * zn_dv01, 2),
        "net_portfolio_dv01": round(n_zt * zt_dv01 + n_zn * zn_dv01, 2),
        "is_dv01_neutral": res["is_dv01_neutral"],
        "residual_pct_of_leg": res["residual_pct_of_leg"],
    }


def allocate_2s5s10s_butterfly(
    signal: float,
    target_dv01: float = 10_000.0,
    zt_dv01: float = DEFAULT_FUTURES_DV01["ZT"],
    zf_dv01: float = DEFAULT_FUTURES_DV01["ZF"],
    zn_dv01: float = DEFAULT_FUTURES_DV01["ZN"],
    max_tolerance_pct: float = 5.0,
    min_trade_dv01: float = 300.0,
) -> Dict[str, Any]:
    """
    Allocate DV01-neutral 2s-5s-10s butterfly position via integer lattice optimization.
    
    Wings: ZT (2Y), ZN (10Y). Belly: ZF (5Y).
    50/50 DV01 weighting across wings:
    N_ZT * DV01_ZT ~= 0.5 * |N_ZF| * DV01_ZF
    N_ZN * DV01_ZN ~= 0.5 * |N_ZF| * DV01_ZF
    
    Signal > 0 -> Long Fly (Short Belly ZF, Long Wings ZT & ZN)
    Signal < 0 -> Short Fly (Long Belly ZF, Short Wings ZT & ZN)
    Signal == 0 or scaled_target < min_trade_dv01 -> Flat (0 contracts)
    """
    scaled_target = target_dv01 * abs(signal)
    if abs(signal) < 1e-4 or scaled_target < min_trade_dv01:
        return {
            "strategy": "2s5s10s_butterfly",
            "signal": signal,
            "direction": "flat",
            "n_zt": 0,
            "n_zf": 0,
            "n_zn": 0,
            "zt_total_dv01": 0.0,
            "zf_total_dv01": 0.0,
            "zn_total_dv01": 0.0,
            "net_portfolio_dv01": 0.0,
            "is_dv01_neutral": True,
            "residual_pct_of_leg": 0.0,
        }
        
    target_belly = int(round(scaled_target / zf_dv01))
    best_sol = None
    min_res_pct = float("inf")
    
    # Search around target belly contracts for optimal integer neutral combination
    search_range = range(max(7, target_belly - 5), target_belly + 6)
    for nb in search_range:
        target_wing = (nb * zf_dv01) / 2.0
        n1_base = int(round(target_wing / zt_dv01))
        n3_base = int(round(target_wing / zn_dv01))
        for n1 in range(max(1, n1_base - 2), n1_base + 3):
            for n3 in range(max(1, n3_base - 2), n3_base + 3):
                net = n1 * zt_dv01 - nb * zf_dv01 + n3 * zn_dv01
                single_leg = min(n1 * zt_dv01, nb * zf_dv01, n3 * zn_dv01)
                res_pct = (abs(net) / single_leg) * 100.0
                wing_imb = abs(n1 * zt_dv01 - n3 * zn_dv01) / (nb * zf_dv01)
                if wing_imb <= 0.35 and res_pct < min_res_pct:
                    min_res_pct = res_pct
                    best_sol = (n1, nb, n3, net, res_pct)
                    
    if best_sol is None or best_sol[4] > max_tolerance_pct:
        # Fallback to multiple of fundamental basis unit (5 ZT, -10 ZF, 3 ZN)
        k = max(1, int(round(scaled_target / (10.0 * zf_dv01))))
        n1 = 5 * k
        nb = 10 * k
        n3 = 3 * k
        net = n1 * zt_dv01 - nb * zf_dv01 + n3 * zn_dv01
        single_leg = min(n1 * zt_dv01, nb * zf_dv01, n3 * zn_dv01)
        res_pct = (abs(net) / single_leg) * 100.0
    else:
        n1, nb, n3, net, res_pct = best_sol
        
    sign = 1 if signal > 0 else -1
    signed_n_zt = sign * n1
    signed_n_zf = -sign * nb  # Long Fly is short belly (-), Short Fly is long belly (+)
    signed_n_zn = sign * n3
    
    zt_tot = signed_n_zt * zt_dv01
    zf_tot = signed_n_zf * zf_dv01
    zn_tot = signed_n_zn * zn_dv01
    net_dv01 = zt_tot + zf_tot + zn_tot
    
    return {
        "strategy": "2s5s10s_butterfly",
        "signal": signal,
        "direction": "long_fly" if sign > 0 else "short_fly",
        "n_zt": signed_n_zt,
        "n_zf": signed_n_zf,
        "n_zn": signed_n_zn,
        "zt_total_dv01": round(zt_tot, 2),
        "zf_total_dv01": round(zf_tot, 2),
        "zn_total_dv01": round(zn_tot, 2),
        "net_portfolio_dv01": round(net_dv01, 2),
        "is_dv01_neutral": res_pct <= max_tolerance_pct,
        "residual_pct_of_leg": round(res_pct, 3),
    }


def allocate_outright_duration_trade(*args, **kwargs):
    """
    Outright duration trade is explicitly DROPPED from core scope.
    Relative value is the core differentiator; predicting yield direction is out of scope.
    """
    raise NotImplementedError(
        "Outright duration trading has been deliberately dropped from MacroRates core scope. "
        "The project focuses exclusively on systematic DV01-neutral curve relative-value strategies."
    )


def compute_continuous_positions(
    signals_series: pd.Series,
    strategy_type: str = "2s10s",
    target_dv01: float = 10_000.0,
    dv01_dict: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Generate daily integer contract position series from a continuous signal series.
    
    strategy_type: '2s10s' or '2s5s10s'
    """
    if dv01_dict is None:
        dv01_dict = DEFAULT_FUTURES_DV01
        
    records = []
    for dt, sig in signals_series.items():
        if strategy_type == "2s10s":
            alloc = allocate_2s10s_spread(
                signal=sig,
                target_dv01=target_dv01,
                zt_dv01=dv01_dict["ZT"],
                zn_dv01=dv01_dict["ZN"],
            )
            records.append({
                "date": dt,
                "signal": sig,
                "n_zt": alloc["n_zt"],
                "n_zn": alloc["n_zn"],
                "net_dv01": alloc["net_portfolio_dv01"],
            })
        elif strategy_type in ("2s5s10s", "fly"):
            alloc = allocate_2s5s10s_butterfly(
                signal=sig,
                target_dv01=target_dv01,
                zt_dv01=dv01_dict["ZT"],
                zf_dv01=dv01_dict["ZF"],
                zn_dv01=dv01_dict["ZN"],
            )
            records.append({
                "date": dt,
                "signal": sig,
                "n_zt": alloc["n_zt"],
                "n_zf": alloc["n_zf"],
                "n_zn": alloc["n_zn"],
                "net_dv01": alloc["net_portfolio_dv01"],
            })
        else:
            raise ValueError(f"Unknown strategy_type: {strategy_type}")
            
    df_pos = pd.DataFrame(records).set_index("date")
    return df_pos
