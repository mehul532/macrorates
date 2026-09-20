"""
Canonical Tenor-to-Maturity and Benchmark Panel Definitions.

Eliminates positional slicing errors by mapping every Treasury tenor
strictly through an immutable dictionary of continuous maturities.
"""

from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np


CANONICAL_TENORS: Dict[str, float] = {
    "DGS1MO": 1.0 / 12.0,
    "DGS3MO": 3.0 / 12.0,
    "DGS6MO": 6.0 / 12.0,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}

# Standard comparison panel that maintains complete uninterrupted daily historical coverage
CORE_BENCHMARK_TENORS: List[str] = [
    "DGS3MO",
    "DGS1",
    "DGS2",
    "DGS3",
    "DGS5",
    "DGS7",
    "DGS10",
    "DGS30",
]


def get_canonical_maturities(tenor_cols: Sequence[str]) -> np.ndarray:
    """
    Return array of maturities looked up strictly by column name.
    
    Guarantees that tenor order and missing tenors never corrupt maturity mappings.
    """
    missing = [c for c in tenor_cols if c not in CANONICAL_TENORS]
    if missing:
        raise KeyError(f"Tenors not recognized in CANONICAL_TENORS: {missing}")
    return np.array([CANONICAL_TENORS[c] for c in tenor_cols], dtype=float)


def compute_observable_spreads(yields_df_or_array: np.ndarray, tenor_cols: Sequence[str]) -> Dict[str, np.ndarray]:
    """
    Compute observable 2s10s slope and 2s5s10s butterfly from yield arrays.
    
    2s10s Slope = y(10Y) - y(2Y)
    2s5s10s Fly = 2 * y(5Y) - y(2Y) - y(10Y)
    """
    idx_map = {c: i for i, c in enumerate(tenor_cols)}
    spreads = {}
    
    if "DGS2" in idx_map and "DGS10" in idx_map:
        i2 = idx_map["DGS2"]
        i10 = idx_map["DGS10"]
        spreads["2s10s"] = yields_df_or_array[..., i10] - yields_df_or_array[..., i2]
        
    if "DGS2" in idx_map and "DGS5" in idx_map and "DGS10" in idx_map:
        i2 = idx_map["DGS2"]
        i5 = idx_map["DGS5"]
        i10 = idx_map["DGS10"]
        spreads["2s5s10s"] = 2.0 * yields_df_or_array[..., i5] - yields_df_or_array[..., i2] - yields_df_or_array[..., i10]
        
    return spreads
