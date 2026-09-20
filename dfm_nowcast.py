"""
Root convenience export for Dynamic Factor Model (DFM) nowcast module.
Re-exports FREDMDLoader, DynamicFactorModelDGR, DFMNowcastSurprise,
and ExtendedMacroCurveRegression from src.macro.dfm_nowcast.
"""

from src.macro.dfm_nowcast import (
    FREDMDLoader,
    DynamicFactorModelDGR,
    DFMNowcastSurprise,
    ExtendedMacroCurveRegression,
    DFMResult,
    FRED_MD_CATEGORIES,
    SERIES_GROUP_MAP,
)

__all__ = [
    "FREDMDLoader",
    "DynamicFactorModelDGR",
    "DFMNowcastSurprise",
    "ExtendedMacroCurveRegression",
    "DFMResult",
    "FRED_MD_CATEGORIES",
    "SERIES_GROUP_MAP",
]
