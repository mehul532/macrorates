"""Macroeconomic surprise generation and curve response module."""

from src.macro.macro_surprises import (
    MACRO_SERIES_SPECS,
    MacroEventHarmonizer,
    MacroRegressionEngine,
    RealTimeMacroIngestor,
    SurpriseEngine,
    CausalMacroResponseEstimator,
    MacroResponseFoldEstimate,
    plot_cpi_impulse_response,
    run_pipeline,
)

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
    "MACRO_SERIES_SPECS",
    "RealTimeMacroIngestor",
    "SurpriseEngine",
    "CausalMacroResponseEstimator",
    "MacroResponseFoldEstimate",
    "MacroEventHarmonizer",
    "MacroRegressionEngine",
    "plot_cpi_impulse_response",
    "run_pipeline",
    "FREDMDLoader",
    "DynamicFactorModelDGR",
    "DFMNowcastSurprise",
    "ExtendedMacroCurveRegression",
    "DFMResult",
    "FRED_MD_CATEGORIES",
    "SERIES_GROUP_MAP",
]

