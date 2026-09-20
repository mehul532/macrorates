"""Macroeconomic surprise generation and curve response module."""

from src.macro.macro_surprises import (
    MACRO_SERIES_SPECS,
    MacroEventHarmonizer,
    MacroRegressionEngine,
    RealTimeMacroIngestor,
    SurpriseEngine,
    plot_cpi_impulse_response,
    run_pipeline,
)

__all__ = [
    "MACRO_SERIES_SPECS",
    "RealTimeMacroIngestor",
    "SurpriseEngine",
    "MacroEventHarmonizer",
    "MacroRegressionEngine",
    "plot_cpi_impulse_response",
    "run_pipeline",
]
