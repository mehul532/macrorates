"""Term structure curve modeling module."""

from src.curve.curve import (
    compare_pca_vs_ns,
    compare_svensson_on_date,
    evaluate_against_gsw,
    fit_pca,
    fit_static_nelson_siegel,
)
from src.curve.nelson_siegel import NelsonSiegelFit, StaticNelsonSiegel, nelson_siegel_loadings, NelsonSiegelAR1Forecaster
from src.curve.pca import PCAResult, YieldCurvePCA, PCAVARForecaster
from src.curve.svensson import SvenssonCurve, SvenssonFit, svensson_loadings, compute_aic_bic
from src.curve.canonical import (
    CANONICAL_TENORS,
    CORE_BENCHMARK_TENORS,
    get_canonical_maturities,
    compute_observable_spreads,
)

__all__ = [
    "CANONICAL_TENORS",
    "CORE_BENCHMARK_TENORS",
    "get_canonical_maturities",
    "compute_observable_spreads",
    "YieldCurvePCA",
    "PCAResult",
    "PCAVARForecaster",
    "StaticNelsonSiegel",
    "NelsonSiegelFit",
    "NelsonSiegelAR1Forecaster",
    "SvenssonCurve",
    "SvenssonFit",
    "nelson_siegel_loadings",
    "svensson_loadings",
    "compute_aic_bic",
    "fit_pca",
    "fit_static_nelson_siegel",
    "compare_pca_vs_ns",
    "evaluate_against_gsw",
    "compare_svensson_on_date",
]
