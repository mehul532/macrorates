"""
Systematic Relative-Value Strategy and Backtest Engine.
"""

from src.strategy.signals import (
    compute_slope_zscore_signals,
    compute_dns_factor_signals,
    compute_svensson_factor_signals,
    compute_macro_surprise_signals,
    generate_all_signals,
    MACRO_EVENT_BETAS,
)
from src.strategy.portfolio import (
    allocate_2s10s_spread,
    allocate_2s5s10s_butterfly,
    allocate_outright_duration_trade,
    compute_continuous_positions,
    DEFAULT_FUTURES_DV01,
)
from src.strategy.etf_prototype import (
    ETF_SPECS,
    ENGINEERING_VALIDATION_TAG,
    generate_synthetic_etf_panel,
    allocate_etf_2s10s_spread,
    allocate_etf_2s5s10s_butterfly,
)
from src.strategy.backtest import (
    CostModelV1Config,
    BacktestResult,
    RelativeValueBacktestEngine,
)
from src.strategy.attribution import (
    run_factor_attribution,
)
from src.strategy.strategy import (
    SystematicRelativeValuePipeline,
)

__all__ = [
    "compute_slope_zscore_signals",
    "compute_dns_factor_signals",
    "compute_svensson_factor_signals",
    "compute_macro_surprise_signals",
    "generate_all_signals",
    "MACRO_EVENT_BETAS",
    "allocate_2s10s_spread",
    "allocate_2s5s10s_butterfly",
    "allocate_outright_duration_trade",
    "compute_continuous_positions",
    "DEFAULT_FUTURES_DV01",
    "ETF_SPECS",
    "ENGINEERING_VALIDATION_TAG",
    "generate_synthetic_etf_panel",
    "allocate_etf_2s10s_spread",
    "allocate_etf_2s5s10s_butterfly",
    "CostModelV1Config",
    "BacktestResult",
    "RelativeValueBacktestEngine",
    "run_factor_attribution",
    "SystematicRelativeValuePipeline",
]
