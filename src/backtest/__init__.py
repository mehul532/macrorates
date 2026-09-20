"""
Institutional Backtesting, Walk-Forward, and Regime Robustness Module.
"""

from src.backtest.regimes import (
    load_episodes_config,
    tag_fed_regimes,
    slice_metrics_by_regime,
)
from src.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardHarness,
)
from src.backtest.contracts import (
    ForecastStatus,
    ForecastRecord,
    ForecastLedger,
    ExecutionTimingAssumption,
    DecisionContract,
)
from src.backtest.backtest import (
    compute_episode_performance,
    compute_regime_robustness_table,
    CostModelV1Config,
    BacktestResult,
    RelativeValueBacktestEngine,
    SyntheticDV01Backtest,
)

__all__ = [
    "load_episodes_config",
    "tag_fed_regimes",
    "slice_metrics_by_regime",
    "WalkForwardConfig",
    "WalkForwardHarness",
    "compute_episode_performance",
    "compute_regime_robustness_table",
    "CostModelV1Config",
    "BacktestResult",
    "RelativeValueBacktestEngine",
    "SyntheticDV01Backtest",
    "ForecastStatus",
    "ForecastRecord",
    "ForecastLedger",
    "ExecutionTimingAssumption",
    "DecisionContract",
]
