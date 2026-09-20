"""
Root-level entry point for Bayesian and Regime-Switching State-Space Nelson-Siegel modeling.

Re-exports core engines from src.state_space.bayesian_state_space and provides
a command-line execution interface for model estimation, benchmarking, and diagnostics.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.fred_ingest import CMT_SERIES_CONFIG
from src.data.pipeline import load_yield_panel
from src.state_space.bayesian_state_space import (
    RegimeSwitchingNelsonSiegel,
    RegimeSwitchingResults,
    BayesianNelsonSiegelSampler,
    BayesianStateSpaceResults,
    compare_all_state_space_models,
)

DEFAULT_MATURITIES = {k: v["maturity_years"] for k, v in CMT_SERIES_CONFIG.items()}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("bayesian_state_space")

__all__ = [
    "RegimeSwitchingNelsonSiegel",
    "RegimeSwitchingResults",
    "BayesianNelsonSiegelSampler",
    "BayesianStateSpaceResults",
    "compare_all_state_space_models",
]


def run_cli():
    parser = argparse.ArgumentParser(description="Estimate Regime-Switching Kim Filter and Bayesian MCMC Dynamic Nelson-Siegel")
    parser.add_argument("--mcmc-draws", type=int, default=300, help="Number of Gibbs MCMC draws (default: 300)")
    parser.add_argument("--burn-in", type=int, default=50, help="MCMC burn-in iterations (default: 50)")
    parser.add_argument("--subsample", type=int, default=1, help="Subsample stride for fast testing (default: 1)")
    args = parser.parse_args()

    logger.info("Loading Treasury yield data...")
    yield_df, _ = load_yield_panel()
    if args.subsample > 1:
        yield_df = yield_df.iloc[::args.subsample].reset_index(drop=True)
        logger.info("Subsampled to %d dates (stride %d)", len(yield_df), args.subsample)

    logger.info("Running comparative state-space evaluation across Point MLE, Kim Filter, and Bayesian MCMC...")
    comparison = compare_all_state_space_models(
        yield_df=yield_df,
        maturities_dict=DEFAULT_MATURITIES,
        n_mcmc_draws=args.mcmc_draws,
        burn_in=args.burn_in,
    )

    print("\n" + "=" * 80)
    print("STATE-SPACE MODEL EXTENSION SCORECARD")
    print("Point MLE Kalman vs. Regime-Switching Kim Filter vs. Bayesian MCMC")
    print("=" * 80)
    for metric, values in comparison["scorecard"].items():
        print(f"\n[{metric}]")
        for k, v in values.items():
            print(f"  - {k:25s}: {v}")

    print("\n" + "=" * 80)
    print("STRESS PERIOD CREDIBLE INTERVAL WIDTH ANALYSIS (95% CI in bp)")
    print("=" * 80)
    for period, stats_dict in comparison["stress_analysis"].items():
        print(f"\nPeriod: {period.upper()}")
        for factor, width_bp in stats_dict.items():
            print(f"  - {factor:25s}: {width_bp:.2f} bps")

    print("\nExecution complete.")


if __name__ == "__main__":
    run_cli()
