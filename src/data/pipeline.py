"""Unified data ingestion and integrity verification pipeline."""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.data.fred_ingest import FREDIngestor
from src.data.gsw_ingest import GSWIngestor
from src.data.metadata import DatasetMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_yield_panel(
    path: Path = Path("data/processed/yield_panel.parquet"),
) -> Tuple[pd.DataFrame, Optional[DatasetMetadata]]:
    """Load the processed CMT yield panel along with its metadata if available."""
    df = pd.read_parquet(path)
    meta_path = path.with_suffix(".metadata.json")
    metadata = DatasetMetadata.load_json(meta_path) if meta_path.exists() else None
    return df, metadata


def load_gsw_panel(
    path: Path = Path("data/processed/gsw_panel.parquet"),
) -> Tuple[pd.DataFrame, Optional[DatasetMetadata]]:
    """Load the processed GSW benchmark panel along with its metadata if available."""
    df = pd.read_parquet(path)
    meta_path = path.with_suffix(".metadata.json")
    metadata = DatasetMetadata.load_json(meta_path) if meta_path.exists() else None
    return df, metadata


def run_pipeline(include_gsw: bool = True) -> None:
    """Run full ingestion and verification pipeline for MacroRates."""
    logger.info("=== Starting FRED CMT Yield Panel Ingestion ===")
    fred_path = FREDIngestor().run()
    logger.info("FRED Ingestion complete: %s", fred_path)

    if include_gsw:
        logger.info("=== Starting Federal Reserve GSW Benchmark Ingestion ===")
        gsw_path = GSWIngestor().run()
        logger.info("GSW Ingestion complete: %s", gsw_path)

    logger.info("=== Ingestion Pipeline Complete & Validated ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MacroRates Data Ingestion Pipeline")
    parser.add_argument("--skip-gsw", action="store_true", help="Skip GSW dataset download")
    args = parser.parse_args()
    run_pipeline(include_gsw=not args.skip_gsw)
