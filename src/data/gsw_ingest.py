"""Federal Reserve Gürkaynak-Sack-Wright (GSW) Svensson benchmark dataset ingestion."""

import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data.gap_detector import GapDetector
from src.data.metadata import create_metadata_record

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GSWIngestor:
    """
    Ingests and validates the Federal Reserve GSW zero-coupon and Svensson parameters dataset.
    
    Note: As documented in the architecture, GSW is an evaluation benchmark, not a model input.
    """

    GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"

    def __init__(
        self,
        raw_dir: Path = Path("data/raw/gsw"),
        processed_dir: Path = Path("data/processed"),
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, force: bool = False, max_retries: int = 3) -> Path:
        """Download raw GSW CSV dataset from the Federal Reserve with local caching."""
        out_file = self.raw_dir / "feds200628.csv"
        if out_file.exists() and not force and out_file.stat().st_size > 1_000_000:
            logger.info("Using cached GSW dataset: %s (%d bytes)", out_file, out_file.stat().st_size)
            return out_file

        for attempt in range(1, max_retries + 1):
            try:
                logger.info("Downloading GSW dataset from Fed (attempt %d/%d)...", attempt, max_retries)
                res = subprocess.run(
                    [
                        "curl",
                        "-s",
                        "-L",
                        "--connect-timeout",
                        "20",
                        "--max-time",
                        "90",
                        "-o",
                        str(out_file),
                        self.GSW_URL,
                    ],
                    check=True,
                )
                if out_file.exists() and out_file.stat().st_size > 1_000_000:
                    logger.info("Successfully downloaded GSW file (%d bytes): %s", out_file.stat().st_size, out_file)
                    return out_file
                else:
                    raise ValueError("Downloaded GSW file is missing or too small")
            except Exception as e:
                logger.warning("GSW download attempt %d failed: %s", attempt, e)
                if attempt == max_retries:
                    raise
                time.sleep(3 * attempt)

        return out_file

    def load_clean_dataset(self) -> Tuple[pd.DataFrame, List[str]]:
        """Parse raw Fed CSV, detect header offset, standardize columns and handle sentinels."""
        raw_path = self.download_dataset()

        # Find header index starting with 'Date,'
        skip_rows = 9
        with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
            for idx in range(30):
                line = f.readline()
                if line.startswith("Date,"):
                    skip_rows = idx
                    break

        logger.info("Parsing GSW CSV (header at line %d)...", skip_rows)
        df = pd.read_csv(
            raw_path,
            skiprows=skip_rows,
            na_values=["NA", "-999.99", ""],
            low_memory=False,
        )

        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"Date": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # Identify parameter and yield columns
        param_cols = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
        zero_yield_cols = [f"SVENY{i:02d}" for i in range(1, 31) if f"SVENY{i:02d}" in df.columns]
        par_yield_cols = [f"SVENPY{i:02d}" for i in range(1, 31) if f"SVENPY{i:02d}" in df.columns]

        selected_cols = ["date"] + [c for c in param_cols if c in df.columns] + zero_yield_cols + par_yield_cols
        clean_df = df[selected_cols].copy()

        for c in selected_cols:
            if c != "date":
                clean_df[c] = pd.to_numeric(clean_df[c], errors="coerce")

        return clean_df, zero_yield_cols

    def build_panel(self) -> Tuple[pd.DataFrame, Any]:
        """Build validated GSW benchmark panel with metadata record."""
        df, zero_cols = self.load_clean_dataset()

        # Audit with GapDetector on key benchmark tenors (e.g. 1Y, 5Y, 10Y, 30Y zero yields)
        sample_tenors = [c for c in ["SVENY01", "SVENY05", "SVENY10", "SVENY30"] if c in df.columns]
        detector = GapDetector(df, date_col="date")
        missingness_report = detector.generate_missingness_report(sample_tenors)

        metadata = create_metadata_record(
            source="Federal Reserve Board (GSW: Gürkaynak, Sack, and Wright)",
            series=[c for c in df.columns if c != "date"],
            units="Continuously compounded zero-coupon yields and coupon-equivalent par yields in percent",
            frequency="Business daily",
            first_observation=df["date"].min().strftime("%Y-%m-%d"),
            last_observation=df["date"].max().strftime("%Y-%m-%d"),
            missingness_report=missingness_report,
            transformation="Svensson 6-parameter curve fitting; evaluation benchmark only (not model input)",
            extra={
                "dataset_type": "GSW_benchmark_panel",
                "reference_paper": "Gürkaynak, Sack, and Wright (2007, FEDS 2006-28)",
                "role": "Evaluation benchmark for model grading and tracking error residuals",
            },
        )

        return df, metadata

    def run(self) -> Path:
        """Execute GSW ingestion and export parquet + metadata."""
        df, metadata = self.build_panel()
        output_parquet = self.processed_dir / "gsw_panel.parquet"
        output_meta = self.processed_dir / "gsw_panel.metadata.json"

        df.to_parquet(output_parquet, index=False)
        metadata.save_json(output_meta)
        logger.info("Saved GSW panel to %s (%d rows)", output_parquet, len(df))
        logger.info("Saved GSW metadata to %s", output_meta)
        return output_parquet


if __name__ == "__main__":
    GSWIngestor().run()
