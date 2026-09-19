"""FRED Constant Maturity Treasury (CMT) yield panel ingestion pipeline."""

import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data.gap_detector import KNOWN_DISCONTINUITIES, GapDetector
from src.data.metadata import create_metadata_record

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard 11 Constant Maturity Treasury Tenors on FRED
CMT_SERIES_CONFIG: Dict[str, Dict[str, Any]] = {
    "DGS1MO": {"tenor": "1M", "maturity_years": 1.0 / 12.0, "name": "1-Month Treasury CMT"},
    "DGS3MO": {"tenor": "3M", "maturity_years": 3.0 / 12.0, "name": "3-Month Treasury CMT"},
    "DGS6MO": {"tenor": "6M", "maturity_years": 6.0 / 12.0, "name": "6-Month Treasury CMT"},
    "DGS1": {"tenor": "1Y", "maturity_years": 1.0, "name": "1-Year Treasury CMT"},
    "DGS2": {"tenor": "2Y", "maturity_years": 2.0, "name": "2-Year Treasury CMT"},
    "DGS3": {"tenor": "3Y", "maturity_years": 3.0, "name": "3-Year Treasury CMT"},
    "DGS5": {"tenor": "5Y", "maturity_years": 5.0, "name": "5-Year Treasury CMT"},
    "DGS7": {"tenor": "7Y", "maturity_years": 7.0, "name": "7-Year Treasury CMT"},
    "DGS10": {"tenor": "10Y", "maturity_years": 10.0, "name": "10-Year Treasury CMT"},
    "DGS20": {"tenor": "20Y", "maturity_years": 20.0, "name": "20-Year Treasury CMT"},
    "DGS30": {"tenor": "30Y", "maturity_years": 30.0, "name": "30-Year Treasury CMT"},
}


class FREDIngestor:
    """Ingests, cleans, validates, and stores FRED CMT daily yield panels."""

    BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    def __init__(
        self,
        raw_dir: Path = Path("data/raw/fred"),
        processed_dir: Path = Path("data/processed"),
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_series(
        self, series_id: str, force_download: bool = False, max_retries: int = 3
    ) -> Path:
        """Download raw CSV for a single FRED series with caching and curl fallback."""
        out_file = self.raw_dir / f"{series_id}.csv"
        if out_file.exists() and not force_download and out_file.stat().st_size > 100:
            logger.info("Using cached raw file: %s", out_file)
            return out_file

        url = self.BASE_URL.format(series_id=series_id)

        for attempt in range(1, max_retries + 1):
            try:
                logger.info("Downloading %s from FRED (attempt %d/%d)...", series_id, attempt, max_retries)
                res = subprocess.run(
                    ["curl", "-s", "-L", "--connect-timeout", "15", "--max-time", "45", url],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                content = res.stdout
                if "observation_date" in content:
                    with open(out_file, "w", encoding="utf-8") as f:
                        f.write(content)
                    logger.info("Successfully downloaded and cached: %s (%d bytes)", out_file, len(content))
                    return out_file
                else:
                    raise ValueError(f"Invalid response content for {series_id}: {content[:100]}")
            except Exception as e:
                logger.warning("Download attempt %d failed for %s: %s", attempt, series_id, e)
                if attempt == max_retries:
                    raise
                time.sleep(2 * attempt)

        return out_file

    def load_clean_series(
        self, series_id: str, enforce_statutory_discontinuities: bool = True
    ) -> pd.DataFrame:
        """
        Load, parse, and clean raw FRED series file.
        
        Args:
            series_id: FRED mnemonic (e.g. DGS10, DGS30).
            enforce_statutory_discontinuities: If True, masks periods where Treasury
                statutorily suspended issuance (e.g. DGS30 Feb 2002 - Feb 2006) with NaN,
                preventing reliance on Treasury synthetic/extrapolated estimates.
        """
        csv_path = self.download_series(series_id)
        df = pd.read_csv(csv_path)

        # Standardize column names
        date_col = [c for c in df.columns if "date" in c.lower()][0]
        val_col = [c for c in df.columns if c != date_col][0]

        df = df.rename(columns={date_col: "date", val_col: series_id})
        df["date"] = pd.to_datetime(df["date"])

        # FRED denotes non-trading days/missing with '.'
        df[series_id] = pd.to_numeric(
            df[series_id].replace(".", np.nan), errors="coerce"
        )

        # Enforce statutory suspension windows if requested
        if enforce_statutory_discontinuities and series_id in KNOWN_DISCONTINUITIES:
            for start, end in KNOWN_DISCONTINUITIES[series_id]:
                mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
                num_masked = df.loc[mask, series_id].dropna().count()
                if num_masked > 0:
                    logger.info(
                        "Enforcing statutory discontinuity for %s: set %d extrapolated quotes to NaN between %s and %s",
                        series_id,
                        num_masked,
                        start,
                        end,
                    )
                    df.loc[mask, series_id] = np.nan

        return df[["date", series_id]].sort_values("date").reset_index(drop=True)

    def build_yield_panel(
        self,
        series_ids: Optional[List[str]] = None,
        start_date: Optional[str] = "1962-01-02",
        end_date: Optional[str] = None,
        enforce_statutory_discontinuities: bool = True,
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Build unified, gap-audited Treasury yield panel.
        
        Returns:
            Tuple of (DataFrame, DatasetMetadata).
        """
        if series_ids is None:
            series_ids = list(CMT_SERIES_CONFIG.keys())

        logger.info("Building CMT yield panel for series: %s", series_ids)
        panel_df: Optional[pd.DataFrame] = None

        for sid in series_ids:
            s_df = self.load_clean_series(sid, enforce_statutory_discontinuities=enforce_statutory_discontinuities)
            if panel_df is None:
                panel_df = s_df
            else:
                panel_df = pd.merge(panel_df, s_df, on="date", how="outer")

        assert panel_df is not None
        panel_df = panel_df.sort_values("date").reset_index(drop=True)

        if start_date:
            panel_df = panel_df[panel_df["date"] >= pd.to_datetime(start_date)].reset_index(drop=True)
        if end_date:
            panel_df = panel_df[panel_df["date"] <= pd.to_datetime(end_date)].reset_index(drop=True)

        # Audit with GapDetector
        detector = GapDetector(panel_df, date_col="date")
        for sid in series_ids:
            detector.check_for_silent_forward_fill(sid)

        missingness_report = detector.generate_missingness_report(series_ids)

        metadata = create_metadata_record(
            source="Federal Reserve Bank of St. Louis (FRED) / U.S. Department of the Treasury",
            series=series_ids,
            units="Percent per annum, annualized",
            frequency="Business daily",
            first_observation=panel_df["date"].min().strftime("%Y-%m-%d"),
            last_observation=panel_df["date"].max().strftime("%Y-%m-%d"),
            missingness_report=missingness_report,
            transformation=(
                "Raw constant maturity par-equivalent yields in percent; "
                "statutory issuance suspensions (e.g. DGS30 2002-2006, DGS20 1987-1993) strictly masked to NaN"
            ),
            extra={
                "dataset_type": "CMT_yield_panel",
                "maturities_years": {k: v["maturity_years"] for k, v in CMT_SERIES_CONFIG.items() if k in series_ids},
            },
        )

        return panel_df, metadata

    def run(self) -> Path:
        """Execute full ingestion pipeline and write parquet + metadata."""
        panel_df, metadata = self.build_yield_panel()
        output_parquet = self.processed_dir / "yield_panel.parquet"
        output_meta = self.processed_dir / "yield_panel.metadata.json"

        panel_df.to_parquet(output_parquet, index=False)
        metadata.save_json(output_meta)
        logger.info("Saved yield panel to %s (%d rows)", output_parquet, len(panel_df))
        logger.info("Saved metadata to %s", output_meta)
        return output_parquet


if __name__ == "__main__":
    FREDIngestor().run()
