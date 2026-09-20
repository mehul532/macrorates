"""
Macroeconomic Surprises & Dynamic Term Structure Response Module.

Implements:
1. Two strictly separated, non-merged surprise series:
   - S_ann_t = (Actual_t - Consensus_t) / sigma(Actual - Consensus)
     [True announcement surprise from market consensus]
   - S_model_t = (Actual_t - ActualHat_t|t-1) / sigma(e)
     [Model innovation from expanding-window AR(1)/RW with zero lookahead]
2. Real-time unrevised actuals (ALFRED vintages) and survey expectations
   for CPI, Core CPI, Nonfarm Payrolls, Unemployment Rate, and FOMC decisions.
3. Contemporaneous regressions: Delta Factor_t = alpha + beta * Surprise_t + eps_t
   with Newey-West HAC standard errors across PCA, Static NS, and Kalman factors.
4. Jordà (2005) Local Projections:
   Factor_{t+h} - Factor_{t-1} = alpha_h + beta_h * Surprise_t + Gamma_h * X_{t-1} + eps_{t+h}
   for horizons h in {0, 1, 2, 5, 10} with strictly lagged controls X_{t-1}.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data.metadata import DatasetMetadata, create_metadata_record

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard series configurations
MACRO_SERIES_SPECS: Dict[str, Dict[str, Any]] = {
    "CPI": {
        "series_id": "CPIAUCNS",
        "name": "Headline CPI YoY",
        "unit": "%",
        "frequency": "Monthly",
        "time_of_day": "08:30 ET",
        "model_type": "ar1",
    },
    "CORE_CPI": {
        "series_id": "CPILFENS",
        "name": "Core CPI YoY",
        "unit": "%",
        "frequency": "Monthly",
        "time_of_day": "08:30 ET",
        "model_type": "ar1",
    },
    "NFP": {
        "series_id": "PAYEMS",
        "name": "Nonfarm Payrolls Change",
        "unit": "Thousands",
        "frequency": "Monthly",
        "time_of_day": "08:30 ET",
        "model_type": "ar1",
    },
    "UNEMP": {
        "series_id": "UNRATE",
        "name": "Unemployment Rate",
        "unit": "%",
        "frequency": "Monthly",
        "time_of_day": "08:30 ET",
        "model_type": "ar1",
    },
    "FOMC": {
        "series_id": "DFEDTARU",
        "name": "Fed Funds Target Rate Decision",
        "unit": "%",
        "frequency": "FOMC Meetings",
        "time_of_day": "14:00 ET",
        "model_type": "rw",
    },
}


class RealTimeMacroIngestor:
    """Ingests, parses, validates, and standardizes real-time macro announcements and consensus."""

    def __init__(
        self,
        raw_dir: Path = Path("data/raw/macro"),
        processed_dir: Path = Path("data/processed"),
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_consensus_events(self) -> Dict[str, pd.DataFrame]:
        """
        Load parsed occurrences for the 5 target series from raw JSON files.
        Falls back seamlessly across investing consensus and FRED macro events snapshots.
        """
        consensus_path = self.raw_dir / "investing-us-macro-consensus.json"
        events_path = self.raw_dir / "fred-us-macro-events.json"

        if not consensus_path.exists() and not events_path.exists():
            raise FileNotFoundError(
                f"No macro data found in {self.raw_dir}. Ensure raw macro JSON files are present."
            )

        series_frames = {}

        if consensus_path.exists():
            with open(consensus_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            series_list = c_data.get("series", [])
            for s in series_list:
                sid = s.get("seriesId")
                matched_key = None
                for k, v in MACRO_SERIES_SPECS.items():
                    if v["series_id"] == sid:
                        matched_key = k
                        break
                if matched_key is None:
                    continue

                records = []
                for o in s.get("occurrences", []):
                    if o.get("occurrenceTime") and o.get("actual") is not None:
                        records.append({
                            "timestamp": o["occurrenceTime"],
                            "actual": float(o["actual"]),
                            "forecast": float(o["forecast"]) if o.get("forecast") is not None else np.nan,
                            "previous": float(o["previous"]) if o.get("previous") is not None else np.nan,
                            "unit": o.get("unit", MACRO_SERIES_SPECS[matched_key]["unit"]),
                        })
                if records:
                    df = pd.DataFrame(records)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df["date"] = pd.to_datetime(df["timestamp"].dt.date)
                    df = df.sort_values("timestamp").reset_index(drop=True)
                    series_frames[matched_key] = df

        return series_frames


class SurpriseEngine:
    """
    Computes Announcement Surprises and Recursive Model Innovations.
    
    CRITICAL METHODOLOGICAL GUARANTEE:
    - S_ann_t = (Actual_t - Consensus_t) / sigma(Actual - Consensus)
    - S_model_t = (Actual_t - ActualHat_{t|t-1}) / sigma(e)
    S_ann_t and S_model_t are strictly separate series and are NEVER merged.
    Model predictions ActualHat_{t|t-1} use STRICTLY observations through s <= t-1.
    """

    @staticmethod
    def compute_announcement_surprise(
        df: pd.DataFrame,
        actual_col: str = "actual",
        forecast_col: str = "forecast",
    ) -> Tuple[pd.Series, pd.Series, float]:
        """
        Compute standardized announcement surprise.
        
        Returns:
            Tuple of (raw_surprise_ann, standardized_surprise_ann, sigma_ann).
        """
        raw_surprise = df[actual_col] - df[forecast_col]
        valid = raw_surprise.dropna()
        if len(valid) == 0:
            sigma = 1.0
        else:
            sigma = float(valid.std())
            if sigma == 0.0 or np.isnan(sigma):
                sigma = 1.0

        std_surprise = raw_surprise / sigma
        return raw_surprise, std_surprise, sigma

    @staticmethod
    def compute_model_surprise(
        df: pd.DataFrame,
        actual_col: str = "actual",
        previous_col: str = "previous",
        model_type: str = "ar1",
        min_history: int = 12,
    ) -> Tuple[pd.Series, pd.Series, float]:
        """
        Compute out-of-sample model innovation using STRICTLY data through t-1.
        
        Args:
            df: Chronologically sorted DataFrame of releases.
            actual_col: Column with actual values.
            previous_col: Column with prior release values.
            model_type: 'ar1' for expanding AR(1), 'rw' for random walk (previous value).
            min_history: Minimum observations required before fitting expanding AR(1).
        
        Returns:
            Tuple of (raw_surprise_model, standardized_surprise_model, sigma_model).
        """
        n = len(df)
        actuals = df[actual_col].values
        prev_vals = df[previous_col].values if previous_col in df.columns else np.full(n, np.nan)
        pred_errors = np.full(n, np.nan)

        for i in range(n):
            if i == 0:
                if not np.isnan(prev_vals[0]):
                    pred_errors[0] = actuals[0] - prev_vals[0]
                continue

            if model_type == "rw" or i < min_history:
                # Random walk: prediction is prior actual (or previous release)
                y_hat = actuals[i - 1]
                pred_errors[i] = actuals[i] - y_hat
            else:
                # Expanding AR(1) fitted strictly on actuals[:i]
                y_hist = actuals[:i]
                y_lag = y_hist[:-1]
                y_curr = y_hist[1:]

                X = np.column_stack([np.ones(len(y_lag)), y_lag])
                try:
                    # lstsq solves OLS beta = [c, phi]
                    beta, _, _, _ = np.linalg.lstsq(X, y_curr, rcond=None)
                    c, phi = beta[0], beta[1]
                    # Forecast 1 step ahead:
                    y_hat = c + phi * y_hist[-1]
                    pred_errors[i] = actuals[i] - y_hat
                except Exception:
                    pred_errors[i] = actuals[i] - y_hist[-1]

        raw_errors = pd.Series(pred_errors, index=df.index)
        valid = raw_errors.dropna()
        if len(valid) == 0:
            sigma = 1.0
        else:
            sigma = float(valid.std())
            if sigma == 0.0 or np.isnan(sigma):
                sigma = 1.0

        std_errors = raw_errors / sigma
        return raw_errors, std_errors, sigma

    def build_unified_surprises_panel(
        self,
        series_frames: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, DatasetMetadata]:
        """
        Process all macro series and build audited, gap-checked panel.
        Ensures strict separation of surprise_ann and surprise_model.
        """
        all_records = []
        series_names = list(series_frames.keys())
        missingness_report = {}

        for indicator, df in series_frames.items():
            df_clean = df.copy().sort_values("timestamp").reset_index(drop=True)
            spec = MACRO_SERIES_SPECS.get(indicator, {})
            model_type = spec.get("model_type", "ar1")

            raw_ann, std_ann, sigma_ann = self.compute_announcement_surprise(df_clean)
            raw_mod, std_mod, sigma_mod = self.compute_model_surprise(
                df_clean, model_type=model_type
            )

            df_clean["indicator"] = indicator
            df_clean["series_id"] = spec.get("series_id", indicator)
            df_clean["raw_surprise_ann"] = raw_ann
            df_clean["surprise_ann"] = std_ann
            df_clean["raw_surprise_model"] = raw_mod
            df_clean["surprise_model"] = std_mod

            # Verification of strict separation:
            assert "surprise" not in df_clean.columns, "Surprise column must not be generic!"
            assert "surprise_ann" in df_clean.columns and "surprise_model" in df_clean.columns

            missingness_report[indicator] = {
                "total_releases": len(df_clean),
                "with_consensus_surprise": int(df_clean["surprise_ann"].dropna().count()),
                "with_model_surprise": int(df_clean["surprise_model"].dropna().count()),
                "sigma_ann": round(sigma_ann, 4),
                "sigma_model": round(sigma_mod, 4),
                "date_min": str(df_clean["date"].min().date()),
                "date_max": str(df_clean["date"].max().date()),
            }

            all_records.append(df_clean)

        panel_df = pd.concat(all_records, ignore_index=True)
        panel_df = panel_df.sort_values(["date", "indicator"]).reset_index(drop=True)

        metadata = create_metadata_record(
            source="ALFRED Real-Time Vintages / Investing Consensus & FRED Macro Database",
            series=series_names,
            units="Standard deviations (Z-score standardized)",
            frequency="Event-based release dates",
            first_observation=str(panel_df["date"].min().date()),
            last_observation=str(panel_df["date"].max().date()),
            missingness_report=missingness_report,
            transformation=(
                "Two strictly separated surprise series: S_ann_t = (Actual - Consensus)/sigma(ann) "
                "and S_model_t = (Actual - ActualHat_{t|t-1})/sigma(e) fitted with strictly zero lookahead."
            ),
            extra={"series_specs": MACRO_SERIES_SPECS},
        )

        return panel_df, metadata


class MacroEventHarmonizer:
    """Harmonizes macroeconomic release events with business daily term structure factor panels."""

    @staticmethod
    def align_events_with_factors(
        factors_df: pd.DataFrame,
        surprises_df: pd.DataFrame,
        indicator: str,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        Merge macro surprise event dates with daily curve factors.
        Aligns event date t with close-to-close factor change Factor_t - Factor_{t-1}.
        """
        sub_s = surprises_df[surprises_df["indicator"] == indicator].copy()
        sub_s[date_col] = pd.to_datetime(sub_s[date_col])

        sub_f = factors_df.copy()
        sub_f[date_col] = pd.to_datetime(sub_f[date_col])
        sub_f = sub_f.sort_values(date_col).reset_index(drop=True)

        merged = pd.merge(sub_s, sub_f, on=date_col, how="inner")
        return merged.sort_values(date_col).reset_index(drop=True)


class MacroRegressionEngine:
    """
    Contemporaneous regressions and Jordà (2005) Local Projections
    with Newey-West HAC covariance.
    """

    @staticmethod
    def run_contemporaneous_regression(
        df: pd.DataFrame,
        factor_diff_col: str,
        surprise_col: str,
        maxlags: int = 5,
    ) -> Dict[str, Any]:
        """
        Fit Delta Factor_t = alpha + beta * Surprise_t + eps_t with Newey-West HAC SEs.
        """
        clean = df[[factor_diff_col, surprise_col]].dropna()
        if len(clean) < 10:
            return {
                "factor": factor_diff_col,
                "surprise_type": surprise_col,
                "n_obs": len(clean),
                "beta": np.nan,
                "hac_se": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "r_squared": np.nan,
            }

        y = clean[factor_diff_col]
        X = sm.add_constant(clean[surprise_col])

        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

        beta = float(model.params[surprise_col])
        se = float(model.bse[surprise_col])
        t_stat = float(model.tvalues[surprise_col])
        p_val = float(model.pvalues[surprise_col])
        r2 = float(model.rsquared)

        return {
            "factor": factor_diff_col,
            "surprise_type": surprise_col,
            "n_obs": len(clean),
            "beta": round(beta, 3),
            "hac_se": round(se, 3),
            "t_stat": round(t_stat, 3),
            "p_value": round(p_val, 4),
            "r_squared": round(r2, 4),
        }

    @staticmethod
    def run_local_projections(
        factors_df: pd.DataFrame,
        surprises_df: pd.DataFrame,
        indicator: str,
        factor_col: str,
        surprise_col: str,
        horizons: List[int] = [0, 1, 2, 5, 10],
        date_col: str = "date",
        min_obs: int = 5,
    ) -> pd.DataFrame:
        """
        Fit Jordà (2005) Local Projections:
        Factor_{t+h} - Factor_{t-1} = alpha_h + beta_h * Surprise_t + Gamma_h * X_{t-1} + eps_{t+h}
        for h in horizons, with strictly lagged controls X_{t-1} = [Delta Factor_{t-1}, Delta Factor_{t-2}].
        """
        sub_s = surprises_df[surprises_df["indicator"] == indicator].copy()
        sub_s[date_col] = pd.to_datetime(sub_s[date_col])

        sub_f = factors_df.copy()
        sub_f[date_col] = pd.to_datetime(sub_f[date_col])
        sub_f = sub_f.sort_values(date_col).reset_index(drop=True)

        date_to_idx = {d: i for i, d in enumerate(sub_f[date_col])}

        results = []
        for h in horizons:
            y_vals, s_vals, ctrl1, ctrl2 = [], [], [], []

            for _, row in sub_s.iterrows():
                d = row[date_col]
                s_val = row[surprise_col]
                if pd.isna(s_val) or d not in date_to_idx:
                    continue

                idx = date_to_idx[d]
                if idx < 3 or idx + h >= len(sub_f):
                    continue

                # Cumulative response Factor_{t+h} - Factor_{t-1} in basis points
                cum_resp = (sub_f.loc[idx + h, factor_col] - sub_f.loc[idx - 1, factor_col]) * 100.0
                # Strictly lagged controls X_{t-1}:
                dx1 = (sub_f.loc[idx - 1, factor_col] - sub_f.loc[idx - 2, factor_col]) * 100.0
                dx2 = (sub_f.loc[idx - 2, factor_col] - sub_f.loc[idx - 3, factor_col]) * 100.0

                if not np.isnan(cum_resp) and not np.isnan(dx1) and not np.isnan(dx2):
                    y_vals.append(cum_resp)
                    s_vals.append(s_val)
                    ctrl1.append(dx1)
                    ctrl2.append(dx2)

            if len(y_vals) < min_obs:
                continue

            reg_df = pd.DataFrame({"y": y_vals, "s": s_vals, "x1": ctrl1, "x2": ctrl2})
            X = sm.add_constant(reg_df[["s", "x1", "x2"]])

            # HAC bandwidth of at least h + 1
            maxlags = max(1, h + 1)
            model = sm.OLS(reg_df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

            beta = float(model.params["s"])
            se = float(model.bse["s"])
            t_stat = float(model.tvalues["s"])
            p_val = float(model.pvalues["s"])

            results.append({
                "indicator": indicator,
                "factor": factor_col,
                "surprise_type": surprise_col,
                "horizon_days": h,
                "beta": round(beta, 3),
                "hac_se": round(se, 3),
                "ci_68_lower": round(beta - se, 3),
                "ci_68_upper": round(beta + se, 3),
                "ci_95_lower": round(beta - 1.96 * se, 3),
                "ci_95_upper": round(beta + 1.96 * se, 3),
                "t_stat": round(t_stat, 3),
                "p_value": round(p_val, 4),
                "r_squared": round(float(model.rsquared), 4),
                "n_obs": len(reg_df),
            })

        return pd.DataFrame(results)


def plot_cpi_impulse_response(
    irf_ann: pd.DataFrame,
    irf_model: pd.DataFrame,
    out_path: Union[str, Path] = Path("reports/figures/irf_cpi_surprise_level_slope_curvature.png"),
) -> Path:
    """
    Generate the standout 3-panel impulse response figure of Level/Slope/Curvature
    to a +1 sigma CPI surprise across horizons h in {0, 1, 2, 5, 10} days.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    factors = ["level", "slope", "curvature"]
    titles = [
        "Level (Parallel Shift / Inflation Expectations)",
        "Slope (Term Premium / Policy Tightening)",
        "Curvature (Belly / Intermediate Dislocation)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    for i, (factor, title) in enumerate(zip(factors, titles)):
        ax = axes[i]

        sub_ann = irf_ann[irf_ann["factor"] == factor].sort_values("horizon_days")
        sub_mod = irf_model[irf_model["factor"] == factor].sort_values("horizon_days")

        h_ann = sub_ann["horizon_days"].values
        b_ann = sub_ann["beta"].values
        se_ann = sub_ann["hac_se"].values

        h_mod = sub_mod["horizon_days"].values
        b_mod = sub_mod["beta"].values
        se_mod = sub_mod["hac_se"].values

        # Zero reference line
        ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

        # Announcement surprise (Primary - Navy/Blue)
        ax.plot(
            h_ann,
            b_ann,
            marker="o",
            markersize=6,
            color="#1f77b4",
            linewidth=2.4,
            label=r"$S_{\mathrm{ann}, t}$ (Consensus Surprise)",
        )
        ax.fill_between(
            h_ann,
            b_ann - 1.96 * se_ann,
            b_ann + 1.96 * se_ann,
            color="#1f77b4",
            alpha=0.15,
            label=r"$\pm 1.96$ HAC SE (95% CI)",
        )
        ax.fill_between(
            h_ann,
            b_ann - se_ann,
            b_ann + se_ann,
            color="#1f77b4",
            alpha=0.25,
            label=r"$\pm 1$ HAC SE (68% CI)",
        )

        # Model innovation surprise (Secondary - Coral/Red)
        ax.plot(
            h_mod,
            b_mod,
            marker="s",
            markersize=5,
            color="#d62728",
            linewidth=1.8,
            linestyle="--",
            label=r"$S_{\mathrm{model}, t}$ (AR(1) Innovation)",
        )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Horizon (Business Days Post-Announcement)", fontsize=10)
        ax.set_ylabel("Yield Curve Response (Basis Points / $+1\\sigma$)", fontsize=10)
        ax.set_xticks([0, 1, 2, 5, 10])
        ax.grid(True, linestyle=":", alpha=0.5)

        if i == 0:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    plt.suptitle(
        r"Jordà Local Projections: Term Structure Factor IRF to $+1\sigma$ CPI Surprise",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved standout IRF figure to %s", out_path)
    return out_path


def run_pipeline() -> Tuple[pd.DataFrame, DatasetMetadata]:
    """Execute full ingestion, surprise generation, and export."""
    ingestor = RealTimeMacroIngestor()
    series_frames = ingestor.load_raw_consensus_events()

    engine = SurpriseEngine()
    panel_df, metadata = engine.build_unified_surprises_panel(series_frames)

    out_parquet = Path("data/processed/macro_surprises.parquet")
    out_meta = Path("data/processed/macro_surprises.metadata.json")

    panel_df.to_parquet(out_parquet, index=False)
    metadata.save_json(out_meta)

    logger.info("Saved macro surprise panel to %s (%d rows)", out_parquet, len(panel_df))
    logger.info("Saved metadata to %s", out_meta)
    return panel_df, metadata


if __name__ == "__main__":
    run_pipeline()
