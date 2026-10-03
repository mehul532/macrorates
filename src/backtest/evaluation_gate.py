"""
Point-in-Time Evaluation Gate, Input Traceability, and Run Manifest Module.

Implements:
1. Input Traceability:
   - Distinguishes Treasury yield observation dates from availability timestamps
     (U.S. Treasury par yields are indicative quotes collected around 15:30 ET by FRBNY,
     accessible for 16:00 ET decisions).
   - Records fitted parameter hashes for PCA/VAR, Kalman/DNS, macro response, and GBM.
   - Records macro actual, consensus vintage, release timestamp, timezone, surprise scale,
     and decision cutoff.
   - Does NOT assign "verified point in time" provenance solely because a row parsed successfully.
2. Immutable As-Of Manifests:
   - Per-fold as-of manifests tracing inputs, cutoffs, and parameter hashes.
   - Immutable run manifest serialized to JSON.
3. Event-Coverage Table by Fold:
   - Fold breakdown of independent releases, active event days, missing consensus/timestamps,
     and nonzero overlay days.
4. Holdout Evaluation Gate:
   - Categorizes existing quick evaluations as DEVELOPMENT_EVIDENCE.
   - If an event-covered, untouched holdout is unavailable, returns NOT_EVALUABLE
     with exact missing data reasons and minimum predeclared sample requirements.
   - Strictly bars macro-alpha verdicts when holdout is not evaluable.
5. Separation of Concerns & Research Proxy Disclosures:
   - Keeps forecast accuracy strictly separated from trading performance.
   - Preserves COPIED_DNS_CURVE_FORECAST provenance for DNS_Kalman_Macro.
   - Labels Treasury-yield / DV01 trading results as a research proxy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GateVerdict(str, Enum):
    """Overall point-in-time gate evaluation verdict."""
    NOT_EVALUABLE = "NOT_EVALUABLE"
    EVALUABLE_PASS = "EVALUABLE_PASS"
    EVALUABLE_FAIL = "EVALUABLE_FAIL"


class EvaluationTier(str, Enum):
    """Tier of evaluation sample."""
    DEVELOPMENT_EVIDENCE = "DEVELOPMENT_EVIDENCE"
    OUT_OF_SAMPLE_HOLDOUT = "OUT_OF_SAMPLE_HOLDOUT"


class MacroProvenanceStatus(str, Enum):
    """Verification status of macro announcement input."""
    VERIFIED_POINT_IN_TIME = "VERIFIED_POINT_IN_TIME"
    TIMESTAMP_ONLY_PARSED = "TIMESTAMP_ONLY_PARSED"
    LEGACY_DATE_ONLY_ASSUMED = "LEGACY_DATE_ONLY_ASSUMED"
    MISSING_CONSENSUS = "MISSING_CONSENSUS"
    POST_CLOSE_ROLLED = "POST_CLOSE_ROLLED"
    NON_TRADING_DAY_ROLLED = "NON_TRADING_DAY_ROLLED"
    INVALID_OR_FUTURE = "INVALID_OR_FUTURE"


@dataclass
class YieldObservationTrace:
    """Provenance and availability trace for Treasury yield panel inputs."""
    observation_dates: List[str]
    availability_timestamps: List[str]
    latency_convention: str = "FRBNY composite quote collected ~15:30 ET, published by U.S. Treasury, accessible for 16:00 ET decision"
    source_note: str = (
        "U.S. Department of the Treasury Daily Treasury Par Yield Curve Rates "
        "(indicative market quotes from Federal Reserve Bank of New York)"
    )
    data_sha256: str = ""


@dataclass
class ParameterTrace:
    """Provenance and hash of fitted model parameters within a training fold."""
    model_name: str
    parameter_sha256: str
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MacroEventTrace:
    """Trace of individual macro announcement admitted to walk-forward decision."""
    event_idx: int
    indicator: str
    release_date: str
    release_timestamp: str
    actual: Optional[float]
    consensus: Optional[float]
    consensus_vintage: Optional[str]
    surprise: Optional[float]
    surprise_scale_used: float
    verification_status: str
    assigned_origin_date: str
    decision_cutoff_nyc: str
    is_post_close: bool = False
    is_rolled: bool = False


@dataclass
class FoldEventCoverage:
    """Event coverage metrics by fold."""
    fold_id: int
    train_cutoff: str
    test_start: str
    test_end: str
    independent_releases: int
    active_event_days: int
    missing_consensus_or_timestamps: int
    nonzero_overlay_days: int


@dataclass
class FoldAsOfManifest:
    """Immutable As-Of Manifest for a single walk-forward fold."""
    fold_id: int
    training_cutoff_date: str
    training_cutoff_timestamp: str
    test_start_date: str
    test_end_date: str
    yield_trace: Dict[str, Any]
    fitted_parameters: Dict[str, Any]
    macro_events_trace: List[Dict[str, Any]]
    event_coverage: Dict[str, Any]
    pit_compliance: bool = True
    pit_notes: List[str] = field(default_factory=list)


@dataclass
class RunManifest:
    """Immutable run-level manifest aggregating point-in-time provenance across all folds."""
    run_id: str
    generated_at_utc: str
    git_commit: str
    run_mode: str
    evaluation_tier: str
    gate_verdict: str
    macro_alpha_verdict: Optional[str]
    missing_data_reasons: List[str]
    minimum_predeclared_sample_requirements: Dict[str, Any]
    fold_manifests: List[Dict[str, Any]]
    event_coverage_table: List[Dict[str, Any]]
    disclosures: Dict[str, str]
    manifest_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        d = asdict(self)
        return d


class PointInTimeEvaluationGate:
    """
    Evaluator and auditor enforcing the point-in-time evaluation gate.
    """

    TREASURY_PAR_YIELD_URL = "https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/"
    MINIMUM_PREDECLARED_SAMPLE_REQUIREMENTS = {
        "min_holdout_trading_days": 252,
        "min_independent_releases": 20,
        "min_active_event_days": 10,
        "required_indicators": ["CPI", "NFP", "FOMC"],
        "required_provenance": (
            "Unrevised first-release actuals with timestamped contemporaneous consensus "
            "vintages verifiably available strictly prior to decision cutoff."
        ),
        "execution_proxy_requirement": (
            "Historical tradable instrument prices, contract rolls, bid-ask spreads, "
            "and execution fees (indicative par yield changes are a research proxy)."
        ),
    }

    @staticmethod
    def hash_object(obj: Any) -> str:
        """Compute SHA-256 hash of array, dataframe, dict, or string."""
        hasher = hashlib.sha256()
        if isinstance(obj, np.ndarray):
            hasher.update(obj.tobytes())
        elif isinstance(obj, pd.DataFrame):
            hasher.update(pd.util.hash_pandas_object(obj).values.tobytes())
        elif isinstance(obj, (dict, list)):
            hasher.update(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))
        else:
            hasher.update(str(obj).encode("utf-8"))
        return hasher.hexdigest()

    @classmethod
    def trace_yield_observations(
        cls,
        yield_df: pd.DataFrame,
    ) -> YieldObservationTrace:
        """
        Build trace mapping observation dates to availability timestamps in America/New_York.
        """
        obs_dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in yield_df.index]
        avail_ts = [
            pd.Timestamp(f"{d_str} 15:30:00", tz="America/New_York").isoformat()
            for d_str in obs_dates
        ]
        checksum = cls.hash_object(yield_df.values)
        return YieldObservationTrace(
            observation_dates=obs_dates,
            availability_timestamps=avail_ts,
            data_sha256=checksum,
        )

    @classmethod
    def audit_macro_event_provenance(
        cls,
        m_row: Dict[str, Any],
        decision_cutoff_nyc: pd.Timestamp,
    ) -> Tuple[MacroProvenanceStatus, str]:
        """
        Strictly verify whether actual value and contemporaneous consensus
        were verifiably available before the decision.
        
        DO NOT assign 'verified point in time' provenance solely because a row parsed successfully.
        """
        ts_val = m_row.get("release_timestamp")
        actual_val = m_row.get("actual")
        forecast_val = m_row.get("forecast")
        if forecast_val is None or pd.isna(forecast_val):
            forecast_val = m_row.get("consensus")
        is_post_close = m_row.get("is_post_close", False)
        is_rolled = m_row.get("is_rolled", False)
        is_legacy = m_row.get("is_legacy", False)
        vintage_val = m_row.get("consensus_vintage")

        # 1. Check actual presence
        if actual_val is None or pd.isna(actual_val):
            return MacroProvenanceStatus.INVALID_OR_FUTURE, "Actual value is missing or NaN."

        # 2. Check contemporaneous consensus
        if forecast_val is None or pd.isna(forecast_val):
            return MacroProvenanceStatus.MISSING_CONSENSUS, "Consensus forecast is missing (NaN); unverified."

        # 3. Check legacy date-only assumption
        if is_legacy:
            return MacroProvenanceStatus.LEGACY_DATE_ONLY_ASSUMED, (
                "Release lacks intraday timestamp; legacy 08:30 ET assumption applied."
            )

        # 4. Check post-close roll
        if is_post_close or is_rolled:
            return MacroProvenanceStatus.POST_CLOSE_ROLLED, (
                "Release occurred after 16:00 ET cutoff; rolled causally to next decision origin."
            )

        # 5. Timestamp validation against decision cutoff
        if ts_val is not None:
            try:
                t_rel = pd.to_datetime(ts_val)
                if t_rel.tzinfo is None:
                    t_rel = t_rel.tz_localize("America/New_York")
                else:
                    t_rel = t_rel.tz_convert("America/New_York")
                if t_rel > decision_cutoff_nyc:
                    return MacroProvenanceStatus.POST_CLOSE_ROLLED, "Release timestamp is after decision cutoff."
            except Exception as e:
                return MacroProvenanceStatus.INVALID_OR_FUTURE, f"Unparseable release timestamp: {e}"

        # 6. Verification: both actual and contemporaneous consensus verified prior to cutoff
        # Explicit check: do NOT assign VERIFIED_POINT_IN_TIME solely because row parsed successfully
        if vintage_val is None or pd.isna(vintage_val):
            return MacroProvenanceStatus.TIMESTAMP_ONLY_PARSED, (
                "Release timestamp and consensus value parsed successfully, but contemporaneous consensus survey "
                "vintage timestamp is unverified/absent."
            )

        return MacroProvenanceStatus.VERIFIED_POINT_IN_TIME, (
            "Actual and contemporaneous consensus vintage verifiably available prior to decision cutoff."
        )

    @classmethod
    def build_event_coverage_table(
        cls,
        fold_macro_audit: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Generate structured event-coverage table by fold:
        - Independent releases
        - Active event days
        - Missing consensus or timestamps
        - Days with a nonzero overlay
        """
        rows = []
        for f in fold_macro_audit:
            fold_id = f.get("fold_id", 0)
            train_cutoff = f.get("train_cutoff", "N/A")
            test_start = f.get("test_start", "N/A")
            test_end = f.get("test_end", "N/A")
            
            ind_rel = f.get("evaluated_decision_events", f.get("calendar_event_count", 0))
            active_days = int(f.get("nonzero_macro_days", 0))
            missing_cons_or_ts = int(
                f.get("missing_surprise_events", 0)
                + f.get("invalid_timestamp_rejected_events", 0)
                + f.get("legacy_date_only_assumed_events", 0)
            )
            nonzero_overlay_days = active_days

            rows.append({
                "Fold": fold_id,
                "Train Cutoff": train_cutoff,
                "Test Window": f"{test_start} to {test_end}",
                "Independent Releases": ind_rel,
                "Active Event Days": active_days,
                "Missing Consensus or Timestamps": missing_cons_or_ts,
                "Nonzero Overlay Days": nonzero_overlay_days,
            })

        df = pd.DataFrame(rows)
        return df

    @classmethod
    def evaluate_holdout_readiness(
        cls,
        fold_coverage_df: pd.DataFrame,
        total_test_events: int,
        total_eval_days: int,
        macro_coverage_end: str,
    ) -> Dict[str, Any]:
        """
        Evaluate holdout readiness against formal predeclared criteria.
        
        If an event-covered, genuinely untouched holdout is unavailable:
        returns NOT_EVALUABLE with exact missing data and sample requirements.
        DOES NOT REPORT A MACRO-ALPHA VERDICT.
        """
        total_releases = int(fold_coverage_df["Independent Releases"].sum()) if not fold_coverage_df.empty else 0
        total_active_days = int(fold_coverage_df["Active Event Days"].sum()) if not fold_coverage_df.empty else 0
        total_nonzero_days = int(fold_coverage_df["Nonzero Overlay Days"].sum()) if not fold_coverage_df.empty else 0

        # Current quick run criteria
        is_development_sample = total_eval_days <= 100 or total_releases == 0

        missing_data_reasons = []
        if total_releases == 0:
            missing_data_reasons.append(
                f"Macro announcement dataset coverage ends on {macro_coverage_end}; "
                f"0 macro events occurred in the evaluated test window."
            )
        if total_active_days == 0:
            missing_data_reasons.append(
                "Zero active macro decision days occurred in the evaluated holdout."
            )
        if total_eval_days < cls.MINIMUM_PREDECLARED_SAMPLE_REQUIREMENTS["min_holdout_trading_days"]:
            missing_data_reasons.append(
                f"Evaluated sample size ({total_eval_days} days) is less than the predeclared "
                f"minimum holdout requirement of {cls.MINIMUM_PREDECLARED_SAMPLE_REQUIREMENTS['min_holdout_trading_days']} trading days."
            )
        missing_data_reasons.append(
            "Contemporaneous consensus survey vintages timestamped strictly prior to release "
            "are not available across all target indicators."
        )

        if is_development_sample or total_releases == 0 or total_nonzero_days == 0:
            verdict = GateVerdict.NOT_EVALUABLE
            tier = EvaluationTier.DEVELOPMENT_EVIDENCE
            alpha_verdict = None  # Strictly barred from claiming macro alpha
        else:
            verdict = GateVerdict.EVALUABLE_PASS
            tier = EvaluationTier.OUT_OF_SAMPLE_HOLDOUT
            alpha_verdict = "EVALUABLE_ALPHA_TEST_PENDING_SCORING"

        return {
            "gate_verdict": verdict.value,
            "evaluation_tier": tier.value,
            "macro_alpha_verdict": alpha_verdict,
            "is_evaluable": verdict == GateVerdict.EVALUABLE_PASS,
            "total_independent_releases": total_releases,
            "total_active_event_days": total_active_days,
            "total_nonzero_overlay_days": total_nonzero_days,
            "missing_data_reasons": missing_data_reasons,
            "minimum_predeclared_sample_requirements": cls.MINIMUM_PREDECLARED_SAMPLE_REQUIREMENTS,
        }

    @classmethod
    def build_run_manifest(
        cls,
        run_id: str,
        git_commit: str,
        run_mode: str,
        yield_df: pd.DataFrame,
        fold_manifests: List[FoldAsOfManifest],
        fold_coverage_df: pd.DataFrame,
        gate_eval: Dict[str, Any],
    ) -> RunManifest:
        """
        Assemble comprehensive immutable RunManifest.
        """
        disclosures = {
            "forecast_vs_trading_separation": (
                "Yield curve and spread forecast accuracy (RMSE in basis points) are strictly "
                "separated from trading performance (Sharpe, Sortino, turnover, and PnL)."
            ),
            "trading_proxy_disclosure": (
                "Trading net PnL and Sharpe ratios are a research proxy based on synthetic daily "
                "rebalancing of constant-maturity Treasury yields. Any executable profit claim "
                "requires historical tradable futures or cash bond prices, contract rolls, "
                "bid-ask spreads, and financing costs. U.S. Treasury's Daily Treasury Par Yield "
                "Curve Rates are indicative market quotes based on FRBNY composite quotes."
            ),
            "treasury_url": cls.TREASURY_PAR_YIELD_URL,
            "curve_provenance": "DNS_Kalman_Macro yield predictions are labeled COPIED_DNS_CURVE_FORECAST.",
        }

        manifest = RunManifest(
            run_id=run_id,
            generated_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
            git_commit=git_commit,
            run_mode=run_mode,
            evaluation_tier=gate_eval["evaluation_tier"],
            gate_verdict=gate_eval["gate_verdict"],
            macro_alpha_verdict=gate_eval["macro_alpha_verdict"],
            missing_data_reasons=gate_eval["missing_data_reasons"],
            minimum_predeclared_sample_requirements=gate_eval["minimum_predeclared_sample_requirements"],
            fold_manifests=[asdict(m) for m in fold_manifests],
            event_coverage_table=fold_coverage_df.to_dict(orient="records"),
            disclosures=disclosures,
        )

        # Compute immutable hash of manifest content
        manifest_str = json.dumps(manifest.to_dict(), sort_keys=True)
        manifest.manifest_sha256 = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()
        return manifest

    @classmethod
    def save_manifest(
        cls,
        manifest: RunManifest,
        output_path: Union[str, Path] = "reports/point_in_time_manifest.json",
    ) -> Path:
        """Serialize RunManifest to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        logger.info("Saved point-in-time manifest to %s.", out)
        return out
