"""
Dynamic Factor Model (DFM) Macro Nowcast & Term Structure Propagation Module.

Implements:
1. Ingestion and stationarity transformation of the McCracken & Ng FRED-MD
   monthly macroeconomic database (120+ indicators across 8 categories).
2. The Doz, Giannone, and Reichlin (2011, 2012) two-step Dynamic Factor Model:
   - Step 1: PCA on balanced subpanel to obtain factor loadings, idiosyncratic
     variances, and VAR(1) transition matrix.
   - Step 2: Time-varying Kalman filter and Rauch-Tung-Striebel (RTS) smoother
     handling mixed publication lags and ragged edges without ad-hoc zero-filling.
3. Factor identification:
   - Factor 1: Real Activity / Growth (anchored to INDPRO, PAYEMS, RPI > 0).
   - Factor 2: Nominal / Inflation (anchored to CPIAUCSL, PCEPI > 0).
4. Strict zero-lookahead recursive real-time nowcast surprises:
   S^DFM_t = (F_t - F_hat_{t|t-1}) / sigma_{t-1}
   using only information available through t-1.
5. Extended term structure regressions & Jordà (2005) local projections:
   Assessing incremental explanatory power of broad nowcast factors beyond
   individual announcement releases (CPI, NFP, FOMC) on Treasury Level, Slope,
   and Curvature.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

logger = logging.getLogger(__name__)

# Standard 8 McCracken & Ng (2016) series categories
FRED_MD_CATEGORIES = {
    1: "Output and Income",
    2: "Labor Market",
    3: "Housing",
    4: "Consumption, Orders, and Inventories",
    5: "Money and Credit",
    6: "Interest Rates and Spreads",
    7: "Prices",
    8: "Stock Market",
}

# Mapping of prominent series to their group number
SERIES_GROUP_MAP: Dict[str, int] = {
    # Group 1: Output and Income
    "RPI": 1, "W875RX1": 1, "DPCERA3M086SBEA": 1, "CMRMTSPLx": 1, "RETAILx": 1,
    "INDPRO": 1, "IPFPNSS": 1, "IPFINAL": 1, "IPCONGD": 1, "IPDCONGD": 1,
    "IPNCONGD": 1, "IPBUSEQ": 1, "IPMAT": 1, "IPDMAT": 1, "IPNMAT": 1,
    "IPMANSICS": 1, "IPB51222S": 1, "IPFUELS": 1, "CUMFNS": 1,
    # Group 2: Labor Market
    "HWI": 2, "HWIURATIO": 2, "CLF16OV": 2, "CE16OV": 2, "UNRATE": 2,
    "UEMPMEAN": 2, "UEMPLT5": 2, "UEMP5TO14": 2, "UEMP15OV": 2, "UEMP15T26": 2,
    "UEMP27OV": 2, "CLAIMSx": 2, "PAYEMS": 2, "USGOOD": 2, "CES1021000001": 2,
    "USCONS": 2, "MANEMP": 2, "DMANEMP": 2, "NDMANEMP": 2, "SRVPRD": 2,
    "USTPU": 2, "USWTRADE": 2, "USTRADE": 2, "USFIRE": 2, "USGOVT": 2,
    "CES0600000007": 2, "AWOTMAN": 2, "AWHMAN": 2, "CES0600000008": 2,
    "CES2000000008": 2, "CES3000000008": 2,
    # Group 3: Housing
    "HOUST": 3, "HOUSTNE": 3, "HOUSTMW": 3, "HOUSTS": 3, "HOUSTW": 3,
    "PERMIT": 3, "PERMITNE": 3, "PERMITMW": 3, "PERMITS": 3, "PERMITW": 3,
    # Group 4: Consumption, Orders, Inventories
    "DPCERA3M086SBEA": 4, "CMRMTSPLx": 4, "RETAILx": 4, "AMDMNOx": 4,
    "ANDENOx": 4, "AMDMUOx": 4, "BUSINVx": 4, "ISRATIOx": 4, "UMCSENTx": 4,
    # Group 5: Money and Credit
    "M1SL": 5, "M2SL": 5, "M2REAL": 5, "BOGMBASE": 5, "TOTRESNS": 5,
    "NONBORRES": 5, "BUSLOANS": 5, "REALLN": 5, "NONREVSL": 5, "CONSPI": 5,
    "DTCOLNVHFNM": 5, "DTCTHFNM": 5, "INVEST": 5,
    # Group 6: Interest Rates and Spreads
    "FEDFUNDS": 6, "CP3Mx": 6, "TB3MS": 6, "TB6MS": 6, "GS1": 6, "GS5": 6,
    "GS10": 6, "AAA": 6, "BAA": 6, "COMPAPFFx": 6, "TB3SMFFM": 6, "TB6SMFFM": 6,
    "T1YFFM": 6, "T5YFFM": 6, "T10YFFM": 6, "AAAFFM": 6, "BAAFFM": 6,
    # Group 7: Prices
    "WPSFD49207": 7, "WPSFD49502": 7, "WPSID61": 7, "WPSID62": 7, "OILPRICEx": 7,
    "PPICMM": 7, "CPIAUCSL": 7, "CPIAPPSL": 7, "CPITRNSL": 7, "CPIMEDSL": 7,
    "CUSR0000SAC": 7, "CUSR0000SAD": 7, "CUSR0000SAS": 7, "CPIULFSL": 7,
    "CUSR0000SA0L2": 7, "CUSR0000SA0L5": 7, "PCEPI": 7, "DDURRG3M086SBEA": 7,
    "DNDGRG3M086SBEA": 7, "DSERRG3M086SBEA": 7,
    # Group 8: Stock Market
    "S&P 500": 8, "S&P: indust": 8, "S&P div yield": 8, "S&P PE ratio": 8, "VIXCLSx": 8,
}


class FREDMDLoader:
    """
    Ingests and transforms the McCracken & Ng (2016) FRED-MD monthly macroeconomic panel.
    Applies official transformation codes (TCODE 1 to 7) to achieve stationarity,
    screens extreme outliers (> 10 * IQR), and standardizes series.
    """

    def __init__(self, raw_filepath: Union[str, Path] = "data/raw/macro/fred_md_current.csv"):
        self.raw_filepath = Path(raw_filepath)
        self.tcodes: Dict[str, int] = {}
        self.raw_df: Optional[pd.DataFrame] = None
        self.transformed_df: Optional[pd.DataFrame] = None
        self.standardized_df: Optional[pd.DataFrame] = None

    def load_and_transform(
        self,
        max_missing_ratio: float = 0.25,
        outlier_iqr_mult: float = 10.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load CSV, apply McCracken-Ng TCODE transformations, filter outliers,
        and drop series with missingness > max_missing_ratio.

        Returns:
            Tuple of (transformed_df, standardized_df)
        """
        if not self.raw_filepath.exists():
            raise FileNotFoundError(f"FRED-MD file not found at: {self.raw_filepath}")

        df_raw = pd.read_csv(self.raw_filepath)
        self.raw_df = df_raw.copy()

        # Parse transformation codes from row 0
        tcode_row = df_raw.iloc[0, 1:]
        self.tcodes = {
            col: int(float(code))
            for col, code in tcode_row.dropna().items()
            if pd.notna(code) and str(code).strip() != ""
        }

        # Data rows begin at row index 1
        data = df_raw.iloc[1:].copy()
        data["sasdate"] = pd.to_datetime(data["sasdate"], errors="coerce")
        data = data.dropna(subset=["sasdate"]).sort_values("sasdate").reset_index(drop=True)
        data = data.set_index("sasdate")

        # Apply TCODE transformations
        transformed: Dict[str, pd.Series] = {}
        for col, tcode in self.tcodes.items():
            if col not in data.columns:
                continue
            s = pd.to_numeric(data[col], errors="coerce")

            if tcode == 1:
                # Level (no transformation)
                res = s
            elif tcode == 2:
                # First difference
                res = s.diff()
            elif tcode == 3:
                # Second difference
                res = s.diff().diff()
            elif tcode == 4:
                # Natural log
                res = np.log(s.where(s > 0))
            elif tcode == 5:
                # First difference of natural log
                res = np.log(s.where(s > 0)).diff()
            elif tcode == 6:
                # Second difference of natural log
                res = np.log(s.where(s > 0)).diff().diff()
            elif tcode == 7:
                # Percentage change
                res = s.pct_change()
            else:
                res = s
            transformed[col] = res

        df_trans = pd.DataFrame(transformed, index=data.index).iloc[2:].copy()

        # Outlier screening: |x - median| > outlier_iqr_mult * IQR -> NaN
        for col in df_trans.columns:
            series = df_trans[col].dropna()
            if len(series) < 10:
                continue
            med = series.median()
            q75, q25 = series.quantile(0.75), series.quantile(0.25)
            iqr = q75 - q25
            if iqr > 0:
                is_outlier = (df_trans[col] - med).abs() > (outlier_iqr_mult * iqr)
                df_trans.loc[is_outlier, col] = np.nan

        # Screen series exceeding missingness threshold
        missing_ratios = df_trans.isna().mean()
        valid_cols = missing_ratios[missing_ratios <= max_missing_ratio].index.tolist()
        df_trans = df_trans[valid_cols].copy()

        # Standardize (zero mean, unit variance)
        means = df_trans.mean()
        stds = df_trans.std()
        df_std = (df_trans - means) / stds

        self.transformed_df = df_trans
        self.standardized_df = df_std

        logger.info(
            f"Loaded FRED-MD panel: {df_trans.shape[0]} months ({df_trans.index[0].strftime('%Y-%m')} to "
            f"{df_trans.index[-1].strftime('%Y-%m')}), {df_trans.shape[1]} series retained."
        )

        return df_trans, df_std

    def get_ragged_edge_summary(self, n_tail: int = 6) -> pd.DataFrame:
        """Analyze missingness pattern and publication lags across the tail periods."""
        if self.transformed_df is None:
            raise ValueError("Call load_and_transform() first.")

        tail_df = self.transformed_df.iloc[-n_tail:].copy()
        summary = []
        for dt, row in tail_df.iterrows():
            total = len(row)
            missing = row.isna().sum()
            summary.append({
                "date": dt.strftime("%Y-%m"),
                "total_series": total,
                "observed_series": total - missing,
                "missing_series": missing,
                "missing_pct": round(missing / total * 100.0, 1),
            })
        return pd.DataFrame(summary)


@dataclass
class DFMResult:
    """Encapsulates Dynamic Factor Model estimation and extraction results."""
    factors_filtered: pd.DataFrame
    factors_smoothed: pd.DataFrame
    loadings: pd.DataFrame
    eigenvalues: np.ndarray
    variance_explained: np.ndarray
    transition_matrix: np.ndarray
    transition_cov: np.ndarray
    idiosyncratic_var: pd.Series
    r_squared_by_series: pd.Series
    growth_col: str
    inflation_col: str


class DynamicFactorModelDGR:
    """
    Two-Step Dynamic Factor Model (Doz, Giannone, and Reichlin 2011, 2012).

    Step 1:
      Computes initial static factor loadings Lambda and factor dynamics (VAR(1))
      via principal components on the pairwise correlation matrix of standardized data.
    Step 2:
      Applies the Kalman filter and Rauch-Tung-Striebel (RTS) smoother with dynamic
      observation selection matrices W_t to extract optimal latent factors even in the
      presence of mixed publication lags and ragged edges.
    """

    def __init__(self, n_factors: int = 2):
        self.n_factors = n_factors
        self.result_: Optional[DFMResult] = None

    def fit(self, standardized_df: pd.DataFrame) -> DFMResult:
        """
        Fit DFM on standardized macroeconomic panel.

        Args:
            standardized_df: Time x Series standardized macro panel (NaNs allowed)
        Returns:
            DFMResult with smoothed/filtered factors, loadings, and diagnostics.
        """
        Z = standardized_df.copy()
        T, N = Z.shape
        r = self.n_factors

        # Step 1: PCA on pairwise correlation matrix
        corr_matrix = np.nan_to_num(Z.corr().values, nan=0.0)
        # Ensure positive semi-definiteness
        vals, vecs = np.linalg.eigh(corr_matrix)
        vals = np.maximum(vals, 1e-6)
        sort_idx = np.argsort(vals)[::-1]
        vals = vals[sort_idx]
        vecs = vecs[:, sort_idx]

        eigenvals = vals[:10]
        var_explained = (vals / vals.sum() * 100.0)[:10]

        # Initial factor loadings Lambda (N x r)
        Lambda = vecs[:, :r] * np.sqrt(vals[:r])

        # Initial factor trajectory via projection F_init = Z_filled @ W
        # W = (Lambda^T Lambda)^-1 Lambda^T
        W = np.linalg.pinv(Lambda.T @ Lambda) @ Lambda.T
        Z_filled = Z.fillna(0.0).values
        F_init = Z_filled @ W.T

        # VAR(1) transition: F_t = A F_{t-1} + u_t
        F_lag = F_init[:-1]
        F_curr = F_init[1:]
        A = np.linalg.lstsq(F_lag, F_curr, rcond=None)[0].T  # r x r
        # Enforce stationarity on transition matrix (spectral radius < 0.99)
        eigs = np.linalg.eigvals(A)
        max_eig = np.max(np.abs(eigs))
        if max_eig >= 1.0:
            A = A * (0.95 / max_eig)

        u = F_curr - (A @ F_lag.T).T
        Q = np.cov(u.T)
        if r == 1:
            Q = np.array([[float(Q)]])
        else:
            Q = np.atleast_2d(Q)
            # Regularize Q to be strictly positive definite
            Q = Q + np.eye(r) * 1e-4

        # Idiosyncratic noise variances Psi = diag(Var(e_i))
        residuals = Z_filled - F_init @ Lambda.T
        Psi = np.var(residuals, axis=0)
        Psi = np.maximum(Psi, 0.02)  # lower bound to prevent singularity

        # Step 2: Kalman Filter with dynamic observation mask for ragged edges
        f_pred = np.zeros((T, r))
        f_filt = np.zeros((T, r))
        P_pred = np.zeros((T, r, r))
        P_filt = np.zeros((T, r, r))

        # Prior distribution
        f_filt[0] = np.zeros(r)
        P_filt[0] = np.eye(r) * 5.0

        Z_mat = Z.values
        for t in range(1, T):
            # 1. State Prediction
            f_pred[t] = A @ f_filt[t - 1]
            P_pred[t] = A @ P_filt[t - 1] @ A.T + Q

            # 2. Measurement Update with observed indices
            obs_idx = np.where(~np.isnan(Z_mat[t]))[0]
            if len(obs_idx) == 0:
                f_filt[t] = f_pred[t]
                P_filt[t] = P_pred[t]
            else:
                H = Lambda[obs_idx, :]  # m_t x r
                R_inv = 1.0 / Psi[obs_idx]  # m_t diagonal inverse
                y = Z_mat[t, obs_idx]  # m_t x 1

                # Efficient Woodbury update in r x r space:
                # K = (P_pred^-1 + H^T R^-1 H)^-1 H^T R^-1
                P_pred_inv = np.linalg.inv(P_pred[t])
                M = np.linalg.inv(P_pred_inv + (H.T * R_inv) @ H)  # r x r posterior covariance
                K = M @ (H.T * R_inv)  # r x m_t Kalman gain

                v = y - H @ f_pred[t]  # innovation
                f_filt[t] = f_pred[t] + K @ v
                P_filt[t] = M

        # RTS Backward Smoother
        f_smooth = np.zeros_like(f_filt)
        P_smooth = np.zeros_like(P_filt)
        f_smooth[-1] = f_filt[-1]
        P_smooth[-1] = P_filt[-1]

        for t in range(T - 2, -1, -1):
            P_pred_inv = np.linalg.inv(P_pred[t + 1])
            J = P_filt[t] @ A.T @ P_pred_inv
            f_smooth[t] = f_filt[t] + J @ (f_smooth[t + 1] - f_pred[t + 1])
            P_smooth[t] = P_filt[t] + J @ (P_smooth[t + 1] - P_pred[t + 1]) @ J.T

        # Factor Identification: Align Factor 1 to Real Growth and Factor 2 to Inflation
        col_names = Z.columns.tolist()
        lambda_df = pd.DataFrame(Lambda, index=col_names, columns=[f"Factor_{i+1}" for i in range(r)])

        # Real Growth: check loading on INDPRO or PAYEMS
        growth_idx = 0
        if "INDPRO" in col_names:
            growth_idx = 0
            if lambda_df.loc["INDPRO", f"Factor_1"] < 0:
                Lambda[:, 0] *= -1.0
                f_filt[:, 0] *= -1.0
                f_smooth[:, 0] *= -1.0
        elif "PAYEMS" in col_names and lambda_df.loc["PAYEMS", f"Factor_1"] < 0:
            Lambda[:, 0] *= -1.0
            f_filt[:, 0] *= -1.0
            f_smooth[:, 0] *= -1.0

        # Nominal Inflation: check loading on CPIAUCSL or PCEPI
        if r > 1:
            inf_target = "CPIAUCSL" if "CPIAUCSL" in col_names else ("PCEPI" if "PCEPI" in col_names else None)
            if inf_target and lambda_df.loc[inf_target, f"Factor_2"] < 0:
                Lambda[:, 1] *= -1.0
                f_filt[:, 1] *= -1.0
                f_smooth[:, 1] *= -1.0

        factor_labels = ["Growth", "Inflation"] + [f"Factor_{i+1}" for i in range(2, r)]
        df_filt = pd.DataFrame(f_filt, index=Z.index, columns=factor_labels[:r])
        df_smooth = pd.DataFrame(f_smooth, index=Z.index, columns=factor_labels[:r])
        df_loadings = pd.DataFrame(Lambda, index=col_names, columns=factor_labels[:r])

        # Variance explained per series
        r2_series = pd.Series(
            [1.0 - (Psi[i] / np.var(Z_filled[:, i])) for i in range(N)],
            index=col_names,
        ).clip(lower=0.0, upper=1.0)

        self.result_ = DFMResult(
            factors_filtered=df_filt,
            factors_smoothed=df_smooth,
            loadings=df_loadings,
            eigenvalues=eigenvals,
            variance_explained=var_explained,
            transition_matrix=A,
            transition_cov=Q,
            idiosyncratic_var=pd.Series(Psi, index=col_names),
            r_squared_by_series=r2_series,
            growth_col="Growth",
            inflation_col="Inflation" if r > 1 else "Growth",
        )
        return self.result_


class DFMNowcastSurprise:
    """
    Constructs real-time nowcast 'surprise' innovations from DFM latent factors:
      S_t^DFM = (F_t - F_hat_{t|t-1}) / sigma_{t-1}
    using strictly recursive expanding windows to guarantee zero lookahead.
    """

    @staticmethod
    def compute_recursive_surprises(
        factors_df: pd.DataFrame,
        factor_cols: Optional[List[str]] = None,
        burn_in_months: int = 60,
    ) -> pd.DataFrame:
        """
        Estimate AR(1) trend on expanding window [0, t-1] and calculate
        out-of-sample standardized surprise at t.

        Args:
            factors_df: Time series DataFrame of latent factors
            factor_cols: Columns to generate surprises for (default: all numeric)
            burn_in_months: Initial window size (default: 60 months = 5 years)

        Returns:
            DataFrame with surprise series labeled 'surprise_<col>' and trend 'trend_<col>'
        """
        if factor_cols is None:
            factor_cols = factors_df.select_dtypes(include=[np.number]).columns.tolist()

        df_out = pd.DataFrame(index=factors_df.index)
        for col in factor_cols:
            s = factors_df[col].dropna()
            if len(s) <= burn_in_months:
                continue

            surprises = pd.Series(np.nan, index=s.index)
            trends = pd.Series(np.nan, index=s.index)

            vals = s.values
            idx = s.index

            # Recursive expanding window
            for t in range(burn_in_months, len(vals)):
                # History strictly through t-1
                y_hist = vals[1:t]
                x_hist = vals[:t - 1]

                if len(y_hist) < 10:
                    continue

                # AR(1) OLS: F_s = alpha + phi * F_{s-1} + e_s
                X_mat = np.column_stack([np.ones_like(x_hist), x_hist])
                try:
                    params, residuals, rank, _ = np.linalg.lstsq(X_mat, y_hist, rcond=None)
                    alpha, phi = params[0], params[1]
                    # Forecast for t
                    f_pred = alpha + phi * vals[t - 1]
                    # Residual standard error from history up to t-1
                    resid = y_hist - (alpha + phi * x_hist)
                    sigma_e = np.std(resid, ddof=2)
                    if sigma_e > 1e-4:
                        surprise_t = (vals[t] - f_pred) / sigma_e
                        surprises.iloc[t] = surprise_t
                        trends.iloc[t] = f_pred
                except Exception:
                    continue

            df_out[f"surprise_{col}"] = surprises
            df_out[f"trend_{col}"] = trends
            df_out[f"raw_factor_{col}"] = s

        return df_out.dropna(how="all")


class ExtendedMacroCurveRegression:
    """
    Evaluates term structure responses to broad DFM macroeconomic nowcast surprises
    alongside discrete point announcement surprises (CPI, NFP, FOMC).

    Estimates:
    1. Contemporaneous regressions with Newey-West HAC standard errors:
       Delta Factor_t = alpha + sum_k beta_k S_k,t + beta_DFM S_DFM,t + eps_t
       and assesses incremental R^2 and partial F-tests.
    2. Jordà (2005) Local Projections tracing dynamic impulse response functions (IRFs)
       across horizons h in {0, 1, 2, 5, 10, 20} business days.
    """

    @staticmethod
    def align_monthly_surprises_to_trading_days(
        daily_curve_df: pd.DataFrame,
        monthly_surprises_df: pd.DataFrame,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        Aligns monthly DFM surprises with daily Treasury curve factors.
        Maps each monthly surprise to the nearest active trading day.
        """
        daily = daily_curve_df.copy()
        if date_col in daily.columns:
            daily[date_col] = pd.to_datetime(daily[date_col])
            daily = daily.set_index(date_col)
        daily = daily.sort_index()

        monthly = monthly_surprises_df.copy()
        if not isinstance(monthly.index, pd.DatetimeIndex):
            if date_col in monthly.columns:
                monthly[date_col] = pd.to_datetime(monthly[date_col])
                monthly = monthly.set_index(date_col)
            else:
                monthly.index = pd.to_datetime(monthly.index)
        monthly = monthly.sort_index()

        # Find nearest preceding/matching trading day for each monthly observation
        aligned_rows = []
        for m_date, row in monthly.iterrows():
            # Find trading day on or immediately preceding m_date (e.g. month-end)
            trading_days = daily.index[daily.index <= m_date]
            if len(trading_days) == 0:
                continue
            matched_date = trading_days[-1]
            rec = row.to_dict()
            rec["matched_trading_date"] = matched_date
            rec["monthly_date"] = m_date
            aligned_rows.append(rec)

        df_aligned = pd.DataFrame(aligned_rows)
        if not df_aligned.empty:
            df_aligned = df_aligned.set_index("matched_trading_date")
        return df_aligned

    @staticmethod
    def run_augmented_contemporaneous_regression(
        daily_factors: pd.DataFrame,
        macro_surprises: pd.DataFrame,
        dfm_surprises_aligned: pd.DataFrame,
        factor_col: str = "level_change",
        maxlags: int = 5,
    ) -> Dict[str, Any]:
        """
        Fits baseline (announcements only) vs augmented (announcements + DFM)
        contemporaneous regressions:
          Delta Factor_t = alpha + beta_ann * S_ann_t + beta_dfm * S_dfm_t + eps_t

        Returns:
            Dictionary with coefficients, HAC t-stats, p-values, R^2, and incremental R^2.
        """
        daily = daily_factors.copy()
        if "date" in daily.columns:
            daily["date"] = pd.to_datetime(daily["date"])
            daily = daily.set_index("date")

        # Pivot announcement surprises: CPI, NFP, FOMC
        sub_s = macro_surprises.copy()
        if "date" in sub_s.columns:
            sub_s["date"] = pd.to_datetime(sub_s["date"])

        # Create wide announcement surprises on trading days
        ann_wide = sub_s.pivot_table(
            index="date",
            columns="indicator",
            values="surprise_ann",
            aggfunc="mean",
        )

        # Merge daily factors, announcement surprises, and DFM surprises
        merged = daily[[factor_col]].join(ann_wide, how="left")
        merged = merged.join(dfm_surprises_aligned, how="left")

        # Fill announcement NaNs with 0 on non-event days
        for ind in ["CPI", "NFP", "FOMC"]:
            if ind in merged.columns:
                merged[ind] = merged[ind].fillna(0.0)

        # DFM surprise columns
        dfm_cols = [c for c in dfm_surprises_aligned.columns if c.startswith("surprise_")]
        for c in dfm_cols:
            if c in merged.columns:
                merged[c] = merged[c].fillna(0.0)

        ann_cols = [c for c in ["CPI", "NFP", "FOMC"] if c in merged.columns]
        req_cols = [factor_col] + ann_cols + dfm_cols
        clean = merged[req_cols].dropna()

        if len(clean) < 30:
            return {"error": "Insufficient observations"}

        y = clean[factor_col] * 100.0  # basis points

        # 1. Baseline Model (Announcements only)
        X_base = sm.add_constant(clean[ann_cols])
        res_base = sm.OLS(y, X_base).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

        # 2. Augmented Model (Announcements + DFM)
        X_aug = sm.add_constant(clean[ann_cols + dfm_cols])
        res_aug = sm.OLS(y, X_aug).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

        r2_base = float(res_base.rsquared)
        r2_aug = float(res_aug.rsquared)
        delta_r2 = r2_aug - r2_base

        # Partial F-test of joint significance of DFM variables
        try:
            r_matrix = np.zeros((len(dfm_cols), len(res_aug.params)))
            for i, c in enumerate(dfm_cols):
                col_pos = list(X_aug.columns).index(c)
                r_matrix[i, col_pos] = 1.0
            f_test = res_aug.f_test(r_matrix)
            f_stat = float(f_test.fvalue)
            f_pval = float(f_test.pvalue)
        except Exception:
            f_stat = np.nan
            f_pval = np.nan

        summary_params = {}
        for param in res_aug.params.index:
            summary_params[param] = {
                "beta": round(float(res_aug.params[param]), 3),
                "hac_se": round(float(res_aug.bse[param]), 3),
                "t_stat": round(float(res_aug.tvalues[param]), 3),
                "p_value": round(float(res_aug.pvalues[param]), 4),
            }

        return {
            "factor": factor_col,
            "n_obs": len(clean),
            "r2_baseline": round(r2_base, 4),
            "r2_augmented": round(r2_aug, 4),
            "delta_r2": round(delta_r2, 4),
            "f_stat": round(f_stat, 3) if not np.isnan(f_stat) else np.nan,
            "f_pvalue": round(f_pval, 4) if not np.isnan(f_pval) else np.nan,
            "parameters": summary_params,
        }

    @staticmethod
    def run_dfm_local_projections(
        daily_factors: pd.DataFrame,
        dfm_surprises_aligned: pd.DataFrame,
        factor_col: str,
        dfm_surprise_col: str,
        horizons: List[int] = [0, 1, 2, 5, 10, 20],
        min_obs: int = 20,
    ) -> pd.DataFrame:
        """
        Fit Jordà (2005) Local Projections for DFM shocks:
          Factor_{t+h} - Factor_{t-1} = alpha_h + beta_h * S_DFM,t + Gamma_h * X_{t-1} + eps_{t+h}
        with lagged factor changes X_{t-1} as controls.
        """
        daily = daily_factors.copy()
        if "date" in daily.columns:
            daily["date"] = pd.to_datetime(daily["date"])
            daily = daily.sort_values("date").reset_index(drop=True)
        else:
            daily = daily.sort_index().reset_index()
            daily.rename(columns={"index": "date"}, inplace=True)

        date_to_idx = {d: i for i, d in enumerate(daily["date"])}

        surprises = dfm_surprises_aligned.copy()
        if not isinstance(surprises.index, pd.DatetimeIndex):
            surprises.index = pd.to_datetime(surprises.index)

        irf_rows = []
        for h in horizons:
            y_vals, s_vals, ctrl1, ctrl2 = [], [], [], []

            for d, row in surprises.iterrows():
                s_val = row.get(dfm_surprise_col, np.nan)
                if pd.isna(s_val) or d not in date_to_idx:
                    continue

                idx = date_to_idx[d]
                if idx < 3 or idx + h >= len(daily):
                    continue

                # Cumulative factor change in basis points: Factor_{t+h} - Factor_{t-1}
                f_curr = daily.loc[idx + h, factor_col]
                f_prev = daily.loc[idx - 1, factor_col]
                cum_resp = (f_curr - f_prev) * 100.0

                # Lagged controls
                dx1 = (daily.loc[idx - 1, factor_col] - daily.loc[idx - 2, factor_col]) * 100.0
                dx2 = (daily.loc[idx - 2, factor_col] - daily.loc[idx - 3, factor_col]) * 100.0

                if not (np.isnan(cum_resp) or np.isnan(dx1) or np.isnan(dx2)):
                    y_vals.append(cum_resp)
                    s_vals.append(s_val)
                    ctrl1.append(dx1)
                    ctrl2.append(dx2)

            if len(y_vals) < min_obs:
                continue

            reg_df = pd.DataFrame({"y": y_vals, "s": s_vals, "x1": ctrl1, "x2": ctrl2})
            X = sm.add_constant(reg_df[["s", "x1", "x2"]])
            maxlags = max(1, h + 1)
            model = sm.OLS(reg_df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

            beta = float(model.params["s"])
            se = float(model.bse["s"])
            t_val = float(model.tvalues["s"])
            p_val = float(model.pvalues["s"])
            ci_lower = beta - 1.96 * se
            ci_upper = beta + 1.96 * se

            irf_rows.append({
                "horizon": h,
                "factor": factor_col,
                "surprise_series": dfm_surprise_col,
                "beta": round(beta, 3),
                "hac_se": round(se, 3),
                "t_stat": round(t_val, 3),
                "p_value": round(p_val, 4),
                "ci_lower": round(ci_lower, 3),
                "ci_upper": round(ci_upper, 3),
                "n_obs": len(y_vals),
            })

        return pd.DataFrame(irf_rows)
