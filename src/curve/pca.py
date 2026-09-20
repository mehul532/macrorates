"""Principal Component Analysis (PCA) for yield curve term structures."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    """Container for yield curve PCA decomposition results."""

    n_components: int
    maturities: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance_ratio: np.ndarray
    loadings: np.ndarray  # Shape: (n_maturities, n_components)
    mean_vector: np.ndarray
    scores: Optional[pd.DataFrame] = None
    on_changes: bool = False

    def variance_explained_summary(self) -> Dict[str, float]:
        """Return R^2_k for k=1, 2, 3."""
        r2 = {}
        for k in range(1, min(self.n_components + 1, 4)):
            r2[f"R2_{k}"] = float(self.cumulative_variance_ratio[k - 1])
        return r2


class YieldCurvePCA:
    """
    Extracts statistical Level, Slope, and Curvature from yield curves.
    
    Supports PCA on raw yield levels and yield changes (daily diffs),
    with automated economic sign alignment:
      - PC1 (Level): Positive loadings across all maturities.
      - PC2 (Slope): Monotonically increasing from short to long (steeper curve = positive score).
      - PC3 (Curvature): Peak loading in intermediate belly tenors (2Y-5Y).
    """

    def __init__(self, n_components: int = 3):
        self.n_components = n_components

    def fit(
        self,
        yield_df: pd.DataFrame,
        maturities_dict: Dict[str, float],
        on_changes: bool = False,
        date_col: str = "date",
    ) -> PCAResult:
        """
        Fit PCA to yield curve panel.
        
        Args:
            yield_df: DataFrame with date and yield columns.
            maturities_dict: Mapping of column name to maturity in years (e.g. {'DGS2': 2.0}).
            on_changes: If True, computes PCA on first-differences (yield changes).
            date_col: Column name containing dates.
        """
        cols = [c for c in yield_df.columns if c in maturities_dict]
        # Sort columns by ascending maturity
        cols = sorted(cols, key=lambda c: maturities_dict[c])
        maturities = np.array([maturities_dict[c] for c in cols])

        df_clean = yield_df[[date_col] + cols].dropna().copy()
        dates = df_clean[date_col].values

        Y = df_clean[cols].values

        if on_changes:
            # Yield changes: Delta y_t = y_t - y_{t-1}
            Y = np.diff(Y, axis=0)
            dates = dates[1:]

        mean_vector = np.mean(Y, axis=0)
        Y_centered = Y - mean_vector

        # SVD: Y_centered = U * S * Vt
        U, S, Vt = np.linalg.svd(Y_centered, full_matrices=False)
        eigenvalues = (S ** 2) / (len(Y_centered) - 1)
        total_variance = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        V = Vt.T[:, :self.n_components]  # Shape: (n_maturities, n_components)
        scores = Y_centered @ V  # Shape: (T, n_components)

        # Economic Sign Alignment:
        # PC1: Level -> overall mean loading should be positive
        if np.sum(V[:, 0]) < 0:
            V[:, 0] *= -1
            scores[:, 0] *= -1

        # PC2: Slope -> Long-term loading minus short-term loading should be positive
        # (higher score = steeper yield curve)
        if self.n_components >= 2:
            if V[-1, 1] - V[0, 1] < 0:
                V[:, 1] *= -1
                scores[:, 1] *= -1

        # PC3: Curvature -> Butterfly loading (belly > wings)
        if self.n_components >= 3:
            # Find index closest to 3-5 years (belly)
            belly_idx = np.argmin(np.abs(maturities - 3.0))
            wing_avg = 0.5 * (V[0, 2] + V[-1, 2])
            if V[belly_idx, 2] < wing_avg:
                V[:, 2] *= -1
                scores[:, 2] *= -1

        score_cols = [f"PC{i+1}" for i in range(self.n_components)]
        scores_df = pd.DataFrame(scores, columns=score_cols)
        scores_df.insert(0, "date", dates)

        return PCAResult(
            n_components=self.n_components,
            maturities=maturities,
            eigenvalues=eigenvalues[:self.n_components],
            explained_variance_ratio=explained_variance_ratio[:self.n_components],
            cumulative_variance_ratio=cumulative_variance_ratio[:self.n_components],
            loadings=V,
            mean_vector=mean_vector,
            scores=scores_df,
            on_changes=on_changes,
        )


class PCAVARForecaster:
    """
    Rolling one-step PCA / VAR(1) term structure forecaster.
    
    RESEARCH INTEGRITY & INFORMATION CONTRACT:
    - Fits PCA mean vector, loadings V, and VAR(1) strictly on training yields y_train.
    - Sequential OOS propagation:
      At step k, forecasts score z_hat_k from z_{k-1}, reconstructs curve y_hat_k = mu + z_hat_k @ V.T.
      Then observes y_test[k], computes z_k = (y_test[k] - mu) @ V.
    - Fixed parameters throughout the evaluation fold.
    """
    
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.pca_model = YieldCurvePCA(n_components=n_components)
        self.pca_res_: Optional[PCAResult] = None
        self.var_intercept_: Optional[np.ndarray] = None
        self.var_A_: Optional[np.ndarray] = None
        self.last_train_score_: Optional[np.ndarray] = None

    def fit(self, y_train_df: pd.DataFrame, maturities_dict: Dict[str, float], date_col: str = "date") -> "PCAVARForecaster":
        """
        Fit PCA decomposition and VAR(1) dynamics strictly on training observations.
        """
        self.pca_res_ = self.pca_model.fit(y_train_df, maturities_dict=maturities_dict, date_col=date_col)
        scores = self.pca_res_.scores.drop(columns=[date_col]).values
        self.last_train_score_ = scores[-1].copy()
        
        # Fit VAR(1) on training PCA scores: z_t = c + A z_{t-1} + e_t
        X_lag = scores[:-1]
        Y_lead = scores[1:]
        X_design = np.column_stack([np.ones(len(X_lag)), X_lag])
        params, _, _, _ = np.linalg.lstsq(X_design, Y_lead, rcond=None)
        
        self.var_intercept_ = params[0]      # Shape (n_components,)
        A_mat = params[1:].T                 # Shape (n_components, n_components)
        eigvals = np.linalg.eigvals(A_mat)
        max_eig = float(np.max(np.abs(eigvals)))
        if max_eig >= 0.999:
            A_mat = A_mat * (0.995 / max_eig)
        self.var_A_ = A_mat
        self.train_scores_ = scores
        return self

    def sequential_predict_and_update(self, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sequentially forecast 1-step curve and PCA scores across test observations.
        
        Returns:
          y_pred: (K, N) predicted yield curves
          scores_pred: (K, n_components) 1-step predicted PCA scores
          scores_obs: (K, n_components) observed PCA scores after projecting y_test
        """
        if self.pca_res_ is None or self.var_intercept_ is None or self.last_train_score_ is None:
            raise ValueError("Forecaster must be fit before forecasting.")
            
        K, N = y_test.shape
        y_pred = np.zeros((K, N))
        scores_pred = np.zeros((K, self.n_components))
        scores_obs = np.zeros((K, self.n_components))
        
        z_curr = self.last_train_score_.copy()
        mu = self.pca_res_.mean_vector
        V = self.pca_res_.loadings
        
        for k in range(K):
            # 1. 1-step forecast from information through k-1
            z_p = self.var_intercept_ + self.var_A_ @ z_curr
            y_p = mu + z_p @ V.T
            
            y_pred[k] = y_p
            scores_pred[k] = z_p
            
            # 2. Observe y_test[k] and project onto frozen training PCA loadings
            y_k = y_test[k]
            mask = ~np.isnan(y_k)
            if np.all(mask):
                z_curr = (y_k - mu) @ V
            else:
                y_fill = np.where(mask, y_k, y_p)
                z_curr = (y_fill - mu) @ V
                
            scores_obs[k] = z_curr
            
        return y_pred, scores_pred, scores_obs

