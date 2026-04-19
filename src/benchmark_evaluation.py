"""
src/benchmark_evaluation.py
─────────────────────────────────────────────────────────────────────────────
Benchmark Comparison Layer — Quant Research Evaluation Extension
─────────────────────────────────────────────────────────────────────────────
Purpose:
    Compare the predictive accuracy of all model outputs (GARCH, ML, Hybrid)
    against a naive rolling-standard-deviation baseline.

    A model earns its keep only if it beats the baseline on RMSE *and*
    correlation.  This module makes that comparison explicit and returns a
    single structured DataFrame row that can be consumed by any downstream
    reporting layer.

Baselines and Models Evaluated
───────────────────────────────
    GARCH     — from column  GARCH_Vol  (or Forecast_Vol if absent)
    ML        — from column  ML_Vol     (or falls back gracefully)
    Hybrid    — from column  Forecast_Vol  (the combined output)
    Rolling   — 20-day rolling std of past returns × √252  (computed here;
                no look-ahead because we use shift(1) before the comparison)

Metrics
───────
    RMSE        — root mean square error vs Realized_Vol
    Correlation — Pearson correlation vs Realized_Vol (higher = better signal)

Input Contract
──────────────
    hybrid_master_df must contain at minimum:
        • Realized_Vol   (annualised decimal vol proxy)
        • Forecast_Vol   (hybrid output)
    Optional columns used when present:
        • GARCH_Vol
        • ML_Vol
        • Log_Return     (required only if Rolling baseline is desired;
                          falls back to NaN if absent)

Usage:
    from src.benchmark_evaluation import BenchmarkEvaluation
    bench = BenchmarkEvaluation()
    result_df = bench.evaluate(hybrid_master_df)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class BenchmarkEvaluation:
    """
    Compares GARCH, ML, and Hybrid model forecasts against a rolling-std
    naïve baseline, returning RMSE and Correlation for each.

    Parameters
    ----------
    rolling_window : int
        Look-back window (in trading days) used for the rolling-std baseline.
        Default is 20 (one calendar month).
    ann_factor : float
        Annualisation factor applied to daily rolling std.
        Default √252 for daily data.
    """

    def __init__(self, rolling_window: int = 20, ann_factor: float = np.sqrt(252)):
        self.rolling_window = rolling_window
        self.ann_factor = ann_factor

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _safe_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
        """RMSE on the intersection of non-NaN index positions."""
        common = y_true.index.intersection(y_pred.index)
        if len(common) < 5:
            return np.nan
        yt = y_true.loc[common].dropna()
        yp = y_pred.loc[common].reindex(yt.index).dropna()
        aligned_idx = yt.index.intersection(yp.index)
        if len(aligned_idx) < 5:
            return np.nan
        return float(np.sqrt(mean_squared_error(yt.loc[aligned_idx], yp.loc[aligned_idx])))

    @staticmethod
    def _safe_corr(y_true: pd.Series, y_pred: pd.Series) -> float:
        """Pearson correlation on the intersection of non-NaN index positions."""
        common = y_true.index.intersection(y_pred.index)
        if len(common) < 5:
            return np.nan
        yt = y_true.loc[common].dropna()
        yp = y_pred.loc[common].reindex(yt.index).dropna()
        aligned_idx = yt.index.intersection(yp.index)
        if len(aligned_idx) < 5:
            return np.nan
        if np.std(yt.loc[aligned_idx]) == 0 or np.std(yp.loc[aligned_idx]) == 0:
            return 0.0
        return float(np.corrcoef(yt.loc[aligned_idx], yp.loc[aligned_idx])[0, 1])

    def _build_rolling_baseline(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a naïve rolling-std volatility baseline from Log_Return.

        Uses a shift(1) so that the rolling window at time T uses only
        information available at T-1 (no look-ahead).

        Returns an annualised decimal series aligned to df.index.
        Falls back to a NaN series if Log_Return is not present.
        """
        if "Log_Return" not in df.columns:
            print(
                "[BenchmarkEvaluation] Log_Return column not found — "
                "Rolling baseline will be NaN."
            )
            return pd.Series(np.nan, index=df.index, name="Rolling_Baseline")

        rolling_std = (
            df["Log_Return"]
            .shift(1)                            # no look-ahead
            .rolling(self.rolling_window)
            .std()
            * self.ann_factor
        )
        rolling_std.name = "Rolling_Baseline"
        print(
            f"[BenchmarkEvaluation] Rolling baseline computed "
            f"(window={self.rolling_window}, non-NaN={rolling_std.notna().sum()})"
        )
        return rolling_std

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the benchmark comparison on the supplied DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at minimum: Realized_Vol, Forecast_Vol.
            Optionally: GARCH_Vol, ML_Vol, Log_Return.
            The input DataFrame is NOT modified.

        Returns
        -------
        pd.DataFrame — single-row summary with columns:
            RMSE_GARCH, RMSE_ML, RMSE_HYBRID, RMSE_ROLLING
            Corr_GARCH, Corr_ML, Corr_HYBRID
            n_obs
        """
        # Work on a copy — never mutate the caller's data
        data = df.copy()

        # ── Validate minimum required columns ──────────────────────────────
        required = {"Realized_Vol", "Forecast_Vol"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"[BenchmarkEvaluation] Missing required columns: {missing}"
            )

        realized = data["Realized_Vol"].dropna()
        print(
            f"[BenchmarkEvaluation] Evaluating {len(realized)} observations "
            f"against {len(data.columns)} columns."
        )

        # ── Extract model series (fall back to NaN if column absent) ───────
        hybrid_vol = data["Forecast_Vol"]

        garch_vol = (
            data["GARCH_Vol"]
            if "GARCH_Vol" in data.columns
            else pd.Series(np.nan, index=data.index)
        )
        ml_vol = (
            data["ML_Vol"]
            if "ML_Vol" in data.columns
            else pd.Series(np.nan, index=data.index)
        )

        # ── Rolling baseline ────────────────────────────────────────────────
        rolling_vol = self._build_rolling_baseline(data)

        # ── Compute metrics ─────────────────────────────────────────────────
        rmse_garch  = self._safe_rmse(realized, garch_vol)
        rmse_ml     = self._safe_rmse(realized, ml_vol)
        rmse_hybrid = self._safe_rmse(realized, hybrid_vol)
        rmse_roll   = self._safe_rmse(realized, rolling_vol)

        corr_garch  = self._safe_corr(realized, garch_vol)
        corr_ml     = self._safe_corr(realized, ml_vol)
        corr_hybrid = self._safe_corr(realized, hybrid_vol)
        if corr_hybrid < 0:
            print("[WARNING] Hybrid correlation negative → model instability detected")

        # ── Log summary ────────────────────────────────────────────────────
        print(
            f"[BenchmarkEvaluation] RMSE  — GARCH:{rmse_garch:.4f}  "
            f"ML:{rmse_ml:.4f}  Hybrid:{rmse_hybrid:.4f}  Rolling:{rmse_roll:.4f}"
        )
        print(
            f"[BenchmarkEvaluation] Corr  — GARCH:{corr_garch:.4f}  "
            f"ML:{corr_ml:.4f}  Hybrid:{corr_hybrid:.4f}"
        )

        # ── Signal strength classification ─────────────────────────────────
        def _classify_signal(corr: float) -> str:
            if np.isnan(corr):
                return "UNAVAILABLE"
            if corr < 0.30:
                return "WEAK (likely noise-driven)"
            if corr < 0.60:
                return "MODERATE"
            return "STRONG"
        
        best_model = min(
            {
                "GARCH": rmse_garch,
                "ML": rmse_ml,
                "HYBRID": rmse_hybrid,
                "ROLLING": rmse_roll
            },
            key=lambda k: np.inf if np.isnan({
                "GARCH": rmse_garch,
                "ML": rmse_ml,
                "HYBRID": rmse_hybrid,
                "ROLLING": rmse_roll
            }[k]) else {
                "GARCH": rmse_garch,
                "ML": rmse_ml,
                "HYBRID": rmse_hybrid,
                "ROLLING": rmse_roll
            }[k]
        )

        # ── Return structured DataFrame ────────────────────────────────────
        result = pd.DataFrame([{
            "RMSE_GARCH":    rmse_garch,
            "RMSE_ML":       rmse_ml,
            "RMSE_HYBRID":   rmse_hybrid,
            "RMSE_ROLLING":  rmse_roll,
            "Corr_GARCH":    corr_garch,
            "Corr_ML":       corr_ml,
            "Corr_HYBRID":   corr_hybrid,
            "Signal_GARCH":  _classify_signal(corr_garch),
            "Signal_ML":     _classify_signal(corr_ml),
            "Signal_HYBRID": _classify_signal(corr_hybrid),
            "n_obs":         int(realized.notna().sum()),
        }])

        print(
            f"[BenchmarkEvaluation] Signal classification — "
            f"GARCH:{result['Signal_GARCH'].iloc[0]}  "
            f"ML:{result['Signal_ML'].iloc[0]}  "
            f"Hybrid:{result['Signal_HYBRID'].iloc[0]}"
        )
        # STATISTICAL SIGNIFICANCE TEST
        try:
            from scipy.stats import ttest_rel

            common_idx = realized.index.intersection(hybrid_vol.index).intersection(ml_vol.index)

            if len(common_idx) > 50:
                t_stat, p_val = ttest_rel(
                    (realized.loc[common_idx] - hybrid_vol.loc[common_idx]),
                    (realized.loc[common_idx] - ml_vol.loc[common_idx])
                )
            else:
                p_val = np.nan

            result['Hybrid_vs_ML_pvalue'] = p_val

        except Exception as e:
            print(f"[WARNING] Statistical test failed: {e}")
            result['Hybrid_vs_ML_pvalue'] = np.nan

        result['Best_Model'] = best_model
        return result
