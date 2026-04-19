"""
src/statistical_tests.py
─────────────────────────────────────────────────────────────────────────────
Statistical Testing Layer — Quant Research Evaluation Extension
─────────────────────────────────────────────────────────────────────────────
Purpose:
    Evaluate the statistical validity of model residuals produced by any
    volatility forecasting model.  Two core tests are implemented:

    1. Ljung-Box Q-test  — detects residual autocorrelation.  A PASS means
       the standardised residuals behave like white noise (no serial structure
       left unexplained by the model).

    2. ARCH-LM test      — detects remaining conditional heteroskedasticity.
       A PASS means the squared residuals show no ARCH effects (the model has
       captured the volatility clustering).

Operates independently.  Consumes model residuals as a plain pandas Series
or numpy array.  Returns a structured result dictionary compatible with the
rest of the pipeline's output schema.

Usage:
    from src.statistical_tests import StatisticalTests
    tester = StatisticalTests()
    results = tester.run(residuals)
"""

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


class StatisticalTests:
    """
    Applies Ljung-Box and ARCH-LM hypothesis tests to model residuals.

    Parameters
    ----------
    lb_lags : int
        Number of lags for the Ljung-Box test (default 10).
    arch_lags : int
        Number of lags for the ARCH-LM test (default 5).
    significance : float
        Significance level for PASS/FAIL classification (default 0.05).
    """

    def __init__(self, lb_lags: int = 10, arch_lags: int = 5, significance: float = 0.05):
        self.lb_lags = lb_lags
        self.arch_lags = arch_lags
        self.significance = significance

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    def _clean(self, residuals) -> np.ndarray:
        """Convert input to a clean 1-D numpy float array, dropping NaNs."""
        if isinstance(residuals, pd.Series):
            arr = residuals.dropna().values.astype(float)
        else:
            arr = np.asarray(residuals, dtype=float)
            arr = arr[~np.isnan(arr)]
        return arr

    def _ljung_box(self, arr: np.ndarray) -> dict:
        """
        Run the Ljung-Box Q-test for residual autocorrelation.

        H₀: No autocorrelation up to lag k.
        Reject H₀ (FAIL) if the minimum p-value across lags < significance.
        We report the p-value at the maximum lag for a single-number summary,
        and flag on the *minimum* p-value across all lags (most conservative).
        """
        try:
            result = acorr_ljungbox(arr, lags=self.lb_lags, return_df=True)
            p_values = result["lb_pvalue"].values
            p_at_max_lag = float(p_values[-1])
            status = "PASS" if p_at_max_lag >= self.significance else "FAIL"

            print(
                f"[StatisticalTests] Ljung-Box: "
                f"p@lag{self.lb_lags}={p_at_max_lag:.4f} → {status}"
            )

            return {
                "LjungBox_p": p_at_max_lag,
                "LjungBox_Result": status,
            }
        except Exception as exc:
            print(f"[StatisticalTests] Ljung-Box test failed: {exc}")
            return {"LjungBox_p": np.nan, "LjungBox_Result": "ERROR"}

    def _arch_lm(self, arr: np.ndarray) -> dict:
        """
        Run the ARCH-LM test for remaining conditional heteroskedasticity.

        H₀: No ARCH effects in residuals.
        Reject H₀ (FAIL) if p-value < significance — means the model has
        NOT fully captured the volatility clustering.
        """
        try:
            lm_stat, lm_pval, f_stat, f_pval = het_arch(arr, nlags=self.arch_lags)
            p = float(lm_pval)
            status = "PASS" if p >= self.significance else "FAIL"
            print(
                f"[StatisticalTests] ARCH-LM: LM_stat={lm_stat:.4f} "
                f"p={p:.4f} → {status}"
            )
            return {
                "ARCH_p": p,
                "ARCH_LM_stat": float(lm_stat),
                "ARCH_Result": status,
            }
        except Exception as exc:
            print(f"[StatisticalTests] ARCH-LM test failed: {exc}")
            return {"ARCH_p": np.nan, "ARCH_LM_stat": np.nan, "ARCH_Result": "ERROR"}

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def run(self, residuals) -> dict:
        """
        Run both tests on the supplied residuals.

        Parameters
        ----------
        residuals : array-like (pd.Series, np.ndarray, list)
            Standardised or raw model residuals.  NaNs are dropped safely.

        Returns
        -------
        dict with keys:
            LjungBox_p        — p-value at maximum lag
            LjungBox_p_min    — minimum p-value across all lags
            LjungBox_Result   — "PASS" | "FAIL" | "ERROR"
            ARCH_p            — ARCH-LM p-value
            ARCH_LM_stat      — ARCH-LM test statistic
            ARCH_Result       — "PASS" | "FAIL" | "ERROR"
            n_obs             — number of residuals used
            Overall_Result    — "PASS" if BOTH tests pass, else "FAIL"
        """
        arr = self._clean(residuals)
        # DO NOT re-standardize (already standardized from GARCH)
        arr = arr

        if len(arr) < max(self.lb_lags, self.arch_lags) + 10:
            print(
                f"[StatisticalTests] Insufficient observations ({len(arr)}) "
                f"— need at least {max(self.lb_lags, self.arch_lags) + 10}."
            )
            return {
                "LjungBox_p": np.nan,
                "LjungBox_Result": "INSUFFICIENT_DATA",
                "ARCH_p": np.nan,
                "ARCH_LM_stat": np.nan,
                "ARCH_Result": "INSUFFICIENT_DATA",
                "n_obs": len(arr),
                "Overall_Result": "INSUFFICIENT_DATA",
            }

        print(f"[StatisticalTests] Running tests on {len(arr)} residuals "
              f"(lb_lags={self.lb_lags}, arch_lags={self.arch_lags}, α={self.significance})")

        lb_results = self._ljung_box(arr)
        arch_results = self._arch_lm(arr)

        both_pass = (
            lb_results["LjungBox_Result"] == "PASS"
            and arch_results["ARCH_Result"] == "PASS"
        )

        output = {
            **lb_results,
            **arch_results,
            "n_obs": len(arr),
            "Overall_Result": "PASS" if both_pass else "FAIL",
        }

        print(f"[StatisticalTests] Overall → {output['Overall_Result']}")
        return output
