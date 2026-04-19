"""
src/economic_analysis.py
─────────────────────────────────────────────────────────────────────────────
Advanced Economic Evaluation Layer — Quant Research Evaluation Extension
─────────────────────────────────────────────────────────────────────────────
Purpose:
    Evaluate the real financial usefulness of a volatility forecast by
    simulating a volatility-targeting portfolio strategy and computing
    institution-grade performance metrics.

Strategy Logic
──────────────
    At each time step T, the portfolio's position size is set to:

        Position_T = clip(target_vol / Forecast_Vol_T, 0, max_leverage)

    The clipping to [0, max_leverage] ensures the strategy never goes short
    or exceeds a configurable leverage cap.

    The position is applied with a one-period lag (shift(1)) to prevent
    look-ahead bias.  Portfolio return at T is:

        R_T = Position_{T-1} × Actual_Return_T

Metrics
───────
    Sharpe Ratio   — annualised: (μ_r × 252) / (σ_r × √252)
    Max Drawdown   — maximum peak-to-trough decline on compound equity curve
    Equity Curve   — cumulative product of (1 + R_t)

Input Contract
──────────────
    df must contain:
        • Forecast_Vol   — annualised, decimal (e.g. 0.18 for 18%)
        • Actual_Return  — log or simple daily return, decimal
    NaNs are handled safely throughout.

Usage:
    from src.economic_analysis import EconomicAnalysis
    analyser = EconomicAnalysis(target_vol=0.20, max_leverage=2.0)
    result_df, metrics = analyser.evaluate(df)
"""

import numpy as np
import pandas as pd


class EconomicAnalysis:
    """
    Volatility-targeting economic strategy evaluator.

    Parameters
    ----------
    target_vol : float
        Annualised target volatility in decimal form (e.g. 0.20 = 20%).
        The position size is scaled so the portfolio aims to reach this vol.
    max_leverage : float
        Maximum allowed leverage (position cap). Default 2.0×.
    min_vol_floor : float
        Minimum forecast vol used in the denominator to avoid division by
        near-zero values. Default 0.01 (1%).
    risk_free_rate : float
        Annualised risk-free rate used in Sharpe computation. Default 0.0.
    """

    def __init__(
        self,
        target_vol: float = 0.20,
        max_leverage: float = 2.0,
        min_vol_floor: float = 0.01,
        risk_free_rate: float = 0.0,
    ):
        if target_vol <= 0:
            raise ValueError("target_vol must be > 0")
        if max_leverage <= 0:
            raise ValueError("max_leverage must be > 0")

        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.min_vol_floor = min_vol_floor
        self.risk_free_rate = risk_free_rate

    # ──────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        """
        Compute the maximum peak-to-trough drawdown on a compound equity curve.

        Returns a positive float representing the magnitude of the worst
        drawdown (e.g. 0.30 = 30% drawdown).
        """
        if equity.empty or equity.isna().all():
            return np.nan
        # Running peak
        running_peak = equity.cummax()
        # Drawdown at each point
        drawdown = (equity - running_peak) / running_peak.replace(0, np.nan)
        return float(-drawdown.min())  # positive magnitude

    @staticmethod
    def _sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Annualised Sharpe ratio: (mean_daily_excess_return × 252)
                                 / (std_daily_return × √252)

        Returns NaN if std is zero.
        """
        if returns.empty or returns.isna().all():
            return np.nan
        excess = returns - (risk_free_rate / 252.0)
        mean_exc = excess.mean()
        std_ret = returns.std()
        if std_ret == 0 or np.isnan(std_ret):
            return np.nan
        return float((mean_exc * 252.0) / (std_ret * np.sqrt(252.0)))

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame):
        """
        Simulate the volatility-targeting strategy and compute performance
        metrics.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns 'Forecast_Vol' and 'Actual_Return'.
            The input DataFrame is NOT modified.

        Returns
        -------
        result_df : pd.DataFrame
            Time-indexed DataFrame with columns:
                Position         — clipped leverage at each step (lagged)
                Strategy_Return  — realised portfolio return at each step
                Equity           — compound equity curve starting at 1.0

        metrics : dict
            Scalar performance summary:
                Sharpe          — annualised Sharpe ratio
                Max_Drawdown    — magnitude of worst peak-to-trough loss
                Ann_Return      — annualised arithmetic mean return
                Ann_Volatility  — annualised return standard deviation
                n_obs           — number of valid observations used
                target_vol      — the target vol parameter used
                avg_position    — average portfolio leverage (gross exposure)
        """
        # ── Validate inputs ────────────────────────────────────────────────
        required = {"Forecast_Vol", "Actual_Return"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"[EconomicAnalysis] Missing required columns: {missing}"
            )

        data = df[["Forecast_Vol", "Actual_Return"]].copy()

        # ── Clean ──────────────────────────────────────────────────────────
        data = data.dropna(subset=["Actual_Return"])
        data["Forecast_Vol"] = data["Forecast_Vol"].ffill().fillna(self.target_vol)

        n_total = len(data)
        print(
            f"[EconomicAnalysis] Running strategy on {n_total} observations "
            f"(target_vol={self.target_vol:.2%}, max_leverage={self.max_leverage:.1f}×)"
        )

        # ── Floor forecast vol to prevent division by near-zero ────────────
        forecast_vol_safe = data["Forecast_Vol"].clip(lower=self.min_vol_floor)

        # SOFT VOL REGIME FILTER (DO NOT KILL SIGNAL)
        vol_mean = forecast_vol_safe.rolling(20).mean()

        signal = (forecast_vol_safe < (vol_mean * 2.2)).astype(float)

        # MOMENTUM FILTER (NEW — CRITICAL)
        returns_ma = data["Actual_Return"].rolling(5).mean()
        momentum_filter = (returns_ma > -0.002).astype(float)

        # COMBINED SIGNAL
        signal = signal * momentum_filter

        # POSITION
        raw_position = signal * (self.target_vol / forecast_vol_safe)
        position = raw_position.clip(lower=0.0, upper=self.max_leverage)
        # Smooth exposure (prevents instability)
        position = position.ewm(alpha=0.2).mean()

        # ── Shift by 1 to prevent look-ahead bias ──────────────────────────
        position_lagged = position.shift(1)
        position_lagged = position_lagged.fillna(0.0)

        # ── Portfolio return ───────────────────────────────────────────────
        strategy_return = (position_lagged * data["Actual_Return"]).fillna(0.0)
        # BUY and HOLD BASELINE 
        buy_hold_return = data["Actual_Return"].fillna(0.0)
        buy_hold_equity = (1.0 + buy_hold_return).cumprod().clip(lower=1e-6)

        # ── Compound equity curve — start at 1.0 ──────────────────────────
        equity = (1.0 + strategy_return).cumprod()
        equity = equity.clip(lower=1e-6)

        # Sanity check: equity must not collapse to zero or go negative
        if equity.min() <= 0:
            print(
                "[EconomicAnalysis] WARNING: Equity curve collapsed. "
                "Check for extreme return inputs."
            )
            # Replace with last valid positive value to keep curve stable
            equity = equity.clip(lower=1e-6)

        # ── Assemble result DataFrame ──────────────────────────────────────
        result_df = pd.DataFrame(
            {
                "Position":        position_lagged,
                "Strategy_Return": strategy_return,
                "Equity":          equity,
                "BuyHold_Return":  buy_hold_return,
                "BuyHold_Equity":  buy_hold_equity
            },
            index=data.index,
        )

        # ── Scalar metrics ─────────────────────────────────────────────────
        sharpe_val   = self._sharpe(strategy_return, self.risk_free_rate)
        mdd_val      = self._max_drawdown(equity)
        ann_ret      = float(strategy_return.mean() * 252)
        ann_vol      = float(strategy_return.std() * np.sqrt(252))
        avg_position = float(position_lagged.mean())
        n_obs        = int(strategy_return.notna().sum())

        buy_hold_curve = (1 + buy_hold_return).cumprod()
        buy_hold_return_series = buy_hold_curve.pct_change().dropna()

        buy_hold_sharpe = (
            buy_hold_return_series.mean() * 252 /
            (buy_hold_return_series.std() * np.sqrt(252) + 1e-8)
        )

        metrics = {
            "Sharpe": sharpe_val,
            "BuyHold_Sharpe": buy_hold_sharpe,
            "Alpha_vs_BuyHold": sharpe_val - buy_hold_sharpe,
            "Max_Drawdown": mdd_val,
            "Ann_Return": ann_ret,
            "Ann_Volatility": ann_vol,
            "n_obs": n_obs,
            "target_vol": self.target_vol,
            "avg_position": avg_position,
        }

        # ── Log summary ────────────────────────────────────────────────────
        print(
            f"[EconomicAnalysis] Sharpe={sharpe_val:.3f}  "
            f"MaxDD={mdd_val:.2%}  "
            f"Ann_Return={ann_ret:.2%}  "
            f"Avg Position={avg_position:.2f}×"
        )

        return result_df, metrics
