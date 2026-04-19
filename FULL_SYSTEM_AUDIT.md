# FULL SYSTEM FORENSIC AUDIT
## Volatility Forecasting, Regime Dynamics & Risk Modeling using GARCH
### Document Type: Quantitative Validity Audit · Statistical Stress Test · Production Readiness Evaluation
### Date: 2026-03-31 | Reviewer: Senior Quantitative Portfolio Reviewer
### Standard: Institutional / Academic Publication Grade

---

> **Audit Scope:** This is NOT a code review. This is a forensic assessment of whether the system produces mathematically valid, statistically sound, and economically meaningful results. Every file is evaluated on evidence. Every issue is classified by severity. Every model-related claim is challenged against its own output data.

---

## EXECUTIVE SUMMARY

| Dimension | Finding |
|---|---|
| Markets Successfully Completed | **1 of 3** (SSE_Composite only) |
| Pipeline Completion Rate | **33%** — CSI_300 and ChiNext failed before forecast stage |
| Primary Signal Correlation | **0.608** (Hybrid vs Realized Vol, SSE_Composite) |
| VaR Backtest | **CRITICAL FAIL** — all three VaR models rejected at p=0.0 |
| Regime Crisis Capture | **19–22%** — functionally useless; GFC 2008 has NO_DATA |
| Economic Feasibility | Vol-managed strategy **increases** drawdown vs static |
| Hybrid "Copula" Label | **Mislabeled** — it is a simple inverse-RMSE weighted mean |
| Production Readiness | **NOT READY** |

---

## PART I — FILE-LEVEL FORENSIC AUDIT

---

### 1. `main.py`

**Purpose:** CLI entry point. Routes to `--run-pipeline` or `--run-gui`.

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | `import subprocess` imported but never used | LOW |
| 2 | Debug path prints (`[DEBUG PATH] Pipeline:`) left in production code | LOW |
| 3 | No logging initialization before `pipe.run_all()` — `src.logger` is imported only in the `__main__` block of `pipeline.py`, not here. When invoked via `main.py --run-pipeline`, logging is never configured. | MEDIUM |
| 4 | No exception handling around `pipe.run_all()` — a single market failure will surface as unhandled if pipeline's internal `continue` does not catch it | MEDIUM |
| 5 | No argument for market selection — forced to run all 3 markets or none | MEDIUM |

**Root Cause:** Entry point was written as a thin wrapper without operational hardening.

**Institutional Improvement:** Add `--market` flag, initialize logging before pipeline call, remove dead imports, add top-level try/except with exit code.

---

### 2. `src/data_loader.py`

**Purpose:** Downloads OHLCV data from yfinance for CSI_300, SSE_Composite, ChiNext. Caches to CSV.

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | No data staleness check — cached CSVs from 2015-2024 are served indefinitely regardless of age | HIGH |
| 2 | yfinance returns auto-adjusted data by default; this is not documented or validated against raw prices | HIGH |
| 3 | `df.columns.droplevel(1)` drops the ticker level of multi-index but does not verify that remaining columns are the expected OHLCV set | MEDIUM |
| 4 | Start date hardcoded to `2015-01-01` — GFC (2008) is **completely excluded from training data** | CRITICAL |
| 5 | Returns empty DataFrame on failure with no retry logic | MEDIUM |
| 6 | No adjustment for Chinese market closures, suspension events, or circuit-breaker gaps | HIGH |
| 7 | Data provenance is not logged (download timestamp, source URL, version) | MEDIUM |

**🔴 CRITICAL FINDING — DATA SCOPE FAILURE:**

The system claims to validate crisis detection including `GFC_2008 (2008-01-01 to 2009-03-01)` but the data window starts `2015-01-01`. The GFC is entirely absent. The `regime_validation.csv` output confirms this:

```
GFC_2008, 2008-01-01, 2009-03-01, [EMPTY], NO_DATA
```

This is not a minor gap. GFC validation was **advertised as a system capability** but is **structurally impossible** with the current data range. Any paper claiming GFC regime capture is factually incorrect.

**Failure Classification: CRITICAL — invalidates the regime validation narrative for the most important historical stress event.**

---

### 3. `src/preprocessor.py`

**Purpose:** Converts raw OHLCV to log returns, applies business-day frequency resampling.

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | `df_processed.asfreq('B')` inserts NaT rows for non-trading days, then `ffill()` forward-fills them. Methodologically acceptable for EM markets but undocumented. | LOW |
| 2 | `df_processed['Close'] = df_processed['Close'].ffill()` is executed twice (once implicitly on line 26, once explicitly on line 28). Redundant. | LOW |
| 3 | `RV = Log_Return ** 2` — this is raw squared return, not realized volatility. True RV requires summing intraday squared returns (Andersen-Bollerslev). The label `RV` is methodologically misleading and propagates through the entire evaluation chain. | HIGH |
| 4 | No outlier detection — circuit-breaker events (e.g., CSI_300 −8% in 2015) are passed through unchecked | MEDIUM |
| 5 | No unit test for the log-return computation or scaling validation | MEDIUM |

**Root Cause:** `RV = Log_Return²` is a surrogate measure. Downstream components (ML target, evaluator) depend on this definition without disclosure.

---

### 4. `src/eda_econometrics.py`

**Purpose:** Runs ADF, Jarque-Bera, ARCH-LM tests; saves diagnostic plots.

**Observed Output (CSI_300):**

| Test | Statistic | p-value | Interpretation |
|---|---|---|---|
| ADF | −10.34 | 2.78e-18 | Stationarity confirmed ✅ |
| Jarque-Bera | 1978.36 | 0.0 | Non-normality / fat tails confirmed ✅ |
| ARCH-LM (5 lags) | 93.07 | 1.52e-18 | Volatility clustering confirmed ✅ |

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | ADF test run without specifying `maxlag` — default lag selection may be inappropriate for this sample size | MEDIUM |
| 2 | No Ljung-Box test for serial autocorrelation in returns | MEDIUM |
| 3 | No ACF/PACF plots of squared returns — does not verify the clustering beyond the LM test | MEDIUM |
| 4 | `plot_return_distribution` title is "Non-Normal Distribution (Fat Tails)" — conclusion is hardcoded before any test runs. Pre-judged output. | HIGH |
| 5 | `plot_volatility_clustering` shows a raw return series, not the autocorrelation of squared returns — does not technically demonstrate clustering | MEDIUM |
| 6 | EDA entirely absent for ChiNext (pipeline failed before this stage) | CRITICAL |

**Assessment:** The three statistical tests confirm prerequisites for GARCH modeling — the strongest module in the system. However, diagnostic gaps leave the residual independence claim unverified.

---

### 5. `src/garch_models.py`

**Purpose:** Fits ARCH, GARCH, EGARCH, GJR-GARCH and saves AIC/BIC comparison.

**Observed Output (CSI_300):**

| Model | AIC | BIC | Log-Likelihood |
|---|---|---|---|
| ARCH | 2929.43 | 2944.13 | −1461.72 |
| **GARCH** | **2856.98** | **2876.58** | **−1424.49** |
| EGARCH | 2876.55 | 2901.05 | −1433.27 |
| GJR-GARCH | 2857.86 | 2882.36 | −1423.93 |

GARCH(1,1) parameters: mu=−0.037, omega=0.055, alpha=0.078, beta=0.873. Persistence = alpha+beta = **0.950**.

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | `returns * 100` scaling applied for fitting; AIC/BIC values are reported on **scaled** data, making them incomparable against any system using decimal returns | HIGH |
| 2 | No standardized residual diagnostics — no Ljung-Box on squared residuals, no in-model ARCH-LM to verify fit adequacy | HIGH |
| 3 | GJR-GARCH is marginally worse than GARCH (2857 vs 2856) — counter-intuitive for Chinese markets known for leverage effects. Not investigated. | MEDIUM |
| 4 | Model selection is by AIC minimum but no out-of-sample validation is performed at this stage | MEDIUM |
| 5 | `save with is_time_series=True` but index is `Model` (categorical string). This should trigger the DatetimeIndex assertion. | HIGH |
| 6 | No AICc (corrected for finite samples) | LOW |

---

### 6. `src/forecast_engine.py` 🔴 CRITICAL MODULE

**Purpose:** Rolling GJR-GARCH(1,1,1) forecast with 1000-step expanding window; annualizes variance.

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | **PARAMETER FREEZE**: Parameters are refitted only every 20 steps (`i % 20 == 0`). Between refits, `am.fix(last_params)` freezes parameters. This converts a true rolling GARCH into a **parameter-locked pseudo-forecast** for 19 of every 20 observations. | CRITICAL |
| 2 | **POST-HOC SMOOTHING**: `forecast_df['Forecast_Vol'] = forecast_df['Forecast_Vol'].rolling(5).mean().bfill()` — applied after forecasting. The `.bfill()` propagates forward-looking values back to the start of the series. This smoothing artificially inflates correlation with realized vol (which is also smoothed). | CRITICAL |
| 3 | Realized volatility proxy is `rolling(5).std() * sqrt(252)` — a 5-day backward window. Reasonable but not a contemporaneous benchmark. | HIGH |
| 4 | `options={'maxiter': 100}` — insufficient for GARCH convergence on financial data. Non-convergence is silently caught and prior forecast is reused. | HIGH |
| 5 | `upper_bound = quantile(0.99)` then clip — systematically underestimates tail volatility events, precisely what the model must capture for risk management. | HIGH |
| 6 | Hardcoded fallback `0.15` (15% annualized vol) when initial forecast fails — silent, appears in output as a legitimate model forecast. | HIGH |
| 7 | `assert 0.05 < mean < 0.60` — range assertion, not correctness check. Garbage within the band passes silently. | MEDIUM |

**🔴 SIGNAL VS NOISE DIAGNOSIS:**

Standalone GARCH Correlation = **0.530** (SSE_Composite). Above the 0.30 threshold — MODERATE SIGNAL — but:
- 5-day post-hoc smoothing artificially inflates correlation (both forecast and realized series are smoothed)
- 20-step parameter freeze means the model is **static for 95% of all forecast steps**
- True GARCH with per-step refitting would produce lower, honest correlation

**Classification: MODERATE SIGNAL — but partially noise-driven via smoothing and parameter freeze. Reported correlation is inflated relative to true model performance.**

---

### 7. `src/ml_models.py` 🔴 CRITICAL MODULE

**Purpose:** Rolling Random Forest on lagged features; predicts realized volatility.

**Observed Feature Importance (SSE_Composite):**

| Feature | Mean Importance | Stability |
|---|---|---|
| Rolling_Var_21 | 0.242 | Stable |
| Vol_10 | 0.190 | Stable |
| Return_5 | 0.152 | Stable |

**Observed Evaluation (SSE_Composite):**

| Model | RMSE | QLIKE |
|---|---|---|
| GARCH | 0.0703 | **0.1738** |
| RF_ML | 0.1233 | **1.5505** |

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | `data['Log_Return'] = data['Log_Return'].clip(-0.1, 0.1)` — clips ±10% daily moves. Chinese circuit-breaker events (−8% hard limit) are within bounds, but the ≈−10% crash events are truncated, systematically biasing the training distribution toward less extreme volatility | HIGH |
| 2 | Target is `Realized_Vol = sqrt(Log_Return²) * sqrt(252)` — this equals `|Log_Return| * sqrt(252)`. The model predicts the current day's instantaneous absolute return scaled to annual, not forward volatility. | HIGH |
| 3 | `StandardScaler` is fit on `X_train` every single iteration (~1000×) despite model only refitting every 50 steps — computational overhead without proportional benefit | MEDIUM |
| 4 | **QLIKE = 1.5505 for RF is buried in evaluation.csv and never surfaced as an alarm anywhere in the pipeline.** QLIKE is the gold-standard volatility loss function. QLIKE > 1.0 is routinely considered high in academic literature. The ML model is **8.9× worse than GARCH under QLIKE** — this critical finding is invisible. | CRITICAL |
| 5 | ML RMSE (0.123) is also 75% worse than GARCH (0.070). The evaluator runs on the overwritten `garch_preds` (actually hybrid data), making the reported GARCH RMSE = 0.070 actually the hybrid's RMSE. True standalone GARCH RMSE is unknown. | CRITICAL |
| 6 | Top features are all lagged volatility measures — the model is effectively a volatility persistence model. No genuine ML signal extraction. | MEDIUM |

**🔴 SIGNAL VS NOISE DIAGNOSIS:**

ML Correlation = **0.526**. Above 0.30 — MODERATE SIGNAL on raw correlation. But QLIKE = 1.55 reveals the model is severely mis-calibrated relative to the actual volatility level.

**Classification: De facto WEAK SIGNAL under the correct loss function (QLIKE). The correlation metric flatters the ML model. Under QLIKE — which penalizes scale errors — the Random Forest fails to extract meaningful volatility signal beyond what a lagged persistence model would provide.**

---

### 8. `src/pipeline.py` — Hybrid Model Combination 🔴 CRITICAL SECTION

**Purpose:** Combines GARCH and ML via inverse-RMSE weighting.

**Observed Weights (SSE_Composite):**

```
RMSE_GARCH = 0.0775   →   Weight_GARCH = 0.5048
RMSE_ML    = 0.0790   →   Weight_ML    = 0.4952
```

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | **CRITICAL MISLABELING**: The hybrid is named "Hybrid Copula" in `model_comparison.csv`. A copula is a statistical structure linking marginal distributions via joint dependence modeling. This model is a **weighted arithmetic mean**. "Hybrid Copula" is factually wrong and would be immediately rejected in academic review. | CRITICAL |
| 2 | Weights are computed on the **full in-sample period** then applied to the same period. This is in-sample weighting — no out-of-sample weight optimization exists. | CRITICAL |
| 3 | Weights are nearly 50/50 (0.5048 vs 0.4952). The "weighting" provides essentially no differentiation — outputs are mathematically near-identical to an unweighted average. | HIGH |
| 4 | `Rolling_Weight_GARCH` and `Rolling_Weight_ML` in output are **scalar constants** replicated across every row. These are labeled "rolling" but are static global values. | HIGH |
| 5 | **`garch_preds = hybrid_master_df` on line 186** overwrites the GARCH forecast variable with hybrid data. All subsequent modules (RiskMetrics, ModelEvaluator, EconomicEvaluation) receive hybrid data while labeled as "GARCH". The entire downstream evaluation is contaminated. | CRITICAL |
| 6 | No Diebold-Mariano test to statistically confirm hybrid superiority | CRITICAL |
| 7 | Comment step numbering is chaotic (STEP 1 appears 4 times, STEP 5 appears twice). Evidence of iterative patching without cleanup. | LOW |

**🔴 HYBRID VALIDITY CHECK:**

| Model | RMSE | MAE | Correlation |
|---|---|---|---|
| GARCH | 0.0775 | 0.0558 | 0.530 |
| ML | 0.0790 | 0.0516 | 0.526 |
| **Hybrid** | **0.0703** | **0.0474** | **0.608** |

The Hybrid shows 8.7% RMSE improvement over GARCH. Marginal but noted. However:
- No DM-test to confirm statistical significance of this difference
- 50/50 near-equal weights make this mathematically indistinguishable from simple averaging
- Correlation improvement from 0.530 → 0.608 may reflect the double-smoothing effect (both series are 5-day rolling averages)

**Economic Reality Check on Hybrid:** The hybrid is called a "Copula" in published output. It is a weighted mean with ~50/50 static weights computed in-sample. This cannot be submitted to any academic journal without immediate rejection for mislabeling.

> **Classification: Statistically marginal improvement, economically non-deployable without DM test, and academically non-viable due to mislabeling.**

---

### 9. `src/regime_model.py` 🔴 CRITICAL MODULE

**Purpose:** Detects volatility regimes. Claims to implement Markov Regime-Switching.

**Observed Crisis Capture Rates:**

| Crisis | Capture Rate | Status |
|---|---|---|
| GFC_2008 | N/A | NO_DATA (data starts 2015) |
| China_Crash_2015 | **19.3%** | VALID |
| COVID_2020 | **22.1%** | VALID |

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | **NOT A MARKOV MODEL**: `from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression` is present but **never called**. The actual implementation is a deterministic rule: `vol_5 > threshold AND vol_5 > vol_20`, smoothed and thresholded. This is a momentum filter. | CRITICAL |
| 2 | **CAPTURE RATES ARE CRITICALLY LOW**: 19.3% for China 2015 crash; 22.1% for COVID. A model claiming to identify high-volatility regimes correctly classifies fewer than 1 in 4 crisis days. This is indistinguishable from a random baseline at 20%. | CRITICAL |
| 3 | **LOOK-AHEAD #1**: `vol_5 = log_return.rolling(5).std().bfill() * sqrt(252)` — `.bfill()` fills early NaN with future values. | HIGH |
| 4 | **LOOK-AHEAD #2**: `threshold = vol_5.rolling(252).quantile(0.6).bfill()` — bfill on a 252-day rolling quantile. The first full year's threshold is populated using future quantile information. | HIGH |
| 5 | **LOOK-AHEAD #3**: `raw_prob.rolling(5).mean().bfill()` — bfill on a smoothed probability series. Three separate forward-contamination points in a single function. | HIGH |
| 6 | GFC_2008 is hardcoded in `crisis_periods` but data starts 2015. Produces silent `NO_DATA` without any pipeline alarm. | HIGH |
| 7 | No regime persistence metric — the model may flip daily without producing stable regime labels. | MEDIUM |
| 8 | Crisis window for China_Crash_2015 spans `2015-06-01 to 2016-02-01` — 8 months. The main crash occurred June–August 2015. A good regime model should identify >70% of these days as high-regime. 19% is failure. | HIGH |

**🔴 SIGNAL VS NOISE DIAGNOSIS:**

Regime capture rates of 19–22% during the most volatile periods in the data history.

A random classifier with a 20% prior would achieve statistically equivalent results by chance.

**Classification: NON-ROBUST MODEL. Crisis capture is indistinguishable from random. The claimed Markov model is never implemented. This module's outputs should not be used to make any risk-management or regime-filtering decisions.**

---

### 10. `src/risk_metrics.py` 🔴 CATASTROPHIC FAILURE

**Purpose:** Computes Normal VaR, Student-t VaR, Empirical VaR at 95% confidence; runs Kupiec, Christoffersen, and DQ backtests.

**Observed Output (SSE_Composite) — THIS IS THE CRITICAL EVIDENCE:**

| VaR Model | Expected Violations | **Actual Violations** | Kupiec p-value | Status |
|---|---|---|---|---|
| VaR_normal | 79.2 | **1,484** | **0.0000** | **FAIL** |
| VaR_student_t | 79.2 | **1,526** | **0.0000** | **FAIL** |
| VaR_empirical | 79.2 | **1,509** | **0.0000** | **FAIL** |

At 1,584 observations, expected 5% violations = **79.2**. Actual violations = **1,484–1,526** = **93.7–96.3% of all observations**.

**Root Cause of Catastrophic Failure: Unit Scale Mismatch**

The VaR computation in `calculate_var_es()` converts annualized forecast_vol to daily: `forecast_vol / sqrt(252)`. With mean annualized vol ~16%, daily vol ~ 1.0%. VaR_normal = `norm.ppf(0.05) * 1.0%` = −1.645%. 

The backtest then checks: `actual_return < -VaR_normal`. Daily returns from yfinance are in decimal (e.g., 0.01 = 1% move). For a −1% day: `−0.01 < +0.01645` → **False**. Wait — that should NOT be a violation.

The actual evidence shows 1,484 violations, which is 93.7% of days. For this to happen the VaR values must be **much smaller** than the actual returns, meaning VaR_normal ≈ 0 or VaR values are positive (not negative losses). Deeper analysis: `norm.ppf(0.05) = −1.645`. So `VaR_normal = −1.645 * (forecast_vol/sqrt(252))`. For a violation to occur: `actual_return < -VaR_normal = 1.645 * daily_vol`. So a violation means the actual return exceeds **+1.645σ** (a positive return). This flips the test — it counts days when returns are POSITIVE and large, not when losses are large. **The sign convention is inverted**: the system is forecasting losses as negative VaR values, and then checking if returns are MORE profitable than −VaR, which will be true 93.7% of the time.

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | **Violation check inverted**: `actual_return < -VaR_normal`. Since `VaR_normal` is already negative (norm.ppf(0.05) is negative), `-VaR_normal` is positive. This checks if the return exceeds a positive threshold (96% of days), not if losses exceed a negative threshold (5% of days). | CRITICAL |
| 2 | Best model selection picks `VaR_normal` as "best" despite Kupiec p=0.0 absolute failure — because all three models are equally failed and it picks the one closest to the threshold by a meaningless metric | HIGH |
| 3 | Violation count of 1,484 is labeled "Underestimation" in the interpretation — it is not underestimation of tail risk, it is a sign error | HIGH |
| 4 | DQ test on a binary series that is 93.7% ones produces a degenerate Logit — DQ p-values are unreliable | HIGH |

**🔴 RISK MODEL VERDICT:**

**ALL THREE VAR MODELS FAIL ABSOLUTELY — THE VIOLATION CHECK IS SIGN-INVERTED.**

This module cannot produce valid risk metrics in its current form. The output is not "conservative" or "underestimating" — it is measuring the wrong thing entirely.

> **This is the most critical single bug in the system. A risk model that produces 93.7% violation rates is worse than no model — it provides false confidence in a system that would be catastrophically wrong in production.**

---

### 11. `src/economic_evaluation.py` 🔴 CRITICAL BUGS

**Purpose:** Evaluates three trading strategies — Static, Volatility-Managed, Regime-Based.

**Observed Output (SSE_Composite):**

| Strategy | Ann. Return | Volatility | Sharpe | Max DD | Avg Weight |
|---|---|---|---|---|---|
| Static Exposure | 3.90% | 16.7% | **0.234** | −31.8% | 1.0x |
| Volatility-Managed | 6.00% | **25.1%** | 0.239 | **−47.8%** | **1.5x** |
| Regime-Based | 3.48% | 14.8% | 0.234 | −33.9% | 0.90x |

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | **UNIT MISMATCH — STRATEGY IS BROKEN**: `target_vol = 20.0` (in percent) vs `df['Forecast_Vol']` in decimal (~0.16). Weight = `20.0 / 0.16 = 125x`. Clipped to 1.5x maximum. **The volatility-managed strategy is a constant 1.5× leveraged long for the entire history. Avg weight = 1.5, Min weight = 1.5, Max weight = 1.5. No dynamic weighting occurs.** | CRITICAL |
| 2 | A constant 1.5× leveraged position is labeled "Volatility-Managed". This is false. The strategy label is factually incorrect — it always lever up, regardless of forecast volatility. | CRITICAL |
| 3 | Evidence: Vol-Managed has HIGHER volatility (25.1%) and DEEPER drawdown (−47.8%) than static (16.7%, −31.8%). A functioning vol-managed strategy should reduce both. | CRITICAL |
| 4 | `mdd = (r.cumsum().cummax() - r.cumsum()).max()` — max drawdown is computed on **summed log returns**, not compound growth. For moderate vol, the difference is small. For high-vol regimes, this systematically understates true drawdown. | HIGH |
| 5 | Regime weight = 0.4x when `Regime_Prob > 0.7`. Since regime detection captures only 19–22% of crisis days, the 0.4x protection executes less than 1 day in 5 during crises. Effective crisis protection rate approaches zero. | HIGH |
| 6 | Strate Sharpe ratios (0.234, 0.239, 0.234) are statistically indistinguishable — the system adds no alpha over buy-and-hold. | HIGH |
| 7 | `df.plot(df['Regime_Return'])` on line 141 will error if `Regime_Return` has NaN not covered by the `dropna` filter on `Dynamic_Return` | MEDIUM |

**Crisis Performance (SSE_Composite):**

| Strategy | Crisis Return | Crisis Sharpe | Crisis Drawdown |
|---|---|---|---|
| Static | 6.9% | 0.323 | 21.8% |
| Vol-Managed | 10.9% | 0.340 | **32.7%** |
| Regime-Based | 5.3% | **0.346** | **10.7%** |

The Vol-Managed strategy has the worst crisis drawdown (32.7%) because it is permanently 1.5× long. The Regime-Based strategy shows better crisis drawdown (10.7%) than static — but only because the high-vol regime happened to coincide with a period where reducing to 0.4x helped. With 22% capture rate, this is not reliably repeatable.

**🔴 ECONOMIC REALITY CHECK:**

> **"Can this be traded?"**

- Volatility-Managed is a broken strategy (constant 1.5× leveraged long). Not tradeable as described.
- Regime-Based executes protective trades only 19–22% of crisis time. Effectively a slightly dampened static strategy.
- All three Sharpe ratios are between 0.234–0.239. The model adds **zero measurable alpha after removing the 1.5× leverage effect**.
- No transaction costs modeled. Even minimal costs would eliminate the 0.005 Sharpe difference between static and vol-managed.

> **Verdict: "Statistically claimed but economically non-viable."**

The system produces three strategies that are economically indistinguishable from a static long position in Chinese equities.

---

### 12. `src/evaluator.py`

**Purpose:** Computes MSE, RMSE, QLIKE for GARCH and ML forecasts.

**Observed Output (SSE_Composite):**

| Model | MSE | RMSE | QLIKE |
|---|---|---|---|
| GARCH | 0.00494 | 0.0703 | **0.1738** |
| RF_ML | 0.01520 | 0.1233 | **1.5505** |

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | **`evaluator.evaluate(garch_preds, 'GARCH', market_name)` is called after `garch_preds = hybrid_master_df`** (pipeline.py line 186). The "GARCH" row actually evaluates the **hybrid forecast**. RMSE 0.0703 belongs to the Hybrid, not standalone GARCH. True GARCH RMSE is unknown. | CRITICAL |
| 2 | **QLIKE = 1.5505 for RF_ML is NEVER FLAGGED**. This is the most important finding in the evaluator and it is silently written to a CSV with no alert, no threshold warning, no pipeline flag. | CRITICAL |
| 3 | No Mincer-Zarnowitz (MZ) regression — the standard academic test for forecast unbiasedness | HIGH |
| 4 | No out-of-sample R² | MEDIUM |
| 5 | Correlation not included in evaluator output | MEDIUM |

---

### 13. `src/model_comparison.py`

**Purpose:** Side-by-side RMSE/MAE/Correlation for GARCH, ML, Hybrid.

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | Model labeled "Hybrid Copula" — incorrect academic terminology | CRITICAL |
| 2 | No Diebold-Mariano test to validate statistical superiority of hybrid | CRITICAL |
| 3 | No Hansen's Model Confidence Set | HIGH |
| 4 | Ranking by RMSE only — QLIKE is computed in evaluator but not used here for selection | MEDIUM |
| 5 | `compare_models` is called with `garch_df` which is already `hybrid_master_df` — effectively comparing hybrid vs ML vs hybrid | CRITICAL |

---

### 14. `src/regime_performance.py`

**Purpose:** Splits forecast accuracy by high/low vol regime.

**Observed Output (SSE_Composite):**

| Model | RMSE (High Vol) | RMSE (Low Vol) |
|---|---|---|
| GARCH | 0.104 | 0.071 |
| ML | **0.126** | 0.066 |
| Hybrid | 0.107 | 0.060 |

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | RMSE is 46–91% higher in high-vol regimes for all models. All models degrade most during the periods where accuracy matters most. | HIGH |
| 2 | ML has the worst high-vol RMSE (0.126) — the model that already has QLIKE = 1.55 also performs worst in the periods that matter most | HIGH |
| 3 | Regime labels are contaminated by three bfill look-ahead calls — regime RMSE splits may be slightly inflated | HIGH |
| 4 | No statistical significance test for regime RMSE difference | MEDIUM |

---

### 15. `src/validation.py` (ValidationEngine)

**Purpose:** Reads `var_comparison.csv`, classifies each VaR model as PASS/WARNING/FAIL.

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | All three VaR models have Kupiec p=0.0 → all correctly identified as FAIL. But the pipeline **continues without interruption** — a total VaR failure does not halt execution or raise a system-level alarm | HIGH |
| 2 | DQ p-value is extracted but not used in PASS/FAIL determination | MEDIUM |
| 3 | Fallback FAIL record is injected when data is missing but does not propagate to any system-level alert | MEDIUM |
| 4 | Interpretation strings are hardcoded template text — not derived from actual computed values | LOW |

---

### 16. `src/interpretation.py`

**Purpose:** Generates academic text interpretation report.

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | **Interpretation is boilerplate text**. Every market with corr > 0.60 receives identical text regardless of context. Section 5 "Final Synthesis" is entirely static hardcoded text, rendering it meaningless. | HIGH |
| 2 | Correlation of 0.608 interpreted as "strong predictive alignment" — this claim is not challenged despite the post-hoc smoothing that inflates the correlation | MEDIUM |
| 3 | **Crisis performance section reads `vol_m['Drawdown_crisis']`** but the actual field is `Drawdown_crisis` with a value of `0.327` (positive float). The interpretation formats this as percentage without sign conversion — readers will misread direction. | HIGH |
| 4 | Report is unstructured text only — no JSON, no structured CSV for programmatic consumption | LOW |

---

### 17. `src/data_validation.py`

**Purpose:** Lightweight post-load validation.

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | Only warns on missing values — does not specify which columns | LOW |
| 2 | Length threshold < 500 is a warning only, not enforced | MEDIUM |
| 3 | Called in `pipeline.py` as `validate_data(df)` but return value is never checked — always returns True | LOW |
| 4 | No check for duplicate index entries | MEDIUM |

---

### 18. `src/utils/validation.py`

**Purpose:** Central serialization and DatetimeIndex enforcement.

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | `assert isinstance(df.index, pd.DatetimeIndex)` in `is_time_series=True` path. `garch_models.py` saves comparison CSV with `is_time_series=True` where the index is the Model name string. This assertion should fail but evidently passes — suggesting silent bypass or a code path that doesn't reach the assertion. | HIGH |
| 2 | `MIN_REQUIRED = 50` row minimum is too low for meaningful rolling-window backtesting requiring 1000+ observations | MEDIUM |
| 3 | `df.index.name = "Date"` forcibly overwrites any existing index name | LOW |
| 4 | No checksum or hash of saved files — post-write corruption is undetectable | LOW |

---

### 19. `src/logger.py`

**Purpose:** Configures file-based logging.

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | Logger is only imported in `pipeline.py`'s `__main__` block — when invoked via `main.py --run-pipeline`, **logging is never initialized**. All `logging.info()` and `logging.error()` calls silently produce no output in production runs. | HIGH |
| 2 | No log rotation — `logs/pipeline.log` grows unbounded | LOW |
| 3 | No stdout handler — print and logging run in parallel, creating duplicate output with different timestamp formats | LOW |

---

### 20. `adversarial_test.py`

**Purpose:** Post-run sanity check of outputs.

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | Checks for `'Violation'` column in `risk_forecasts.csv`, but this column is **never created by `risk_metrics.py`** — the VaR consistency test always silently skips | HIGH |
| 2 | Vol spike threshold is 60% but values are displayed as raw decimals (e.g., "0.35%") — any reader expecting annualized percentages receives meaningless numbers | MEDIUM |
| 3 | No assertions — all findings are written to a text file only. This script cannot fail regardless of how catastrophic the outputs are. | HIGH |
| 4 | Regime test uses `Regime_1_Prob > 0.7` but pipeline uses `Regime_Prob > 0.6` for switching — threshold inconsistency between auditing and production | MEDIUM |
| 5 | ChiNext listed as a test market but has no output files — silently produces empty results | MEDIUM |

---

### 21. `refactor2.py`

**Purpose:** Batch string replacement used to fix `validate_and_save` call signatures.

**Issues Identified:**

| # | Issue | Severity |
|---|---|---|
| 1 | This file is evidence that internal API call signatures were broken and mass-patched — sign of unstable internal API design history | MEDIUM |
| 2 | Glob pattern is hardcoded to an absolute Windows path — not portable | LOW |
| 3 | No backup of original files before patching — destructive in-place modification | MEDIUM |
| 4 | Several fix pairs have identical before/after strings (e.g., preprocessor.py entry) — dead fix entries | LOW |

---

## PART II — GLOBAL SYSTEM ANALYSIS

---

### Data Flow Trace

```
yfinance (2015–2024)
    ↓ [CRITICAL: GFC 2008 absent — entire regime validation narrative invalid]
DataLoader.fetch_market_data()
    ↓ [HIGH: no staleness check; no adjustment verification]
Preprocessor.process()
    ↓ [HIGH: RV = Log_Return² mislabeled; dual bfill]
EDAEconometrics.analyze()
    ↓ [CLEAN: ADF/JB/ARCH-LM valid, prerequisites confirmed]
GARCHModels.compare_models()
    ↓ [HIGH: scaling bakes AIC values in percentage units]
RegimeModel.detect_regimes()
    ↓ [CRITICAL: not Markov; bfill look-ahead ×3; capture 19–22%]
ForecastEngine.rolling_forecast()
    ↓ [CRITICAL: 20-step param freeze; post-hoc 5-day smoothing inflates correlation]
MLModels.run_ml()
    ↓ [CRITICAL: QLIKE 8.9× worse than GARCH; never surfaced as alert]
Pipeline — Hybrid Combination
    ↓ [CRITICAL: in-sample weighting; 50/50; wrong "Copula" label]
        ↓ garch_preds OVERWRITTEN to hybrid → ALL downstream uses hybrid data
RiskMetrics.backtest()
    ↓ [CATASTROPHIC: sign-inverted violation check; 1,484 violations vs expected 79]
EconomicEvaluation.evaluate()
    ↓ [CRITICAL: Vol-Managed is permanent 1.5× long due to unit mismatch]
InterpretationLayer.generate_report()
    ↓ [HIGH: boilerplate text; hardcoded conclusions]
ValidationEngine.validate_results()
    ↓ [HIGH: FAIL detected but pipeline continues; no alarm escalation]
```

---

### Three Markets Status

| Market | Pipeline Completed | Critical Artifact Status |
|---|---|---|
| CSI_300 | ❌ **Partial** — stopped after EDA/GARCH/Regime stage | No forecast, no risk, no economic output |
| SSE_Composite | ✅ **Full** — 28 output files generated | Complete but with critical bugs throughout |
| ChiNext | ❌ **Not started** — no output files found | Total pipeline failure |

**Critical Implication:** `final_model_summary.csv` contains only SSE_Composite. All conclusions about system performance are drawn from a **single-market, single-run observation**. Cross-market generalizability cannot be claimed. No averaging across markets is possible. The paper's core quantitative claims rest on one time series.

---

### Performance Bottlenecks

| Issue | Location | Severity |
|---|---|---|
| RF scaler refitted every step (~1000×) | `ml_models.py:85-87` | MEDIUM |
| GARCH param freeze every 20 steps | `forecast_engine.py:35` | CRITICAL (accuracy impact) |
| Sequential market processing (no parallelism) | `pipeline.py:run_all()` | MEDIUM |
| `shutil.rmtree("results/metrics")` on every pipeline init | `pipeline.py:33` | MEDIUM |

---

## PART III — TRUSTWORTHINESS SCORES

---

### 🔴 Data Integrity Score: **52 / 100**

| Component | Score | Evidence |
|---|---|---|
| Source data validity | 70 | yfinance real data; unadjusted/unvalidated |
| Temporal scope | 20 | 2015–2024 entirely misses GFC 2008 |
| Look-ahead contamination | 35 | Three bfill look-aheads in regime detection |
| Return calculation | 75 | Log returns computed correctly |
| RV proxy accuracy | 45 | RV=Log_Return² ≠ true realized vol |
| Duplicate/NaN handling | 65 | Deduplication present in utils/validation.py |
| Data provenance | 30 | No timestamps, no hashes, no version logging |

---

### 🔴 Model Reliability Score: **41 / 100**

| Component | Score | Evidence |
|---|---|---|
| GARCH specification | 60 | Valid GJR-GARCH structure; persistence = 0.95 ✓ |
| GARCH forecast fidelity | 35 | 20-step freeze + post-hoc smoothing |
| ML signal quality | 20 | QLIKE = 1.55; worst high-vol RMSE |
| Regime detection | 10 | Not a Markov model; 19–22% capture |
| Hybrid methodology | 25 | IS weighting; 50/50 weights; eval contaminated |
| Forecast correlation (reported) | 55 | 0.608 — moderate but inflated by smoothing |
| OOS validation | 0 | No formal train/test split reported separately |
| Statistical superiority proof | 0 | No DM test, no MCS, no MZ regression |

---

### 🔴 Risk Model Validity Score: **8 / 100**

| Component | Score | Evidence |
|---|---|---|
| VaR Normal model | 0 | Kupiec p=0.0; 1,484 of 1,584 days flagged as violations |
| VaR Student-t model | 0 | Kupiec p=0.0; 1,526 violations |
| VaR Empirical model | 0 | Kupiec p=0.0; 1,509 violations |
| Backtest framework (code) | 45 | Kupiec + Christoffersen + DQ implemented correctly |
| Scale coherence | 0 | Sign-inverted violation check is root cause |
| Response to failure | 15 | FAIL detected; pipeline continues without alarm |

---

### **Would you trust this system with real capital? ❌ NO.**

**Justification:**

1. **VaR module is catastrophically broken.** A system that reports 93.7% of trading days as "VaR violations" would trigger margin calls, regulatory flags, and immediate shutdown at any institutional desk on Day 1.

2. **Vol-Managed strategy increases risk.** The flagship economic strategy — meant to reduce drawdown — instead produces a constant 1.5× leveraged position due to a unit mismatch bug, deepening drawdown from −31.8% to −47.8%.

3. **Regime model captures fewer than 1 in 4 crisis days.** A risk early-warning system with 22% crisis recall provides no actionable edge over a random signal.

4. **Only 1 of 3 target markets completes the full pipeline.** A system that fails to run on 67% of its target universe cannot support cross-market analysis claims.

5. **The core model evaluation is contaminated.** After line 186 of pipeline.py, all "GARCH" evaluations actually measure the hybrid. True GARCH performance is unmeasured.

6. **ML model is 8.9× worse than GARCH under QLIKE — silently.** Under the standard academic loss function for volatility, the ML component destroys value. This is never surfaced.

---

## PART IV — FINAL VERDICT

---

### Engineering Score: **55 / 100**

The pipeline is architecturally modular — 11 sequential stages, clean separation of concerns per module, a central validation utility, and organized output directories. The validate_and_save abstraction shows engineering intent. Points deducted for: 2 of 3 markets failing silently, look-ahead contamination via production bfill, chaotic step-numbering in pipeline.py (STEP 1 appears 4×), the GARCH evaluation silently measuring hybrid data, the logger never activating in production runs, non-portable absolute paths, and a 20-step parameter-freeze that undermines forecast credibility.

---

### Quant Rigor Score: **28 / 100**

The prerequisite econometric tests (ADF, JB, ARCH-LM) are correct and confirm GARCH suitability. GARCH persistence of 0.95 is appropriate for Chinese equity volatility. The QLIKE loss function is implemented — a sign of quantitative awareness. Beyond that: no DM test, no MZ regression, no MCS, no IS/OOS split validation, a VaR module with a sign-inverted violation check, a regime classifier that is not a Markov model, in-sample hybrid weighting, post-hoc forecast smoothing, three bfill look-ahead contaminations, ML QLIKE = 1.55 never surfaced, and an interpretation layer filled with boilerplate text.

---

### Production Readiness Score: **18 / 100**

Pipeline completes for 1 of 3 markets. VaR validation fails at 93.7% violation rate. No transaction cost or slippage model. No walk-forward or IS/OOS validation. No live data feed connector. No Docker or reproducibility layer. No unit tests or integration tests. No CI/CD. Vol-Managed strategy increases drawdown vs static. Regime strategy economically inert. Logging never activates via main.py. The `results/metrics` self-destruction pattern risks data loss on restart.

---

## ⚠️ PUBLISHABLE WITH MAJOR REVISIONS

The system cannot be submitted in current form. The architectural intent — a multi-model hybrid pipeline with regime annotation, formal VaR backtesting, and economic strategy simulation — is academically ambitious and structurally sound in concept. However, the following **14 blocking issues** must be resolved before any submission:

| Priority | Issue | Required Fix |
|---|---|---|
| P0 | VaR sign-inverted violation check | Correct to: `actual_return < VaR_normal` (no negation), or ensure VaR is stored as a loss (negative value) |
| P0 | Vol-Managed unit mismatch | Fix `target_vol=20.0` to `target_vol=0.20` or convert all forecast_vol to percent scale |
| P0 | "Hybrid Copula" mislabeling | Rename to "Inverse-RMSE Weighted Ensemble" — rewrite in all reports, CSVs, and interpretation text |
| P0 | In-sample hybrid weighting | Implement rolling OOS weight optimization with proper train/update split |
| P0 | Regime model is not Markov | Implement `MarkovRegression` from statsmodels (already imported), or explicitly label as "Rule-Based Regime Filter" — cannot claim Markov Switching in paper |
| P0 | GFC 2008 absent from data | Extend `start_date` to 2005-01-01, OR remove all GFC references from crisis validation |
| P0 | GARCH eval measures Hybrid | Fix pipeline.py line 186 — preserve `garch_preds_original` separately from hybrid for correct evaluation |
| P1 | Three bfill look-ahead contaminations | Replace `bfill()` with `ffill()` or cold-start initialization in `regime_model.py` |
| P1 | Post-hoc 5-day smoothing in forecast engine | Remove or move to a separate `Forecast_Vol_Smoothed` column; preserve raw point-in-time forecast |
| P1 | 20-step parameter freeze | Execute full GARCH fitting per step, or disclose refit interval explicitly in methodology |
| P1 | QLIKE=1.55 for ML never alerted | Add QLIKE threshold check (e.g., > 0.5 → WARNING) in evaluator.py and interpretation.py |
| P1 | Add Diebold-Mariano test | Required before claiming hybrid statistical superiority over GARCH |
| P2 | Complete all 3 markets | Multi-market results are required for any generalizability claim |
| P2 | Add Mincer-Zarnowitz regression | Required for forecast unbiasedness claim |

---

*Audit completed: 2026-03-31*
*Codebase reviewed: 21 files across src/, root, and output artifacts*
*Evidence base: Full source code + SSE_Composite empirical outputs (the sole complete pipeline run)*
