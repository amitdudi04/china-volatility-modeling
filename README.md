# Empirical Analysis of Volatility and Risk Dynamics in Chinese Equity Markets

**Author:** Amit Kumar Dudi  
**Research area:** Quantitative Finance · Financial Econometrics · Chinese Equity Markets · Volatility & Tail Risk  
**Latest research:** **v2.2 — retrospective reanalysis of the original 2015–2024 market-data archive**

> **Repository version notice:** This default-branch README has been updated to reflect the corrected and tested **v2.2 research results**. The **v2.2 implementation, checked output files, and current offline dashboard are on the [research v2.2 branch](https://github.com/amitdudi04/china-volatility-modeling/tree/research/phase1-2-data-timing-2026-10-12)**, not yet merged into `main`. The Python pipeline and historical reports currently on `main` are **legacy implementations** and should not be used to reproduce or substantiate the corrected findings. Review the [research correction PR](https://github.com/amitdudi04/china-volatility-modeling/pull/1) for the code changes.

## 1. Research overview

**Research question:** Can econometric and machine-learning models produce useful **next-session variance forecasts** for Chinese equity markets, and do those forecasts improve downside-risk measurement or a simple volatility-managed allocation rule?

The revised study compares a genuine **GARCH(1,1)** conditional-variance model, a **Random Forest**, a **log-variance Ridge** regression, a fixed **Ridge–Random Forest ensemble**, and a 21-session historical-variance benchmark. All methods are evaluated against the *same future return-based variance proxy* on the *same dates*.

The project measures forecasting losses, uncertainty, conditional Value at Risk (VaR), parametric Expected Shortfall (ES), and the performance of an illustrative volatility-targeted exposure rule. **The main contribution is a transparent and reproducible comparison, including negative findings—not a claim of guaranteed risk protection or an institutional trading system.**

## 2. Markets, data, and evaluation period

The corrected analysis uses the **original archived data**. It does not pretend that newer 2025–2026 observations or missing earlier index histories were available.

| Market | Actual instrument | Historical coverage | Price observations |
| --- | --- | --- | ---: |
| CSI 300 | `000300.SS` price index | 11 Mar 2021–30 Dec 2024 | 925 |
| SSE Composite | `000001.SS` price index | 5 Jan 2015–30 Dec 2024 | 2,428 |
| ChiNext exposure | `159915.SZ` **ETF proxy**, not the ChiNext index | 5 Jan 2015–30 Dec 2024 | 2,428 |

**Evaluation:** 240 retrospective walk-forward prediction origins per market, **2 January–27 December 2024**, for **720 market–date forecast observations** in total. The subsequent observed trading session supplies each next-session target.

**Important qualification:** The 2024 observations were examined during research development, and one Ridge stabilization rule was introduced after observing an extreme retrospective forecast. Consequently, these are *chronologically out-of-sample forecasts at each historical origin*, but **not an untouched, preregistered, or independent final holdout**.

## 3. Methodology

1. **Causal data preparation:** Daily log returns, lagged market features, and trailing 5-/21-session volatility information. The forecast target is `252 × [ln(P[t+1]/P[t])]^2`, a **noisy proxy** for annualized next-session conditional variance.
2. **Walk-forward estimation:** Models are fitted using information available through close `t`; every training target must have matured by its forecast origin. No future return is used to predict itself.
3. **Forecast comparisons:** Student-t **GARCH(1,1)**, Random Forest, HAR-**inspired** log-variance Ridge (**not canonical HAR-RV**), a **fixed 50/50 Ridge–RF combination** (**not** adaptive weighting or a copula), and a 21-session benchmark.
4. **Loss and uncertainty:** Same-date annualized-variance **RMSE**, **MAE**, and **QLIKE**, plus paired five-session moving-block bootstrap intervals on differences in QLIKE versus the historical benchmark.
5. **Tail risk and economics:** Origin-specific fitted, **standardized Student-t GARCH 95% VaR**, analytical parametric ES, Kupiec/Christoffersen exceedance diagnostics, and illustrative no-leverage volatility targeting with turnover costs.

The exact model parameters, origin dates, training conventions, post-hoc amendments, and limitations are documented in the [full research methodology and results](https://github.com/amitdudi04/china-volatility-modeling/blob/research/phase1-2-data-timing-2026-10-12/docs/RESEARCH_V22_ARCHIVED_RESULTS.md).

## 4. Verified forecasting results

**2024 retrospective results; 240 observations per market. Lower RMSE and QLIKE are better. RMSE is measured on annualized *variance*, not volatility.**

| Market | Lowest RMSE model (RMSE) | Lowest QLIKE model (QLIKE) |
| --- | --- | --- |
| CSI 300 | Ridge–RF 50/50 (**0.1423**) | GARCH(1,1) (**−2.4822**) |
| SSE Composite | GARCH(1,1) (**0.1306**) | GARCH(1,1) (**−2.5192**) |
| ChiNext ETF proxy | Random Forest (**0.8323**) | GARCH(1,1) (**−1.2169**) |

**Interpretation:** GARCH achieves the lowest QLIKE in all three markets, but the best RMSE model varies. Its bootstrap QLIKE improvement over the 21-session benchmark has a 95% interval below zero for CSI 300 and SSE Composite, **not** for ChiNext. This evidence is *conditional on retrospective model development*, without a completely independent holdout or a comprehensive adjustment for model selection/multiple comparisons.

The Ridge specification performs poorly on QLIKE in stressed conditions, and its fixed-weight ensemble is **not consistently superior**. Negative outcomes are reported rather than removed.

## 5. Tail-risk findings

The corrected tail-risk study refits the Student-t GARCH distribution **at each of the 720 forecast origins** and uses origin-specific fitted location, conditional scale, and degrees of freedom. The Student-t quantile is standardized to unit innovation variance rather than using an arbitrary fixed degrees-of-freedom assumption.

**95% fitted Student-t GARCH VaR diagnostics (2024):**

| Market | Exceedances / 240 | Kupiec p-value | Christoffersen independence p-value | Joint conditional-coverage p-value |
| --- | ---: | ---: | ---: | ---: |
| CSI 300 | 7 | 0.1094 | 0.5157 | 0.2248 |
| SSE Composite | 12 | 1.0000 | 0.2599 | 0.5301 |
| ChiNext ETF proxy | 8 | 0.2084 | 0.4566 | 0.3436 |

The tests **do not reject** these VaR coverage hypotheses at the 5% level; this **does not prove** that tail risk is universally calibrated or that the models provide adequate capital protection. **Parametric Expected Shortfall forecasts were generated, but no formal ES backtest was completed.** The separate exploratory Gaussian VaR results are also retained and clearly distinguished.

## 6. Economic significance: mixed results

Illustrative 2024 strategy: exposure `min(1, 20% annual volatility target / predicted annual volatility)`, **no leverage**, idle cash earning zero, and **10 basis points per one-way traded unit**. Returns use the archived price series, not verified investable dividend-adjusted total returns.

| Market | Static annualized return | GARCH-managed annualized return | Static max drawdown | Managed max drawdown |
| --- | ---: | ---: | ---: | ---: |
| CSI 300 | 19.08% | 17.44% | −14.41% | −14.41% |
| SSE Composite | 15.83% | 11.77% | −14.73% | −14.73% |
| ChiNext ETF proxy | 20.97% | 5.12% | −25.50% | −17.54% |

The managed strategy did **not** improve absolute returns in these markets. It reduced ChiNext drawdown, but at a substantial return and Sharpe-ratio cost; CSI 300 and SSE did not show a maximum-drawdown improvement in this sample. The simulation omits some real-world implementation frictions and is **not a directly tradable portfolio backtest**.

## 7. Reproducibility and research materials

**Current verified research files are on the v2.2 branch:**

| Resource | Contents |
| --- | --- |
| [Full research report](https://github.com/amitdudi04/china-volatility-modeling/blob/research/phase1-2-data-timing-2026-10-12/docs/RESEARCH_V22_ARCHIVED_RESULTS.md) | Methodology, complete forecast losses, uncertainty, VaR/ES, economic findings, limitations |
| [Verified results and offline dashboard](https://github.com/amitdudi04/china-volatility-modeling/tree/research/phase1-2-data-timing-2026-10-12/outputs/research_v22) | 720 saved forecasts, source-hash manifests, model comparisons, risk and economic results, `dashboard.html` |
| [Senior finance-research review](https://github.com/amitdudi04/china-volatility-modeling/blob/research/phase1-2-data-timing-2026-10-12/docs/FINAL_PANEL_REVIEW.md) | Critical appraisal and academic/interview readiness |
| [File-by-file structural audit](https://github.com/amitdudi04/china-volatility-modeling/blob/research/phase1-2-data-timing-2026-10-12/docs/FINAL_FILE_AUDIT.csv) | Tracked-file checksums, classifications, parse results, legacy alerts |
| [Research protocol](https://github.com/amitdudi04/china-volatility-modeling/blob/research/phase1-2-data-timing-2026-10-12/config/research_protocol_v22.json) | Frozen archived-data scope and disclosed exploratory amendments |
| [Code and reproducibility tests](https://github.com/amitdudi04/china-volatility-modeling/tree/research/phase1-2-data-timing-2026-10-12/tests) | Chronology, actual GARCH, source integrity, risk, costs, and publication-claim checks |

To run the **corrected** code, first check out the research branch; do **not** run `main`'s legacy `main.py` and treat its reports as v2.2 results.

```bash
git clone https://github.com/amitdudi04/china-volatility-modeling.git
cd china-volatility-modeling
git switch research/phase1-2-data-timing-2026-10-12
python -m pip install "numpy>=1.24,<3" "pandas>=2,<3" "scikit-learn>=1.4,<2" "arch>=7,<9"
python -m unittest discover -s tests -v
python main.py --verify-v22
python main.py --run-gui
```

The final offline dashboard uses bundled, checksum-verified research outputs. Research branch checks passed **57 automated tests** on both the connected desktop and GitHub Actions at the audited October 2026 commit. The v2.2 files remain in a [draft pull request](https://github.com/amitdudi04/china-volatility-modeling/pull/1); they have **not been merged into the default branch**.

## 8. Research limitations and integrity

- **No untouched post-2024 holdout:** 2024 is a retrospective, already-examined evaluation sample.
- **Instrument differences:** CSI 300 begins in 2021, and the ChiNext instrument is an ETF proxy with an unverified total-return adjustment.
- **No formal ES backtest:** Parametric ES and VaR exceedance/independence checks do not establish complete tail-risk validation.
- **Model scope:** Ridge is HAR-inspired, not canonical HAR-RV; the ensemble is fixed-weight, not a copula.
- **Economic realism:** The exposure sensitivity is illustrative, not an implementable net-total-return strategy.

**Conclusion:** The corrected project documents reproducible empirical risk research, including where advanced models and volatility-managed exposure **do not** outperform simple benchmarks. Its value is in transparent estimation, causal timing, rigorous comparisons, and accurately bounded financial conclusions.
