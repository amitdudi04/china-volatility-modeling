# Volatility Forecasting and Risk Dynamics in Chinese Equity Markets

**Research v2.2 — retrospective reanalysis of the original archived 2015–2024 dataset**

This repository studies whether econometric and machine-learning forecasts of **next-session conditional variance** improve out-of-sample risk forecasting and a simple volatility-targeted equity-exposure rule. This is an academic research exercise, **not** a claim of institutional-grade capital protection.

> **Research integrity:** The earlier README's descriptions of an "adaptive hybrid copula," RMSE-weighted parameters, universal Gaussian VaR failure, and universally beneficial volatility management were **not supported by the legacy implementation or its results**. These claims have been withdrawn in this reanalysis. The original legacy result files are preserved for comparison, but the corrected 2024 experiment and its limitations are documented in [Research v2.2 — full archived results](docs/RESEARCH_V22_ARCHIVED_RESULTS.md).

## Markets and archived data

| Market | Instrument actually studied | Raw historical coverage |
| --- | --- | --- |
| CSI 300 | `000300.SS` **price index** | 2021-03-11 to 2024-12-30 (925 observations) |
| SSE Composite | `000001.SS` **price index** | 2015-01-05 to 2024-12-30 (2,428 observations) |
| ChiNext | `159915.SZ` **ETF proxy** — **not** the ChiNext price index | 2015-01-05 to 2024-12-30 (2,428 observations) |

These are exactly the archived CSVs in `data/raw/`; no missing CSI history is fabricated, imputed, or assumed. The project originally predates the requested September 2026 extension, which is **not part of the completed study**. The ETF raw close must not be treated as a verified dividend-adjusted total-return series.

## Corrected forecasting design

At the close of session `t`, forecast annualized next-session conditional variance. Each model's score uses the same observed annualized next-session squared log return: `252 * log(Close[t+1]/Close[t])**2`. The one-day squared return is a **noisy** volatility proxy. Training labels are available only when their target end date is on or before the forecast origin.

Each market has **240 retrospectively evaluated 2024 origins** from January 2 to December 27. GARCH and the other models use expanding training with only information available at each forecast origin. The 2024 history has already been examined during research development and is **not an untouched or blind holdout**.

Five forecasts are compared: a 21-session trailing variance baseline, genuine one-step Student-t GARCH(1,1), Random Forest, HAR-**inspired** log-variance Ridge (**not canonical HAR-RV**), and a fixed 50/50 Ridge–RF blend (**not an adaptive copula**). The Ridge forecast has a disclosed causal magnitude guardrail that was introduced after observing a large retrospective outlier.

## Key empirical findings

**2024 retrospective results, 240 origins per market; lower is better.**

| Market | Lowest annual-variance RMSE | Lowest QLIKE |
| --- | --- | --- |
| CSI 300 | Fixed Ridge–RF blend (0.1423) | GARCH(1,1) (-2.4822) |
| SSE Composite | GARCH(1,1) (0.1306) | GARCH(1,1) (-2.5192) |
| ChiNext ETF proxy | Random Forest (0.8323) | GARCH(1,1) (-1.2169) |

Paired five-session moving-block bootstrap intervals for GARCH's QLIKE difference versus the simple 21-session baseline exclude zero for CSI 300 and SSE, but **not** for ChiNext. This is exploratory retrospective inference; model-development and multiple-comparison uncertainty remain.

**Risk:** An exploratory **Gaussian**, zero-conditional-mean 95% VaR test produced 7, 11 and 6 GARCH VaR breaches (out of 240) for CSI 300, SSE and the ChiNext ETF respectively. The Kupiec test alone cannot establish tail-risk adequacy. The original blanket claim that Gaussian VaR universally fails is not supported.

**Economics:** At an illustrative 20% annual volatility target, 100% maximum exposure and 10 bps per unit turnover, GARCH-managed annualized returns were lower than static exposure in all three markets. ChiNext drawdown fell from approximately -25.50% to -17.54%, but annualized return and Sharpe also fell. For CSI and SSE, maximum drawdowns were essentially unchanged in this 2024 sample. **No universal performance or capital-protection benefit is claimed.**

Full model-by-model RMSE/MAE/QLIKE values, block-bootstrap intervals, per-date forecasts, Gaussian and per-origin fitted Student-t VaR/parametric ES diagnostics, conditional-coverage testing, zero/5/10/20 bps sensitivity and exact calculation assumptions are in [the full v2.2 results report](docs/RESEARCH_V22_ARCHIVED_RESULTS.md) and [research v2.2 outputs](outputs/research_v22/).

## Reproduction

Install NumPy, pandas, scikit-learn, SciPy and `arch` (the tested dependency ranges appear in `.github/workflows/research-v2-tests.yml`).

```bash
python -m unittest discover -s tests -v
python scripts/run_archived_v22.py --market CSI_300 --out-dir staging/research_v22
python scripts/run_archived_v22.py --market SSE_Composite --out-dir staging/research_v22
python scripts/run_archived_v22.py --market ChiNext --out-dir staging/research_v22
python scripts/summarize_archived_v22.py --source-dir staging/research_v22 --out-dir staging/v22_summary
python scripts/assess_archived_v22_risk.py --source-dir staging/research_v22 --out-dir staging/v22_risk\npython main.py --verify-v22\npython main.py --run-gui  # opens the offline research dashboard
```

The experimental scripts refuse to overwrite existing releases. Separate scripts can reassemble an interrupted run from complete timestamp- and SHA-verified chunks. Main entry point `python main.py --run-pipeline` and the optional PyQt GUI still run the **legacy** implementation; their archived reports are not interchangeable with the corrected v2.2 results.

## Repository guide and limitations

- `config/research_protocol_v22.json`: active original-data study design and disclosed retrospective amendments.
- `src/research_design.py`: forward target construction and mature-label rules.
- `src/oos_forecasting_v2.py`: aligned benchmark, Ridge, RF, blend and actual GARCH variance forecasts.
- `src/research_statistics_v22.py`: dependence-aware paired QLIKE loss intervals.
- `src/risk_economic_v22.py`: ex-ante Gaussian VaR and simple-return cost-aware exposure analysis.
- `scripts/run_archived_v22.py`, `scripts/summarize_archived_v22.py`, `scripts/assess_archived_v22_risk.py`: reproducibility workflow.
- `docs/RESEARCH_V22_ARCHIVED_RESULTS.md`: full findings, evidence and caveats.
- `docs/FINAL_PANEL_REVIEW.md` and `docs/FINAL_FILE_AUDIT.csv`: admissions-style methodological review and tracked-file structural audit.
- `outputs/research_v22/`: saved forecasts, hashes, comparisons, fitted Student-t conditional VaR/ES, risk/economic sensitivity and self-contained `dashboard.html`.\n- `main.py --run-gui`: verified browser dashboard; `--run-pipeline` regenerates v2.2 into a new staging directory; superseded legacy pipeline requires explicit `--run-legacy-pipeline --allow-legacy-overwrite`.

**Not yet established:** an independent post-2024 holdout, **formal Expected Shortfall backtesting**, canonical HAR-RV, full risk-model calibration across multiple windows, or a realistic tradable index/ETF total-return simulation. Fitted Student-t VaR and Christoffersen diagnostics now exist; parametric ES is calculated but not formally backtested. The verified offline HTML dashboard is the default; the optional Qt interface remains environment-dependent. These limitations are part of the research record. Negative results remain valid findings.
