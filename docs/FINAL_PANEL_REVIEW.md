# Senior Finance-Research / MSc Finance Admissions Panel Review — 12 October 2026

**Subject:** `china-volatility-modeling`, research correction branch `research/phase1-2-data-timing-2026-10-12`

**Scope:** Academic, quantitative and code-review standards appropriate to a competitive MSc Finance research portfolio. This is an **independent-style technical review of the repository**, not an actual PHBS or Fudan admissions decision, a peer-reviewed publication, or a guarantee of acceptance or a numerical admissions rating.

## Executive determination

**Conditional research-portfolio readiness: STRONG IMPROVEMENT, WITH MATERIAL LIMITATIONS.**

The corrected archived-data experiment is now internally reproducible across three instruments, and the misleading legacy interpretation is explicitly archived rather than being passed off as current evidence. The repo now distinguishes actual fitted GARCH conditional variance from Ridge-style regressions, forecasts genuinely future daily variance proxies, re-estimates model parameters using information available at each forecast date, verifies 2024 observation alignment, compares forecast losses on identical sessions, computes block-bootstrap uncertainty and connects the conditional variance estimates to cautiously interpreted VaR and economic results.

The empirical sample remains archival and retrospective. A candidate should **not** describe these results as the product of a preregistered blind holdout: 2024 was examined during development and the log-Ridge safeguard was introduced after seeing an explosive 2024 forecast. The original CSI 300 price history is shorter, the ChiNext series is an ETF proxy rather than the index, and tradable total-return conventions have not been established.

**Panel disposition:** Appropriate to discuss as a genuine quantitative-finance project with explicit negative and mixed findings. Not suitable for claims of universally superior hybrid forecasting, guaranteed capital preservation, inefficiency proved by ARCH-LM, or validated institutional VaR/ES capital buffers.

## 1. File-by-file audit approach and extent

The machine-readable inventory at [FINAL_FILE_AUDIT.csv](FINAL_FILE_AUDIT.csv) lists **every tracked file at the audited Git commit**, with SHA-256, path, byte size, format-specific structural checks and a role classification (`ACTIVE_RESEARCH_V22`, `LEGACY_ARTIFACT_RETAINED_NOT_V22_EVIDENCE`, etc.). The reproducible scanner is `scripts/inventory_review_v22.py`; it checks AST parseability of Python, CSV readability, JSON parsing, UTF-8 documents, PNG headers, and DOCX package integrity. Those are **structural checks, not full manual statistical verification of every historic image/paper/table**.

Substantive forensic review focused on the finance-critical code and evidence flows:
- **Frozen data:** `data/raw/*.csv`, `config/research_protocol_v22.json`, `src/research_design.py`, `scripts/run_archived_v22.py`.
- **Forecasting:** `src/oos_forecasting_v2.py`, per-date prediction CSVs and model-loss reconciliation.
- **Statistics and risk:** `src/research_statistics_v22.py`, `src/risk_economic_v22.py`, `src/student_t_risk_v22.py`, risk aggregation scripts, VaR/ES and block-bootstrap result files.
- **Economic assessment:** simulated static and volatility-managed exposures, next-session alignment, costs, compounding, wealth peak-to-trough drawdown, and reproducible sensitivities.
- **Dissemination:** `README.md`, `main.py`, new offline dashboard, optional Qt interface, the v2.2 report, old interpretation text files, prior figures and Word report.

### Known legacy-file alerts — retained, not repaired by fabricating values

The initial exhaustive audit found a superseded Word report containing contradicted claims; three legacy `model_weights.csv` files with entirely blank columns; and a legacy CSI `regime_validation.csv` with an entirely blank column. Such archived contents must **never be represented as current v2.2 research**. The Word file was moved unchanged to `docs/legacy_results_analysis_2026_SUPERSEDED.docx`, with [LEGACY_STUDY_NOTICE.md](LEGACY_STUDY_NOTICE.md). The original `outputs/results`, `outputs/reports`, `outputs/figures` and `outputs/plots` remain historical and explicitly disclaimed in `outputs/README.md`.

## 2. Research question and dataset identity

The defensible claim is **forecasting conditional-variance proxies**, not forecasting policy shocks or proving market efficiency classifications.

| Instrument | Actual instrument identity | Raw archive | Observations | v2.2 retrospective evaluation |
| --- | --- | --- | ---: | --- |
| CSI 300 | `000300.SS` price index | 2021-03-11–2024-12-30 | 925 | 240 origins in 2024 |
| SSE Composite | `000001.SS` price index | 2015-01-05–2024-12-30 | 2,428 | 240 origins in 2024 |
| ChiNext | `159915.SZ` ETF proxy — **NOT** the ChiNext index | 2015-01-05–2024-12-30 | 2,428 | 240 origins in 2024 |

Causal target definition: for close-of-session `t`, `252 × log(P[t+1]/P[t])²`. The future squared log return is known only at the next session close. Models are trained using labels whose `target_end_date` is at or before origin `t`, and are compared on the identical 2024 dates. This is **prequential relative to the origin but retrospectively examined**.

### Important limits

The one-day squared return is noisy; CSI 300's 2021 start means much less training history; ChiNext ETF close omits dividend-adjusted total returns; 2024 is one particular volatility regime rather than a multi-cycle final holdout. The earlier proposed 2015–2026 dataset recovery was abandoned at the user's direction because the original project predates those observations; it is **not a missing prerequisite for finishing this archival reanalysis**.

## 3. Modeling conclusions

Models: annualized 21-day variance baseline, genuine per-origin fitted Student-t GARCH(1,1), seeded 40-tree Random Forest, log-variance Ridge with historical-information safeguard, and **fixed** 50/50 Ridge–RF ensemble. No adaptive weight learning or copula estimation is claimed. Ridge is HAR-inspired only; no canonical HAR-RV claim is allowed.

All numeric loss data are saved under `outputs/research_v22/`. Lower is better.

| Market | Lowest RMSE | Lowest QLIKE |
| --- | --- | --- |
| CSI 300 | Ridge–RF (0.1423) | GARCH (-2.4822) |
| SSE Composite | GARCH (0.1306) | GARCH (-2.5192) |
| ChiNext ETF | Random Forest (0.8323) | GARCH (-1.2169) |

A five-session paired moving-block QLIKE bootstrap (2,000 replications) reported GARCH-vs-trailing-baseline 95% mean-loss-difference intervals of **[-0.2395, -0.0048]** (CSI), **[-0.4041, -0.0170]** (SSE) and **[-0.2284, 0.0331]** (ChiNext). This provides conditional support for GARCH improvement over the baseline in CSI and SSE, not ChiNext, **without accounting for repeated model selection or all multiplicity**.

The Ridge predictor has poor QLIKE in all three markets, especially during stressed months. This must remain in the results rather than be hidden or smoothed away. An excellent interview answer is *why squared-error and QLIKE differ, and why model blending was not reliably beneficial*.

## 4. Tail-risk evaluation

The risk comparison properly distinguishes the initial zero-mean **Gaussian** VaR panel from subsequently fitted **Student-t** conditional GARCH VaR.

The Student-t extension refits GARCH per forecast origin and extracts contemporaneously available mean, sigma and degrees of freedom. The Student-t quantile is standardized by `sqrt((nu-2)/nu)`, rather than naively using an unstandardized SciPy t-quantile. The conditional variance series reconciles with the saved GARCH forecasts.

| Market | Student-t 95% breaches / 240 | Kupiec p | Christoffersen independence p | Joint conditional-coverage p |
| --- | ---: | ---: | ---: | ---: |
| CSI 300 | 7 | 0.1094 | 0.5157 | 0.2248 |
| SSE | 12 | 1.0000 | 0.2599 | 0.5301 |
| ChiNext ETF | 8 | 0.2084 | 0.4566 | 0.3436 |

None rejects its respective coverage null at 5% on these samples. Such nonrejection does **not** prove risk adequacy, especially with relatively few breaches, retrospective model decisions, no regulatory calibration test and no formal Expected Shortfall backtest. **Parametric Expected Shortfall forecasts exist; only descriptive breach checks are reported.** Explicitly prohibit descriptions like "universally successful institutional VaR validation."

## 5. Economic interpretation — mixed, not promotional

Next-session exposure is `min(1, 20% annual target / predicted annual volatility)`; cash yield zero, no leverage, simple-return compounding, 10 basis points per one-way traded exposure unit. The index/ETF price series are not consistent verified investable total-return portfolios.

| Market | Static annual return | GARCH-managed annual return | Static max DD | Managed max DD |
| --- | ---: | ---: | ---: | ---: |
| CSI 300 | 19.08% | 17.44% | -14.41% | -14.41% |
| SSE | 15.83% | 11.77% | -14.73% | -14.73% |
| ChiNext ETF | 20.97% | 5.12% | -25.50% | -17.54% |

CSI Sharpe improves from 0.923 to 1.023, but return declines; SSE return and Sharpe decline with no maximum drawdown benefit; ChiNext drawdown improves while return and Sharpe fall sharply. A capital-preservation story is therefore **conditional**, not general. The limitations of benchmark index returns, ETF dividends, cash financing, impact, taxes and turnover must be acknowledged.

## 6. Software and publication consistency

- The current `README.md` and `docs/RESEARCH_V22_ARCHIVED_RESULTS.md` describe the **same** archived study and refrain from promising abnormal outperformance.
- `main.py --verify-v22` validates published archived inputs/outputs and hashes; `main.py --run-gui` opens a **self-contained offline browser dashboard** (no external services); `main.py --run-pipeline` runs v2.2 again into a new directory.
- The superseded legacy pipeline must be explicitly requested and explicitly authorized to overwrite original files.
- The optional PyQt6 GUI is not required to read results and was **not successfully validated** on the connected Windows headless Qt graphics setup. It must not be claimed verified on every machine.
- Tests cover source identity, chronological labels, fixed forecasting dates, GARCH one-step scaling, walk-forward reproducibility, VaR/ES math, Christoffersen dependence, returns alignment, transaction costs, dashboard hashes and no-overwrite guards.
- A static, machine-readable file register and output provenance notices preserve older code and prior findings for reviewer traceability; they do not silently erase negative legacy findings.

## 7. Admissions-style assessment (subjective, non-official)

| Area | Panel weight | Assessment |
| --- | ---: | --- |
| Data transparency, instrument identification | 20 | 16–17: data fixed and honestly labeled; CSI history and ETF adjustment limit scope |
| Chronology and econometric implementation | 20 | 17–18: genuine GARCH and causal features; no canonical HAR |
| Forecast loss evidence and uncertainty | 20 | 15–16: common-date evaluation and block bootstrap; examined retrospective period and post-hoc Ridge guardrail |
| Risk and economic reasoning | 20 | 16–17: Student-t VaR/ES math, hit diagnostics and trading costs; no formal ES test/investable return implementation |
| Reproducibility, engineering, integrity | 20 | 17–18: checked artifacts, tests, read-only versioning and transparent corrections; legacy technical debt retained |
| **Indicative outcome** | **100** | **Approximately 82–86, conditional on independent review; NOT guaranteed 90+** |

This rubric is not an admissions acceptance probability. An academically candid 84-style project can be stronger in an interview than a superficially polished 95-style project with unverifiable causal claims.

## 8. Remaining improvements required before claiming an unqualified “90+”

1. **Genuine independent evaluation** using additional verified data or a new predefined external dataset (not possible to establish solely from previously examined 2024 files). If the archival-only scope must remain fixed, **do not claim this gate has been passed**.
2. Formal joint VaR–ES backtesting and/or ES loss evaluation, including adequate sample interpretation and liquidity/friction assumptions.
3. Robustness to alternative estimation windows, volatility proxy/horizon, crisis/stress dates, bootstrap block lengths and multiple-comparison adjustments, with all exploratory tuning explicitly disclosed.
4. A canonical HAR model only if the project wants to compare **HAR-RV** as a named methodology.
5. Independent reviewer rerun in a clean Python environment and final browser UI inspection across operating systems (tests establish data contracts, not screen-perfect rendering).

**Bottom line for MSc Finance reviewers:** The project now demonstrates the ability to discover and correct material quantitative errors, implement defensible econometrics, quantify uncertainty and acknowledge negative strategy outcomes. It should be presented as rigorous exploratory archived-market research—not as a proven institutional trading system or evidence of universal market inefficiency.
