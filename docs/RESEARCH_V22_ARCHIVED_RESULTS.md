# Research v2.2 — Archived 2024 Retrospective Reanalysis

**Status:** Fully computed, internally reconciled, retrospectively evaluated; **not** an untouched holdout or a validated investable strategy.

## Why the data scope changed

The original project was developed on historical market CSVs ending 2024-12-30. Source recovery through September 2026 proved infeasible from Yahoo Finance and EastMoney on both GitHub Actions and the connected desktop. At the user's direction, the study was returned to the **original archived data** instead of manufacturing missing history. This is recorded in `config/research_protocol_v22.json`. The earlier proposed v2.1 three-index/2026 expansion is superseded, not secretly treated as completed.

| Market | Actual instrument | Original data | Rows | Forecast origins |
| --- | --- | --- | ---: | ---: |
| CSI 300 | `000300.SS` CSI 300 **price index** | 2021-03-11 to 2024-12-30 | 925 | 240 |
| SSE Composite | `000001.SS` **price index** | 2015-01-05 to 2024-12-30 | 2,428 | 240 |
| ChiNext | `159915.SZ` **ETF proxy, not the ChiNext index** | 2015-01-05 to 2024-12-30 | 2,428 | 240 |

**Frozen final forecast origins:** 2024-01-02 through 2024-12-27, covering 240 observations per market. The next-session price on 2024-12-30 completes the final label. CSI 300 has shorter pre-2024 training history, a cross-market limitation. Raw `Close` is used exactly as archived; the ETF series excludes a verified dividend-adjusted total-return convention. Published legacy result CSVs remain unchanged.

## Experiment and causal timing

At market close on session `t`, returns `r_t = ln(P_t/P_(t-1))` and rolling 5/21-session return/variance features are available. The target is **annualized next-session squared log return**, `252 × r_(t+1)^2`, a noisy observable proxy for conditional variance (not a direct measurement of latent variance).

Training is strictly expanding and uses historical training rows only if their **`target_end_date <= t`**. No current `t+1` return is used before generating the forecast. All models predict the same annual-variance target, scored on identical origin dates. The original price files are not updated, backfilled, or re-downloaded.

Models and benchmarks:
- **21-session trailing squared-return variance:** simple reference, with observations available at close `t`.
- **Student-t GARCH(1,1):** genuine one-step ahead conditional variance, fitted through close `t` (returns expressed in percent for arch estimation; rescaled to annual decimal variance).
- **Random Forest:** 40 trees, seeded, expanding mature-label training, no globally fitted scaler.
- **Log-variance Ridge:** HAR-inspired lagged features; **NOT a canonical HAR-RV model**.
- **50/50 Ridge / Random Forest blend:** fixed convex average of their variance predictions, **NOT adaptive weighting and NOT a copula**.

**Important post-hoc amendment:** A first full 2024 diagnostic exposed an extreme Ridge log-variance prediction (annual variance above 873) in October 2024. Subsequently a **causal, training-history-only** magnitude guardrail was added to the Ridge forecast and the ensemble was rerun. This adjustment was prompted by previously observed evaluation data. All model results are **retrospective** and must not be presented as independent, blind, or preregistered validation. This also makes the results exploratory where model development was influenced by the retrospective period.

## Full 2024 model-loss results

All quantities use the annualized **variance** target. RMSE/MAE values are **not volatility RMSE**. Lower loss is better. QLIKE is `ln(v_pred)+v_realized/v_pred` evaluated on the same 240 dates.

| Market | Model | RMSE | MAE | QLIKE |
| --- | --- | ---: | ---: | ---: |
| CSI 300 | GARCH(1,1) | 0.149850 | 0.050884 | **-2.482182** |
| CSI 300 | Random Forest | 0.145817 | 0.049043 | -2.461559 |
| CSI 300 | 21-session baseline | 0.160989 | 0.057207 | -2.370183 |
| CSI 300 | 50/50 Ridge–RF | **0.142314** | 0.042396 | -2.254125 |
| CSI 300 | Log-variance Ridge | 0.152916 | **0.040227** | 0.250500 |
| SSE | GARCH(1,1) | **0.130621** | 0.046643 | **-2.519237** |
| SSE | Random Forest | 0.131649 | 0.051114 | -2.453540 |
| SSE | 21-session baseline | 0.138313 | 0.049299 | -2.345413 |
| SSE | 50/50 Ridge–RF | 0.131751 | 0.041519 | -2.214136 |
| SSE | Log-variance Ridge | 0.139619 | **0.037542** | 5.513494 |
| ChiNext ETF | GARCH(1,1) | 0.899570 | 0.249584 | **-1.216900** |
| ChiNext ETF | Random Forest | **0.832338** | **0.207826** | -1.186441 |
| ChiNext ETF | 21-session baseline | 0.949133 | 0.271437 | -1.140991 |
| ChiNext ETF | 50/50 Ridge–RF | 0.953906 | 0.233058 | -0.776553 |
| ChiNext ETF | Log-variance Ridge | 1.137654 | 0.269596 | 10.157529 |

The rankings are loss-sensitive: GARCH has the lowest QLIKE for **all three** markets, whereas the ensemble has the lowest CSI RMSE and Random Forest has the lowest ChiNext RMSE. Log-variance Ridge has particularly poor QLIKE, so the claim that blending methods universally improves predictive quality is **not supported**.

## Paired dependence-aware forecast-loss uncertainty

Circular moving-block bootstrap, block length 5 sessions, 2,000 replications, seed 2026. Difference = `model QLIKE - 21-session baseline QLIKE`. Negative favours the model. Percentile intervals are **conditional** on already-observed model development, a single chosen horizon and multiple comparisons.

| Market | Model | Mean QLIKE difference | 95% interval | Excludes zero? |
| --- | --- | ---: | --- | --- |
| CSI 300 | GARCH | -0.111999 | [-0.239532, -0.004794] | Yes |
| CSI 300 | Random Forest | -0.091376 | [-0.316635, 0.086749] | No |
| SSE | GARCH | -0.173824 | [-0.404136, -0.017049] | Yes |
| SSE | Random Forest | -0.108127 | [-0.339553, 0.045647] | No |
| ChiNext ETF | GARCH | -0.075909 | [-0.228379, 0.033067] | No |
| ChiNext ETF | Random Forest | -0.045450 | [-0.208654, 0.098547] | No |

These intervals suggest improvements over the simple benchmark for GARCH in CSI 300 and SSE under these retrospective design assumptions. They are **not** a substitute for an untouched external validation period, and should not be presented as universally significant after multiple-comparison adjustment.

## Exploratory Normal VaR (95%)

These values concern **Gaussian** next-session VaR with zero conditional mean and predicted daily sigma `sqrt(v_ann/252)`. They do **not** test fitted Student-t VaR, Expected Shortfall, Christoffersen independence, or DQ coverage. Expected breaches: approximately 12/240.

| Market | GARCH breaches | RF breaches | Rolling baseline breaches | Ridge breaches |
| --- | ---: | ---: | ---: | ---: |
| CSI 300 | 7 | 11 | 9 | 42 |
| SSE | 11 | 12 | 14 | 47 |
| ChiNext ETF | 6 | 12 | 9 | 47 |

The Kupiec test is provided per model in `normal_95pct_var_coverage.csv`. A nonrejected Kupiec test is **not proof** of safe or calibrated tails, particularly without independence tests. The Ridge forecasts systematically understate some risk outcomes; the prior claim that Gaussian VaR universally fails on every market is contradicted by this preliminary evaluation.

## Per-origin fitted Student-t VaR and parametric Expected Shortfall (new)

The saved Student-t GARCH study refits **GARCH(1,1) with Student-t conditional innovations at each of the 240 2024 forecast origins per market**, recovers that origin's conditional mean, conditional volatility and fitted degrees of freedom, and verifies its variance against the original saved annualized GARCH forecast. The Student-t conditional quantile is rescaled to **unit innovation variance** using `sqrt((nu - 2)/nu)`; no fixed arbitrary degrees-of-freedom assumption is used. The lower-tail parametric conditional Expected Shortfall is computed analytically. Its performance is **not formally backtested**.

| Market | VaR breaches / 240 | Kupiec p-value | Christoffersen independence p-value | Joint conditional-coverage p-value |
| --- | ---: | ---: | ---: | ---: |
| CSI 300 | 7 | 0.1094 | 0.5157 | 0.2248 |
| SSE Composite | 12 | 1.0000 | 0.2599 | 0.5301 |
| ChiNext ETF proxy | 8 | 0.2084 | 0.4566 | 0.3436 |

These tests do not reject the hypothesized Student-t VaR coverage behavior at the 5% level; **nonrejection is not proof of calibrated tail risk, and neither a small number of exceedances nor an examined retrospective sample can validate ES**. See `outputs/research_v22/student_t_garch_tail_forecasts.csv`, `student_t_var_backtests.csv`, `student_t_es_descriptive.csv` and `student_t_tail_manifest.json`. All 720 rows and saved input forecast hashes are verified before inclusion.

## Illustrative long-only volatility targeting: 10 bps per traded unit

At the close of `t`, determine exposure for session `t+1` as `min(1, 0.20 / sqrt(predicted_ann_variance))`. Cap leverage at 100%; idle cash return zero. Charge 10 basis points per unit change in exposure and compound **simple returns** calculated as `exp(log_return)-1`. Max drawdown uses wealth relative to prior maximum, including initial wealth of 1.

| Market | Static annual return | GARCH-managed annual return | Static max DD | Managed max DD | Static Sharpe | Managed Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CSI 300 | 19.08% | 17.44% | -14.41% | -14.41% | 0.923 | 1.023 |
| SSE | 15.83% | 11.77% | -14.73% | -14.73% | 0.833 | 0.762 |
| ChiNext ETF proxy | 20.97% | 5.12% | -25.50% | -17.54% | 0.639 | 0.336 |

**Mixed result:** CSI risk-adjusted Sharpe improves, but absolute return declines and drawdown remains the same. SSE Sharpe and return decline with no drawdown benefit. ChiNext drawdown declines, but its annualized return and Sharpe also decline materially. Consequently a claim of **universal capital preservation or strategy superiority would be false**. These are not directly implementable index trades: tracking products, transaction-cost assumptions, ETF distributions, cash interest, and tax/market impact are not fully represented. Source raw prices are not verified adjusted total-return prices. Sensitivities at 0/5/10/20 bps and GARCH vs RF signal models are saved separately.

## Reproducibility and artifacts

Requirements: Python, NumPy, pandas, scikit-learn, `arch`, SciPy. Use the pinned bounds from `.github/workflows/research-v2-tests.yml`.

```bash
python -m unittest discover -s tests -v
python scripts/run_archived_v22.py --market CSI_300 --out-dir staging/research_v22
python scripts/run_archived_v22.py --market SSE_Composite --out-dir staging/research_v22
python scripts/run_archived_v22.py --market ChiNext --out-dir staging/research_v22
python scripts/summarize_archived_v22.py --source-dir staging/research_v22 --out-dir staging/v22_summary
python scripts/assess_archived_v22_risk.py --source-dir staging/research_v22 --out-dir staging/v22_risk
```

These scripts **refuse to overwrite existing releases**. On a rerun use new staging directory names, then compare hashes before publishing. The saved final archival release under `outputs/research_v22/` has **3 × 240 per-origin forecast rows**, manifests, full loss tables, bootstrap confidence intervals, Gaussian VaR diagnostics, and exposure-cost sensitivities. ChiNext's complete 240-row forecast was independently reproduced by four 60-row batches with **zero numerical difference**.

**Remaining limitations:** no new post-2024 holdout, no canonical HAR-RV implementation, no **formal ES backtest** (Student-t VaR and Christoffersen diagnostics are now available), no transaction-level or dividend-adjusted investable backtest, and the old GUI/reports have not yet been regenerated from v2.2 outputs. The 2024 period was examined during model correction; interpret all rankings as exploratory retrospective evidence. No admissions score should be inflated solely by adding these files.
