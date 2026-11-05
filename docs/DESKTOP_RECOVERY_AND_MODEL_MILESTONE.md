# Research v2.1 — Desktop data inventory and model-engine milestone (2026-10-12)

## Source snapshot and scope
- Default GitHub branch main remains on commit c7d78bf4e59d3e1c08eedd429339398f82157f97.
- New worktree on the authorized desktop: G:\\china-volatility-research-v2 (detached working copy of the research branch).
- Legacy project remains: G:\\Volatility Forecasting, Regime Dynamics & Risk Modeling using GARCH.
- This report records local *inspection* and new forecast-engine tests, **not** a recovered v2.1 market-data release.

## Read-only desktop source inventory

| Location | Data observed | First | Last | Suitability |
| --- | --- | --- | --- | --- |
| Original China Volatility data/raw/CSI_300.csv | CSI 300 index | 2021-03-11 | 2024-12-30 | Legacy, too short |
| Original China Volatility data/raw/SSE_Composite.csv | SSE index | 2015-01-05 | 2024-12-30 | Legacy, ends 2024 |
| Original China Volatility data/raw/ChiNext.csv | 159915 ETF proxy | 2015-01-05 | 2024-12-30 | Not v2.1 index 399006 |
| Separate G:\\2- Equity Market Volatility Forecasting Using GARCH Models/data/raw_data/csi300.csv | CSI 300 index | 2021-03-11 | 2025-12-30 | Partial recovery only |
| News-Driven Alpha local data/csi300_features.csv | CSI 300 derived prices/features | 2025-08-04 | 2026-07-03 | Partial and independently derived |

The other two local volatility directories are separate codebases and should not
be silently treated as the v2.1 frozen index dataset. No complete 2015–September
2026 three-index price file was established in these inspected project caches.

## Direct provider attempts from desktop
- Yahoo Finance request of 000300.SS from 2015-01-01 through 2026-09-30: 0 prices, provider returned its generic possibly-delisted/no-price-data response. **This is not evidence that the CSI 300 index is delisted.**
- AKShare/EastMoney sh000300 from 2015-01-01 through 2026-09-30: all three bounded attempts failed with upstream connection reset.
- Both requests were executed on the authorized desktop; attempts did not write data into source archives.

## Next-session variance forecasting engine
- New src/oos_forecasting_v2.py implements a one-step next-session variance experiment. Models: rolling-21 variance, log-variance Ridge using lagged HAR-inspired features, Random Forest, equal Ridge–RF blend, and actual Student-t GARCH(1,1).
- Fit data use the mature-label rule target_end_date <= prediction origin. Full history for GARCH excludes future returns. Prediction and target dates are saved explicitly.
- Annualization and GARCH percent-return squared scaling are handled explicitly.
- A common annual-variance RMSE/MAE/QLIKE evaluator operates on same-date predictions.
- 27 local unittest tests pass, including a real GARCH convergence/forecast smoke test, Windows/Linux manifest key normalization and retrospective feature causality.
- GitHub research-v2 workflow success on commit 9e7a6f5213dd1aa68650913ddf725c3d94392242:
  https://github.com/amitdudi04/china-volatility-modeling/actions/runs/38254121091

## Local numeric smoke checks — NOT final model evidence
The script scripts/run_local_v2_diagnostics.py **only permits** the legacy
CSI 300 and SSE price-index datasets, with 2024 forecast origins. It does not
accept the legacy ChiNext ETF as the v2.1 ChiNext index.

Initial six forecast origins (2024-01-02 to 2024-01-09) ran across both
CSI 300 and SSE. After stabilizing Ridge via log-variance modeling,
eight SSE forecast origins (2024-01-02 to 2024-01-11) ran successfully.
The source code was verified, but tiny smoke samples are NOT sufficient for
model rankings, inference, or risk/strategy claims.

## Remaining research gates
1. Acquire and independently validate one consistent price-index dataset for
   all three instruments covering 2015–September 2026. Produce checksum manifest.
2. Run full pre-registered 2023–24 retrospective validation and candidate
   2025–26 evaluation on identical market forecast dates.
3. Replace heuristic Ridge baseline with canonical HAR specification if HAR
   claims are to be made; currently label it HAR-inspired log-variance Ridge.
4. Add walk-forward model comparison intervals, convergence reporting,
   VaR/ES risk analysis, transaction costs and economic exposure backtesting.
5. Reconcile dashboard and documentation against genuine regenerated results;
   keep the draft PR unmerged until these gates pass.
