# Research v2 — Phase 1/2 provenance and evaluation protocol

> **Protocol amendment 2026-10-12:** This document records the original legacy data inventory and the earlier v2.0 candidate design. For the corrected research v2.1, ChiNext is the actual price index `399006.SZ`, not the legacy ETF proxy `159915.SZ`. See [DATA_RECOVERY_V2.md](DATA_RECOVERY_V2.md) and [research_protocol_v2.json](../config/research_protocol_v2.json). No v2 models were fit before this amendment.

Status: **provisional protocol, not a rerun of the published research** (2026-10-12).

## Legacy snapshot / immutable comparison point
- Default branch `main` at commit `c7d78bf4e59d3e1c08eedd429339398f82157f97`.
- No existing raw CSV, chart, result, README, GUI file, or model script has been modified on this branch.
- Old forecast RMSE/correlation/Sharpe/VaR outcomes must not be described as v2 findings.
- The existing results through 2024 have been inspected, so 2023–2024 is a retrospective validation period, **not an untouched holdout**.

## Observed raw market-file inventory
| File | Git blob SHA at snapshot | Rows | First session | Last session | Data caveat |
| --- | --- | ---: | --- | --- | --- |
| `data/raw/CSI_300.csv` | `ded070ea74bc891063adaa08281f852d595e1e7c` | 925 | 2021-03-11 | 2024-12-30 | Earlier CSI history missing |
| `data/raw/SSE_Composite.csv` | `27b5734880b2bba2770122cb589c2be7bb428a60` | 2428 | 2015-01-05 | 2024-12-30 | Existing 2015–24 sample |
| `data/raw/ChiNext.csv` | `72c1117c8c650144c90af70aac0f6f4da0b77a5e` | 2428 | 2015-01-05 | 2024-12-30 | This is ticker 159915.SZ, **an ETF proxy** |

Observed on existing raw CSVs: dates strictly ascending and unique; no empty cells, nonpositive closes, or elementary OHLC rule violations. These checks do **not** establish external source accuracy, corporate-action correctness, or complete trading-calendar coverage.

## Market contract
- CSI 300: `000300.SS` price index.
- SSE Composite: `000001.SS` price index.
- ChiNext: `159915.SZ`, an ETF proxy, **not** the price index. If a pure index comparison is desired, source and verify the relevant index separately.
- Research session timezone: `Asia/Shanghai`.
- Existing `Close` series is a legacy input, not necessarily an adjusted total-return series. Do not mix vendor adjustment methods in new history.
- Collect provider, download UTC timestamp, instrument identity, field definitions, adjustment setting, price range and both SHA-256 and Git blob hashes when new data is acquired.
- Freeze a coherent overlapping sample for cross-market comparisons. The CSI history deficiency is a **data-readiness blocker**, not a reason to silently drop it.

## Forecast / target contract (implemented in `src/research_design.py`)
At close t:
- Features are known through return r[t], with r[t] = log(P[t]/P[t-1]).
- **Primary**: annualized conditional variance for next session, evaluated against 252*r[t+1]^2.
- **Secondary**: forward five-session RMS volatility, sqrt(252/5 * sum(r[t+j]^2, j=1..5)).
- Each row stores both `origin_date` and `target_end_date`.
- Training may only include observations whose targets are completed by the forecast origin.
- A split includes a label only when its origin and end date are both inside that split. Thus 5-day labels crossing a boundary are excluded.
- Rolling volatility calculated from past returns is a **feature**, not a future target.
- Never backfill missing observations or substitute current-day VaR targets for ex-ante risk forecasts.
- Score all models against the **same** target on identical dates. Keep separate target horizons in separate reports.

## Proposed chronology — blocked until price extensions validated
- Development: 2015–2022.
- Chronological retrospective validation: 2023–2024; the period has already been examined.
- Candidate new post-legacy evaluation: 2025-01-01 to 2026-09-30, subject to verifiable source data and advance fixation of model choices.
- A new period is not automatically untouched: document any use for tuning explicitly.
- 2025–2026 was **not** sourced or evaluated as part of this commit.

## Tests and run instructions
```bash
python -m unittest discover -s tests -v
```
This first test suite checks synthetic target formulas, five-session horizon alignment, prefix truncation (no future features), split crossing, training label maturity, duplicates and invalid prices. Passing these tests does not prove that the legacy model pipeline is leakage-free.

## Phase 2 remaining and research release gate
1. Recover source-verifiable CSI history and provenance-checked 2025–2026 observations for all three instruments.
2. Verify price adjustment/corporate actions, exchange-calendar alignment, and source consistency.
3. Generate instrument-wise and common-date research datasets; audit sample sizes and train/validation/test completeness.
4. Integrate research v2 targets with HAR / actual rolling GARCH / Random Forest code **without modifying the legacy claims in place**.
5. Run the full code, economic backtest, and reporting pipeline; only then replace any model-performance conclusions.
