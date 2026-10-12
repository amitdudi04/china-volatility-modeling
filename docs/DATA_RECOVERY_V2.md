# Research v2.1 — coherent three-price-index recovery

**Protocol amended on 2026-10-12, before fitting corrected models.**
The legacy raw data, results, and main branch are unchanged.

## Change from the legacy study

The legacy ChiNext series was ticker `159915.SZ`, an ETF proxy. Its EastMoney
forward-adjusted-price endpoint failed in the hosted recovery environment.
For a cleaner **three market-index comparison**, v2.1 uses the *actual ChiNext
price index* `399006`, accessed as `sz399006`.

| Market | EastMoney symbol | Source | Return series |
| --- | --- | --- | --- |
| CSI 300 | `sh000300` | AKShare `stock_zh_index_daily_em` | price-index close log returns |
| SSE Composite | `sh000001` | same | price-index close log returns |
| ChiNext index | `sz399006` | same | price-index close log returns |

These are **price-index** series, not total-return indices or directly investable
ETFs. The three series are collected from one historical-data interface.
No Yahoo/ETF histories are spliced into this panel.

Source documentation:
- AKShare official interface: https://github.com/akfamily/akshare/blob/main/docs/data/index/index.md
- Index provider lists ChiNext code 399006: https://www.cnindex.com.cn/eng/indices/sese/scale/index.html

## Current runnable commands

```bash
python -m pip install "numpy>=1.24,<3" "pandas>=2,<3" "akshare>=1.16,<2"
python -m unittest discover -s tests -v
python scripts/fetch_research_data_v2_akshare.py --output-dir staging/research-v2-eastmoney-20261012
```

The alternate Yahoo-only source is retained as a documented unsuccessful
attempt rather than used for the final v2.1 panel. The EastMoney-only GitHub
Actions workflow produces a distinct 30-day downloadable artifact **only when
all three instruments pass**, with three source files, 18 model-ready CSVs
(market × horizon × split), and a SHA-256 `manifest.json`.

## Acceptance gates

- Each index must have valid, complete and chronologically ordered source rows.
- Earliest date by 2015-01-15; latest date by 2026-09-25 and before 2026-10-01.
- At least 2,600 unique sessions per index, at least 2,600 shared sessions.
- No missing values, nonfinite prices, nonpositive prices, negative volumes,
  duplicate/weekend dates or unadjusted OHLC inconsistencies.
- Index price jumps larger than absolute 35% daily log return are a **stop-for-review**
  condition, not an instruction to clip or edit returns.
- Construct future one- and five-session targets on the *common actual trading calendar*
  with recorded origin and target-end dates; refuse boundary-crossing outcomes.
- Require at least 1,000 development observations (2015–2022) and at least
  300 each in retrospective validation (2023–2024) and candidate post-legacy
  evaluation (2025–Sep 2026).
- No automatic substitution of missing historical observations or source changes.

## Research limitations

The observed next-day squared-return target is a noisy proxy for latent
variance. Five-session forward realized-volatility outcomes overlap and
require leakage-safe label maturity and serial-dependence-aware uncertainty
analysis. The old 2023–24 outputs have already been examined. The new
2025–26 sample should be treated as a **candidate post-legacy evaluation**,
not claimed untouched without separate documented evidence.

A successful staged download does **not** prove the corrected forecasting,
VaR, or trading simulations are valid. Freeze data artifacts and their hashes,
then run true OOS models, diagnostics and economic evaluation before rewriting
README claims or merging the draft PR as final research.
