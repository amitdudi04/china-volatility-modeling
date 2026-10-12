# Research v2 — source-data recovery and model-ready datasets

**This is an input-recovery stage, NOT a new model-result release.**
The legacy main branch, `data/raw` and `outputs/results` remain unchanged.

## Markets and provider mapping

| Market | Source ticker | Instrument type | Research series |
| --- | --- | --- | --- |
| CSI 300 | `000300.SS` | Price index | Yahoo `Adj Close` |
| SSE Composite | `000001.SS` | Price index | Yahoo `Adj Close` |
| ChiNext | `159915.SZ` | E Fund ChiNext **ETF** proxy | Yahoo `Adj Close` |

The new workflow uses `yfinance.download(auto_adjust=False, repair=False, threads=False)`
to retain both the vendor's original `Close` (stored as `Raw_Close`) and
`Adj Close` (stored as the research `Close`). It requests 2015-01-01 through
2026-09-30 inclusive, by specifying an exclusive 2026-10-01 end date.

**Important:** Vendor-adjusted prices are not necessarily a verified investable total-return
series. The research must explicitly assess ETF distributions, corporate actions and
vendor corrections before presenting strategy returns as implementable.

## Download and validation

```bash
python -m pip install "numpy>=1.24,<3" "pandas>=2,<3" "yfinance>=0.2.54,<1"
python -m unittest discover -s tests -v
python scripts/fetch_research_data_v2.py --output-dir staging/research-v2-20261012
```

The external workflow is `Research v2 - external data recovery`. Successful runs
upload 30-day artifacts containing the three source CSVs, 18 model-ready
market × horizon × period CSVs, and `manifest.json` with SHA-256 hashes and
provider settings.

The script refuses preexisting output directories and does not modify legacy data.
It blocks data that is incomplete, late/short, duplicated, out-of-order,
nonfinite, nonpositive, includes weekends, has inconsistent unadjusted OHLC,
or contains adjusted daily absolute log returns above 35% without review.
It requires data from early 2015 through at least September 25, 2026,
at least 2,600 observations in each source and common trading calendar, and
adequate mature labels in each chronological sample.

All three series are **intersected on their common actual observed trading sessions**
before one- and five-session future labels are constructed. The labels carry
prediction-origin and target-end dates and cannot cross a declared split boundary.

## Chronology

- Development: 2015–2022.
- Retrospective validation: 2023–2024, **already inspected in the legacy project**.
- Candidate new evaluation: 2025–Sep 2026. Newer does **not** automatically mean
  untouched; preserve a record of all decisions/tuning on this period.

## Research caveats and next gate

The one-day squared-return target is a noisy proxy for latent conditional variance.
Five-session forward RMS targets overlap and require appropriate uncertainty
estimators / leakage-safe training maturity. A complete provider download
does not establish successful OOS GARCH/ML fitting or strategy superiority.

**Do not merge this as a finished results release** until checks on sourced
history, 2025–26 sampling, forward forecasts, VaR, and economic simulations pass.
This workflow cannot guarantee Yahoo availability; on network or provider failure,
it must fail visibly rather than synthesize missing prices. Staged GitHub Actions
artifacts expire after 30 days; make an independently verified archival copy if
using them for publication.
