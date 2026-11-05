# Research Methodology

## Research question and instruments
This study examines the relative forecasting performance of statistical and machine-learning models for next-session variance in Chinese equity markets. Instruments are the CSI 300 price index (`000300.SS`), SSE Composite price index (`000001.SS`) and ChiNext ETF proxy (`159915.SZ`). The ETF is not the ChiNext index, and its archived closing prices are not confirmed dividend-adjusted total returns. All three price datasets end on 30 December 2024; the CSI 300 dataset begins in March 2021 while the others begin in January 2015.

## Forecast origin, information set and target
Forecast origins are the 240 market sessions from 2 January to 27 December 2024, with the next trading close providing each target. At the close of session `t`, models use only information available through that close. The annualized target is `252 × [log(P[t+1]/P[t])]^2`, a noisy next-session squared-return proxy for latent conditional variance. Each training row's target-end date must precede or equal the forecast origin; model comparisons share identical origin and target dates.

## Models and estimation
A 21-session trailing-variance estimate serves as the simple benchmark. Alternatives include fitted per-origin Student-t GARCH(1,1), Random Forest, log-variance Ridge regression using HAR-inspired historical features, and a fixed 50/50 Ridge–Random Forest average. The Ridge model is **not** canonical HAR realized variance and the blend is **not** an adaptive ensemble or copula. A Ridge historical-range guardrail was added after an extreme prediction was observed in retrospective 2024 development. Although the guardrail is computed only from historical information at each origin, its *selection* was informed by the evaluation period. The whole 2024 exercise is exploratory rather than an untouched holdout.

## Accuracy and uncertainty
Models are compared with RMSE, MAE and QLIKE for annualized **variance**, not standard deviation. A paired circular moving-block bootstrap with 5-session blocks and 2,000 replicates evaluates QLIKE differences relative to the 21-session baseline. The resulting intervals are conditional on the chosen specifications and do not fully adjust for selection across markets, metrics and models.

## Tail risk and economic analysis
The risk extension uses per-origin Student-t GARCH fitted means, conditional variance and degrees of freedom to estimate standardized 95% VaR and parametric Expected Shortfall. Kupiec unconditional coverage and Christoffersen independence/conditional coverage tests evaluate VaR violations. **No formal ES backtest is claimed.** A stylized long-only, unlevered exposure strategy targets 20% annualized forecast volatility and compares zero, 5, 10 and 20 basis-point cost scenarios. Cash is assumed to earn zero. Turnover uses changes in target exposure rather than drift-adjusted post-return holdings, and the price-index/ETF returns are not established as directly investable total-return series.

The full numerical evidence appears in [Results](RESULTS.md); the principal empirical qualifications appear in [Limitations](LIMITATIONS.md).
