# Research Limitations

The 2024 evaluation period was examined during development. The Ridge training-history guardrail was introduced after observing an extreme forecast, so subsequent results are retrospective and cannot be treated as untouched, preregistered out-of-sample evidence. The paired bootstrap does not resolve model-selection uncertainty or all multiple comparisons.

The forecast target is next-session squared log return, a noisy observable proxy rather than latent conditional variance. CSI 300 has a shorter historical training sample than SSE Composite and the ChiNext ETF. The ChiNext instrument is `159915.SZ`, an ETF proxy rather than the index, and its archived closes have no verified dividend-adjusted total-return convention. The instruments therefore differ economically as well as statistically.

Student-t GARCH VaR nonrejection at the 5% level does not prove dependable future tail-risk coverage, particularly with 7, 12 and 8 observed exceedances across 240 observations per instrument. Expected Shortfall values are parametric forecasts, not formally backtested ES results.

The economic simulation is a simplified research exercise: it assumes a zero cash yield, price-return index proxies, and proportional costs based on changes in *target* portfolio exposure. It excludes drift-adjusted rebalancing, dividends, fund expenses, borrowing and funding costs, taxes, slippage, market impact and some tradeability restrictions. It does not demonstrate achievable investment profits or universal drawdown protection.

Null and unfavorable model comparisons, including Ridge's poor QLIKE and the mixed volatility-managed return outcomes, are retained in [Results](RESULTS.md). Reproducible calculations cannot by themselves cure the foregoing external-validity constraints.
