# Volatility Forecasting and Risk Analysis in Chinese Equity Markets

This project compares statistical and machine-learning methods for forecasting equity-market volatility in China. It uses historical prices from the CSI 300, SSE Composite, and a ChiNext ETF to evaluate forecast accuracy, estimate downside risk, and test a simple volatility-based position-sizing strategy.

The main models are GARCH(1,1), Random Forest, log-variance Ridge regression, and a fixed Ridge–Random Forest combination. A 21-session historical-variance estimate serves as the baseline.

## 1. Project overview

The project follows five steps:

1. **Prepare the data:** Calculate daily log returns, lagged features, and rolling volatility measures from historical closing prices.
2. **Forecast next-session variance:** Fit the models using only information available at each forecast date.
3. **Compare predictions:** Evaluate all forecasts against the same next-session squared-return target using RMSE, MAE, and QLIKE.
4. **Measure downside risk:** Estimate 95% Value at Risk (VaR), calculate parametric Expected Shortfall (ES), and check VaR exceedances.
5. **Test a volatility-based strategy:** Compare a fixed equity exposure with a volatility-targeted exposure after trading costs.

## 2. Dataset

The project uses the original historical price files. The three instruments have different starting dates:

| Market | Instrument used | Date range | Observations |
| --- | --- | --- | ---: |
| CSI 300 | `000300.SS` index | Mar 2021–Dec 2024 | 925 |
| SSE Composite | `000001.SS` index | Jan 2015–Dec 2024 | 2,428 |
| ChiNext | `159915.SZ` ETF proxy | Jan 2015–Dec 2024 | 2,428 |

The ChiNext series is an **ETF, not the ChiNext price index**. Its closing prices are not confirmed as dividend-adjusted total returns.

Models were evaluated at **240 dates in 2024 for each market** (720 market-date observations overall). Each forecast uses information available through that day's close and is compared with the return from the following trading session.

The evaluation is historical and walk-forward. Because the 2024 results were examined during model development, this period should **not be described as an untouched holdout**.

## 3. Models and evaluation

| Method | Purpose |
| --- | --- |
| 21-session rolling variance | Simple historical baseline |
| GARCH(1,1) with Student-t innovations | Estimate changing conditional variance |
| Random Forest | Model nonlinear relationships in lagged market features |
| Log-variance Ridge | Linear benchmark using lagged volatility-related features |
| Ridge + Random Forest (50/50) | Compare a fixed average of two model forecasts |

The models predict annualized next-session variance. The evaluation target is:

```text
252 × [log(Close[t+1] / Close[t])]^2
```

This is an observable but noisy approximation of conditional variance. RMSE and MAE measure forecast errors, while QLIKE gives a different perspective on variance-forecast quality.

The Ridge specification uses features inspired by HAR models; it is **not a standard HAR-RV implementation**. The 50/50 combination uses fixed weights and is **not an adaptive copula model**.

## 4. Forecasting results

The table below gives the lowest RMSE and lowest QLIKE model in each market. Both metrics are calculated against the same 2024 next-session variance target; **lower values are better**.

| Market | Lowest RMSE | Lowest QLIKE |
| --- | --- | --- |
| CSI 300 | Ridge–RF 50/50 (0.1423) | GARCH(1,1) (−2.4822) |
| SSE Composite | GARCH(1,1) (0.1306) | GARCH(1,1) (−2.5192) |
| ChiNext ETF | Random Forest (0.8323) | GARCH(1,1) (−1.2169) |

GARCH had the lowest QLIKE across all three markets. However, the lowest RMSE model differed by market. The Ridge model and its combination with Random Forest did not consistently improve on simpler alternatives.

A five-session moving-block bootstrap was used to compare QLIKE against the rolling-variance baseline. The estimated GARCH improvements had 95% intervals below zero for CSI 300 and SSE Composite, but not for the ChiNext ETF. These comparisons are exploratory because the evaluation period was already examined during model development.

## 5. Value at Risk and Expected Shortfall

The project fits a Student-t GARCH distribution at each forecast date to estimate 95% VaR and parametric ES. The VaR forecasts are checked using Kupiec and Christoffersen tests.

| Market | VaR breaches / 240 | Kupiec p-value | Conditional-coverage p-value |
| --- | ---: | ---: | ---: |
| CSI 300 | 7 | 0.1094 | 0.2248 |
| SSE Composite | 12 | 1.0000 | 0.5301 |
| ChiNext ETF | 8 | 0.2084 | 0.3436 |

The tests did not reject the specified VaR coverage conditions at the 5% level. This is not proof of accurate risk estimates under every market condition. **Expected Shortfall was calculated but not formally backtested.**

## 6. Volatility-based strategy

A simple comparison uses GARCH forecasts to reduce or maintain equity exposure. The strategy targets 20% annual volatility, allows no leverage, assumes zero return on unused cash, and applies 10 basis points of cost per unit of turnover.

| Market | Static annualized return | Volatility-managed return | Static max drawdown | Managed max drawdown |
| --- | ---: | ---: | ---: | ---: |
| CSI 300 | 19.08% | 17.44% | −14.41% | −14.41% |
| SSE Composite | 15.83% | 11.77% | −14.73% | −14.73% |
| ChiNext ETF | 20.97% | 5.12% | −25.50% | −17.54% |

The volatility-managed approach reduced the ChiNext drawdown, but its return was substantially lower. It did not improve maximum drawdown for CSI 300 or SSE Composite during this period. These are illustrative calculations based on the stored price series, not a fully tradable total-return backtest.

## 7. Project files and dashboard

The updated model scripts, test suite, results, and offline dashboard are currently available in the [updated project files](https://github.com/amitdudi04/china-volatility-modeling/tree/research/phase1-2-data-timing-2026-10-12).

| File or folder | What it contains |
| --- | --- |
| [Model scripts](https://github.com/amitdudi04/china-volatility-modeling/tree/research/phase1-2-data-timing-2026-10-12/src) | Feature preparation, volatility forecasts, and risk calculations |
| [Results](https://github.com/amitdudi04/china-volatility-modeling/tree/research/phase1-2-data-timing-2026-10-12/outputs/research_v22) | Predictions, model comparisons, VaR/ES outputs, and strategy figures |
| [Dashboard](https://github.com/amitdudi04/china-volatility-modeling/blob/research/phase1-2-data-timing-2026-10-12/outputs/research_v22/dashboard.html) | Offline view of forecast, risk, and strategy results |
| [Detailed results](https://github.com/amitdudi04/china-volatility-modeling/blob/research/phase1-2-data-timing-2026-10-12/docs/RESEARCH_V22_ARCHIVED_RESULTS.md) | Methods, numerical tables, and additional analysis |
| [Tests](https://github.com/amitdudi04/china-volatility-modeling/tree/research/phase1-2-data-timing-2026-10-12/tests) | Checks for data alignment, forecast calculations, risk metrics, and saved outputs |

To run the updated version locally:

```bash
git clone https://github.com/amitdudi04/china-volatility-modeling.git
cd china-volatility-modeling
git switch research/phase1-2-data-timing-2026-10-12

python -m pip install "numpy>=1.24,<3" "pandas>=2,<3" "scikit-learn>=1.4,<2" "arch>=7,<9"
python -m unittest discover -s tests -v
python main.py --verify-v22
python main.py --run-gui
```

The dashboard works offline using saved output files. **The default `main` branch still contains the earlier pipeline**; switch to the branch above to run the updated models and view their corresponding results.

## 8. Limitations

The CSI 300 history is shorter than that of the other instruments, and the ChiNext data represent an ETF rather than the price index. The 2024 evaluation is a previously examined historical period, not an independent holdout. Daily squared returns are noisy variance targets, and one Ridge forecast safeguard was adjusted after reviewing a 2024 outlier. ES has not been formally backtested, and the strategy results exclude several real-world costs and total-return adjustments.

Overall, the project shows how different volatility models compare across these three markets, how their forecasts can be applied to risk estimation, and where the results do—and do not—support a volatility-based allocation approach.
