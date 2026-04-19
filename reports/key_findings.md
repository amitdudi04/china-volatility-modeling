# 📊 Key Findings: Volatility & Risk Modeling in China

## 1. Strong Volatility Clustering
* Empirical testing (ARCH-LM, Jarque-Bera) confirms significant leptokurtic behavior and variance memory.
* High retail participation translates structural news into prolonged, persistent volatility rather than immediate efficient repricing.

## 2. Frequent Regime Shifts
* Markov-Switching models identified distinct variance states strongly influenced by policy interventions and structural breakdowns.
* The "Crisis" regime historically maps directly to 2008 and 2015, highlighting exogenous intervention shocks versus naturally expanding risk.

## 3. GARCH Underestimates Tail Risk
* Standard normal GARCH failed the Christoffersen Conditional Coverage test across indices.
* When violations of the 95% Value at Risk (VaR) threshold occurred, they clustered dependently, meaning losses cascaded day-over-day during crises.

## 4. ML Unstable in Crises
* While Random Forest and XGBoost architectures outperformed GARCH (RMSE/QLIKE) during stable bull markets by recognizing complex lag features...
* They experienced severe failure bounds during unseen policy interventions, lacking the long-memory persistence parameter native to GARCH models.

## 5. Regime Models Improve Stability
* Separating statistical analysis into 2-State probability matrices effectively isolated structural reality from generalized "noise".

## 6. Volatility Targeting Improves Sharpe & Drawdown
* Inverse-volatility dynamic exposure (reducing portfolio weight when forward-GARCH variance predicts spikes) successfully bridged the gap between econometrics and action.
* Dynamically managed portfolios consistently reduced Maximum Drawdowns compared to 100% passive strategies.
