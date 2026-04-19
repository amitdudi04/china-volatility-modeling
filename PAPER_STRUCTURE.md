# RESEARCH PAPER: Volatility Forecasting, Regime Dynamics, and Risk Modeling in Chinese Equity Markets

## CORE RESEARCH QUESTION
> **How do volatility forecasting models perform across policy-driven market regimes in China, and do improvements in statistical forecasting accuracy translate into economically meaningful gains in risk management?**

---

### 1. Introduction
*   **Context:** Uniqueness of the Chinese equity market (strong government intervention, retail investor dominance, high policy sensitivity).
*   **Problem Statement:** Addressing the difficulty of forecasting volatility in an inefficient, policy-driven market characterized by frequent regime shifts.
*   **Objectives:** Evaluate the economic value of various volatility forecasting models, including ML architectures, in risk management specifically for Chinese indices.
*   **Contributions:** Detailed empirical validation of regime-dependent forecasting behavior, explicitly testing the limit of standard GARCH vs non-linear ML techniques.

### 2. Literature Review
*   Volatility clustering & GARCH family models.
*   Regime-switching methodologies (Markov implementations).
*   Behavioral finance in China, the 2008 & 2015 cycles, and COVID effects.
*   Machine learning applications in quantitative finance vs traditional econometrics.
*   The gap: Translating statistical accuracy (RMSE/QLIKE) into economic value (reduced VaR violations).

### 3. Methodology
*   **Data Generation:** CSI 300, SSE Composite, and ChiNext spanning 2005 to 2025.
*   **Econometric Infrastructure:** ARCH-LM tests for structural persistence, ADF for integration, and Jarque-Bera for heavy tails.
*   **Modeling Engine:** ARCH(1), GARCH(1,1), EGARCH(1,1), and GJR-GARCH. Evaluation via AIC/BIC and Max-LogLikelihood.
*   **Regime Dynamics:** 2-State Markov Regression linking policy shifts to conditional variance states.
*   **Forecasting Scheme:** Rolling window ($N=1000$) 1-step ahead estimation bridging to out-of-sample ML (RF baseline) evaluations.

### 4. Hypotheses Revisited
1.  **H1 (Clustering):** Strong volatility clustering where GARCH > naïve estimators.
2.  **H2 (Instability):** China exhibits more frequent volatility regime-shifting than developed counterparts.
3.  **H3 (ML Limitations):** ML excels in stable nonlinearity but abruptly breaks down during policy-driven volatility explosions.
4.  **H4 (Tail Underestimation):** Conventional GARCH chronically breaches standard VaR and ES thresholds due to excess skewness.
5.  **H5 (Inefficiency):** Persistent patterns are directly reflective of retail speculative flows vs SOE interventions.
6.  **H6 (Economic Application):** The translation from lower numerical loss to improved risk-adjusted downside protection is tangible and verifiable.

### 5. Results & Empirical Analysis
*   *Reference auto-generated `metrics/` outputs in the repository.*
*   Discuss AIC/BIC minimizations.
*   Plot review: `figures/` directory visual evidence of the 2015 and 2020 regimes.

### 6. Economic Interpretation
*   **Key Question Answered:** "Does better volatility forecasting reduce real financial risk in China?"
*   Analyzing Kupiec test (VaR Violations). An underestimating model leads to devastating capital depletion. Better modeling in China explicitly prevents systemic collapse for institutional traders during state-driven selloffs.

### 7. Conclusion
*   Summary of findings.
*   The limitation of statistical models in a structurally controlled, emerging-nature market.
*   Final statement confirming the central hypothesis/SOP line.

---
**SOP LINE:**
> "My research analyzes how volatility forecasting models behave under policy-driven regime shifts in Chinese equity markets and whether improved statistical accuracy translates into meaningful gains in risk management."
