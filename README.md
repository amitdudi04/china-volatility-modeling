# Quantitative Volatility Forecasting & Regime Dynamics Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-orange)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-green)
![Arch](https://img.shields.io/badge/Stats-ARCH-yellow)

An institutional-grade quantitative research framework designed to forecast financial volatility, dynamically map structural regime transitions, and evaluate extreme tail-risk matrices. By architecting a unified analytical pipeline, it decouples pure statistical persistence from non-linear crisis shocks natively across equity manifolds.

## 🎯 Project Objective
The central objective of this system is to dynamically evaluate and integrate classical parametric architectures (e.g., GARCH) alongside non-linear machine learning ensembles (Random Forest). It executes an adaptive **Hybrid Copula** framework that mathematically re-weights predictions across high-volatility and low-volatility structural regimes, offering robust capital preservation strategies exclusively tailored for emerging and structurally inefficient distribution networks (e.g., CSI-300, ChiNext).

## 🧠 Architectural Models
This repository mathematically binds three separate projection layers securely into a unified tracking metric:
- **Baseline Parametric (GARCH)**: Standard variance persistence tracking (`EGARCH`, `GJR-GARCH`) capturing volatility clustering statically.
- **Machine Learning (Random Forest)**: Evaluates non-linear shock decays iteratively over a localized rolling `1000-day` operational matrix, actively mapping localized structure deviations via lagged inputs.
- **Hybrid Adaptive Engine**: Analytically evaluates full-spectrum root-mean-square error vectors (`Global Inverse-RMSE`) dynamically aggregating orthogonal outputs together.

## 🔬 Key Results & Evaluation Telemetry
The repository rigidly generates the following key performance bounds natively per market:
* `model_comparison.csv` — Comprehensive structural validation (`RMSE`, `MAE`, `Pearson Correlation`) continuously ranking outputs geometrically.
* `regime_performance.csv` — Isolates the tracking error precisely across `High Volatility` vs `Low Volatility` manifolds decoupling crisis-survival behavior natively.
* `validation_report.csv` — Statistically validates the Value-at-Risk parameters utilizing robust unconditional spacing limits (`Kupiec`, `Christoffersen`, and `Dynamic Quantile` logic limits).
* `interpretation.txt` — Automatically mathematically translates raw tracking datasets into formatted, detached academic prose strictly engineered for institutional review.

## 🖥 Bloomberg-Style Diagnostics Terminal
The infrastructure utilizes a native `PyQt6` dashboard actively ingesting the downstream `.csv` telemetry strings bypassing image-rendering errors natively. Execution tracks entirely mathematically utilizing `matplotlib FigureCanvas`.

*To launch the GUI:*
```bash
python main.py --run-gui
```
> The dashboard enforces rigorous `[400:600]` width bounds cleanly separating quantitative tracking layers (`Sharpe`, `VaR`, `Kupiec`) from dark-mode graphical telemetry globally.

### ⚙ Execution Protocol
The pipeline sequentially maps arrays continuously handling trailing `NaNs` systematically. To execute the target forecasting backbone without UI deployment natively:
```bash
python main.py --run-pipeline
```

## 📂 Structural Tree
```text
outputs/
├── plots/                  # Visual algebraic tracking arrays
├── reports/                # Academic translation and formatting limits
└── results/                # Core temporal matrices (.csv)
```
*Note: Ensure identical raw `.csv` extraction inputs map sequentially into `data/raw/` pre-booting.*
