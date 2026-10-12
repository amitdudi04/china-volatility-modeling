"""Causal one-session volatility forecast model comparisons.

Input: prepare_session_dataset(raw, horizon=1).  Every model predicts
the same annualized next-session conditional variance (not volatility).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.research_design import completed_training_rows

FEATURES = [
    "lag1_return", "lag1_squared_return",
    "trailing_5_variance_ann", "trailing_21_variance_ann",
    "trailing_5_vol_ann", "trailing_21_vol_ann",
    "trailing_5_mean_return", "trailing_21_mean_return",
]

def _clip_variance(v):
    v = float(v)
    if not np.isfinite(v):
        raise ValueError("Nonfinite variance forecast")
    return max(v, 1e-10)

def forecast_one_origin(frame, origin, min_training=252,
                        include_garch=True, n_estimators=100):
    origin = pd.Timestamp(origin)
    if origin not in frame.index:
        raise ValueError("Forecast origin not in dataset")
    required = FEATURES + ["return_at_origin","target_variance_ann","target_end_date"]
    if not set(required).issubset(frame.columns):
        raise ValueError("Missing model-ready columns")
    train = completed_training_rows(frame, origin).dropna(
        subset=FEATURES + ["target_variance_ann"])
    if len(train) < min_training:
        raise ValueError("Insufficient mature training rows")
    if (pd.to_datetime(train["target_end_date"]) > origin).any():
        raise AssertionError("Future training target")
    current = frame.loc[[origin],FEATURES]
    X, y = train[FEATURES], train["target_variance_ann"]
    if (y < 0).any() or current.isna().any().any():
        raise ValueError("Invalid causal features or variance targets")
    # A log-variance Ridge benchmark, using HAR-inspired lagged features.\n    # Not a canonical HAR-RV model.
    # Exponentiation enforces strictly positive forecasts without clipping
    # potentially large negative variance predictions to an epsilon floor.
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10))
    ridge.fit(X,np.log(np.maximum(y.to_numpy(dtype=float),1e-8)))
    raw_logvar = _clip_variance(np.exp(np.clip(ridge.predict(current)[0], -100, 100)))
    # A causal, historical-only forecast magnitude guardrail: avoid Ridge
    # extrapolations hundreds of times above observed training variance.
    # Added AFTER a 2024 retrospective diagnostic revealed explosive
    # extrapolation. This is exploratory, NOT preregistered evaluation.
    baseline = _clip_variance(frame.loc[origin,"trailing_21_variance_ann"])
    upper = max(5.0*baseline,3.0*float(np.quantile(y,0.99)),1e-5)
    lower = max(1e-8, min(baseline/10.0,float(np.quantile(y,0.01))))
    har = float(np.clip(raw_logvar,lower,upper))
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=17,
                               max_features=.8, min_samples_leaf=8, n_jobs=1)
    rf.fit(X,y)
    ml = _clip_variance(rf.predict(current)[0])
    result = {
        "origin_date": origin,
        "target_end_date": pd.Timestamp(frame.loc[origin,"target_end_date"]),
        "training_observations": len(train),
        "rolling_21_variance": _clip_variance(frame.loc[origin,"trailing_21_variance_ann"]),
        "logvar_ridge_variance": har,
        "ridge_uncapped_diagnostic":raw_logvar,
        "ridge_guardrail_upper":upper,
        "ridge_was_capped":bool(abs(har-raw_logvar)>1e-12),
        "random_forest_variance": ml,
        "ridge_rf_equal_variance": .5*(har+ml),
        "realized_next_variance": float(frame.loc[origin,"target_variance_ann"]),
    }
    if include_garch:
        from arch import arch_model
        returns = frame.loc[:origin,"return_at_origin"].dropna()
        if len(returns) < min_training:
            raise ValueError("Insufficient historical returns for GARCH")
        model=arch_model(returns.to_numpy()*100,mean="Constant",vol="GARCH",
                         p=1,o=0,q=1,dist="t",rescale=False)
        fit=model.fit(disp="off",show_warning=False)
        if fit.convergence_flag != 0:
            raise RuntimeError("GARCH did not converge")
        # Model uses 100*decimal log returns. Convert percent^2 to decimal^2.
        conditional=float(fit.forecast(horizon=1,reindex=False).variance.iloc[-1,0])
        result["garch11_variance"]=_clip_variance(conditional*252/10000)
    return result

def walk_forward(frame, eval_start, eval_end, *,
                 min_training=252, max_origins=None, include_garch=True,
                 n_estimators=100):
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Invalid chronological index")
    if max_origins is not None and max_origins < 1:
        raise ValueError("Invalid number of origins")
    candidates=frame.loc[
        (frame.index>=pd.Timestamp(eval_start))
        & (frame.index<=pd.Timestamp(eval_end))
        & (pd.to_datetime(frame["target_end_date"])<=pd.Timestamp(eval_end))]
    if max_origins is not None:
        candidates=candidates.iloc[:max_origins]
    if candidates.empty:
        raise ValueError("No eligible observations")
    rows=[forecast_one_origin(frame,t,min_training,include_garch,n_estimators)
          for t in candidates.index]
    out=pd.DataFrame(rows).set_index("origin_date")
    if not out.index.equals(candidates.index):
        raise AssertionError("Timestamp alignment mismatch")
    if out.isna().any().any():
        raise ValueError("Incomplete forecasts")
    return out

def score_variance_predictions(predictions):
    truth=predictions["realized_next_variance"].to_numpy(dtype=float)
    rows=[]
    for name in predictions:
        if not name.endswith("_variance") or name=="realized_next_variance":
            continue
        pred=predictions[name].to_numpy(dtype=float)
        if not(np.isfinite(pred).all() and np.isfinite(truth).all()
               and (pred>0).all() and (truth>=0).all()):
            raise ValueError("Invalid variance data")
        error=truth-pred
        rows.append({"model":name,"n":len(pred),
                    "rmse":float(np.sqrt(np.mean(error**2))),
                    "mae":float(np.mean(np.abs(error))),
                    "qlike":float(np.mean(np.log(pred)+truth/pred))})
    return pd.DataFrame(rows).sort_values("qlike",kind="stable").reset_index(drop=True)
