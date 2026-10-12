"""Exploratory 2024 ex-ante Gaussian VaR and long-only exposure sensitivity.

Not a calibrated Student-t VaR or trading recommendation. Model conditional
variance is annualized; VaR uses next-session daily sigma = sqrt(v_ann/252).
Historical 2024 observations were inspected and are NOT an untouched holdout.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from scipy.special import xlogy


def align_next_session_returns(pred:pd.DataFrame, prices:pd.DataFrame)->pd.Series:
    if "Close" not in prices or "target_end_date" not in pred:
        raise ValueError("Missing price or forecast target-end date")
    if pred.index.has_duplicates or prices.index.has_duplicates:
        raise ValueError("Duplicate forecast or return dates")
    if not pred.index.is_monotonic_increasing or not prices.index.is_monotonic_increasing:
        raise ValueError("Unsorted dates")
    origin_i=prices.index.get_indexer(pred.index)
    target_i=prices.index.get_indexer(pd.to_datetime(pred["target_end_date"]))
    if (origin_i<0).any() or (target_i<0).any() or not np.all(target_i==origin_i+1):
        raise ValueError("Forecast target must be exactly the next actual market session")
    a=prices["Close"].to_numpy(float)
    if not np.isfinite(a).all() or (a<=0).any():
        raise ValueError("Invalid price levels")
    next_log=np.log(a[target_i]/a[origin_i])
    if not np.isfinite(next_log).all():
        raise ValueError("Invalid realized return")
    return pd.Series(next_log,index=pred.index,name="next_session_log_return")


def kupiec_unconditional_coverage(breaches,alpha=.05)->dict:
    b=np.asarray(breaches,dtype=bool)
    if b.ndim!=1 or len(b)==0 or not 0<alpha<1:
        raise ValueError("Invalid VaR test data")
    x=int(b.sum());n=len(b);observed=x/n
    # 0*log(0)=0 with xlogy, including exact zero/all breach edge cases.
    log_null=float(xlogy(x,alpha)+xlogy(n-x,1-alpha))
    log_alt=float(xlogy(x,observed)+xlogy(n-x,1-observed))
    lr=max(0.,2*(log_alt-log_null))
    return {"observations":n,"breaches":x,"expected_rate":alpha,
            "observed_rate":observed,"kupiec_lr":lr,
            "kupiec_pvalue":float(chi2.sf(lr,1))}


def normal_var_coverage(pred:pd.DataFrame, log_return:pd.Series,
                        alpha=.05)->pd.DataFrame:
    if not pred.index.equals(log_return.index):
        raise ValueError("Unaligned prediction and realized return origins")
    returns=log_return.to_numpy(dtype=float)
    rows=[]
    for name in pred.columns:
        if not name.endswith("_variance") or name=="realized_next_variance":
            continue
        v=pred[name].to_numpy(dtype=float)
        if (v<=0).any() or not np.isfinite(v).all():
            raise ValueError("Invalid conditional variance")
        # Zero-return conditional-mean assumption. No claim of fitted Student-t.
        cutoff=float(norm.ppf(alpha))*np.sqrt(v/252.0)
        stats=kupiec_unconditional_coverage(returns<cutoff,alpha=alpha)
        rows.append({"model":name,**stats})
    return pd.DataFrame(rows)


def economic_exposure_sensitivity(
    pred:pd.DataFrame, log_return:pd.Series,
    forecast_col:str="garch11_variance",
    target_annual_vol:float=.20,
    cost_bps:int=10,
)->pd.DataFrame:
    if not pred.index.equals(log_return.index):
        raise ValueError("Unaligned strategy dates")
    if forecast_col not in pred or not 0<target_annual_vol<=1 or cost_bps<0:
        raise ValueError("Invalid forecast/strategy settings")
    annual=np.sqrt(pred[forecast_col].to_numpy(float))
    if not np.isfinite(annual).all() or (annual<=0).any():
        raise ValueError("Invalid forecasts")
    # At close t choose t+1 exposure. Long-only, no leverage or financing.
    managed=np.minimum(1.0,target_annual_vol/annual)
    simple=np.expm1(log_return.to_numpy(float))
    if (simple<=-1).any():
        raise ValueError("Invalid simple return")
    records=[]
    for name,weights in (("static",np.ones(len(managed))),("volatility_managed",managed)):
        # Assume 100% initial exposure. Changes cost bps per unit one-way
        # turnover; cash interest, fund fees and dividends are not simulated.
        prior=np.r_[1.0,weights[:-1]]
        turnover=np.abs(weights-prior)
        net=weights*simple-(cost_bps/10000.)*turnover
        if (1+net<=0).any():
            raise ValueError("Strategy bankrupt under specified model")
        equity=np.cumprod(1+net)
        highs=np.maximum.accumulate(np.r_[1.,equity])[1:]
        max_dd=float(np.min(equity/highs-1))
        sigma=float(np.std(net,ddof=1)) if len(net)>1 else 0.
        records.append({
            "strategy":name,
            "forecast_model":forecast_col if name!="static" else "none",
            "n":len(weights),"cost_bps":cost_bps,
            "vol_target":target_annual_vol if name!="static" else float("nan"),
            "cumulative_return":float(equity[-1]-1),
            "annualized_return":float(equity[-1]**(252/len(net))-1),
            "annualized_vol":float(np.sqrt(252)*sigma),
            "sharpe_zero_cash":float(np.sqrt(252)*np.mean(net)/sigma) if sigma>0 else float("nan"),
            "max_drawdown":max_dd,
            "average_exposure":float(np.mean(weights)),
            "total_one_way_turnover":float(np.sum(turnover)),
        })
    return pd.DataFrame(records)
