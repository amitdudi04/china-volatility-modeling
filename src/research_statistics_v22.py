"""Dependence-aware uncertainty for *retrospective* forecast loss differences.

Circular moving-block bootstrap on paired same-day QLIKE loss differences.
This DOES NOT convert examined 2024 observations into an untouched holdout.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def bootstrap_qlike_differences(
    predictions:pd.DataFrame, baseline:str="rolling_21_variance",
    *, blocks:int=5, replications:int=2000, seed:int=2026
)->pd.DataFrame:
    if not isinstance(predictions,pd.DataFrame) or predictions.empty:
        raise ValueError("Missing per-origin predictions")
    if blocks<2 or replications<100 or len(predictions)<max(40,blocks*4):
        raise ValueError("Insufficient sample, block length or replications")
    if baseline not in predictions:
        raise ValueError("Unknown reference forecast")
    truth=predictions["realized_next_variance"].to_numpy(float)
    base=predictions[baseline].to_numpy(float)
    if not np.isfinite(truth).all() or not np.isfinite(base).all():
        raise ValueError("Invalid truth/reference")
    if (base<=0).any() or (truth<0).any():
        raise ValueError("Negative target or invalid forecast")
    def loss(p):
        return np.log(p)+truth/p
    reference=loss(base)
    rng=np.random.default_rng(seed)
    n=len(predictions)
    nblocks=int(np.ceil(n/blocks))
    starts=rng.integers(0,n,size=(replications,nblocks))
    offsets=np.arange(blocks,dtype=int)
    bootstrap_indices=((starts[:,:,None]+offsets[None,None,:])%n).reshape(
        replications,nblocks*blocks)[:,:n]
    rows=[]
    for name in predictions.columns:
        if not name.endswith("_variance") or name in ("realized_next_variance",baseline):
            continue
        pred=predictions[name].to_numpy(float)
        if not np.isfinite(pred).all() or (pred<=0).any():
            raise ValueError(f"Invalid forecast {name}")
        diff=loss(pred)-reference
        means=np.mean(diff[bootstrap_indices],axis=1)
        low,high=np.quantile(means,[.025,.975])
        rows.append({
            "model":name,"reference":baseline,"n":n,
            "block_length":blocks,"replications":replications,
            "mean_qlike_difference":float(np.mean(diff)),
            "ci95_low":float(low),"ci95_high":float(high),
            "ci_excludes_zero":bool(low>0 or high<0),
            "interpretation":"negative mean favours model; uncertainty conditional on selected retrospective methods",
        })
    return pd.DataFrame(rows).sort_values("mean_qlike_difference",kind="stable").reset_index(drop=True)
