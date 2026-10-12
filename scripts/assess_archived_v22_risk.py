"""Retrospective 2024 ex-ante normal VaR and long-only exposure sensitivities.

Only reads completed, checksummed archived v2.2 full forecast artifacts.
Not an untouched holdout, Student-t calibration, or investable backtest.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.run_archived_v22 import load_legacy_data
from src.risk_economic_v22 import (
    align_next_session_returns,normal_var_coverage,
    economic_exposure_sensitivity,
)

MARKETS=("CSI_300","SSE_Composite","ChiNext")

def assess(source_dir:Path,out_dir:Path)->dict:
    source_dir,out_dir=Path(source_dir),Path(out_dir)
    if out_dir.exists():
        raise FileExistsError("Refusing to overwrite risk/strategy report")
    risk,econ=[],[]
    sources={}
    for market in MARKETS:
        stem=f"{market}_2024_complete"
        path=source_dir/(stem+"_predictions.csv")
        manifest=json.loads((source_dir/(stem+"_manifest.json")).read_text())
        if manifest["status"]!="retrospective_2024_full":
            raise ValueError("Partial models are not eligible for risk analysis")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest["predictions_sha256"]:
            raise ValueError("Source forecast hash mismatch")
        prices,info=load_legacy_data(market)
        if info["sha256"]!=manifest["source"]["sha256"]:
            raise ValueError("Source price file changed since model fitting")
        pred=pd.read_csv(path,parse_dates=["origin_date","target_end_date"]).set_index("origin_date")
        if len(pred)!=manifest["total_eligible_2024_origins"]:
            raise ValueError("Incomplete annual forecast series")
        realized=align_next_session_returns(pred,prices)
        r=normal_var_coverage(pred,realized,alpha=.05)
        r.insert(0,"market",market)
        risk.append(r)
        for model in ("garch11_variance","random_forest_variance"):
            for bps in (0,5,10,20):
                e=economic_exposure_sensitivity(
                    pred,realized,forecast_col=model,
                    target_annual_vol=.20,cost_bps=bps)
                e.insert(0,"market",market)
                # Static is the same comparator for each model, repeated with
                # explicit model/cost labels for easier within-case audit.
                econ.append(e)
        sources[market]={"sha256":info["sha256"],"n_origins":len(pred)}
    out_dir.mkdir(parents=True,exist_ok=False)
    v=pd.concat(risk,ignore_index=True)
    s=pd.concat(econ,ignore_index=True)
    v.to_csv(out_dir/"normal_95pct_var_coverage.csv",index=False)
    s.to_csv(out_dir/"exposure_strategy_sensitivity.csv",index=False)
    summary={
        "scope":"exploratory archived 2024 prequential only",
        "not_untouched_holdout":True,
        "assumptions":[
            "Normal conditional VaR at 95%, zero expected daily return; not Student-t VaR.",
            "Kupiec unconditional coverage only; coverage nonrejection is not evidence of safe tail model.",
            "No leverage: exposure min(1, 20% target annual vol / predicted vol).",
            "Long-only and zero yield on idle cash; close-t reweight for next session t+1.",
            "Transaction cost per unit one-way change in exposure: 0/5/10/20 bps.",
            "No dividends, borrow costs, interest, taxes, market impact or ETF trading frictions.",
            "SSE/CSI are non-investable price indices and ChiNext is raw-price ETF proxy.",
            "Target and strategy settings are exploratory; not optimized on an untouched holdout.",
        ],
        "sources":sources,
    }
    (out_dir/"risk_economic_assumptions.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("Completed archived 2024 risk and economic sensitivity",flush=True)
    print(v[["market","model","observations","breaches","observed_rate","kupiec_pvalue"]].to_string(index=False),flush=True)
    print(s.loc[(s.strategy=="volatility_managed")&(s.cost_bps==10),
            ["market","forecast_model","annualized_return","max_drawdown","sharpe_zero_cash"]].to_string(index=False),flush=True)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    x=p.parse_args();assess(x.source_dir,x.out_dir)

if __name__=="__main__": main()
