"""Refit one-step Student-t GARCH density at each archived forecast origin.

Uses same model as src/oos_forecasting_v2.py and verifies its saved
variance forecast before using fitted mu/nu. Does not modify frozen outputs.
Runs can be divided into --limit chunks and assembled after checksum checks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from arch import arch_model

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_archived_v22 import load_legacy_data
from src.research_design import prepare_session_dataset
from src.gui.v22_data import load_research_market
from src.risk_economic_v22 import align_next_session_returns
from src.student_t_risk_v22 import standardized_t_var_es,christoffersen_tests

def compute(market,offset,limit,output_dir):
    if offset<0 or limit<1:
        raise ValueError("Require nonnegative offset and positive chunk length")
    d=load_research_market(market)
    pred=d["predictions"]
    labels=align_next_session_returns(pred,d["prices"])
    full=prepare_session_dataset(d["prices"],1)
    selected=pred.iloc[offset:offset+limit]
    if selected.empty:
        raise ValueError("No selected origins")
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    stem=f"{market}_tail_part_{offset:04d}_{len(selected):03d}"
    path=output_dir/(stem+".csv")
    man=output_dir/(stem+".json")
    if path.exists() or man.exists():
        raise FileExistsError("Risk report already exists; no overwrites")
    rows=[]
    for num,(origin,p) in enumerate(selected.iterrows(),1):
        returns=full.loc[:origin,"return_at_origin"].dropna().to_numpy(float)
        fitted=arch_model(returns*100,mean="Constant",vol="GARCH",p=1,o=0,q=1,
                          dist="t",rescale=False).fit(disp="off",show_warning=False)
        if fitted.convergence_flag!=0:
            raise RuntimeError(f"{market} {origin}: GARCH failed to converge")
        variance=float(fitted.forecast(horizon=1,reindex=False).variance.iloc[-1,0])
        ann=variance*252/10000.
        if not np.isclose(ann,p["garch11_variance"],rtol=1e-9,atol=1e-11):
            raise ValueError(f"Saved GARCH variance does not match refit at {origin}")
        nu=float(fitted.params["nu"])
        mu=float(fitted.params["mu"])/100.
        sigma=float(np.sqrt(ann/252))
        var,es=standardized_t_var_es(mu,sigma,nu)
        realized=float(labels.loc[origin])
        rows.append({"origin_date":origin,"target_end_date":p["target_end_date"],
                     "annualized_conditional_variance":ann,
                     "estimated_t_df":nu,"estimated_daily_log_mean":mu,
                     "estimated_daily_sigma":sigma,
                     "student_t_var_95":var,"student_t_es_95":es,
                     "next_log_return":realized,
                     "breach":int(realized<var)})
        if num==1 or num%20==0 or num==len(selected):
            print(f"{market}: Student-t VaR {num}/{len(selected)}; origin {origin.date()}",flush=True)
    out=pd.DataFrame(rows).set_index("origin_date")
    out.to_csv(path,float_format="%.15g")
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={"market":market,"forecast_origins":len(out),
              "offset":offset,"first":str(out.index.min().date()),
              "last":str(out.index.max().date()),
              "source_prediction_sha256":d["manifest"]["predictions_sha256"],
              "output_sha256":digest,
              "status":"historical_retrospective_not_untouched_holdout",
              "model":"refitted per origin GARCH(1,1), Student-t conditional dist",
              "scale":"raw daily log-return, standardized Student-t nu; annualized variance input",
              "es_status":"Parametric forecast with descriptive exceedance diagnostics; NOT formally backtested ES"}
    man.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("SAVED",path,flush=True)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--market",required=True,choices=("CSI_300","SSE_Composite","ChiNext"))
    p.add_argument("--offset",type=int,default=0)
    p.add_argument("--limit",type=int,default=60)
    p.add_argument("--out-dir",type=Path,default=Path("staging/v22_tails"))
    a=p.parse_args()
    compute(a.market,a.offset,a.limit,a.out_dir)
if __name__=="__main__":
    main()
