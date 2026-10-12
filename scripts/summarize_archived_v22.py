"""Validate three fully completed 2024 archive experiments and summarize them.

No model selection, retraining, new information, or 'blind holdout' claims.
Do not run on partial diagnostic chunks; require 3 complete manifests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.oos_forecasting_v2 import score_variance_predictions
from src.research_statistics_v22 import bootstrap_qlike_differences

MARKETS=("CSI_300","SSE_Composite","ChiNext")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def summarize(source_dir:Path,destination_dir:Path,blocks=5,replications=2000):
    source_dir,destination_dir=Path(source_dir),Path(destination_dir)
    if destination_dir.exists():
        raise FileExistsError("Refuse to overwrite previously generated report")
    all_scores,all_ci,all_manifests=[],[],[]
    dates_by_market={}
    for market in MARKETS:
        stem=f"{market}_2024_complete"
        f=source_dir/(stem+"_predictions.csv")
        m=source_dir/(stem+"_manifest.json")
        manifest=json.loads(m.read_text(encoding="utf-8"))
        if manifest["status"]!="retrospective_2024_full" or manifest["source"]["market"]!=market:
            raise ValueError(f"Invalid partial or mislabelled experiment {market}")
        if sha(f)!=manifest["predictions_sha256"]:
            raise ValueError(f"Predictions corrupted for {market}")
        scores_path=source_dir/(stem+"_loss_metrics.csv")
        if sha(scores_path)!=manifest["scores_sha256"]:
            raise ValueError(f"Metric file hash mismatch: {market}")
        pred=pd.read_csv(f,parse_dates=["origin_date","target_end_date"]).set_index("origin_date")
        if len(pred)!=manifest["total_eligible_2024_origins"]:
            raise ValueError(f"Incomplete run for {market}")
        if pred.index.has_duplicates or not pred.index.is_monotonic_increasing:
            raise ValueError(f"Duplicate or unsorted origins for {market}")
        if not(pred["target_end_date"]>pred.index).all():
            raise ValueError(f"Non-future target or corrupt time index for {market}")
        recomputed=score_variance_predictions(pred)
        reported=pd.read_csv(scores_path)
        merged=recomputed.merge(reported,on="model",suffixes=("_new","_old"),validate="one_to_one")
        for field in ("rmse","mae","qlike"):
            if (merged[field+"_new"]-merged[field+"_old"]).abs().max()>1e-10:
                raise ValueError(f"Inconsistent saved metric: {market}, {field}")
        recomputed.insert(0,"market",market)
        ci=bootstrap_qlike_differences(pred,blocks=blocks,
                                       replications=replications,seed=2026)
        ci.insert(0,"market",market)
        all_scores.append(recomputed)
        all_ci.append(ci)
        all_manifests.append(manifest)
        dates_by_market[market]=pred.index

    common=dates_by_market[MARKETS[0]]
    for market in MARKETS[1:]:
        common=common.intersection(dates_by_market[market])
    if len(common)<200:
        raise ValueError("Insufficient common 2024 trading origins")
    destination_dir.mkdir(parents=True,exist_ok=False)
    a=pd.concat(all_scores,ignore_index=True)
    b=pd.concat(all_ci,ignore_index=True)
    a.to_csv(destination_dir/"2024_model_loss_summary.csv",index=False)
    b.to_csv(destination_dir/"2024_bootstrap_qlike_differences.csv",index=False)
    audit={"protocol":"research-protocol-v2.2-legacy-archive",
           "status":"retrospective_OOS_not_blind",
           "markets":list(MARKETS),
           "all_three_common_evaluation_origins":len(common),
           "common_evaluation_start":str(common.min().date()),
           "common_evaluation_end":str(common.max().date()),
           "input_manifest_sha256":{
             m["source"]["market"]:sha(source_dir/(m["source"]["market"]+"_2024_complete_manifest.json"))
             for m in all_manifests},
           "limitations":[
               "Historical 2024 outcomes and research design have been inspected: no claim of untouched holdout.",
               "CSI 300 source history starts in 2021 and differs from other market histories.",
               "ChiNext is 159915.SZ ETF proxy, not a price index; raw close omits dividend adjustments.",
               "Ridge historical-range guardrail introduced after retrospective extreme-forecast observation.",
               "One-session squared returns are noisy estimators of latent conditional variance.",
               "QLIKE block bootstrap ignores model-selection uncertainty and does not establish strategy utility.",
           ]}
    (destination_dir/"evaluation_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print("Verified full retrospective runs for 3 markets; common origins",len(common),flush=True)
    print(a.to_string(index=False),flush=True)
    print(b[["market","model","mean_qlike_difference","ci95_low","ci95_high","ci_excludes_zero"]].to_string(index=False),flush=True)
    return audit

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--replications",type=int,default=2000)
    x=parser.parse_args()
    summarize(x.source_dir,x.out_dir,replications=x.replications)
if __name__=="__main__":
    main()
