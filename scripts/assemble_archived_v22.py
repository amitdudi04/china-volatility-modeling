"""Assemble nonoverlapping archived-2024 forecast chunks with exact coverage.

Fail closed if any source checksum, hyperparameter, date, or output overlaps.
Assembled results remain retrospective, NOT blind holdout evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_archived_v22 import load_legacy_data
from src.research_design import prepare_session_dataset
from src.oos_forecasting_v2 import score_variance_predictions

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def assemble(market:str, parts_dir:Path, target_dir:Path):
    parts_dir,target_dir=Path(parts_dir),Path(target_dir)
    raw,src=load_legacy_data(market)
    x=prepare_session_dataset(raw,1)
    expected=x.index[(x.index>=pd.Timestamp("2024-01-01"))
                     &(x.index<=pd.Timestamp("2024-12-31"))
                     &(pd.to_datetime(x["target_end_date"])<=pd.Timestamp("2024-12-31"))]
    fragments=[]; records=[]; n_estimators=None; min_training=None
    names=sorted(parts_dir.glob(f"{market}_2024_part_*_manifest.json"))
    if not names:
        raise ValueError("No chunk manifests found")
    for mpath in names:
        m=json.loads(mpath.read_text())
        if m["status"]!="diagnostic_partial_only" or m["source"]["sha256"]!=src["sha256"]:
            raise ValueError(f"Invalid chunk source/identity: {mpath}")
        if m["total_eligible_2024_origins"]!=len(expected):
            raise ValueError("Wrong total number of eligible origins")
        if n_estimators is None:
            n_estimators=m["n_estimators"];min_training=m["min_training"]
        elif n_estimators!=m["n_estimators"] or min_training!=m["min_training"]:
            raise ValueError("Chunks used different model parameters")
        f=mpath.with_name(mpath.name.replace("_manifest.json","_predictions.csv"))
        s=mpath.with_name(mpath.name.replace("_manifest.json","_loss_metrics.csv"))
        if sha(f)!=m["predictions_sha256"] or sha(s)!=m["scores_sha256"]:
            raise ValueError("Chunk forecast or metric SHA256 mismatch")
        frame=pd.read_csv(f,parse_dates=["origin_date","target_end_date"]).set_index("origin_date")
        if len(frame)!=m["forecast_origins"]:
            raise ValueError("Chunk row count mismatch")
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("Chunk is not ordered or has duplicates")
        if not(frame["target_end_date"]>frame.index).all():
            raise ValueError("One or more labels are not future observations")
        fragments.append(frame)
        records.append({"manifest":mpath.name,"manifest_sha256":sha(mpath)})
    merged=pd.concat(fragments).sort_index()
    if merged.index.has_duplicates:
        raise ValueError("Overlapping origin dates between forecast chunks")
    if not merged.index.equals(expected):
        missing=expected.difference(merged.index)
        surplus=merged.index.difference(expected)
        raise ValueError(f"Missing {len(missing)} and surplus {len(surplus)} origin dates")
    score=score_variance_predictions(merged)
    target_dir.mkdir(parents=True,exist_ok=True)
    stem=f"{market}_2024_complete"
    forecast_path=target_dir/(stem+"_predictions.csv")
    score_path=target_dir/(stem+"_loss_metrics.csv")
    manifest_path=target_dir/(stem+"_manifest.json")
    for path in (forecast_path,score_path,manifest_path):
        if path.exists():
            raise FileExistsError("Refusing to replace completed forecast artifact")
    merged.to_csv(forecast_path,float_format="%.15g")
    score.to_csv(score_path,index=False,float_format="%.15g")
    final={
        "protocol_version":"research-protocol-v2.2-legacy-archive",
        "status":"retrospective_2024_full",
        "not_an_untouched_holdout":True,
        "source":src,
        "total_eligible_2024_origins":len(expected),
        "forecast_origins":len(merged),
        "first_origin":str(expected.min().date()),"last_origin":str(expected.max().date()),
        "min_training":min_training,"n_estimators":n_estimators,
        "method":"identical per-origin daily expanding models, assembled from validated independent chunks",
        "merged_from_parts":records,
        "predictions_sha256":sha(forecast_path),
        "scores_sha256":sha(score_path),
        "limitations":["Not an untouched holdout","ChiNext is 159915.SZ raw-price ETF proxy"],
    }
    manifest_path.write_text(json.dumps(final,indent=2)+"\n")
    print("VALIDATED_COMPLETE",market,len(merged),"origins across",len(records),"chunks",flush=True)
    print(score.to_string(index=False),flush=True)
    return final

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--market",choices=("CSI_300","SSE_Composite","ChiNext"),required=True)
    p.add_argument("--parts-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args();assemble(args.market,args.parts_dir,args.out_dir)
if __name__=="__main__":
    main()
