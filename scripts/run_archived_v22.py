"""Retrospective archived-data walk-forward experiment, v2.2.

Runs on the ORIGINAL tracked market CSVs, not on newly downloaded data.
ChiNext is the 159915.SZ ETF proxy, NOT the price index. Outcomes are
prequential OOS relative to each origin, but the previously examined 2024
period is NOT a blind/untouched academic holdout.

Usage:
  python scripts/run_archived_v22.py --market CSI_300 --out-dir staging/v22
  python scripts/run_archived_v22.py --market SSE_Composite --out-dir staging/v22
  python scripts/run_archived_v22.py --market ChiNext --out-dir staging/v22
The result files are versioned, non-destructive, and refuse overwrites.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.research_design import prepare_session_dataset
from src.oos_forecasting_v2 import forecast_one_origin,score_variance_predictions

PROTOCOL_FILE=ROOT/"config"/"research_protocol_v22.json"


def load_legacy_data(market: str, repo_root: Path = ROOT) -> tuple[pd.DataFrame, dict]:
    protocol=json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    if protocol["schema_version"]!="research-protocol-v2.2-legacy-archive":
        raise ValueError("Wrong active research protocol")
    if market not in protocol["markets"]:
        raise ValueError("Unknown archived market")
    spec=protocol["markets"][market]
    path=Path(repo_root)/spec["path"]
    if not path.is_file():
        raise FileNotFoundError(f"Archived data missing: {path}")
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    data=pd.read_csv(path,parse_dates=["Date"])
    mandatory={"Date","Close","Open","High","Low","Volume"}
    if not mandatory.issubset(data.columns):
        raise ValueError(f"{market}: missing data columns")
    if len(data)!=spec["rows"]:
        raise ValueError(f"{market}: expected {spec['rows']} rows, found {len(data)}")
    if data["Date"].duplicated().any() or not data["Date"].is_monotonic_increasing:
        raise ValueError(f"{market}: invalid date ordering")
    if data["Date"].min().strftime("%Y-%m-%d")!=spec["first"]:
        raise ValueError(f"{market}: first observed date differs from protocol")
    if data["Date"].max().strftime("%Y-%m-%d")!=spec["last"]:
        raise ValueError(f"{market}: last observed date differs from protocol")
    if data[["Open","High","Low","Close","Volume"]].isna().any().any():
        raise ValueError(f"{market}: missing raw data")
    if (data[["Open","High","Low","Close"]]<=0).any().any():
        raise ValueError(f"{market}: nonpositive prices")
    if (data["Volume"]<0).any():
        raise ValueError(f"{market}: negative volume")
    if ((data["High"]<data[["Open","Close","Low"]].max(axis=1))
        | (data["Low"]>data[["Open","Close","High"]].min(axis=1))).any():
        raise ValueError(f"{market}: inconsistent raw OHLC")
    if (data["Date"].dt.dayofweek>=5).any():
        raise ValueError(f"{market}: weekend observations")
    data=data.set_index("Date")
    info={
        "market":market,
        "source_relative_path":spec["path"],
        "instrument":spec["label"],
        "symbol":spec["ticker"],
        "rows":len(data),
        "first":spec["first"],"last":spec["last"],
        "sha256":sha,
        "git_blob_sha_at_legacy_snapshot":spec["git_blob_sha"],
    }
    return data,info


def run_archived(market: str, *, limit: int=0, offset: int=0,
                 min_training: int=252, n_estimators: int=40,
                 output_dir: Path=Path("staging/research_v22")):
    if limit<0 or offset<0 or n_estimators<1 or min_training<1:
        raise ValueError("Invalid experiment argument")
    data,source=load_legacy_data(market)
    frame=prepare_session_dataset(data,horizon=1)
    period=frame.loc[
        (frame.index>=pd.Timestamp("2024-01-01"))
        & (frame.index<=pd.Timestamp("2024-12-31"))
        & (pd.to_datetime(frame["target_end_date"])<=pd.Timestamp("2024-12-31"))]
    if period.empty:
        raise ValueError("No archived retrospective evaluation rows")
    total=len(period)
    chosen=period.index[offset:offset+limit if limit else None]
    if len(chosen)==0:
        raise ValueError("Selected empty forecast block")
    # Run the entire period without slicing for an eligible final release.
    complete=(offset==0 and len(chosen)==total)
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    stem=f"{market}_2024_{'complete' if complete else f'part_{offset:04d}_{len(chosen):03d}'}"
    pred_path=output_dir/(stem+"_predictions.csv")
    score_path=output_dir/(stem+"_loss_metrics.csv")
    manifest_path=output_dir/(stem+"_manifest.json")
    for path in (pred_path,score_path,manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite results: {path}")
    forecasts=[]
    for n,origin in enumerate(chosen,1):
        record=forecast_one_origin(
            frame,origin,min_training=min_training,
            include_garch=True,n_estimators=n_estimators)
        forecasts.append(record)
        if n==1 or n%20==0 or n==len(chosen):
            print(f"{market}: forecast {n}/{len(chosen)} origin={origin.date()}",flush=True)
    pred=pd.DataFrame(forecasts).set_index("origin_date")
    if not pred.index.equals(chosen):
        raise AssertionError("Incorrect forecast origin alignment")
    if not(pred["target_end_date"]>pred.index).all():
        raise AssertionError("Invalid forward-label chronology")
    metrics=score_variance_predictions(pred)
    pred.to_csv(pred_path,float_format="%.15g")
    metrics.to_csv(score_path,index=False,float_format="%.15g")
    manifest={
        "protocol_version":"research-protocol-v2.2-legacy-archive",
        "status":"retrospective_2024_full" if complete else "diagnostic_partial_only",
        "not_an_untouched_holdout":True,
        "source":source,
        "total_eligible_2024_origins":total,
        "forecast_origins":len(chosen),
        "first_origin":str(chosen.min().date()),
        "last_origin":str(chosen.max().date()),
        "min_training":min_training,
        "n_estimators":n_estimators,
        "model_fitting":"expanding, daily refits; no targets with end date beyond forecast origin",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "predictions_sha256":hashlib.sha256(pred_path.read_bytes()).hexdigest(),
        "scores_sha256":hashlib.sha256(score_path.read_bytes()).hexdigest(),
        "limitations":[
            "Retrospective prequential evaluation, NOT previously untouched holdout",
            "CSI history starts 2021; SSE and ChiNext ETF start 2015",
            "ChiNext market uses raw-price ETF proxy 159915.SZ, not the index",
            "One-day squared log return is a noisy latent-variance proxy",
            "Model/strategy superiority has not been established by significance tests",
        ],
    }
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(metrics.to_string(index=False),flush=True)
    print("SAVED",pred_path,score_path,manifest_path,flush=True)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market",choices=("CSI_300","SSE_Composite","ChiNext"),required=True)
    parser.add_argument("--limit",type=int,default=0,help="0 = all 2024 origins")
    parser.add_argument("--offset",type=int,default=0)
    parser.add_argument("--min-training",type=int,default=252)
    parser.add_argument("--n-estimators",type=int,default=40)
    parser.add_argument("--out-dir",type=Path,default=Path("staging/research_v22"))
    opts=parser.parse_args(argv)
    run_archived(opts.market,limit=opts.limit,offset=opts.offset,
                 min_training=opts.min_training,n_estimators=opts.n_estimators,
                 output_dir=opts.out_dir)

if __name__=="__main__":
    main()
