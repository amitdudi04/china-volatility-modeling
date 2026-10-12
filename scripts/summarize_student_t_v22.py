"""Combine per-origin Student-t GARCH tail risk forecasts with strict provenance.

Cannot publish incomplete or overlapping chunks. Calculated ES values are
model forecasts; descriptive exceedance statistics are NOT ES backtests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.gui.v22_data import MARKETS,load_research_market
from src.risk_economic_v22 import align_next_session_returns
from src.student_t_risk_v22 import christoffersen_tests
from src.utils.archived_digest import archived_file_sha256

def sha(p:Path):
    return archived_file_sha256(p)

def aggregate(parts:Path,dest:Path):
    parts,dest=Path(parts),Path(dest)
    if dest.exists():
        raise FileExistsError(f"Refuse overwrite of tail study: {dest}")
    tables=[];coverage=[];es_description=[];sources={}
    for market in MARKETS:
        release=load_research_market(market)
        pred=release["predictions"]
        expected_return=align_next_session_returns(pred,release["prices"])
        fragments=[];provenance=[]
        for off in (0,120):
            stem=f"{market}_tail_part_{off:04d}_120"
            m=parts/(stem+".json");f=parts/(stem+".csv")
            source=json.loads(m.read_text("utf-8"))
            if source["market"]!=market or source["offset"]!=off or source["forecast_origins"]!=120:
                raise ValueError(f"Incomplete/mislabelled Student-t risk part: {stem}")
            if source["source_prediction_sha256"]!=release["manifest"]["predictions_sha256"]:
                raise ValueError("Underlying GARCH forecasts have changed since risk refit")
            if sha(f)!=source["output_sha256"]:
                raise ValueError(f"Student-t risk CSV checksum mismatch: {stem}")
            frag=pd.read_csv(f,parse_dates=["origin_date","target_end_date"]).set_index("origin_date")
            fragments.append(frag);provenance.append({"manifest":m.name,"sha256":sha(m)})
        combo=pd.concat(fragments).sort_index()
        if combo.index.has_duplicates or not combo.index.equals(pred.index):
            raise ValueError(f"Missed/duplicated risk dates: {market}")
        if not combo["target_end_date"].equals(pred["target_end_date"]):
            raise ValueError(f"Risk target labels mismatch {market}")
        if not np.allclose(combo["annualized_conditional_variance"],pred["garch11_variance"],
                           rtol=1e-9,atol=1e-11):
            raise ValueError(f"Re-fitted GARCH forecasts inconsistent: {market}")
        if not np.allclose(combo["next_log_return"],expected_return,rtol=1e-11,atol=1e-13):
            raise ValueError(f"Incorrect return alignment: {market}")
        if not (combo["student_t_es_95"]<combo["student_t_var_95"]).all():
            raise ValueError("ES must lie in loss tail beyond VaR")
        actual_breaches=combo["next_log_return"]<combo["student_t_var_95"]
        if not np.array_equal(actual_breaches.to_numpy(int),combo["breach"].to_numpy(int)):
            raise ValueError("Published VaR hits inconsistent")
        stats=christoffersen_tests(actual_breaches,alpha=.05)
        coverage.append({"market":market,"distribution":"standardized fitted Student-t GARCH",
                         **stats,"nu_min":float(combo["estimated_t_df"].min()),
                         "nu_median":float(combo["estimated_t_df"].median()),
                         "nu_max":float(combo["estimated_t_df"].max())})
        sub=combo.loc[actual_breaches]
        es_description.append({
            "market":market,"n_exceedances":len(sub),
            "mean_realized_exceedance_log_return":float(sub["next_log_return"].mean()) if len(sub) else float("nan"),
            "mean_model_ES_on_exceedance_sessions":float(sub["student_t_es_95"].mean()) if len(sub) else float("nan"),
            "formal_ES_backtest":"NOT PERFORMED",
        })
        flat=combo.reset_index()
        flat.insert(0,"market",market)
        tables.append(flat)
        sources[market]={"source_prediction_sha256":release["manifest"]["predictions_sha256"],
                         "tail_part_manifests":provenance}
    dest.mkdir(parents=True,exist_ok=False)
    all_rows=pd.concat(tables,ignore_index=True)
    all_rows.to_csv(dest/"student_t_garch_tail_forecasts.csv",index=False,float_format="%.15g")
    pd.DataFrame(coverage).to_csv(dest/"student_t_var_backtests.csv",index=False)
    pd.DataFrame(es_description).to_csv(dest/"student_t_es_descriptive.csv",index=False)
    manifest={
        "scope":"2024 retrospectively evaluated, NOT blind holdout",
        "Student_t_distribution":"per-origin fitted conditional mean, nu, standardized variance",
        "var_alpha":.05,"row_count":len(all_rows),
        "markets":list(MARKETS),"source_part_checksums":sources,
        "results_sha256":{
            x:sha(dest/x) for x in ("student_t_garch_tail_forecasts.csv",
                                  "student_t_var_backtests.csv",
                                  "student_t_es_descriptive.csv")
        },
        "limitations":[
            "Prior retrospective 2024 model tuning; not an untouched holdout.",
            "Christoffersen VaR conditional-coverage includes LRuc and LRind; confidence not guaranteed by high p.",
            "Expected Shortfall is parametric forecast only, not subject to formal ES backtest here.",
            "No funding/transaction cost calibration inferred from the fitted conditional distribution.",
        ],
    }
    (dest/"student_t_tail_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n","utf-8")
    print("PASS: validated",len(all_rows),"daily Student-t GARCH risk forecasts")
    print(pd.DataFrame(coverage)[["market","observations","breaches","observed_rate",
           "kupiec_pvalue","christoffersen_ind_pvalue","christoffersen_cc_pvalue"]].to_string(index=False))
    return manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    args=parser.parse_args();aggregate(args.parts_dir,args.out_dir)
if __name__=="__main__":
    main()
