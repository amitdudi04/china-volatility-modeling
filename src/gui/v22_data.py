"""Read-only, fail-closed v2.2 research-output adapter for the dashboard.

Legacy outputs/results files must not be used for final model conclusions.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scripts.run_archived_v22 import load_legacy_data

MARKETS=("CSI_300","SSE_Composite","ChiNext")
PREDICTION_MODELS=(
    "garch11_variance","random_forest_variance",
    "rolling_21_variance","ridge_rf_equal_variance","logvar_ridge_variance",
)
def digest(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_research_market(market:str,root:Path|None=None)->dict:
    if market not in MARKETS:
        raise ValueError("Unsupported market")
    root=Path(root) if root is not None else Path(__file__).resolve().parents[2]
    research=root/"outputs"/"research_v22"
    base=market+"_2024_complete"
    pred_path=research/(base+"_predictions.csv")
    score_path=research/(base+"_loss_metrics.csv")
    manifest=json.loads((research/(base+"_manifest.json")).read_text(encoding="utf-8"))
    prices,info=load_legacy_data(market,root)
    if manifest["status"]!="retrospective_2024_full":
        raise ValueError("This dashboard only accepts complete v2.2 releases")
    if info["sha256"]!=manifest["source"]["sha256"]:
        raise ValueError("Raw historical prices differ from validated release")
    if digest(pred_path)!=manifest["predictions_sha256"]:
        raise ValueError("Forecast CSV checksum mismatch")
    if digest(score_path)!=manifest["scores_sha256"]:
        raise ValueError("Metrics checksum mismatch")
    pred=pd.read_csv(pred_path,parse_dates=["origin_date","target_end_date"]).set_index("origin_date")
    if len(pred)!=240 or pred.index.has_duplicates or not pred.index.is_monotonic_increasing:
        raise ValueError("Invalid forecast dates")
    if not pred.index.equals(prices.loc["2024-01-01":"2024-12-31"].index[:-1]):
        raise ValueError("Forecast origins differ from actual market sessions")
    if not (pd.to_datetime(pred["target_end_date"])>pred.index).all():
        raise ValueError("Forecasts must precede realized targets")
    scores=pd.read_csv(score_path)
    summary=pd.read_csv(research/"2024_model_loss_summary.csv")
    risk=pd.read_csv(research/"normal_95pct_var_coverage.csv")
    econ=pd.read_csv(research/"exposure_strategy_sensitivity.csv")
    ci=pd.read_csv(research/"2024_bootstrap_qlike_differences.csv")
    if set(scores["model"])!=set(PREDICTION_MODELS):
        raise ValueError("Unknown model types in release")
    if len(summary[summary.market==market])!=5 or len(risk[risk.market==market])!=5:
        raise ValueError("Incomplete summary or risk records")
    if len(econ[econ.market==market])!=16:
        raise ValueError("Incomplete strategy sensitivity")
    for col in PREDICTION_MODELS:
        if col not in pred or (pred[col]<=0).any() or not np.isfinite(pred[col]).all():
            raise ValueError(f"Nonpositive or invalid variance in {col}")
    if not pred.index.equals(pd.to_datetime(pred.index)):
        raise ValueError("Invalid origin index")
    return {"market":market,"prices":prices,"source":info,"predictions":pred,
            "scores":scores,"model_summary":summary[summary.market==market].copy(),
            "risk":risk[risk.market==market].copy(),
            "strategy":econ[econ.market==market].copy(),
            "bootstrap":ci[ci.market==market].copy(),
            "manifest":manifest,
            "scope":"Retrospective historical 2024 prequential, NOT untouched holdout; exploratory post-hoc model correction.",
            "instrument_note":"ChiNext uses 159915.SZ ETF proxy (raw close), NOT the ChiNext index." if market=="ChiNext" else "Price index; not directly tradeable."}
