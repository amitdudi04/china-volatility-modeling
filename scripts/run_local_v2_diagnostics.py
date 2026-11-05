"""Run a LOCAL diagnostic on legacy files, without publishing model results.

This intentionally does not claim research v2.1 is complete.  The legacy
ChiNext file is an ETF, whereas v2.1 specifies the actual ChiNext index; do
not pass it here as a market index. This diagnostic supports CSI_300 and
SSE_Composite only, on available 2024 observations.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.research_design import prepare_session_dataset
from src.oos_forecasting_v2 import walk_forward,score_variance_predictions

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root",required=True)
    parser.add_argument("--market",required=True,choices=("CSI_300","SSE_Composite"))
    parser.add_argument("--max-origins",type=int,default=8)
    parser.add_argument("--min-training",type=int,default=252)
    parser.add_argument("--n-estimators",type=int,default=30)
    args=parser.parse_args(argv)
    path=Path(args.legacy_root)/"data"/"raw"/f"{args.market}.csv"
    raw=pd.read_csv(path,parse_dates=["Date"]).set_index("Date")
    if raw.index.max()>pd.Timestamp("2024-12-31"):
        parser.error("This diagnostic is restricted to legacy 2024 data")
    frame=prepare_session_dataset(raw,horizon=1)
    forecasts=walk_forward(
        frame,"2024-01-01","2024-12-31",min_training=args.min_training,
        max_origins=args.max_origins,include_garch=True,
        n_estimators=args.n_estimators)
    scores=score_variance_predictions(forecasts)
    print("LEGACY-DATA DIAGNOSTIC ONLY: not v2.1 three-index OOS results")
    print("MARKET",args.market,"INPUT_DATE_RANGE",
          raw.index.min().date(),raw.index.max().date())
    print("ORIGINS",len(forecasts),"FIRST",forecasts.index.min().date(),
          "LAST",forecasts.index.max().date())
    print(scores.to_string(index=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
