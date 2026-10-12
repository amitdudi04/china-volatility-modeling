"""China volatility research CLI (current research v2.2 by default).

Legacy pipeline is preserved for forensic comparisons, but must never
accidentally overwrite the audited legacy research outputs.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path

def verify_v22():
    from src.gui.v22_data import MARKETS,load_research_market
    for market in MARKETS:
        d=load_research_market(market)
        print(f"PASS {market}: {len(d['predictions'])} verified predictions; {d['source']['rows']} raw observations")
    print("Research v2.2: saved retrospective datasets verify; not an untouched holdout.")

def rerun_v22(root:Path):
    from scripts.run_archived_v22 import run_archived
    from scripts.summarize_archived_v22 import summarize
    from scripts.assess_archived_v22_risk import assess
    source=root/"full"
    if root.exists():
        raise FileExistsError(f"Refuse to overwrite existing v2.2 rerun folder: {root}")
    for market in ("CSI_300","SSE_Composite","ChiNext"):
        run_archived(market,output_dir=source)
    summarize(source,root/"summary")
    assess(source,root/"risk")
    print(f"Reanalysis finished: {root}; compare the new manifest hashes before publishing.")

def main(argv=None):
    parser=argparse.ArgumentParser(description="Chinese equity variance research; verified v2.2 archived dataset")
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument("--run-gui",action="store_true",help="Open the verified offline browser research dashboard (no Qt dependency)")
    actions.add_argument("--run-qt-gui",action="store_true",help="Try experimental v2.2 PyQt dashboard, dependent on local Qt graphics support")
    actions.add_argument("--verify-v22",action="store_true",help="Validate 3 market files and checksum-verified research predictions")
    actions.add_argument("--run-pipeline",action="store_true",help="Rerun the corrected v2.2 retrospective experiment (new outputs only)")
    actions.add_argument("--run-legacy-gui",action="store_true",help="Open historical, superseded GUI for comparison only")
    actions.add_argument("--run-legacy-pipeline",action="store_true",help="Run superseded pipeline; may rewrite legacy outputs")
    parser.add_argument("--output-dir",type=Path,default=None,help="New, unused directory for --run-pipeline")
    parser.add_argument("--allow-legacy-overwrite",action="store_true",
                        help="Explicitly authorize superseded pipeline to replace legacy outputs")
    args=parser.parse_args(argv)

    if args.run_gui:
        from scripts.build_v22_dashboard import export_dashboard
        import webbrowser
        file=Path("outputs/research_v22/dashboard.html").resolve()
        if not file.is_file():
            export_dashboard(file)
        else:
            # Opening an existing page still checks its underlying research inputs.
            verify_v22()
        print(f"Opening offline verified dashboard: {file}")
        webbrowser.open(file.as_uri())
        return 0
    if args.run_qt_gui:
        from src.gui.v22_dashboard import main as launch
        return launch()
    if args.verify_v22:
        verify_v22()
        return 0
    if args.run_pipeline:
        output=args.output_dir
        if output is None:
            output=Path("staging")/("v22_rerun_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
        rerun_v22(output)
        return 0
    if args.run_legacy_gui:
        print("WARNING: legacy dashboard uses older, superseded outputs with unsupported claims.")
        from src.gui import main as launch
        return launch()
    if args.run_legacy_pipeline:
        if not args.allow_legacy_overwrite:
            parser.error("--run-legacy-pipeline requires --allow-legacy-overwrite; original outputs could be replaced.")
        print("WARNING: running superseded legacy research; its outputs are NOT verified v2.2 evidence.")
        from src.pipeline import Pipeline
        Pipeline().run_all()
        return 0
    parser.print_help()
    return 0

if __name__=="__main__":
    raise SystemExit(main())
