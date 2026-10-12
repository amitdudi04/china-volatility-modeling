"""Review the metadata, parseability and role of every tracked repository file.

This does NOT claim every mathematical inference/figure was independently
validated. High-risk code, tabular release consistency and legacy document
claims require separate semantic review; see docs/FINAL_PANEL_REVIEW.md.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ROOT_RESOLVED=str(ROOT)
LEGACY_PATH_PREFIXES=("outputs/results/","outputs/reports/","outputs/plots/",
                      "outputs/figures/","data/processed/")
CURRENT_PREFIXES=("outputs/research_v22/","tests/","config/research_protocol_v22.json",
                  "src/oos_forecasting_v2.py","src/research_design.py",
                  "src/research_statistics_v22.py","src/risk_economic_v22.py",
                  "src/student_t_risk_v22.py","src/gui/v22_",
                  "scripts/run_archived_v22.py","scripts/summarize_archived_v22.py",
                  "scripts/assess_archived_v22_risk.py","scripts/build_v22_dashboard.py",
                  "docs/RESEARCH_V22_ARCHIVED_RESULTS.md")

def role(rel:str)->str:
    if rel.startswith(LEGACY_PATH_PREFIXES) or rel=="docs/results_analysis.docx":
        return "LEGACY_ARTIFACT_RETAINED_NOT_V22_EVIDENCE"
    if rel.startswith(CURRENT_PREFIXES) or rel=="README.md" or rel=="main.py":
        return "ACTIVE_RESEARCH_V22"
    if rel.startswith("src/") or rel.startswith("data/raw/"):
        return "LEGACY_PIPELINE_OR_ARCHIVED_SOURCE"
    if rel.startswith("config/research_protocol_v2") or rel.startswith("docs/DATA_RECOVERY_V2"):
        return "SUPERSEDED_2026_EXTENSION_PROPOSAL"
    return "SUPPORT_OR_LEGACY_RESEARCH"

def inventory(output:Path):
    output=Path(output)
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite audit inventory: {output}")
    tracked=subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).decode("utf-8").split("\0")
    tracked=[x for x in tracked if x]
    results=[]
    for rel in tracked:
        file=ROOT/rel
        entry={"path":rel,"role":role(rel),"extension":file.suffix.lower(),
               "bytes":file.stat().st_size if file.is_file() else 0,
               "sha256":hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else "",
               "structure":"UNEXAMINED","rows_or_nodes":"","alert":""}
        try:
            suffix=file.suffix.lower()
            if suffix==".py":
                code=file.read_text(encoding="utf-8")
                tree=ast.parse(code,filename=rel)
                entry["structure"]="AST_PASS"
                entry["rows_or_nodes"]=len(tree.body)
            elif suffix==".csv":
                df=pd.read_csv(file,low_memory=False)
                entry["structure"]="PARSE_PASS"
                entry["rows_or_nodes"]=len(df)
                if not len(df):entry["alert"]="EMPTY_CSV"
                if df.columns.duplicated().any():entry["alert"]="DUPLICATE_CSV_HEADERS"
                if df.isna().all(axis=0).any():entry["alert"]="ALL_MISSING_COLUMN"
            elif suffix==".json":
                obj=json.loads(file.read_text(encoding="utf-8"))
                entry["structure"]="JSON_PASS"
                entry["rows_or_nodes"]=len(obj) if hasattr(obj,"__len__") else 1
            elif suffix==".md":
                content=file.read_text(encoding="utf-8")
                entry["structure"]="UTF8_PASS"
                entry["rows_or_nodes"]=len(content.splitlines())
                missing=[]
                for target in re.findall(r"(?<!\\)\[[^\]]+\]\(([^)]+)\)",content):
                    target=target.split("#")[0].split("?")[0]
                    if not target or "://" in target or target.startswith(("mailto:","#")):
                        continue
                    if not (file.parent/target).exists():
                        missing.append(target)
                if missing:
                    entry["alert"]="BROKEN_RELATIVE_LINK:"+",".join(missing[:3])
            elif suffix in (".txt",".gitignore",".yml",".yaml"):
                content=file.read_text(encoding="utf-8")
                entry["structure"]="UTF8_PASS"
                entry["rows_or_nodes"]=len(content.splitlines())
            elif suffix==".png":
                with file.open("rb") as fh:
                    head=fh.read(24)
                if not (head[:8]==b"\x89PNG\r\n\x1a\n" and head[12:16]==b"IHDR"):
                    raise ValueError("Invalid PNG signature/IHDR")
                entry["structure"]="PNG_HEADER_PASS"
                entry["rows_or_nodes"]=f"{int.from_bytes(head[16:20],'big')}x{int.from_bytes(head[20:24],'big')}"
            elif suffix==".docx":
                with zipfile.ZipFile(file) as z:
                    problem=z.testzip()
                    if problem:raise ValueError(f"ZIP member CRC error {problem}")
                    content=z.read("word/document.xml").decode("utf-8")
                entry["structure"]="DOCX_PACKAGE_CRC_PASS"
                entry["rows_or_nodes"]=content.count("<w:p")
                if any(key in content for key in ("Adaptive Hybrid","universal","institutional")):
                    entry["alert"]="LEGACY_CLAIMS_CONFLICT_WITH_V22"
            else:
                entry["structure"]="BINARY_HASH_ONLY"
        except Exception as err:
            entry["structure"]="VALIDATION_FAILED"
            entry["alert"]=type(err).__name__+":"+str(err)[:150]
        results.append(entry)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8") as stream:
        wr=csv.DictWriter(stream,fieldnames=list(results[0]))
        wr.writeheader();wr.writerows(results)
    summary={}
    for row in results:
        summary[row["structure"]]=summary.get(row["structure"],0)+1
    warnings=[x for x in results if x["alert"]]
    print("TRACKED_FILE_COUNT",len(results),"STATUS",summary)
    print("ALERTS",len(warnings))
    for item in warnings[:35]:
        print(item["path"],item["alert"])
    print("INVENTORY",output)
    return results

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    opts=p.parse_args();inventory(opts.out)
