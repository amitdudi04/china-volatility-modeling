"""Acquire frozen candidate research inputs; NEVER overwrite legacy CSVs.

Run: python scripts/fetch_research_data_v2.py --output-dir staging/research-v2-20261012
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yfinance as yf
from src.data_contract import (
    MARKETS, DATA_START, DATA_END_EXCLUSIVE,
    standardize_download, write_immutable_staging,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    if output.exists():
        parser.error("Destination exists: refusing to overwrite previous release")
    sources = {}
    for market, ticker in MARKETS.items():
        print(f"Downloading {market} ({ticker}), {DATA_START} to {DATA_END_EXCLUSIVE} exclusive", flush=True)
        raw = yf.download(
            ticker, start=DATA_START, end=DATA_END_EXCLUSIVE,
            auto_adjust=False, repair=False, threads=False, progress=False,
        )
        sources[market] = standardize_download(raw, ticker)
        print(f"  raw sessions: {len(sources[market])}", flush=True)
    manifest = write_immutable_staging(output, sources, {
        "provider": "Yahoo Finance via yfinance",
        "yfinance_version": str(yf.__version__),
        "requested_start": DATA_START,
        "requested_end_exclusive": DATA_END_EXCLUSIVE,
        "auto_adjust": False,
        "research_close_field": "Adj Close",
        "raw_close_field": "Close",
        "repair": False,
        "tickers": MARKETS,
    })
    print(f"PASS: validated staged input release: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
