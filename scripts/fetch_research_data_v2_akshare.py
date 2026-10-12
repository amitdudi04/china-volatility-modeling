"""Recover THREE price-index histories from a single EastMoney interface.

Index set: CSI 300 (sh000300), SSE Composite (sh000001), ChiNext (sz399006).
This is research v2.1, which explicitly replaces the legacy ETF *proxy* with
the real ChiNext index BEFORE any model estimation or new result publication.
No source is blended with Yahoo Finance, and legacy files are not overwritten.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.data_contract import MARKETS, standardize_download, write_immutable_staging

DATE_START = "20150101"
DATE_END = "20260930"
INDEX_SYMBOLS = {
    "CSI_300":"sh000300",
    "SSE_Composite":"sh000001",
    "ChiNext":"sz399006",
}


def _index_frame(raw: pd.DataFrame, market: str) -> pd.DataFrame:
    mandatory = ["date", "open", "close", "high", "low", "volume"]
    if raw is None or raw.empty or not set(mandatory).issubset(raw.columns):
        raise ValueError(f"{market}: missing EastMoney index OHLCV history")
    frame = raw.loc[:,mandatory].copy()
    frame.index = pd.to_datetime(frame.pop("date"), errors="raise")
    frame = frame.rename(columns={
        "open":"Open", "high":"High", "low":"Low",
        "close":"Close", "volume":"Volume",
    })
    # A price index has no separate ETF adjustment; not a total-return index.
    frame["Adj Close"] = frame["Close"]
    return standardize_download(frame, MARKETS[market])


def _fetch_index_with_retries(ak, ticker: str, attempts: int = 3):
    """Retry transient provider/transport errors; never invent missing records."""
    for number in range(1, attempts+1):
        try:
            response = ak.stock_zh_index_daily_em(
                symbol=ticker, start_date=DATE_START, end_date=DATE_END)
            if response is None or response.empty:
                raise ValueError(f"{ticker}: provider returned empty history")
            return response
        except Exception as exc:
            if number == attempts:
                raise RuntimeError(
                    f"{ticker}: all {attempts} EastMoney history requests failed") from exc
            print(f"{ticker}: attempt {number}/{attempts} failed: {type(exc).__name__}. Retrying.", flush=True)
            time.sleep(2*number)
    raise AssertionError("Unreachable")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists():
        parser.error("Staging output exists; refusing overwrite")
    import akshare as ak
    frames = {}
    for market,ticker in INDEX_SYMBOLS.items():
        print(f"Fetching EastMoney price index {market} ({ticker})", flush=True)
        raw = _fetch_index_with_retries(ak, ticker)
        frames[market] = _index_frame(raw, market)
        print(f"  received {len(frames[market])} index sessions", flush=True)
    manifest = write_immutable_staging(Path(args.output_dir), frames, {
        "source": "AKShare index interface to EastMoney",
        "akshare_version": str(getattr(ak, "__version__", "unknown")),
        "api": "stock_zh_index_daily_em",
        "index_symbols": INDEX_SYMBOLS,
        "index_price_field": "close: unadjusted index price, not total return",
        "research_protocol": "v2.1 amended 2026-10-12 before model fitting",
        "requested_start": DATE_START, "requested_end": DATE_END,
    })
    print(f"PASS: three-index staged release: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
