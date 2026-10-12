"""Single-vendor China market recovery through AKShare / EastMoney.

Does not merge Yahoo Finance data with EastMoney data. The three instruments
must all pass source and chronology validation, otherwise no release is made.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.data_contract import MARKETS, standardize_download, write_immutable_staging

DATE_START = "20150101"
DATE_END = "20260930"
INDEX_SYMBOLS = {"CSI_300":"sh000300", "SSE_Composite":"sh000001"}


def _index_frame(raw: pd.DataFrame, market: str) -> pd.DataFrame:
    required = ["date", "open", "close", "high", "low", "volume"]
    if raw is None or raw.empty or not set(required).issubset(raw.columns):
        raise ValueError(f"{market}: missing EastMoney index OHLCV history")
    data = raw.loc[:, required].copy()
    data.index = pd.to_datetime(data.pop("date"), errors="raise")
    data = data.rename(columns={
        "open":"Open", "high":"High", "low":"Low",
        "close":"Close", "volume":"Volume",
    })
    # Index daily price series, NOT an investable total-return series.
    data["Adj Close"] = data["Close"]
    return standardize_download(data, MARKETS[market])


def _etf_frame(raw: pd.DataFrame, adjusted: pd.DataFrame) -> pd.DataFrame:
    required = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
    if raw is None or adjusted is None or raw.empty or adjusted.empty:
        raise ValueError("ChiNext: ETF raw or qfq-adjusted price series missing")
    if not set(required).issubset(raw.columns) or not {"日期","收盘"}.issubset(adjusted.columns):
        raise ValueError("ChiNext: unexpected provider columns")
    data = raw.loc[:,required].copy().set_index("日期")
    forward_adjusted_close = adjusted.set_index("日期")["收盘"]
    data.index = pd.to_datetime(data.index, errors="raise")
    forward_adjusted_close.index = pd.to_datetime(forward_adjusted_close.index, errors="raise")
    if (data.index.has_duplicates or forward_adjusted_close.index.has_duplicates
        or set(data.index) != set(forward_adjusted_close.index)):
        raise ValueError("ChiNext: raw and adjusted ETF date calendars differ")
    data = data.rename(columns={
        "开盘":"Open","最高":"High","最低":"Low",
        "收盘":"Close","成交量":"Volume",
    })
    data["Adj Close"] = forward_adjusted_close.reindex(data.index)
    return standardize_download(data, MARKETS["ChiNext"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists():
        parser.error("Output already exists; refusing overwrite")
    import akshare as ak
    sources = {}
    for market, ticker in INDEX_SYMBOLS.items():
        print(f"EastMoney index {market}: {ticker}", flush=True)
        raw = ak.stock_zh_index_daily_em(
            symbol=ticker, start_date=DATE_START, end_date=DATE_END)
        sources[market] = _index_frame(raw, market)
        print(f"  {len(sources[market])} source sessions", flush=True)
    print("EastMoney ETF 159915: requesting raw and qfq adjusted histories", flush=True)
    raw_etf = ak.fund_etf_hist_em(
        symbol="159915", period="daily",
        start_date=DATE_START, end_date=DATE_END, adjust="")
    adjusted_etf = ak.fund_etf_hist_em(
        symbol="159915", period="daily",
        start_date=DATE_START, end_date=DATE_END, adjust="qfq")
    sources["ChiNext"] = _etf_frame(raw_etf, adjusted_etf)
    print(f'  {len(sources["ChiNext"])} ETF sessions', flush=True)
    manifest = write_immutable_staging(Path(args.output_dir), sources, {
        "provider": "AKShare interfaces accessing EastMoney",
        "akshare_version": str(getattr(ak, "__version__", "unknown")),
        "index_api": "stock_zh_index_daily_em",
        "index_symbols": INDEX_SYMBOLS,
        "etf_api": "fund_etf_hist_em",
        "etf_code": "159915",
        "etf_original_adjustment": "",
        "etf_research_adjustment": "qfq, EastMoney forward adjusted",
        "index_research_close_type": "unadjusted price index",
        "requested_start": DATE_START, "requested_end": DATE_END,
    })
    print(f"PASS: staged source data and checksums: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
