"""Strict external-market data validation and research release assembly.

This module does not overwrite legacy data/raw or legacy results.
Provider accuracy and ETF corporate actions require independent source review.
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from src.research_design import prepare_session_dataset, chronological_partition

MARKETS = {
    "CSI_300": "000300.SS",
    "SSE_Composite": "000001.SS",
    "ChiNext": "399006.SZ",  # ChiNext price index; v2 protocol amendment
}
DATA_START = "2015-01-01"
DATA_END_EXCLUSIVE = "2026-10-01"
TRAIN_END = "2022-12-31"
VALIDATION_START, VALIDATION_END = "2023-01-01", "2024-12-31"
POST_LEGACY_START, POST_LEGACY_END = "2025-01-01", "2026-09-30"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standardize_download(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Do not silently replace missing adjusted closes with raw closes."""
    if frame is None or frame.empty:
        raise ValueError(f"{ticker}: empty provider response")
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        elif ticker in data.columns.get_level_values(0):
            data = data.xs(ticker, axis=1, level=0)
        else:
            raise ValueError(f"{ticker}: ticker not present in MultiIndex columns")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: missing DatetimeIndex")
    if data.index.tz is not None:
        data.index = data.index.tz_convert("Asia/Shanghai").tz_localize(None)
    data.index = data.index.normalize()
    mandatory = ["Close", "Adj Close", "Open", "High", "Low", "Volume"]
    if not set(mandatory).issubset(data.columns):
        raise ValueError(f"{ticker}: missing mandatory columns: {set(mandatory)-set(data.columns)}")
    data = data.loc[:, mandatory].rename(
        columns={"Close": "Raw_Close", "Adj Close": "Close"})
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError(f"{ticker}: unsorted or duplicate dates")
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data.isna().any().any() or not np.isfinite(data.to_numpy()).all():
        raise ValueError(f"{ticker}: missing/nonfinite prices; no automatic fill allowed")
    for column in ("Raw_Close", "Close", "Open", "High", "Low"):
        if (data[column] <= 0).any():
            raise ValueError(f"{ticker}: nonpositive {column}")
    if (data["Volume"] < 0).any():
        raise ValueError(f"{ticker}: negative volume")
    # Adj Close need not be between unadjusted high and low.
    if ((data["High"] < data[["Open", "Raw_Close", "Low"]].max(axis=1)) |
        (data["Low"] > data[["Open", "Raw_Close", "High"]].min(axis=1))).any():
        raise ValueError(f"{ticker}: inconsistent unadjusted OHLC")
    if not data.index.isin(pd.date_range(DATA_START, periods=8000, freq="B")).all():
        raise ValueError(f"{ticker}: weekend observation")
    data.index.name = "Date"
    return data


def validate_coverage(frame: pd.DataFrame, market: str) -> dict:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
        raise ValueError(f"{market}: invalid source data")
    if frame.index[0] > pd.Timestamp("2015-01-15"):
        raise ValueError(f"{market}: history missing, starts {frame.index[0].date()}")
    if frame.index[-1] < pd.Timestamp("2026-09-25"):
        raise ValueError(f"{market}: recent history missing, ends {frame.index[-1].date()}")
    if frame.index[-1] >= pd.Timestamp(DATA_END_EXCLUSIVE):
        raise ValueError(f"{market}: exceeds frozen study end")
    if len(frame) < 2600:
        raise ValueError(f"{market}: too few sessions: {len(frame)}")
    returns = np.log(frame["Close"]/frame["Close"].shift(1)).dropna()
    suspicious = returns.loc[returns.abs() > 0.35]
    if len(suspicious):
        raise ValueError(f"{market}: {len(suspicious)} adjusted-return jumps exceed 35%; inspect source")
    return {
        "market": market, "rows": len(frame),
        "first": str(frame.index[0].date()), "last": str(frame.index[-1].date()),
        "adjusted_close_min": float(frame["Close"].min()),
        "adjusted_close_max": float(frame["Close"].max()),
        "max_abs_log_return": float(returns.abs().max()),
        "extreme_return_gt_20pct": int((returns.abs() > .20).sum()),
    }


def build_common_session_datasets(markets: dict[str, pd.DataFrame],
                                  horizons=(1, 5)) -> tuple[dict, dict]:
    if set(markets) != set(MARKETS):
        raise ValueError(f"Expected markets {sorted(MARKETS)}, got {sorted(markets)}")
    common = markets["CSI_300"].index
    for frame in markets.values():
        common = common.intersection(frame.index)
    common = common.sort_values()
    if len(common) < 2600:
        raise ValueError(f"Insufficient three-way common sessions: {len(common)}")
    output = {}
    counts = {
        "common_sessions": len(common), "start": str(common[0].date()),
        "end": str(common[-1].date()), "markets": {},
    }
    for market, frame in markets.items():
        aligned = frame.loc[common, ["Close"]].copy()
        by_split = {}
        for horizon in horizons:
            master = prepare_session_dataset(aligned, horizon=horizon)
            split_counts = {}
            for label, start, end in [
                ("development", DATA_START, TRAIN_END),
                ("retrospective_validation", VALIDATION_START, VALIDATION_END),
                ("post_legacy_evaluation", POST_LEGACY_START, POST_LEGACY_END),
            ]:
                split = chronological_partition(master, start, end)
                split_counts[label] = len(split)
                floor = 1000 if label == "development" else 300
                if len(split) < floor:
                    raise ValueError(
                        f"{market} horizon {horizon}: insufficient {label} observations ({len(split)})")
                if (pd.to_datetime(split["target_end_date"]) > pd.Timestamp(end)).any():
                    raise AssertionError("A target crossed the split boundary")
                by_split[(horizon, label)] = split
            counts["markets"].setdefault(market, {})[str(horizon)] = split_counts
        output[market] = by_split
    return output, counts


def write_immutable_staging(root: Path, markets: dict[str, pd.DataFrame],
                            provider_metadata: dict) -> Path:
    """Create staged input files, provenance hashes and split-aware datasets."""
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"Refuse to overwrite staged release: {root}")
    quality = {market: validate_coverage(frame, market)
               for market, frame in markets.items()}
    model_ready, summary = build_common_session_datasets(markets)
    root.mkdir(parents=True)
    (root/"source_data").mkdir()
    (root/"model_ready").mkdir()
    hashes = {}
    for market, frame in markets.items():
        path = root/"source_data"/f"{market}.csv"
        frame.to_csv(path, float_format="%.15g")
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
        for (horizon, split_name), split in model_ready[market].items():
            path = root/"model_ready"/f"{market}_h{horizon}_{split_name}.csv"
            split.to_csv(path, float_format="%.15g", date_format="%Y-%m-%d")
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    manifest = {
        "release_type": "staged_market_inputs_not_model_results",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider_metadata,
        "quality": quality,
        "common_session_summary": summary,
        "sha256": hashes,
        "important_limitations": [
            "Provider-adjusted close is not a validated investable total-return series.",
            "Post-legacy dates are not automatically an untouched evaluation period.",
            "No model has been fitted or risk/strategy performance evaluated.",
        ],
    }
    path = root/"manifest.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    return path
