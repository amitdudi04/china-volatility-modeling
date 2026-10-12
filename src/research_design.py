"""Causal target construction and chronological research splits.

All dataset rows are indexed by prediction origin t. Model inputs use returns
known by close t; targets depend exclusively on future sessions t+1...t+h.
"""
import numpy as np
import pandas as pd


def prepare_session_dataset(raw: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("raw must have a DatetimeIndex")
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if "Close" not in raw:
        raise ValueError("Missing Close column")
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("Duplicate or unsorted dates")
    close = raw["Close"].astype(float)
    if raw.empty or not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Invalid close prices")

    r = np.log(close / close.shift(1))
    out = pd.DataFrame(index=raw.index)
    out.index.name = "origin_date"
    out["close_at_origin"] = close
    out["return_at_origin"] = r
    out["lag1_return"] = r
    out["lag1_squared_return"] = r.pow(2)
    out["trailing_5_variance_ann"] = 252 * r.pow(2).rolling(5).mean()
    out["trailing_21_variance_ann"] = 252 * r.pow(2).rolling(21).mean()
    out["trailing_5_vol_ann"] = np.sqrt(out["trailing_5_variance_ann"])
    out["trailing_21_vol_ann"] = np.sqrt(out["trailing_21_variance_ann"])
    out["trailing_5_mean_return"] = r.rolling(5).mean()
    out["trailing_21_mean_return"] = r.rolling(21).mean()

    # At t, none of these future squared returns is available as a feature.
    future_sq = pd.concat([r.shift(-j).pow(2) for j in range(1, horizon + 1)], axis=1)
    out["target_variance_ann"] = 252 * future_sq.mean(axis=1, skipna=False)
    out["target_vol_ann"] = np.sqrt(out["target_variance_ann"])
    out["target_end_date"] = pd.Series(raw.index, index=raw.index).shift(-horizon)
    if horizon == 1:
        out["next_session_log_return"] = r.shift(-1)
        out["target_squared_log_return"] = future_sq.iloc[:, 0]

    feature_cols = [
        "lag1_return", "lag1_squared_return", "trailing_5_variance_ann",
        "trailing_21_variance_ann", "trailing_5_vol_ann", "trailing_21_vol_ann",
        "trailing_5_mean_return", "trailing_21_mean_return"
    ]
    valid = out[feature_cols + ["target_variance_ann", "target_end_date"]].notna().all(axis=1)
    return out.loc[valid].copy()


def chronological_partition(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError("Invalid split dates")
    if "target_end_date" not in frame:
        raise ValueError("Missing target_end_date")
    target_end = pd.to_datetime(frame["target_end_date"])
    mask = (frame.index >= start_ts) & (frame.index <= end_ts)
    mask &= (target_end >= start_ts) & (target_end <= end_ts)
    return frame.loc[mask].copy()


def completed_training_rows(frame: pd.DataFrame, current_origin: pd.Timestamp) -> pd.DataFrame:
    t = pd.Timestamp(current_origin)
    # Targets are usable only after their completion date.
    mask = (frame.index < t) & (pd.to_datetime(frame["target_end_date"]) <= t)
    return frame.loc[mask].copy()


def verify_feature_causality(raw: pd.DataFrame, horizon: int = 1,
                             prefix_sessions: int | None = None) -> bool:
    cols = [
        "close_at_origin", "return_at_origin", "lag1_return",
        "lag1_squared_return", "trailing_5_variance_ann",
        "trailing_21_variance_ann", "trailing_5_vol_ann",
        "trailing_21_vol_ann", "trailing_5_mean_return",
        "trailing_21_mean_return"
    ]
    if prefix_sessions is None:
        prefix_sessions = max(35, len(raw)//2)
    if not 25 < prefix_sessions < len(raw)-horizon:
        raise ValueError("Invalid prefix length")
    short = prepare_session_dataset(raw.iloc[:prefix_sessions], horizon)
    long = prepare_session_dataset(raw, horizon)
    common = short.index.intersection(long.index)
    if len(common) == 0:
        raise ValueError("No comparable rows")
    pd.testing.assert_frame_equal(short.loc[common, cols], long.loc[common, cols], check_exact=True)
    return True
