import pandas as pd
import os

def load_time_series_csv(path):
    """
    Centralized safe loader for time-series data.
    Ensures DatetimeIndex, sorted order, and lack of duplicates.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if df.empty:
        raise ValueError(f"Empty DataFrame: {path}")

    # REMOVE DUPLICATES FIRST
    df = df[~df.index.duplicated(keep='first')]

    # ASSERT DATETIME
    assert isinstance(df.index, pd.DatetimeIndex), \
        f"DatetimeIndex lost: {path}"

    # SORT
    df = df.sort_index()

    return df

def validate_and_save(df, path, is_time_series=False, **kwargs):
    """
    Core validation root wrapping CSV serialization structurally.
    Ensures DatetimeIndex preservation and strict data integrity.
    """
    assert df is not None, "DataFrame is None"
    assert not df.empty, "Empty DataFrame"

    if is_time_series:
        # SECTION 6 — FINAL SYSTEM VALIDATION (MANDATORY for Time-Series/Forecasts)
        if df.isna().all().any():
            raise ValueError("Invalid dataset (all NaN column detected)")
        
        MIN_REQUIRED = 10  # allow small rolling outputs
        if len(df) < MIN_REQUIRED:
            print(f"[WARNING] Small dataset (len={len(df)}) — continuing")
            
        assert isinstance(df.index, pd.DatetimeIndex), "Missing DatetimeIndex"
        
        # DatetimeIndex Contract (Section 2)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        df.index.name = "Date"
        
        # Ensure index=True for time series unless explicitly overridden
        if 'index' not in kwargs:
            kwargs['index'] = True
        df.to_csv(path, **kwargs)
    else:
        # Tabular data (Summary tables, model comparison)
        if 'index' not in kwargs:
            kwargs['index'] = False
        df.to_csv(path, **kwargs)
