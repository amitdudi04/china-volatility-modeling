import pandas as pd

def validate_data(df):
    if df.isnull().sum().sum() > 0:
        print("[WARNING] Missing values detected")

    if len(df) < 200:
        raise ValueError(f"Critical: dataset too small ({len(df)})")

    if 'Log_Return' in df.columns and df['Log_Return'].std() == 0:
        raise ValueError("Zero variance detected")

    return True
