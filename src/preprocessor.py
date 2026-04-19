from src.utils.validation import validate_and_save
import pandas as pd
import numpy as np
import os

class Preprocessor:
    def __init__(self, data_dir="data/processed"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def process(self, df, market_name):
        """Prepare raw price data into log returns and perform basic cleaning."""
        print(f"Preprocessing data for {market_name}...")
        
        # Ensure 'Close' price exists
        if 'Close' not in df.columns:
            raise KeyError(f"Close column missing in {market_name} data.")
            
        # Forward fill missing values (common in EM holidays)
        df_processed = df.copy()
        
        # Address index frequency warnings
        df_processed.index = pd.to_datetime(df_processed.index)

        # DO NOT FORCE BUSINESS DAYS
        df_processed = df_processed.sort_index()
        df_processed = df_processed.ffill()

        # ONLY ensure Close exists
        df_processed = df_processed.dropna(subset=['Close'])
        
        # Compute log returns
        df_processed['Log_Return'] = np.log(df_processed['Close'] / df_processed['Close'].shift(1))
        
        # Drop the first row which has NaN log return
        df_processed = df_processed.dropna(subset=['Log_Return'])
        
        # Calculate Realized Volatility Proxy (Daily squared returns)
        df_processed['RV'] = df_processed['Log_Return'] ** 2
        
        # Optional: Save preprocessed data
        out_path = os.path.join(self.data_dir, f"{market_name}_processed.csv")
        validate_and_save(df_processed, out_path, is_time_series=True)
        print(f"Saved processed {market_name} data to {out_path}.")
        print(f"[DEBUG PREPROCESSOR] Final rows for {market_name}: {len(df_processed)}")
        return df_processed

    def describe(self, df, market_name):
        """Output expected descriptive statistics for paper methodology."""
        desc = df['Log_Return'].describe()
        desc['Skewness'] = df['Log_Return'].skew()
        desc['Kurtosis'] = df['Log_Return'].kurtosis()
        
        print(f"\\n--- {market_name} Descriptive Stats ---")
        print(desc)
        return desc
