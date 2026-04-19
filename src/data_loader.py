from src.utils.validation import validate_and_save
import yfinance as yf
import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Target markets (China Focus)
        self.markets = {
            "CSI_300": "000300.SS",
            "SSE_Composite": "000001.SS",
            "ChiNext": "159915.SZ"  # ETF proxy (works reliably)
        }
        self.start_date = "2015-01-01"
        self.end_date = "2024-12-31"

    def fetch_market_data(self, market_name, ticker):
        """Fetch data from yfinance and save locally per market to ensure isolation."""
        file_path = os.path.join(self.data_dir, f"{market_name}.csv")
        if os.path.exists(file_path):
            print(f"Data for {market_name} already exists. Loading from {file_path}")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            print(f"[DEBUG DATALOADER] {market_name} index type: {type(df.index)}")
            return df
            
        print(f"Downloading {market_name} ({ticker}) data from {self.start_date} to {self.end_date}...")
        try:
            df = yf.download(ticker, start=self.start_date, end=self.end_date)
            # Flatten multi-indexed columns if yfinance returns them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df is None or df.empty:
                raise ValueError(f"{market_name} download returned empty DataFrame")    
            validate_and_save(df, file_path, is_time_series=True)
            print(f"Saved {market_name} data to {file_path}")
            print(f"[DEBUG DATALOADER] {market_name} index type: {type(df.index)}")
            return df
        except Exception as e:
            print(f"[WARNING] {market_name} download failed — skipping ({str(e)})")
            return None

    def run(self):
        datasets = {}
        for name, ticker in self.markets.items():
            print(f"[DEBUG DATALOADER] Attempting to load {name}...")
            try:
                df = self.fetch_market_data(name, ticker)
                if df is None:
                    print(f"[WARNING] Skipping {name} — no data returned")
                    continue

                if df.empty:
                    print(f"[WARNING] Skipping {name} — dataframe is empty")
                    continue

                print(f"[DEBUG DATALOADER] Success: {name} has {len(df)} rows")
                datasets[name] = df
            except Exception as e:
                print(f"[WARNING] Skipping {name}: {e}")
                continue
        return datasets

if __name__ == "__main__":
    loader = DataLoader()
    loader.run()
