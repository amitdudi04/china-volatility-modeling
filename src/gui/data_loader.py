import os
import pandas as pd
from src.utils.validation import load_time_series_csv

class GUIDataLoader:
    def __init__(self, base_outputs_dir="outputs/results"):
        self.base_outputs_dir = base_outputs_dir
        self.data_cache = {}

    def load_market_data(self, market):
        if market in self.data_cache:
            return self.data_cache[market]

        self.data_cache[market] = {}
        market_res_dir = os.path.join(self.base_outputs_dir, market)
        
        paths = {
            "hybrid": os.path.join(market_res_dir, f"{market}_hybrid_forecast.csv"),
            "weights": os.path.join(market_res_dir, "model_weights.csv"),
            "feat": os.path.join(market_res_dir, "feature_stability.csv"),
            "var_res": os.path.join(market_res_dir, "var_results.csv"),
            "var_comp": os.path.join(market_res_dir, "var_comparison.csv"),
            "risk_for": os.path.join(market_res_dir, "risk_forecasts.csv"),
            "econ": os.path.join(market_res_dir, "economic_performance.csv"),
            "econ_ts": os.path.join(market_res_dir, f"{market}_economic_timeseries.csv"),
            "sys": os.path.join(market_res_dir, "validation_report.csv"),
            "raw": f"data/processed/{market}_processed.csv",
            "stat": os.path.join(market_res_dir, "statistical_tests.csv")
        }
        
        ts_keys = ["hybrid", "risk_for", "econ_ts", "raw"]
        for key, path in paths.items():
            if os.path.exists(path):
                if key in ts_keys:
                    try:
                        self.data_cache[market][key] = load_time_series_csv(path)
                    except Exception as e:
                        print(f"[GUI WARNING] Failed loading {key}: {e}")
                        self.data_cache[market][key] = pd.DataFrame()
                else:
                    self.data_cache[market][key] = pd.read_csv(path)
            else:
                self.data_cache[market][key] = pd.DataFrame()
        return self.data_cache[market]
