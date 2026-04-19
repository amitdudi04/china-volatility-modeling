from src.utils.validation import validate_and_save
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from src.utils import get_market_path

class RegimePerformance:
    def __init__(self):
        pass

    def evaluate_regimes(self, garch_df, ml_df, hybrid_df, regime_labels_df, market_name):
        print(f"Executing explicit Regime Performance Analysis structurally for {market_name}...")
        
        # Align all model forecasts strictly alongside dynamic Regime labels mathematically
        aligned = pd.concat([
            garch_df['Forecast_Vol'].rename('GARCH'),
            ml_df['Forecast_Vol'].rename('ML'),
            hybrid_df['Forecast_Vol'].rename('Hybrid'),
            hybrid_df['Realized_Vol'].rename('Realized'),
            regime_labels_df['Regime_Label'].rename('Regime')
        ], axis=1).dropna()
        
        if aligned.empty:
            raise ValueError(f"CRITICAL TEAR: Zero overlapping temporal forecasts found for Regime split on: {market_name}")

        high_vol = aligned[aligned['Regime'] == 1]
        low_vol = aligned[aligned['Regime'] == 0]
        
        def safe_rmse(df, pred_col):
            if df.empty: return np.nan
            return np.sqrt(mean_squared_error(df['Realized'], df[pred_col]))
            
        results = []
        for model in ['GARCH', 'ML', 'Hybrid']:
            results.append({
                'Model': model,
                'RMSE_HighVol': safe_rmse(high_vol, model),
                'RMSE_LowVol': safe_rmse(low_vol, model)
            })
            
        perf_df = pd.DataFrame(results)
        
        market_dir = get_market_path(market_name)
        validate_and_save(perf_df, os.path.join(market_dir, "regime_performance.csv"), is_time_series=False, index=False)
        
        print(f"[SUCCESS] Regime-based performance evaluation complete natively.")
        return perf_df
