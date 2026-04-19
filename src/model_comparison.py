from src.utils.validation import validate_and_save
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils import get_market_path

class ModelComparison:
    def __init__(self):
        pass

    def evaluate(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Pearson correlation ignoring perfectly identical arrays strictly to avoid div-0 bounds structurally
        if np.std(y_pred) == 0 or np.std(y_true) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            
        return rmse, mae, corr

    def compare_models(self, garch_df, ml_df, hybrid_df, market_name):
        print(f"Executing explicit Model Comparison Engine algebraically for {market_name}...")
        
        # Geometrically binding dataframes to intersecting temporal windows safely
        aligned = pd.concat([
            garch_df['Forecast_Vol'].rename('GARCH'),
            ml_df['Forecast_Vol'].rename('ML'),
            hybrid_df['Forecast_Vol'].rename('Hybrid'),
            hybrid_df['Realized_Vol'].rename('Realized')
        ], axis=1).dropna()
        
        if aligned.empty:
            raise ValueError(f"CRITICAL TEAR: Zero overlapping forecasts located for Market: {market_name}")

        y_true = aligned['Realized']
        
        g_rmse, g_mae, g_corr = self.evaluate(y_true, aligned['GARCH'])
        m_rmse, m_mae, m_corr = self.evaluate(y_true, aligned['ML'])
        h_rmse, h_mae, h_corr = self.evaluate(y_true, aligned['Hybrid'])
        
        results = [
            {'Model': 'GARCH', 'RMSE': g_rmse, 'MAE': g_mae, 'Correlation': g_corr},
            {'Model': 'Machine Learning', 'RMSE': m_rmse, 'MAE': m_mae, 'Correlation': m_corr},
            {'Model': 'Adaptive Hybrid (RMSE-weighted)', 'RMSE': h_rmse, 'MAE': h_mae, 'Correlation': h_corr}
        ]
        
        comp_df = pd.DataFrame(results)
        
        # RANKING
        comp_df['Rank'] = comp_df['RMSE'].rank()
        
        # SORTING inherently enforces rank 1 explicitly top
        comp_df = comp_df.sort_values('Rank').reset_index(drop=True)
        
        market_dir = get_market_path(market_name)
        validate_and_save(comp_df, os.path.join(market_dir, "model_comparison.csv"), is_time_series=False, index=False)
        
        print(f"[SUCCESS] Model Comparison completed natively. Extracted Winner: {comp_df.iloc[0]['Model']}")
        return comp_df
