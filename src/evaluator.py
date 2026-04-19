from src.utils.validation import validate_and_save
import numpy as np
import pandas as pd
import os

from src.utils import get_market_path

class ModelEvaluator:
    def evaluate(self, forecast_df, model_name, market_name):
        market_dir = get_market_path(market_name)
        df = forecast_df.dropna().copy()
        
        # Strictly evaluating using Volatility
        true_vol = df['Realized_Vol']
        pred_vol = df['Forecast_Vol']
        
        mse = np.mean((true_vol - pred_vol)**2)
        rmse = np.sqrt(mse)
        
        # QLIKE Loss Function
        safe_pred = np.where(pred_vol <= 0, 1e-8, pred_vol)
        safe_true = np.where(true_vol <= 0, 1e-8, true_vol)
        qlike = np.mean(safe_true / safe_pred - np.log(safe_true / safe_pred) - 1)
        
        corr = np.corrcoef(true_vol, pred_vol)[0,1]
        metrics = {
            'Market': market_name,
            'Model_Type': model_name,
            'MSE': mse,
            'RMSE': rmse,
            'QLIKE': qlike,
            'Correlation': corr
        }
        
        df_res = pd.DataFrame([metrics])
        out_path = os.path.join(market_dir, f"{market_name}_evaluation.csv")
        
        # Append if exists
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            df_res = pd.concat([existing, df_res])
            df_res = df_res.drop_duplicates(subset=["Market", "Model_Type"], keep="last").reset_index(drop=True)
        
        validate_and_save(df_res, out_path, is_time_series=False, index=False)
        return df_res
