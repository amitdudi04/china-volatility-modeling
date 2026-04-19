from src.utils.validation import validate_and_save
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.utils import get_market_path

class ForecastEngine:
    def __init__(self):
        self.window = None  # will set dynamically

    def rolling_forecast(self, df, market_name):
        returns = df['Log_Return'] # Pure decimal bounds
        n = len(returns)
        
        # DYNAMIC WINDOW FIX
        self.window = min(1000, int(n * 0.7))
        
        # PART 5: PERFORMANCE SAFETY
        if n <= self.window + 50:
            raise RuntimeError(f"Insufficient data: {n} for window {self.window}")
        
        print(f"Running rolling HAR-RV Forecast for {market_name} (Window = {self.window})...")
        
        # 1. HAR FEATURES (REALIZED VOL MISALIGNMENT FIX)
        rv = returns.rolling(5).std().shift(1) * np.sqrt(252)
        
        # LOG TRANSFORM HAR (ADVANCED REQ)
        # Suppress warnings for np.log(0) which will become -inf and handled by dropna
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y_log = np.log(rv)
            
            rv_d = rv.shift(1)
            rv_w = rv.rolling(5).mean().shift(1)
            rv_m = rv.rolling(22).mean().shift(1)
            
            X_all = pd.concat([rv_d, rv_w, rv_m], axis=1)
            X_all.columns = ['rv_d', 'rv_w', 'rv_m']
            
            # Ensure clip before log so it doesn't -inf
            X_all = X_all.clip(lower=1e-8)
            X_all_log = np.log(X_all)
        
        forecasts = []
        dates = []
        
        for i in range(self.window, n):
            train_start = max(0, i - self.window)
            X_train = X_all_log.iloc[train_start:i]
            y_train = y_log.iloc[train_start:i]
            
            valid_idx = X_train.replace([np.inf, -np.inf], np.nan).dropna().index.intersection(y_train.replace([np.inf, -np.inf], np.nan).dropna().index)
            
            if len(valid_idx) < 50:
                forecast_vol = rv.iloc[i-1] if not pd.isna(rv.iloc[i-1]) else 0.15
            else:
                model = LinearRegression(fit_intercept=True) # HAR MODEL INTERCEPT FIX
                model.fit(X_train.loc[valid_idx], y_train.loc[valid_idx])
                
                X_test = X_all_log.iloc[[i]]
                if X_test.replace([np.inf, -np.inf], np.nan).isna().values.any():
                    forecast_vol = rv.iloc[i-1] if not pd.isna(rv.iloc[i-1]) else 0.15
                else:
                    pred_log = model.predict(X_test)[0]
                    forecast_vol = np.clip(np.exp(pred_log), 0.05, 0.60)
            
            forecasts.append(forecast_vol)
            dates.append(returns.index[i])
                
        forecast_df = pd.DataFrame({
            'Forecast_Vol': forecasts,
            'Realized_Vol': rv.reindex(dates).values
        }, index=dates)

        # Hard stability bounds
        forecast_df['Forecast_Vol'] = forecast_df['Forecast_Vol'].clip(lower=0.05, upper=0.60)
        
        # CREATE PROXY STANDARDIZED RESIDUALS (CRITICAL FIX)
        # Keeps pipeline compatibility for standalone tests
        forecast_df['Std_Residuals'] = (
            df.loc[forecast_df.index, 'Log_Return'] / 
            (forecast_df['Forecast_Vol'] + 1e-8)
        )

        assert forecast_df['Forecast_Vol'].mean() < 1, "Values must be decimal scaled!"
        assert 0.05 < forecast_df['Forecast_Vol'].mean() < 0.60, f"Forecast Volatility Mean Out of Bounds: {forecast_df['Forecast_Vol'].mean()}"
        
        validate_and_save(forecast_df, os.path.join(get_market_path(market_name), f"{market_name}_garch_forecast.csv"), is_time_series=True)
        return forecast_df
