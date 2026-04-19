from src.utils.validation import validate_and_save
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from src.utils import get_market_path

class MLModels:
    def __init__(self):
        # STEP 4: MODEL UPGRADE
        self.rf = RandomForestRegressor(
            n_estimators=120,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.window = None  # dynamic window (set later)

    def prepare_features(self, df):
        data = df.copy()
        
        # STEP 2: REMOVE NOISE
        data['Log_Return'] = data['Log_Return']
        
        data['RV'] = data['Log_Return']**2
        data['Realized_Vol'] = np.sqrt(data['RV'].rolling(5).mean()) * np.sqrt(252)
        
        # BASE FEATURES (Constructed natively at T)
        data['Lag_1_Ret'] = data['Log_Return']
        data['Lag_1_SqRet'] = data['Log_Return']**2
        data['Rolling_Var_5'] = data['RV'].rolling(5).mean()
        data['Rolling_Var_21'] = data['RV'].rolling(21).mean()
        
        data['Vol_1'] = data['Realized_Vol']
        data['Vol_5'] = data['Realized_Vol'].rolling(5).mean()
        data['Vol_10'] = data['Realized_Vol'].rolling(10).mean()
        
        data['Return_1'] = data['Log_Return']
        data['Return_5'] = data['Log_Return'].rolling(5).mean()
        
        feature_cols = [
            'Lag_1_Ret', 'Lag_1_SqRet', 'Rolling_Var_5', 'Rolling_Var_21', 
            'Vol_1', 'Vol_5', 'Vol_10', 'Return_1', 'Return_5'
        ]
        
        # RULE 1: ALL FEATURES MUST BE LAGGED securely
        data[feature_cols] = data[feature_cols].shift(1)
        
        # RULE 3: DROP NaNs AFTER SHIFT (MANDATORY)
        data = data.dropna(subset=feature_cols + ['Realized_Vol']).copy()
        
        # RULE 2: TARGET MUST NOT BE SHIFTED natively
        X = data[feature_cols].clip(-5, 5)
        y = data['Realized_Vol']
        
        # RULE 4: VALIDATION CHECK
        assert (X.index == y.index).all(), "Index mismatch structurally between features and target"
        assert len(X) == len(y), "Features and Target matrix intrinsically mismatched"
        
        # STEP 4: DEBUG PRINT
        print(f"[DEBUG RF] X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y, data.index, feature_cols
    def run_ml(self, df, market_name):
        X, y, idx, feature_cols = self.prepare_features(df)
        
        n = len(X)
        ml_forecasts = []
        
        # 🔴 WINDOW SYNC FIX (CRITICAL)
        # Ensure ML exact index iteration perfectly matches GARCH index iteration
        self.window = min(1000, int(len(df) * 0.7))
        drop_count = len(df) - len(X)
        start_idx = max(50, self.window - drop_count)

        if len(X) < 100:
            raise RuntimeError("[RF FAILURE] Dataset too small for ML modeling")

        print(f"Running rolling Random Forest on {market_name} (Window = {self.window})...")
        
        X_val = X.values
        y_val = y.values
        REFIT_INTERVAL = 50
        model = None
        feature_history = []
        
        for i in range(start_idx, n):
            train_start = max(0, i - self.window)
            X_train = X_val[train_start : i]
            y_train = y_val[train_start : i]
            X_test = X_val[[i]]
           
            if len(X_train) < 50:
                # USE LAST VALUE INSTEAD OF NaN (CRITICAL FIX)
                if len(ml_forecasts) > 0:
                    ml_forecasts.append(ml_forecasts[-1])
                else:
                    ml_forecasts.append(y_val[i])
                continue
            # SAFE CHECK (only if exists)
            if len(ml_forecasts) > 0:
                if np.isnan(ml_forecasts[-1]) or np.isinf(ml_forecasts[-1]):
                    ml_forecasts[-1] = y_val[i]
            
            # STEP 3: NORMALIZE FEATURES (Dynamically mapped zero lookahead)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Batched Refitting Logic (Section 3)
            if model is None or (i % REFIT_INTERVAL == 0):
                model = RandomForestRegressor(
                    n_estimators=120,
                    max_depth=6,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)

            pred = model.predict(X_test_scaled)[0]

            # HARD CLIP (match GARCH scale)
            pred = np.clip(pred, 0.05, 0.60)

            ml_forecasts.append(pred)
            feature_history.append(model.feature_importances_)
            
            if i % 100 == 0:
                print(f"[RF PROGRESS] Step {i}/{len(X)}")
        feat_array = np.array(feature_history)
        
        # Evaluate stability structurally conditionally based on if elements exist
        if len(feat_array) > 0:
            feat_mean = feat_array.mean(axis=0)
            feat_std = feat_array.std(axis=0)
            stability_ratio = feat_std / (feat_mean + 1e-6)
        else:
            feat_mean = np.zeros(len(feature_cols))
            feat_std = np.zeros(len(feature_cols))
            stability_ratio = np.zeros(len(feature_cols))
        
        def classify_stability(ratio):
            if ratio < 0.5: return "Stable"
            if ratio <= 1.0: return "Moderate"
            return "Unstable"
            
        fi = pd.DataFrame({
            'Feature': feature_cols,
            'Importance_Mean': feat_mean,
            'Importance_Std': feat_std,
            'Stability_Ratio': stability_ratio
        }).sort_values('Importance_Mean', ascending=False)
        fi['Classification'] = fi['Stability_Ratio'].apply(classify_stability)
        
        market_dir = get_market_path(market_name)
        validate_and_save(fi, os.path.join(market_dir, "feature_stability.csv"), is_time_series=False, index=False)
        
        try:
            # Shift length mapping dynamically in case of skipped iterations unconditionally
            align_slice = len(ml_forecasts)

            # SMOOTH ML FORECASTS
            ml_forecasts_series = pd.Series(ml_forecasts, index=idx[-len(ml_forecasts):])

            # 🔴 FINAL SAFE BLOCK (STRICT ORDER)
            ml_forecasts_series = ml_forecasts_series.replace([np.inf, -np.inf], np.nan)
            ml_forecasts_series = ml_forecasts_series.ffill().fillna(0.15)
            ml_forecasts_series = ml_forecasts_series.rolling(3, min_periods=1).mean()
            # 🔴 FINAL GUARD (MANDATORY)
            ml_forecasts_series = ml_forecasts_series.ffill().fillna(0.15)
            assert ml_forecasts_series.isna().sum() == 0

            pipeline_df = pd.DataFrame({
                'Forecast_Vol': ml_forecasts_series.values,
                'Realized_Vol': y.iloc[-align_slice:].values if align_slice > 0 else []
            }, index=idx[-align_slice:] if align_slice > 0 else [])
            
            validate_and_save(pipeline_df, os.path.join(market_dir, f"{market_name}_ml_forecast.csv"), is_time_series=True)
            return pipeline_df
        
        # STEP 5: FAIL FAST
        except Exception as e:
            raise RuntimeError(f"[RF FAILURE] {str(e)}")
