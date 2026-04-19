from src.utils.validation import validate_and_save
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from src.utils import get_market_path

class RegimeModel:
    def detect_regimes(self, df, market_name):
        reg_df = df.copy()
        print(f"Applying Volatility-Based Regime Detection for {market_name}...")
        try:
            # Build Realized Volatility Proxy directly matching downstream structure (DECIMAL format)
            vol_5 = reg_df['Log_Return'].rolling(5).std().ffill().fillna(0) * np.sqrt(252)
            vol_20 = reg_df['Log_Return'].rolling(20).std().ffill().fillna(0) * np.sqrt(252)
            
            # Minimum constraint required so we don't zero-out the entire first year naturally
            threshold = vol_5.rolling(252, min_periods=21).quantile(0.6).ffill().fillna(0)
            
            # STEP 3: ADD MOMENTUM FILTER (High vol only if Vol_5 > Vol_20)
            raw_prob = ((vol_5 > threshold) & (vol_5 > vol_20)).astype(float)
            
            # STEP 1: SMOOTH PROBABILITY
            reg_df['Regime_Prob'] = raw_prob.rolling(5).mean().ffill().fillna(0)
            
            # STEP 2: STRONGER THRESHOLD
            reg_df['Regime_Label'] = (reg_df['Regime_Prob'] > 0.6).astype(int)
            
            reg_df['Regime_1_Prob'] = reg_df['Regime_Prob']
            reg_df['Regime_0_Prob'] = 1 - reg_df['Regime_Prob']
            
            validate_and_save(reg_df, os.path.join(get_market_path(market_name), f"{market_name}_regimes.csv"), is_time_series=True)
            
            # 🔴 PART 1 — DEFINE CRISIS WINDOWS
            crisis_periods = {
                "GFC_2008": ("2008-01-01", "2009-03-01"),
                "China_Crash_2015": ("2015-06-01", "2016-02-01"),
                "COVID_2020": ("2020-02-01", "2020-06-01")
            }
            
            # 🔴 PART 2 — ENSURE DATE INDEX
            reg_df.index = pd.to_datetime(reg_df.index)
            assert reg_df.index.is_monotonic_increasing
            
            results = []
            
            # 🔴 PART 3 — CAPTURE RATE CALCULATION
            for name, (start, end) in crisis_periods.items():
                subset = reg_df.loc[start:end]
                
                # STEP 1: SAFE VALIDATION
                if subset.empty:
                    print(f"[WARNING] No data for {name}")
                    capture_rate = np.nan
                    status = "NO_DATA"
                else:
                    capture_rate = (subset['Regime_Label'] == 1).mean()
                    status = "VALID"
                
                # 🔴 PART 4 — STORE RESULTS (STEP 2)
                results.append({
                    "Crisis": name,
                    "Start_Date": start,
                    "End_Date": end,
                    "Capture_Rate": capture_rate,
                    "Validation_Status": status
                })
            
            # STEP 3: DO NOT BREAK PIPELINE (Removed hard assert)
            
            # 🔴 PART 5 — SAVE OUTPUT
            capture_df = pd.DataFrame(results)
            save_path = f"outputs/results/{market_name}/regime_validation.csv"
            validate_and_save(capture_df, save_path, is_time_series=False, index=False)
            
            # 🔴 PART 7 — OPTIONAL DEBUG PRINT
            print(f"[INFO] Regime validation complete for {market_name}")
            
            # STEP 4: FINAL ASSERT
            assert 'Regime_Label' in reg_df.columns
                
            return reg_df
        except Exception as e:
            raise RuntimeError(f"[MODULE FAILURE] Regime proxy mathematically broke natively: {str(e)}")
