import pandas as pd
import numpy as np
import os

with open("final_review_output.txt", "w") as f:
    def log(msg=""):
        f.write(msg + "\n")
        
    for market in ["CSI_300", "SSE_Composite", "ChiNext"]:
        log(f"=== {market} ===")
        market_dir = f"outputs/results/{market}"
        
        # 1. VOLATILITY ALIGNMENT
        vol_file = os.path.join(market_dir, "volatility_comparison.csv")
        if os.path.exists(vol_file):
            df_vol = pd.read_csv(vol_file).dropna()
            corr = df_vol['Forecast_Vol'].corr(df_vol['Realized_Vol'])
            f_mean = df_vol['Forecast_Vol'].mean()
            r_mean = df_vol['Realized_Vol'].mean()
            log(f"Vol Correlation: {corr:.4f}")
            log(f"Mean Forecast Vol: {f_mean:.2f}% | Mean Realized Vol: {r_mean:.2f}%")
            
        # 2. VaR VALIDATION
        var_file = os.path.join(market_dir, "var_results.csv")
        if os.path.exists(var_file):
            df_v = pd.read_csv(var_file)
            if 'Actual_Violations' in df_v.columns and 'Expected_Violations' in df_v.columns:
                actual = df_v['Actual_Violations'].iloc[0]
                expected = df_v['Expected_Violations'].iloc[0]
                log(f"VaR Violations: {actual} (Expected: {expected})")
                log(f"VaR Interpretation: {df_v['Interpretation'].iloc[0]}")
            
        # 3. REGIME FAILURE RATE
        econ_perf = os.path.join(market_dir, "economic_performance.csv")
        econ_ts = os.path.join(market_dir, f"{market}_economic_timeseries.csv")
        if os.path.exists(econ_perf):
            df_perf = pd.read_csv(econ_perf)
            status = df_perf['Regime_Status'].iloc[0] if 'Regime_Status' in df_perf.columns else "UNKNOWN"
            log(f"Regime Status: {status}")
            
        # 4. STRATEGY PERFORMANCE
        if os.path.exists(econ_perf):
            log("Performance:")
            df_perf = pd.read_csv(econ_perf)
            for idx, row in df_perf.iterrows():
                if row['Strategy'] in ['Static Exposure', 'Volatility-Managed']:
                    log(f"  {row['Strategy']}: Ret={row['Ann Return']:.2f}%, Sharpe={row['Sharpe']:.2f}, MDD={row['Max DD']:.2f}%")
                    
        # 4.5. STRESS TEST PERFORMANCE
        stress_file = os.path.join(market_dir, f"{market}_stress_test.csv")
        if os.path.exists(stress_file):
            log("\nStress Test (Regime == 1 Isolated):")
            df_st = pd.read_csv(stress_file)
            for idx, row in df_st.iterrows():
                log(f"  {row['Strategy']}: Ret={row['Ann_Return']:.2f}%, Sharpe={row['Sharpe']:.2f} (Diff: {row['Sharpe_Diff']:.2f}), MDD={row['Max_DD']:.2f}% (Diff: {row['MDD_Diff']:.2f})")
        
        # 4.6 Feature Stability Check
        feat_file = os.path.join(market_dir, "feature_importance.csv")
        if os.path.exists(feat_file):
            df_f = pd.read_csv(feat_file)
            log("\nFeature Stability Matrix:")
            for idx, row in df_f.iterrows():
                log(f"  {row['Feature']}: Mean={row['Importance_Mean']:.4f}, Std={row['Importance_Std']:.4f} -> [{row['Classification']}]")
        
        # 4.7 Crisis Capture
        crisis_file = os.path.join(market_dir, f"{market}_crisis_capture.csv")
        if os.path.exists(crisis_file):
            df_c = pd.read_csv(crisis_file)
            log("\nCrisis Capture Overlap:")
            for idx, row in df_c.iterrows():
                log(f"  {row['Crisis']}: {row['Capture_Rate']*100:.1f}% ({row['Captured_Days']} / {row['Total_Days']} days)")
                
        # 5. TURNOVER
        if os.path.exists(econ_ts):
            df_ts = pd.read_csv(econ_ts)
            if 'Weight' in df_ts.columns:
                turnover = df_ts['Weight'].diff().abs().mean() * 100
                log(f"Avg Turnover: {turnover:.2f}%")
        log("\n")
