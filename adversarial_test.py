import pandas as pd
import numpy as np
import os

with open("adversarial_report.txt", "w") as f:
    def log(msg=""):
        f.write(msg + "\n")
        
    def test_market(market):
        log(f"--- ADVERSARIAL TEST: {market} ---")
        market_dir = f"outputs/results/{market}"
        
        # 1. VOLATILITY
        vol_file = os.path.join(market_dir, "volatility_comparison.csv")
        if os.path.exists(vol_file):
            df_vol = pd.read_csv(vol_file)
            f_max = df_vol['Forecast_Vol'].max()
            f_min = df_vol['Forecast_Vol'].min()
            r_max = df_vol['Realized_Vol'].max()
            
            log(f"Vol Max (Forecast): {f_max:.2f}%")
            log(f"Vol Min (Forecast): {f_min:.2f}%")
            log(f"Vol Max (Realized): {r_max:.2f}%")
            if f_max > 60: log("!! FORECAST VOL SPIKE > 60% DETECTED !!")
            if r_max > 60: log("!! REALIZED VOL SPIKE > 60% DETECTED !!")
            
        # 2. STRATEGY TURNOVER
        econ_ts_file = os.path.join(market_dir, f"{market}_economic_timeseries.csv")
        if os.path.exists(econ_ts_file):
            df_ts = pd.read_csv(econ_ts_file)
            if 'Weight' in df_ts.columns:
                turnover = df_ts['Weight'].diff().abs()
                avg_turnover = turnover.mean()
                max_turnover = turnover.max()
                log(f"Strategy Avg Daily Turnover: {avg_turnover*100:.2f}%")
                log(f"Strategy Max Daily Turnover: {max_turnover*100:.2f}%")
                if max_turnover > 0.5: log("!! EXTREME TURNOVER DETECTED !!")
                
        # 3. REGIME STABILITY
        regime_file = os.path.join(market_dir, f"{market}_regimes.csv")
        if os.path.exists(regime_file):
            df_reg = pd.read_csv(regime_file)
            if 'Regime_1_Prob' in df_reg.columns:
                df_reg['Regime'] = (df_reg['Regime_1_Prob'] > 0.7).astype(int)
                switches = (df_reg['Regime'].diff().abs() > 0).sum()
                avg_dur = len(df_reg) / max(1, switches)
                log(f"Regime Switches: {switches}")
                log(f"Avg Regime Duration: {avg_dur:.1f} days")
                if avg_dur < 10: log("!! HIGH REGIME INSTABILITY DETECTED !!")
                
        # 4. VaR CONSISTENCY
        risk_file = os.path.join(market_dir, "risk_forecasts.csv")
        if os.path.exists(risk_file):
            df_v = pd.read_csv(risk_file)
            if 'Violation' in df_v.columns:
                viol_pct = df_v['Violation'].mean() * 100
                log(f"VaR Violation Freq: {viol_pct:.2f}%")
                violations = df_v['Violation'].astype(int)
                consecutive = (violations * violations.shift(1)).sum()
                log(f"Consecutive VaR Failures (Clustering): {consecutive}")
                if consecutive > max(1, len(df_v) * 0.01):
                    log("!! SEVERE VaR CLUSTERING DETECTED !!")
                    
        # 5. METRIC ROBUSTNESS
        econ_file = os.path.join(market_dir, "economic_performance.csv")
        if os.path.exists(econ_file):
            df_econ = pd.read_csv(econ_file)
            log("\nSharpe Ranges:")
            if 'Strategy' in df_econ.columns and 'Sharpe' in df_econ.columns:
                for idx, row in df_econ.iterrows():
                    log(f"  {row['Strategy']}: {row['Sharpe']:.2f}")
            log("\nMax DD Ranges:")
            if 'Strategy' in df_econ.columns and 'Max DD' in df_econ.columns:
                for idx, row in df_econ.iterrows():
                    log(f"  {row['Strategy']}: {row['Max DD']:.2f}")
            
        log("\n")

    for m in ["CSI_300", "SSE_Composite", "ChiNext"]:
        test_market(m)
