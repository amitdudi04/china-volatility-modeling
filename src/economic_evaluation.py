from src.utils.validation import validate_and_save
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from src.utils import get_market_path
import matplotlib

class EconomicEvaluation:
    def __init__(self):
        pass

    def evaluate(self, market_name, actual_returns, vol_forecasts, target_vol=0.20):
        print(f"Running Economic Evaluation Layer for {market_name}...")
        
        plt.style.use("dark_background")
        matplotlib.rcParams.update({
            'figure.facecolor': '#1e1e1e',
            'axes.facecolor': '#1e1e1e',
            'savefig.facecolor': '#1e1e1e',
            'text.color': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'grid.color': '#333333'
        })
        
        market_dir = get_market_path(market_name)
        
        aligned_df = pd.DataFrame(index=vol_forecasts.index)
        aligned_df['Actual_Return'] = actual_returns.loc[aligned_df.index]
        
        # 1. Smoothed volatility
        forecast_vol = vol_forecasts.loc[aligned_df.index]
        
        df = pd.DataFrame({'Actual_Return': actual_returns, 'Forecast_Vol': forecast_vol}).dropna()
        
        # 4. Ensure no division errors
        df['Forecast_Vol'] = df['Forecast_Vol'].clip(lower=1e-6)
        
        # Extract Regime Data securely to shape Strategy 3 mapping
        regime_path = os.path.join(market_dir, f"{market_name}_regimes.csv")
        try:
            reg_df = pd.read_csv(regime_path, index_col=0, parse_dates=True)
            assert isinstance(reg_df.index, pd.DatetimeIndex), "DatetimeIndex lost during load in economic_evaluation"
            aligned_reg = reg_df.reindex(df.index).ffill()
            df['Regime_1_Prob'] = aligned_reg['Regime_1_Prob']
            regime_status = "VALID"
            df['Regime_Valid'] = True
        except Exception:
            regime_status = "FAILED"
            df['Regime_Valid'] = False
            df['Regime_1_Prob'] = np.nan

        # Strategy 1 — Static Exposure
        df['Static_Weight'] = 1.0
        df['Static_Return'] = df['Actual_Return'] * df['Static_Weight']
        
        # Strategy 2 — Volatility-Adjusted Exposure
        df['Forecast_Vol'] = df['Forecast_Vol'].clip(lower=0.05, upper=0.60)
        raw_weight = target_vol / (df['Forecast_Vol'] + 1e-6)
        raw_weight = raw_weight.clip(0.1, 2.0)
        
        # Direct volatility targeting
        df['Dynamic_Weight'] = target_vol / df['Forecast_Vol']
        df['Dynamic_Weight'] = df['Dynamic_Weight'].clip(0.2, 2.0)
        df['Actual_Return'] = df['Actual_Return'].fillna(0)
        # Light smoothing (prevents noise)
        df['Dynamic_Weight'] = df['Dynamic_Weight'].ewm(alpha=0.2).mean()

        # Momentum filter (Sharpe boost — MUST come BEFORE return calc)
        momentum = df['Actual_Return'].rolling(5).mean()
        df['Dynamic_Weight'] *= (momentum > -0.002)

        # Avoid trading in extreme volatility spikes

        df['Dynamic_Return'] = (df['Dynamic_Weight'].shift(1) * df['Actual_Return']).fillna(0)
        
        # Strategy 3 — Regime-Based Exposure
        if df['Regime_Valid'].all() == False:
            df['Regime_Weight'] = np.nan
            df['Regime_Return'] = np.nan
        else:
            df['Regime_Weight'] = np.where(df['Regime_1_Prob'] > 0.7, 0.4, 1.0)
            df['Regime_Return'] = (df['Regime_Weight'].shift(1) * df['Actual_Return']).fillna(0)
        
        df = df.dropna(subset=['Actual_Return', 'Forecast_Vol', 'Dynamic_Return'])
        
        def compute_metrics(r):
            ann_ret = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
            mdd = (r.cumsum().cummax() - r.cumsum()).max()
            if mdd < 0: mdd = abs(mdd) # Absolute formatting standard
            return ann_ret, ann_vol, sharpe, mdd

        s_ret, s_vol, s_sharpe, s_mdd = compute_metrics(df['Static_Return'])
        d_ret, d_vol, d_sharpe, d_mdd = compute_metrics(df['Dynamic_Return'])
        
        if regime_status == "VALID":
            r_ret, r_vol, r_sharpe, r_mdd = compute_metrics(df['Regime_Return'].dropna())
        else:
            r_ret, r_vol, r_sharpe, r_mdd = np.nan, np.nan, np.nan, np.nan
        
        res = {
            'Strategy': ['Static Exposure', 'Volatility-Managed', 'Regime-Based'],
            'Ann Return': [s_ret, d_ret, r_ret],
            'Volatility': [s_vol, d_vol, r_vol],
            'Sharpe': [s_sharpe, d_sharpe, r_sharpe],
            'Max DD': [-s_mdd, -d_mdd, -r_mdd],
            'Avg Weight': [1.0, df['Dynamic_Weight'].mean(), df['Regime_Weight'].mean()],
            'Min Weight': [1.0, df['Dynamic_Weight'].min(), df['Regime_Weight'].min()],
            'Max Weight': [1.0, df['Dynamic_Weight'].max(), df['Regime_Weight'].max()],
            'Regime_Status': [regime_status] * 3
        }
        res_df = pd.DataFrame(res)
        res_df = res_df.drop_duplicates().reset_index(drop=True)
        validate_and_save(res_df, os.path.join(market_dir, "economic_performance.csv"), is_time_series=False, index=False)
        
        if regime_status == "VALID":
            # Part 4 STRESS TEST: Validate physical edge constraints cleanly limiting evaluation arrays onto pure crisis regimes
            high_vol_mask = df['Regime_1_Prob'] >= 0.5
            if high_vol_mask.sum() > 10:
                stress_df = df[high_vol_mask].copy()
                
                s_ret_st, _, s_sharpe_st, s_mdd_st = compute_metrics(stress_df['Static_Return'])
                d_ret_st, _, d_sharpe_st, d_mdd_st = compute_metrics(stress_df['Dynamic_Return'])
                r_ret_st, _, r_sharpe_st, r_mdd_st = compute_metrics(stress_df['Regime_Return'].dropna())
                
                stress_res = {
                    'Strategy': ['Static', 'Vol-managed', 'Regime-based'],
                    'Return_crisis': [s_ret_st, d_ret_st, r_ret_st],
                    'Sharpe_crisis': [s_sharpe_st, d_sharpe_st, r_sharpe_st],
                    'Drawdown_crisis': [s_mdd_st, d_mdd_st, r_mdd_st]
                }
                pd.DataFrame(stress_res).to_csv(os.path.join(market_dir, "economic_crisis_performance.csv"), index=False)
        
        # Outputs specific for time-series UI tracking limits
        ts_df = df[['Dynamic_Weight', 'Forecast_Vol', 'Dynamic_Return']].copy()
        ts_df.index.name = 'Date'
        ts_df.rename(columns={'Dynamic_Weight': 'Weight', 'Forecast_Vol': 'Forecast_Vol', 'Dynamic_Return': 'Return'}, inplace=True)
        validate_and_save(ts_df, os.path.join(market_dir, f"{market_name}_economic_timeseries.csv"), is_time_series=True)
        
        # PLOT: Drawdown
        fig, ax = plt.subplots(figsize=(10, 5))
        def get_dd(r):
            return (r.cumsum().cummax() - r.cumsum())
            
        ax.plot(df.index, get_dd(df['Static_Return']), label='Static Exposure', color='#aaaaaa', alpha=0.9)
        ax.plot(df.index, get_dd(df['Dynamic_Return']), label='Volatility-Managed', color='blue', alpha=0.7)
        ax.plot(df.index, get_dd(df['Regime_Return']), label='Regime-Based', color='#ffa500', alpha=0.9, linewidth=1.5)
        ax.set_title(f"{market_name}: Strategy Drawdown Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        import matplotlib.ticker as mticker
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        
        plt.tight_layout()
        plt.savefig(os.path.join(market_dir, f"{market_name}_drawdown.png"))
        plt.close()
        
        # PLOT: Cumulative Returns
        fig, ax = plt.subplots(figsize=(10, 5))
        # SECTION 5 — EQUITY CURVE STABILITY FIX
        df['Cum_Static'] = (1 + df['Static_Return'].fillna(0)).cumprod()
        df['Cum_Dynamic'] = (1 + df['Dynamic_Return'].fillna(0)).cumprod()
        df['Cum_Regime'] = (1 + df['Regime_Return'].fillna(0)).cumprod() if regime_status == "VALID" else np.nan
        
        assert not df['Cum_Static'].isna().all(), "Static equity curve invalid"
        assert not df['Cum_Dynamic'].isna().all(), "Dynamic equity curve invalid"

        ax.plot(df.index, df['Cum_Static'], label='Static Exposure', color='#aaaaaa', alpha=0.9)
        ax.plot(df.index, df['Cum_Dynamic'], label='Volatility-Managed', color='blue', alpha=0.7)
        if regime_status == "VALID":
            ax.plot(df.index, df['Cum_Regime'], label='Regime-Based', color='#ffa500', alpha=0.9, linewidth=1.5)
        ax.set_title(f"{market_name}: Cumulative Portfolio Growth")
        ax.set_xlabel("Date")
        ax.set_ylabel("Multiplier (1.0 = Base)")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(market_dir, f"{market_name}_cumulative_returns.png"))
        plt.close()
        
        # PLOT: Weight Over Time
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['Dynamic_Weight'], label='Vol-Managed Target Exposure', color='blue', alpha=0.5)
        ax.plot(df.index, df['Regime_Weight'], label='Regime-Based Exposure Constraints', color='#ffa500', alpha=0.8, drawstyle='steps-post')
        ax.axhline(1.0, color='#aaaaaa', linestyle='--', label='Static 1.0x baseline')
        ax.set_title(f"{market_name}: Portfolio Exposure Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight (%)")
        
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(market_dir, f"{market_name}_weight_over_time.png"))
        plt.close()
        
        import matplotlib.ticker as mticker
        
        # PLOT: Forecast vs Actual Return (Volatility)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        realized_vol = df['Actual_Return'].rolling(window=5).std() * np.sqrt(252)
        
        forecast_vol = df['Forecast_Vol']
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility (%)")
        ax.plot(df.index, realized_vol, label="Realized Volatility", color='#aaaaaa', alpha=0.5)
        ax.plot(df.index, forecast_vol, label="Forecast Volatility", color='#ffa500', alpha=0.9)
        
        try:
            regime_path = os.path.join(market_dir, f"{market_name}_regimes.csv")
            if os.path.exists(regime_path):
                reg_df = pd.read_csv(regime_path, index_col=0, parse_dates=True)
                assert isinstance(reg_df.index, pd.DatetimeIndex), "DatetimeIndex lost during load in economic_evaluation (regime plot)"
                aligned_reg = reg_df.reindex(df.index).ffill()
                if 'Regime_1_Prob' in aligned_reg.columns:
                    high_vol_series = aligned_reg['Regime_1_Prob'] > 0.7
                    ax.fill_between(df.index, 0, forecast_vol.max(), where=high_vol_series, color='red', alpha=0.1, label="High Vol Regime")
        except Exception:
            pass
            
        try:
            ts_start, ts_end = df.index.min(), df.index.max()
            t1 = pd.Timestamp("2015-06-01")
            if ts_start < t1 < ts_end:
                ax.axvline(t1, linestyle='--', color='red', alpha=0.5, label="2015 Bubble Crash")
            t2 = pd.Timestamp("2020-03-01")
            if ts_start < t2 < ts_end:
                ax.axvline(t2, linestyle='--', color='yellow', alpha=0.5, label="Covid-19 Shock")
        except Exception:
            pass
            
        ax.set_title(f"{market_name}: Volatility Forecast vs Realized Trajectory")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        try:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        except Exception:
            pass
        plt.tight_layout()
        plt.savefig(os.path.join(market_dir, f"{market_name}_forecast_vs_actual.png"))
        plt.close()

        return res_df
