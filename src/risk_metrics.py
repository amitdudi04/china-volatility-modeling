from src.utils.validation import validate_and_save
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import t
from src.utils import get_market_path
import os

class RiskMetrics:
    def __init__(self):
        pass

    def calculate_var_es(self, forecasts_df, market_name):
        print(f"Calculating Risk Metrics for {market_name}...")
        
        df = forecasts_df.copy().dropna()
        
        # Daily conversion for VaR computation correctly mapping against daily raw percentages
        forecast_vol = df['Forecast_Vol'] / np.sqrt(252)
        forecast_vol = forecast_vol.clip(lower=1e-4)
        
        # Compute Normal VaR
        df['VaR_normal'] = -abs(stats.norm.ppf(0.05)) * forecast_vol 
        df['VaR_student_t'] = -abs(stats.t.ppf(0.05, df=6)) * forecast_vol
        
        market_dir = get_market_path(market_name)
        validate_and_save(df, os.path.join(market_dir, "risk_forecasts.csv"), is_time_series=True)
        
        # Add Diagnostic Output
        diag_df = df[['Forecast_Vol', 'Realized_Vol']].copy()
        
        # Scaling to Annualized % for the dashboard (already daily % from forecast_engine)
        # Note: Forecast_Vol and Realized_Vol are already Annualized %
        
        mean_vol = df['Forecast_Vol'].mean()
        print(f"[DEBUG] Mean Vol: {mean_vol}")
        assert 0.05 < mean_vol < 0.60, f"Invalid volatility: {mean_vol}"
        
        mean_realized = df['Realized_Vol'].mean()
        assert 0.05 < mean_realized < 0.60, f"Invalid realized volatility: {mean_realized}"
        
        diag_df.index.name = 'Date'
        validate_and_save(diag_df, os.path.join(market_dir, "volatility_comparison.csv"), is_time_series=False)
        
        return df

    def christoffersen_test(self, violations):
        v = violations.astype(int).values
        n00, n01, n10, n11 = 0, 0, 0, 0
        for i in range(1, len(v)):
            if v[i-1] == 0 and v[i] == 0: n00 += 1
            elif v[i-1] == 0 and v[i] == 1: n01 += 1
            elif v[i-1] == 1 and v[i] == 0: n10 += 1
            elif v[i-1] == 1 and v[i] == 1: n11 += 1
        pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0
        def safe_log(x): return np.log(x) if x > 0 else -1e10
        ll_null = (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
        ll_alt = n00 * safe_log(1 - pi_01) + n01 * safe_log(pi_01) + n10 * safe_log(1 - pi_11) + n11 * safe_log(pi_11)
        lr_ind = -2 * (ll_null - ll_alt)
        p_val_ind = 1 - stats.chi2.cdf(lr_ind, df=1)
        return lr_ind, p_val_ind

    def dq_test(self, violations):
        v = violations.astype(int).values
        if v.sum() == 0 or v.sum() == len(v): return np.nan
        import statsmodels.api as sm
        y = v[1:]
        X = sm.add_constant(v[:-1])
        try:
            model = sm.Logit(y, X)
            res = model.fit(disp=0)
            if len(res.pvalues) > 1:
                return res.pvalues[1]
            return np.nan
        except:
            return np.nan

    def backtest(self, returns, risk_df, market_name):
        aligned_df = risk_df.copy()
        aligned_df['Actual_Return'] = returns.loc[aligned_df.index]
        
        # Compute Empirical VaR (rolling window on actual returns)
        aligned_df['VaR_empirical'] = returns.rolling(250).quantile(0.05).loc[aligned_df.index]
        aligned_df['VaR_empirical'] = aligned_df['VaR_empirical'].bfill()

        expected_count_95 = int(0.05 * len(aligned_df))
        
        def get_kupiec(vc):
            if vc == 0:
                pf = 1e-6
            else:
                pf = vc / len(aligned_df)

            pf = min(max(pf, 1e-6), 1 - 1e-6)
            lr = -2 * (
                (vc * np.log(0.05) + (len(aligned_df) - vc) * np.log(0.95)) - 
                (vc * np.log(pf) + (len(aligned_df) - vc) * np.log(1 - pf))
            )
            return 1 - stats.chi2.cdf(abs(lr), df=1), lr

        models = ['VaR_normal', 'VaR_student_t', 'VaR_empirical']
        comp_results = []

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for m_name in models:
                viol = aligned_df['Actual_Return'] < aligned_df[m_name]
                v_count = sum(viol)
                k_p, k_lr = get_kupiec(v_count)
                c_lr, c_p = self.christoffersen_test(viol)
                dq_p = self.dq_test(viol)
                
                comp_results.append({
                    'Model': m_name,
                    'Actual_Violations': v_count,
                    'Kupiec_pvalue': k_p,
                    'Christoffersen_pvalue': c_p,
                    'DQ_pvalue': dq_p,
                    'Distance_to_0.05': abs(k_p - 0.05) if not np.isnan(k_p) else 999
                })

        comp_df = pd.DataFrame(comp_results)
        
        market_dir = get_market_path(market_name)
        comp_df = comp_df.drop_duplicates().reset_index(drop=True)
        validate_and_save(comp_df, os.path.join(market_dir, "var_comparison.csv"), is_time_series=False, index=False)
        
        # Select best model: Kupiec p-val > 0.05 and Christoffersen > 0.05. Pick one with Kupiec closest to 0.05-0.5
        valid = comp_df[(comp_df['Kupiec_pvalue'] > 0.05) & (comp_df['Christoffersen_pvalue'] > 0.05)]
        if not valid.empty:
            best_model_name = valid.sort_values('Distance_to_0.05').iloc[0]['Model']
        else:
            best_model_name = comp_df.sort_values('Kupiec_pvalue', ascending=False).iloc[0]['Model']

        best_stats = comp_df[comp_df['Model'] == best_model_name].iloc[0]
        
        deviation = best_stats['Actual_Violations'] - expected_count_95
        if deviation < -2:
            note = "Conservative VaR (overestimates risk)"
        elif deviation > 2:
            note = "VaR underestimates tail risk (excess violations)"
        else:
            note = "VaR statistically acceptable"
            
        k_p = best_stats['Kupiec_pvalue']
        c_p = best_stats['Christoffersen_pvalue']
        
        if k_p < 0.01 or c_p < 0.01:
            model_pass = "FAIL"
        elif k_p < 0.05 or c_p < 0.05:
            model_pass = "WARNING"
        else:
            model_pass = "PASS"
            
        results = {
            'Market': market_name,
            'Best_VaR_Model': best_model_name,
            'N_Observations': len(aligned_df),
            'VaR_Confidence': '95%',
            'Expected_Violations': expected_count_95,
            'Actual_Violations': best_stats['Actual_Violations'],
            'Deviation': deviation,
            'Interpretation': note,
            'kupiec_pvalue': best_stats['Kupiec_pvalue'],
            'christoffersen_pvalue': best_stats['Christoffersen_pvalue'],
            'dq_pvalue': best_stats['DQ_pvalue'],
            'model_pass': model_pass
        }
        res_df = pd.DataFrame([results])
        
        res_df = res_df.drop_duplicates().reset_index(drop=True)
        validate_and_save(res_df, os.path.join(market_dir, "var_results.csv"), is_time_series=False, index=False)
        
        print(f"Risk Backtest Complete. Best Model: {best_model_name} | Target Kupiec p-val: {best_stats['Kupiec_pvalue']:.4f}")
        return res_df
