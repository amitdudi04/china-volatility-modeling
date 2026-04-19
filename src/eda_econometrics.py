from src.utils.validation import validate_and_save
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from scipy.stats import jarque_bera
from src.utils import get_market_path

class EDAEconometrics:
    def plot_price_trends(self, df, market_name):
        market_dir = get_market_path(market_name)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['Close'], color='blue', linewidth=1)
        ax.set_title(f"{market_name}: Structural Price Trends under Policy Regimes")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(market_dir, f"{market_name}_price_trend.png"))
        plt.close()

    def plot_return_distribution(self, df, market_name):
        import matplotlib.ticker as mticker
        market_dir = get_market_path(market_name)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['Log_Return'], bins=100, color='red', alpha=0.7, density=True)
        ax.set_title(f"{market_name}: Non-Normal Return Distribution (Fat Tails)")
        ax.set_xlabel("Log Return")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(market_dir, f"{market_name}_return_dist.png"))
        plt.close()

    def plot_volatility_clustering(self, df, market_name):
        market_dir = get_market_path(market_name)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['Log_Return'], color='black', linewidth=0.5)
        ax.set_title(f"{market_name}: Volatility Clustering in Chinese Mkts")
        ax.set_xlabel("Date")
        ax.set_ylabel("Log Return")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(market_dir, f"{market_name}_vol_clustering.png"))
        plt.close()

    def run_tests(self, df, market_name):
        returns = df['Log_Return']
        results = {}
        
        # 1. ADF Test (Stationarity)
        adf_stat, adf_pval, *_ = adfuller(returns.dropna())
        results['ADF_Stat'] = adf_stat
        results['ADF_pvalue'] = adf_pval
        
        # 2. Jarque-Bera Test (Fat tails / Non-normality)
        jb_stat, jb_pval = jarque_bera(returns.dropna())
        results['JB_Stat'] = jb_stat
        results['JB_pvalue'] = jb_pval
        
        # 3. ARCH-LM Test (Volatility clustering)
        # Using 5 lags standard for high freq/daily
        arch_test, arch_pval, _, _ = het_arch(returns.dropna(), nlags=5)
        results['ARCH_LM_Stat'] = arch_test
        results['ARCH_LM_pvalue'] = arch_pval

        res_df = pd.DataFrame([results], index=[market_name])
        validate_and_save(res_df, os.path.join(get_market_path(market_name), f"{market_name}_econometric_tests.csv"), is_time_series=False)
        return res_df

    def analyze(self, df, market_name):
        print(f"Running EDA and Econometric Tests for {market_name}...")
        self.plot_price_trends(df, market_name)
        self.plot_return_distribution(df, market_name)
        self.plot_volatility_clustering(df, market_name)
        return self.run_tests(df, market_name)
