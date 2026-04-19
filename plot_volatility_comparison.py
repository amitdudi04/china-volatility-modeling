import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_market(market_name, file_path, output_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Plot settings for academic style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Check if necessary columns exist
    if 'Realized_Vol' in df.columns and 'Forecast_Vol' in df.columns:
        ax.plot(df.index, df['Realized_Vol'], label='Realized Volatility', color='black', alpha=0.6, linewidth=1)
        ax.plot(df.index, df['Forecast_Vol'], label='Forecast Volatility', color='darkred', alpha=0.9, linewidth=1.2)
        
        ax.set_title(f"Forecast vs Realized Volatility — {market_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Volatility")
        ax.legend(loc='upper left', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Saved: {output_path}")
    else:
        print(f"Missing columns in {file_path}")
    
    plt.close()

if __name__ == "__main__":
    os.makedirs('outputs/figures', exist_ok=True)
    
    csi_path = r'outputs/results/CSI_300/CSI_300_hybrid_forecast.csv'
    chinext_path = r'outputs/results/ChiNext/ChiNext_hybrid_forecast.csv'
    
    plot_market('CSI 300', csi_path, 'outputs/figures/csi_volatility.png')
    plot_market('ChiNext', chinext_path, 'outputs/figures/chinext_volatility.png')
