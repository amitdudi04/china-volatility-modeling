import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pandas as pd
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=10, height=4, dpi=120):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#121212')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#1e1e1e')
        self.axes.tick_params(colors='white', labelsize=10)
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')
        for spine in self.axes.spines.values():
            spine.set_color('#333333')
        super(MplCanvas, self).__init__(self.fig)

class ChartWidget(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border-radius: 8px;
                border: 1px solid #2a2a2a;
            }
        """)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        
        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet("color: #e6e6e6; font-size: 16px; font-weight: bold; font-family: Segoe UI; background: transparent; border: none;")
        self.layout.addWidget(self.title_lbl)
        
        self.canvas = MplCanvas(self)
        self.layout.addWidget(self.canvas)
        self.ax = self.canvas.axes
    
    def _format_ax(self):
        if len(self.ax.lines) > 0 or len(self.ax.patches) > 0:
            self.ax.legend(facecolor='#1e1e1e', edgecolor='#333', labelcolor='white')
        self.ax.grid(color='#333333', linestyle='--', alpha=0.5)
        self.canvas.draw()

    def plot_forecast_vs_realized(self, df):
        self.ax.clear()
        if not df.empty and 'Forecast_Vol' in df.columns and 'Realized_Vol' in df.columns:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.set_index(df.columns[0], inplace=True)
                df.index = pd.to_datetime(df.index)
            self.ax.plot(df.index, df['Realized_Vol'], label='Realized Vol', color='#555555', alpha=0.7, lw=1.5)
            self.ax.plot(df.index, df['Forecast_Vol'], label='Forecast Vol', color='#ff9900', lw=2.0)
        else:
            self.ax.text(0.5, 0.5, "Data Unavailable", color='white', ha='center', va='center')
        self._format_ax()

    def plot_feature_importance(self, df):
        self.ax.clear()
        if not df.empty and 'Feature' in df.columns and 'Importance_Mean' in df.columns:
            f_names = df['Feature'][:10][::-1]
            f_vals = df['Importance_Mean'][:10][::-1]
            self.ax.barh(f_names, f_vals, color='#4285f4')
            self.ax.set_xlabel("Mean Importance")
        else:
            self.ax.text(0.5, 0.5, "Feature Data Unavailable", color='white', ha='center', va='center')
        self._format_ax()

    def plot_var_breaches(self, raw_df, risk_df):
        self.ax.clear()
        if not risk_df.empty and not raw_df.empty:
            if not isinstance(risk_df.index, pd.DatetimeIndex):
                risk_df.set_index(risk_df.columns[0], inplace=True)
            if not isinstance(raw_df.index, pd.DatetimeIndex):
                raw_df.set_index(raw_df.columns[0], inplace=True)
                
            aligned = raw_df.loc[raw_df.index.intersection(risk_df.index)]
            aligned_risk = risk_df.loc[aligned.index]
            
            self.ax.scatter(aligned.index, aligned['Log_Return'], color='#aaaaaa', s=4, alpha=0.6, label='Actual Returns')
            if 'VaR_student_t' in aligned_risk.columns:
                self.ax.plot(aligned_risk.index, -aligned_risk['VaR_student_t'], color='#ea4335', lw=2.0, label='VaR Limit (Student-t)')
                
            import matplotlib.ticker as ticker
            self.ax.xaxis.set_major_locator(ticker.MaxNLocator(8))
        else:
            self.ax.text(0.5, 0.5, "VaR / Return Data Unavailable", color='white', ha='center', va='center')
        self._format_ax()

    def plot_equity_curve(self, ts_df):
        self.ax.clear()
        if not ts_df.empty and 'Date' in ts_df.columns:
            ts_df.set_index('Date', inplace=True)
            ts_df.index = pd.to_datetime(ts_df.index)
            
            if 'Cum_Static' in ts_df.columns: 
                self.ax.plot(ts_df.index, ts_df['Cum_Static'], color='#555555', lw=1.5, label='Baseline')
            if 'Cum_Vol' in ts_df.columns: 
                self.ax.plot(ts_df.index, ts_df['Cum_Vol'], color='#4285f4', lw=2.0, label='Vol-Managed')
            if 'Cum_Regime' in ts_df.columns: 
                self.ax.plot(ts_df.index, ts_df['Cum_Regime'], color='#34a853', lw=2.5, label='Regime-Managed')
        else:
            self.ax.text(0.5, 0.5, "Economic Timeseries Data Unavailable", color='white', ha='center', va='center')
        self._format_ax()
