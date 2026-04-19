import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QSplitter)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

from src.gui.data_loader import GUIDataLoader
from src.gui.widgets.kpi_cards import KPIStrip
from src.gui.widgets.charts import ChartWidget
from src.gui.widgets.tables import DataTable
from src.gui.widgets.insight_panel import InsightPanel

class QuantResearchDashboard(QMainWindow):
    def __init__(self, base_outputs_dir="outputs/results"):
        super().__init__()
        self.data_loader = GUIDataLoader(base_outputs_dir)
        
        self.setWindowTitle("Bloomberg Terminal - Quant Diagnostics")
        self.resize(1600, 1000)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; color: #ffffff; }
            QWidget { background-color: #121212; color: #ffffff; }
            QTabWidget::pane { border: 1px solid #333; }
            QTabBar::tab { background: #1e1e1e; color: #888; padding: 12px 24px; border: 1px solid #333; border-bottom: none; font-family: Consolas; font-size: 14px; font-weight: bold; }
            QTabBar::tab:selected { background: #2c2c2c; color: #ffffff; border-top: 3px solid #ff9900; }
            QSplitter::handle { background-color: #333; height: 3px; width: 3px; margin: 5px; }
        """)
        
        self.markets = ["CSI_300", "SSE_Composite", "ChiNext"]
        self.initUI()
        
    def initUI(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Header
        header_layout = QHBoxLayout()
        lbl_title = QLabel("QUANTITATIVE RESEARCH TERMINAL")
        lbl_title.setStyleSheet("font-size: 24pt; font-family: Segoe UI; font-weight: bold; color: #ff9900; background-color: transparent;")
        header_layout.addWidget(lbl_title)
        
        self.market_combo = QComboBox()
        self.market_combo.setFont(QFont("Consolas", 14))
        self.market_combo.setStyleSheet("background-color: #2c2c2c; color: white; border: 1px solid #555; padding: 5px; min-width: 200px;")
        self.market_combo.addItems(self.markets)
        self.market_combo.currentTextChanged.connect(self.update_dashboard)
        
        combo_layout = QHBoxLayout()
        combo_lbl = QLabel("MARKET:")
        combo_lbl.setStyleSheet("font-family: Consolas; font-weight: bold; font-size: 16px; background-color: transparent;")
        combo_layout.addWidget(combo_lbl)
        combo_layout.addWidget(self.market_combo)
        combo_layout.addStretch()
        header_layout.addLayout(combo_layout)
        
        main_layout.addLayout(header_layout)
        
        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        self.tab_overview = QWidget(); self.tab_overview.setLayout(QVBoxLayout(self.tab_overview))
        self.tab_model = QWidget(); self.tab_model.setLayout(QVBoxLayout(self.tab_model))
        self.tab_risk = QWidget(); self.tab_risk.setLayout(QVBoxLayout(self.tab_risk))
        self.tab_economic = QWidget(); self.tab_economic.setLayout(QVBoxLayout(self.tab_economic))
        self.tab_system = QWidget(); self.tab_system.setLayout(QVBoxLayout(self.tab_system))
        
        self.tabs.addTab(self.tab_overview, "Overview")
        self.tabs.addTab(self.tab_model, "Model Diagnostics")
        self.tabs.addTab(self.tab_risk, "Risk Metrics")
        self.tabs.addTab(self.tab_economic, "Economic Performance")
        self.tabs.addTab(self.tab_system, "System Diagnostics")
        
        if self.markets:
            self.update_dashboard(self.markets[0])

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout():
                    self.clear_layout(child.layout())

    def update_dashboard(self, market):
        for t in [self.tab_overview, self.tab_model, self.tab_risk, self.tab_economic, self.tab_system]:
            self.clear_layout(t.layout())
        
        data = self.data_loader.load_market_data(market)
        
        self.build_overview_tab(market, data)
        self.build_model_tab(market, data)
        self.build_risk_tab(market, data)
        self.build_economic_tab(market, data)
        self.build_system_tab(market, data)

    def get_v_splitter(self):
        splitter = QSplitter(Qt.Orientation.Vertical)
        return splitter

    def get_h_splitter(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        return splitter

    # ---------------- TAB 1: OVERVIEW ----------------
    def build_overview_tab(self, market, data):
        layout = self.tab_overview.layout()
        
        w_df = data.get("weights", pd.DataFrame())
        var_df = data.get("var_res", pd.DataFrame())
        econ_df = data.get("econ", pd.DataFrame())
        hybrid_df = data.get("hybrid", pd.DataFrame())
        
        best_model = "UNKNOWN"
        if not w_df.empty:
            best_model = "GARCH" if w_df['RMSE_GARCH'].iloc[0] < w_df['RMSE_ML'].iloc[0] else "ML"
            
        mean_vol = hybrid_df['Forecast_Vol'].mean() if not hybrid_df.empty and 'Forecast_Vol' in hybrid_df.columns else np.nan
        s_shp = econ_df['Sharpe'].values[-1] if not econ_df.empty and 'Sharpe' in econ_df.columns else np.nan
        s_mdd = econ_df['Max DD'].values[-1] if not econ_df.empty and 'Max DD' in econ_df.columns else np.nan
        kup_p = var_df['kupiec_pvalue'].values[0] if not var_df.empty and 'kupiec_pvalue' in var_df.columns else np.nan
        
        corr = np.nan
        if not hybrid_df.empty and 'Forecast_Vol' in hybrid_df.columns and 'Realized_Vol' in hybrid_df.columns:
            corr = np.corrcoef(hybrid_df['Forecast_Vol'], hybrid_df['Realized_Vol'])[0, 1]

        # Top: KPIs
        kpi_strip = KPIStrip()
        kpi_strip.add_kpi("Sharpe Ratio", f"{s_shp:.2f}", s_shp > 0 if not pd.isna(s_shp) else None)
        kpi_strip.add_kpi("Max Drawdown", f"{s_mdd:.2f}%", s_mdd > -15 if not pd.isna(s_mdd) else None)
        kpi_strip.add_kpi("Mean Volatility", f"{mean_vol:.4f}", mean_vol < 0.3 if not pd.isna(mean_vol) else None)
        var_acc = kup_p > 0.05 if not pd.isna(kup_p) else False
        kpi_strip.add_kpi("VaR Check", "PASS" if var_acc else "FAIL", var_acc)
        kpi_strip.add_kpi("Correlation", f"{corr:.4f}", corr > 0.5 if not pd.isna(corr) else None)
        kpi_strip.add_stretch()
        layout.addWidget(kpi_strip)

        # Middle & Bottom Splitter
        v_split = self.get_v_splitter()
        
        # Middle: Chart
        chart = ChartWidget("Forecast vs Realized Volatility")
        chart.plot_forecast_vs_realized(hybrid_df)
        v_split.addWidget(chart)
        
        # Bottom: Table + Insight
        h_split = self.get_h_splitter()
        summary_data = pd.DataFrame({
            "Metric": ["Best Model", "Correlation", "Mean Volatility", "Sharpe Ratio", "Max Drawdown", "Kupiec p-value"],
            "Value": [str(best_model), f"{corr:.4f}", f"{mean_vol:.4f}", f"{s_shp:.2f}", f"{s_mdd:.2f}%", f"{kup_p:.4f}"]
        })
        table = DataTable("EXECUTIVE SUMMARY", summary_data)
        h_split.addWidget(table)
        
        insight = InsightPanel()
        insight.generate_overview_insight(market, corr, kup_p, best_model)
        h_split.addWidget(insight)
        
        h_split.setSizes([400, 600])
        v_split.addWidget(h_split)
        v_split.setSizes([600, 300])
        
        layout.addWidget(v_split)

    # ---------------- TAB 2: MODEL DIAGNOSTICS ----------------
    def build_model_tab(self, market, data):
        layout = self.tab_model.layout()
        
        w_df = data.get("weights", pd.DataFrame())
        feat_df = data.get("feat", pd.DataFrame())
        
        g_rmse = w_df['RMSE_GARCH'].iloc[0] if not w_df.empty else np.nan
        m_rmse = w_df['RMSE_ML'].iloc[0] if not w_df.empty else np.nan
        
        # Top: KPIs
        kpi_strip = KPIStrip()
        kpi_strip.add_kpi("GARCH RMSE", f"{g_rmse:.4f}")
        kpi_strip.add_kpi("ML RMSE", f"{m_rmse:.4f}")
        kpi_strip.add_kpi("Dominant Model", "GARCH" if g_rmse < m_rmse else "ML", True)
        kpi_strip.add_stretch()
        layout.addWidget(kpi_strip)

        v_split = self.get_v_splitter()
        
        # Middle: Chart
        chart = ChartWidget("Random Forest Feature Importance (Top 10)")
        chart.plot_feature_importance(feat_df)
        v_split.addWidget(chart)
        
        # Bottom: Table + Insight
        h_split = self.get_h_splitter()
        table = DataTable("FEATURE IMPORTANCE STABILITY", feat_df.head(15) if not feat_df.empty else pd.DataFrame())
        h_split.addWidget(table)
        
        insight = InsightPanel()
        insight.generate_model_insight(market, w_df, feat_df)
        h_split.addWidget(insight)
        
        h_split.setSizes([500, 500])
        v_split.addWidget(h_split)
        v_split.setSizes([600, 300])
        
        layout.addWidget(v_split)

    # ---------------- TAB 3: RISK METRICS ----------------
    def build_risk_tab(self, market, data):
        layout = self.tab_risk.layout()
        
        var_res = data.get("var_res", pd.DataFrame())
        var_comp = data.get("var_comp", pd.DataFrame())
        raw_df = data.get("raw", pd.DataFrame())
        risk_df = data.get("risk_for", pd.DataFrame())
        
        kup_p = var_res['kupiec_pvalue'].values[0] if not var_res.empty and 'kupiec_pvalue' in var_res.columns else np.nan
        chris_p = var_res['christoffersen_pvalue'].values[0] if not var_res.empty and 'christoffersen_pvalue' in var_res.columns else np.nan
        
        kpi_strip = KPIStrip()
        kpi_strip.add_kpi("Kupiec Test", "PASS" if kup_p > 0.05 else "FAIL", kup_p > 0.05 if not pd.isna(kup_p) else None)
        kpi_strip.add_kpi("Christoffersen Test", "PASS" if chris_p > 0.05 else "FAIL", chris_p > 0.05 if not pd.isna(chris_p) else None)
        kpi_strip.add_kpi("Expected Violations", f"{var_res['Expected_Violations'].values[0]:.1f}" if not var_res.empty and 'Expected_Violations' in var_res.columns else "N/A")
        kpi_strip.add_kpi("Actual Violations", f"{var_res['Actual_Violations'].values[0]}" if not var_res.empty and 'Actual_Violations' in var_res.columns else "N/A")
        kpi_strip.add_stretch()
        layout.addWidget(kpi_strip)

        v_split = self.get_v_splitter()
        
        chart = ChartWidget("Value-at-Risk (VaR) vs Sub-zero Returns")
        chart.plot_var_breaches(raw_df, risk_df)
        v_split.addWidget(chart)
        
        h_split = self.get_h_splitter()
        table = DataTable("VAR MODEL COMPARISON", var_comp)
        h_split.addWidget(table)
        
        insight = InsightPanel()
        insight.generate_risk_insight(market, var_res)
        h_split.addWidget(insight)
        
        h_split.setSizes([500, 500])
        v_split.addWidget(h_split)
        v_split.setSizes([600, 300])
        
        layout.addWidget(v_split)

    # ---------------- TAB 4: ECONOMIC PERFORMANCE ----------------
    def build_economic_tab(self, market, data):
        layout = self.tab_economic.layout()
        
        econ_df = data.get("econ", pd.DataFrame())
        ts_df = data.get("econ_ts", pd.DataFrame())
        
        s_shp = econ_df['Sharpe'].values[-1] if not econ_df.empty and 'Sharpe' in econ_df.columns else np.nan
        s_mdd = econ_df['Max DD'].values[-1] if not econ_df.empty and 'Max DD' in econ_df.columns else np.nan
        s_ret = econ_df['Ann. Return'].values[-1] if not econ_df.empty and 'Ann. Return' in econ_df.columns else np.nan
        
        kpi_strip = KPIStrip()
        kpi_strip.add_kpi("Strategy Sharpe", f"{s_shp:.2f}", s_shp > 0 if not pd.isna(s_shp) else None)
        kpi_strip.add_kpi("Strategy Drawdown", f"{s_mdd:.2f}%", s_mdd > -15 if not pd.isna(s_mdd) else None)
        kpi_strip.add_kpi("Annualized Return", f"{s_ret:.2f}%", s_ret > 0 if not pd.isna(s_ret) else None)
        kpi_strip.add_stretch()
        layout.addWidget(kpi_strip)

        v_split = self.get_v_splitter()
        
        chart = ChartWidget("Out-Of-Sample Cumulative Equity Curve")
        chart.plot_equity_curve(ts_df)
        v_split.addWidget(chart)
        
        h_split = self.get_h_splitter()
        table = DataTable("STRATEGY PERFORMANCE COMPARISON", econ_df)
        h_split.addWidget(table)
        
        insight = InsightPanel()
        insight.generate_economic_insight(market, econ_df)
        h_split.addWidget(insight)
        
        h_split.setSizes([500, 500])
        v_split.addWidget(h_split)
        v_split.setSizes([600, 300])
        
        layout.addWidget(v_split)

    # ---------------- TAB 5: SYSTEM DIAGNOSTICS ----------------
    def build_system_tab(self, market, data):
        layout = self.tab_system.layout()
        
        sys_df = data.get("sys", pd.DataFrame())
        stat_df = data.get("stat", pd.DataFrame())
        var_res = data.get("var_res", pd.DataFrame())
        econ_df = data.get("econ", pd.DataFrame())
        
        kpi_strip = KPIStrip()
        
        # Panel 1: Statistical Tests
        ljung_p = stat_df['LjungBox_p'].values[0] if not stat_df.empty and 'LjungBox_p' in stat_df.columns else np.nan
        arch_p = stat_df['ARCH_p'].values[0] if not stat_df.empty and 'ARCH_p' in stat_df.columns else np.nan
        
        kpi_strip.add_kpi("Ljung-Box Stat", "PASS" if ljung_p > 0.05 else "FAIL", True if ljung_p > 0.05 else False)
        kpi_strip.add_kpi("ARCH LM Stat", "PASS" if arch_p > 0.05 else "FAIL", True if arch_p > 0.05 else "warning")
        
        # Panel 2: Risk Validation
        kup_p = var_res['kupiec_pvalue'].values[0] if not var_res.empty and 'kupiec_pvalue' in var_res.columns else np.nan
        chris_p = var_res['christoffersen_pvalue'].values[0] if not var_res.empty and 'christoffersen_pvalue' in var_res.columns else np.nan
        
        kpi_strip.add_kpi("Kupiec VaR", "PASS" if kup_p > 0.05 else "FAIL", True if kup_p > 0.05 else False)
        kpi_strip.add_kpi("Christoffersen", "PASS" if chris_p > 0.05 else "FAIL", True if chris_p > 0.05 else False)
        
        # Panel 3: Economic Performance
        s_shp = econ_df['Sharpe'].values[-1] if not econ_df.empty and 'Sharpe' in econ_df.columns else np.nan
        s_mdd = econ_df['Max DD'].values[-1] if not econ_df.empty and 'Max DD' in econ_df.columns else np.nan
        
        kpi_strip.add_kpi("Strategy Sharpe", "GOOD" if s_shp > 0 else "FAIL", True if s_shp > 0 else False)
        kpi_strip.add_kpi("Max Drawdown", "ACCEPTABLE" if s_mdd > -20 else "FAIL", True if s_mdd > -20 else False)
        
        kpi_strip.add_stretch()
        layout.addWidget(kpi_strip)

        v_split = self.get_v_splitter()
        
        # In the absence of a chart, use a combined top layout or empty chart box
        # System doesn't need a massive chart, so we'll present tables.
        
        top_h_split = self.get_h_splitter()
        table_sys = DataTable("VALIDATION REPORT", sys_df)
        top_h_split.addWidget(table_sys)
        
        table_stat = DataTable("STATISTICAL RESIDUAL TESTS", stat_df)
        top_h_split.addWidget(table_stat)
        
        v_split.addWidget(top_h_split)
        
        insight = InsightPanel("SYSTEM AUDIT AND DIAGNOSTICS")
        insight.generate_system_insight(market, sys_df, stat_df)
        v_split.addWidget(insight)
        v_split.setSizes([600, 300])
        
        layout.addWidget(v_split)
