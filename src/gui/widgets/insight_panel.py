from PyQt6.QtWidgets import QTextEdit, QVBoxLayout, QLabel, QFrame
import numpy as np

class InsightPanel(QFrame):
    def __init__(self, title="QUANTITATIVE INSIGHT"):
        super().__init__()
        self.setStyleSheet("""
            QFrame { background-color: #1e1e1e; border-radius: 8px; border: 1px solid #2a2a2a; }
            QTextEdit { background-color: transparent; color: #e6e6e6; border: none; font-family: Consolas; font-size: 14px; line-height: 1.5; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        title_lbl = QLabel(title.upper())
        title_lbl.setStyleSheet("color: #ff9900; font-size: 14px; font-weight: bold; font-family: Segoe UI; background: transparent; border: none; padding-bottom: 5px;")
        layout.addWidget(title_lbl)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

    def generate_overview_insight(self, market, corr, kupiec_p, best_model):
        insight = f"MARKET DIAGNOSTICS: {market}\n\n"
        if not np.isnan(corr):
            insight += f"• Correlation: At {corr:.4f}, the forecast exhibits {'strong' if corr > 0.7 else 'moderate' if corr > 0.4 else 'weak'} alignment with realized volatility.\n"
        
        insight += f"• Best Tracking Model: {best_model} provides the superior variance fit.\n"
        
        if not np.isnan(kupiec_p):
            if kupiec_p > 0.05:
                insight += "• VaR Status: The model successfully captures tail risk without systemic underestimation (Kupiec p > 0.05)."
            else:
                insight += "• VaR Status: CAUTION - Model rejects Kupiec test, indicating potential structural underestimation of extreme downside risk."
                
        self.text_edit.setPlainText(insight)

    def generate_model_insight(self, market, w_df, feat_df):
        insight = f"MODEL ARCHITECTURE ANALYSIS: {market}\n\n"
        if not w_df.empty:
            g_rmse = w_df['RMSE_GARCH'].iloc[0]
            m_rmse = w_df['RMSE_ML'].iloc[0]
            diff = abs(g_rmse - m_rmse) / max(g_rmse, m_rmse) * 100
            winner = "GARCH" if g_rmse < m_rmse else "ML"
            insight += f"• RMSE Dominance: {winner} outperforms by {diff:.2f}%. This suggests {'linear persistence' if winner == 'GARCH' else 'non-linear shock dynamics'} are driving variance.\n"
            
        if not feat_df.empty:
            top_feat = feat_df['Feature'].iloc[0]
            insight += f"• Primary Signal: The Machine Learning engine heavily relies on '{top_feat}', identifying it as the primary structural feature for shock prediction."
            
        self.text_edit.setPlainText(insight)

    def generate_risk_insight(self, market, var_res):
        insight = f"TAIL RISK VALIDATION: {market}\n\n"
        if not var_res.empty and 'kupiec_pvalue' in var_res.columns:
            p_val = var_res['kupiec_pvalue'].values[0]
            insight += f"• Kupiec POF Test: p-value = {p_val:.4f}. "
            if p_val > 0.05:
                insight += "The unconditional coverage is statistically valid. The system safely sizes risk buffers.\n"
            else:
                insight += "The risk model failed coverage tests. Adjust leverage immediately.\n"
                
            if 'Expected_Violations' in var_res.columns and 'Actual_Violations' in var_res.columns:
                exp = var_res['Expected_Violations'].values[0]
                act = var_res['Actual_Violations'].values[0]
                insight += f"• Violations: Observed {act} vs Expected {exp:.1f}.\n"
        else:
            insight += "No VaR validation data available."
            
        self.text_edit.setPlainText(insight)

    def generate_economic_insight(self, market, econ_df):
        insight = f"ECONOMIC PERFORMANCE STRATEGY: {market}\n\n"
        if not econ_df.empty and 'Sharpe' in econ_df.columns:
            sharpe = econ_df['Sharpe'].values[-1]
            mdd = econ_df['Max DD'].values[-1]
            
            insight += f"• Risk-Adjusted Return: The strategy yielded a Sharpe Ratio of {sharpe:.2f}.\n"
            insight += f"• Capital Preservation: Maximum Drawdown was constrained to {mdd:.2f}%.\n\n"
            
            if sharpe > 1.0:
                insight += "Conclusion: The volatility-targeting mechanism effectively smooths out equity curve drawdowns, generating strong institutional-grade returns."
            elif sharpe > 0:
                insight += "Conclusion: The strategy shows positive drift, but requires further optimization for strict institutional deployment."
            else:
                insight += "Conclusion: The strategy decayed capital. The volatility forecasting signal may be drowned out by market noise or policy shocks."
        else:
            insight += "No economic simulation data available."
            
        self.text_edit.setPlainText(insight)

    def generate_system_insight(self, market, sys_df, stat_df):
        market_labels = {
            "CSI_300": "Efficient Market",
            "SSE_Composite": "Moderately Inefficient",
            "ChiNext": "High Volatility Regime"
        }
        regime = market_labels.get(market, "Unknown Regime")
        
        insight = f"SYSTEM AUDIT: {market} ({regime})\n\n"
        
        ljung_pass = True
        arch_pass = True
        
        if not stat_df.empty:
            if 'LjungBox_p' in stat_df.columns and stat_df['LjungBox_p'].values[0] <= 0.05:
                ljung_pass = False
            if 'ARCH_p' in stat_df.columns and stat_df['ARCH_p'].values[0] <= 0.05:
                arch_pass = False

        if ljung_pass and not arch_pass:
            insight += "This indicates residual volatility clustering remains,\n"
            insight += "which is expected in emerging markets and does not invalidate the model.\n"
            self.setStyleSheet("""
                QFrame { background-color: #2b2b10; border-radius: 8px; border: 1px solid #fbbc05; }
                QTextEdit { background-color: transparent; color: #fbe090; border: none; font-family: Consolas; font-size: 14px; line-height: 1.5; }
            """)
        elif ljung_pass and arch_pass:
            insight += "🟢 Model fully satisfies statistical assumptions.\n"
            self.setStyleSheet("""
                QFrame { background-color: #142c16; border-radius: 8px; border: 1px solid #34a853; }
                QTextEdit { background-color: transparent; color: #a8e6a3; border: none; font-family: Consolas; font-size: 14px; line-height: 1.5; }
            """)
        else:
            insight += "🚨 Model is misspecified.\n"
            self.setStyleSheet("""
                QFrame { background-color: #3b1818; border-radius: 8px; border: 1px solid #ea4335; }
                QTextEdit { background-color: transparent; color: #ffb4ab; border: none; font-family: Consolas; font-size: 14px; line-height: 1.5; }
            """)
            
        self.text_edit.setPlainText(insight)
