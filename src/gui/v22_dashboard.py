"""Verified 2024 research dashboard — never loads legacy model outputs.

An interactive visual reader for the saved, checksummed retrospective research.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QTabWidget,
    QComboBox,QLabel,QTableWidget,QTableWidgetItem,QHeaderView,QTextEdit,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from src.gui.v22_data import load_research_market,MARKETS
from src.risk_economic_v22 import align_next_session_returns

LABELS={
    "garch11_variance":"GARCH(1,1), Student-t",
    "random_forest_variance":"Random Forest",
    "rolling_21_variance":"21-session rolling baseline",
    "ridge_rf_equal_variance":"Equal-weight Ridge / RF",
    "logvar_ridge_variance":"Log-variance Ridge (guardrailed)",
}
def show_table(df:pd.DataFrame)->QTableWidget:
    frame=df.reset_index(drop=True).replace({np.nan:"—"})
    table=QTableWidget(len(frame),len(frame.columns))
    table.setHorizontalHeaderLabels([str(c).replace("_"," ") for c in frame.columns])
    for r in range(len(frame)):
        for c in range(len(frame.columns)):
            val=frame.iat[r,c]
            display=f"{val:.4f}" if isinstance(val,(float,np.floating)) else str(val)
            table.setItem(r,c,QTableWidgetItem(display))
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.setAlternatingRowColors(True)
    table.setSortingEnabled(True)
    return table

def text_box(text:str)->QTextEdit:
    widget=QTextEdit()
    widget.setReadOnly(True)
    widget.setPlainText(text)
    widget.setMinimumHeight(115)
    return widget

def line_chart(x,series,title,y_label)->Canvas:
    fig=Figure(figsize=(11,4),tight_layout=True)
    ax=fig.add_subplot(111)
    for name,y in series.items():
        ax.plot(x,y,label=name,linewidth=1.3)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(alpha=.2)
    ax.legend(loc="best",fontsize=8)
    return Canvas(fig)

def wealth_curves(dataset,market):
    from src.risk_economic_v22 import align_next_session_returns
    pred=dataset["predictions"]
    r=align_next_session_returns(pred,dataset["prices"])
    simple=np.expm1(r.to_numpy(float))
    vol=np.sqrt(pred["garch11_variance"].to_numpy(float))
    weights=np.minimum(1.0,.20/vol)
    turnover=np.abs(weights-np.r_[1.0,weights[:-1]])
    static=np.cumprod(1+simple)
    managed=np.cumprod(1+weights*simple-.001*turnover)
    return {"Static (index / raw ETF close)":static,"GARCH-managed (10bps)":managed}

class ResearchDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("China Volatility | Research v2.2 — Retrospective (Not Blind Holdout)")
        self.resize(1420,900)
        self.setStyleSheet("QMainWindow { background: #121820; } QWidget { color: #ededed; } QTableWidget { background: #222b34; gridline-color:#41505c; } QHeaderView::section { background: #283c50; color:#ffffff; } QTextEdit { background:#1b2934; }")
        container=QWidget();self.setCentralWidget(container)
        layout=QVBoxLayout(container)
        banner=QLabel("RESEARCH V2.2 | HISTORICAL 2024 WALK-FORWARD | NOT AN UNTOUCHED HOLDOUT")
        banner.setStyleSheet("font-size:17px;font-weight:bold;color:#efbd75;padding:10px")
        layout.addWidget(banner)
        bar=QHBoxLayout();bar.addWidget(QLabel("Market:"))
        self.market=QComboBox();self.market.addItems(MARKETS)
        bar.addWidget(self.market);bar.addStretch();layout.addLayout(bar)
        self.tabs=QTabWidget();layout.addWidget(self.tabs)
        self.market.currentTextChanged.connect(self.refresh)
        self.refresh(MARKETS[0])

    def refresh(self,market):
        try:
            d=load_research_market(market)
        except Exception as exc:
            self.tabs.clear()
            view=QWidget();lay=QVBoxLayout(view)
            lay.addWidget(text_box("RESEARCH DATA VERIFICATION FAILED.\n"+str(exc)))
            self.tabs.addTab(view,"Data error")
            return
        self.tabs.clear()
        src=d["source"];scores=d["model_summary"].sort_values("qlike")
        best_qlike=scores.iloc[0];best_rmse=scores.sort_values("rmse").iloc[0]
        overview=QWidget();v=QVBoxLayout(overview)
        v.addWidget(text_box(
            f"{market}: {src['instrument']} | {src['first']} to {src['last']} | {src['rows']} raw rows\n"
            f"240 2024 forecast origins | primary label = annualized next-session squared log return\n"
            f"Lowest retrospective QLIKE: {LABELS.get(best_qlike['model'],best_qlike['model'])}, {best_qlike['qlike']:.4f}\n"
            f"Lowest retrospective RMSE: {LABELS.get(best_rmse['model'],best_rmse['model'])}, {best_rmse['rmse']:.4f}\n"
            "Do not infer statistically established or blind-sample model superiority solely from this ranking.\n"
            +d["instrument_note"]))
        v.addWidget(show_table(scores[["model","n","rmse","mae","qlike"]]))
        self.tabs.addTab(overview,"Research overview")

        predictions=d["predictions"]
        model=QWidget();v=QVBoxLayout(model)
        v.addWidget(text_box("All curves use the same annualized VARIANCE units; realized one-day squared returns are noisy. GARCH is actual fitted GARCH, Ridge is HAR-inspired only, and the blend is fixed 50/50 (not adaptive or copula)."))
        v.addWidget(line_chart(predictions.index,
            {"Realized next-session variance":predictions["realized_next_variance"],
             "GARCH(1,1)":predictions["garch11_variance"],
             "Random Forest":predictions["random_forest_variance"]},
             "Next-session annualized variance: realized proxy vs forecasts","Annualized variance"))
        v.addWidget(show_table(d["bootstrap"][["model","mean_qlike_difference","ci95_low","ci95_high","ci_excludes_zero"]]))
        self.tabs.addTab(model,"Forecast diagnostics")

        risk=QWidget();v=QVBoxLayout(risk)
        v.addWidget(text_box(
            "EXPLORATORY 95% GAUSSIAN VaR: Zero forecast mean, sigma_daily = sqrt(predicted annual variance/252).\n"
            "Kupiec unconditional coverage checks the exceedance RATE only. A large p-value does NOT demonstrate safety, return-independence, tail calibration or adequate Expected Shortfall.\n"
            "Results in this tab are NOT fitted Student-t VaR or Christoffersen tests."))
        v.addWidget(show_table(d["risk"][["model","observations","breaches","observed_rate","kupiec_pvalue"]]))
        self.tabs.addTab(risk,"Tail risk")

        econ=QWidget();v=QVBoxLayout(econ)
        t=d["strategy"];t=t[(t["cost_bps"]==10)
            &(t["forecast_model"].isin(("none","garch11_variance")))]
        t=t[((t["strategy"]=="static")|(t["forecast_model"]=="garch11_variance"))].drop_duplicates(subset=["strategy","forecast_model"])
        v.addWidget(text_box(
            "EXPLORATORY volatility targeting only: Target annual vol 20%, exposure capped at 100%; no leverage; cash return zero; 10 basis points per one-way turnover.\n"
            "Index prices are not directly tradeable; ETF raw Close omits an adjusted total-return convention. These are not investable net-of-all-fees portfolio returns.\n"
            "Compare both absolute return AND risk reduction before drawing any economic conclusion."))
        v.addWidget(line_chart(predictions.index,wealth_curves(d,market),
                                "Illustrative 2024 wealth paths (starting at 1)","Wealth / initial wealth"))
        v.addWidget(show_table(t[["strategy","annualized_return","annualized_vol","sharpe_zero_cash","max_drawdown","average_exposure"]]))
        self.tabs.addTab(econ,"Economic sensitivity")

        audit=QWidget();v=QVBoxLayout(audit)
        v.addWidget(text_box(
            f"SOURCE: {src['source_relative_path']}\n"
            f"SHA-256: {src['sha256']}\n"
            f"Instrument identity: {src['instrument']}\n"
            f"Evaluation: {d['scope']}\n\n"
            "Provenance, loss tables, source checks and model results live under outputs/research_v22/.\n"
            "Legacy src/pipeline.py / outputs/results are superseded for quantitative claims.\n"
            "Remaining limits: no independent new-period test; ChiNext is an ETF proxy; no fully investable total-return simulation; no corrected Student-t ES diagnostics in this GUI."))
        self.tabs.addTab(audit,"Provenance and limits")

def main():
    app=QApplication.instance() or QApplication(sys.argv)
    win=ResearchDashboard()
    win.show()
    return app.exec()

if __name__=="__main__":
    raise SystemExit(main())
