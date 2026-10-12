"""Verify key README and academic-report assertions from real v2.2 files.

Use market-specific table labels to prevent selecting unrelated tables with
the same market name. Historical claims remain appropriately qualified.
"""
from pathlib import Path
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RESEARCH=ROOT/"outputs"/"research_v22"
README=(ROOT/"README.md").read_text(encoding="utf-8")
REPORT=(ROOT/"docs"/"RESEARCH_V22_ARCHIVED_RESULTS.md").read_text(encoding="utf-8")
MARKETS={
    "CSI_300":{"readme":"CSI 300","tail":"CSI 300","econ":"CSI 300"},
    "SSE_Composite":{"readme":"SSE Composite","tail":"SSE Composite","econ":"SSE"},
    "ChiNext":{"readme":"ChiNext ETF proxy","tail":"ChiNext ETF proxy","econ":"ChiNext ETF proxy"},
}

class PublishedClaimTests(unittest.TestCase):
    def test_readme_winner_figures_derive_from_csv(self):
        scores=pd.read_csv(RESEARCH/"2024_model_loss_summary.csv")
        for market,case in MARKETS.items():
            with self.subTest(market=market):
                sample=scores.loc[scores.market==market]
                self.assertEqual(len(sample),5)
                self.assertEqual(set(sample.n),{240})
                a=sample.sort_values("rmse").iloc[0]
                b=sample.sort_values("qlike").iloc[0]
                # Numeric delimiters distinguish the performance summary
                # from the instrument/coverage table above it.
                text=next(line for line in README.splitlines()
                    if line.startswith("| "+case["readme"]+" |")
                    and f"({a.rmse:.4f})" in line)
                self.assertIn(f"({b.qlike:.4f})",text.replace("−", "-"))

    def test_student_t_var_backtest_figures_derive_from_csv(self):
        data=pd.read_csv(RESEARCH/"student_t_var_backtests.csv")
        for market,case in MARKETS.items():
            with self.subTest(market=market):
                row=data.loc[data.market==market].iloc[0]
                prefix=(f'| {case["tail"]} | {int(row.breaches)} '
                        f'| {row.kupiec_pvalue:.4f} |')
                line=next(x for x in REPORT.splitlines() if x.startswith(prefix))
                self.assertIn(f"| {row.christoffersen_ind_pvalue:.4f} |",line)
                self.assertIn(f"| {row.christoffersen_cc_pvalue:.4f} |",line)
                self.assertEqual(int(row.observations),240)

    def test_economic_headlines_derived_from_net_cost_csv(self):
        data=pd.read_csv(RESEARCH/"exposure_strategy_sensitivity.csv")
        for market,case in MARKETS.items():
            with self.subTest(market=market):
                sample=data[(data.market==market)&(data.cost_bps==10)]
                static=sample.loc[sample.strategy=="static"].iloc[0]
                managed=sample.loc[(sample.strategy=="volatility_managed")&
                    (sample.forecast_model=="garch11_variance")].iloc[0]
                prefix=f'| {case["econ"]} | {100*static.annualized_return:.2f}% |'
                line=next(x for x in REPORT.splitlines() if x.startswith(prefix))
                for value in (managed.annualized_return,static.max_drawdown,managed.max_drawdown):
                    self.assertIn(f"{100*value:.2f}%",line)

    def test_legacy_word_and_scope_are_explicitly_disclaimed(self):
        self.assertFalse((ROOT/"docs"/"results_analysis.docx").exists())
        self.assertTrue((ROOT/"docs"/"legacy_results_analysis_2026_SUPERSEDED.docx").exists())
        self.assertIn("not an untouched",README.lower())
        self.assertIn("**not** an untouched",REPORT.lower())
        self.assertTrue((ROOT/"docs"/"FINAL_PANEL_REVIEW.md").exists())

if __name__=="__main__":
    unittest.main()
