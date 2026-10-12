"""Publication-level verification that README tables match frozen v2.2 results.

The tests are intentionally derived from numerical outputs, not copied from
a separate manually written performance table.
"""
from pathlib import Path
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RESEARCH=ROOT/"outputs"/"research_v22"
README=(ROOT/"README.md").read_text(encoding="utf-8")
REPORT=(ROOT/"docs"/"RESEARCH_V22_ARCHIVED_RESULTS.md").read_text(encoding="utf-8")

MARKETS={
    "CSI_300":{"readme":"CSI 300","report":"CSI 300",
               "rmse_name":"Fixed Ridge–RF blend","qlike_name":"GARCH(1,1)"},
    "SSE_Composite":{"readme":"SSE Composite","report":"SSE",
                     "rmse_name":"GARCH(1,1)","qlike_name":"GARCH(1,1)"},
    "ChiNext":{"readme":"ChiNext ETF proxy","report":"ChiNext ETF",
               "rmse_name":"Random Forest","qlike_name":"GARCH(1,1)"},
}

class PublishedClaimTests(unittest.TestCase):
    def test_readme_winning_model_and_loss_numbers(self):
        scores=pd.read_csv(RESEARCH/"2024_model_loss_summary.csv")
        for market,case in MARKETS.items():
            with self.subTest(market=market):
                sub=scores.loc[scores.market==market]
                self.assertEqual(len(sub),5)
                self.assertEqual(set(sub.n),{240})
                a=sub.sort_values("rmse").iloc[0]
                b=sub.sort_values("qlike").iloc[0]
                line=next(x for x in README.splitlines() if x.startswith("| "+case["readme"]+" |"))
                self.assertIn(case["rmse_name"],line)
                self.assertIn(case["qlike_name"],line)
                self.assertIn(f"{a.rmse:.4f}",line)
                self.assertIn(f"{b.qlike:.4f}",line)

    def test_formal_student_t_var_results_match_report(self):
        data=pd.read_csv(RESEARCH/"student_t_var_backtests.csv")
        for market,case in MARKETS.items():
            with self.subTest(market=market):
                row=data.loc[data.market==market].iloc[0]
                line=next(x for x in REPORT.splitlines() if
                          x.startswith("| "+case["report"]+" |") and
                          " | 0." in x and " | " in x and
                          str(int(row.breaches)) in x)
                self.assertIn(f"{row.kupiec_pvalue:.4f}",line)
                self.assertIn(f"{row.christoffersen_ind_pvalue:.4f}",line)
                self.assertIn(f"{row.christoffersen_cc_pvalue:.4f}",line)
                self.assertEqual(int(row.observations),240)

    def test_economic_headlines_match_archived_cost_sensitivity(self):
        data=pd.read_csv(RESEARCH/"exposure_strategy_sensitivity.csv")
        for market,case in MARKETS.items():
            with self.subTest(market=market):
                sub=data[(data.market==market)&(data.cost_bps==10)]
                static=sub[sub.strategy=="static"].iloc[0]
                managed=sub[(sub.strategy=="volatility_managed")&
                            (sub.forecast_model=="garch11_variance")].iloc[0]
                report=next(x for x in REPORT.splitlines()
                            if x.startswith("| "+case["report"]+" |") and "%" in x
                            and f"{100*static.annualized_return:.2f}%" in x)
                self.assertIn(f"{100*managed.annualized_return:.2f}%",report)
                self.assertIn(f"{100*static.max_drawdown:.2f}%",report)
                self.assertIn(f"{100*managed.max_drawdown:.2f}%",report)

    def test_legacy_word_doc_is_not_presented_as_current(self):
        self.assertFalse((ROOT/"docs"/"results_analysis.docx").exists())
        self.assertTrue((ROOT/"docs"/"legacy_results_analysis_2026_SUPERSEDED.docx").exists())
        self.assertIn("not an untouched",README.lower())
        self.assertIn("not",REPORT.lower())
        self.assertTrue((ROOT/"docs"/"FINAL_PANEL_REVIEW.md").exists())

if __name__=="__main__":
    unittest.main()
