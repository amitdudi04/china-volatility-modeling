"""Version-2.2 data freeze and retrospective outcome boundary tests."""
import json
from pathlib import Path
import unittest
import pandas as pd
from src.research_design import prepare_session_dataset
from scripts.run_archived_v22 import load_legacy_data

ROOT=Path(__file__).resolve().parents[1]

class Archive2024Tests(unittest.TestCase):
    def test_source_files_match_frozen_identity_and_dates(self):
        for market in ("CSI_300","SSE_Composite","ChiNext"):
            with self.subTest(market=market):
                raw,meta=load_legacy_data(market)
                self.assertEqual(raw.index.max(),pd.Timestamp("2024-12-30"))
                self.assertEqual(meta["rows"],len(raw))
                self.assertEqual(len(meta["sha256"]),64)
                self.assertFalse(raw.index.has_duplicates)

    def test_chinext_is_explicit_etf_not_index(self):
        _,meta=load_legacy_data("ChiNext")
        self.assertEqual(meta["symbol"],"159915.SZ")
        self.assertIn("ETF PROXY",meta["instrument"])

    def test_no_2025_or_2026_dates_in_retrospective_frames(self):
        for market in ("CSI_300","SSE_Composite","ChiNext"):
            with self.subTest(market=market):
                raw,_=load_legacy_data(market)
                frame=prepare_session_dataset(raw,horizon=1)
                self.assertTrue((pd.to_datetime(frame["target_end_date"])<=
                                 pd.Timestamp("2024-12-30")).all())
                ev=frame.loc[frame.index>=pd.Timestamp("2024-01-01")]
                self.assertGreater(len(ev),200)
                self.assertTrue((ev.index<pd.to_datetime(ev["target_end_date"])).all())

    def test_2023_can_be_validation_but_not_blind(self):
        p=json.loads((ROOT/"config"/"research_protocol_v22.json").read_text())
        self.assertIn("retrospective",p["windows"]["retrospective_evaluation"]["note"].lower())
        self.assertEqual(p["markets"]["CSI_300"]["rows"],925)

    def test_model_csvs_are_not_rewritten_during_loading(self):
        path=ROOT/"data"/"raw"/"CSI_300.csv"
        before=path.read_bytes()
        load_legacy_data("CSI_300")
        self.assertEqual(before,path.read_bytes())

if __name__=="__main__":
    unittest.main()
