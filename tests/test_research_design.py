"""Leakage and timestamp contract tests for research v2 (synthetic only)."""
import unittest
import numpy as np
import pandas as pd
from src.research_design import (
    prepare_session_dataset, chronological_partition,
    completed_training_rows, verify_feature_causality
)


class ResearchDesignTests(unittest.TestCase):
    @staticmethod
    def sample():
        dates = pd.bdate_range("2021-01-01", periods=140)
        values = 100 * np.exp(np.cumsum(0.01*np.sin(np.arange(140)*0.31)))
        return pd.DataFrame({"Close": values}, index=dates)

    def test_next_day_label(self):
        raw = self.sample()
        f = prepare_session_dataset(raw)
        origin = f.index[0]
        k = raw.index.get_loc(origin)
        expected = float(np.log(raw["Close"].iloc[k+1]/raw["Close"].iloc[k]))
        self.assertAlmostEqual(f.loc[origin, "next_session_log_return"], expected)
        self.assertAlmostEqual(f.loc[origin, "target_variance_ann"], 252*expected**2)
        self.assertEqual(pd.Timestamp(f.loc[origin, "target_end_date"]), raw.index[k+1])

    def test_five_session_future_target(self):
        raw = self.sample()
        f = prepare_session_dataset(raw, 5)
        origin = f.index[5]
        k = raw.index.get_loc(origin)
        a = raw["Close"].iloc[k+1:k+6].to_numpy()
        b = raw["Close"].iloc[k:k+5].to_numpy()
        expected = np.sqrt(252*np.mean(np.log(a/b)**2))
        self.assertAlmostEqual(f.loc[origin, "target_vol_ann"], expected)
        self.assertEqual(pd.Timestamp(f.loc[origin, "target_end_date"]), raw.index[k+5])

    def test_prefix_truncation_does_not_change_features(self):
        self.assertTrue(verify_feature_causality(self.sample(), 1))
        self.assertTrue(verify_feature_causality(self.sample(), 5))

    def test_split_excludes_unfinished_future_label(self):
        f = prepare_session_dataset(self.sample(), 5)
        end = str(f.index[45].date())
        split = chronological_partition(f, "2021-01-01", end)
        self.assertTrue((pd.to_datetime(split["target_end_date"]) <= pd.Timestamp(end)).all())

    def test_training_outcomes_available_before_fit(self):
        f = prepare_session_dataset(self.sample(), 5)
        origin = f.index[60]
        tr = completed_training_rows(f, origin)
        self.assertTrue((pd.to_datetime(tr["target_end_date"]) <= origin).all())
        self.assertTrue((tr.index < origin).all())

    def test_duplicate_dates_rejected(self):
        raw = self.sample()
        with self.assertRaises(ValueError):
            prepare_session_dataset(pd.concat([raw.iloc[:3], raw.iloc[2:]]))

    def test_invalid_close_rejected(self):
        raw = self.sample()
        raw.iloc[7, 0] = 0
        with self.assertRaises(ValueError):
            prepare_session_dataset(raw)


if __name__ == "__main__":
    unittest.main()
