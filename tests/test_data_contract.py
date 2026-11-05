"""Synthetic-only tests for the external-data recovery gate."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd
from src.data_contract import (
    MARKETS, standardize_download, validate_coverage,
    build_common_session_datasets, write_immutable_staging, sha256_file,
)


class DataContractTests(unittest.TestCase):
    @staticmethod
    def source(start="2015-01-05", end="2026-09-30"):
        dates = pd.bdate_range(start, end)
        steps = np.arange(len(dates))
        price = 100 * np.exp(np.cumsum(.0002 + .002*np.sin(.127*steps)))
        return pd.DataFrame({
            "Close": price, "Adj Close": price*.98,
            "Open": price*.998, "High": price*1.01,
            "Low": price*.99, "Volume": np.full(len(dates), 250000.),
        }, index=dates)

    def test_adjusted_close_is_explicit(self):
        frame = standardize_download(self.source(), "000300.SS")
        self.assertAlmostEqual(frame["Close"].iloc[0], frame["Raw_Close"].iloc[0]*.98)

    def test_missing_adjusted_close_fails(self):
        with self.assertRaisesRegex(ValueError, "missing mandatory"):
            standardize_download(self.source().drop(columns=["Adj Close"]), "000300.SS")

    def test_duplicate_dates_fail(self):
        frame = self.source().iloc[:45]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            standardize_download(pd.concat([frame, frame.iloc[[3]]]), "000300.SS")

    def test_missing_values_never_filled(self):
        frame = self.source().iloc[:60]
        frame.loc[frame.index[5], "Close"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing/nonfinite"):
            standardize_download(frame, "000300.SS")

    def test_truncated_csi_history_rejected(self):
        frame = standardize_download(self.source(start="2021-03-11"), "000300.SS")
        with self.assertRaisesRegex(ValueError, "history missing"):
            validate_coverage(frame, "CSI_300")

    def test_complete_synthetic_history_passes(self):
        frame = standardize_download(self.source(), "000300.SS")
        quality = validate_coverage(frame, "CSI_300")
        self.assertGreater(quality["rows"], 2600)
        self.assertEqual(quality["last"], "2026-09-30")

    def test_suspicious_unadjusted_corporate_action_blocks_release(self):
        frame = self.source()
        frame.loc[frame.index[800]:, "Adj Close"] *= 2
        standardized = standardize_download(frame, "000300.SS")
        with self.assertRaisesRegex(ValueError, "jumps exceed"):
            validate_coverage(standardized, "CSI_300")

    def test_common_trading_calendar_and_target_boundaries(self):
        full = standardize_download(self.source(), "000300.SS")
        markets = {market: full.copy() for market in MARKETS}
        markets["ChiNext"] = markets["ChiNext"].drop(markets["ChiNext"].index[350])
        datasets, audit = build_common_session_datasets(markets)
        self.assertEqual(audit["common_sessions"], len(full)-1)
        for market in MARKETS:
            for horizon in (1,5):
                frame = datasets[market][(horizon,"post_legacy_evaluation")]
                self.assertGreater(len(frame), 300)
                self.assertTrue((pd.to_datetime(frame["target_end_date"]) <= pd.Timestamp("2026-09-30")).all())

    def test_hash_manifest_and_no_overwrite(self):
        full = standardize_download(self.source(), "000300.SS")
        markets = {market: full.copy() for market in MARKETS}
        with TemporaryDirectory() as directory:
            dest = Path(directory)/"research-release"
            manifest = write_immutable_staging(dest, markets, {"provider": "synthetic"})
            data = json.loads(manifest.read_text())
            self.assertEqual(data["release_type"], "staged_market_inputs_not_model_results")
            self.assertEqual(len(data["sha256"]), 21)
            self.assertEqual(data["sha256"]["source_data/CSI_300.csv"],
                             sha256_file(dest/"source_data"/"CSI_300.csv"))
            with self.assertRaises(FileExistsError):
                write_immutable_staging(dest, markets, {"provider": "synthetic"})


if __name__ == "__main__":
    unittest.main()
