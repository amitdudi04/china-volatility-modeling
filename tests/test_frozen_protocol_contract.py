"""Prevent a protocol/market identity mismatch at future releases."""
import json
from pathlib import Path
import unittest
from src.data_contract import (
    MARKETS, DATA_START, DATA_END_EXCLUSIVE,
    TRAIN_END, VALIDATION_START, VALIDATION_END,
    POST_LEGACY_START, POST_LEGACY_END,
)
from scripts.fetch_research_data_v2_akshare import INDEX_SYMBOLS


class FrozenProtocolContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file = Path(__file__).resolve().parents[1]/"config"/"research_protocol_v2.json"
        cls.protocol = json.loads(file.read_text(encoding="utf-8"))

    def test_three_market_identifiers_match_frozen_protocol(self):
        p = self.protocol
        self.assertEqual(p["schema_version"], "research-protocol-v2.1")
        self.assertEqual(set(p["markets"]), set(MARKETS))
        for market, ticker in MARKETS.items():
            self.assertEqual(p["markets"][market]["ticker"], ticker)
        self.assertEqual(set(INDEX_SYMBOLS), set(MARKETS))
        self.assertEqual(INDEX_SYMBOLS["ChiNext"], "sz399006")

    def test_frozen_date_boundaries_match_code(self):
        p = self.protocol["windows"]
        self.assertEqual(DATA_START, "2015-01-01")
        self.assertEqual(DATA_END_EXCLUSIVE, "2026-10-01")
        self.assertEqual(TRAIN_END, p["development"]["end"])
        self.assertEqual((VALIDATION_START, VALIDATION_END),
                         (p["validation"]["start"], p["validation"]["end"]))
        self.assertEqual((POST_LEGACY_START, POST_LEGACY_END),
                         (p["new_holdout"]["start"], p["new_holdout"]["end"]))

    def test_legacy_commit_and_amendment_are_explicit(self):
        p = self.protocol
        self.assertEqual(p["legacy_main_commit"],
                         "c7d78bf4e59d3e1c08eedd429339398f82157f97")
        self.assertEqual(p["amendment_date"], "2026-10-12")
        self.assertFalse(p["data_ready"])


if __name__ == "__main__":
    unittest.main()
