"""Independent EastMoney index mapping tests; no external network calls."""
import unittest
import pandas as pd
from scripts.fetch_research_data_v2_akshare import _index_frame, INDEX_SYMBOLS


class EastMoneyIndexMappingTests(unittest.TestCase):
    @staticmethod
    def source():
        dates = pd.bdate_range("2020-01-01", periods=40)
        return pd.DataFrame({
            "date":dates, "open":100., "high":110., "low":90.,
            "close":102., "volume":500.,
        })

    def test_all_three_symbols_are_real_indices(self):
        self.assertEqual(INDEX_SYMBOLS, {
            "CSI_300":"sh000300",
            "SSE_Composite":"sh000001",
            "ChiNext":"sz399006",
        })

    def test_index_fields_canonicalized_for_all_three(self):
        for market in INDEX_SYMBOLS:
            with self.subTest(market=market):
                result = _index_frame(self.source(), market)
                self.assertTrue(result["Close"].equals(result["Raw_Close"]))
                self.assertEqual(result.index.name, "Date")

    def test_missing_index_column_fails(self):
        for market in INDEX_SYMBOLS:
            with self.subTest(market=market):
                with self.assertRaisesRegex(ValueError, "missing EastMoney"):
                    _index_frame(self.source().drop(columns=["close"]), market)


if __name__ == "__main__":
    unittest.main()
