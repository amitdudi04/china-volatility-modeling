"""EastMoney schema mapping tests — no external network calls."""
import unittest
import pandas as pd
from scripts.fetch_research_data_v2_akshare import _index_frame, _etf_frame


class EastMoneyMappingTests(unittest.TestCase):
    @staticmethod
    def index_history():
        dates = pd.bdate_range("2020-01-01", periods=40)
        return pd.DataFrame({
            "date":dates, "open":100., "high":110., "low":90.,
            "close":102., "volume":500.,
        })

    @staticmethod
    def etf_histories():
        dates = pd.bdate_range("2020-01-01", periods=40)
        raw = pd.DataFrame({
            "日期":dates, "开盘":100., "收盘":102.,
            "最高":110., "最低":90., "成交量":1000.,
        })
        qfq = pd.DataFrame({"日期":dates, "收盘":100.})
        return raw, qfq

    def test_index_schema_and_price_index(self):
        result = _index_frame(self.index_history(), "CSI_300")
        self.assertTrue(result["Close"].equals(result["Raw_Close"]))
        self.assertEqual(result.index.name, "Date")

    def test_adjusted_etf_close_distinct_from_raw_close(self):
        raw, qfq = self.etf_histories()
        result = _etf_frame(raw, qfq)
        self.assertEqual(result["Close"].iloc[0], 100.)
        self.assertEqual(result["Raw_Close"].iloc[0], 102.)

    def test_missing_etf_adjusted_calendar_must_fail(self):
        raw, qfq = self.etf_histories()
        with self.assertRaisesRegex(ValueError, "date calendars differ"):
            _etf_frame(raw, qfq.iloc[:-1])


if __name__ == "__main__":
    unittest.main()
