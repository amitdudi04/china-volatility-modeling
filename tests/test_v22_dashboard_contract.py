"""Release-schema and GUI data contract checks (no graphical display required)."""
import io
import contextlib
import unittest
from src.gui.v22_data import MARKETS,load_research_market
from main import main

class V22DashboardContractTests(unittest.TestCase):
    def test_all_research_releases_are_verifiable(self):
        for market in MARKETS:
            with self.subTest(market=market):
                d=load_research_market(market)
                self.assertEqual(len(d["predictions"]),240)
                self.assertEqual(len(d["scores"]),5)
                self.assertEqual(len(d["risk"]),5)
                self.assertEqual(len(d["strategy"]),16)
                self.assertIn("retrospective",d["scope"].lower())

    def test_chinext_is_not_called_an_index(self):
        self.assertIn("ETF PROXY",load_research_market("ChiNext")["source"]["instrument"])
        self.assertIn("NOT the ChiNext index",load_research_market("ChiNext")["instrument_note"])

    def test_cli_checksum_verification(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--verify-v22"]),0)
        self.assertIn("PASS ChiNext",output.getvalue())

    def test_legacy_pipeline_requires_explicit_overwrite_permission(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["--run-legacy-pipeline"])
        self.assertEqual(caught.exception.code,2)

if __name__=="__main__":
    unittest.main()
