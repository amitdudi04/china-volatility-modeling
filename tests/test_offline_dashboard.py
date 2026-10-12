"""Offline dashboard release integrity — browser UI does not need Qt."""
import tempfile
import unittest
from pathlib import Path
from scripts.build_v22_dashboard import export_dashboard

class OfflineDashboardTests(unittest.TestCase):
    def test_three_verified_markets_embedded_in_one_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"dashboard.html"
            export_dashboard(file)
            html=file.read_text("utf-8")
            self.assertIn("Retrospective",html)
            for name in ("CSI_300","SSE_Composite","ChiNext"):
                self.assertIn(name,html)
            self.assertIn("not a claim of adaptive copula",html)
            self.assertGreater(len(html),30000)
            with self.assertRaises(FileExistsError):
                export_dashboard(file)

if __name__=="__main__":
    unittest.main()
