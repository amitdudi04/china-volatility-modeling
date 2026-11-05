"""Cross-platform SHA-256 compatibility of Windows CRLF archived manifests."""
import json
import tempfile
import unittest
from pathlib import Path
from src.utils.archived_digest import archived_file_sha256

ROOT=Path(__file__).resolve().parents[1]

class ArchivedPortableDigestTests(unittest.TestCase):
    def test_same_archived_csv_digest_across_git_eol(self):
        with tempfile.TemporaryDirectory() as dirname:
            path=Path(dirname)/"same.csv"
            path.write_bytes(b"Date,Close\r\n2024-01-02,100\r\n2024-01-03,101\r\n")
            win=archived_file_sha256(path)
            path.write_bytes(b"Date,Close\n2024-01-02,100\n2024-01-03,101\n")
            self.assertEqual(win,archived_file_sha256(path))
            path.write_bytes(b"Date,Close\n2024-01-02,100\n2024-01-03,102\n")
            self.assertNotEqual(win,archived_file_sha256(path))

    def test_all_original_prices_match_immutable_stored_sha(self):
        release=ROOT/"outputs"/"research_v22"
        for market in ("CSI_300","SSE_Composite","ChiNext"):
            with self.subTest(market=market):
                m=json.loads((release/f"{market}_2024_complete_manifest.json").read_text())
                path=ROOT/m["source"]["source_relative_path"]
                self.assertEqual(archived_file_sha256(path),m["source"]["sha256"])
                self.assertEqual(archived_file_sha256(release/f"{market}_2024_complete_predictions.csv"),
                                 m["predictions_sha256"])

    def test_fitted_student_t_manifest_matches_csvs(self):
        release=ROOT/"outputs"/"research_v22"
        m=json.loads((release/"student_t_tail_manifest.json").read_text())
        for name,sha in m["results_sha256"].items():
            with self.subTest(path=name):
                self.assertEqual(archived_file_sha256(release/name),sha)

if __name__=="__main__":
    unittest.main()
