import unittest
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import asos_pipeline as ap


class TestAsosPipeline(unittest.TestCase):
    def test_calculate_stockout_metrics(self):
        count, rate = ap.calculate_stockout_metrics(
            "UK 4 - Out of stock,UK 6,UK 8 - Out of stock"
        )
        self.assertEqual(count, 2)
        self.assertAlmostEqual(rate, 2 / 3, places=6)

    def test_extract_brand_simple(self):
        s = pd.Series(["Dress by New Look with details", "No brand info"])
        out = ap.extract_brand_simple(s)
        self.assertEqual(out.iloc[0], "New Look")
        self.assertEqual(out.iloc[1], "Unknown")

    def test_extract_brand_multiword(self):
        s = pd.Series(["Jacket by River Island classic fit"])
        out = ap.extract_brand_multiword(s)
        self.assertIn("River", out.iloc[0])

    def test_weighted_risk_by_price_band(self):
        df = pd.DataFrame(
            {
                "price_band": ["0-20", "100+"],
                "Revenue_Risk_Score": [100.0, 100.0],
            }
        )
        out = ap.weighted_risk_by_price_band(df)
        self.assertAlmostEqual(float(out.iloc[0]), 100.0)
        self.assertAlmostEqual(float(out.iloc[1]), 160.0)

    def test_priority_score_exists(self):
        df = pd.DataFrame(
            {
                "sku": [1, 1, 2, 2],
                "Brand": ["A", "A", "B", "B"],
                "name": ["n1", "n1", "n2", "n2"],
                "price": [10.0, 12.0, 100.0, 90.0],
                "Stockout_Rate": [0.1, 0.2, 0.6, 0.7],
                "Stockout_Count": [1, 2, 4, 5],
                "Revenue_Risk_Score": [10.0, 24.0, 400.0, 450.0],
            }
        )
        out = ap.build_sku_priority_table(df)
        self.assertIn("Priority_Score", out.columns)
        self.assertGreaterEqual(float(out["Priority_Score"].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
