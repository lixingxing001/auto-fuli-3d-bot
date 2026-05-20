import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.features import extract_features, normalize_number, pattern_label


class FeatureTests(unittest.TestCase):
    def test_normalize_number(self) -> None:
        self.assertEqual(normalize_number("7"), "007")
        self.assertEqual(normalize_number(42), "042")
        self.assertEqual(normalize_number("314"), "314")

    def test_extract_features(self) -> None:
        features = extract_features("338")
        self.assertEqual(features.digits, (3, 3, 8))
        self.assertEqual(features.sum_value, 14)
        self.assertEqual(features.span, 5)
        self.assertEqual(pattern_label(features.pattern), "组三")


if __name__ == "__main__":
    unittest.main()
