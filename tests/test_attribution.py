import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.attribution import parse_months, parse_years


class AttributionTests(unittest.TestCase):
    def test_parse_months(self) -> None:
        self.assertEqual(parse_months("5,3,3"), [3, 5])

    def test_parse_years(self) -> None:
        self.assertEqual(parse_years("2026,2024"), [2024, 2026])

    def test_parse_months_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_months("0,13")


if __name__ == "__main__":
    unittest.main()

