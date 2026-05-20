import unittest
from datetime import date
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.models import Draw
from fuli3d_bot.validation import validate_draws


class ValidationTests(unittest.TestCase):
    def test_detects_issue_gap(self) -> None:
        draws = [
            Draw(issue="2026001", draw_date=date(2026, 1, 1), number="123"),
            Draw(issue="2026003", draw_date=date(2026, 1, 3), number="456"),
        ]
        report = validate_draws(draws)
        self.assertEqual(report.issue_gaps, ["2026001->2026003"])
        self.assertEqual(report.date_gaps, ["2026001->2026003:2d"])

    def test_detects_duplicate_issue(self) -> None:
        draws = [
            Draw(issue="2026001", draw_date=date(2026, 1, 1), number="123"),
            Draw(issue="2026001", draw_date=date(2026, 1, 2), number="456"),
        ]
        report = validate_draws(draws)
        self.assertFalse(report.ok)
        self.assertEqual(report.duplicate_issues, ["2026001"])


if __name__ == "__main__":
    unittest.main()
