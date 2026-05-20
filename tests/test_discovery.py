import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.discovery import run_formula_discovery, save_discovery_reports  # noqa: E402
from fuli3d_bot.models import Draw  # noqa: E402


def make_transition_draws(count: int) -> list[Draw]:
    start = date(2026, 1, 1)
    return [
        Draw(
            issue=f"2026{index + 1:03d}",
            draw_date=start + timedelta(days=index),
            number=str(index % 10) * 3,
        )
        for index in range(count)
    ]


class FormulaDiscoveryTests(unittest.TestCase):
    def test_discovery_finds_constructed_transition_formula(self) -> None:
        report = run_formula_discovery(
            make_transition_draws(120),
            windows=[30],
            min_history=30,
            show_top=5,
        )

        self.assertEqual(report.summary.conclusion_status, "candidate_found")
        self.assertGreaterEqual(report.summary.selected_test_hits, 15)
        self.assertLess(report.summary.selected_test_p_value, 0.05)

    def test_save_discovery_reports(self) -> None:
        report = run_formula_discovery(
            make_transition_draws(80),
            windows=[30],
            min_history=30,
            show_top=3,
        )
        with TemporaryDirectory() as temp_dir:
            json_path, md_path = save_discovery_reports(
                report,
                temp_dir,
                {"draw_rows": 80},
            )

            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())


if __name__ == "__main__":
    unittest.main()
