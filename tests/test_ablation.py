import unittest
from datetime import date, timedelta
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.ablation import default_ablation_cases, run_ablation  # noqa: E402
from fuli3d_bot.models import Draw  # noqa: E402


def make_draws(count: int) -> list[Draw]:
    start = date(2026, 1, 1)
    return [
        Draw(
            issue=f"2026{index + 1:03d}",
            draw_date=start + timedelta(days=index),
            number=f"{(index * 37) % 1000:03d}",
        )
        for index in range(count)
    ]


class AblationTests(unittest.TestCase):
    def test_default_ablation_cases_include_baseline_and_only_cases(self) -> None:
        names = [case.name for case in default_ablation_cases()]

        self.assertIn("baseline", names)
        self.assertIn("no_recent_position", names)
        self.assertIn("position_only", names)
        self.assertIn("shape_stack", names)

    def test_run_ablation_returns_summary_and_rows(self) -> None:
        report = run_ablation(
            make_draws(35),
            top_values=[10],
            windows=[5],
            training_window=30,
            recent_window=20,
            min_history=20,
        )

        self.assertTrue(report.summary.best_case)
        self.assertGreater(len(report.rows), 0)
        self.assertTrue(any(row.case == "baseline" for row in report.rows))
        self.assertTrue(any(row.segment == "recent" for row in report.rows))


if __name__ == "__main__":
    unittest.main()
