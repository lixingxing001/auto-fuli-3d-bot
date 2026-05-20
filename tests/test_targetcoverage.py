import unittest
from datetime import date, timedelta
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.models import Draw  # noqa: E402
from fuli3d_bot.targetcoverage import (  # noqa: E402
    required_top_n,
    run_target_coverage,
    summarize_target,
)


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


class TargetCoverageTests(unittest.TestCase):
    def test_required_top_n_for_target_rate(self) -> None:
        self.assertEqual(required_top_n(0.65), 650)
        self.assertEqual(required_top_n(0.001), 1)

    def test_summarize_target_includes_negative_expected_roi(self) -> None:
        summary = summarize_target(0.65, stake_per_number=2.0, payout_per_hit=1040.0)

        self.assertEqual(summary.required_top_n, 650)
        self.assertAlmostEqual(summary.theoretical_hit_rate, 0.65)
        self.assertAlmostEqual(summary.expected_roi, -0.48)
        self.assertEqual(summary.feasibility_status, "coverage_only")

    def test_run_target_coverage_returns_target_rows(self) -> None:
        report = run_target_coverage(
            make_draws(35),
            target_rate=0.65,
            compare_top_values=[10],
            windows=[5],
            training_window=30,
            recent_window=20,
            min_history=20,
        )

        self.assertEqual(report.summary.required_top_n, 650)
        self.assertTrue(any(row.top_n == 650 and row.segment == "all" for row in report.rows))
        self.assertTrue(any(row.top_n == 650 and row.segment == "recent" for row in report.rows))


if __name__ == "__main__":
    unittest.main()
