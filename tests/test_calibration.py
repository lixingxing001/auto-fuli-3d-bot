import unittest
from datetime import date, timedelta
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.calibration import (  # noqa: E402
    CalibrationPick,
    rank_buckets,
    run_calibration,
    summarize_rank_buckets,
    summarize_top_hits,
)
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


class CalibrationTests(unittest.TestCase):
    def test_summarize_top_hits(self) -> None:
        picks = [
            CalibrationPick("1", None, "001", 5, 1.0),
            CalibrationPick("2", None, "002", 20, 0.8),
            CalibrationPick("3", None, "003", 300, 0.1),
        ]
        row = summarize_top_hits(picks, top_n=20)

        self.assertEqual(row.rounds, 3)
        self.assertEqual(row.hits, 2)
        self.assertAlmostEqual(row.expected_hits, 0.06)
        self.assertEqual(row.max_losing_streak, 1)

    def test_summarize_rank_buckets(self) -> None:
        picks = [
            CalibrationPick("1", None, "001", 5, 1.0),
            CalibrationPick("2", None, "002", 150, 0.8),
            CalibrationPick("3", None, "003", 950, 0.1),
        ]
        rows = summarize_rank_buckets(picks, bucket_size=100)

        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0].bucket, "1-100")
        self.assertEqual(rows[0].hits, 1)
        self.assertEqual(rows[-1].bucket, "901-1000")
        self.assertEqual(rows[-1].hits, 1)

    def test_rank_buckets_handles_tail(self) -> None:
        self.assertEqual(rank_buckets(333)[-1], (1000, 1000))

    def test_run_calibration_returns_report(self) -> None:
        report = run_calibration(
            make_draws(35),
            top_values=[10, 20],
            windows=[5],
            training_window=30,
            recent_window=20,
            min_history=20,
            bucket_size=250,
        )

        self.assertEqual(report.summary.rounds, 5)
        self.assertEqual(len([row for row in report.top_rows if row.segment == "all"]), 2)
        self.assertEqual(len([row for row in report.bucket_rows if row.segment == "all"]), 4)


if __name__ == "__main__":
    unittest.main()
