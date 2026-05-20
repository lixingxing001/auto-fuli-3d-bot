import unittest
from datetime import date, timedelta
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.models import Draw  # noqa: E402
from fuli3d_bot.variantstress import (  # noqa: E402
    parse_variant_names,
    run_variant_stress,
    selected_cases,
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


class VariantStressTests(unittest.TestCase):
    def test_parse_variant_names(self) -> None:
        self.assertEqual(parse_variant_names("baseline, no_repeat_penalty"), ["baseline", "no_repeat_penalty"])

    def test_selected_cases_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            selected_cases(["missing"])

    def test_run_variant_stress_returns_segments(self) -> None:
        report = run_variant_stress(
            make_draws(35),
            variant_names=["baseline", "no_repeat_penalty"],
            top_values=[10],
            windows=[5],
            years=[2026],
            training_window=30,
            recent_window=20,
            min_history=20,
        )

        self.assertTrue(report.summary.best_variant)
        self.assertTrue(any(row.segment == "all" for row in report.rows))
        self.assertTrue(any(row.segment == "year" for row in report.rows))
        self.assertTrue(any(row.segment == "recent" for row in report.rows))


if __name__ == "__main__":
    unittest.main()
