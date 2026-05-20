import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.lawcheck import run_law_check, save_law_check_reports  # noqa: E402
from fuli3d_bot.models import Draw  # noqa: E402


def make_formula_draws(count: int) -> list[Draw]:
    start = date(2026, 1, 1)
    value = 17
    draws: list[Draw] = []
    for index in range(count):
        draws.append(
            Draw(
                issue=f"2026{index + 1:03d}",
                draw_date=start + timedelta(days=index),
                number=f"{value:03d}",
            )
        )
        value = (value + 7) % 1000
    return draws


class LawCheckTests(unittest.TestCase):
    def test_law_check_detects_constructed_affine_formula(self) -> None:
        report = run_law_check(
            make_formula_draws(80),
            max_lag=5,
            min_formula_history=10,
            split_ratio=0.7,
        )

        affine = next(row for row in report.formula_tests if row.name == "最佳线性同余公式")
        self.assertEqual(affine.test_hits, affine.test_rounds)
        self.assertEqual(affine.status, "validated")
        self.assertGreaterEqual(report.summary.validated_formulas, 1)

    def test_save_law_check_reports(self) -> None:
        report = run_law_check(
            make_formula_draws(40),
            max_lag=3,
            min_formula_history=8,
            split_ratio=0.7,
        )
        with TemporaryDirectory() as temp_dir:
            json_path, md_path = save_law_check_reports(
                report,
                temp_dir,
                {"draw_rows": 40},
            )

            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())


if __name__ == "__main__":
    unittest.main()
