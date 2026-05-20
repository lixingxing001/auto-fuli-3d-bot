import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.experiment import build_cases, parse_int_list


class ExperimentTests(unittest.TestCase):
    def test_parse_int_list(self) -> None:
        self.assertEqual(parse_int_list("10, 20,30"), [10, 20, 30])

    def test_build_cases(self) -> None:
        cases = build_cases([10, 20], [300], [60, 120])
        self.assertEqual(len(cases), 4)
        self.assertEqual(cases[0].top_n, 10)
        self.assertEqual(cases[-1].recent_window, 120)


if __name__ == "__main__":
    unittest.main()
