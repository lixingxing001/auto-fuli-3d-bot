import unittest
from datetime import date, timedelta
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.baselines import deterministic_random_ranker, position_hot_ranker
from fuli3d_bot.benchmark import draw_months
from fuli3d_bot.models import Draw
from fuli3d_bot.strategy import StrategyConfig


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


class BaselineTests(unittest.TestCase):
    def test_position_hot_ranker_returns_top_n(self) -> None:
        picks = position_hot_ranker(make_draws(40), 7, StrategyConfig(recent_window=20))
        self.assertEqual(len(picks), 7)
        self.assertEqual(len(set(picks)), 7)

    def test_random_ranker_is_deterministic(self) -> None:
        draws = make_draws(40)
        config = StrategyConfig(recent_window=20)
        first = deterministic_random_ranker(draws, 10, config)
        second = deterministic_random_ranker(draws, 10, config)
        self.assertEqual(first, second)

    def test_draw_months_filters_year(self) -> None:
        draws = make_draws(40)
        self.assertEqual(draw_months(draws, 2026), [(2026, 1), (2026, 2)])


if __name__ == "__main__":
    unittest.main()
