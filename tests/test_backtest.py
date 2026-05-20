import unittest
from datetime import date, timedelta
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.backtest import run_backtest
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


class BacktestTests(unittest.TestCase):
    def test_backtest_returns_metrics(self) -> None:
        draws = make_draws(50)
        result = run_backtest(
            draws,
            top_n=10,
            training_window=30,
            config=StrategyConfig(min_history=20, recent_window=20),
        )
        self.assertEqual(result.rounds, 20)
        self.assertGreater(result.stake, 0)
        self.assertEqual(len(result.picks), 20)


if __name__ == "__main__":
    unittest.main()
