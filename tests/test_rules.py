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
from fuli3d_bot.rules import matches_filters, parse_rule_filters, rule_label, run_rule_recency
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


class RuleTests(unittest.TestCase):
    def test_parse_rule_filters(self) -> None:
        filters = parse_rule_filters("ones=5,sum=15,pattern=组六")
        self.assertEqual(rule_label(filters), "ones=5,sum=15,pattern=组六")
        self.assertTrue(matches_filters("915", filters))
        self.assertFalse(matches_filters("916", filters))

    def test_backtest_uses_variable_candidate_stake(self) -> None:
        def ranker(_draws, _top_n, _config):
            return ["001", "002"]

        result = run_backtest(
            make_draws(35),
            top_n=10,
            training_window=30,
            config=StrategyConfig(min_history=20, recent_window=20),
            ranker=ranker,
        )
        self.assertEqual(result.rounds, 5)
        self.assertEqual(result.stake, 20.0)
        self.assertAlmostEqual(result.expected_hits, 0.01)

    def test_rule_recency_returns_window_rows(self) -> None:
        rows = run_rule_recency(
            make_draws(40),
            rules=["pattern=zuliu"],
            windows=[5, 10],
            top_n=5,
            pool_size=50,
            training_window=30,
            recent_window=20,
            min_history=20,
        )
        self.assertEqual([row.window_size for row in rows], [5, 10])
        self.assertEqual(rows[0].rounds, 5)


if __name__ == "__main__":
    unittest.main()
