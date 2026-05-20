import unittest
from datetime import date, timedelta
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.gate import (  # noqa: E402
    ACTIVE,
    BLOCKED,
    WATCH,
    RuleGateThresholds,
    build_gated_candidates,
    evaluate_rule_gate,
)
from fuli3d_bot.models import Draw  # noqa: E402
from fuli3d_bot.rules import RuleRecencyRow  # noqa: E402
from fuli3d_bot.strategy import StrategyConfig  # noqa: E402


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


def make_recency_row(rule: str, window_size: int, hits: int, lift: float, roi: float) -> RuleRecencyRow:
    stake = 100.0
    return RuleRecencyRow(
        rule=rule,
        window_size=window_size,
        start_issue=f"2026{window_size:03d}",
        end_issue=f"2026{window_size + 1:03d}",
        top_n=20,
        pool_size=200,
        training_window=300,
        recent_window=60,
        rounds=window_size,
        avg_candidates=3.0,
        empty_rounds=0,
        hits=hits,
        expected_hits=hits / lift if lift else 1.0,
        hit_rate=hits / window_size,
        hit_lift=lift,
        hit_z_score=1.0 if lift >= 1 else -1.0,
        stake=stake,
        pnl=stake * roi,
        roi=roi,
        expected_roi=-0.48,
        pnl_vs_random_expected=0.0,
        max_losing_streak=window_size,
    )


class GateTests(unittest.TestCase):
    def test_evaluate_rule_gate_classifies_rules(self) -> None:
        rows = [
            make_recency_row("active_rule", 60, 1, 1.20, -0.20),
            make_recency_row("active_rule", 120, 2, 1.30, 0.10),
            make_recency_row("watch_rule", 60, 1, 1.10, -0.30),
            make_recency_row("watch_rule", 120, 0, 0.00, -1.00),
            make_recency_row("blocked_rule", 60, 0, 0.00, -1.00),
            make_recency_row("blocked_rule", 120, 0, 0.00, -1.00),
        ]
        statuses = evaluate_rule_gate(
            rows,
            gate_windows=[60, 120],
            thresholds=RuleGateThresholds(min_hits=1, min_lift=1.0, min_roi=-0.48),
        )
        by_rule = {item.rule: item for item in statuses}

        self.assertEqual(by_rule["active_rule"].status, ACTIVE)
        self.assertEqual(by_rule["active_rule"].passed_windows, [60, 120])
        self.assertEqual(by_rule["watch_rule"].status, WATCH)
        self.assertEqual(by_rule["watch_rule"].passed_windows, [60])
        self.assertEqual(by_rule["blocked_rule"].status, BLOCKED)
        self.assertEqual(by_rule["blocked_rule"].failed_windows, [60, 120])

    def test_build_gated_candidates_filters_ranked_pool(self) -> None:
        candidates = build_gated_candidates(
            make_draws(40),
            active_rules=["pattern=zuliu"],
            top_n=5,
            pool_size=1000,
            config=StrategyConfig(min_history=20, recent_window=20),
        )

        self.assertEqual(len(candidates), 5)
        self.assertTrue(all(item.pattern == "组六" for item in candidates))
        self.assertTrue(all(item.matched_rules == ["pattern=组六"] for item in candidates))

    def test_build_gated_candidates_returns_empty_without_active_rules(self) -> None:
        candidates = build_gated_candidates(
            make_draws(40),
            active_rules=[],
            top_n=5,
            pool_size=1000,
            config=StrategyConfig(min_history=20, recent_window=20),
        )

        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
