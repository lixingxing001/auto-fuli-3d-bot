import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.daily import (  # noqa: E402
    PlayRow,
    PlayStats,
    StrategyCandidate,
    StrategySelection,
    _play_meta,
    _play_payout,
    _play_rows_html,
    build_action_filter,
    _strategy_variants,
    build_confidence_gate,
    build_daily_report,
    choose_strategy_candidate,
    save_daily_report,
    select_daily_strategy,
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


def make_stats(
    rounds: int,
    hits: int,
    expected_hits: float,
    roi: float = -0.5,
) -> PlayStats:
    return PlayStats(
        rounds=rounds,
        hits=hits,
        hit_rate=hits / rounds if rounds else 0.0,
        expected_hits=expected_hits,
        expected_hit_rate=expected_hits / rounds if rounds else 0.0,
        hit_lift=hits / expected_hits if expected_hits else 0.0,
        stake=float(rounds * 2),
        payout=0.0,
        pnl=0.0,
        roi=roi,
    )


def make_gate_row(play_id: str, all_stats: PlayStats, recent60: PlayStats, recent120: PlayStats) -> PlayRow:
    return PlayRow(
        play_id=play_id,
        name=play_id,
        selection="812",
        active=True,
        order_required="测试",
        stake=2.0,
        prize_text="测试",
        net_profit_text="测试",
        theoretical_hit_rate=0.001,
        expected_roi=-0.48,
        all_stats=all_stats,
        recent60_stats=recent60,
        recent120_stats=recent120,
        note="测试",
    )


def make_strategy_candidate(name: str, status: str, z: float) -> StrategyCandidate:
    return StrategyCandidate(
        name=name,
        label=name,
        status=status,
        reason="测试",
        primary_number="812",
        top_n=10,
        rounds=100,
        hits=2,
        expected_hits=1.0,
        hit_rate=0.02,
        hit_lift=2.0,
        hit_z_score=z,
        roi=-0.1,
        recent60_hits=1,
        recent120_hits=1,
        annual_failures=0,
    )


def make_strategy_selection(status: str) -> StrategySelection:
    return StrategySelection(
        active_name="baseline",
        active_label="默认策略",
        status=status,
        cache_status="disabled",
        summary="测试",
        candidates=[make_strategy_candidate("baseline", "blocked", 0.5)],
    )


class DailyTests(unittest.TestCase):
    def test_group_meta_uses_number_pattern(self) -> None:
        zuliu = _play_meta("group", "812")
        zusan = _play_meta("group", "811")
        baozi = _play_meta("group", "888")

        self.assertEqual(zuliu["name"], "组选6")
        self.assertEqual(zusan["name"], "组选3")
        self.assertFalse(baozi["active"])

    def test_play_payout_for_direct_and_group(self) -> None:
        self.assertEqual(_play_payout("direct", "812", "812"), (True, 1040.0))
        self.assertEqual(_play_payout("direct", "812", "281"), (False, 0.0))
        self.assertEqual(_play_payout("group", "812", "281"), (True, 173.0))

    def test_build_daily_report_returns_rows(self) -> None:
        report = build_daily_report(
            make_draws(35),
            top_n=5,
            training_window=30,
            recent_window=20,
            min_history=20,
        )

        self.assertTrue(report.primary.number)
        self.assertEqual(len(report.alternatives), 4)
        self.assertTrue(any(row.play_id == "direct" for row in report.play_rows))
        self.assertTrue(report.confidence_gate.label)
        self.assertTrue(report.action_filter.label)
        self.assertTrue(report.strategy_selection.active_name)

    def test_save_daily_report_writes_snapshot(self) -> None:
        report = build_daily_report(
            make_draws(35),
            top_n=5,
            training_window=30,
            recent_window=20,
            min_history=20,
        )
        with TemporaryDirectory() as temp_dir:
            save_daily_report(report, temp_dir, {"draw_rows": 35})
            snapshot = Path(temp_dir) / "snapshots" / f"prediction_{report.next_issue_hint}.json"

            self.assertTrue(snapshot.exists())

    def test_confidence_gate_blocks_when_direct_recent_fails(self) -> None:
        rows = [
            make_gate_row("direct", make_stats(900, 2, 0.9), make_stats(60, 0, 0.06), make_stats(120, 0, 0.12)),
            make_gate_row("group", make_stats(900, 4, 5.4), make_stats(60, 0, 0.36), make_stats(120, 0, 0.72)),
            make_gate_row("package", make_stats(900, 4, 5.4), make_stats(60, 0, 0.36), make_stats(120, 0, 0.72)),
        ]

        gate = build_confidence_gate(rows)

        self.assertEqual(gate.status, "blocked")
        self.assertEqual(gate.label, "低置信")

    def test_confidence_gate_watch_when_direct_passes(self) -> None:
        rows = [
            make_gate_row("direct", make_stats(900, 3, 0.9), make_stats(60, 1, 0.06), make_stats(120, 1, 0.12)),
            make_gate_row("group", make_stats(900, 4, 5.4), make_stats(60, 0, 0.36), make_stats(120, 0, 0.72)),
            make_gate_row("package", make_stats(900, 4, 5.4), make_stats(60, 0, 0.36), make_stats(120, 0, 0.72)),
        ]

        gate = build_confidence_gate(rows)

        self.assertEqual(gate.status, "watch")
        self.assertEqual(gate.confidence, "中")

    def test_strategy_choice_prefers_ready_candidate(self) -> None:
        selected = choose_strategy_candidate(
            [
                make_strategy_candidate("baseline", "blocked", 0.5),
                make_strategy_candidate("no_repeat_penalty", "ready", 2.5),
            ]
        )

        self.assertEqual(selected.name, "no_repeat_penalty")

    def test_strategy_choice_falls_back_to_baseline(self) -> None:
        selected = choose_strategy_candidate(
            [
                make_strategy_candidate("baseline", "blocked", 0.5),
                make_strategy_candidate("no_repeat_penalty", "watch", 2.5),
            ]
        )

        self.assertEqual(selected.name, "baseline")

    def test_strategy_variant_pool_has_distinct_feature_sets(self) -> None:
        names = [variant.name for variant in _strategy_variants(recent_window=20, min_history=20)]

        self.assertIn("baseline", names)
        self.assertIn("no_repeat_penalty", names)
        self.assertIn("recent_position_only", names)
        self.assertIn("shape_stack", names)
        self.assertGreaterEqual(len(names), 8)

    def test_action_filter_allows_small_action_when_all_gates_pass(self) -> None:
        rows = [
            make_gate_row("direct", make_stats(900, 3, 0.9), make_stats(60, 1, 0.06), make_stats(120, 1, 0.12)),
            make_gate_row("group", make_stats(900, 4, 5.4), make_stats(60, 0, 0.36), make_stats(120, 0, 0.72)),
            make_gate_row("package", make_stats(900, 4, 5.4), make_stats(60, 0, 0.36), make_stats(120, 0, 0.72)),
        ]
        gate = build_confidence_gate(rows)
        action = build_action_filter(gate, make_strategy_selection("active"), rows)

        self.assertEqual(action.status, "action")
        self.assertEqual(action.label, "可小注")

    def test_action_filter_observes_when_only_auxiliary_passes(self) -> None:
        rows = [
            make_gate_row("direct", make_stats(900, 2, 0.9), make_stats(60, 0, 0.06), make_stats(120, 0, 0.12)),
            make_gate_row("group", make_stats(900, 6, 5.4), make_stats(60, 1, 0.36), make_stats(120, 1, 0.72)),
            make_gate_row("package", make_stats(900, 6, 5.4, roi=0.1), make_stats(60, 1, 0.36), make_stats(120, 1, 0.72)),
        ]
        gate = build_confidence_gate(rows)
        action = build_action_filter(gate, make_strategy_selection("fallback"), rows)

        self.assertEqual(action.status, "observe")
        self.assertEqual(action.stake_level, "零")

    def test_action_filter_skips_when_no_signal_passes(self) -> None:
        rows = [
            make_gate_row("direct", make_stats(900, 0, 0.9), make_stats(60, 0, 0.06), make_stats(120, 0, 0.12)),
            make_gate_row("group", make_stats(900, 0, 5.4), make_stats(60, 0, 0.36), make_stats(120, 0, 0.72)),
            make_gate_row("package", make_stats(900, 0, 5.4), make_stats(60, 0, 0.36), make_stats(120, 0, 0.72)),
        ]
        gate = build_confidence_gate(rows)
        action = build_action_filter(gate, make_strategy_selection("fallback"), rows)

        self.assertEqual(action.status, "skip")
        self.assertEqual(action.label, "不出手")

    def test_strategy_cache_roundtrip(self) -> None:
        draws = make_draws(40)
        with TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "strategy_cache.json"
            first_selection, _first_config, first_picks = select_daily_strategy(
                draws,
                top_n=5,
                training_window=30,
                recent_window=20,
                min_history=20,
                limit_rows=40,
                cache_path=cache_path,
            )
            second_selection, _second_config, second_picks = select_daily_strategy(
                draws,
                top_n=5,
                training_window=30,
                recent_window=20,
                min_history=20,
                limit_rows=40,
                cache_path=cache_path,
            )

            self.assertEqual(first_selection.cache_status, "miss")
            self.assertEqual(second_selection.cache_status, "hit")
            self.assertEqual(first_selection.active_name, second_selection.active_name)
            self.assertEqual(len(first_picks), len(second_picks))
            self.assertTrue(cache_path.exists())

    def test_inactive_play_row_hides_backtest_stats(self) -> None:
        stats = PlayStats(
            rounds=10,
            hits=3,
            hit_rate=0.3,
            expected_hits=1.0,
            expected_hit_rate=0.1,
            hit_lift=3.0,
            stake=20.0,
            payout=30.0,
            pnl=10.0,
            roi=0.5,
        )
        html = _play_rows_html(
            [
                PlayRow(
                    play_id="big_small",
                    name="猜大小",
                    selection="和值9到18无大小投注",
                    active=False,
                    order_required="不要求顺序",
                    stake=2.0,
                    prize_text="不适用",
                    net_profit_text="不适用",
                    theoretical_hit_rate=0.0,
                    expected_roi=0.0,
                    all_stats=stats,
                    recent60_stats=stats,
                    recent120_stats=stats,
                    note="测试",
                )
            ]
        )

        self.assertIn("本期不适用", html)
        self.assertNotIn("30.00%", html)
        self.assertNotIn("50.00%", html)


if __name__ == "__main__":
    unittest.main()
