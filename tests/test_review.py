import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.models import Draw  # noqa: E402
from fuli3d_bot.review import build_review_report, save_review_report  # noqa: E402


def write_snapshot(
    path: Path,
    issue: str,
    number: str,
    alternatives: list[str] | None = None,
    full_ranking: list[dict] | None = None,
) -> None:
    ranked_numbers = [number, *(alternatives or [])]
    saved_ranking = full_ranking or [
        {"rank": index + 1, "number": item, "score": 1.0 / (index + 1)}
        for index, item in enumerate(ranked_numbers)
    ]
    payload = {
        "meta": {"draw_rows": 10},
        "report": {
            "latest_issue": "2026129",
            "latest_date": "2026-05-19",
            "next_issue_hint": issue,
            "primary": {"number": number},
            "alternatives": [{"number": item} for item in (alternatives or [])],
            "full_ranking": saved_ranking,
            "action_filter": {"label": "可小注", "stake_level": "低"},
            "confidence_gate": {"label": "可关注"},
            "strategy_selection": {"active_label": "测试策略"},
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class ReviewTests(unittest.TestCase):
    def test_review_marks_direct_and_group_hits(self) -> None:
        with TemporaryDirectory() as temp_dir:
            predictions = Path(temp_dir) / "snapshots"
            predictions.mkdir()
            write_snapshot(predictions / "prediction_2026130.json", "2026130", "812", ["888", "818"])
            draws = [Draw("2026130", date(2026, 5, 20), "812")]

            report = build_review_report(draws, predictions)

            self.assertEqual(report.summary.reviewed, 1)
            self.assertEqual(report.summary.direct_hits, 1)
            self.assertEqual(report.summary.group_hits, 1)
            self.assertEqual(report.summary.ranked, 1)
            self.assertEqual(report.summary.mean_actual_rank, 1.0)
            self.assertEqual(report.summary.top100_count, 1)
            self.assertEqual(report.rows[0].actual_rank, 1)
            self.assertTrue(report.rows[0].top3_hit)

    def test_review_tracks_actual_rank_for_misses(self) -> None:
        with TemporaryDirectory() as temp_dir:
            predictions = Path(temp_dir) / "snapshots"
            predictions.mkdir()
            write_snapshot(
                predictions / "prediction_2026130.json",
                "2026130",
                "812",
                ["888", "818"],
                full_ranking=[
                    {"rank": 1, "number": "812", "score": 1.0},
                    {"rank": 750, "number": "267", "score": -0.2},
                ],
            )
            draws = [Draw("2026130", date(2026, 5, 20), "267")]

            report = build_review_report(draws, predictions)

            self.assertEqual(report.rows[0].status, "miss")
            self.assertEqual(report.rows[0].actual_rank, 750)
            self.assertEqual(report.rows[0].actual_score, -0.2)
            self.assertEqual(report.summary.mean_actual_rank, 750.0)
            self.assertEqual(report.summary.median_actual_rank, 750.0)
            self.assertEqual(report.summary.top500_count, 0)
            self.assertEqual(report.summary.top500_rate, 0.0)

    def test_review_keeps_pending_when_actual_missing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            predictions = Path(temp_dir) / "snapshots"
            predictions.mkdir()
            write_snapshot(predictions / "prediction_2026130.json", "2026130", "812")

            report = build_review_report([], predictions)

            self.assertEqual(report.summary.pending, 1)
            self.assertEqual(report.rows[0].status, "pending")

    def test_save_review_report_outputs_json_and_html(self) -> None:
        with TemporaryDirectory() as temp_dir:
            predictions = Path(temp_dir) / "snapshots"
            output = Path(temp_dir) / "review"
            predictions.mkdir()
            write_snapshot(predictions / "prediction_2026130.json", "2026130", "812")
            report = build_review_report([], predictions)

            json_path, html_path = save_review_report(report, output, {"predictions_dir": str(predictions)})

            self.assertTrue(json_path.exists())
            self.assertTrue(html_path.exists())


if __name__ == "__main__":
    unittest.main()
