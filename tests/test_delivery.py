import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fuli3d_bot.delivery import build_delivery_report  # noqa: E402
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


class DeliveryTests(unittest.TestCase):
    def test_build_delivery_report_marks_analysis_only_when_reports_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_delivery_report(make_draws(300), reports_dir=temp_dir)

        self.assertEqual(report.mode, "analysis_only")
        self.assertFalse(report.can_recommend)
        self.assertTrue(any(item.status == "missing" for item in report.checks))

    def test_build_delivery_report_can_recommend_when_checks_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "gate").mkdir()
            (root / "calibration").mkdir()
            (root / "ablation").mkdir()
            (root / "variantstress").mkdir()
            (root / "gate" / "gate_report.json").write_text(
                json.dumps({"report": {"active_rules": ["sum=15"], "recommendation_mode": "rule_gated"}}),
                encoding="utf-8",
            )
            (root / "calibration" / "calibration_report.json").write_text(
                json.dumps(
                    {
                        "report": {
                            "summary": {
                                "signal_status": "strong",
                                "mean_actual_rank": 450.0,
                                "top_bucket_lift": 1.3,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "ablation" / "ablation_report.json").write_text(
                json.dumps(
                    {
                        "report": {
                            "summary": {
                                "recommendation_status": "candidate",
                                "best_case": "baseline",
                                "best_top_n": 10,
                                "best_hit_z_score": 2.2,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "variantstress" / "variantstress_report.json").write_text(
                json.dumps(
                    {
                        "report": {
                            "summary": {
                                "delivery_status": "recommendation_candidate",
                                "best_variant": "baseline",
                                "best_top_n": 10,
                                "annual_failures": 0,
                                "recent_failures": 0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = build_delivery_report(make_draws(300), reports_dir=root)

        self.assertEqual(report.mode, "recommendation_ready")
        self.assertTrue(report.can_recommend)


if __name__ == "__main__":
    unittest.main()
