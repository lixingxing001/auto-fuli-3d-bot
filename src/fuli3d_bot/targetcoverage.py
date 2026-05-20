from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path

from .calibration import CalibrationTopRow, collect_calibration_picks, summarize_top_hits
from .models import Draw


RANK_TOTAL = 1000


@dataclass(frozen=True)
class TargetCoverageSummary:
    target_rate: float
    required_top_n: int
    theoretical_hit_rate: float
    stake_per_draw: float
    expected_payout_per_draw: float
    expected_loss_per_draw: float
    expected_roi: float
    feasibility_status: str
    verdict: str


@dataclass(frozen=True)
class TargetCoverageReport:
    summary: TargetCoverageSummary
    rows: list[CalibrationTopRow]


def required_top_n(target_rate: float) -> int:
    if not 0 < target_rate <= 1:
        raise ValueError("target_rate must be greater than 0 and no more than 1")
    return min(RANK_TOTAL, ceil(target_rate * RANK_TOTAL))


def summarize_target(
    target_rate: float,
    stake_per_number: float,
    payout_per_hit: float,
) -> TargetCoverageSummary:
    top_n = required_top_n(target_rate)
    theoretical_hit_rate = top_n / RANK_TOTAL
    stake_per_draw = top_n * stake_per_number
    expected_payout_per_draw = theoretical_hit_rate * payout_per_hit
    expected_loss_per_draw = stake_per_draw - expected_payout_per_draw
    expected_roi = (expected_payout_per_draw - stake_per_draw) / stake_per_draw
    if top_n >= 500:
        feasibility_status = "coverage_only"
        verdict = (
            f"目标可通过覆盖{top_n}个号码实现，但这是扩大覆盖面，"
            "没有提供预测优势，期望收益仍为负。"
        )
    else:
        feasibility_status = "model_target"
        verdict = "目标候选数量低于一半号码，需要模型提供真实排序优势。"

    return TargetCoverageSummary(
        target_rate=target_rate,
        required_top_n=top_n,
        theoretical_hit_rate=theoretical_hit_rate,
        stake_per_draw=stake_per_draw,
        expected_payout_per_draw=expected_payout_per_draw,
        expected_loss_per_draw=expected_loss_per_draw,
        expected_roi=expected_roi,
        feasibility_status=feasibility_status,
        verdict=verdict,
    )


def run_target_coverage(
    draws: list[Draw],
    target_rate: float = 0.65,
    compare_top_values: list[int] | None = None,
    windows: list[int] | None = None,
    training_window: int = 300,
    recent_window: int = 60,
    min_history: int = 300,
    stake_per_number: float = 2.0,
    payout_per_hit: float = 1040.0,
) -> TargetCoverageReport:
    summary = summarize_target(target_rate, stake_per_number, payout_per_hit)
    top_values = set(compare_top_values or [10, 20, 50, 100, 200, 500])
    top_values.add(summary.required_top_n)
    unique_top_values = sorted(top_values)
    unique_windows = sorted(set(windows or [60, 120, 240, 360]))

    picks = collect_calibration_picks(
        draws,
        training_window=training_window,
        recent_window=recent_window,
        min_history=min_history,
    )
    rows: list[CalibrationTopRow] = []
    for top_n in unique_top_values:
        rows.append(
            summarize_top_hits(
                picks,
                top_n=top_n,
                stake_per_number=stake_per_number,
                payout_per_hit=payout_per_hit,
            )
        )
        for window_size in unique_windows:
            rows.append(
                summarize_top_hits(
                    picks,
                    top_n=top_n,
                    window_size=window_size,
                    stake_per_number=stake_per_number,
                    payout_per_hit=payout_per_hit,
                )
            )

    rows.sort(key=lambda row: (0 if row.segment == "all" else 1, row.top_n, row.window_size or 0))
    return TargetCoverageReport(summary=summary, rows=rows)


def save_target_coverage_reports(
    report: TargetCoverageReport,
    output_dir: str | Path,
    meta: dict,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "targetcoverage_report.json"
    md_path = report_dir / "targetcoverage_report.md"
    payload = {
        "meta": meta,
        "report": asdict(report),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_target_coverage_markdown(report, meta), encoding="utf-8")
    return json_path, md_path


def render_target_coverage_markdown(report: TargetCoverageReport, meta: dict) -> str:
    summary = report.summary
    all_rows = [row for row in report.rows if row.segment == "all"]
    recent_rows = [row for row in report.rows if row.segment == "recent"]
    all_rows.sort(key=lambda row: row.top_n)
    recent_rows.sort(key=lambda row: (row.window_size or 0, row.top_n))

    lines = [
        "# 福彩3D目标覆盖率报告",
        "",
        "## 参数",
        "",
        f"* 数据行数: {meta.get('draw_rows')}",
        f"* target_rate: {summary.target_rate:.2%}",
        f"* required_top_n: {summary.required_top_n}",
        f"* training_window: {meta.get('training_window')}",
        f"* recent_window: {meta.get('recent_window')}",
        "",
        "## 目标测算",
        "",
        f"* theoretical_hit_rate: {summary.theoretical_hit_rate:.2%}",
        f"* stake_per_draw: {summary.stake_per_draw:.2f}",
        f"* expected_payout_per_draw: {summary.expected_payout_per_draw:.2f}",
        f"* expected_loss_per_draw: {summary.expected_loss_per_draw:.2f}",
        f"* expected_roi: {summary.expected_roi:.2%}",
        f"* feasibility_status: {summary.feasibility_status}",
        f"* verdict: {summary.verdict}",
        "",
        "## 全局覆盖结果",
        "",
        "| top | rounds | hits | hit_rate | exp_hits | expected_rate | lift | z | stake | pnl | roi |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        lines.append(
            "| "
            f"{row.top_n} | {row.rounds} | {row.hits} | {row.hit_rate:.2%} | "
            f"{row.expected_hits:.2f} | {row.expected_hit_rate:.2%} | "
            f"{row.hit_lift:.3f} | {row.hit_z_score:.3f} | {row.stake:.2f} | "
            f"{row.pnl:.2f} | {row.roi:.2%} |"
        )

    lines.extend(
        [
            "",
            "## 近期覆盖结果",
            "",
            "| window | top | rounds | hits | hit_rate | expected_rate | lift | z | roi |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in recent_rows:
        lines.append(
            "| "
            f"{row.window_size} | {row.top_n} | {row.rounds} | {row.hits} | "
            f"{row.hit_rate:.2%} | {row.expected_hit_rate:.2%} | "
            f"{row.hit_lift:.3f} | {row.hit_z_score:.3f} | {row.roi:.2%} |"
        )

    lines.extend(
        [
            "",
            "## 判读",
            "",
            "* 65% 命中率可以靠覆盖 650 个号码达到，数学期望仍然亏损。",
            "* top 越大，命中率越接近覆盖比例，预测含量越低。",
            "* 这个报告用于拆穿目标口径，不能作为实盘或购彩建议。",
        ]
    )
    return "\n".join(lines) + "\n"
