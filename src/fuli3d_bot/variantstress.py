from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .ablation import AblationCase, collect_case_picks, default_ablation_cases
from .calibration import CalibrationPick, summarize_top_hits
from .models import Draw


@dataclass(frozen=True)
class VariantStressRow:
    variant: str
    description: str
    segment: str
    year: int | None
    window_size: int | None
    top_n: int
    rounds: int
    start_issue: str
    end_issue: str
    hits: int
    expected_hits: float
    hit_lift: float
    hit_z_score: float
    roi: float
    pnl_vs_random_expected: float
    max_losing_streak: int


@dataclass(frozen=True)
class VariantStressSummary:
    best_variant: str
    best_top_n: int
    best_hit_z_score: float
    annual_failures: int
    recent_failures: int
    delivery_status: str
    verdict: str


@dataclass(frozen=True)
class VariantStressReport:
    summary: VariantStressSummary
    rows: list[VariantStressRow]


def parse_variant_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("variants cannot be empty")
    return names


def selected_cases(
    names: list[str],
    recent_window: int = 60,
    min_history: int = 300,
) -> list[AblationCase]:
    cases = {case.name: case for case in default_ablation_cases(recent_window, min_history)}
    missing = [name for name in names if name not in cases]
    if missing:
        raise ValueError(f"unknown variants: {missing}")
    return [cases[name] for name in names]


def _row_from_pick_subset(
    case: AblationCase,
    segment: str,
    year: int | None,
    window_size: int | None,
    top_n: int,
    picks: list[CalibrationPick],
) -> VariantStressRow:
    top_row = summarize_top_hits(picks, top_n=top_n)
    return VariantStressRow(
        variant=case.name,
        description=case.description,
        segment=segment,
        year=year,
        window_size=window_size,
        top_n=top_n,
        rounds=top_row.rounds,
        start_issue=top_row.start_issue,
        end_issue=top_row.end_issue,
        hits=top_row.hits,
        expected_hits=top_row.expected_hits,
        hit_lift=top_row.hit_lift,
        hit_z_score=top_row.hit_z_score,
        roi=top_row.roi,
        pnl_vs_random_expected=top_row.pnl_vs_random_expected,
        max_losing_streak=top_row.max_losing_streak,
    )


def _draw_year(draw: Draw) -> int | None:
    return draw.draw_date.year if draw.draw_date else None


def summarize_variant_stress(rows: list[VariantStressRow]) -> VariantStressSummary:
    all_rows = [row for row in rows if row.segment == "all"]
    ranked = sorted(
        all_rows,
        key=lambda row: (
            -row.hit_z_score,
            -row.pnl_vs_random_expected,
            row.max_losing_streak,
        ),
    )
    best = ranked[0] if ranked else None
    if best is None:
        return VariantStressSummary("", 0, 0.0, 0, 0, "analysis_only", "没有可评估结果。")

    annual_rows = [
        row for row in rows if row.segment == "year" and row.variant == best.variant and row.top_n == best.top_n
    ]
    recent_rows = [
        row
        for row in rows
        if row.segment == "recent" and row.variant == best.variant and row.top_n == best.top_n
    ]
    annual_failures = sum(1 for row in annual_rows if row.hit_z_score <= 0)
    recent_failures = sum(1 for row in recent_rows if row.hit_z_score <= 0)

    if best.hit_z_score >= 2.0 and annual_failures == 0 and recent_failures == 0:
        delivery_status = "recommendation_candidate"
        verdict = "最佳变体通过全局、年度和近期窗口初筛，可进入下一轮严格验收。"
    elif best.hit_z_score >= 2.0:
        delivery_status = "analysis_only"
        verdict = "最佳变体全局达标，但年度或近期窗口存在失败，应交付为分析观察工具。"
    else:
        delivery_status = "analysis_only"
        verdict = "没有变体达到全局 z>=2，应交付为分析观察工具。"

    return VariantStressSummary(
        best_variant=best.variant,
        best_top_n=best.top_n,
        best_hit_z_score=best.hit_z_score,
        annual_failures=annual_failures,
        recent_failures=recent_failures,
        delivery_status=delivery_status,
        verdict=verdict,
    )


def run_variant_stress(
    draws: list[Draw],
    variant_names: list[str],
    top_values: list[int],
    windows: list[int],
    years: list[int],
    training_window: int = 300,
    recent_window: int = 60,
    min_history: int = 300,
) -> VariantStressReport:
    eval_draws = draws[training_window:]
    cases = selected_cases(variant_names, recent_window=recent_window, min_history=min_history)
    unique_top_values = sorted(set(top_values))
    unique_windows = sorted(set(windows))
    unique_years = sorted(set(years))

    rows: list[VariantStressRow] = []
    for case in cases:
        picks = collect_case_picks(draws, case.config, training_window=training_window)
        for top_n in unique_top_values:
            rows.append(_row_from_pick_subset(case, "all", None, None, top_n, picks))
            for year in unique_years:
                year_picks = [
                    pick
                    for pick, draw in zip(picks, eval_draws)
                    if _draw_year(draw) == year
                ]
                if not year_picks:
                    continue
                rows.append(_row_from_pick_subset(case, "year", year, None, top_n, year_picks))
            for window_size in unique_windows:
                if window_size <= 0:
                    raise ValueError("window sizes must be positive")
                rows.append(
                    _row_from_pick_subset(
                        case,
                        "recent",
                        None,
                        window_size,
                        top_n,
                        picks[-window_size:],
                    )
                )

    rows.sort(
        key=lambda row: (
            0 if row.segment == "all" else 1 if row.segment == "year" else 2,
            row.top_n,
            row.year or 0,
            row.window_size or 0,
            -row.hit_z_score,
            row.variant,
        )
    )
    return VariantStressReport(summary=summarize_variant_stress(rows), rows=rows)


def save_variant_stress_reports(
    report: VariantStressReport,
    output_dir: str | Path,
    meta: dict,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "variantstress_report.json"
    md_path = report_dir / "variantstress_report.md"
    payload = {
        "meta": meta,
        "report": asdict(report),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_variant_stress_markdown(report, meta), encoding="utf-8")
    return json_path, md_path


def render_variant_stress_markdown(report: VariantStressReport, meta: dict) -> str:
    all_rows = [row for row in report.rows if row.segment == "all"]
    yearly_rows = [row for row in report.rows if row.segment == "year"]
    recent_rows = [row for row in report.rows if row.segment == "recent"]
    all_rows.sort(key=lambda row: (row.top_n, -row.hit_z_score, row.variant))
    yearly_rows.sort(key=lambda row: (row.top_n, row.year or 0, -row.hit_z_score, row.variant))
    recent_rows.sort(key=lambda row: (row.top_n, row.window_size or 0, -row.hit_z_score, row.variant))
    summary = report.summary

    lines = [
        "# 福彩3D变体压力测试报告",
        "",
        "## 参数",
        "",
        f"* 数据行数: {meta.get('draw_rows')}",
        f"* variants: {meta.get('variants')}",
        f"* top_values: {meta.get('top_values')}",
        f"* years: {meta.get('years')}",
        f"* windows: {meta.get('windows')}",
        f"* training_window: {meta.get('training_window')}",
        f"* recent_window: {meta.get('recent_window')}",
        "",
        "## 总结",
        "",
        f"* best_variant: {summary.best_variant}",
        f"* best_top_n: {summary.best_top_n}",
        f"* best_hit_z_score: {summary.best_hit_z_score:.3f}",
        f"* annual_failures: {summary.annual_failures}",
        f"* recent_failures: {summary.recent_failures}",
        f"* delivery_status: {summary.delivery_status}",
        f"* verdict: {summary.verdict}",
        "",
        "## 全局",
        "",
        "| top | variant | rounds | hits | exp_hits | lift | z | roi | pnl_vs_random | max_loss |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        lines.append(
            "| "
            f"{row.top_n} | {row.variant} | {row.rounds} | {row.hits} | "
            f"{row.expected_hits:.2f} | {row.hit_lift:.3f} | {row.hit_z_score:.3f} | "
            f"{row.roi:.2%} | {row.pnl_vs_random_expected:.2f} | {row.max_losing_streak} |"
        )

    lines.extend(
        [
            "",
            "## 年度",
            "",
            "| top | year | variant | rounds | hits | exp_hits | lift | z | roi | max_loss |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in yearly_rows:
        lines.append(
            "| "
            f"{row.top_n} | {row.year} | {row.variant} | {row.rounds} | {row.hits} | "
            f"{row.expected_hits:.2f} | {row.hit_lift:.3f} | {row.hit_z_score:.3f} | "
            f"{row.roi:.2%} | {row.max_losing_streak} |"
        )

    lines.extend(
        [
            "",
            "## 近期",
            "",
            "| top | window | variant | rounds | hits | exp_hits | lift | z | roi | max_loss |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in recent_rows:
        lines.append(
            "| "
            f"{row.top_n} | {row.window_size} | {row.variant} | {row.rounds} | {row.hits} | "
            f"{row.expected_hits:.2f} | {row.hit_lift:.3f} | {row.hit_z_score:.3f} | "
            f"{row.roi:.2%} | {row.max_losing_streak} |"
        )

    lines.extend(
        [
            "",
            "## 判读",
            "",
            "* 全局 z>=2 只是入场条件，年度和近期窗口失败会降级为观察。",
            "* max_loss 很长时，即使命中统计占优，也不能作为稳定推荐。",
            "* delivery_status 为 analysis_only 时，交付重点应放在数据、报告和闸门。",
        ]
    )
    return "\n".join(lines) + "\n"
