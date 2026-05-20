from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .calibration import CalibrationPick, summarize_top_hits
from .models import Draw
from .strategy import StrategyConfig, rank_numbers


@dataclass(frozen=True)
class AblationCase:
    name: str
    description: str
    config: StrategyConfig


@dataclass(frozen=True)
class AblationRow:
    case: str
    description: str
    segment: str
    window_size: int | None
    top_n: int
    rounds: int
    hits: int
    expected_hits: float
    hit_lift: float
    hit_z_score: float
    roi: float
    pnl_vs_random_expected: float
    max_losing_streak: int


@dataclass(frozen=True)
class AblationSummary:
    best_case: str
    best_top_n: int
    best_hit_z_score: float
    baseline_top20_z: float
    recommendation_status: str
    verdict: str


@dataclass(frozen=True)
class AblationReport:
    summary: AblationSummary
    rows: list[AblationRow]


def _zero_config(base: StrategyConfig) -> StrategyConfig:
    return replace(
        base,
        position_weight=0.0,
        recent_position_weight=0.0,
        sum_weight=0.0,
        span_weight=0.0,
        pattern_weight=0.0,
        omission_weight=0.0,
        repeat_penalty=0.0,
    )


def default_ablation_cases(recent_window: int = 60, min_history: int = 300) -> list[AblationCase]:
    base = StrategyConfig(recent_window=recent_window, min_history=min_history)
    zero = _zero_config(base)
    return [
        AblationCase("baseline", "当前默认权重", base),
        AblationCase(
            "no_long_position",
            "移除长期位置频率",
            replace(base, position_weight=0.0),
        ),
        AblationCase(
            "no_recent_position",
            "移除近期位置热度",
            replace(base, recent_position_weight=0.0),
        ),
        AblationCase("no_sum", "移除和值分布", replace(base, sum_weight=0.0)),
        AblationCase("no_span", "移除跨度分布", replace(base, span_weight=0.0)),
        AblationCase("no_pattern", "移除形态分布", replace(base, pattern_weight=0.0)),
        AblationCase("no_omission", "移除遗漏分散", replace(base, omission_weight=0.0)),
        AblationCase("no_repeat_penalty", "移除近期重号惩罚", replace(base, repeat_penalty=0.0)),
        AblationCase(
            "position_only",
            "仅保留长期位置频率",
            replace(zero, position_weight=base.position_weight),
        ),
        AblationCase(
            "recent_position_only",
            "仅保留近期位置热度",
            replace(zero, recent_position_weight=base.recent_position_weight),
        ),
        AblationCase("sum_only", "仅保留和值分布", replace(zero, sum_weight=base.sum_weight)),
        AblationCase("span_only", "仅保留跨度分布", replace(zero, span_weight=base.span_weight)),
        AblationCase(
            "pattern_only",
            "仅保留形态分布",
            replace(zero, pattern_weight=base.pattern_weight),
        ),
        AblationCase(
            "omission_only",
            "仅保留遗漏分散",
            replace(zero, omission_weight=base.omission_weight),
        ),
        AblationCase(
            "position_stack",
            "仅保留长期和近期位置",
            replace(
                zero,
                position_weight=base.position_weight,
                recent_position_weight=base.recent_position_weight,
            ),
        ),
        AblationCase(
            "shape_stack",
            "仅保留和值跨度形态",
            replace(
                zero,
                sum_weight=base.sum_weight,
                span_weight=base.span_weight,
                pattern_weight=base.pattern_weight,
            ),
        ),
    ]


def collect_case_picks(
    draws: list[Draw],
    config: StrategyConfig,
    training_window: int = 300,
) -> list[CalibrationPick]:
    if len(draws) <= training_window:
        raise ValueError(
            f"need more draw rows than training_window, got {len(draws)} rows and window {training_window}"
        )
    active_config = replace(config, min_history=min(config.min_history, training_window))
    picks: list[CalibrationPick] = []
    for index in range(training_window, len(draws)):
        training_draws = draws[index - training_window : index]
        actual = draws[index]
        ranked = rank_numbers(training_draws, top_n=1000, config=active_config)
        match = next(item for item in ranked if item.number == actual.number)
        picks.append(
            CalibrationPick(
                issue=actual.issue,
                draw_date=actual.draw_date.isoformat() if actual.draw_date else None,
                actual_number=actual.number,
                actual_rank=match.rank,
                actual_score=match.score,
            )
        )
    return picks


def _row_from_top(case: AblationCase, top_row) -> AblationRow:
    return AblationRow(
        case=case.name,
        description=case.description,
        segment=top_row.segment,
        window_size=top_row.window_size,
        top_n=top_row.top_n,
        rounds=top_row.rounds,
        hits=top_row.hits,
        expected_hits=top_row.expected_hits,
        hit_lift=top_row.hit_lift,
        hit_z_score=top_row.hit_z_score,
        roi=top_row.roi,
        pnl_vs_random_expected=top_row.pnl_vs_random_expected,
        max_losing_streak=top_row.max_losing_streak,
    )


def summarize_ablation(rows: list[AblationRow]) -> AblationSummary:
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
    baseline_top20 = next(
        (row for row in all_rows if row.case == "baseline" and row.top_n == 20),
        None,
    )
    baseline_z = baseline_top20.hit_z_score if baseline_top20 else 0.0

    if best and best.hit_z_score >= 2.0:
        recommendation_status = "candidate"
        verdict = "存在达到 z>=2 的消融方案，需要继续做年度和近期窗口复核。"
    elif best and best.hit_z_score > 0:
        recommendation_status = "analysis_only"
        verdict = "消融只能找到弱信号，当前应交付为分析观察工具。"
    else:
        recommendation_status = "analysis_only"
        verdict = "消融没有找到有效排序信号，当前应冻结推荐输出。"

    return AblationSummary(
        best_case=best.case if best else "",
        best_top_n=best.top_n if best else 0,
        best_hit_z_score=best.hit_z_score if best else 0.0,
        baseline_top20_z=baseline_z,
        recommendation_status=recommendation_status,
        verdict=verdict,
    )


def run_ablation(
    draws: list[Draw],
    top_values: list[int],
    windows: list[int],
    training_window: int = 300,
    recent_window: int = 60,
    min_history: int = 300,
) -> AblationReport:
    cases = default_ablation_cases(recent_window=recent_window, min_history=min_history)
    unique_top_values = sorted(set(top_values))
    unique_windows = sorted(set(windows))
    rows: list[AblationRow] = []
    for case in cases:
        picks = collect_case_picks(draws, config=case.config, training_window=training_window)
        for top_n in unique_top_values:
            rows.append(_row_from_top(case, summarize_top_hits(picks, top_n=top_n)))
            for window_size in unique_windows:
                rows.append(
                    _row_from_top(
                        case,
                        summarize_top_hits(picks, top_n=top_n, window_size=window_size),
                    )
                )

    rows.sort(
        key=lambda row: (
            0 if row.segment == "all" else 1,
            row.top_n,
            -row.hit_z_score,
            row.case,
            row.window_size or 0,
        )
    )
    return AblationReport(summary=summarize_ablation(rows), rows=rows)


def save_ablation_reports(
    report: AblationReport,
    output_dir: str | Path,
    meta: dict,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "ablation_report.json"
    md_path = report_dir / "ablation_report.md"
    payload = {
        "meta": meta,
        "report": asdict(report),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_ablation_markdown(report, meta), encoding="utf-8")
    return json_path, md_path


def render_ablation_markdown(report: AblationReport, meta: dict) -> str:
    all_rows = [row for row in report.rows if row.segment == "all"]
    recent_rows = [row for row in report.rows if row.segment == "recent"]
    all_rows.sort(key=lambda row: (row.top_n, -row.hit_z_score, row.case))
    recent_rows.sort(key=lambda row: (row.window_size or 0, row.top_n, -row.hit_z_score, row.case))
    summary = report.summary

    lines = [
        "# 福彩3D权重消融报告",
        "",
        "## 参数",
        "",
        f"* 数据行数: {meta.get('draw_rows')}",
        f"* top_values: {meta.get('top_values')}",
        f"* windows: {meta.get('windows')}",
        f"* training_window: {meta.get('training_window')}",
        f"* recent_window: {meta.get('recent_window')}",
        "",
        "## 总结",
        "",
        f"* best_case: {summary.best_case}",
        f"* best_top_n: {summary.best_top_n}",
        f"* best_hit_z_score: {summary.best_hit_z_score:.3f}",
        f"* baseline_top20_z: {summary.baseline_top20_z:.3f}",
        f"* recommendation_status: {summary.recommendation_status}",
        f"* verdict: {summary.verdict}",
        "",
        "## 全局消融排行",
        "",
        "| top | case | description | rounds | hits | exp_hits | lift | z | roi | pnl_vs_random | max_loss |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        lines.append(
            "| "
            f"{row.top_n} | {row.case} | {row.description} | {row.rounds} | "
            f"{row.hits} | {row.expected_hits:.2f} | {row.hit_lift:.3f} | "
            f"{row.hit_z_score:.3f} | {row.roi:.2%} | "
            f"{row.pnl_vs_random_expected:.2f} | {row.max_losing_streak} |"
        )

    lines.extend(
        [
            "",
            "## 近期窗口消融排行",
            "",
            "| window | top | case | hits | exp_hits | lift | z | roi | max_loss |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in recent_rows:
        lines.append(
            "| "
            f"{row.window_size} | {row.top_n} | {row.case} | {row.hits} | "
            f"{row.expected_hits:.2f} | {row.hit_lift:.3f} | "
            f"{row.hit_z_score:.3f} | {row.roi:.2%} | {row.max_losing_streak} |"
        )

    lines.extend(
        [
            "",
            "## 判读",
            "",
            "* 如果移除某个权重后 z 值上升，说明该权重可能在当前窗口拖累排序。",
            "* 单项 only 方案能帮助判断信号来源，但不能直接当成最终策略。",
            "* 只有全局、近期窗口和年度压力都稳定，才允许进入推荐层。",
            "* recommendation_status 为 analysis_only 时，应交付分析工具和风险闸门。",
        ]
    )
    return "\n".join(lines) + "\n"
