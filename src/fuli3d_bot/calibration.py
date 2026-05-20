from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from math import sqrt
from pathlib import Path

from .models import Draw
from .strategy import StrategyConfig, rank_numbers


RANK_TOTAL = 1000


@dataclass(frozen=True)
class CalibrationPick:
    issue: str
    draw_date: str | None
    actual_number: str
    actual_rank: int
    actual_score: float


@dataclass(frozen=True)
class CalibrationTopRow:
    segment: str
    window_size: int | None
    start_issue: str
    end_issue: str
    top_n: int
    rounds: int
    hits: int
    expected_hits: float
    hit_rate: float
    expected_hit_rate: float
    hit_lift: float
    hit_z_score: float
    stake: float
    pnl: float
    roi: float
    expected_roi: float
    pnl_vs_random_expected: float
    max_losing_streak: int


@dataclass(frozen=True)
class CalibrationBucketRow:
    segment: str
    window_size: int | None
    start_issue: str
    end_issue: str
    bucket: str
    start_rank: int
    end_rank: int
    rounds: int
    hits: int
    expected_hits: float
    hit_rate: float
    expected_hit_rate: float
    hit_lift: float
    hit_z_score: float


@dataclass(frozen=True)
class CalibrationSummary:
    rounds: int
    mean_actual_rank: float
    median_actual_rank: float
    top_bucket_lift: float
    bottom_bucket_lift: float
    monotonic_violations: int
    signal_status: str
    verdict: str


@dataclass(frozen=True)
class CalibrationReport:
    summary: CalibrationSummary
    top_rows: list[CalibrationTopRow]
    bucket_rows: list[CalibrationBucketRow]
    picks: list[CalibrationPick]


def _hit_z_score(hits: int, expected_hits: float, variance: float) -> float:
    return (hits - expected_hits) / sqrt(variance) if variance > 0 else 0.0


def _max_losing_streak(values: list[bool]) -> int:
    current = 0
    longest = 0
    for hit in values:
        if hit:
            current = 0
            continue
        current += 1
        longest = max(longest, current)
    return longest


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[midpoint])
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


def _scope_picks(picks: list[CalibrationPick], window_size: int | None) -> list[CalibrationPick]:
    if window_size is None:
        return picks
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    return picks[-window_size:]


def _scope_name(window_size: int | None) -> str:
    return "all" if window_size is None else "recent"


def _scope_issues(picks: list[CalibrationPick]) -> tuple[str, str]:
    if not picks:
        return "", ""
    return picks[0].issue, picks[-1].issue


def collect_calibration_picks(
    draws: list[Draw],
    training_window: int = 300,
    recent_window: int = 60,
    min_history: int = 300,
) -> list[CalibrationPick]:
    if training_window <= 0:
        raise ValueError("training_window must be positive")
    if len(draws) <= training_window:
        raise ValueError(
            f"need more draw rows than training_window, got {len(draws)} rows and window {training_window}"
        )

    config = StrategyConfig(recent_window=recent_window, min_history=min_history)
    config = replace(config, min_history=min(config.min_history, training_window))
    picks: list[CalibrationPick] = []
    for index in range(training_window, len(draws)):
        training_draws = draws[index - training_window : index]
        actual = draws[index]
        ranked = rank_numbers(training_draws, top_n=RANK_TOTAL, config=config)
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


def summarize_top_hits(
    picks: list[CalibrationPick],
    top_n: int,
    window_size: int | None = None,
    stake_per_number: float = 2.0,
    payout_per_hit: float = 1040.0,
) -> CalibrationTopRow:
    if not 0 < top_n <= RANK_TOTAL:
        raise ValueError("top_n must be between 1 and 1000")

    scoped = _scope_picks(picks, window_size)
    rounds = len(scoped)
    start_issue, end_issue = _scope_issues(scoped)
    hit_flags = [item.actual_rank <= top_n for item in scoped]
    hits = sum(1 for hit in hit_flags if hit)
    probability = top_n / RANK_TOTAL
    expected_hits = rounds * probability
    variance = rounds * probability * (1.0 - probability)
    stake = rounds * top_n * stake_per_number
    payout = hits * payout_per_hit
    pnl = payout - stake
    expected_pnl = expected_hits * payout_per_hit - stake

    return CalibrationTopRow(
        segment=_scope_name(window_size),
        window_size=window_size,
        start_issue=start_issue,
        end_issue=end_issue,
        top_n=top_n,
        rounds=rounds,
        hits=hits,
        expected_hits=expected_hits,
        hit_rate=hits / rounds if rounds else 0.0,
        expected_hit_rate=probability,
        hit_lift=hits / expected_hits if expected_hits else 0.0,
        hit_z_score=_hit_z_score(hits, expected_hits, variance),
        stake=stake,
        pnl=pnl,
        roi=pnl / stake if stake else 0.0,
        expected_roi=expected_pnl / stake if stake else 0.0,
        pnl_vs_random_expected=pnl - expected_pnl,
        max_losing_streak=_max_losing_streak(hit_flags),
    )


def rank_buckets(bucket_size: int = 100) -> list[tuple[int, int]]:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    ranges: list[tuple[int, int]] = []
    start = 1
    while start <= RANK_TOTAL:
        end = min(RANK_TOTAL, start + bucket_size - 1)
        ranges.append((start, end))
        start = end + 1
    return ranges


def summarize_rank_buckets(
    picks: list[CalibrationPick],
    bucket_size: int = 100,
    window_size: int | None = None,
) -> list[CalibrationBucketRow]:
    scoped = _scope_picks(picks, window_size)
    rounds = len(scoped)
    start_issue, end_issue = _scope_issues(scoped)
    rows: list[CalibrationBucketRow] = []
    for start_rank, end_rank in rank_buckets(bucket_size):
        hits = sum(1 for item in scoped if start_rank <= item.actual_rank <= end_rank)
        size = end_rank - start_rank + 1
        probability = size / RANK_TOTAL
        expected_hits = rounds * probability
        variance = rounds * probability * (1.0 - probability)
        rows.append(
            CalibrationBucketRow(
                segment=_scope_name(window_size),
                window_size=window_size,
                start_issue=start_issue,
                end_issue=end_issue,
                bucket=f"{start_rank}-{end_rank}",
                start_rank=start_rank,
                end_rank=end_rank,
                rounds=rounds,
                hits=hits,
                expected_hits=expected_hits,
                hit_rate=hits / rounds if rounds else 0.0,
                expected_hit_rate=probability,
                hit_lift=hits / expected_hits if expected_hits else 0.0,
                hit_z_score=_hit_z_score(hits, expected_hits, variance),
            )
        )
    return rows


def summarize_calibration(
    picks: list[CalibrationPick],
    top_rows: list[CalibrationTopRow],
    bucket_rows: list[CalibrationBucketRow],
) -> CalibrationSummary:
    ranks = [item.actual_rank for item in picks]
    all_buckets = [row for row in bucket_rows if row.segment == "all"]
    all_buckets.sort(key=lambda row: row.start_rank)
    lift_values = [row.hit_lift for row in all_buckets]
    monotonic_violations = sum(
        1 for index in range(1, len(lift_values)) if lift_values[index] > lift_values[index - 1]
    )
    top20 = next((row for row in top_rows if row.segment == "all" and row.top_n == 20), None)
    top_bucket_lift = lift_values[0] if lift_values else 0.0
    bottom_bucket_lift = lift_values[-1] if lift_values else 0.0

    if top20 and top20.hit_z_score >= 2.0 and top_bucket_lift > 1.0:
        signal_status = "strong"
        verdict = "Top20 和头部分桶同时显示优势，基础评分具备继续优化资格。"
    elif top20 and top20.hit_z_score > 0 and top_bucket_lift > 1.0:
        signal_status = "weak"
        verdict = "基础评分有局部信号，但强度不足，不能直接升级成推荐。"
    else:
        signal_status = "failed"
        verdict = "基础评分排序能力不足，应先调整特征权重和评分结构。"

    return CalibrationSummary(
        rounds=len(picks),
        mean_actual_rank=sum(ranks) / len(ranks) if ranks else 0.0,
        median_actual_rank=_median(ranks),
        top_bucket_lift=top_bucket_lift,
        bottom_bucket_lift=bottom_bucket_lift,
        monotonic_violations=monotonic_violations,
        signal_status=signal_status,
        verdict=verdict,
    )


def run_calibration(
    draws: list[Draw],
    top_values: list[int],
    windows: list[int],
    training_window: int = 300,
    recent_window: int = 60,
    min_history: int = 300,
    bucket_size: int = 100,
    stake_per_number: float = 2.0,
    payout_per_hit: float = 1040.0,
) -> CalibrationReport:
    picks = collect_calibration_picks(
        draws,
        training_window=training_window,
        recent_window=recent_window,
        min_history=min_history,
    )
    unique_top_values = sorted(set(top_values))
    unique_windows = sorted(set(windows))
    if any(value > RANK_TOTAL for value in unique_top_values):
        raise ValueError("top values must be between 1 and 1000")

    top_rows: list[CalibrationTopRow] = []
    for top_n in unique_top_values:
        top_rows.append(
            summarize_top_hits(
                picks,
                top_n=top_n,
                stake_per_number=stake_per_number,
                payout_per_hit=payout_per_hit,
            )
        )
        for window_size in unique_windows:
            top_rows.append(
                summarize_top_hits(
                    picks,
                    top_n=top_n,
                    window_size=window_size,
                    stake_per_number=stake_per_number,
                    payout_per_hit=payout_per_hit,
                )
            )

    bucket_rows = summarize_rank_buckets(picks, bucket_size=bucket_size)
    for window_size in unique_windows:
        bucket_rows.extend(
            summarize_rank_buckets(picks, bucket_size=bucket_size, window_size=window_size)
        )

    summary = summarize_calibration(picks, top_rows, bucket_rows)
    return CalibrationReport(
        summary=summary,
        top_rows=top_rows,
        bucket_rows=bucket_rows,
        picks=picks,
    )


def save_calibration_reports(
    report: CalibrationReport,
    output_dir: str | Path,
    meta: dict,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "calibration_report.json"
    md_path = report_dir / "calibration_report.md"
    payload = {
        "meta": meta,
        "report": asdict(report),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_calibration_markdown(report, meta), encoding="utf-8")
    return json_path, md_path


def render_calibration_markdown(report: CalibrationReport, meta: dict) -> str:
    top_all = [row for row in report.top_rows if row.segment == "all"]
    top_recent = [row for row in report.top_rows if row.segment == "recent"]
    buckets_all = [row for row in report.bucket_rows if row.segment == "all"]
    buckets_recent = [row for row in report.bucket_rows if row.segment == "recent"]
    top_all.sort(key=lambda row: row.top_n)
    top_recent.sort(key=lambda row: (row.window_size or 0, row.top_n))
    buckets_all.sort(key=lambda row: row.start_rank)
    buckets_recent.sort(key=lambda row: (row.window_size or 0, row.start_rank))

    summary = report.summary
    lines = [
        "# 福彩3D基础评分校准报告",
        "",
        "## 参数",
        "",
        f"* 数据行数: {meta.get('draw_rows')}",
        f"* top_values: {meta.get('top_values')}",
        f"* windows: {meta.get('windows')}",
        f"* training_window: {meta.get('training_window')}",
        f"* recent_window: {meta.get('recent_window')}",
        f"* bucket_size: {meta.get('bucket_size')}",
        "",
        "## 总结",
        "",
        f"* rounds: {summary.rounds}",
        f"* mean_actual_rank: {summary.mean_actual_rank:.2f}",
        f"* median_actual_rank: {summary.median_actual_rank:.2f}",
        f"* top_bucket_lift: {summary.top_bucket_lift:.3f}",
        f"* bottom_bucket_lift: {summary.bottom_bucket_lift:.3f}",
        f"* monotonic_violations: {summary.monotonic_violations}",
        f"* signal_status: {summary.signal_status}",
        f"* verdict: {summary.verdict}",
        "",
        "## 全局 TopN 命中",
        "",
        "| top | rounds | issues | hits | exp_hits | lift | z | roi | pnl_vs_random | max_loss |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_all:
        lines.append(
            "| "
            f"{row.top_n} | {row.rounds} | {row.start_issue}-{row.end_issue} | "
            f"{row.hits} | {row.expected_hits:.2f} | {row.hit_lift:.3f} | "
            f"{row.hit_z_score:.3f} | {row.roi:.2%} | "
            f"{row.pnl_vs_random_expected:.2f} | {row.max_losing_streak} |"
        )

    lines.extend(
        [
            "",
            "## 近期窗口 TopN 命中",
            "",
            "| window | top | rounds | issues | hits | exp_hits | lift | z | roi | max_loss |",
            "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_recent:
        lines.append(
            "| "
            f"{row.window_size} | {row.top_n} | {row.rounds} | "
            f"{row.start_issue}-{row.end_issue} | {row.hits} | {row.expected_hits:.2f} | "
            f"{row.hit_lift:.3f} | {row.hit_z_score:.3f} | {row.roi:.2%} | "
            f"{row.max_losing_streak} |"
        )

    lines.extend(
        [
            "",
            "## 全局评分分桶",
            "",
            "| rank_bucket | rounds | hits | exp_hits | lift | z |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in buckets_all:
        lines.append(
            "| "
            f"{row.bucket} | {row.rounds} | {row.hits} | {row.expected_hits:.2f} | "
            f"{row.hit_lift:.3f} | {row.hit_z_score:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 近期评分分桶",
            "",
            "| window | rank_bucket | rounds | hits | exp_hits | lift | z |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in buckets_recent:
        lines.append(
            "| "
            f"{row.window_size} | {row.bucket} | {row.rounds} | {row.hits} | "
            f"{row.expected_hits:.2f} | {row.hit_lift:.3f} | {row.hit_z_score:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 判读",
            "",
            "* mean_actual_rank 低于 500 才说明排序整体向前移动。",
            "* TopN 的 z 值需要跨窗口为正，单一窗口命中不能支撑升级。",
            "* 分桶 lift 应从高分桶到低分桶大体下降，反向跳升越多，排序越不稳定。",
            "* signal_status 为 failed 时，优先改评分结构，暂停输出增强推荐。",
        ]
    )
    return "\n".join(lines) + "\n"
