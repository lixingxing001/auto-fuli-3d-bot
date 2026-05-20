from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .backtest import run_backtest
from .features import NumberFeatures, extract_features, pattern_label
from .models import BacktestPick, Draw
from .strategy import StrategyConfig


@dataclass(frozen=True)
class HitDetail:
    issue: str
    date: str | None
    number: str
    candidate_rank: int
    sum_value: int
    span: int
    pattern: str
    odd_count: int
    big_count: int
    digits: tuple[int, int, int]


@dataclass(frozen=True)
class BucketComparison:
    name: str
    value: str
    target_hits: int
    target_total: int
    target_rate: float
    compare_hits: int
    compare_total: int
    compare_rate: float
    lift: float


@dataclass(frozen=True)
class FeaturePressure:
    name: str
    value: str
    target_feature_rounds: int
    target_feature_hits: int
    target_hit_rate: float
    compare_feature_rounds: int
    compare_feature_hits: int
    compare_hit_rate: float
    hit_rate_lift: float


@dataclass(frozen=True)
class AttributionReport:
    target_year: int
    target_months: list[int]
    compare_years: list[int]
    top_n: int
    training_window: int
    recent_window: int
    target_rounds: int
    target_hits: int
    hit_details: list[HitDetail]
    target_summary: dict[str, dict[str, int]]
    comparison: list[BucketComparison]
    pressure: list[FeaturePressure]


def parse_months(value: str) -> list[int]:
    months = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not months:
        raise ValueError("months cannot be empty")
    invalid = [month for month in months if month < 1 or month > 12]
    if invalid:
        raise ValueError(f"invalid months: {invalid}")
    return sorted(set(months))


def parse_years(value: str) -> list[int]:
    years = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not years:
        raise ValueError("years cannot be empty")
    return sorted(set(years))


def _pick_in_target(pick: BacktestPick, year: int, months: set[int]) -> bool:
    return (
        pick.draw_date is not None
        and pick.draw_date.year == year
        and pick.draw_date.month in months
    )


def _draw_in_years(draw: Draw, years: set[int]) -> bool:
    return draw.draw_date is not None and draw.draw_date.year in years


def _feature_buckets(features: NumberFeatures) -> dict[str, str]:
    return {
        "sum": str(features.sum_value),
        "span": str(features.span),
        "pattern": pattern_label(features.pattern),
        "odd_count": str(features.odd_count),
        "big_count": str(features.big_count),
        "hundreds": str(features.digits[0]),
        "tens": str(features.digits[1]),
        "ones": str(features.digits[2]),
    }


def _count_buckets(features_items: list[NumberFeatures]) -> dict[str, Counter[str]]:
    counters: dict[str, Counter[str]] = {
        "sum": Counter(),
        "span": Counter(),
        "pattern": Counter(),
        "odd_count": Counter(),
        "big_count": Counter(),
        "hundreds": Counter(),
        "tens": Counter(),
        "ones": Counter(),
    }
    for features in features_items:
        for name, value in _feature_buckets(features).items():
            counters[name][value] += 1
    return counters


def _serialize_summary(counters: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {name: dict(counter.most_common()) for name, counter in counters.items()}


def _build_comparison(
    hit_features: list[NumberFeatures],
    compare_features: list[NumberFeatures],
) -> list[BucketComparison]:
    hit_counters = _count_buckets(hit_features)
    compare_counters = _count_buckets(compare_features)
    rows: list[BucketComparison] = []
    target_total = len(hit_features)
    compare_total = len(compare_features)

    for name, counter in hit_counters.items():
        for value, target_hits in counter.most_common():
            compare_hits = compare_counters[name][value]
            target_rate = target_hits / target_total if target_total else 0.0
            compare_rate = compare_hits / compare_total if compare_total else 0.0
            lift = target_rate / compare_rate if compare_rate else 0.0
            rows.append(
                BucketComparison(
                    name=name,
                    value=value,
                    target_hits=target_hits,
                    target_total=target_total,
                    target_rate=target_rate,
                    compare_hits=compare_hits,
                    compare_total=compare_total,
                    compare_rate=compare_rate,
                    lift=lift,
                )
            )

    rows.sort(key=lambda row: (-row.lift, -row.target_hits, row.name, row.value))
    return rows


def _pick_has_bucket(pick: BacktestPick, name: str, value: str) -> bool:
    return _feature_buckets(extract_features(pick.actual_number)).get(name) == value


def _build_pressure(
    target_picks: list[BacktestPick],
    compare_picks: list[BacktestPick],
    comparison: list[BucketComparison],
    max_features: int = 12,
) -> list[FeaturePressure]:
    rows: list[FeaturePressure] = []
    seen: set[tuple[str, str]] = set()
    for item in comparison:
        key = (item.name, item.value)
        if key in seen:
            continue
        seen.add(key)
        target_feature = [pick for pick in target_picks if _pick_has_bucket(pick, item.name, item.value)]
        compare_feature = [pick for pick in compare_picks if _pick_has_bucket(pick, item.name, item.value)]
        target_hits = sum(1 for pick in target_feature if pick.hit)
        compare_hits = sum(1 for pick in compare_feature if pick.hit)
        target_rate = target_hits / len(target_feature) if target_feature else 0.0
        compare_rate = compare_hits / len(compare_feature) if compare_feature else 0.0
        hit_rate_lift = target_rate / compare_rate if compare_rate else 0.0
        rows.append(
            FeaturePressure(
                name=item.name,
                value=item.value,
                target_feature_rounds=len(target_feature),
                target_feature_hits=target_hits,
                target_hit_rate=target_rate,
                compare_feature_rounds=len(compare_feature),
                compare_feature_hits=compare_hits,
                compare_hit_rate=compare_rate,
                hit_rate_lift=hit_rate_lift,
            )
        )
        if len(rows) >= max_features:
            break
    return rows


def run_hit_attribution(
    draws: list[Draw],
    target_year: int,
    target_months: list[int],
    compare_years: list[int],
    top_n: int = 20,
    training_window: int = 300,
    recent_window: int = 60,
    min_history: int = 300,
) -> AttributionReport:
    config = StrategyConfig(recent_window=recent_window, min_history=min_history)
    target_month_set = set(target_months)
    result = run_backtest(
        draws,
        top_n=top_n,
        training_window=training_window,
        config=config,
        eval_filter=lambda draw: (
            draw.draw_date is not None
            and draw.draw_date.year == target_year
            and draw.draw_date.month in target_month_set
        ),
    )
    compare_year_set = set(compare_years)
    compare_result = run_backtest(
        draws,
        top_n=top_n,
        training_window=training_window,
        config=config,
        eval_filter=lambda draw: (
            draw.draw_date is not None and draw.draw_date.year in compare_year_set
        ),
    )

    hit_details: list[HitDetail] = []
    hit_features: list[NumberFeatures] = []
    for pick in result.picks:
        if not pick.hit:
            continue
        features = extract_features(pick.actual_number)
        hit_features.append(features)
        hit_details.append(
            HitDetail(
                issue=pick.issue,
                date=pick.draw_date.isoformat() if pick.draw_date else None,
                number=pick.actual_number,
                candidate_rank=pick.candidates.index(pick.actual_number) + 1,
                sum_value=features.sum_value,
                span=features.span,
                pattern=pattern_label(features.pattern),
                odd_count=features.odd_count,
                big_count=features.big_count,
                digits=features.digits,
            )
        )

    compare_features = [
        draw.features
        for draw in draws
        if _draw_in_years(draw, compare_year_set)
    ]
    target_summary = _serialize_summary(_count_buckets(hit_features))
    comparison = _build_comparison(hit_features, compare_features)
    pressure = _build_pressure(result.picks, compare_result.picks, comparison)

    return AttributionReport(
        target_year=target_year,
        target_months=target_months,
        compare_years=compare_years,
        top_n=top_n,
        training_window=training_window,
        recent_window=recent_window,
        target_rounds=result.rounds,
        target_hits=result.hits,
        hit_details=hit_details,
        target_summary=target_summary,
        comparison=comparison,
        pressure=pressure,
    )


def save_attribution_report(
    report: AttributionReport,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "hit_attribution.json"
    md_path = report_dir / "hit_attribution.md"
    json_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def render_markdown(report: AttributionReport) -> str:
    lines = [
        "# 福彩3D命中归因报告",
        "",
        "## 参数",
        "",
        f"* 目标年份: {report.target_year}",
        f"* 目标月份: {report.target_months}",
        f"* 对照年份: {report.compare_years}",
        f"* top_n: {report.top_n}",
        f"* training_window: {report.training_window}",
        f"* recent_window: {report.recent_window}",
        f"* 目标轮次: {report.target_rounds}",
        f"* 目标命中: {report.target_hits}",
        "",
        "## 命中明细",
        "",
        "| date | issue | number | rank | sum | span | pattern | odd | big | digits |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for hit in report.hit_details:
        lines.append(
            "| "
            f"{hit.date} | {hit.issue} | {hit.number} | {hit.candidate_rank} | "
            f"{hit.sum_value} | {hit.span} | {hit.pattern} | "
            f"{hit.odd_count} | {hit.big_count} | {hit.digits} |"
        )

    lines.extend(
        [
            "",
            "## 特征富集",
            "",
            "| feature | value | target_hits | target_rate | compare_hits | compare_rate | lift |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report.comparison[:30]:
        lines.append(
            "| "
            f"{row.name} | {row.value} | {row.target_hits} | {row.target_rate:.2%} | "
            f"{row.compare_hits} | {row.compare_rate:.2%} | {row.lift:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 条件命中压力检查",
            "",
            "| feature | value | target_rounds | target_hits | target_rate | compare_rounds | compare_hits | compare_rate | lift |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report.pressure:
        lines.append(
            "| "
            f"{row.name} | {row.value} | {row.target_feature_rounds} | "
            f"{row.target_feature_hits} | {row.target_hit_rate:.2%} | "
            f"{row.compare_feature_rounds} | {row.compare_feature_hits} | "
            f"{row.compare_hit_rate:.2%} | {row.hit_rate_lift:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 判读",
            "",
            "* 高 lift 只说明目标命中样本中富集，样本量小的时候很容易失真。",
            "* 如果富集特征在对照年份没有复现，不能把它升级为稳定规则。",
            "* 候选排名靠前比候选排名靠后更值得继续拆解。",
        ]
    )
    return "\n".join(lines) + "\n"
