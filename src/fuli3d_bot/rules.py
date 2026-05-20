from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .backtest import model_ranker, run_backtest
from .features import NumberFeatures, pattern_label
from .models import BacktestResult, Draw
from .strategy import CANDIDATE_FEATURE_MAP, StrategyConfig


FEATURE_ALIASES = {
    "sum": "sum",
    "sum_value": "sum",
    "span": "span",
    "pattern": "pattern",
    "odd": "odd_count",
    "odd_count": "odd_count",
    "big": "big_count",
    "big_count": "big_count",
    "hundreds": "hundreds",
    "hundred": "hundreds",
    "h": "hundreds",
    "tens": "tens",
    "ten": "tens",
    "t": "tens",
    "ones": "ones",
    "one": "ones",
    "o": "ones",
}

PATTERN_ALIASES = {
    "baozi": "baozi",
    "豹子": "baozi",
    "zusan": "zusan",
    "组三": "zusan",
    "zuliu": "zuliu",
    "组六": "zuliu",
}


@dataclass(frozen=True)
class RuleFilter:
    name: str
    value: str


@dataclass(frozen=True)
class RuleBacktestRow:
    rule: str
    segment: str
    year: int | None
    top_n: int
    training_window: int
    recent_window: int
    rounds: int
    avg_candidates: float
    empty_rounds: int
    hits: int
    expected_hits: float
    hit_rate: float
    hit_lift: float
    hit_z_score: float
    stake: float
    pnl: float
    roi: float
    expected_roi: float
    pnl_vs_random_expected: float
    max_losing_streak: int


@dataclass(frozen=True)
class RuleRecencyRow:
    rule: str
    window_size: int
    start_issue: str
    end_issue: str
    top_n: int
    pool_size: int
    training_window: int
    recent_window: int
    rounds: int
    avg_candidates: float
    empty_rounds: int
    hits: int
    expected_hits: float
    hit_rate: float
    hit_lift: float
    hit_z_score: float
    stake: float
    pnl: float
    roi: float
    expected_roi: float
    pnl_vs_random_expected: float
    max_losing_streak: int


def parse_rule_filters(rule: str) -> list[RuleFilter]:
    parts = [part.strip() for part in rule.split(",") if part.strip()]
    if not parts:
        raise ValueError("rule cannot be empty")

    filters: list[RuleFilter] = []
    for part in parts:
        if "=" not in part:
            raise ValueError(f"invalid rule part: {part!r}")
        raw_name, raw_value = [item.strip() for item in part.split("=", 1)]
        name = FEATURE_ALIASES.get(raw_name)
        if name is None:
            raise ValueError(f"unknown rule feature: {raw_name}")
        value = raw_value
        if name == "pattern":
            if raw_value not in PATTERN_ALIASES:
                raise ValueError(f"unknown pattern value: {raw_value}")
            value = PATTERN_ALIASES[raw_value]
        elif name in {"sum", "span", "odd_count", "big_count", "hundreds", "tens", "ones"}:
            if not raw_value.isdigit():
                raise ValueError(f"rule value must be numeric for {name}: {raw_value}")
            value = str(int(raw_value))
        filters.append(RuleFilter(name=name, value=value))
    return filters


def feature_value(features: NumberFeatures, name: str) -> str:
    if name == "sum":
        return str(features.sum_value)
    if name == "span":
        return str(features.span)
    if name == "pattern":
        return features.pattern
    if name == "odd_count":
        return str(features.odd_count)
    if name == "big_count":
        return str(features.big_count)
    if name == "hundreds":
        return str(features.digits[0])
    if name == "tens":
        return str(features.digits[1])
    if name == "ones":
        return str(features.digits[2])
    raise ValueError(f"unsupported feature: {name}")


def matches_filters(number: str, filters: list[RuleFilter]) -> bool:
    features = CANDIDATE_FEATURE_MAP[number]
    return all(feature_value(features, item.name) == item.value for item in filters)


def rule_label(filters: list[RuleFilter]) -> str:
    parts: list[str] = []
    for item in filters:
        value = pattern_label(item.value) if item.name == "pattern" else item.value
        parts.append(f"{item.name}={value}")
    return ",".join(parts)


def make_rule_ranker(filters: list[RuleFilter], pool_size: int):
    def ranker(draws: list[Draw], top_n: int, config: StrategyConfig) -> list[str]:
        ranked = model_ranker(draws, pool_size, config)
        return [number for number in ranked if matches_filters(number, filters)][:top_n]

    return ranker


def _row_from_result(
    rule: str,
    segment: str,
    year: int | None,
    top_n: int,
    training_window: int,
    recent_window: int,
    result: BacktestResult,
) -> RuleBacktestRow:
    candidate_counts = [len(pick.candidates) for pick in result.picks]
    avg_candidates = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
    empty_rounds = sum(1 for count in candidate_counts if count == 0)
    return RuleBacktestRow(
        rule=rule,
        segment=segment,
        year=year,
        top_n=top_n,
        training_window=training_window,
        recent_window=recent_window,
        rounds=result.rounds,
        avg_candidates=avg_candidates,
        empty_rounds=empty_rounds,
        hits=result.hits,
        expected_hits=result.expected_hits,
        hit_rate=result.hit_rate,
        hit_lift=result.hit_lift,
        hit_z_score=result.hit_z_score,
        stake=result.stake,
        pnl=result.pnl,
        roi=result.roi,
        expected_roi=result.expected_roi,
        pnl_vs_random_expected=result.pnl_vs_random_expected,
        max_losing_streak=result.max_losing_streak,
    )


def _candidate_stats(result: BacktestResult) -> tuple[float, int]:
    candidate_counts = [len(pick.candidates) for pick in result.picks]
    avg_candidates = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
    empty_rounds = sum(1 for count in candidate_counts if count == 0)
    return avg_candidates, empty_rounds


def _recency_row_from_result(
    rule: str,
    window_size: int,
    start_issue: str,
    end_issue: str,
    top_n: int,
    pool_size: int,
    training_window: int,
    recent_window: int,
    result: BacktestResult,
) -> RuleRecencyRow:
    avg_candidates, empty_rounds = _candidate_stats(result)
    return RuleRecencyRow(
        rule=rule,
        window_size=window_size,
        start_issue=start_issue,
        end_issue=end_issue,
        top_n=top_n,
        pool_size=pool_size,
        training_window=training_window,
        recent_window=recent_window,
        rounds=result.rounds,
        avg_candidates=avg_candidates,
        empty_rounds=empty_rounds,
        hits=result.hits,
        expected_hits=result.expected_hits,
        hit_rate=result.hit_rate,
        hit_lift=result.hit_lift,
        hit_z_score=result.hit_z_score,
        stake=result.stake,
        pnl=result.pnl,
        roi=result.roi,
        expected_roi=result.expected_roi,
        pnl_vs_random_expected=result.pnl_vs_random_expected,
        max_losing_streak=result.max_losing_streak,
    )


def run_rule_backtests(
    draws: list[Draw],
    rules: list[str],
    years: list[int],
    top_n: int = 20,
    pool_size: int = 200,
    training_window: int = 300,
    recent_window: int = 60,
    min_history: int = 300,
) -> list[RuleBacktestRow]:
    config = StrategyConfig(recent_window=recent_window, min_history=min_history)
    rows: list[RuleBacktestRow] = []
    for rule in rules:
        filters = parse_rule_filters(rule)
        label = rule_label(filters)
        ranker = make_rule_ranker(filters, pool_size)
        full_result = run_backtest(
            draws,
            top_n=top_n,
            training_window=training_window,
            config=config,
            ranker=ranker,
        )
        rows.append(
            _row_from_result(label, "all", None, top_n, training_window, recent_window, full_result)
        )
        for year in years:
            yearly_result = run_backtest(
                draws,
                top_n=top_n,
                training_window=training_window,
                config=config,
                ranker=ranker,
                eval_filter=lambda draw, active_year=year: (
                    draw.draw_date is not None and draw.draw_date.year == active_year
                ),
            )
            if yearly_result.rounds == 0:
                continue
            rows.append(
                _row_from_result(
                    label,
                    "year",
                    year,
                    top_n,
                    training_window,
                    recent_window,
                    yearly_result,
                )
            )
    rows.sort(
        key=lambda row: (
            0 if row.segment == "all" else 1,
            row.rule,
            row.year or 0,
        )
    )
    return rows


def run_rule_recency(
    draws: list[Draw],
    rules: list[str],
    windows: list[int],
    top_n: int = 20,
    pool_size: int = 200,
    training_window: int = 300,
    recent_window: int = 60,
    min_history: int = 300,
) -> list[RuleRecencyRow]:
    if not draws:
        raise ValueError("draws cannot be empty")

    config = StrategyConfig(recent_window=recent_window, min_history=min_history)
    rows: list[RuleRecencyRow] = []
    unique_windows = sorted(set(windows))
    for rule in rules:
        filters = parse_rule_filters(rule)
        label = rule_label(filters)
        ranker = make_rule_ranker(filters, pool_size)
        for window_size in unique_windows:
            if window_size <= 0:
                raise ValueError("window sizes must be positive")
            target_draws = draws[-window_size:]
            target_issues = {draw.issue for draw in target_draws}
            result = run_backtest(
                draws,
                top_n=top_n,
                training_window=training_window,
                config=config,
                ranker=ranker,
                eval_filter=lambda draw, issues=target_issues: draw.issue in issues,
            )
            rows.append(
                _recency_row_from_result(
                    label,
                    window_size,
                    target_draws[0].issue,
                    target_draws[-1].issue,
                    top_n,
                    pool_size,
                    training_window,
                    recent_window,
                    result,
                )
            )

    rows.sort(key=lambda row: (row.rule, row.window_size))
    return rows


def save_rulebacktest_reports(
    rows: list[RuleBacktestRow],
    output_dir: str | Path,
    meta: dict,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "rulebacktest_results.json"
    md_path = report_dir / "rulebacktest_results.md"
    payload = {
        "meta": meta,
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(rows, meta), encoding="utf-8")
    return json_path, md_path


def save_rulerecency_reports(
    rows: list[RuleRecencyRow],
    output_dir: str | Path,
    meta: dict,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "rulerecency_results.json"
    md_path = report_dir / "rulerecency_results.md"
    payload = {
        "meta": meta,
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_recency_markdown(rows, meta), encoding="utf-8")
    return json_path, md_path


def render_markdown(rows: list[RuleBacktestRow], meta: dict) -> str:
    overall = [row for row in rows if row.segment == "all"]
    yearly = [row for row in rows if row.segment == "year"]
    overall.sort(key=lambda row: (-row.hit_z_score, -row.pnl_vs_random_expected))
    yearly.sort(key=lambda row: (row.rule, row.year or 0))

    lines = [
        "# 福彩3D规则候选回测报告",
        "",
        "## 参数",
        "",
        f"* 数据行数: {meta.get('draw_rows')}",
        f"* rules: {meta.get('rules')}",
        f"* years: {meta.get('years')}",
        f"* top_n: {meta.get('top_n')}",
        f"* pool_size: {meta.get('pool_size')}",
        f"* training_window: {meta.get('training_window')}",
        f"* recent_window: {meta.get('recent_window')}",
        "",
        "## 全局排行",
        "",
        "| rule | rounds | avg_candidates | empty | hits | exp_hits | lift | z | roi | pnl_vs_random | max_loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            "| "
            f"{row.rule} | {row.rounds} | {row.avg_candidates:.2f} | {row.empty_rounds} | "
            f"{row.hits} | {row.expected_hits:.2f} | {row.hit_lift:.3f} | "
            f"{row.hit_z_score:.3f} | {row.roi:.2%} | "
            f"{row.pnl_vs_random_expected:.2f} | {row.max_losing_streak} |"
        )

    lines.extend(
        [
            "",
            "## 年度压力检查",
            "",
            "| rule | year | rounds | avg_candidates | hits | exp_hits | lift | z | roi | max_loss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in yearly:
        lines.append(
            "| "
            f"{row.rule} | {row.year} | {row.rounds} | {row.avg_candidates:.2f} | "
            f"{row.hits} | {row.expected_hits:.2f} | {row.hit_lift:.3f} | "
            f"{row.hit_z_score:.3f} | {row.roi:.2%} | {row.max_losing_streak} |"
        )

    lines.extend(
        [
            "",
            "## 判读",
            "",
            "* avg_candidates 太低时，单次命中会让 ROI 和 z 值剧烈波动。",
            "* 规则必须在对照年份也稳定，才有资格进入推荐策略。",
            "* 如果规则只优化了 2025 年 3 月和 5 月，那就是回看偏差。",
        ]
    )
    return "\n".join(lines) + "\n"


def render_recency_markdown(rows: list[RuleRecencyRow], meta: dict) -> str:
    ranked = sorted(
        rows,
        key=lambda row: (
            row.window_size,
            -row.hit_z_score,
            -row.pnl_vs_random_expected,
        ),
    )

    lines = [
        "# 福彩3D规则时效性检测报告",
        "",
        "## 参数",
        "",
        f"* 数据行数: {meta.get('draw_rows')}",
        f"* rules: {meta.get('rules')}",
        f"* windows: {meta.get('windows')}",
        f"* top_n: {meta.get('top_n')}",
        f"* pool_size: {meta.get('pool_size')}",
        f"* training_window: {meta.get('training_window')}",
        f"* recent_window: {meta.get('recent_window')}",
        "",
        "## 最近窗口表现",
        "",
        "| window | rule | issues | rounds | avg_candidates | empty | hits | exp_hits | lift | z | roi | max_loss |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            "| "
            f"{row.window_size} | {row.rule} | {row.start_issue}-{row.end_issue} | "
            f"{row.rounds} | {row.avg_candidates:.2f} | {row.empty_rounds} | "
            f"{row.hits} | {row.expected_hits:.2f} | {row.hit_lift:.3f} | "
            f"{row.hit_z_score:.3f} | {row.roi:.2%} | {row.max_losing_streak} |"
        )

    lines.extend(
        [
            "",
            "## 判读",
            "",
            "* 最近 60 和 120 期权重最高，长窗口好看但短窗口失效，说明规则可能已经退潮。",
            "* avg_candidates 很低时，零命中和单次命中都会放大波动。",
            "* 当前推荐过滤器应优先选择短窗口仍未明显失效的规则。",
        ]
    )
    return "\n".join(lines) + "\n"
