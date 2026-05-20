from __future__ import annotations

from dataclasses import dataclass, replace
from math import log

from .features import (
    NumberFeatures,
    all_numbers,
    extract_features,
    pattern_label,
    theoretical_distributions,
)
from .models import Draw, Recommendation
from .stats import HistoryStats, build_stats


THEORY = theoretical_distributions()
THEORY_TOTAL = 1000
CANDIDATE_FEATURES = [(number, extract_features(number)) for number in all_numbers()]
CANDIDATE_FEATURE_MAP = dict(CANDIDATE_FEATURES)


@dataclass(frozen=True)
class StrategyConfig:
    recent_window: int = 60
    exact_recent_window: int = 30
    min_history: int = 30
    position_weight: float = 0.65
    recent_position_weight: float = 0.55
    sum_weight: float = 0.50
    span_weight: float = 0.35
    pattern_weight: float = 0.25
    omission_weight: float = 0.12
    repeat_penalty: float = 0.75


def with_recent_window(config: StrategyConfig, recent_window: int) -> StrategyConfig:
    return replace(config, recent_window=recent_window)


def _bounded_log_ratio(observed_count: int, observed_total: int, expected_count: int, bucket_count: int) -> float:
    expected_prob = expected_count / THEORY_TOTAL
    prior_strength = max(20.0, float(bucket_count))
    observed_prob = (observed_count + expected_prob * prior_strength) / (
        observed_total + prior_strength
    )
    value = log(observed_prob / expected_prob)
    return max(-1.25, min(1.25, value))


def _position_component(stats: HistoryStats, position_index: int, digit: int, recent: bool) -> float:
    counts = stats.recent_position_counts if recent else stats.position_counts
    total = stats.recent_window if recent else stats.total
    return _bounded_log_ratio(
        counts[position_index][digit],
        total,
        THEORY[f"position_{position_index}"][digit],
        10,
    )


def _bucket_component(count: int, total: int, expected_count: int, bucket_count: int) -> float:
    return _bounded_log_ratio(count, total, expected_count, bucket_count)


def _omission_component(stats: HistoryStats, position_index: int, digit: int) -> float:
    omission = stats.omissions[position_index][digit]
    expected_gap = 10.0
    if omission <= 2:
        return -0.35
    if omission <= expected_gap:
        return omission / expected_gap * 0.35
    return min(0.70, 0.35 + (omission - expected_gap) / expected_gap * 0.12)


def score_features(
    features: NumberFeatures,
    stats: HistoryStats,
    config: StrategyConfig,
) -> tuple[float, list[str], list[str]]:
    parts: list[tuple[str, float]] = []

    position_score = sum(
        _position_component(stats, index, digit, recent=False)
        for index, digit in enumerate(features.digits)
    )
    parts.append(("长期位置频率", position_score * config.position_weight))

    recent_position_score = sum(
        _position_component(stats, index, digit, recent=True)
        for index, digit in enumerate(features.digits)
    )
    parts.append(("近期位置热度", recent_position_score * config.recent_position_weight))

    sum_score = _bucket_component(
        stats.sum_counts[features.sum_value],
        stats.total,
        THEORY["sum"][features.sum_value],
        len(THEORY["sum"]),
    )
    parts.append(("和值分布", sum_score * config.sum_weight))

    span_score = _bucket_component(
        stats.span_counts[features.span],
        stats.total,
        THEORY["span"][features.span],
        len(THEORY["span"]),
    )
    parts.append(("跨度分布", span_score * config.span_weight))

    pattern_score = _bucket_component(
        stats.pattern_counts[features.pattern],
        stats.total,
        THEORY["pattern"][features.pattern],
        len(THEORY["pattern"]),
    )
    parts.append(("形态分布", pattern_score * config.pattern_weight))

    omission_score = sum(
        _omission_component(stats, index, digit)
        for index, digit in enumerate(features.digits)
    )
    parts.append(("遗漏分散", omission_score * config.omission_weight))

    repeat_count = stats.exact_recent_counts[features.number]
    repeat_penalty = repeat_count * config.repeat_penalty
    if repeat_penalty:
        parts.append(("近期重号惩罚", -repeat_penalty))

    score = sum(value for _name, value in parts)
    leading = sorted(parts, key=lambda item: abs(item[1]), reverse=True)[:3]
    reasons = [f"{name}: {value:+.3f}" for name, value in leading]
    reasons.append(
        f"和值{features.sum_value}, 跨度{features.span}, {pattern_label(features.pattern)}"
    )

    risk_notes = [
        "历史分布只能解释评分来源，不能证明下一期概率已经改变",
        "候选数量越多，命中率会上升，投入也会同步放大",
    ]
    return score, reasons, risk_notes


def score_number(number: str, stats: HistoryStats, config: StrategyConfig) -> tuple[float, list[str], list[str]]:
    return score_features(extract_features(number), stats, config)


def rank_numbers(draws: list[Draw], top_n: int = 20, config: StrategyConfig | None = None) -> list[Recommendation]:
    active_config = config or StrategyConfig()
    if len(draws) < active_config.min_history:
        raise ValueError(
            f"at least {active_config.min_history} draw rows are required, got {len(draws)}"
        )

    scoped_draws = draws
    stats = build_stats(
        scoped_draws,
        recent_window=active_config.recent_window,
        exact_recent_window=active_config.exact_recent_window,
    )
    scored: list[tuple[str, float, list[str], list[str]]] = []
    for number, features in CANDIDATE_FEATURES:
        score, reasons, risk_notes = score_features(features, stats, active_config)
        scored.append((number, score, reasons, risk_notes))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return [
        Recommendation(
            rank=index + 1,
            number=number,
            score=score,
            features=CANDIDATE_FEATURE_MAP[number],
            reasons=reasons,
            risk_notes=risk_notes,
        )
        for index, (number, score, reasons, risk_notes) in enumerate(scored[:top_n])
    ]
