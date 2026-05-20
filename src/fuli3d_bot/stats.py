from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .features import POSITIONS
from .models import Draw


@dataclass(frozen=True)
class HistoryStats:
    total: int
    recent_window: int
    position_counts: tuple[Counter[int], Counter[int], Counter[int]]
    recent_position_counts: tuple[Counter[int], Counter[int], Counter[int]]
    sum_counts: Counter[int]
    recent_sum_counts: Counter[int]
    span_counts: Counter[int]
    recent_span_counts: Counter[int]
    pattern_counts: Counter[str]
    recent_pattern_counts: Counter[str]
    exact_recent_counts: Counter[str]
    omissions: tuple[dict[int, int], dict[int, int], dict[int, int]]


def build_stats(
    draws: list[Draw],
    recent_window: int = 60,
    exact_recent_window: int | None = None,
) -> HistoryStats:
    if not draws:
        raise ValueError("draws cannot be empty")

    recent = draws[-recent_window:]
    exact_recent_size = exact_recent_window if exact_recent_window is not None else recent_window
    exact_recent = draws[-exact_recent_size:] if exact_recent_size > 0 else []

    position_counts = [Counter(), Counter(), Counter()]
    recent_position_counts = [Counter(), Counter(), Counter()]
    sum_counts: Counter[int] = Counter()
    recent_sum_counts: Counter[int] = Counter()
    span_counts: Counter[int] = Counter()
    recent_span_counts: Counter[int] = Counter()
    pattern_counts: Counter[str] = Counter()
    recent_pattern_counts: Counter[str] = Counter()

    for draw in draws:
        features = draw.features
        sum_counts[features.sum_value] += 1
        span_counts[features.span] += 1
        pattern_counts[features.pattern] += 1
        for index, digit in enumerate(features.digits):
            position_counts[index][digit] += 1

    for draw in recent:
        features = draw.features
        recent_sum_counts[features.sum_value] += 1
        recent_span_counts[features.span] += 1
        recent_pattern_counts[features.pattern] += 1
        for index, digit in enumerate(features.digits):
            recent_position_counts[index][digit] += 1

    omissions: list[dict[int, int]] = []
    for position_index, _position_name in enumerate(POSITIONS):
        latest_distance = {digit: len(draws) for digit in range(10)}
        for distance, draw in enumerate(reversed(draws)):
            digit = draw.features.digits[position_index]
            if latest_distance[digit] == len(draws):
                latest_distance[digit] = distance
        omissions.append(latest_distance)

    return HistoryStats(
        total=len(draws),
        recent_window=len(recent),
        position_counts=(position_counts[0], position_counts[1], position_counts[2]),
        recent_position_counts=(
            recent_position_counts[0],
            recent_position_counts[1],
            recent_position_counts[2],
        ),
        sum_counts=sum_counts,
        recent_sum_counts=recent_sum_counts,
        span_counts=span_counts,
        recent_span_counts=recent_span_counts,
        pattern_counts=pattern_counts,
        recent_pattern_counts=recent_pattern_counts,
        exact_recent_counts=Counter(draw.number for draw in exact_recent),
        omissions=(omissions[0], omissions[1], omissions[2]),
    )
