from __future__ import annotations

from hashlib import blake2b

from .models import Draw
from .stats import build_stats
from .strategy import CANDIDATE_FEATURES, StrategyConfig


def _rank_by_score(scored: list[tuple[str, float]], top_n: int) -> list[str]:
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [number for number, _score in scored[:top_n]]


def position_hot_ranker(draws: list[Draw], top_n: int, config: StrategyConfig) -> list[str]:
    stats = build_stats(draws, recent_window=config.recent_window)
    scored = []
    for number, features in CANDIDATE_FEATURES:
        score = sum(
            stats.recent_position_counts[index][digit]
            for index, digit in enumerate(features.digits)
        )
        scored.append((number, float(score)))
    return _rank_by_score(scored, top_n)


def position_cold_ranker(draws: list[Draw], top_n: int, config: StrategyConfig) -> list[str]:
    stats = build_stats(draws, recent_window=config.recent_window)
    scored = []
    for number, features in CANDIDATE_FEATURES:
        score = sum(
            stats.omissions[index][digit]
            for index, digit in enumerate(features.digits)
        )
        scored.append((number, float(score)))
    return _rank_by_score(scored, top_n)


def sum_hot_ranker(draws: list[Draw], top_n: int, config: StrategyConfig) -> list[str]:
    stats = build_stats(draws, recent_window=config.recent_window)
    scored = [
        (
            number,
            float(stats.recent_sum_counts[features.sum_value])
            + float(stats.sum_counts[features.sum_value]) * 0.10,
        )
        for number, features in CANDIDATE_FEATURES
    ]
    return _rank_by_score(scored, top_n)


def span_hot_ranker(draws: list[Draw], top_n: int, config: StrategyConfig) -> list[str]:
    stats = build_stats(draws, recent_window=config.recent_window)
    scored = [
        (
            number,
            float(stats.recent_span_counts[features.span])
            + float(stats.span_counts[features.span]) * 0.10,
        )
        for number, features in CANDIDATE_FEATURES
    ]
    return _rank_by_score(scored, top_n)


def pattern_hot_ranker(draws: list[Draw], top_n: int, config: StrategyConfig) -> list[str]:
    stats = build_stats(draws, recent_window=config.recent_window)
    scored = [
        (
            number,
            float(stats.recent_pattern_counts[features.pattern])
            + float(stats.pattern_counts[features.pattern]) * 0.10,
        )
        for number, features in CANDIDATE_FEATURES
    ]
    return _rank_by_score(scored, top_n)


def deterministic_random_ranker(draws: list[Draw], top_n: int, config: StrategyConfig) -> list[str]:
    seed = draws[-1].issue if draws else "empty"
    scored = []
    for number, _features in CANDIDATE_FEATURES:
        digest = blake2b(f"{seed}:{number}".encode("utf-8"), digest_size=8).digest()
        score = int.from_bytes(digest, "big")
        scored.append((number, float(score)))
    return _rank_by_score(scored, top_n)


BASELINE_RANKERS = {
    "position_hot": position_hot_ranker,
    "position_cold": position_cold_ranker,
    "sum_hot": sum_hot_ranker,
    "span_hot": span_hot_ranker,
    "pattern_hot": pattern_hot_ranker,
    "random_fixed": deterministic_random_ranker,
}


def parse_ranker_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in BASELINE_RANKERS and name != "model"]
    if unknown:
        valid = ", ".join(["model", *BASELINE_RANKERS.keys()])
        raise ValueError(f"unknown ranker names: {', '.join(unknown)}. valid: {valid}")
    return names

