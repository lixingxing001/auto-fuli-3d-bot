from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from typing import Any

from .features import NumberFeatures, extract_features, normalize_number


@dataclass(frozen=True)
class Draw:
    issue: str
    draw_date: date | None
    number: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "number", normalize_number(self.number))

    @cached_property
    def features(self) -> NumberFeatures:
        return extract_features(self.number)


@dataclass(frozen=True)
class Recommendation:
    rank: int
    number: str
    score: float
    features: NumberFeatures
    reasons: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BacktestPick:
    issue: str
    draw_date: date | None
    actual_number: str
    candidates: list[str]
    hit: bool
    stake: float
    payout: float
    pnl: float


@dataclass(frozen=True)
class BacktestResult:
    rounds: int
    hits: int
    hit_rate: float
    expected_hit_rate: float
    expected_hits: float
    hit_lift: float
    hit_z_score: float
    stake: float
    payout: float
    pnl: float
    roi: float
    expected_pnl: float
    expected_roi: float
    pnl_vs_random_expected: float
    max_drawdown: float
    max_losing_streak: int
    picks: list[BacktestPick]
    meta: dict[str, Any] = field(default_factory=dict)
