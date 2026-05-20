from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


POSITIONS = ("hundreds", "tens", "ones")
PATTERN_LABELS = {
    "baozi": "豹子",
    "zusan": "组三",
    "zuliu": "组六",
}


@dataclass(frozen=True)
class NumberFeatures:
    number: str
    digits: tuple[int, int, int]
    sum_value: int
    span: int
    pattern: str
    odd_count: int
    big_count: int


def normalize_number(value: str | int) -> str:
    raw = str(value).strip()
    if not raw:
        raise ValueError("number is empty")
    if not raw.isdigit():
        raise ValueError(f"number must contain digits only: {raw!r}")
    if len(raw) > 3:
        raise ValueError(f"number must be in 000..999: {raw!r}")
    return raw.zfill(3)


def classify_pattern(digits: tuple[int, int, int]) -> str:
    unique_count = len(set(digits))
    if unique_count == 1:
        return "baozi"
    if unique_count == 2:
        return "zusan"
    return "zuliu"


def extract_features(number: str | int) -> NumberFeatures:
    normalized = normalize_number(number)
    digits = tuple(int(ch) for ch in normalized)
    if len(digits) != 3:
        raise ValueError(f"invalid normalized number: {normalized!r}")

    return NumberFeatures(
        number=normalized,
        digits=digits,  # type: ignore[arg-type]
        sum_value=sum(digits),
        span=max(digits) - min(digits),
        pattern=classify_pattern(digits),  # type: ignore[arg-type]
        odd_count=sum(digit % 2 for digit in digits),
        big_count=sum(1 for digit in digits if digit >= 5),
    )


def pattern_label(pattern: str) -> str:
    return PATTERN_LABELS.get(pattern, pattern)


def all_numbers() -> list[str]:
    return [f"{number:03d}" for number in range(1000)]


def theoretical_distributions() -> dict[str, Counter]:
    sums: Counter[int] = Counter()
    spans: Counter[int] = Counter()
    patterns: Counter[str] = Counter()
    position_digits: list[Counter[int]] = [Counter(), Counter(), Counter()]

    for number in all_numbers():
        features = extract_features(number)
        sums[features.sum_value] += 1
        spans[features.span] += 1
        patterns[features.pattern] += 1
        for index, digit in enumerate(features.digits):
            position_digits[index][digit] += 1

    return {
        "sum": sums,
        "span": spans,
        "pattern": patterns,
        "position_0": position_digits[0],
        "position_1": position_digits[1],
        "position_2": position_digits[2],
    }

