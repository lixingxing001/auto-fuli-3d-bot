from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from math import erfc, exp, isfinite, lgamma, log, log1p, sqrt
from pathlib import Path

from .features import all_numbers, extract_features, theoretical_distributions
from .models import Draw


RANK_TOTAL = 1000


@dataclass(frozen=True)
class DistributionTest:
    name: str
    sample_size: int
    categories: int
    statistic: float
    df: int
    p_value: float
    adjusted_alpha: float
    status: str
    strongest_bucket: str
    strongest_observed: int
    strongest_expected: float
    note: str


@dataclass(frozen=True)
class SerialTest:
    name: str
    sample_size: int
    lags: int
    statistic: float
    df: int
    p_value: float
    adjusted_alpha: float
    status: str
    max_abs_correlation: float
    strongest_lag: int
    note: str


@dataclass(frozen=True)
class TransitionTest:
    name: str
    sample_size: int
    states: int
    statistic: float
    df: int
    p_value: float
    adjusted_alpha: float
    status: str
    strongest_transition: str
    strongest_residual: float
    note: str


@dataclass(frozen=True)
class FormulaTest:
    name: str
    formula: str
    train_rounds: int
    train_hits: int
    test_rounds: int
    test_hits: int
    expected_test_hits: float
    test_hit_rate: float
    expected_hit_rate: float
    lift: float
    z_score: float
    p_value: float
    adjusted_alpha: float
    status: str
    note: str


@dataclass(frozen=True)
class LawCheckSummary:
    rows: int
    latest_issue: str
    latest_date: str | None
    latest_number: str
    alpha: float
    adjusted_alpha: float
    total_tests: int
    distribution_rejections: int
    serial_rejections: int
    transition_rejections: int
    validated_formulas: int
    best_formula_name: str
    best_formula_hits: int
    best_formula_expected_hits: float
    best_formula_p_value: float
    conclusion_status: str
    verdict: str


@dataclass(frozen=True)
class LawCheckReport:
    summary: LawCheckSummary
    distribution_tests: list[DistributionTest]
    serial_tests: list[SerialTest]
    transition_tests: list[TransitionTest]
    formula_tests: list[FormulaTest]


def _gammaincc(a: float, x: float) -> float:
    if x < 0.0 or a <= 0.0:
        raise ValueError("invalid gamma parameters")
    if x == 0.0:
        return 1.0
    eps = 1e-14
    tiny = 1e-300
    max_iter = 300
    gln = lgamma(a)
    if x < a + 1.0:
        ap = a
        delta = 1.0 / a
        total = delta
        for _index in range(max_iter):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * eps:
                lower = total * exp(-x + a * log(x) - gln)
                return max(0.0, min(1.0, 1.0 - lower))
        lower = total * exp(-x + a * log(x) - gln)
        return max(0.0, min(1.0, 1.0 - lower))

    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for index in range(1, max_iter + 1):
        an = -index * (index - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            upper = exp(-x + a * log(x) - gln) * h
            return max(0.0, min(1.0, upper))
    upper = exp(-x + a * log(x) - gln) * h
    return max(0.0, min(1.0, upper))


def _chi_square_p_value(statistic: float, df: int) -> float:
    if statistic < 0.0 or df <= 0:
        return 1.0
    return _gammaincc(df / 2.0, statistic / 2.0)


def _normal_upper_tail(z_score: float) -> float:
    return 0.5 * erfc(z_score / sqrt(2.0))


def _binomial_upper_tail(hits: int, rounds: int, probability: float) -> float:
    if rounds <= 0:
        return 1.0
    if hits <= 0:
        return 1.0
    if probability <= 0.0:
        return 0.0 if hits > 0 else 1.0
    if probability >= 1.0:
        return 1.0 if hits <= rounds else 0.0

    log_p0 = rounds * log1p(-probability)
    if log_p0 < -745:
        expected = rounds * probability
        variance = rounds * probability * (1.0 - probability)
        z_score = (hits - expected) / sqrt(variance) if variance > 0.0 else 0.0
        return _normal_upper_tail(z_score)

    probability_mass = exp(log_p0)
    lower = probability_mass
    for index in range(0, hits - 1):
        probability_mass *= (
            (rounds - index)
            / (index + 1)
            * probability
            / (1.0 - probability)
        )
        lower += probability_mass
    return max(0.0, min(1.0, 1.0 - lower))


def _p_status(p_value: float, alpha: float, adjusted_alpha: float) -> str:
    if p_value < adjusted_alpha:
        return "rejected_after_correction"
    if p_value < alpha:
        return "weak_signal"
    return "compatible"


def _formula_status(
    test_hits: int,
    expected_hits: float,
    p_value: float,
    alpha: float,
    adjusted_alpha: float,
) -> str:
    if test_hits <= expected_hits:
        return "no_edge"
    if p_value < adjusted_alpha:
        return "validated"
    if p_value < alpha:
        return "weak_edge"
    return "no_statistical_edge"


def _strongest_bucket(
    labels: list[str],
    observed: list[int],
    expected: list[float],
) -> tuple[str, int, float]:
    best_index = 0
    best_gap = -1.0
    for index, (obs, exp_value) in enumerate(zip(observed, expected)):
        if exp_value <= 0.0:
            continue
        gap = abs(obs - exp_value) / sqrt(exp_value)
        if gap > best_gap:
            best_index = index
            best_gap = gap
    return labels[best_index], observed[best_index], expected[best_index]


def _distribution_test(
    name: str,
    counts: Counter,
    expected_counts: dict,
    sample_size: int,
    note: str = "",
) -> DistributionTest:
    labels = [str(label) for label in expected_counts]
    observed = [counts.get(label, 0) for label in expected_counts]
    expected = [
        sample_size * expected_counts[label] / RANK_TOTAL
        for label in expected_counts
    ]
    statistic = sum(
        ((obs - exp_value) ** 2) / exp_value
        for obs, exp_value in zip(observed, expected)
        if exp_value > 0.0
    )
    bucket, bucket_observed, bucket_expected = _strongest_bucket(labels, observed, expected)
    df = max(1, len(labels) - 1)
    return DistributionTest(
        name=name,
        sample_size=sample_size,
        categories=len(labels),
        statistic=statistic,
        df=df,
        p_value=_chi_square_p_value(statistic, df),
        adjusted_alpha=0.0,
        status="pending",
        strongest_bucket=bucket,
        strongest_observed=bucket_observed,
        strongest_expected=bucket_expected,
        note=note,
    )


def _feature_theory_counts(feature_name: str) -> dict:
    counts: Counter = Counter()
    for number in all_numbers():
        features = extract_features(number)
        counts[getattr(features, feature_name)] += 1
    return dict(counts)


def build_distribution_tests(draws: list[Draw]) -> list[DistributionTest]:
    sample_size = len(draws)
    theory = theoretical_distributions()
    tests: list[DistributionTest] = []
    for position_index, name in enumerate(("百位数字均匀性", "十位数字均匀性", "个位数字均匀性")):
        counts = Counter(draw.features.digits[position_index] for draw in draws)
        tests.append(
            _distribution_test(
                name,
                counts,
                dict(theory[f"position_{position_index}"]),
                sample_size,
            )
        )

    tests.append(
        _distribution_test(
            "和值分布",
            Counter(draw.features.sum_value for draw in draws),
            dict(theory["sum"]),
            sample_size,
        )
    )
    tests.append(
        _distribution_test(
            "跨度分布",
            Counter(draw.features.span for draw in draws),
            dict(theory["span"]),
            sample_size,
        )
    )
    tests.append(
        _distribution_test(
            "形态分布",
            Counter(draw.features.pattern for draw in draws),
            dict(theory["pattern"]),
            sample_size,
        )
    )
    tests.append(
        _distribution_test(
            "奇数个数分布",
            Counter(draw.features.odd_count for draw in draws),
            _feature_theory_counts("odd_count"),
            sample_size,
        )
    )
    tests.append(
        _distribution_test(
            "大号个数分布",
            Counter(draw.features.big_count for draw in draws),
            _feature_theory_counts("big_count"),
            sample_size,
        )
    )
    tests.append(
        _distribution_test(
            "直选号码整体频率",
            Counter(draw.number for draw in draws),
            {number: 1 for number in all_numbers()},
            sample_size,
            "单格期望约为样本量除以1000，主要用于发现极端偏差。",
        )
    )
    return tests


def _autocorrelation(values: list[float], lag: int) -> float:
    if lag <= 0 or lag >= len(values):
        return 0.0
    mean_value = sum(values) / len(values)
    denominator = sum((value - mean_value) ** 2 for value in values)
    if denominator <= 0.0:
        return 0.0
    numerator = sum(
        (values[index] - mean_value) * (values[index - lag] - mean_value)
        for index in range(lag, len(values))
    )
    return numerator / denominator


def _ljung_box_test(name: str, values: list[float], max_lag: int) -> SerialTest:
    sample_size = len(values)
    safe_lags = max(1, min(max_lag, sample_size - 2))
    correlations = [
        _autocorrelation(values, lag)
        for lag in range(1, safe_lags + 1)
    ]
    statistic = sample_size * (sample_size + 2) * sum(
        (correlation ** 2) / (sample_size - lag)
        for lag, correlation in enumerate(correlations, start=1)
    )
    strongest_lag = max(
        range(1, safe_lags + 1),
        key=lambda lag: abs(correlations[lag - 1]),
    )
    max_abs = abs(correlations[strongest_lag - 1])
    return SerialTest(
        name=name,
        sample_size=sample_size,
        lags=safe_lags,
        statistic=statistic,
        df=safe_lags,
        p_value=_chi_square_p_value(statistic, safe_lags),
        adjusted_alpha=0.0,
        status="pending",
        max_abs_correlation=max_abs,
        strongest_lag=strongest_lag,
        note="Ljung-Box检验用于观察多阶自相关。",
    )


def build_serial_tests(draws: list[Draw], max_lag: int) -> list[SerialTest]:
    rows = [
        _ljung_box_test(
            name,
            [float(draw.features.digits[index]) for draw in draws],
            max_lag,
        )
        for index, name in enumerate(("百位数字自相关", "十位数字自相关", "个位数字自相关"))
    ]
    rows.extend(
        [
            _ljung_box_test("和值自相关", [float(draw.features.sum_value) for draw in draws], max_lag),
            _ljung_box_test("跨度自相关", [float(draw.features.span) for draw in draws], max_lag),
            _ljung_box_test("号码整数自相关", [float(int(draw.number)) for draw in draws], max_lag),
        ]
    )
    return rows


def _transition_test(name: str, values: list[str], states: list[str]) -> TransitionTest:
    table: dict[str, Counter] = {state: Counter() for state in states}
    row_counts: Counter = Counter()
    col_counts: Counter = Counter()
    for previous, current in zip(values, values[1:]):
        table[previous][current] += 1
        row_counts[previous] += 1
        col_counts[current] += 1

    total = len(values) - 1
    statistic = 0.0
    strongest_transition = ""
    strongest_residual = 0.0
    for previous in states:
        for current in states:
            expected = row_counts[previous] * col_counts[current] / total if total else 0.0
            if expected <= 0.0:
                continue
            observed = table[previous][current]
            residual = (observed - expected) / sqrt(expected)
            statistic += residual ** 2
            if abs(residual) > abs(strongest_residual):
                strongest_residual = residual
                strongest_transition = f"{previous}->{current}"

    df = max(1, (len(states) - 1) * (len(states) - 1))
    return TransitionTest(
        name=name,
        sample_size=total,
        states=len(states),
        statistic=statistic,
        df=df,
        p_value=_chi_square_p_value(statistic, df),
        adjusted_alpha=0.0,
        status="pending",
        strongest_transition=strongest_transition,
        strongest_residual=strongest_residual,
        note="检验上一期状态与下一期状态是否独立。",
    )


def build_transition_tests(draws: list[Draw]) -> list[TransitionTest]:
    digit_states = [str(value) for value in range(10)]
    tests = [
        _transition_test(
            name,
            [str(draw.features.digits[index]) for draw in draws],
            digit_states,
        )
        for index, name in enumerate(("百位数字转移独立性", "十位数字转移独立性", "个位数字转移独立性"))
    ]
    tests.append(
        _transition_test(
            "形态转移独立性",
            [draw.features.pattern for draw in draws],
            ["baozi", "zusan", "zuliu"],
        )
    )
    return tests


def _formula_result(
    name: str,
    formula: str,
    train_rounds: int,
    train_hits: int,
    test_rounds: int,
    test_hits: int,
    note: str,
) -> FormulaTest:
    expected_rate = 1.0 / RANK_TOTAL
    expected_hits = test_rounds * expected_rate
    variance = test_rounds * expected_rate * (1.0 - expected_rate)
    z_score = (test_hits - expected_hits) / sqrt(variance) if variance > 0.0 else 0.0
    p_value = _binomial_upper_tail(test_hits, test_rounds, expected_rate)
    lift = test_hits / expected_hits if expected_hits else 0.0
    return FormulaTest(
        name=name,
        formula=formula,
        train_rounds=train_rounds,
        train_hits=train_hits,
        test_rounds=test_rounds,
        test_hits=test_hits,
        expected_test_hits=expected_hits,
        test_hit_rate=test_hits / test_rounds if test_rounds else 0.0,
        expected_hit_rate=expected_rate,
        lift=lift,
        z_score=z_score,
        p_value=p_value,
        adjusted_alpha=0.0,
        status="pending",
        note=note,
    )


def _previous_exact_formula(draws: list[Draw]) -> FormulaTest:
    hits = sum(1 for previous, current in zip(draws, draws[1:]) if previous.number == current.number)
    rounds = max(0, len(draws) - 1)
    return _formula_result(
        "上一期原样重复",
        "n[t] = n[t-1]",
        train_rounds=0,
        train_hits=0,
        test_rounds=rounds,
        test_hits=hits,
        note="检测完全重复上一期号码。",
    )


def _rolling_exact_mode_formula(draws: list[Draw], min_history: int) -> FormulaTest:
    counter = Counter(draw.number for draw in draws[:min_history])
    hits = 0
    rounds = 0
    for index in range(min_history, len(draws)):
        prediction = counter.most_common(1)[0][0]
        actual = draws[index].number
        hits += int(prediction == actual)
        rounds += 1
        counter[actual] += 1
    return _formula_result(
        "滚动最高频直选号",
        "n[t] = 历史最高频直选号码",
        train_rounds=min_history,
        train_hits=0,
        test_rounds=rounds,
        test_hits=hits,
        note="每期只用当期之前的数据更新。",
    )


def _rolling_position_mode_formula(draws: list[Draw], min_history: int) -> FormulaTest:
    counters = [
        Counter(draw.features.digits[position] for draw in draws[:min_history])
        for position in range(3)
    ]
    hits = 0
    rounds = 0
    for index in range(min_history, len(draws)):
        prediction = "".join(str(counter.most_common(1)[0][0]) for counter in counters)
        actual = draws[index]
        hits += int(prediction == actual.number)
        rounds += 1
        for position, digit in enumerate(actual.features.digits):
            counters[position][digit] += 1
    return _formula_result(
        "滚动位置热号组合",
        "n[t] = 各位置历史最高频数字拼接",
        train_rounds=min_history,
        train_hits=0,
        test_rounds=rounds,
        test_hits=hits,
        note="这是最朴素的热号公式。",
    )


def _rolling_transition_mode_formula(draws: list[Draw], min_history: int) -> FormulaTest:
    position_counts = [
        Counter(draw.features.digits[position] for draw in draws[:min_history])
        for position in range(3)
    ]
    transitions: list[dict[int, Counter]] = [defaultdict(Counter) for _position in range(3)]
    for index in range(1, min_history):
        previous = draws[index - 1].features.digits
        current = draws[index].features.digits
        for position in range(3):
            transitions[position][previous[position]][current[position]] += 1

    hits = 0
    rounds = 0
    for index in range(min_history, len(draws)):
        previous_digits = draws[index - 1].features.digits
        predicted_digits: list[str] = []
        for position, previous_digit in enumerate(previous_digits):
            options = transitions[position].get(previous_digit, Counter())
            if options:
                predicted_digits.append(str(options.most_common(1)[0][0]))
            else:
                predicted_digits.append(str(position_counts[position].most_common(1)[0][0]))
        prediction = "".join(predicted_digits)
        actual = draws[index]
        hits += int(prediction == actual.number)
        rounds += 1
        for position, digit in enumerate(actual.features.digits):
            position_counts[position][digit] += 1
            transitions[position][previous_digits[position]][digit] += 1

    return _formula_result(
        "滚动Markov位置转移",
        "digit[t,pos] = argmax P(digit | previous_digit,pos)",
        train_rounds=min_history,
        train_hits=0,
        test_rounds=rounds,
        test_hits=hits,
        note="按每个位置的上一期数字预测下一期数字。",
    )


def _best_affine_mod1000_formula(draws: list[Draw], split_ratio: float) -> FormulaTest:
    values = [int(draw.number) for draw in draws]
    split = max(2, min(len(values) - 1, int(len(values) * split_ratio)))
    train_pairs = [(values[index - 1], values[index]) for index in range(1, split)]
    test_pairs = [(values[index - 1], values[index]) for index in range(split, len(values))]
    counts: Counter[tuple[int, int]] = Counter()
    for multiplier in range(RANK_TOTAL):
        for previous, current in train_pairs:
            offset = (current - multiplier * previous) % RANK_TOTAL
            counts[(multiplier, offset)] += 1

    (best_multiplier, best_offset), train_hits = counts.most_common(1)[0]
    test_hits = sum(
        1
        for previous, current in test_pairs
        if (best_multiplier * previous + best_offset) % RANK_TOTAL == current
    )
    return _formula_result(
        "最佳线性同余公式",
        f"n[t] = ({best_multiplier} * n[t-1] + {best_offset}) mod 1000",
        train_rounds=len(train_pairs),
        train_hits=train_hits,
        test_rounds=len(test_pairs),
        test_hits=test_hits,
        note="先在前段样本寻找最优a和b，再只看后段留出样本。",
    )


def _best_digit_affine_formula(draws: list[Draw], split_ratio: float) -> FormulaTest:
    split = max(2, min(len(draws) - 1, int(len(draws) * split_ratio)))
    configs: list[tuple[int, int]] = []
    train_hits_by_position: list[int] = []
    for position in range(3):
        counts: Counter[tuple[int, int]] = Counter()
        for index in range(1, split):
            previous = draws[index - 1].features.digits[position]
            current = draws[index].features.digits[position]
            for multiplier in range(10):
                offset = (current - multiplier * previous) % 10
                counts[(multiplier, offset)] += 1
        config, hits = counts.most_common(1)[0]
        configs.append(config)
        train_hits_by_position.append(hits)

    train_hits = 0
    for index in range(1, split):
        predicted = "".join(
            str((multiplier * draws[index - 1].features.digits[position] + offset) % 10)
            for position, (multiplier, offset) in enumerate(configs)
        )
        train_hits += int(predicted == draws[index].number)

    test_hits = 0
    for index in range(split, len(draws)):
        predicted = "".join(
            str((multiplier * draws[index - 1].features.digits[position] + offset) % 10)
            for position, (multiplier, offset) in enumerate(configs)
        )
        test_hits += int(predicted == draws[index].number)

    formula_parts = [
        f"d{position}=({multiplier}*prev+{offset}) mod 10"
        for position, (multiplier, offset) in enumerate(configs)
    ]
    return _formula_result(
        "最佳逐位线性公式",
        "; ".join(formula_parts),
        train_rounds=split - 1,
        train_hits=train_hits,
        test_rounds=len(draws) - split,
        test_hits=test_hits,
        note="每个位置独立拟合一条mod10线性公式，再组合成直选号码。",
    )


def build_formula_tests(
    draws: list[Draw],
    min_history: int,
    split_ratio: float,
) -> list[FormulaTest]:
    safe_min_history = max(2, min(min_history, len(draws) - 2))
    return [
        _previous_exact_formula(draws),
        _rolling_exact_mode_formula(draws, safe_min_history),
        _rolling_position_mode_formula(draws, safe_min_history),
        _rolling_transition_mode_formula(draws, safe_min_history),
        _best_affine_mod1000_formula(draws, split_ratio),
        _best_digit_affine_formula(draws, split_ratio),
    ]


def _apply_statuses(
    distribution_tests: list[DistributionTest],
    serial_tests: list[SerialTest],
    transition_tests: list[TransitionTest],
    formula_tests: list[FormulaTest],
    alpha: float,
) -> tuple[list[DistributionTest], list[SerialTest], list[TransitionTest], list[FormulaTest], float]:
    total_tests = (
        len(distribution_tests)
        + len(serial_tests)
        + len(transition_tests)
        + len(formula_tests)
    )
    adjusted_alpha = alpha / max(1, total_tests)
    distribution_tests = [
        replace(
            row,
            adjusted_alpha=adjusted_alpha,
            status=_p_status(row.p_value, alpha, adjusted_alpha),
        )
        for row in distribution_tests
    ]
    serial_tests = [
        replace(
            row,
            adjusted_alpha=adjusted_alpha,
            status=_p_status(row.p_value, alpha, adjusted_alpha),
        )
        for row in serial_tests
    ]
    transition_tests = [
        replace(
            row,
            adjusted_alpha=adjusted_alpha,
            status=_p_status(row.p_value, alpha, adjusted_alpha),
        )
        for row in transition_tests
    ]
    formula_tests = [
        replace(
            row,
            adjusted_alpha=adjusted_alpha,
            status=_formula_status(
                row.test_hits,
                row.expected_test_hits,
                row.p_value,
                alpha,
                adjusted_alpha,
            ),
        )
        for row in formula_tests
    ]
    return distribution_tests, serial_tests, transition_tests, formula_tests, adjusted_alpha


def run_law_check(
    draws: list[Draw],
    max_lag: int = 10,
    alpha: float = 0.05,
    min_formula_history: int = 300,
    split_ratio: float = 0.7,
) -> LawCheckReport:
    if len(draws) < 30:
        raise ValueError("law check needs at least 30 draw rows")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not 0.1 <= split_ratio <= 0.9:
        raise ValueError("split_ratio must be between 0.1 and 0.9")

    distribution_tests = build_distribution_tests(draws)
    serial_tests = build_serial_tests(draws, max_lag=max_lag)
    transition_tests = build_transition_tests(draws)
    formula_tests = build_formula_tests(
        draws,
        min_history=min_formula_history,
        split_ratio=split_ratio,
    )
    (
        distribution_tests,
        serial_tests,
        transition_tests,
        formula_tests,
        adjusted_alpha,
    ) = _apply_statuses(
        distribution_tests,
        serial_tests,
        transition_tests,
        formula_tests,
        alpha,
    )

    distribution_rejections = sum(
        1 for row in distribution_tests if row.status == "rejected_after_correction"
    )
    serial_rejections = sum(
        1 for row in serial_tests if row.status == "rejected_after_correction"
    )
    transition_rejections = sum(
        1 for row in transition_tests if row.status == "rejected_after_correction"
    )
    validated_formulas = sum(1 for row in formula_tests if row.status == "validated")
    best_formula = max(
        formula_tests,
        key=lambda row: (
            row.test_hits - row.expected_test_hits,
            row.z_score,
            row.test_hits,
        ),
    )
    if validated_formulas:
        conclusion_status = "formula_validated"
        verdict = "发现至少一个留出样本显著公式，需要继续做跨年份压力测试。"
    elif distribution_rejections or serial_rejections or transition_rejections:
        conclusion_status = "statistical_deviation_only"
        verdict = "发现统计偏差，但这些偏差还没有转化成可验证的直选公式。"
    else:
        conclusion_status = "random_compatible"
        verdict = "没有发现能通过数学检验的稳定公式，当前证据更接近随机独立序列。"

    latest = draws[-1]
    total_tests = (
        len(distribution_tests)
        + len(serial_tests)
        + len(transition_tests)
        + len(formula_tests)
    )
    summary = LawCheckSummary(
        rows=len(draws),
        latest_issue=latest.issue,
        latest_date=latest.draw_date.isoformat() if latest.draw_date else None,
        latest_number=latest.number,
        alpha=alpha,
        adjusted_alpha=adjusted_alpha,
        total_tests=total_tests,
        distribution_rejections=distribution_rejections,
        serial_rejections=serial_rejections,
        transition_rejections=transition_rejections,
        validated_formulas=validated_formulas,
        best_formula_name=best_formula.name,
        best_formula_hits=best_formula.test_hits,
        best_formula_expected_hits=best_formula.expected_test_hits,
        best_formula_p_value=best_formula.p_value,
        conclusion_status=conclusion_status,
        verdict=verdict,
    )
    return LawCheckReport(
        summary=summary,
        distribution_tests=distribution_tests,
        serial_tests=serial_tests,
        transition_tests=transition_tests,
        formula_tests=formula_tests,
    )


def save_law_check_reports(
    report: LawCheckReport,
    output_dir: str | Path,
    meta: dict,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "lawcheck_report.json"
    md_path = report_dir / "lawcheck_report.md"
    payload = {
        "meta": meta,
        "report": asdict(report),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_law_check_markdown(report, meta), encoding="utf-8")
    return json_path, md_path


def _format_p(value: float) -> str:
    if not isfinite(value):
        return "nan"
    if value < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _status_label(status: str) -> str:
    labels = {
        "rejected_after_correction": "显著偏离",
        "weak_signal": "弱信号",
        "compatible": "未显著",
        "validated": "留出验证通过",
        "weak_edge": "弱优势",
        "no_edge": "无优势",
        "no_statistical_edge": "未显著",
    }
    return labels.get(status, status)


def render_law_check_markdown(report: LawCheckReport, meta: dict) -> str:
    summary = report.summary
    lines = [
        "# 福彩3D历史规律数学检验",
        "",
        "## 数据范围",
        "",
        f"* 数据行数: {summary.rows}",
        f"* 最新期号: {summary.latest_issue}",
        f"* 最新日期: {summary.latest_date}",
        f"* 最新号码: {summary.latest_number}",
        f"* 显著性阈值 alpha: {summary.alpha:.3f}",
        f"* 多重检验修正阈值: {summary.adjusted_alpha:.6f}",
        "",
        "## 总结",
        "",
        f"* 结论状态: {summary.conclusion_status}",
        f"* 分布显著偏离数: {summary.distribution_rejections}",
        f"* 自相关显著偏离数: {summary.serial_rejections}",
        f"* 转移显著偏离数: {summary.transition_rejections}",
        f"* 留出验证通过公式数: {summary.validated_formulas}",
        f"* 最强公式: {summary.best_formula_name}",
        f"* 最强公式留出命中: {summary.best_formula_hits}",
        f"* 随机期望命中: {summary.best_formula_expected_hits:.2f}",
        f"* 最强公式 p_value: {_format_p(summary.best_formula_p_value)}",
        f"* 判读: {summary.verdict}",
        "",
        "## 分布检验",
        "",
        "| 项目 | p_value | 状态 | 最强偏差项 | 观察 | 期望 | chi2 |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in report.distribution_tests:
        lines.append(
            "| "
            f"{row.name} | {_format_p(row.p_value)} | {_status_label(row.status)} | "
            f"{row.strongest_bucket} | {row.strongest_observed} | "
            f"{row.strongest_expected:.2f} | {row.statistic:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 自相关检验",
            "",
            "| 项目 | p_value | 状态 | 最强lag | 最大相关绝对值 | Q |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in report.serial_tests:
        lines.append(
            "| "
            f"{row.name} | {_format_p(row.p_value)} | {_status_label(row.status)} | "
            f"{row.strongest_lag} | {row.max_abs_correlation:.4f} | {row.statistic:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 转移独立性检验",
            "",
            "| 项目 | p_value | 状态 | 最强转移 | 标准化残差 | chi2 |",
            "|---|---:|---|---|---:|---:|",
        ]
    )
    for row in report.transition_tests:
        lines.append(
            "| "
            f"{row.name} | {_format_p(row.p_value)} | {_status_label(row.status)} | "
            f"{row.strongest_transition} | {row.strongest_residual:.3f} | {row.statistic:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 公式留出检验",
            "",
            "| 公式 | 留出命中 | 随机期望 | 命中率 | lift | z | p_value | 状态 | 表达式 |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in sorted(report.formula_tests, key=lambda item: item.z_score, reverse=True):
        lines.append(
            "| "
            f"{row.name} | {row.test_hits}/{row.test_rounds} | "
            f"{row.expected_test_hits:.2f} | {row.test_hit_rate:.3%} | "
            f"{row.lift:.3f} | {row.z_score:.3f} | {_format_p(row.p_value)} | "
            f"{_status_label(row.status)} | {row.formula} |"
        )

    lines.extend(
        [
            "",
            "## 严格判读规则",
            "",
            "* 单个偏差只说明历史样本里有不均匀现象。",
            "* 能用于预测的公式必须在留出样本上显著优于随机直选 0.1%。",
            "* 发现偏差后还要跨年份复验，避免把历史噪声当成公式。",
        ]
    )
    return "\n".join(lines) + "\n"
