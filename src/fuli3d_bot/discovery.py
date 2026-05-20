from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from math import erfc, exp, log1p, sqrt
from pathlib import Path

from .models import Draw


RANK_TOTAL = 1000
DIGITS = tuple(range(10))


@dataclass(frozen=True)
class FormulaWeights:
    window: int
    hot: float
    transition: float
    sum_link: float
    span_link: float
    pattern_link: float
    omission: float
    repeat: float

    @property
    def name(self) -> str:
        parts = [
            f"w{self.window}",
            f"h{self.hot:g}",
            f"t{self.transition:g}",
            f"s{self.sum_link:g}",
            f"sp{self.span_link:g}",
            f"p{self.pattern_link:g}",
            f"o{self.omission:g}",
            f"r{self.repeat:g}",
        ]
        return "_".join(parts).replace("-", "m").replace(".", "p")

    @property
    def expression(self) -> str:
        terms = [
            f"{self.hot:g}*position_hot",
            f"{self.transition:g}*transition",
            f"{self.sum_link:g}*prev_sum_link",
            f"{self.span_link:g}*prev_span_link",
            f"{self.pattern_link:g}*prev_pattern_link",
            f"{self.omission:g}*omission",
            f"{self.repeat:g}*repeat_flag",
        ]
        return (
            "digit[pos] = argmax_d score(pos,d); "
            f"window={self.window}; score="
            + " + ".join(terms)
        )


@dataclass(frozen=True)
class SegmentMetrics:
    rounds: int
    hits: int
    expected_hits: float
    hit_rate: float
    expected_hit_rate: float
    lift: float
    z_score: float
    p_value: float


@dataclass(frozen=True)
class DiscoveredFormula:
    name: str
    expression: str
    window: int
    weights: FormulaWeights
    train: SegmentMetrics
    validation: SegmentMetrics
    test: SegmentMetrics
    total: SegmentMetrics
    status: str
    note: str


@dataclass(frozen=True)
class DiscoverySummary:
    rows: int
    latest_issue: str
    latest_date: str | None
    latest_number: str
    min_history: int
    windows: list[int]
    formulas_searched: int
    train_rounds: int
    validation_rounds: int
    test_rounds: int
    alpha: float
    validation_adjusted_alpha: float
    selected_formula: str
    selected_expression: str
    selected_validation_hits: int
    selected_validation_expected_hits: float
    selected_validation_p_value: float
    selected_test_hits: int
    selected_test_expected_hits: float
    selected_test_p_value: float
    selected_test_lift: float
    conclusion_status: str
    verdict: str


@dataclass(frozen=True)
class DiscoveryReport:
    summary: DiscoverySummary
    top_formulas: list[DiscoveredFormula]


@dataclass(frozen=True)
class _FormulaCounts:
    weights: FormulaWeights
    train_hits: int
    validation_hits: int
    test_hits: int
    total_hits: int


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
        return 1.0

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


def _segment_metrics(rounds: int, hits: int, probability: float = 1.0 / RANK_TOTAL) -> SegmentMetrics:
    expected_hits = rounds * probability
    variance = rounds * probability * (1.0 - probability)
    z_score = (hits - expected_hits) / sqrt(variance) if variance > 0.0 else 0.0
    return SegmentMetrics(
        rounds=rounds,
        hits=hits,
        expected_hits=expected_hits,
        hit_rate=hits / rounds if rounds else 0.0,
        expected_hit_rate=probability,
        lift=hits / expected_hits if expected_hits else 0.0,
        z_score=z_score,
        p_value=_binomial_upper_tail(hits, rounds, probability),
    )


def _split_sizes(total_rounds: int) -> tuple[int, int, int]:
    train_rounds = int(total_rounds * 0.6)
    validation_rounds = int(total_rounds * 0.2)
    test_rounds = total_rounds - train_rounds - validation_rounds
    return train_rounds, validation_rounds, test_rounds


def _segment_index(row_index: int, train_rounds: int, validation_rounds: int) -> str:
    if row_index < train_rounds:
        return "train"
    if row_index < train_rounds + validation_rounds:
        return "validation"
    return "test"


def _ratio(count: int, total: int, bucket_count: int = 10) -> float:
    prior = 0.5
    return (count + prior) / (total + prior * bucket_count) if total >= 0 else 0.0


def _omission(draws: list[Draw], index: int, position: int, digit: int, window: int) -> float:
    start = max(0, index - window)
    gap = 0
    for cursor in range(index - 1, start - 1, -1):
        gap += 1
        if draws[cursor].features.digits[position] == digit:
            return min(gap, window) / window
    return 1.0


def _window_components(draws: list[Draw], index: int, window: int) -> list[list[tuple[float, float, float, float, float, float, float]]]:
    start = max(0, index - window)
    history = draws[start:index]
    history_size = len(history)
    previous = draws[index - 1].features

    position_counts = [Counter(draw.features.digits[position] for draw in history) for position in range(3)]
    transition_counts = [defaultdict(Counter) for _position in range(3)]
    transition_totals = [Counter() for _position in range(3)]
    sum_counts = [defaultdict(Counter) for _position in range(3)]
    sum_totals = Counter()
    span_counts = [defaultdict(Counter) for _position in range(3)]
    span_totals = Counter()
    pattern_counts = [defaultdict(Counter) for _position in range(3)]
    pattern_totals = Counter()

    pair_start = max(1, start + 1)
    for cursor in range(pair_start, index):
        previous_features = draws[cursor - 1].features
        current_features = draws[cursor].features
        sum_totals[previous_features.sum_value] += 1
        span_totals[previous_features.span] += 1
        pattern_totals[previous_features.pattern] += 1
        for position in range(3):
            previous_digit = previous_features.digits[position]
            current_digit = current_features.digits[position]
            transition_counts[position][previous_digit][current_digit] += 1
            transition_totals[position][previous_digit] += 1
            sum_counts[position][previous_features.sum_value][current_digit] += 1
            span_counts[position][previous_features.span][current_digit] += 1
            pattern_counts[position][previous_features.pattern][current_digit] += 1

    result: list[list[tuple[float, float, float, float, float, float, float]]] = []
    for position in range(3):
        previous_digit = previous.digits[position]
        position_result = []
        for digit in DIGITS:
            position_result.append(
                (
                    _ratio(position_counts[position][digit], history_size),
                    _ratio(
                        transition_counts[position][previous_digit][digit],
                        transition_totals[position][previous_digit],
                    ),
                    _ratio(
                        sum_counts[position][previous.sum_value][digit],
                        sum_totals[previous.sum_value],
                    ),
                    _ratio(
                        span_counts[position][previous.span][digit],
                        span_totals[previous.span],
                    ),
                    _ratio(
                        pattern_counts[position][previous.pattern][digit],
                        pattern_totals[previous.pattern],
                    ),
                    _omission(draws, index, position, digit, window),
                    1.0 if digit == previous_digit else 0.0,
                )
            )
        result.append(position_result)
    return result


def _predict_number(
    components: list[list[tuple[float, float, float, float, float, float, float]]],
    weights: FormulaWeights,
) -> str:
    predicted_digits: list[str] = []
    for position_components in components:
        best_digit = 0
        best_score = float("-inf")
        for digit, values in enumerate(position_components):
            score = (
                values[0] * weights.hot
                + values[1] * weights.transition
                + values[2] * weights.sum_link
                + values[3] * weights.span_link
                + values[4] * weights.pattern_link
                + values[5] * weights.omission
                + values[6] * weights.repeat
            )
            if score > best_score:
                best_score = score
                best_digit = digit
        predicted_digits.append(str(best_digit))
    return "".join(predicted_digits)


def generate_formula_weights(windows: list[int]) -> list[FormulaWeights]:
    formulas: list[FormulaWeights] = []
    for window in windows:
        for hot in (0.0, 1.0, 2.0):
            for transition in (0.0, 1.0, 2.0):
                for sum_link in (0.0, 1.0):
                    for span_link in (0.0, 1.0):
                        for pattern_link in (0.0, 1.0):
                            for omission in (0.0, 0.5):
                                for repeat in (0.0, -0.5):
                                    if not any(
                                        abs(value) > 1e-12
                                        for value in (
                                            hot,
                                            transition,
                                            sum_link,
                                            span_link,
                                            pattern_link,
                                            omission,
                                            repeat,
                                        )
                                    ):
                                        continue
                                    formulas.append(
                                        FormulaWeights(
                                            window=window,
                                            hot=hot,
                                            transition=transition,
                                            sum_link=sum_link,
                                            span_link=span_link,
                                            pattern_link=pattern_link,
                                            omission=omission,
                                            repeat=repeat,
                                        )
                                    )
    return formulas


def _evaluate_window_formulas(
    draws: list[Draw],
    formulas: list[FormulaWeights],
    window: int,
    min_history: int,
    train_rounds: int,
    validation_rounds: int,
) -> list[_FormulaCounts]:
    counts = {
        formula.name: [0, 0, 0, 0]
        for formula in formulas
    }
    for row_index, draw_index in enumerate(range(min_history, len(draws))):
        components = _window_components(draws, draw_index, window)
        actual = draws[draw_index].number
        segment = _segment_index(row_index, train_rounds, validation_rounds)
        for formula in formulas:
            prediction = _predict_number(components, formula)
            if prediction != actual:
                continue
            hit_counts = counts[formula.name]
            if segment == "train":
                hit_counts[0] += 1
            elif segment == "validation":
                hit_counts[1] += 1
            else:
                hit_counts[2] += 1
            hit_counts[3] += 1

    return [
        _FormulaCounts(
            weights=formula,
            train_hits=counts[formula.name][0],
            validation_hits=counts[formula.name][1],
            test_hits=counts[formula.name][2],
            total_hits=counts[formula.name][3],
        )
        for formula in formulas
    ]


def _counts_to_formula(
    counts: _FormulaCounts,
    train_rounds: int,
    validation_rounds: int,
    test_rounds: int,
    adjusted_alpha: float,
    alpha: float,
) -> DiscoveredFormula:
    train = _segment_metrics(train_rounds, counts.train_hits)
    validation = _segment_metrics(validation_rounds, counts.validation_hits)
    test = _segment_metrics(test_rounds, counts.test_hits)
    total = _segment_metrics(train_rounds + validation_rounds + test_rounds, counts.total_hits)
    if validation.hits <= validation.expected_hits:
        status = "rejected_on_validation"
        note = "验证段没有超过随机期望。"
    elif validation.p_value >= adjusted_alpha:
        status = "validation_not_significant"
        note = "验证段没有通过搜索规模修正。"
    elif test.hits <= test.expected_hits:
        status = "failed_on_test"
        note = "测试段回落到随机期望以下。"
    elif test.p_value >= alpha:
        status = "test_not_significant"
        note = "测试段没有达到独立显著性。"
    else:
        status = "candidate_formula"
        note = "验证段和测试段均显示正向优势，需要继续跨年份复验。"
    return DiscoveredFormula(
        name=counts.weights.name,
        expression=counts.weights.expression,
        window=counts.weights.window,
        weights=counts.weights,
        train=train,
        validation=validation,
        test=test,
        total=total,
        status=status,
        note=note,
    )


def _formula_sort_key(formula: DiscoveredFormula) -> tuple[float, float, int, float]:
    return (
        formula.validation.z_score,
        formula.test.z_score,
        formula.validation.hits,
        formula.total.z_score,
    )


def run_formula_discovery(
    draws: list[Draw],
    windows: list[int] | None = None,
    min_history: int = 300,
    alpha: float = 0.05,
    show_top: int = 20,
) -> DiscoveryReport:
    if len(draws) <= min_history + 30:
        raise ValueError("formula discovery needs more rows than min_history plus 30")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    active_windows = sorted(set(windows or [30, 60, 120, 240]))
    for window in active_windows:
        if window <= 0:
            raise ValueError("windows must be positive")

    total_rounds = len(draws) - min_history
    train_rounds, validation_rounds, test_rounds = _split_sizes(total_rounds)
    formulas = generate_formula_weights(active_windows)
    adjusted_alpha = alpha / max(1, len(formulas))
    grouped: dict[int, list[FormulaWeights]] = defaultdict(list)
    for formula in formulas:
        grouped[formula.window].append(formula)

    evaluated: list[DiscoveredFormula] = []
    for window in active_windows:
        formula_counts = _evaluate_window_formulas(
            draws,
            grouped[window],
            window=window,
            min_history=min_history,
            train_rounds=train_rounds,
            validation_rounds=validation_rounds,
        )
        evaluated.extend(
            _counts_to_formula(
                counts,
                train_rounds=train_rounds,
                validation_rounds=validation_rounds,
                test_rounds=test_rounds,
                adjusted_alpha=adjusted_alpha,
                alpha=alpha,
            )
            for counts in formula_counts
        )

    ranked = sorted(evaluated, key=_formula_sort_key, reverse=True)
    selected = ranked[0]
    if selected.status == "candidate_formula":
        conclusion_status = "candidate_found"
        verdict = "发现一个通过验证段和测试段的候选公式，下一步必须做跨年份和更新期追踪。"
    elif selected.validation.p_value < alpha and selected.test.hits > selected.test.expected_hits:
        conclusion_status = "weak_candidate"
        verdict = "发现弱候选公式，但没有通过搜索规模修正，风险主要是过拟合。"
    else:
        conclusion_status = "no_valid_formula"
        verdict = "搜索空间内没有发现能证明优于随机直选的稳定公式。"

    latest = draws[-1]
    summary = DiscoverySummary(
        rows=len(draws),
        latest_issue=latest.issue,
        latest_date=latest.draw_date.isoformat() if latest.draw_date else None,
        latest_number=latest.number,
        min_history=min_history,
        windows=active_windows,
        formulas_searched=len(formulas),
        train_rounds=train_rounds,
        validation_rounds=validation_rounds,
        test_rounds=test_rounds,
        alpha=alpha,
        validation_adjusted_alpha=adjusted_alpha,
        selected_formula=selected.name,
        selected_expression=selected.expression,
        selected_validation_hits=selected.validation.hits,
        selected_validation_expected_hits=selected.validation.expected_hits,
        selected_validation_p_value=selected.validation.p_value,
        selected_test_hits=selected.test.hits,
        selected_test_expected_hits=selected.test.expected_hits,
        selected_test_p_value=selected.test.p_value,
        selected_test_lift=selected.test.lift,
        conclusion_status=conclusion_status,
        verdict=verdict,
    )
    return DiscoveryReport(
        summary=summary,
        top_formulas=ranked[:show_top],
    )


def save_discovery_reports(
    report: DiscoveryReport,
    output_dir: str | Path,
    meta: dict,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "formula_discovery_report.json"
    md_path = report_dir / "formula_discovery_report.md"
    payload = {
        "meta": meta,
        "report": asdict(report),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_discovery_markdown(report, meta), encoding="utf-8")
    return json_path, md_path


def _format_p(value: float) -> str:
    if value < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _metric_cell(metrics: SegmentMetrics) -> str:
    return (
        f"{metrics.hits}/{metrics.rounds} "
        f"exp {metrics.expected_hits:.2f}, "
        f"lift {metrics.lift:.2f}, "
        f"p {_format_p(metrics.p_value)}"
    )


def render_discovery_markdown(report: DiscoveryReport, meta: dict) -> str:
    summary = report.summary
    lines = [
        "# 自定义公式探索报告",
        "",
        "## 数据范围",
        "",
        f"* 数据行数: {summary.rows}",
        f"* 最新期号: {summary.latest_issue}",
        f"* 最新日期: {summary.latest_date}",
        f"* 最新号码: {summary.latest_number}",
        f"* 最小历史窗口: {summary.min_history}",
        f"* 搜索窗口: {', '.join(str(item) for item in summary.windows)}",
        f"* 搜索公式数: {summary.formulas_searched}",
        f"* 验证段修正阈值: {summary.validation_adjusted_alpha:.8f}",
        "",
        "## 结论",
        "",
        f"* 结论状态: {summary.conclusion_status}",
        f"* 选中公式: {summary.selected_formula}",
        f"* 表达式: {summary.selected_expression}",
        f"* 验证命中: {summary.selected_validation_hits}, 期望 {summary.selected_validation_expected_hits:.2f}, p {_format_p(summary.selected_validation_p_value)}",
        f"* 测试命中: {summary.selected_test_hits}, 期望 {summary.selected_test_expected_hits:.2f}, lift {summary.selected_test_lift:.2f}, p {_format_p(summary.selected_test_p_value)}",
        f"* 判读: {summary.verdict}",
        "",
        "## 候选公式排行榜",
        "",
        "| 排名 | 名称 | 状态 | 训练 | 验证 | 测试 | 表达式 |",
        "|---:|---|---|---|---|---|---|",
    ]
    for index, formula in enumerate(report.top_formulas, start=1):
        lines.append(
            "| "
            f"{index} | {formula.name} | {formula.status} | "
            f"{_metric_cell(formula.train)} | {_metric_cell(formula.validation)} | "
            f"{_metric_cell(formula.test)} | {formula.expression} |"
        )
    lines.extend(
        [
            "",
            "## 纪律",
            "",
            "* 公式发现只允许使用开奖前的滚动历史。",
            "* 公式筛选按验证段排序，测试段只用于最终压力检查。",
            "* 若测试段没有显著优势，该公式只能继续观察。",
        ]
    )
    return "\n".join(lines) + "\n"
