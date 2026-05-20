from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .features import pattern_label
from .models import Draw, Recommendation
from .rules import RuleFilter, RuleRecencyRow, matches_filters, parse_rule_filters, rule_label, run_rule_recency
from .strategy import StrategyConfig, rank_numbers


ACTIVE = "active"
WATCH = "watch"
BLOCKED = "blocked"


@dataclass(frozen=True)
class RuleGateThresholds:
    min_hits: int = 1
    min_lift: float = 1.0
    min_roi: float = -0.48


@dataclass(frozen=True)
class RuleGateStatus:
    rule: str
    status: str
    passed_windows: list[int]
    failed_windows: list[int]
    reasons: list[str]
    rows: list[RuleRecencyRow]


@dataclass(frozen=True)
class GateCandidate:
    rank: int
    model_rank: int
    number: str
    score: float
    sum_value: int
    span: int
    pattern: str
    matched_rules: list[str]
    reasons: list[str]


@dataclass(frozen=True)
class GateReport:
    recommendation_mode: str
    active_rules: list[str]
    watch_rules: list[str]
    blocked_rules: list[str]
    statuses: list[RuleGateStatus]
    gated_candidates: list[GateCandidate]
    base_candidates: list[GateCandidate]


def _status_rank(status: str) -> int:
    return {ACTIVE: 0, WATCH: 1, BLOCKED: 2}.get(status, 99)


def _window_fail_reasons(row: RuleRecencyRow, thresholds: RuleGateThresholds) -> list[str]:
    reasons: list[str] = []
    if row.hits < thresholds.min_hits:
        reasons.append(f"{row.window_size}期命中{row.hits}低于{thresholds.min_hits}")
    if row.hit_lift < thresholds.min_lift:
        reasons.append(
            f"{row.window_size}期lift {row.hit_lift:.3f}低于{thresholds.min_lift:.3f}"
        )
    if row.roi < thresholds.min_roi:
        reasons.append(f"{row.window_size}期ROI {row.roi:.2%}低于{thresholds.min_roi:.2%}")
    return reasons


def _passes_window(row: RuleRecencyRow, thresholds: RuleGateThresholds) -> bool:
    return not _window_fail_reasons(row, thresholds)


def evaluate_rule_gate(
    rows: list[RuleRecencyRow],
    gate_windows: list[int],
    thresholds: RuleGateThresholds | None = None,
) -> list[RuleGateStatus]:
    if not gate_windows:
        raise ValueError("gate_windows cannot be empty")

    active_thresholds = thresholds or RuleGateThresholds()
    required_windows = sorted(set(gate_windows))
    rows_by_rule: dict[str, list[RuleRecencyRow]] = {}
    for row in rows:
        rows_by_rule.setdefault(row.rule, []).append(row)

    statuses: list[RuleGateStatus] = []
    for rule, rule_rows in rows_by_rule.items():
        by_window = {row.window_size: row for row in rule_rows}
        passed_windows: list[int] = []
        failed_windows: list[int] = []
        reasons: list[str] = []

        for window_size in required_windows:
            row = by_window.get(window_size)
            if row is None:
                failed_windows.append(window_size)
                reasons.append(f"{window_size}期缺少检测结果")
                continue
            fail_reasons = _window_fail_reasons(row, active_thresholds)
            if fail_reasons:
                failed_windows.append(window_size)
                reasons.extend(fail_reasons)
            else:
                passed_windows.append(window_size)

        non_gate_passed = sorted(
            row.window_size
            for row in rule_rows
            if row.window_size not in required_windows and _passes_window(row, active_thresholds)
        )
        if len(passed_windows) == len(required_windows):
            status = ACTIVE
            reasons = [f"通过{','.join(str(item) for item in required_windows)}期闸门"]
        elif passed_windows or non_gate_passed:
            status = WATCH
            if non_gate_passed:
                reasons.append(
                    f"非闸门窗口{','.join(str(item) for item in non_gate_passed)}达标，仅作观察"
                )
            if not reasons:
                reasons = ["有窗口达标，但短窗口组合仍未全部通过"]
        else:
            status = BLOCKED
            if not reasons:
                reasons = ["所有检测窗口均未达到闸门阈值"]

        statuses.append(
            RuleGateStatus(
                rule=rule,
                status=status,
                passed_windows=passed_windows,
                failed_windows=failed_windows,
                reasons=reasons,
                rows=sorted(rule_rows, key=lambda item: item.window_size),
            )
        )

    statuses.sort(key=lambda item: (_status_rank(item.status), item.rule))
    return statuses


def _candidate_from_recommendation(
    item: Recommendation,
    rank: int,
    matched_rules: list[str],
) -> GateCandidate:
    features = item.features
    return GateCandidate(
        rank=rank,
        model_rank=item.rank,
        number=item.number,
        score=item.score,
        sum_value=features.sum_value,
        span=features.span,
        pattern=pattern_label(features.pattern),
        matched_rules=matched_rules,
        reasons=item.reasons,
    )


def _compiled_rules(rules: list[str]) -> list[tuple[str, list[RuleFilter]]]:
    compiled: list[tuple[str, list[RuleFilter]]] = []
    for rule in rules:
        filters = parse_rule_filters(rule)
        compiled.append((rule_label(filters), filters))
    return compiled


def build_base_candidates(
    draws: list[Draw],
    top_n: int,
    config: StrategyConfig,
) -> list[GateCandidate]:
    ranked = rank_numbers(draws, top_n=top_n, config=config)
    return [
        _candidate_from_recommendation(item, rank=index + 1, matched_rules=[])
        for index, item in enumerate(ranked)
    ]


def build_gated_candidates(
    draws: list[Draw],
    active_rules: list[str],
    top_n: int,
    pool_size: int,
    config: StrategyConfig,
) -> list[GateCandidate]:
    if not active_rules:
        return []

    compiled_rules = _compiled_rules(active_rules)
    ranked_pool = rank_numbers(draws, top_n=pool_size, config=config)
    candidates: list[GateCandidate] = []
    for item in ranked_pool:
        matched_rules = [
            label for label, filters in compiled_rules if matches_filters(item.number, filters)
        ]
        if not matched_rules:
            continue
        candidates.append(
            _candidate_from_recommendation(
                item,
                rank=len(candidates) + 1,
                matched_rules=matched_rules,
            )
        )
        if len(candidates) >= top_n:
            break
    return candidates


def run_gate(
    draws: list[Draw],
    rules: list[str],
    windows: list[int],
    gate_windows: list[int],
    top_n: int = 20,
    pool_size: int = 200,
    training_window: int = 300,
    recent_window: int = 60,
    min_history: int = 300,
    thresholds: RuleGateThresholds | None = None,
) -> GateReport:
    rows = run_rule_recency(
        draws,
        rules=rules,
        windows=windows,
        top_n=top_n,
        pool_size=pool_size,
        training_window=training_window,
        recent_window=recent_window,
        min_history=min_history,
    )
    statuses = evaluate_rule_gate(rows, gate_windows=gate_windows, thresholds=thresholds)
    active_rules = [item.rule for item in statuses if item.status == ACTIVE]
    watch_rules = [item.rule for item in statuses if item.status == WATCH]
    blocked_rules = [item.rule for item in statuses if item.status == BLOCKED]

    config = StrategyConfig(recent_window=recent_window, min_history=min_history)
    gated_candidates = build_gated_candidates(
        draws,
        active_rules=active_rules,
        top_n=top_n,
        pool_size=pool_size,
        config=config,
    )
    base_candidates = build_base_candidates(draws, top_n=top_n, config=config)
    recommendation_mode = "rule_gated" if active_rules else "base_model_only"

    return GateReport(
        recommendation_mode=recommendation_mode,
        active_rules=active_rules,
        watch_rules=watch_rules,
        blocked_rules=blocked_rules,
        statuses=statuses,
        gated_candidates=gated_candidates,
        base_candidates=base_candidates,
    )


def save_gate_reports(
    report: GateReport,
    output_dir: str | Path,
    meta: dict,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "gate_report.json"
    md_path = report_dir / "gate_report.md"
    payload = {
        "meta": meta,
        "report": asdict(report),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_gate_markdown(report, meta), encoding="utf-8")
    return json_path, md_path


def render_gate_markdown(report: GateReport, meta: dict) -> str:
    lines = [
        "# 福彩3D规则闸门报告",
        "",
        "## 参数",
        "",
        f"* 数据行数: {meta.get('draw_rows')}",
        f"* rules: {meta.get('rules')}",
        f"* windows: {meta.get('windows')}",
        f"* gate_windows: {meta.get('gate_windows')}",
        f"* top_n: {meta.get('top_n')}",
        f"* pool_size: {meta.get('pool_size')}",
        f"* min_hits: {meta.get('min_hits')}",
        f"* min_lift: {meta.get('min_lift')}",
        f"* min_roi: {meta.get('min_roi')}",
        "",
        "## 闸门结论",
        "",
        f"* recommendation_mode: {report.recommendation_mode}",
        f"* active_rules: {report.active_rules}",
        f"* watch_rules: {report.watch_rules}",
        f"* blocked_rules: {report.blocked_rules}",
        "",
        "## 规则状态",
        "",
        "| status | rule | passed_windows | failed_windows | reasons |",
        "|---|---|---|---|---|",
    ]
    for status in report.statuses:
        lines.append(
            "| "
            f"{status.status} | {status.rule} | {status.passed_windows} | "
            f"{status.failed_windows} | {'; '.join(status.reasons)} |"
        )

    lines.extend(
        [
            "",
            "## 窗口明细",
            "",
            "| rule | window | rounds | avg_candidates | hits | exp_hits | lift | z | roi |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for status in report.statuses:
        for row in status.rows:
            lines.append(
                "| "
                f"{row.rule} | {row.window_size} | {row.rounds} | {row.avg_candidates:.2f} | "
                f"{row.hits} | {row.expected_hits:.2f} | {row.hit_lift:.3f} | "
                f"{row.hit_z_score:.3f} | {row.roi:.2%} |"
            )

    lines.extend(
        [
            "",
            "## 当前规则候选",
            "",
        ]
    )
    if report.gated_candidates:
        lines.extend(
            [
                "| rank | model_rank | number | score | features | matched_rules |",
                "|---:|---:|---|---:|---|---|",
            ]
        )
        for item in report.gated_candidates:
            lines.append(
                "| "
                f"{item.rank} | {item.model_rank} | {item.number} | {item.score:+.3f} | "
                f"和值{item.sum_value}, 跨度{item.span}, {item.pattern} | "
                f"{'; '.join(item.matched_rules)} |"
            )
    else:
        lines.append("* gated_candidates 为空，短窗口未通过的规则不参与当前过滤。")

    lines.extend(
        [
            "",
            "## 基础模型候选观察",
            "",
            "| rank | number | score | features |",
            "|---:|---|---:|---|",
        ]
    )
    for item in report.base_candidates:
        lines.append(
            "| "
            f"{item.rank} | {item.number} | {item.score:+.3f} | "
            f"和值{item.sum_value}, 跨度{item.span}, {item.pattern} |"
        )

    lines.extend(
        [
            "",
            "## 判读",
            "",
            "* active 需要所有 gate_windows 同时满足命中数、lift 和 ROI 阈值。",
            "* watch 说明存在局部信号，但短窗口证据还不足。",
            "* blocked 说明规则退潮信号明显，继续用于当前推荐会放大回看偏差。",
            "* base_model_only 只表示保留基础模型观察，不能当成规则增强推荐。",
        ]
    )
    return "\n".join(lines) + "\n"
