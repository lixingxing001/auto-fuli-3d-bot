from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import Draw
from .validation import validate_draws


@dataclass(frozen=True)
class DeliveryCheck:
    name: str
    status: str
    detail: str
    source: str


@dataclass(frozen=True)
class DeliveryReport:
    mode: str
    can_recommend: bool
    latest_issue: str
    latest_date: str | None
    latest_number: str
    checks: list[DeliveryCheck]
    verdict: str
    next_actions: list[str]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _check_data(draws: list[Draw]) -> DeliveryCheck:
    report = validate_draws(draws)
    if report.ok and report.rows >= 300:
        status = "pass"
        detail = (
            f"rows={report.rows}, duplicate=0, out_of_order=false, "
            f"issue_gap_count={len(report.issue_gaps)}, date_gap_count={len(report.date_gaps)}"
        )
    else:
        status = "fail"
        detail = (
            f"rows={report.rows}, duplicate={len(report.duplicate_issues)}, "
            f"out_of_order={str(report.out_of_order).lower()}"
        )
    return DeliveryCheck("data", status, detail, "data/history.csv")


def _check_gate(reports_dir: Path) -> DeliveryCheck:
    source = "reports/gate/gate_report.json"
    payload = _read_json(reports_dir / "gate" / "gate_report.json")
    if payload is None:
        return DeliveryCheck("gate", "missing", "缺少规则闸门报告", source)
    report = payload.get("report", {})
    active_rules = report.get("active_rules", [])
    mode = report.get("recommendation_mode", "")
    if active_rules:
        return DeliveryCheck("gate", "pass", f"active_rules={active_rules}, mode={mode}", source)
    return DeliveryCheck("gate", "fail", f"active_rules=[], mode={mode}", source)


def _check_calibration(reports_dir: Path) -> DeliveryCheck:
    source = "reports/calibration/calibration_report.json"
    payload = _read_json(reports_dir / "calibration" / "calibration_report.json")
    if payload is None:
        return DeliveryCheck("calibration", "missing", "缺少基础评分校准报告", source)
    summary = payload.get("report", {}).get("summary", {})
    signal_status = summary.get("signal_status", "")
    mean_rank = summary.get("mean_actual_rank", 0.0)
    top_lift = summary.get("top_bucket_lift", 0.0)
    if signal_status == "strong":
        status = "pass"
    elif signal_status == "weak":
        status = "watch"
    else:
        status = "fail"
    detail = f"signal_status={signal_status}, mean_actual_rank={mean_rank:.2f}, top_bucket_lift={top_lift:.3f}"
    return DeliveryCheck("calibration", status, detail, source)


def _check_ablation(reports_dir: Path) -> DeliveryCheck:
    source = "reports/ablation/ablation_report.json"
    payload = _read_json(reports_dir / "ablation" / "ablation_report.json")
    if payload is None:
        return DeliveryCheck("ablation", "missing", "缺少权重消融报告", source)
    summary = payload.get("report", {}).get("summary", {})
    status = "watch" if summary.get("recommendation_status") == "candidate" else "fail"
    detail = (
        f"best_case={summary.get('best_case')}, best_top_n={summary.get('best_top_n')}, "
        f"best_hit_z_score={summary.get('best_hit_z_score', 0.0):.3f}, "
        f"recommendation_status={summary.get('recommendation_status')}"
    )
    return DeliveryCheck("ablation", status, detail, source)


def _check_variantstress(reports_dir: Path) -> DeliveryCheck:
    source = "reports/variantstress/variantstress_report.json"
    payload = _read_json(reports_dir / "variantstress" / "variantstress_report.json")
    if payload is None:
        return DeliveryCheck("variantstress", "missing", "缺少变体压力测试报告", source)
    summary = payload.get("report", {}).get("summary", {})
    status = "pass" if summary.get("delivery_status") == "recommendation_candidate" else "fail"
    detail = (
        f"best_variant={summary.get('best_variant')}, best_top_n={summary.get('best_top_n')}, "
        f"annual_failures={summary.get('annual_failures')}, recent_failures={summary.get('recent_failures')}, "
        f"delivery_status={summary.get('delivery_status')}"
    )
    return DeliveryCheck("variantstress", status, detail, source)


def _check_targetcoverage(reports_dir: Path) -> DeliveryCheck:
    source = "reports/targetcoverage/targetcoverage_report.json"
    payload = _read_json(reports_dir / "targetcoverage" / "targetcoverage_report.json")
    if payload is None:
        return DeliveryCheck("targetcoverage", "missing", "缺少目标覆盖率报告", source)
    summary = payload.get("report", {}).get("summary", {})
    status = "coverage_only"
    detail = (
        f"target_rate={summary.get('target_rate', 0.0):.2%}, "
        f"required_top_n={summary.get('required_top_n')}, "
        f"expected_loss_per_draw={summary.get('expected_loss_per_draw', 0.0):.2f}, "
        f"feasibility_status={summary.get('feasibility_status')}"
    )
    return DeliveryCheck("targetcoverage", status, detail, source)


def build_delivery_report(draws: list[Draw], reports_dir: str | Path = "reports") -> DeliveryReport:
    root = Path(reports_dir)
    checks = [
        _check_data(draws),
        _check_gate(root),
        _check_calibration(root),
        _check_ablation(root),
        _check_variantstress(root),
        _check_targetcoverage(root),
    ]
    by_name = {item.name: item for item in checks}
    can_recommend = (
        by_name["data"].status == "pass"
        and by_name["gate"].status == "pass"
        and by_name["calibration"].status == "pass"
        and by_name["variantstress"].status == "pass"
    )
    latest = draws[-1]
    if can_recommend:
        mode = "recommendation_ready"
        verdict = "当前证据通过推荐闸门，可进入下一轮人工验收。"
        next_actions = [
            "复跑 fetch、gate、calibration、variantstress，确认数据刷新后结论未变。",
            "输出候选前保留报告路径和参数，禁止脱离报告直接使用号码。",
        ]
    else:
        mode = "analysis_only"
        verdict = "当前可交付为分析观察工具，推荐输出保持关闭。"
        next_actions = [
            "每日先刷新数据，再复跑 gate、calibration、variantstress。",
            "只有 active_rules、strong calibration、recommendation_candidate 同时成立，才进入推荐验收。",
            "继续收集样本，重点观察 2026 年窗口能否改善。",
        ]

    return DeliveryReport(
        mode=mode,
        can_recommend=can_recommend,
        latest_issue=latest.issue,
        latest_date=latest.draw_date.isoformat() if latest.draw_date else None,
        latest_number=latest.number,
        checks=checks,
        verdict=verdict,
        next_actions=next_actions,
    )


def save_delivery_report(
    report: DeliveryReport,
    output_dir: str | Path,
    meta: dict,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "delivery_report.json"
    md_path = report_dir / "delivery_report.md"
    payload = {
        "meta": meta,
        "report": asdict(report),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_delivery_markdown(report, meta), encoding="utf-8")
    return json_path, md_path


def render_delivery_markdown(report: DeliveryReport, meta: dict) -> str:
    lines = [
        "# 福彩3D交付状态报告",
        "",
        "## 参数",
        "",
        f"* 数据行数: {meta.get('draw_rows')}",
        f"* reports_dir: {meta.get('reports_dir')}",
        f"* latest_issue: {report.latest_issue}",
        f"* latest_date: {report.latest_date}",
        f"* latest_number: {report.latest_number}",
        "",
        "## 结论",
        "",
        f"* mode: {report.mode}",
        f"* can_recommend: {str(report.can_recommend).lower()}",
        f"* verdict: {report.verdict}",
        "",
        "## 检查项",
        "",
        "| check | status | detail | source |",
        "|---|---|---|---|",
    ]
    for item in report.checks:
        lines.append(f"| {item.name} | {item.status} | {item.detail} | {item.source} |")

    lines.extend(
        [
            "",
            "## 下一步",
            "",
        ]
    )
    for action in report.next_actions:
        lines.append(f"* {action}")

    lines.extend(
        [
            "",
            "## 交付边界",
            "",
            "* 当前交付物包括数据拉取、校验、回测、归因、规则闸门、校准、消融、变体压力测试和交付状态报告。",
            "* can_recommend=false 时，CLI 输出的号码只能作为评分观察样本。",
            "* 项目不包含真实购彩、自动下单、资金管理或任何外部执行入口。",
        ]
    )
    return "\n".join(lines) + "\n"
