from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .features import extract_features
from .models import Draw
from .strategy import rank_numbers


@dataclass(frozen=True)
class ReviewRow:
    issue: str
    status: str
    predicted_number: str
    actual_number: str | None
    actual_date: str | None
    source_issue: str
    source_date: str | None
    action_label: str
    stake_level: str
    confidence_label: str
    strategy_label: str
    direct_hit: bool
    group_hit: bool
    top3_hit: bool
    top5_hit: bool
    actual_rank: int | None
    actual_score: float | None


@dataclass(frozen=True)
class ReviewSummary:
    snapshots: int
    reviewed: int
    pending: int
    direct_hits: int
    group_hits: int
    top3_hits: int
    top5_hits: int
    action_days: int
    action_direct_hits: int
    ranked: int
    mean_actual_rank: float | None
    median_actual_rank: float | None
    top100_count: int
    top500_count: int
    direct_hit_rate: float
    group_hit_rate: float
    action_direct_hit_rate: float
    top100_rate: float
    top500_rate: float


@dataclass(frozen=True)
class ReviewReport:
    summary: ReviewSummary
    rows: list[ReviewRow]


def _snapshot_files(predictions_dir: str | Path) -> list[Path]:
    path = Path(predictions_dir)
    if not path.exists():
        return []
    return sorted(path.glob("prediction_*.json"))


def _same_group(predicted: str, actual: str) -> bool:
    predicted_features = extract_features(predicted)
    actual_features = extract_features(actual)
    if predicted_features.pattern not in {"zuliu", "zusan"}:
        return False
    return (
        predicted_features.pattern == actual_features.pattern
        and sorted(predicted) == sorted(actual)
    )


def _load_snapshot(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rank_from_saved_ranking(report: dict, actual_number: str) -> tuple[int, float | None] | None:
    for item in report.get("full_ranking", []):
        if str(item.get("number")) == actual_number:
            rank = item.get("rank")
            if rank is None:
                return None
            score = item.get("score")
            return int(rank), float(score) if score is not None else None
    return None


def _draws_through_issue(draws: list[Draw], source_issue: str) -> list[Draw] | None:
    for index, draw in enumerate(draws):
        if draw.issue == source_issue:
            return draws[: index + 1]
    return None


def _strategy_config_from_snapshot(payload: dict):
    meta = payload.get("meta", {})
    report = payload.get("report", {})
    recent_window = int(meta.get("recent_window", 60))
    min_history = int(meta.get("min_history", 300))
    strategy_name = str(
        report.get("strategy_selection", {}).get("active_name", "baseline")
    )
    from .daily import _strategy_variants

    variants = {
        variant.name: variant.config
        for variant in _strategy_variants(
            recent_window=recent_window,
            min_history=min_history,
        )
    }
    return variants.get(strategy_name, variants["baseline"])


def _rank_from_rebuilt_ranking(
    payload: dict,
    actual_number: str,
    draws: list[Draw],
) -> tuple[int, float | None] | None:
    report = payload["report"]
    source_issue = str(report.get("latest_issue", ""))
    source_draws = _draws_through_issue(draws, source_issue)
    if source_draws is None:
        return None
    try:
        config = _strategy_config_from_snapshot(payload)
        recommendations = rank_numbers(source_draws, top_n=1000, config=config)
    except (KeyError, TypeError, ValueError):
        return None
    for item in recommendations:
        if item.number == actual_number:
            return item.rank, item.score
    return None


def _actual_rank(
    payload: dict,
    actual_number: str,
    draws: list[Draw],
) -> tuple[int, float | None] | None:
    report = payload["report"]
    saved_rank = _rank_from_saved_ranking(report, actual_number)
    if saved_rank is not None:
        return saved_rank
    return _rank_from_rebuilt_ranking(payload, actual_number, draws)


def _review_row(
    payload: dict,
    actual_by_issue: dict[str, Draw],
    draws: list[Draw],
) -> ReviewRow:
    report = payload["report"]
    issue = str(report["next_issue_hint"])
    primary = str(report["primary"]["number"])
    alternatives = [str(item["number"]) for item in report.get("alternatives", [])]
    actual = actual_by_issue.get(issue)
    actual_number = actual.number if actual else None
    candidates = [primary, *alternatives]
    status = "pending" if actual is None else "hit" if actual_number == primary else "miss"
    rank_result = _actual_rank(payload, actual_number, draws) if actual_number else None

    action_filter = report.get("action_filter", {})
    confidence_gate = report.get("confidence_gate", {})
    strategy_selection = report.get("strategy_selection", {})
    return ReviewRow(
        issue=issue,
        status=status,
        predicted_number=primary,
        actual_number=actual_number,
        actual_date=actual.draw_date.isoformat() if actual and actual.draw_date else None,
        source_issue=str(report.get("latest_issue", "")),
        source_date=report.get("latest_date"),
        action_label=str(action_filter.get("label", "")),
        stake_level=str(action_filter.get("stake_level", "")),
        confidence_label=str(confidence_gate.get("label", "")),
        strategy_label=str(strategy_selection.get("active_label", "")),
        direct_hit=bool(actual_number == primary) if actual_number else False,
        group_hit=_same_group(primary, actual_number) if actual_number else False,
        top3_hit=actual_number in candidates[:3] if actual_number else False,
        top5_hit=actual_number in candidates[:5] if actual_number else False,
        actual_rank=rank_result[0] if rank_result else None,
        actual_score=rank_result[1] if rank_result else None,
    )


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _summarize(rows: list[ReviewRow]) -> ReviewSummary:
    reviewed_rows = [row for row in rows if row.status != "pending"]
    action_rows = [row for row in reviewed_rows if row.action_label == "可小注"]
    direct_hits = sum(1 for row in reviewed_rows if row.direct_hit)
    group_hits = sum(1 for row in reviewed_rows if row.group_hit)
    action_direct_hits = sum(1 for row in action_rows if row.direct_hit)
    actual_ranks = [
        row.actual_rank
        for row in reviewed_rows
        if row.actual_rank is not None
    ]
    top100_count = sum(1 for rank in actual_ranks if rank <= 100)
    top500_count = sum(1 for rank in actual_ranks if rank <= 500)
    reviewed = len(reviewed_rows)
    action_days = len(action_rows)
    ranked = len(actual_ranks)
    return ReviewSummary(
        snapshots=len(rows),
        reviewed=reviewed,
        pending=sum(1 for row in rows if row.status == "pending"),
        direct_hits=direct_hits,
        group_hits=group_hits,
        top3_hits=sum(1 for row in reviewed_rows if row.top3_hit),
        top5_hits=sum(1 for row in reviewed_rows if row.top5_hit),
        action_days=action_days,
        action_direct_hits=action_direct_hits,
        ranked=ranked,
        mean_actual_rank=sum(actual_ranks) / ranked if ranked else None,
        median_actual_rank=_median(actual_ranks),
        top100_count=top100_count,
        top500_count=top500_count,
        direct_hit_rate=direct_hits / reviewed if reviewed else 0.0,
        group_hit_rate=group_hits / reviewed if reviewed else 0.0,
        action_direct_hit_rate=action_direct_hits / action_days if action_days else 0.0,
        top100_rate=top100_count / ranked if ranked else 0.0,
        top500_rate=top500_count / ranked if ranked else 0.0,
    )


def build_review_report(draws: list[Draw], predictions_dir: str | Path) -> ReviewReport:
    actual_by_issue = {draw.issue: draw for draw in draws}
    rows = [
        _review_row(_load_snapshot(path), actual_by_issue, draws)
        for path in _snapshot_files(predictions_dir)
    ]
    rows.sort(key=lambda row: row.issue)
    return ReviewReport(summary=_summarize(rows), rows=rows)


def save_review_report(
    report: ReviewReport,
    output_dir: str | Path,
    meta: dict,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "review_report.json"
    html_path = report_dir / "review_report.html"
    payload = {
        "meta": meta,
        "report": asdict(report),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_review_html(report, meta), encoding="utf-8")
    return json_path, html_path


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def _format_optional_number(value: float | None) -> str:
    return "暂无" if value is None else f"{value:.1f}"


def _rank_band(rank: int | None) -> str:
    if rank is None:
        return "无排名"
    if rank <= 100:
        return "前100"
    if rank <= 500:
        return "前500"
    return "后500"


def _rows_html(rows: list[ReviewRow]) -> str:
    if not rows:
        return "<tr><td colspan='11' class='empty'>暂无预测快照</td></tr>"
    html_rows = []
    for row in rows:
        actual = row.actual_number if row.actual_number is not None else "待开奖"
        hit = "命中" if row.direct_hit else "未命中" if row.status != "pending" else "待复盘"
        actual_rank = f"#{row.actual_rank}" if row.actual_rank is not None else "暂无"
        actual_score = (
            f"评分 {row.actual_score:.3f}" if row.actual_score is not None else _rank_band(row.actual_rank)
        )
        html_rows.append(
            "<tr>"
            f"<td>{row.issue}</td>"
            f"<td><strong>{row.predicted_number}</strong><span>{row.source_issue}</span></td>"
            f"<td>{actual}<span>{row.actual_date or ''}</span></td>"
            f"<td>{actual_rank}<span>{actual_score}</span></td>"
            f"<td>{row.action_label}<span>{row.stake_level}投入</span></td>"
            f"<td>{row.confidence_label}</td>"
            f"<td>{row.strategy_label}</td>"
            f"<td>{hit}</td>"
            f"<td>{'是' if row.group_hit else '否'}</td>"
            f"<td>{'是' if row.top3_hit else '否'}</td>"
            f"<td>{'是' if row.top5_hit else '否'}</td>"
            "</tr>"
        )
    return "\n".join(html_rows)


def render_review_html(report: ReviewReport, meta: dict) -> str:
    summary = report.summary
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>福彩3D预测复盘</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #65758b;
      --line: #d7dee8;
      --brand: #17324d;
      --gold: #9b7624;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.55;
    }}
    .topbar {{ background: var(--brand); color: #fff; border-bottom: 4px solid var(--gold); }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 0 24px; }}
    .topbar .wrap {{ min-height: 72px; display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
    h1 {{ margin: 0; font-size: 22px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 10px; margin-bottom: 18px; }}
    .card, .table-box {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 1px 2px rgba(31, 41, 51, 0.05); }}
    .card {{ padding: 14px; min-height: 96px; }}
    .card span, td span {{ display: block; color: var(--muted); font-size: 12px; }}
    .card strong {{ display: block; margin-top: 8px; font-size: 22px; }}
    .table-box {{ overflow-x: auto; }}
    table {{ width: 100%; min-width: 900px; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: #41526b; background: #f7f9fb; white-space: nowrap; }}
    tr:last-child td {{ border-bottom: 0; }}
    .empty {{ text-align: center; color: var(--muted); }}
    @media (max-width: 900px) {{
      .wrap, main {{ padding-left: 14px; padding-right: 14px; }}
      table {{ font-size: 12px; }}
    }}
  </style>
</head>
<body>
<div class="topbar">
  <div class="wrap">
    <h1>福彩3D预测复盘</h1>
    <span>快照目录: {meta.get("predictions_dir")}</span>
  </div>
</div>
<main>
  <section class="grid">
    <div class="card"><span>快照数</span><strong>{summary.snapshots}</strong></div>
    <div class="card"><span>已复盘</span><strong>{summary.reviewed}</strong></div>
    <div class="card"><span>待开奖</span><strong>{summary.pending}</strong></div>
    <div class="card"><span>直选命中率</span><strong>{_format_percent(summary.direct_hit_rate)}</strong></div>
    <div class="card"><span>出手日命中率</span><strong>{_format_percent(summary.action_direct_hit_rate)}</strong></div>
    <div class="card"><span>平均真实排名</span><strong>{_format_optional_number(summary.mean_actual_rank)}</strong></div>
    <div class="card"><span>真实号前500</span><strong>{_format_percent(summary.top500_rate)}</strong></div>
  </section>
  <div class="table-box">
    <table>
      <thead>
        <tr>
          <th>预测期号</th><th>预测号</th><th>开奖号</th><th>真实排名</th><th>出手建议</th><th>置信</th><th>策略</th><th>直选</th><th>组选</th><th>Top3</th><th>Top5</th>
        </tr>
      </thead>
      <tbody>{_rows_html(report.rows)}</tbody>
    </table>
  </div>
</main>
</body>
</html>
"""
