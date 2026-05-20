from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .backtest import run_backtest
from .models import Draw
from .strategy import StrategyConfig


@dataclass(frozen=True)
class ExperimentCase:
    top_n: int
    training_window: int
    recent_window: int


@dataclass(frozen=True)
class ExperimentRow:
    rank: int
    top_n: int
    training_window: int
    recent_window: int
    rounds: int
    hits: int
    expected_hits: float
    hit_rate: float
    expected_hit_rate: float
    hit_lift: float
    hit_z_score: float
    roi: float
    expected_roi: float
    pnl: float
    pnl_vs_random_expected: float
    max_drawdown: float
    max_losing_streak: int


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("value must contain at least one integer")
    numbers = [int(item) for item in items]
    if any(number <= 0 for number in numbers):
        raise ValueError("all values must be positive integers")
    return numbers


def build_cases(
    top_values: list[int],
    training_windows: list[int],
    recent_windows: list[int],
) -> list[ExperimentCase]:
    return [
        ExperimentCase(top_n=top_n, training_window=training_window, recent_window=recent_window)
        for top_n in top_values
        for training_window in training_windows
        for recent_window in recent_windows
    ]


def run_experiments(
    draws: list[Draw],
    cases: list[ExperimentCase],
    min_history: int = 300,
    stake_per_number: float = 2.0,
    payout_per_hit: float = 1040.0,
) -> list[ExperimentRow]:
    rows: list[ExperimentRow] = []
    for case in cases:
        if len(draws) <= case.training_window:
            continue
        config = StrategyConfig(recent_window=case.recent_window, min_history=min_history)
        result = run_backtest(
            draws,
            top_n=case.top_n,
            training_window=case.training_window,
            stake_per_number=stake_per_number,
            payout_per_hit=payout_per_hit,
            config=config,
        )
        rows.append(
            ExperimentRow(
                rank=0,
                top_n=case.top_n,
                training_window=case.training_window,
                recent_window=case.recent_window,
                rounds=result.rounds,
                hits=result.hits,
                expected_hits=result.expected_hits,
                hit_rate=result.hit_rate,
                expected_hit_rate=result.expected_hit_rate,
                hit_lift=result.hit_lift,
                hit_z_score=result.hit_z_score,
                roi=result.roi,
                expected_roi=result.expected_roi,
                pnl=result.pnl,
                pnl_vs_random_expected=result.pnl_vs_random_expected,
                max_drawdown=result.max_drawdown,
                max_losing_streak=result.max_losing_streak,
            )
        )

    rows.sort(
        key=lambda item: (
            -item.hit_z_score,
            -item.pnl_vs_random_expected,
            -item.roi,
            item.max_losing_streak,
        )
    )
    return [
        ExperimentRow(rank=index + 1, **{k: v for k, v in asdict(row).items() if k != "rank"})
        for index, row in enumerate(rows)
    ]


def save_experiment_reports(
    rows: list[ExperimentRow],
    output_dir: str | Path,
    meta: dict,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "experiment_results.json"
    md_path = report_dir / "experiment_results.md"

    payload = {
        "meta": meta,
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(rows, meta), encoding="utf-8")
    return json_path, md_path


def render_markdown(rows: list[ExperimentRow], meta: dict) -> str:
    lines = [
        "# 福彩3D策略实验报告",
        "",
        "## 参数",
        "",
        f"* 数据行数: {meta.get('draw_rows')}",
        f"* top_n: {meta.get('top_values')}",
        f"* training_window: {meta.get('training_windows')}",
        f"* recent_window: {meta.get('recent_windows')}",
        f"* limit_rows: {meta.get('limit_rows')}",
        "",
        "## 排行",
        "",
        "| rank | top | train | recent | rounds | hits | exp_hits | lift | z | roi | pnl_vs_random | max_loss |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.rank} | {row.top_n} | {row.training_window} | {row.recent_window} | "
            f"{row.rounds} | {row.hits} | {row.expected_hits:.2f} | "
            f"{row.hit_lift:.3f} | {row.hit_z_score:.3f} | {row.roi:.2%} | "
            f"{row.pnl_vs_random_expected:.2f} | {row.max_losing_streak} |"
        )
    lines.extend(
        [
            "",
            "## 判读",
            "",
            "* 优先看 z 值和跨参数稳定性，单个 ROI 好看不够。",
            "* z 值低于 2 时，优势证据偏弱。",
            "* ROI 仍为负时，只能说明亏损低于随机基准，不能说明有正收益能力。",
        ]
    )
    return "\n".join(lines) + "\n"

