from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .backtest import model_ranker, run_backtest
from .baselines import BASELINE_RANKERS
from .models import BacktestResult, Draw
from .strategy import StrategyConfig


@dataclass(frozen=True)
class BenchmarkRow:
    ranker: str
    segment: str
    year: int | None
    month: int | None
    top_n: int
    training_window: int
    recent_window: int
    rounds: int
    hits: int
    expected_hits: float
    hit_rate: float
    hit_lift: float
    hit_z_score: float
    roi: float
    expected_roi: float
    pnl: float
    pnl_vs_random_expected: float
    max_drawdown: float
    max_losing_streak: int


def _row_from_result(
    ranker: str,
    segment: str,
    year: int | None,
    month: int | None,
    top_n: int,
    training_window: int,
    recent_window: int,
    result: BacktestResult,
) -> BenchmarkRow:
    return BenchmarkRow(
        ranker=ranker,
        segment=segment,
        year=year,
        month=month,
        top_n=top_n,
        training_window=training_window,
        recent_window=recent_window,
        rounds=result.rounds,
        hits=result.hits,
        expected_hits=result.expected_hits,
        hit_rate=result.hit_rate,
        hit_lift=result.hit_lift,
        hit_z_score=result.hit_z_score,
        roi=result.roi,
        expected_roi=result.expected_roi,
        pnl=result.pnl,
        pnl_vs_random_expected=result.pnl_vs_random_expected,
        max_drawdown=result.max_drawdown,
        max_losing_streak=result.max_losing_streak,
    )


def available_rankers(names: list[str]) -> dict:
    rankers = {"model": model_ranker}
    rankers.update(BASELINE_RANKERS)
    return {name: rankers[name] for name in names}


def draw_years(draws: list[Draw]) -> list[int]:
    return sorted({draw.draw_date.year for draw in draws if draw.draw_date is not None})


def draw_months(draws: list[Draw], year: int | None = None) -> list[tuple[int, int]]:
    return sorted(
        {
            (draw.draw_date.year, draw.draw_date.month)
            for draw in draws
            if draw.draw_date is not None and (year is None or draw.draw_date.year == year)
        }
    )


def run_benchmark(
    draws: list[Draw],
    ranker_names: list[str],
    top_n: int = 20,
    training_window: int = 300,
    recent_window: int = 60,
    min_history: int = 300,
    stake_per_number: float = 2.0,
    payout_per_hit: float = 1040.0,
    include_yearly: bool = True,
    include_monthly: bool = False,
    monthly_year: int | None = None,
) -> list[BenchmarkRow]:
    config = StrategyConfig(recent_window=recent_window, min_history=min_history)
    rows: list[BenchmarkRow] = []
    rankers = available_rankers(ranker_names)

    for name, ranker in rankers.items():
        full_result = run_backtest(
            draws,
            top_n=top_n,
            training_window=training_window,
            stake_per_number=stake_per_number,
            payout_per_hit=payout_per_hit,
            config=config,
            ranker=ranker,
        )
        rows.append(
            _row_from_result(
                name,
                "all",
                None,
                None,
                top_n,
                training_window,
                recent_window,
                full_result,
            )
        )

        if not include_yearly:
            yearly_items: list[int] = []
        else:
            yearly_items = draw_years(draws)

        for year in yearly_items:
            yearly_result = run_backtest(
                draws,
                top_n=top_n,
                training_window=training_window,
                stake_per_number=stake_per_number,
                payout_per_hit=payout_per_hit,
                config=config,
                ranker=ranker,
                eval_filter=lambda draw, active_year=year: (
                    draw.draw_date is not None and draw.draw_date.year == active_year
                ),
            )
            if yearly_result.rounds == 0:
                continue
            rows.append(
                _row_from_result(
                    name,
                    "year",
                    year,
                    None,
                    top_n,
                    training_window,
                    recent_window,
                    yearly_result,
                )
            )

        if not include_monthly:
            continue

        for year, month in draw_months(draws, monthly_year):
            monthly_result = run_backtest(
                draws,
                top_n=top_n,
                training_window=training_window,
                stake_per_number=stake_per_number,
                payout_per_hit=payout_per_hit,
                config=config,
                ranker=ranker,
                eval_filter=lambda draw, active_year=year, active_month=month: (
                    draw.draw_date is not None
                    and draw.draw_date.year == active_year
                    and draw.draw_date.month == active_month
                ),
            )
            if monthly_result.rounds == 0:
                continue
            rows.append(
                _row_from_result(
                    name,
                    "month",
                    year,
                    month,
                    top_n,
                    training_window,
                    recent_window,
                    monthly_result,
                )
            )

    return rows


def save_benchmark_reports(
    rows: list[BenchmarkRow],
    output_dir: str | Path,
    meta: dict,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "benchmark_results.json"
    md_path = report_dir / "benchmark_results.md"
    payload = {
        "meta": meta,
        "rows": [asdict(row) for row in rows],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(rows, meta), encoding="utf-8")
    return json_path, md_path


def render_markdown(rows: list[BenchmarkRow], meta: dict) -> str:
    overall = [row for row in rows if row.segment == "all"]
    yearly = [row for row in rows if row.segment == "year"]
    monthly = [row for row in rows if row.segment == "month"]
    overall.sort(key=lambda row: (-row.hit_z_score, -row.pnl_vs_random_expected))
    yearly.sort(key=lambda row: (row.year or 0, row.ranker))
    monthly.sort(key=lambda row: (row.year or 0, row.month or 0, row.ranker))

    lines = [
        "# 福彩3D基准对照报告",
        "",
        "## 参数",
        "",
        f"* 数据行数: {meta.get('draw_rows')}",
        f"* rankers: {meta.get('rankers')}",
        f"* top_n: {meta.get('top_n')}",
        f"* training_window: {meta.get('training_window')}",
        f"* recent_window: {meta.get('recent_window')}",
        f"* limit_rows: {meta.get('limit_rows')}",
        "",
        "## 全局排行",
        "",
        "| ranker | rounds | hits | exp_hits | lift | z | roi | pnl_vs_random | max_loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            "| "
            f"{row.ranker} | {row.rounds} | {row.hits} | {row.expected_hits:.2f} | "
            f"{row.hit_lift:.3f} | {row.hit_z_score:.3f} | {row.roi:.2%} | "
            f"{row.pnl_vs_random_expected:.2f} | {row.max_losing_streak} |"
        )

    lines.extend(
        [
            "",
            "## 年度分段",
            "",
            "| year | ranker | rounds | hits | exp_hits | lift | z | roi | max_loss |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in yearly:
        lines.append(
            "| "
            f"{row.year} | {row.ranker} | {row.rounds} | {row.hits} | "
            f"{row.expected_hits:.2f} | {row.hit_lift:.3f} | "
            f"{row.hit_z_score:.3f} | {row.roi:.2%} | {row.max_losing_streak} |"
        )

    if monthly:
        lines.extend(
            [
                "",
                "## 月度分段",
                "",
                "| month | ranker | rounds | hits | exp_hits | lift | z | roi | max_loss |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in monthly:
            lines.append(
                "| "
                f"{row.year}-{row.month:02d} | {row.ranker} | {row.rounds} | {row.hits} | "
                f"{row.expected_hits:.2f} | {row.hit_lift:.3f} | "
                f"{row.hit_z_score:.3f} | {row.roi:.2%} | {row.max_losing_streak} |"
            )

    lines.extend(
        [
            "",
            "## 判读",
            "",
            "* 综合策略必须赢过弱基准，才值得继续扩展。",
            "* 年度分段如果大面积低于随机期望，说明策略稳定性不足。",
            "* 固定随机基准只用于程序对照，数学期望仍以 expected 指标为准。",
        ]
    )
    return "\n".join(lines) + "\n"
