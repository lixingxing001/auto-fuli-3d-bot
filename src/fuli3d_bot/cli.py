from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .ablation import run_ablation, save_ablation_reports
from .attribution import parse_months, parse_years, run_hit_attribution, save_attribution_report
from .benchmark import run_benchmark, save_benchmark_reports
from .backtest import run_backtest
from .baselines import parse_ranker_names
from .calibration import run_calibration, save_calibration_reports
from .delivery import build_delivery_report, save_delivery_report
from .daily import build_daily_report, save_daily_report
from .experiment import build_cases, parse_int_list, run_experiments, save_experiment_reports
from .discovery import run_formula_discovery, save_discovery_reports
from .fetcher import fetch_and_write
from .features import pattern_label
from .gate import RuleGateThresholds, run_gate, save_gate_reports
from .lawcheck import run_law_check, save_law_check_reports
from .repository import load_draws
from .review import build_review_report, save_review_report
from .rules import (
    run_rule_backtests,
    run_rule_recency,
    save_rulebacktest_reports,
    save_rulerecency_reports,
)
from .stats import build_stats
from .strategy import StrategyConfig, rank_numbers, score_number
from .targetcoverage import run_target_coverage, save_target_coverage_reports
from .validation import validate_draws
from .variantstress import (
    parse_variant_names,
    run_variant_stress,
    save_variant_stress_reports,
)


DEFAULT_DATA = Path("data/history.csv")


def add_data_argument(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--data",
        default=argparse.SUPPRESS,
        help="CSV历史开奖文件",
    )


def _load(path: str) -> list:
    return load_draws(path)


def cmd_summary(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    stats = build_stats(draws, recent_window=args.recent_window)
    validation = validate_draws(draws)
    latest = draws[-1]
    output = {
        "rows": len(draws),
        "latest_issue": latest.issue,
        "latest_date": latest.draw_date.isoformat() if latest.draw_date else None,
        "latest_number": latest.number,
        "recent_window": stats.recent_window,
        "validation": {
            "ok": validation.ok,
            "duplicate_issues": validation.duplicate_issues,
            "issue_gap_count": len(validation.issue_gaps),
            "date_gap_count": len(validation.date_gaps),
            "out_of_order": validation.out_of_order,
        },
        "pattern_counts": dict(stats.pattern_counts),
        "sum_top": stats.sum_counts.most_common(10),
        "span_top": stats.span_counts.most_common(10),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    result = fetch_and_write(
        output_path=args.output,
        page_size=args.page_size,
        max_pages=args.max_pages,
        limit=args.limit,
        timeout=args.timeout,
    )
    validation = validate_draws(result.draws)
    latest = result.draws[-1]
    output = {
        "output": args.output,
        "rows": len(result.draws),
        "remote_total": result.total_remote,
        "pages_fetched": result.pages_fetched,
        "source_url": result.source_url,
        "latest_issue": latest.issue,
        "latest_date": latest.draw_date.isoformat() if latest.draw_date else None,
        "latest_number": latest.number,
        "validation": {
            "ok": validation.ok,
            "duplicate_issues": validation.duplicate_issues,
            "issue_gaps_preview": validation.issue_gaps[:20],
            "issue_gap_count": len(validation.issue_gaps),
            "date_gaps_preview": validation.date_gaps[:20],
            "date_gap_count": len(validation.date_gaps),
            "out_of_order": validation.out_of_order,
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    config = StrategyConfig(recent_window=args.recent_window, min_history=args.min_history)
    recommendations = rank_numbers(draws, top_n=args.top, config=config)
    print(f"数据行数: {len(draws)}")
    print(f"候选数量: {args.top}")
    print("风险提示: 历史特征只用于评分和回测，不能证明下一期开奖概率已经改变")
    print()
    for item in recommendations:
        features = item.features
        print(
            f"{item.rank:02d}. {item.number}  分数 {item.score:+.3f}  "
            f"和值 {features.sum_value:02d}  跨度 {features.span}  {pattern_label(features.pattern)}"
        )
        for reason in item.reasons:
            print(f"    * {reason}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    config = StrategyConfig(recent_window=args.recent_window, min_history=args.min_history)
    result = run_backtest(
        draws,
        top_n=args.top,
        training_window=args.training_window,
        stake_per_number=args.stake_per_number,
        payout_per_hit=args.payout_per_hit,
        config=config,
    )
    output = asdict(result)
    if not args.show_picks:
        output.pop("picks", None)
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    config = StrategyConfig(recent_window=args.recent_window, min_history=args.min_history)
    stats = build_stats(
        draws,
        recent_window=args.recent_window,
        exact_recent_window=config.exact_recent_window,
    )
    score, reasons, risk_notes = score_number(args.number, stats, config)
    print(f"号码: {args.number.zfill(3)}")
    print(f"分数: {score:+.3f}")
    print("评分来源:")
    for reason in reasons:
        print(f"  * {reason}")
    print("风险边界:")
    for note in risk_notes:
        print(f"  * {note}")
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    source_rows = len(draws)
    if args.limit_rows and args.limit_rows > 0:
        draws = draws[-args.limit_rows :]

    top_values = parse_int_list(args.top_values)
    training_windows = parse_int_list(args.training_windows)
    recent_windows = parse_int_list(args.recent_windows)
    cases = build_cases(top_values, training_windows, recent_windows)
    rows = run_experiments(
        draws,
        cases,
        min_history=args.min_history,
        stake_per_number=args.stake_per_number,
        payout_per_hit=args.payout_per_hit,
    )
    meta = {
        "source_rows": source_rows,
        "draw_rows": len(draws),
        "limit_rows": args.limit_rows,
        "top_values": top_values,
        "training_windows": training_windows,
        "recent_windows": recent_windows,
        "case_count": len(cases),
        "completed_count": len(rows),
    }
    json_path, md_path = save_experiment_reports(rows, args.output_dir, meta)
    output = {
        "json": str(json_path),
        "markdown": str(md_path),
        "meta": meta,
        "top_rows": [asdict(row) for row in rows[: args.show_top]],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    source_rows = len(draws)
    if args.limit_rows and args.limit_rows > 0:
        draws = draws[-args.limit_rows :]

    ranker_names = parse_ranker_names(args.rankers)
    rows = run_benchmark(
        draws,
        ranker_names=ranker_names,
        top_n=args.top,
        training_window=args.training_window,
        recent_window=args.recent_window,
        min_history=args.min_history,
        stake_per_number=args.stake_per_number,
        payout_per_hit=args.payout_per_hit,
        include_yearly=not args.no_yearly,
        include_monthly=args.monthly,
        monthly_year=args.monthly_year,
    )
    meta = {
        "source_rows": source_rows,
        "draw_rows": len(draws),
        "limit_rows": args.limit_rows,
        "rankers": ranker_names,
        "top_n": args.top,
        "training_window": args.training_window,
        "recent_window": args.recent_window,
        "include_yearly": not args.no_yearly,
        "include_monthly": args.monthly,
        "monthly_year": args.monthly_year,
    }
    json_path, md_path = save_benchmark_reports(rows, args.output_dir, meta)
    overall = [row for row in rows if row.segment == "all"]
    overall.sort(key=lambda row: (-row.hit_z_score, -row.pnl_vs_random_expected))
    output = {
        "json": str(json_path),
        "markdown": str(md_path),
        "meta": meta,
        "overall": [asdict(row) for row in overall],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_attribution(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    if args.limit_rows and args.limit_rows > 0:
        draws = draws[-args.limit_rows :]

    report = run_hit_attribution(
        draws,
        target_year=args.year,
        target_months=parse_months(args.months),
        compare_years=parse_years(args.compare_years),
        top_n=args.top,
        training_window=args.training_window,
        recent_window=args.recent_window,
        min_history=args.min_history,
    )
    json_path, md_path = save_attribution_report(report, args.output_dir)
    output = {
        "json": str(json_path),
        "markdown": str(md_path),
        "target_rounds": report.target_rounds,
        "target_hits": report.target_hits,
        "hit_details": [asdict(item) for item in report.hit_details],
        "top_comparison": [asdict(item) for item in report.comparison[:10]],
        "top_pressure": [asdict(item) for item in report.pressure[:10]],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_rulebacktest(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    source_rows = len(draws)
    if args.limit_rows and args.limit_rows > 0:
        draws = draws[-args.limit_rows :]

    rules = [item.strip() for item in args.rules.split(";") if item.strip()]
    years = parse_years(args.years)
    rows = run_rule_backtests(
        draws,
        rules=rules,
        years=years,
        top_n=args.top,
        pool_size=args.pool_size,
        training_window=args.training_window,
        recent_window=args.recent_window,
        min_history=args.min_history,
    )
    meta = {
        "source_rows": source_rows,
        "draw_rows": len(draws),
        "limit_rows": args.limit_rows,
        "rules": rules,
        "years": years,
        "top_n": args.top,
        "pool_size": args.pool_size,
        "training_window": args.training_window,
        "recent_window": args.recent_window,
    }
    json_path, md_path = save_rulebacktest_reports(rows, args.output_dir, meta)
    overall = [row for row in rows if row.segment == "all"]
    overall.sort(key=lambda row: (-row.hit_z_score, -row.pnl_vs_random_expected))
    output = {
        "json": str(json_path),
        "markdown": str(md_path),
        "meta": meta,
        "overall": [asdict(row) for row in overall],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_rulerecency(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    source_rows = len(draws)
    if args.limit_rows and args.limit_rows > 0:
        draws = draws[-args.limit_rows :]

    rules = [item.strip() for item in args.rules.split(";") if item.strip()]
    windows = parse_int_list(args.windows)
    rows = run_rule_recency(
        draws,
        rules=rules,
        windows=windows,
        top_n=args.top,
        pool_size=args.pool_size,
        training_window=args.training_window,
        recent_window=args.recent_window,
        min_history=args.min_history,
    )
    meta = {
        "source_rows": source_rows,
        "draw_rows": len(draws),
        "limit_rows": args.limit_rows,
        "rules": rules,
        "windows": windows,
        "top_n": args.top,
        "pool_size": args.pool_size,
        "training_window": args.training_window,
        "recent_window": args.recent_window,
    }
    json_path, md_path = save_rulerecency_reports(rows, args.output_dir, meta)
    ranked = sorted(rows, key=lambda row: (row.window_size, -row.hit_z_score, -row.pnl_vs_random_expected))
    output = {
        "json": str(json_path),
        "markdown": str(md_path),
        "meta": meta,
        "rows": [asdict(row) for row in ranked],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    source_rows = len(draws)
    if args.limit_rows and args.limit_rows > 0:
        draws = draws[-args.limit_rows :]

    rules = [item.strip() for item in args.rules.split(";") if item.strip()]
    windows = parse_int_list(args.windows)
    gate_windows = parse_int_list(args.gate_windows)
    thresholds = RuleGateThresholds(
        min_hits=args.min_hits,
        min_lift=args.min_lift,
        min_roi=args.min_roi,
    )
    report = run_gate(
        draws,
        rules=rules,
        windows=windows,
        gate_windows=gate_windows,
        top_n=args.top,
        pool_size=args.pool_size,
        training_window=args.training_window,
        recent_window=args.recent_window,
        min_history=args.min_history,
        thresholds=thresholds,
    )
    meta = {
        "source_rows": source_rows,
        "draw_rows": len(draws),
        "limit_rows": args.limit_rows,
        "rules": rules,
        "windows": windows,
        "gate_windows": gate_windows,
        "top_n": args.top,
        "pool_size": args.pool_size,
        "training_window": args.training_window,
        "recent_window": args.recent_window,
        "min_hits": args.min_hits,
        "min_lift": args.min_lift,
        "min_roi": args.min_roi,
    }
    json_path, md_path = save_gate_reports(report, args.output_dir, meta)
    output = {
        "json": str(json_path),
        "markdown": str(md_path),
        "meta": meta,
        "recommendation_mode": report.recommendation_mode,
        "active_rules": report.active_rules,
        "watch_rules": report.watch_rules,
        "blocked_rules": report.blocked_rules,
        "gated_candidates": [asdict(item) for item in report.gated_candidates],
        "base_candidates": [asdict(item) for item in report.base_candidates],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_calibration(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    source_rows = len(draws)
    if args.limit_rows and args.limit_rows > 0:
        draws = draws[-args.limit_rows :]

    top_values = parse_int_list(args.top_values)
    windows = parse_int_list(args.windows)
    report = run_calibration(
        draws,
        top_values=top_values,
        windows=windows,
        training_window=args.training_window,
        recent_window=args.recent_window,
        min_history=args.min_history,
        bucket_size=args.bucket_size,
        stake_per_number=args.stake_per_number,
        payout_per_hit=args.payout_per_hit,
    )
    meta = {
        "source_rows": source_rows,
        "draw_rows": len(draws),
        "limit_rows": args.limit_rows,
        "top_values": top_values,
        "windows": windows,
        "training_window": args.training_window,
        "recent_window": args.recent_window,
        "min_history": args.min_history,
        "bucket_size": args.bucket_size,
        "stake_per_number": args.stake_per_number,
        "payout_per_hit": args.payout_per_hit,
    }
    json_path, md_path = save_calibration_reports(report, args.output_dir, meta)
    output = {
        "json": str(json_path),
        "markdown": str(md_path),
        "meta": meta,
        "summary": asdict(report.summary),
        "top_rows": [asdict(row) for row in report.top_rows],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    source_rows = len(draws)
    if args.limit_rows and args.limit_rows > 0:
        draws = draws[-args.limit_rows :]

    top_values = parse_int_list(args.top_values)
    windows = parse_int_list(args.windows)
    report = run_ablation(
        draws,
        top_values=top_values,
        windows=windows,
        training_window=args.training_window,
        recent_window=args.recent_window,
        min_history=args.min_history,
    )
    meta = {
        "source_rows": source_rows,
        "draw_rows": len(draws),
        "limit_rows": args.limit_rows,
        "top_values": top_values,
        "windows": windows,
        "training_window": args.training_window,
        "recent_window": args.recent_window,
        "min_history": args.min_history,
    }
    json_path, md_path = save_ablation_reports(report, args.output_dir, meta)
    all_rows = [row for row in report.rows if row.segment == "all"]
    all_rows.sort(key=lambda row: (-row.hit_z_score, -row.pnl_vs_random_expected))
    output = {
        "json": str(json_path),
        "markdown": str(md_path),
        "meta": meta,
        "summary": asdict(report.summary),
        "top_global_rows": [asdict(row) for row in all_rows[: args.show_top]],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_variantstress(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    source_rows = len(draws)
    if args.limit_rows and args.limit_rows > 0:
        draws = draws[-args.limit_rows :]

    variants = parse_variant_names(args.variants)
    top_values = parse_int_list(args.top_values)
    windows = parse_int_list(args.windows)
    years = parse_years(args.years)
    report = run_variant_stress(
        draws,
        variant_names=variants,
        top_values=top_values,
        windows=windows,
        years=years,
        training_window=args.training_window,
        recent_window=args.recent_window,
        min_history=args.min_history,
    )
    meta = {
        "source_rows": source_rows,
        "draw_rows": len(draws),
        "limit_rows": args.limit_rows,
        "variants": variants,
        "top_values": top_values,
        "windows": windows,
        "years": years,
        "training_window": args.training_window,
        "recent_window": args.recent_window,
        "min_history": args.min_history,
    }
    json_path, md_path = save_variant_stress_reports(report, args.output_dir, meta)
    output = {
        "json": str(json_path),
        "markdown": str(md_path),
        "meta": meta,
        "summary": asdict(report.summary),
        "rows": [asdict(row) for row in report.rows],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_delivery(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    report = build_delivery_report(draws, reports_dir=args.reports_dir)
    meta = {
        "draw_rows": len(draws),
        "reports_dir": args.reports_dir,
    }
    json_path, md_path = save_delivery_report(report, args.output_dir, meta)
    output = {
        "json": str(json_path),
        "markdown": str(md_path),
        "meta": meta,
        "report": asdict(report),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_targetcoverage(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    source_rows = len(draws)
    if args.limit_rows and args.limit_rows > 0:
        draws = draws[-args.limit_rows :]

    compare_top_values = parse_int_list(args.compare_top_values)
    windows = parse_int_list(args.windows)
    report = run_target_coverage(
        draws,
        target_rate=args.target_rate,
        compare_top_values=compare_top_values,
        windows=windows,
        training_window=args.training_window,
        recent_window=args.recent_window,
        min_history=args.min_history,
        stake_per_number=args.stake_per_number,
        payout_per_hit=args.payout_per_hit,
    )
    meta = {
        "source_rows": source_rows,
        "draw_rows": len(draws),
        "limit_rows": args.limit_rows,
        "target_rate": args.target_rate,
        "compare_top_values": compare_top_values,
        "windows": windows,
        "training_window": args.training_window,
        "recent_window": args.recent_window,
        "min_history": args.min_history,
        "stake_per_number": args.stake_per_number,
        "payout_per_hit": args.payout_per_hit,
    }
    json_path, md_path = save_target_coverage_reports(report, args.output_dir, meta)
    target_rows = [row for row in report.rows if row.top_n == report.summary.required_top_n]
    output = {
        "json": str(json_path),
        "markdown": str(md_path),
        "meta": meta,
        "summary": asdict(report.summary),
        "target_rows": [asdict(row) for row in target_rows],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_daily(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    strategy_cache_path = Path(args.output_dir) / "strategy_cache.json"
    report = build_daily_report(
        draws,
        top_n=args.top,
        training_window=args.training_window,
        recent_window=args.recent_window,
        min_history=args.min_history,
        strategy_cache_path=strategy_cache_path,
    )
    meta = {
        "draw_rows": len(draws),
        "top_n": args.top,
        "training_window": args.training_window,
        "recent_window": args.recent_window,
        "min_history": args.min_history,
        "strategy_cache": str(strategy_cache_path),
    }
    json_path, html_path = save_daily_report(report, args.output_dir, meta)
    output = {
        "json": str(json_path),
        "html": str(html_path),
        "meta": meta,
        "primary": asdict(report.primary),
        "alternatives": [asdict(item) for item in report.alternatives[:3]],
        "mode": report.mode,
        "strategy_selection": asdict(report.strategy_selection),
        "confidence_gate": asdict(report.confidence_gate),
        "action_filter": asdict(report.action_filter),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    report = build_review_report(draws, predictions_dir=args.predictions_dir)
    meta = {
        "draw_rows": len(draws),
        "predictions_dir": args.predictions_dir,
        "output_dir": args.output_dir,
    }
    json_path, html_path = save_review_report(report, args.output_dir, meta)
    output = {
        "json": str(json_path),
        "html": str(html_path),
        "meta": meta,
        "summary": asdict(report.summary),
        "rows": [asdict(row) for row in report.rows],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_lawcheck(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    report = run_law_check(
        draws,
        max_lag=args.max_lag,
        alpha=args.alpha,
        min_formula_history=args.min_formula_history,
        split_ratio=args.split_ratio,
    )
    meta = {
        "draw_rows": len(draws),
        "max_lag": args.max_lag,
        "alpha": args.alpha,
        "min_formula_history": args.min_formula_history,
        "split_ratio": args.split_ratio,
        "output_dir": args.output_dir,
    }
    json_path, md_path = save_law_check_reports(report, args.output_dir, meta)
    output = {
        "json": str(json_path),
        "markdown": str(md_path),
        "meta": meta,
        "summary": asdict(report.summary),
        "formula_tests": [asdict(row) for row in report.formula_tests],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    windows = parse_int_list(args.windows)
    report = run_formula_discovery(
        draws,
        windows=windows,
        min_history=args.min_history,
        alpha=args.alpha,
        show_top=args.show_top,
    )
    meta = {
        "draw_rows": len(draws),
        "windows": windows,
        "min_history": args.min_history,
        "alpha": args.alpha,
        "show_top": args.show_top,
        "output_dir": args.output_dir,
    }
    json_path, md_path = save_discovery_reports(report, args.output_dir, meta)
    output = {
        "json": str(json_path),
        "markdown": str(md_path),
        "meta": meta,
        "summary": asdict(report.summary),
        "top_formulas": [asdict(row) for row in report.top_formulas],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fuli3d",
        description="福彩3D量化分析与候选号码生成工具",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="CSV历史开奖文件")

    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="从中国福彩网公开接口拉取福彩3D历史开奖")
    fetch.add_argument("--output", default=str(DEFAULT_DATA), help="输出CSV路径")
    fetch.add_argument("--page-size", type=int, default=100)
    fetch.add_argument("--max-pages", type=int)
    fetch.add_argument("--limit", type=int, help="只保留最近N期")
    fetch.add_argument("--timeout", type=int, default=20)
    fetch.set_defaults(func=cmd_fetch)

    summary = subparsers.add_parser("summary", help="查看历史数据摘要")
    add_data_argument(summary)
    summary.add_argument("--recent-window", type=int, default=60)
    summary.set_defaults(func=cmd_summary)

    recommend = subparsers.add_parser("recommend", help="生成候选号码")
    add_data_argument(recommend)
    recommend.add_argument("--top", type=int, default=20)
    recommend.add_argument("--recent-window", type=int, default=60)
    recommend.add_argument("--min-history", type=int, default=30)
    recommend.set_defaults(func=cmd_recommend)

    backtest = subparsers.add_parser("backtest", help="执行滚动回测")
    add_data_argument(backtest)
    backtest.add_argument("--top", type=int, default=20)
    backtest.add_argument("--training-window", type=int, default=120)
    backtest.add_argument("--recent-window", type=int, default=60)
    backtest.add_argument("--min-history", type=int, default=30)
    backtest.add_argument("--stake-per-number", type=float, default=2.0)
    backtest.add_argument("--payout-per-hit", type=float, default=1040.0)
    backtest.add_argument("--show-picks", action="store_true")
    backtest.set_defaults(func=cmd_backtest)

    explain = subparsers.add_parser("explain", help="解释某个号码的当前评分")
    add_data_argument(explain)
    explain.add_argument("number")
    explain.add_argument("--recent-window", type=int, default=60)
    explain.add_argument("--min-history", type=int, default=30)
    explain.set_defaults(func=cmd_explain)

    experiment = subparsers.add_parser("experiment", help="批量回测参数组合并生成报告")
    add_data_argument(experiment)
    experiment.add_argument("--top-values", default="10,20,30")
    experiment.add_argument("--training-windows", default="300,500")
    experiment.add_argument("--recent-windows", default="60,120")
    experiment.add_argument("--min-history", type=int, default=300)
    experiment.add_argument("--limit-rows", type=int, default=1200, help="只使用最近N行，0表示完整历史")
    experiment.add_argument("--stake-per-number", type=float, default=2.0)
    experiment.add_argument("--payout-per-hit", type=float, default=1040.0)
    experiment.add_argument("--output-dir", default="reports")
    experiment.add_argument("--show-top", type=int, default=10)
    experiment.set_defaults(func=cmd_experiment)

    benchmark = subparsers.add_parser("benchmark", help="对照弱基准并输出年度分段报告")
    add_data_argument(benchmark)
    benchmark.add_argument(
        "--rankers",
        default="model,position_hot,position_cold,sum_hot,span_hot,pattern_hot,random_fixed",
    )
    benchmark.add_argument("--top", type=int, default=20)
    benchmark.add_argument("--training-window", type=int, default=300)
    benchmark.add_argument("--recent-window", type=int, default=60)
    benchmark.add_argument("--min-history", type=int, default=300)
    benchmark.add_argument("--limit-rows", type=int, default=1200, help="只使用最近N行，0表示完整历史")
    benchmark.add_argument("--stake-per-number", type=float, default=2.0)
    benchmark.add_argument("--payout-per-hit", type=float, default=1040.0)
    benchmark.add_argument("--output-dir", default="reports/benchmark")
    benchmark.add_argument("--no-yearly", action="store_true")
    benchmark.add_argument("--monthly", action="store_true", help="输出月度分段")
    benchmark.add_argument("--monthly-year", type=int, help="只输出指定年份的月度分段")
    benchmark.set_defaults(func=cmd_benchmark)

    attribution = subparsers.add_parser("attribution", help="分析指定年月命中号码的特征归因")
    add_data_argument(attribution)
    attribution.add_argument("--year", type=int, required=True)
    attribution.add_argument("--months", required=True, help="逗号分隔月份，例如 3,5")
    attribution.add_argument("--compare-years", default="2024,2026")
    attribution.add_argument("--top", type=int, default=20)
    attribution.add_argument("--training-window", type=int, default=300)
    attribution.add_argument("--recent-window", type=int, default=60)
    attribution.add_argument("--min-history", type=int, default=300)
    attribution.add_argument("--limit-rows", type=int, default=1200)
    attribution.add_argument("--output-dir", default="reports/attribution")
    attribution.set_defaults(func=cmd_attribution)

    rulebacktest = subparsers.add_parser("rulebacktest", help="回测归因特征过滤规则")
    add_data_argument(rulebacktest)
    rulebacktest.add_argument(
        "--rules",
        required=True,
        help="分号分隔规则，例如 ones=5;sum=15;pattern=zuliu;ones=5,sum=15",
    )
    rulebacktest.add_argument("--years", default="2024,2025,2026")
    rulebacktest.add_argument("--top", type=int, default=20)
    rulebacktest.add_argument("--pool-size", type=int, default=200)
    rulebacktest.add_argument("--training-window", type=int, default=300)
    rulebacktest.add_argument("--recent-window", type=int, default=60)
    rulebacktest.add_argument("--min-history", type=int, default=300)
    rulebacktest.add_argument("--limit-rows", type=int, default=1200)
    rulebacktest.add_argument("--output-dir", default="reports/rulebacktest")
    rulebacktest.set_defaults(func=cmd_rulebacktest)

    rulerecency = subparsers.add_parser("rulerecency", help="检测规则在最近N期是否退潮")
    add_data_argument(rulerecency)
    rulerecency.add_argument(
        "--rules",
        required=True,
        help="分号分隔规则，例如 ones=5;sum=15;ones=5,sum=15",
    )
    rulerecency.add_argument("--windows", default="60,120,240,360")
    rulerecency.add_argument("--top", type=int, default=20)
    rulerecency.add_argument("--pool-size", type=int, default=200)
    rulerecency.add_argument("--training-window", type=int, default=300)
    rulerecency.add_argument("--recent-window", type=int, default=60)
    rulerecency.add_argument("--min-history", type=int, default=300)
    rulerecency.add_argument("--limit-rows", type=int, default=1200)
    rulerecency.add_argument("--output-dir", default="reports/rulerecency")
    rulerecency.set_defaults(func=cmd_rulerecency)

    gate = subparsers.add_parser("gate", help="用近期窗口闸门控制规则过滤候选")
    add_data_argument(gate)
    gate.add_argument(
        "--rules",
        required=True,
        help="分号分隔规则，例如 ones=5,sum=15,pattern=zuliu;sum=15",
    )
    gate.add_argument("--windows", default="60,120,240,360")
    gate.add_argument("--gate-windows", default="60,120")
    gate.add_argument("--min-hits", type=int, default=1)
    gate.add_argument("--min-lift", type=float, default=1.0)
    gate.add_argument("--min-roi", type=float, default=-0.48)
    gate.add_argument("--top", type=int, default=20)
    gate.add_argument("--pool-size", type=int, default=200)
    gate.add_argument("--training-window", type=int, default=300)
    gate.add_argument("--recent-window", type=int, default=60)
    gate.add_argument("--min-history", type=int, default=300)
    gate.add_argument("--limit-rows", type=int, default=1200)
    gate.add_argument("--output-dir", default="reports/gate")
    gate.set_defaults(func=cmd_gate)

    calibration = subparsers.add_parser("calibration", help="检验基础评分排序能力")
    add_data_argument(calibration)
    calibration.add_argument("--top-values", default="10,20,50,100")
    calibration.add_argument("--windows", default="60,120,240,360")
    calibration.add_argument("--training-window", type=int, default=300)
    calibration.add_argument("--recent-window", type=int, default=60)
    calibration.add_argument("--min-history", type=int, default=300)
    calibration.add_argument("--bucket-size", type=int, default=100)
    calibration.add_argument("--limit-rows", type=int, default=1200)
    calibration.add_argument("--stake-per-number", type=float, default=2.0)
    calibration.add_argument("--payout-per-hit", type=float, default=1040.0)
    calibration.add_argument("--output-dir", default="reports/calibration")
    calibration.set_defaults(func=cmd_calibration)

    ablation = subparsers.add_parser("ablation", help="评估各评分权重是否拖累排序")
    add_data_argument(ablation)
    ablation.add_argument("--top-values", default="10,20,50")
    ablation.add_argument("--windows", default="60,120,240,360")
    ablation.add_argument("--training-window", type=int, default=300)
    ablation.add_argument("--recent-window", type=int, default=60)
    ablation.add_argument("--min-history", type=int, default=300)
    ablation.add_argument("--limit-rows", type=int, default=1200)
    ablation.add_argument("--output-dir", default="reports/ablation")
    ablation.add_argument("--show-top", type=int, default=10)
    ablation.set_defaults(func=cmd_ablation)

    variantstress = subparsers.add_parser("variantstress", help="按年度和近期窗口压力测试权重变体")
    add_data_argument(variantstress)
    variantstress.add_argument("--variants", default="baseline,no_repeat_penalty")
    variantstress.add_argument("--top-values", default="10,20,50")
    variantstress.add_argument("--windows", default="60,120,240,360")
    variantstress.add_argument("--years", default="2023,2024,2025,2026")
    variantstress.add_argument("--training-window", type=int, default=300)
    variantstress.add_argument("--recent-window", type=int, default=60)
    variantstress.add_argument("--min-history", type=int, default=300)
    variantstress.add_argument("--limit-rows", type=int, default=1200)
    variantstress.add_argument("--output-dir", default="reports/variantstress")
    variantstress.set_defaults(func=cmd_variantstress)

    delivery = subparsers.add_parser("delivery", help="汇总当前项目是否达到交付或推荐条件")
    add_data_argument(delivery)
    delivery.add_argument("--reports-dir", default="reports")
    delivery.add_argument("--output-dir", default="reports/delivery")
    delivery.set_defaults(func=cmd_delivery)

    targetcoverage = subparsers.add_parser("targetcoverage", help="测算达到目标命中率需要覆盖多少号码")
    add_data_argument(targetcoverage)
    targetcoverage.add_argument("--target-rate", type=float, default=0.65)
    targetcoverage.add_argument("--compare-top-values", default="10,20,50,100,200,500")
    targetcoverage.add_argument("--windows", default="60,120,240,360")
    targetcoverage.add_argument("--training-window", type=int, default=300)
    targetcoverage.add_argument("--recent-window", type=int, default=60)
    targetcoverage.add_argument("--min-history", type=int, default=300)
    targetcoverage.add_argument("--limit-rows", type=int, default=1200)
    targetcoverage.add_argument("--stake-per-number", type=float, default=2.0)
    targetcoverage.add_argument("--payout-per-hit", type=float, default=1040.0)
    targetcoverage.add_argument("--output-dir", default="reports/targetcoverage")
    targetcoverage.set_defaults(func=cmd_targetcoverage)

    daily = subparsers.add_parser("daily", help="生成每日预测页面和玩法收益测算")
    add_data_argument(daily)
    daily.add_argument("--top", type=int, default=10)
    daily.add_argument("--training-window", type=int, default=300)
    daily.add_argument("--recent-window", type=int, default=60)
    daily.add_argument("--min-history", type=int, default=300)
    daily.add_argument("--output-dir", default="reports/daily")
    daily.set_defaults(func=cmd_daily)

    review = subparsers.add_parser("review", help="复盘每日预测快照和实际开奖结果")
    add_data_argument(review)
    review.add_argument("--predictions-dir", default="reports/daily/snapshots")
    review.add_argument("--output-dir", default="reports/review")
    review.set_defaults(func=cmd_review)

    lawcheck = subparsers.add_parser("lawcheck", help="检验历史开奖是否存在可验证数学规律")
    add_data_argument(lawcheck)
    lawcheck.add_argument("--max-lag", type=int, default=10)
    lawcheck.add_argument("--alpha", type=float, default=0.05)
    lawcheck.add_argument("--min-formula-history", type=int, default=300)
    lawcheck.add_argument("--split-ratio", type=float, default=0.7)
    lawcheck.add_argument("--output-dir", default="reports/lawcheck")
    lawcheck.set_defaults(func=cmd_lawcheck)

    discover = subparsers.add_parser("discover", help="自动探索自定义历史公式")
    add_data_argument(discover)
    discover.add_argument("--windows", default="30,60,120,240")
    discover.add_argument("--min-history", type=int, default=300)
    discover.add_argument("--alpha", type=float, default=0.05)
    discover.add_argument("--show-top", type=int, default=20)
    discover.add_argument("--output-dir", default="reports/formula_discovery")
    discover.set_defaults(func=cmd_discover)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
