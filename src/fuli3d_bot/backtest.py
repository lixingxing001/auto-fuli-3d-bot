from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from math import sqrt

from .models import BacktestPick, BacktestResult, Draw
from .strategy import StrategyConfig, rank_numbers


Ranker = Callable[[list[Draw], int, StrategyConfig], list[str]]


def model_ranker(draws: list[Draw], top_n: int, config: StrategyConfig) -> list[str]:
    return [item.number for item in rank_numbers(draws, top_n=top_n, config=config)]


def max_drawdown_from_equity(equity_values: list[float]) -> float:
    peak = 0.0
    max_drawdown = 0.0
    for value in equity_values:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, peak - value)
    return max_drawdown


def run_backtest(
    draws: list[Draw],
    top_n: int = 20,
    training_window: int = 120,
    stake_per_number: float = 2.0,
    payout_per_hit: float = 1040.0,
    config: StrategyConfig | None = None,
    ranker: Ranker | None = None,
    eval_filter: Callable[[Draw], bool] | None = None,
) -> BacktestResult:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if training_window <= 0:
        raise ValueError("training_window must be positive")
    if len(draws) <= training_window:
        raise ValueError(
            f"need more draw rows than training_window, got {len(draws)} rows and window {training_window}"
        )

    active_config = config or StrategyConfig()
    active_config = replace(active_config, min_history=min(active_config.min_history, training_window))
    active_ranker = ranker or model_ranker

    picks: list[BacktestPick] = []
    equity_values: list[float] = []
    equity = 0.0
    losing_streak = 0
    max_losing_streak = 0

    for index in range(training_window, len(draws)):
        training_draws = draws[index - training_window : index]
        actual = draws[index]
        if eval_filter and not eval_filter(actual):
            continue
        candidates = active_ranker(training_draws, top_n, active_config)
        hit = actual.number in candidates
        stake = stake_per_number * len(candidates)
        payout = payout_per_hit if hit else 0.0
        pnl = payout - stake

        if hit:
            losing_streak = 0
        else:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)

        equity += pnl
        equity_values.append(equity)
        picks.append(
            BacktestPick(
                issue=actual.issue,
                draw_date=actual.draw_date,
                actual_number=actual.number,
                candidates=candidates,
                hit=hit,
                stake=stake,
                payout=payout,
                pnl=pnl,
            )
        )

    total_stake = sum(pick.stake for pick in picks)
    total_payout = sum(pick.payout for pick in picks)
    total_pnl = total_payout - total_stake
    hits = sum(1 for pick in picks if pick.hit)
    rounds = len(picks)
    roi = total_pnl / total_stake if total_stake else 0.0
    expected_probabilities = [min(1.0, len(pick.candidates) / 1000.0) for pick in picks]
    expected_hits = sum(expected_probabilities)
    expected_hit_rate = expected_hits / rounds if rounds else 0.0
    hit_lift = (hits / expected_hits) if expected_hits else 0.0
    variance = sum(probability * (1.0 - probability) for probability in expected_probabilities)
    hit_z_score = (hits - expected_hits) / sqrt(variance) if variance > 0 else 0.0
    expected_pnl = expected_hits * payout_per_hit - total_stake
    expected_roi = expected_pnl / total_stake if total_stake else 0.0

    return BacktestResult(
        rounds=rounds,
        hits=hits,
        hit_rate=hits / rounds if rounds else 0.0,
        expected_hit_rate=expected_hit_rate,
        expected_hits=expected_hits,
        hit_lift=hit_lift,
        hit_z_score=hit_z_score,
        stake=total_stake,
        payout=total_payout,
        pnl=total_pnl,
        roi=roi,
        expected_pnl=expected_pnl,
        expected_roi=expected_roi,
        pnl_vs_random_expected=total_pnl - expected_pnl,
        max_drawdown=max_drawdown_from_equity(equity_values),
        max_losing_streak=max_losing_streak,
        picks=picks,
        meta={
            "top_n": top_n,
            "training_window": training_window,
            "stake_per_number": stake_per_number,
            "payout_per_hit": payout_per_hit,
        },
    )
