from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import date
from math import sqrt
from pathlib import Path

from .backtest import run_backtest
from .features import extract_features, pattern_label, theoretical_distributions
from .models import BacktestPick, Draw, Recommendation
from .strategy import StrategyConfig, rank_numbers


THEORY = theoretical_distributions()
SUM_PRIZES = {
    0: 1040,
    1: 345,
    2: 172,
    3: 104,
    4: 69,
    5: 49,
    6: 37,
    7: 29,
    8: 23,
    9: 19,
    10: 16,
    11: 15,
    12: 15,
    13: 14,
    14: 14,
    15: 15,
    16: 15,
    17: 16,
    18: 19,
    19: 23,
    20: 29,
    21: 37,
    22: 49,
    23: 69,
    24: 104,
    25: 172,
    26: 345,
    27: 1040,
}


@dataclass(frozen=True)
class DailyPrediction:
    rank: int
    number: str
    score: float
    sum_value: int
    span: int
    pattern: str
    reasons: list[str]


@dataclass(frozen=True)
class RankingEntry:
    rank: int
    number: str
    score: float
    sum_value: int
    span: int
    pattern: str


@dataclass(frozen=True)
class PrimaryCooldownRecord:
    number: str
    consecutive_misses: int
    last_issue: str
    last_actual_number: str
    reason: str


@dataclass(frozen=True)
class PrimaryCooldownState:
    threshold: int
    reviewed_snapshots: int
    records: list[PrimaryCooldownRecord]

    @property
    def active_numbers(self) -> set[str]:
        return {record.number for record in self.records}


@dataclass(frozen=True)
class PlayStats:
    rounds: int
    hits: int
    hit_rate: float
    expected_hits: float
    expected_hit_rate: float
    hit_lift: float
    stake: float
    payout: float
    pnl: float
    roi: float


@dataclass(frozen=True)
class PlayRow:
    play_id: str
    name: str
    selection: str
    active: bool
    order_required: str
    stake: float
    prize_text: str
    net_profit_text: str
    theoretical_hit_rate: float
    expected_roi: float
    all_stats: PlayStats
    recent60_stats: PlayStats
    recent120_stats: PlayStats
    note: str


@dataclass(frozen=True)
class GateCheck:
    name: str
    status: str
    metric: str
    detail: str


@dataclass(frozen=True)
class ConfidenceGate:
    status: str
    label: str
    confidence: str
    action: str
    summary: str
    checks: list[GateCheck]


@dataclass(frozen=True)
class ActionFilter:
    status: str
    label: str
    stake_level: str
    recommendation: str
    reason: str
    allowed_scope: str


@dataclass(frozen=True)
class StrategyVariant:
    name: str
    label: str
    config: StrategyConfig


@dataclass(frozen=True)
class StrategyCandidate:
    name: str
    label: str
    status: str
    reason: str
    primary_number: str
    top_n: int
    rounds: int
    hits: int
    expected_hits: float
    hit_rate: float
    hit_lift: float
    hit_z_score: float
    roi: float
    recent60_hits: int
    recent120_hits: int
    annual_failures: int


@dataclass(frozen=True)
class StrategySelection:
    active_name: str
    active_label: str
    status: str
    cache_status: str
    summary: str
    candidates: list[StrategyCandidate]


@dataclass(frozen=True)
class DailyReport:
    latest_issue: str
    latest_date: str | None
    latest_number: str
    next_issue_hint: str
    mode: str
    primary: DailyPrediction
    alternatives: list[DailyPrediction]
    play_rows: list[PlayRow]
    confidence_gate: ConfidenceGate
    action_filter: ActionFilter
    strategy_selection: StrategySelection
    full_ranking: list[RankingEntry]
    primary_cooldown: PrimaryCooldownState
    source_notes: list[str]


def _next_issue_hint(issue: str) -> str:
    if len(issue) >= 5 and issue.isdigit():
        return f"{int(issue) + 1}"
    return "下一期"


def _prediction_from_recommendation(item: Recommendation) -> DailyPrediction:
    features = item.features
    return DailyPrediction(
        rank=item.rank,
        number=item.number,
        score=item.score,
        sum_value=features.sum_value,
        span=features.span,
        pattern=pattern_label(features.pattern),
        reasons=item.reasons,
    )


def _ranking_entry_from_recommendation(item: Recommendation) -> RankingEntry:
    features = item.features
    return RankingEntry(
        rank=item.rank,
        number=item.number,
        score=item.score,
        sum_value=features.sum_value,
        span=features.span,
        pattern=pattern_label(features.pattern),
    )


def _empty_primary_cooldown(threshold: int = 2, reviewed_snapshots: int = 0) -> PrimaryCooldownState:
    return PrimaryCooldownState(
        threshold=threshold,
        reviewed_snapshots=reviewed_snapshots,
        records=[],
    )


def _snapshot_files(predictions_dir: str | Path) -> list[Path]:
    path = Path(predictions_dir)
    if not path.exists():
        return []
    return sorted(path.glob("prediction_*.json"))


def _safe_load_snapshot(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_primary_cooldown_state(
    draws: list[Draw],
    predictions_dir: str | Path,
    threshold: int = 2,
) -> PrimaryCooldownState:
    active_threshold = max(1, threshold)
    actual_by_issue = {draw.issue: draw for draw in draws}
    reviewed: list[tuple[str, str, str, bool]] = []
    for path in _snapshot_files(predictions_dir):
        payload = _safe_load_snapshot(path)
        if payload is None:
            continue
        try:
            report = payload["report"]
            issue = str(report["next_issue_hint"])
            primary = str(report["primary"]["number"])
        except (KeyError, TypeError):
            continue
        actual = actual_by_issue.get(issue)
        if actual is None:
            continue
        reviewed.append((issue, primary, actual.number, primary == actual.number))

    if not reviewed:
        return _empty_primary_cooldown(active_threshold, 0)

    last_issue, last_primary, last_actual, last_hit = reviewed[-1]
    if last_hit:
        return _empty_primary_cooldown(active_threshold, len(reviewed))

    consecutive_misses = 0
    for _issue, primary, _actual, hit in reversed(reviewed):
        if primary != last_primary or hit:
            break
        consecutive_misses += 1

    if consecutive_misses < active_threshold:
        return _empty_primary_cooldown(active_threshold, len(reviewed))

    return PrimaryCooldownState(
        threshold=active_threshold,
        reviewed_snapshots=len(reviewed),
        records=[
            PrimaryCooldownRecord(
                number=last_primary,
                consecutive_misses=consecutive_misses,
                last_issue=last_issue,
                last_actual_number=last_actual,
                reason=f"主号连续 {consecutive_misses} 次未命中，暂停作为下一期主号。",
            )
        ],
    )


def _unordered_key(number: str) -> tuple[str, ...]:
    return tuple(sorted(number))


def _sum_category(sum_value: int) -> str | None:
    if sum_value <= 8:
        return "小"
    if sum_value >= 19:
        return "大"
    return None


def _parity_category(number: str) -> str | None:
    digits = [int(ch) for ch in number]
    if all(digit % 2 == 0 for digit in digits):
        return "全偶"
    if all(digit % 2 == 1 for digit in digits):
        return "全奇"
    return None


def _play_meta(play_id: str, predicted: str) -> dict:
    features = extract_features(predicted)
    digits = features.digits
    sorted_digits = "".join(sorted(predicted))
    if play_id == "direct":
        return {
            "active": True,
            "name": "单选/直选",
            "selection": predicted,
            "order_required": "要求顺序",
            "stake": 2.0,
            "prize_text": "1040",
            "net_profit_text": "1038",
            "probability": 1 / 1000,
            "expected_payout": 1040 / 1000,
            "note": "买一个三位数，百位十位个位必须完全一致。",
        }
    if play_id == "group":
        if features.pattern == "zuliu":
            return {
                "active": True,
                "name": "组选6",
                "selection": sorted_digits,
                "order_required": "顺序不限",
                "stake": 2.0,
                "prize_text": "173",
                "net_profit_text": "171",
                "probability": 6 / 1000,
                "expected_payout": 173 * 6 / 1000,
                "note": "三个数字各不相同，任意排列命中。",
            }
        if features.pattern == "zusan":
            return {
                "active": True,
                "name": "组选3",
                "selection": sorted_digits,
                "order_required": "顺序不限",
                "stake": 2.0,
                "prize_text": "346",
                "net_profit_text": "344",
                "probability": 3 / 1000,
                "expected_payout": 346 * 3 / 1000,
                "note": "三个数字里有两个相同，任意排列命中。",
            }
        return {
            "active": False,
            "name": "组选",
            "selection": sorted_digits,
            "order_required": "顺序不限",
            "stake": 2.0,
            "prize_text": "不适用",
            "net_profit_text": "不适用",
            "probability": 0.0,
            "expected_payout": 0.0,
            "note": "豹子号不能按组选3或组选6处理。",
        }
    if play_id == "package":
        if features.pattern == "zuliu":
            return {
                "active": True,
                "name": "包选6",
                "selection": predicted,
                "order_required": "全中要求顺序，组中顺序不限",
                "stake": 2.0,
                "prize_text": "全中606, 组中86",
                "net_profit_text": "604 或 84",
                "probability": 6 / 1000,
                "expected_payout": (606 + 86 * 5) / 1000,
                "note": "同一组三个不同数字，直选中全中，其它排列中组中。",
            }
        if features.pattern == "zusan":
            return {
                "active": True,
                "name": "包选3",
                "selection": predicted,
                "order_required": "全中要求顺序，组中顺序不限",
                "stake": 2.0,
                "prize_text": "全中693, 组中173",
                "net_profit_text": "691 或 171",
                "probability": 3 / 1000,
                "expected_payout": (693 + 173 * 2) / 1000,
                "note": "同一组三个含对子数字，直选中全中，其它排列中组中。",
            }
        return {
            "active": False,
            "name": "包选",
            "selection": predicted,
            "order_required": "视号码形态",
            "stake": 2.0,
            "prize_text": "不适用",
            "net_profit_text": "不适用",
            "probability": 0.0,
            "expected_payout": 0.0,
            "note": "豹子号不按包选3或包选6处理。",
        }
    if play_id.startswith("1d_"):
        index = {"1d_h": 0, "1d_t": 1, "1d_o": 2}[play_id]
        label = ["百位", "十位", "个位"][index]
        return {
            "active": True,
            "name": f"1D {label}",
            "selection": f"{label}={digits[index]}",
            "order_required": "要求位置",
            "stake": 2.0,
            "prize_text": "10",
            "net_profit_text": "8",
            "probability": 1 / 10,
            "expected_payout": 10 / 10,
            "note": "只猜一个固定位置的数字。",
        }
    if play_id.startswith("2d_"):
        pairs = {
            "2d_ht": ((0, 1), "百十"),
            "2d_to": ((1, 2), "十个"),
            "2d_ho": ((0, 2), "百个"),
        }
        indexes, label = pairs[play_id]
        selection = "".join(str(digits[index]) for index in indexes)
        return {
            "active": True,
            "name": f"2D {label}",
            "selection": f"{label}={selection}",
            "order_required": "要求位置",
            "stake": 2.0,
            "prize_text": "104",
            "net_profit_text": "102",
            "probability": 1 / 100,
            "expected_payout": 104 / 100,
            "note": "猜两个固定位置的数字。",
        }
    if play_id == "sum":
        sum_value = features.sum_value
        probability = THEORY["sum"][sum_value] / 1000
        prize = SUM_PRIZES[sum_value]
        return {
            "active": True,
            "name": "和值",
            "selection": f"和值={sum_value}",
            "order_required": "不要求顺序",
            "stake": 2.0,
            "prize_text": str(prize),
            "net_profit_text": str(prize - 2),
            "probability": probability,
            "expected_payout": prize * probability,
            "note": "只看三个数字相加之和。",
        }
    if play_id == "big_small":
        category = _sum_category(features.sum_value)
        active = category is not None
        probability = sum(
            count
            for sum_value, count in THEORY["sum"].items()
            if _sum_category(sum_value) == category
        ) / 1000 if active else 0.0
        return {
            "active": active,
            "name": "猜大小",
            "selection": category or "和值9到18无大小投注",
            "order_required": "不要求顺序",
            "stake": 2.0,
            "prize_text": "6" if active else "不适用",
            "net_profit_text": "4" if active else "不适用",
            "probability": probability,
            "expected_payout": 6 * probability if active else 0.0,
            "note": "和值0到8为小，19到27为大，中间区间不适用。",
        }
    if play_id == "odd_even":
        category = _parity_category(predicted)
        active = category is not None
        return {
            "active": active,
            "name": "猜奇偶",
            "selection": category or "混合奇偶无此投注",
            "order_required": "不要求顺序",
            "stake": 2.0,
            "prize_text": "8" if active else "不适用",
            "net_profit_text": "6" if active else "不适用",
            "probability": 125 / 1000 if active else 0.0,
            "expected_payout": 8 * 125 / 1000 if active else 0.0,
            "note": "三个数字全部同为奇数或全部同为偶数才适用。",
        }
    raise ValueError(f"unknown play_id: {play_id}")


def _play_payout(play_id: str, predicted: str, actual: str) -> tuple[bool, float]:
    meta = _play_meta(play_id, predicted)
    if not meta["active"]:
        return False, 0.0
    predicted_features = extract_features(predicted)
    actual_features = extract_features(actual)
    if play_id == "direct":
        return actual == predicted, 1040.0 if actual == predicted else 0.0
    if play_id == "group":
        hit = _unordered_key(actual) == _unordered_key(predicted) and actual_features.pattern == predicted_features.pattern
        if not hit:
            return False, 0.0
        return True, 173.0 if predicted_features.pattern == "zuliu" else 346.0
    if play_id == "package":
        if _unordered_key(actual) != _unordered_key(predicted) or actual_features.pattern != predicted_features.pattern:
            return False, 0.0
        if actual == predicted:
            return True, 606.0 if predicted_features.pattern == "zuliu" else 693.0
        return True, 86.0 if predicted_features.pattern == "zuliu" else 173.0
    if play_id.startswith("1d_"):
        index = {"1d_h": 0, "1d_t": 1, "1d_o": 2}[play_id]
        hit = predicted[index] == actual[index]
        return hit, 10.0 if hit else 0.0
    if play_id.startswith("2d_"):
        indexes = {
            "2d_ht": (0, 1),
            "2d_to": (1, 2),
            "2d_ho": (0, 2),
        }[play_id]
        hit = all(predicted[index] == actual[index] for index in indexes)
        return hit, 104.0 if hit else 0.0
    if play_id == "sum":
        hit = predicted_features.sum_value == actual_features.sum_value
        return hit, float(SUM_PRIZES[predicted_features.sum_value]) if hit else 0.0
    if play_id == "big_small":
        category = _sum_category(predicted_features.sum_value)
        hit = category is not None and category == _sum_category(actual_features.sum_value)
        return hit, 6.0 if hit else 0.0
    if play_id == "odd_even":
        category = _parity_category(predicted)
        hit = category is not None and category == _parity_category(actual)
        return hit, 8.0 if hit else 0.0
    raise ValueError(f"unknown play_id: {play_id}")


def _empty_stats() -> PlayStats:
    return PlayStats(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _stats_for_play(play_id: str, picks: list[BacktestPick]) -> PlayStats:
    rounds = 0
    hits = 0
    stake = 0.0
    payout = 0.0
    expected_hits = 0.0
    for pick in picks:
        if not pick.candidates:
            continue
        predicted = pick.candidates[0]
        meta = _play_meta(play_id, predicted)
        if not meta["active"]:
            continue
        rounds += 1
        stake += float(meta["stake"])
        expected_hits += float(meta["probability"])
        hit, prize = _play_payout(play_id, predicted, pick.actual_number)
        if hit:
            hits += 1
            payout += prize
    if rounds == 0:
        return _empty_stats()
    pnl = payout - stake
    expected_hit_rate = expected_hits / rounds
    return PlayStats(
        rounds=rounds,
        hits=hits,
        hit_rate=hits / rounds,
        expected_hits=expected_hits,
        expected_hit_rate=expected_hit_rate,
        hit_lift=(hits / expected_hits) if expected_hits else 0.0,
        stake=stake,
        payout=payout,
        pnl=pnl,
        roi=pnl / stake if stake else 0.0,
    )


def build_play_rows(primary_number: str, picks: list[BacktestPick]) -> list[PlayRow]:
    play_ids = [
        "direct",
        "group",
        "package",
        "1d_h",
        "1d_t",
        "1d_o",
        "2d_ht",
        "2d_to",
        "2d_ho",
        "sum",
        "big_small",
        "odd_even",
    ]
    rows: list[PlayRow] = []
    for play_id in play_ids:
        meta = _play_meta(play_id, primary_number)
        expected_roi = (float(meta["expected_payout"]) - float(meta["stake"])) / float(meta["stake"])
        rows.append(
            PlayRow(
                play_id=play_id,
                name=meta["name"],
                selection=meta["selection"],
                active=bool(meta["active"]),
                order_required=meta["order_required"],
                stake=float(meta["stake"]),
                prize_text=meta["prize_text"],
                net_profit_text=meta["net_profit_text"],
                theoretical_hit_rate=float(meta["probability"]),
                expected_roi=expected_roi if meta["active"] else 0.0,
                all_stats=_stats_for_play(play_id, picks),
                recent60_stats=_stats_for_play(play_id, picks[-60:]),
                recent120_stats=_stats_for_play(play_id, picks[-120:]),
                note=meta["note"],
            )
        )
    return rows


def _gate_check(name: str, passed: bool, metric: str, detail: str) -> GateCheck:
    return GateCheck(
        name=name,
        status="通过" if passed else "未通过",
        metric=metric,
        detail=detail,
    )


def build_confidence_gate(play_rows: list[PlayRow]) -> ConfidenceGate:
    direct = _find_play_row(play_rows, "direct")
    group = _find_play_row(play_rows, "group")
    package = _find_play_row(play_rows, "package")

    direct_has_recent = direct.recent60_stats.hits > 0 and direct.recent120_stats.hits > 0
    direct_lift_ok = direct.all_stats.hit_lift >= 1.20
    group_watch = group.active and group.all_stats.hit_lift >= 1.00 and group.recent60_stats.hits > 0
    package_watch = package.active and package.all_stats.roi >= 0 and package.recent60_stats.hits > 0

    checks = [
        _gate_check(
            "直选短窗",
            direct_has_recent,
            f"近60期 {direct.recent60_stats.hits}，近120期 {direct.recent120_stats.hits}",
            "单号需要短窗有命中，才允许提高置信。",
        ),
        _gate_check(
            "直选长期",
            direct_lift_ok,
            f"lift {direct.all_stats.hit_lift:.2f}",
            "长期命中需要明显高于随机理论。",
        ),
        _gate_check(
            "组选辅助",
            group_watch,
            f"{group.name} 近60期 {group.recent60_stats.hits}",
            "顺序不限玩法只作辅助信号。",
        ),
        _gate_check(
            "包选辅助",
            package_watch,
            f"ROI {package.all_stats.roi:.2%}",
            "包选回测为正时进入关注，但仍需样本扩充。",
        ),
    ]

    if direct_has_recent and direct_lift_ok:
        return ConfidenceGate(
            status="watch",
            label="可关注",
            confidence="中",
            action="允许小注观察，等待下一次命中验证。",
            summary="直选短窗和长期信号同时过关，可进入小样本观察。",
            checks=checks,
        )

    if group_watch or package_watch:
        return ConfidenceGate(
            status="observe",
            label="辅助观察",
            confidence="低",
            action="只按观察处理，直选不加仓。",
            summary="辅助玩法有一点信号，直选短窗仍未过关。",
            checks=checks,
        )

    return ConfidenceGate(
        status="blocked",
        label="低置信",
        confidence="低",
        action="只记录，不建议扩大投入。",
        summary="直选短窗未过关，当前预测只能作为复盘样本。",
        checks=checks,
    )


def build_action_filter(
    confidence_gate: ConfidenceGate,
    strategy_selection: StrategySelection,
    play_rows: list[PlayRow],
) -> ActionFilter:
    direct = _find_play_row(play_rows, "direct")
    group = _find_play_row(play_rows, "group")
    package = _find_play_row(play_rows, "package")

    if strategy_selection.status == "active" and confidence_gate.status == "watch":
        return ActionFilter(
            status="action",
            label="可小注",
            stake_level="低",
            recommendation="允许小额记录型出手，单期投入必须固定。",
            reason="策略启用闸门和直选置信闸门同时通过。",
            allowed_scope=f"直选 {direct.selection}",
        )

    if confidence_gate.status == "observe" and (group.recent60_stats.hits > 0 or package.recent60_stats.hits > 0):
        return ActionFilter(
            status="observe",
            label="观望",
            stake_level="零",
            recommendation="不建议直选出手，只记录主号和辅助玩法表现。",
            reason="辅助玩法有短窗信号，直选短窗仍未过关。",
            allowed_scope=f"记录 {direct.selection}，辅助观察 {group.name} {group.selection}",
        )

    return ActionFilter(
        status="skip",
        label="不出手",
        stake_level="零",
        recommendation="今天只更新样本，不做投注动作。",
        reason="策略或直选置信闸门未通过。",
        allowed_scope=f"记录 {direct.selection}",
    )


def apply_primary_cooldown(
    action_filter: ActionFilter,
    primary: DailyPrediction,
    cooldown_state: PrimaryCooldownState,
) -> ActionFilter:
    if not cooldown_state.records:
        return action_filter
    cooled = "、".join(record.number for record in cooldown_state.records)
    detail = "；".join(
        f"{record.number} 连续 {record.consecutive_misses} 次未命中"
        for record in cooldown_state.records
    )
    return ActionFilter(
        status="observe",
        label="观望",
        stake_level="零",
        recommendation="主号冷却生效，本期只记录，不做直选出手。",
        reason=f"{detail}，已暂停作为主号。",
        allowed_scope=f"记录 {primary.number}，冷却观察 {cooled}",
    )


def _strategy_variants(recent_window: int, min_history: int) -> list[StrategyVariant]:
    base = StrategyConfig(recent_window=recent_window, min_history=min_history)
    zero = replace(
        base,
        position_weight=0.0,
        recent_position_weight=0.0,
        sum_weight=0.0,
        span_weight=0.0,
        pattern_weight=0.0,
        omission_weight=0.0,
        repeat_penalty=0.0,
    )
    return [
        StrategyVariant("baseline", "默认策略", base),
        StrategyVariant("no_repeat_penalty", "去除重号惩罚", replace(base, repeat_penalty=0.0)),
        StrategyVariant("no_omission", "去除遗漏权重", replace(base, omission_weight=0.0)),
        StrategyVariant("no_span", "去除跨度权重", replace(base, span_weight=0.0)),
        StrategyVariant(
            "position_stack",
            "长期加近期位置",
            replace(
                zero,
                position_weight=base.position_weight,
                recent_position_weight=base.recent_position_weight,
            ),
        ),
        StrategyVariant(
            "recent_position_only",
            "仅近期位置",
            replace(zero, recent_position_weight=base.recent_position_weight),
        ),
        StrategyVariant("sum_only", "仅和值分布", replace(zero, sum_weight=base.sum_weight)),
        StrategyVariant("span_only", "仅跨度分布", replace(zero, span_weight=base.span_weight)),
        StrategyVariant(
            "shape_stack",
            "和值跨度形态",
            replace(
                zero,
                sum_weight=base.sum_weight,
                span_weight=base.span_weight,
                pattern_weight=base.pattern_weight,
            ),
        ),
    ]


def _max_losing_streak(flags: list[bool]) -> int:
    current = 0
    longest = 0
    for flag in flags:
        if flag:
            current = 0
            continue
        current += 1
        longest = max(longest, current)
    return longest


def _strategy_stats_from_picks(picks: list[BacktestPick], top_n: int) -> dict:
    rounds = len(picks)
    if rounds == 0:
        return {
            "rounds": 0,
            "hits": 0,
            "expected_hits": 0.0,
            "hit_rate": 0.0,
            "hit_lift": 0.0,
            "hit_z_score": 0.0,
            "roi": 0.0,
            "max_losing_streak": 0,
        }

    flags = [pick.actual_number in pick.candidates[:top_n] for pick in picks]
    hits = sum(1 for flag in flags if flag)
    probability = top_n / 1000.0
    expected_hits = rounds * probability
    variance = rounds * probability * (1.0 - probability)
    stake = rounds * top_n * 2.0
    payout = hits * 1040.0
    roi = (payout - stake) / stake if stake else 0.0
    return {
        "rounds": rounds,
        "hits": hits,
        "expected_hits": expected_hits,
        "hit_rate": hits / rounds,
        "hit_lift": hits / expected_hits if expected_hits else 0.0,
        "hit_z_score": (hits - expected_hits) / sqrt(variance) if variance > 0 else 0.0,
        "roi": roi,
        "max_losing_streak": _max_losing_streak(flags),
    }


def _annual_failures(picks: list[BacktestPick], top_n: int) -> int:
    grouped: dict[int, list[BacktestPick]] = {}
    for pick in picks:
        if pick.draw_date is None:
            continue
        grouped.setdefault(pick.draw_date.year, []).append(pick)
    failures = 0
    for yearly_picks in grouped.values():
        if len(yearly_picks) < 30:
            continue
        stats = _strategy_stats_from_picks(yearly_picks, top_n)
        if stats["hit_z_score"] <= 0:
            failures += 1
    return failures


def _candidate_status(
    all_stats: dict,
    recent60: dict,
    recent120: dict,
    annual_failures: int,
) -> tuple[str, str]:
    if (
        all_stats["hit_z_score"] >= 2.0
        and recent60["hit_z_score"] > 0
        and recent120["hit_z_score"] > 0
        and annual_failures == 0
    ):
        return "ready", "全局、短窗和年度压力通过。"
    if all_stats["hit_z_score"] >= 2.0:
        return "watch", "全局达标，年度或短窗仍有缺口。"
    return "blocked", "全局 z 值未达到启用线。"


def choose_strategy_candidate(candidates: list[StrategyCandidate]) -> StrategyCandidate:
    ready = [item for item in candidates if item.status == "ready"]
    if ready:
        return sorted(
            ready,
            key=lambda item: (-item.hit_z_score, -item.hit_lift, item.annual_failures, item.name),
        )[0]
    baseline = next((item for item in candidates if item.name == "baseline"), None)
    if baseline is not None:
        return baseline
    return sorted(candidates, key=lambda item: (-item.hit_z_score, item.name))[0]


def _strategy_cache_key(
    draws: list[Draw],
    top_n: int,
    training_window: int,
    recent_window: int,
    min_history: int,
    limit_rows: int,
    strategy_top_n: int,
) -> dict:
    variants = _strategy_variants(recent_window=recent_window, min_history=min_history)
    first = draws[0]
    latest = draws[-1]
    return {
        "version": 2,
        "rows": len(draws),
        "first_issue": first.issue,
        "first_date": first.draw_date.isoformat() if first.draw_date else None,
        "first_number": first.number,
        "latest_issue": latest.issue,
        "latest_date": latest.draw_date.isoformat() if latest.draw_date else None,
        "latest_number": latest.number,
        "top_n": top_n,
        "strategy_top_n": strategy_top_n,
        "training_window": training_window,
        "recent_window": recent_window,
        "min_history": min_history,
        "limit_rows": limit_rows,
        "variants": [
            {
                "name": variant.name,
                "config": asdict(variant.config),
            }
            for variant in variants
        ],
    }


def _strategy_selection_from_dict(payload: dict) -> StrategySelection:
    return StrategySelection(
        active_name=payload["active_name"],
        active_label=payload["active_label"],
        status=payload["status"],
        cache_status=payload.get("cache_status", "hit"),
        summary=payload["summary"],
        candidates=[StrategyCandidate(**item) for item in payload["candidates"]],
    )


def _pick_from_dict(payload: dict) -> BacktestPick:
    raw_date = payload.get("draw_date")
    return BacktestPick(
        issue=payload["issue"],
        draw_date=date.fromisoformat(raw_date) if raw_date else None,
        actual_number=payload["actual_number"],
        candidates=list(payload["candidates"]),
        hit=bool(payload["hit"]),
        stake=float(payload["stake"]),
        payout=float(payload["payout"]),
        pnl=float(payload["pnl"]),
    )


def _pick_to_dict(pick: BacktestPick) -> dict:
    return {
        "issue": pick.issue,
        "draw_date": pick.draw_date.isoformat() if pick.draw_date else None,
        "actual_number": pick.actual_number,
        "candidates": pick.candidates,
        "hit": pick.hit,
        "stake": pick.stake,
        "payout": pick.payout,
        "pnl": pick.pnl,
    }


def _load_strategy_cache(
    cache_path: str | Path | None,
    cache_key: dict,
    config_by_name: dict[str, StrategyConfig],
) -> tuple[StrategySelection, StrategyConfig, list[BacktestPick]] | None:
    if cache_path is None:
        return None
    path = Path(cache_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("cache_key") != cache_key:
            return None
        selection = _strategy_selection_from_dict(payload["strategy_selection"])
        if selection.active_name not in config_by_name:
            return None
        cached_selection = replace(selection, cache_status="hit")
        picks = [_pick_from_dict(item) for item in payload["selected_picks"]]
        return cached_selection, config_by_name[cached_selection.active_name], picks
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _save_strategy_cache(
    cache_path: str | Path | None,
    cache_key: dict,
    selection: StrategySelection,
    selected_picks: list[BacktestPick],
) -> None:
    if cache_path is None:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "strategy_selection": asdict(replace(selection, cache_status="hit")),
        "selected_picks": [_pick_to_dict(pick) for pick in selected_picks],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def select_daily_strategy(
    draws: list[Draw],
    top_n: int,
    training_window: int,
    recent_window: int,
    min_history: int,
    limit_rows: int = 1200,
    cache_path: str | Path | None = None,
) -> tuple[StrategySelection, StrategyConfig, list[BacktestPick]]:
    eval_draws = draws[-limit_rows:] if len(draws) > limit_rows else draws
    strategy_top_n = max(10, min(50, top_n))
    variants = _strategy_variants(recent_window=recent_window, min_history=min_history)
    candidates: list[StrategyCandidate] = []
    result_by_name: dict[str, list[BacktestPick]] = {}
    config_by_name = {variant.name: variant.config for variant in variants}
    cache_key = _strategy_cache_key(
        draws,
        top_n=top_n,
        training_window=training_window,
        recent_window=recent_window,
        min_history=min_history,
        limit_rows=limit_rows,
        strategy_top_n=strategy_top_n,
    )
    cached = _load_strategy_cache(cache_path, cache_key, config_by_name)
    if cached is not None:
        return cached

    for variant in variants:
        result = run_backtest(
            eval_draws,
            top_n=strategy_top_n,
            training_window=training_window,
            config=variant.config,
        )
        result_by_name[variant.name] = result.picks
        all_stats = _strategy_stats_from_picks(result.picks, strategy_top_n)
        recent60 = _strategy_stats_from_picks(result.picks[-60:], strategy_top_n)
        recent120 = _strategy_stats_from_picks(result.picks[-120:], strategy_top_n)
        annual_failures = _annual_failures(result.picks, strategy_top_n)
        status, reason = _candidate_status(all_stats, recent60, recent120, annual_failures)
        primary_number = rank_numbers(draws, top_n=1, config=variant.config)[0].number
        candidates.append(
            StrategyCandidate(
                name=variant.name,
                label=variant.label,
                status=status,
                reason=reason,
                primary_number=primary_number,
                top_n=strategy_top_n,
                rounds=all_stats["rounds"],
                hits=all_stats["hits"],
                expected_hits=all_stats["expected_hits"],
                hit_rate=all_stats["hit_rate"],
                hit_lift=all_stats["hit_lift"],
                hit_z_score=all_stats["hit_z_score"],
                roi=all_stats["roi"],
                recent60_hits=recent60["hits"],
                recent120_hits=recent120["hits"],
                annual_failures=annual_failures,
            )
        )

    selected = choose_strategy_candidate(candidates)
    if selected.status == "ready":
        summary = f"已启用 {selected.label}，因为它通过策略启用闸门。"
        status = "active"
    else:
        summary = "没有候选策略通过启用闸门，继续使用默认策略并保持观察。"
        status = "fallback"
        selected = next(item for item in candidates if item.name == "baseline")

    selection = StrategySelection(
        active_name=selected.name,
        active_label=selected.label,
        status=status,
        cache_status="miss" if cache_path is not None else "disabled",
        summary=summary,
        candidates=sorted(candidates, key=lambda item: (-item.hit_z_score, item.name)),
    )
    config = config_by_name[selected.name]
    picks = result_by_name[selected.name]
    _save_strategy_cache(cache_path, cache_key, selection, picks)
    return selection, config, picks


def build_daily_report(
    draws: list[Draw],
    top_n: int = 10,
    training_window: int = 300,
    recent_window: int = 60,
    min_history: int = 300,
    strategy_cache_path: str | Path | None = None,
    primary_cooldown: PrimaryCooldownState | None = None,
) -> DailyReport:
    strategy_selection, config, selected_picks = select_daily_strategy(
        draws,
        top_n=top_n,
        training_window=training_window,
        recent_window=recent_window,
        min_history=min_history,
        cache_path=strategy_cache_path,
    )
    all_recommendations = rank_numbers(draws, top_n=1000, config=config)
    cooldown_state = primary_cooldown or _empty_primary_cooldown()
    cooled_numbers = cooldown_state.active_numbers
    selected_primary = next(
        (item for item in all_recommendations if item.number not in cooled_numbers),
        all_recommendations[0],
    )
    recommendations = [
        item
        for item in all_recommendations
        if item.number != selected_primary.number
    ][: max(0, top_n - 1)]
    latest = draws[-1]
    primary = _prediction_from_recommendation(selected_primary)
    alternatives = [_prediction_from_recommendation(item) for item in recommendations]
    play_rows = build_play_rows(primary.number, selected_picks)
    confidence_gate = build_confidence_gate(play_rows)
    action_filter = apply_primary_cooldown(
        build_action_filter(confidence_gate, strategy_selection, play_rows),
        primary,
        cooldown_state,
    )
    return DailyReport(
        latest_issue=latest.issue,
        latest_date=latest.draw_date.isoformat() if latest.draw_date else None,
        latest_number=latest.number,
        next_issue_hint=_next_issue_hint(latest.issue),
        mode="analysis_only",
        primary=primary,
        alternatives=alternatives,
        play_rows=play_rows,
        confidence_gate=confidence_gate,
        action_filter=action_filter,
        strategy_selection=strategy_selection,
        full_ranking=[
            _ranking_entry_from_recommendation(item)
            for item in all_recommendations
        ],
        primary_cooldown=cooldown_state,
        source_notes=[
            "规则和固定奖金以公开规则为准，地方派奖或限额销售不纳入本报告。",
            "奖金表来源: https://mzj.gz.gov.cn/gzfcw/gczn/fcsd/content/post_8631790.html",
            "命中率来自最近1200行数据的walk-forward回测。",
        ],
    )


def save_daily_report(
    report: DailyReport,
    output_dir: str | Path,
    meta: dict,
) -> tuple[Path, Path]:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "daily_prediction.json"
    html_path = report_dir / "daily_prediction.html"
    snapshot_dir = report_dir / "snapshots"
    snapshot_path = snapshot_dir / f"prediction_{report.next_issue_hint}.json"
    payload = {
        "meta": meta,
        "report": asdict(report),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_daily_html(report, meta), encoding="utf-8")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path, html_path


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def _format_money(value: float) -> str:
    return f"{value:.2f}"


def _format_count(stats: PlayStats) -> str:
    return f"{stats.hits}/{stats.rounds}" if stats.rounds else "0/0"


def _find_play_row(rows: list[PlayRow], play_id: str) -> PlayRow:
    match = next((row for row in rows if row.play_id == play_id), None)
    if match is None:
        raise ValueError(f"missing play row: {play_id}")
    return match


def _best_play_row(rows: list[PlayRow], prefix: str) -> PlayRow:
    candidates = [row for row in rows if row.play_id.startswith(prefix) and row.active]
    if not candidates:
        return _empty_play_row(prefix)
    return max(
        candidates,
        key=lambda row: (
            row.recent60_stats.hit_rate,
            row.all_stats.hit_rate,
            row.all_stats.roi,
        ),
    )


def _empty_play_row(play_id: str) -> PlayRow:
    stats = _empty_stats()
    return PlayRow(
        play_id=play_id,
        name="暂无",
        selection="本期不适用",
        active=False,
        order_required="本期不适用",
        stake=0.0,
        prize_text="本期不适用",
        net_profit_text="本期不适用",
        theoretical_hit_rate=0.0,
        expected_roi=0.0,
        all_stats=stats,
        recent60_stats=stats,
        recent120_stats=stats,
        note="本期不适用",
    )


def _summary_card(title: str, value: str, detail: str, tone: str = "") -> str:
    class_name = f"summary-card {tone}".strip()
    return (
        f"<div class='{class_name}'>"
        f"<span>{title}</span>"
        f"<strong>{value}</strong>"
        f"<em>{detail}</em>"
        "</div>"
    )


def _gate_checks_html(gate: ConfidenceGate) -> str:
    items = []
    for check in gate.checks:
        css = "pass" if check.status == "通过" else "fail"
        items.append(
            "<li>"
            f"<span class='check-dot {css}'>{check.status}</span>"
            f"<strong>{check.name}</strong>"
            f"<em>{check.metric}</em>"
            "</li>"
        )
    return "\n".join(items)


def _strategy_status_text(status: str) -> str:
    return {
        "ready": "可启用",
        "watch": "观察",
        "blocked": "未达标",
    }.get(status, status)


def _cache_status_text(status: str) -> str:
    return {
        "hit": "缓存命中",
        "miss": "缓存已刷新",
        "disabled": "未启用缓存",
    }.get(status, status)


def _action_tone(status: str) -> str:
    return {
        "action": "action-go",
        "observe": "action-watch",
        "skip": "action-stop",
    }.get(status, "action-watch")


def _strategy_candidates_html(selection: StrategySelection) -> str:
    rows = []
    for item in selection.candidates:
        active = "当前启用" if item.name == selection.active_name else _strategy_status_text(item.status)
        rows.append(
            "<tr>"
            f"<td><strong>{item.label}</strong><span>{active}</span></td>"
            f"<td class='num'>{item.primary_number}</td>"
            f"<td>{item.hits}/{item.rounds}<span>Top{item.top_n} 命中</span></td>"
            f"<td>{item.hit_z_score:.3f}<span>z 值</span></td>"
            f"<td>{item.recent60_hits}/{item.recent120_hits}<span>近60/近120命中</span></td>"
            f"<td>{item.annual_failures}<span>年度失败数</span></td>"
            f"<td>{item.reason}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _play_verdict(row: PlayRow) -> str:
    if row.play_id == "direct":
        return "主号码观察，近60期仍在空窗"
    if row.play_id == "group":
        return "顺序不限，命中频率高于直选"
    if row.play_id == "package":
        if row.all_stats.roi >= 0:
            return "回测略占优，样本仍需继续验证"
        return "回测未占优，仅作辅助"
    if row.play_id.startswith("1d_"):
        return "命中频率更高，奖金低"
    if row.play_id.startswith("2d_"):
        return "比直选更宽，收益压力仍高"
    if row.play_id == "sum":
        return "当前回测偏弱，仅辅助参考"
    return "仅辅助参考"


def _play_rows_html(rows: list[PlayRow]) -> str:
    active_rows = [row for row in rows if row.active]
    if not active_rows:
        return "<tr><td colspan='6' class='empty'>本期不适用</td></tr>"

    visible_ids = {"direct", "group", "package", "1d_h", "1d_t", "1d_o", "2d_ht", "2d_to", "2d_ho", "sum"}
    cells: list[str] = []
    for row in [item for item in active_rows if item.play_id in visible_ids]:
        roi_class = "good" if row.all_stats.roi >= 0 else "bad"
        cells.append(
            "<tr>"
            f"<td><strong>{row.name}</strong><span>{row.note}</span></td>"
            f"<td><b>{row.selection}</b><span>{row.order_required}</span></td>"
            f"<td>{_format_percent(row.theoretical_hit_rate)}<span>理论值</span></td>"
            f"<td>{_format_percent(row.all_stats.hit_rate)}<span>回测 {_format_count(row.all_stats)}</span></td>"
            f"<td>{_format_percent(row.recent60_stats.hit_rate)}<span>近60期 {_format_count(row.recent60_stats)}</span></td>"
            f"<td><b>{_format_money(row.stake)}元</b><span>奖金 {row.prize_text}</span></td>"
            f"<td class='{roi_class}'>{_format_percent(row.all_stats.roi)}<span>{_play_verdict(row)}</span></td>"
            "</tr>"
        )
    return "\n".join(cells)


def _alternatives_html(items: list[DailyPrediction]) -> str:
    rows = []
    for item in items[:4]:
        rows.append(
            "<tr>"
            f"<td>{item.rank}</td>"
            f"<td class='num'>{item.number}</td>"
            f"<td>和值{item.sum_value}, 跨度{item.span}, {item.pattern}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _cooldown_html(cooldown: PrimaryCooldownState) -> str:
    if not cooldown.records:
        return ""
    rows = "".join(
        f"<li><strong>{record.number}</strong><span>{record.reason} 最近开奖 {record.last_issue}，实际 {record.last_actual_number}</span></li>"
        for record in cooldown.records
    )
    return f"""
      <div class="cooldown-strip">
        <div>
          <span>主号冷却</span>
          <strong>已触发</strong>
          <em>连续 {cooldown.threshold} 次未命中的主号暂停进入主位，只保留观察。</em>
        </div>
        <ul>{rows}</ul>
      </div>
    """


def render_daily_html(report: DailyReport, meta: dict) -> str:
    primary = report.primary
    gate = report.confidence_gate
    action = report.action_filter
    strategy = report.strategy_selection
    direct = _find_play_row(report.play_rows, "direct")
    group = _find_play_row(report.play_rows, "group")
    package = _find_play_row(report.play_rows, "package")
    best_one_d = _best_play_row(report.play_rows, "1d_")
    best_two_d = _best_play_row(report.play_rows, "2d_")
    digit_cards = "".join(f"<span>{digit}</span>" for digit in primary.number)
    summary_cards = "".join(
        [
            _summary_card(
                "直选回测",
                _format_percent(direct.all_stats.hit_rate),
                f"{_format_count(direct.all_stats)}，近60期 {_format_count(direct.recent60_stats)}",
                "alert",
            ),
            _summary_card(
                "组选参考",
                group.selection,
                f"{group.name}，回测 {_format_percent(group.all_stats.hit_rate)}",
                "neutral",
            ),
            _summary_card(
                "包选参考",
                package.selection,
                f"{package.name}，回测ROI {_format_percent(package.all_stats.roi)}",
                "neutral",
            ),
            _summary_card(
                "位置参考",
                best_one_d.selection,
                f"{best_one_d.name}，近60期 {_format_percent(best_one_d.recent60_stats.hit_rate)}",
                "neutral",
            ),
            _summary_card(
                "二位参考",
                best_two_d.selection,
                f"{best_two_d.name}，近60期 {_format_percent(best_two_d.recent60_stats.hit_rate)}",
                "neutral",
            ),
        ]
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>福彩3D每日预测简报</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #65758b;
      --line: #d7dee8;
      --brand: #0b5f56;
      --brand-soft: #e8f4f2;
      --gold: #9b7624;
      --red: #a61b2b;
      --green: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.55;
    }}
    .topbar {{
      background: #17324d;
      color: #fff;
      border-bottom: 4px solid var(--gold);
    }}
    .wrap {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 0 24px;
    }}
    .topbar .wrap {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      min-height: 72px;
    }}
    .topbar h1 {{
      font-size: 22px;
      margin: 0;
      font-weight: 700;
    }}
    .topbar span {{
      color: #d8e1eb;
      font-size: 13px;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    .overview {{
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 16px;
      align-items: stretch;
      margin-bottom: 24px;
    }}
    .primary-panel,
    .brief-panel,
    .table-box {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(31, 41, 51, 0.05);
    }}
    .primary-panel {{
      padding: 22px;
    }}
    .brief-panel {{
      padding: 18px;
    }}
    .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .issue {{
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border: 1px solid #bbd8d3;
      border-radius: 999px;
      color: var(--brand);
      background: var(--brand-soft);
      font-size: 13px;
      font-weight: 700;
    }}
    .digits {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin: 18px 0 16px;
    }}
    .digits span {{
      height: 94px;
      display: grid;
      place-items: center;
      border: 1px solid #b9d2ce;
      border-radius: 8px;
      background: #f8fbfb;
      color: var(--brand);
      font-size: 56px;
      line-height: 1;
      font-weight: 800;
      font-family: Consolas, "Microsoft YaHei", monospace;
    }}
    .primary-meta {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .primary-meta strong {{
      display: block;
      color: var(--ink);
      font-size: 16px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .action-strip {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 12px;
      border: 1px solid var(--line);
    }}
    .action-strip.action-go {{
      background: #ecfdf5;
      border-color: #9ad8c8;
    }}
    .action-strip.action-watch {{
      background: #fffaf0;
      border-color: #e7d39a;
    }}
    .action-strip.action-stop {{
      background: #fff6f7;
      border-color: #efc6cb;
    }}
    .action-strip span,
    .action-strip em {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-style: normal;
    }}
    .action-strip strong {{
      display: block;
      color: var(--ink);
      font-size: 26px;
      margin: 4px 0;
    }}
    .action-strip b {{
      min-width: 84px;
      text-align: center;
      border-radius: 999px;
      padding: 7px 10px;
      font-size: 13px;
      color: var(--ink);
      background: #fff;
      border: 1px solid var(--line);
    }}
    .gate-strip {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      border: 1px solid #efc6cb;
      border-radius: 8px;
      background: #fff6f7;
      padding: 12px;
      margin-bottom: 12px;
    }}
    .gate-strip span,
    .gate-strip em {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-style: normal;
    }}
    .gate-strip strong {{
      display: block;
      color: var(--red);
      font-size: 22px;
      margin: 4px 0;
    }}
    .gate-strip b {{
      min-width: 72px;
      text-align: center;
      color: var(--red);
      background: #fff;
      border: 1px solid #efc6cb;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 13px;
    }}
    .cooldown-strip {{
      display: grid;
      grid-template-columns: 180px 1fr;
      gap: 12px;
      border: 1px solid #d6c28b;
      border-radius: 8px;
      background: #fffaf0;
      padding: 12px;
      margin-bottom: 12px;
    }}
    .cooldown-strip span,
    .cooldown-strip em,
    .cooldown-strip li span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-style: normal;
    }}
    .cooldown-strip strong {{
      display: block;
      margin: 4px 0;
      color: #6f4e08;
      font-size: 18px;
    }}
    .cooldown-strip ul {{
      display: grid;
      gap: 8px;
      list-style: none;
      margin: 0;
      padding: 0;
    }}
    .cooldown-strip li {{
      border-left: 3px solid var(--gold);
      padding-left: 10px;
    }}
    .gate-checks {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      padding: 0;
      margin: 0 0 14px;
      list-style: none;
    }}
    .gate-checks li {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 10px;
      min-height: 82px;
    }}
    .gate-checks strong,
    .gate-checks em {{
      display: block;
      font-style: normal;
    }}
    .gate-checks strong {{
      margin: 7px 0 3px;
      font-size: 14px;
    }}
    .gate-checks em {{
      color: var(--muted);
      font-size: 12px;
    }}
    .check-dot {{
      display: inline-flex;
      align-items: center;
      height: 22px;
      padding: 0 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .check-dot.pass {{
      color: var(--green);
      background: #ecfdf5;
    }}
    .check-dot.fail {{
      color: var(--red);
      background: #fff1f2;
    }}
    .summary-card {{
      min-height: 104px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 12px;
    }}
    .summary-card.alert {{
      border-color: #efc6cb;
      background: #fff6f7;
    }}
    .summary-card span,
    .summary-card em,
    td span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-style: normal;
    }}
    .summary-card strong {{
      display: block;
      margin: 8px 0 6px;
      font-size: 21px;
      color: var(--ink);
      overflow-wrap: anywhere;
    }}
    .section-title {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin: 24px 0 10px;
    }}
    .section-title h2 {{
      margin: 0;
      font-size: 18px;
    }}
    .section-title p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .table-box {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; min-width: 760px; }}
    th,
    td {{ border-bottom: 1px solid var(--line); padding: 12px; text-align: left; vertical-align: top; }}
    th {{ color: #41526b; background: #f7f9fb; font-weight: 700; white-space: nowrap; }}
    tr:last-child td {{ border-bottom: 0; }}
    td strong {{ color: var(--ink); }}
    .num {{
      font-family: Consolas, monospace;
      font-weight: 800;
      font-size: 22px;
      color: var(--brand);
    }}
    .good {{ color: var(--green); font-weight: 700; }}
    .bad {{ color: var(--red); font-weight: 700; }}
    .empty {{ text-align: center; color: var(--muted); }}
    .notice {{
      border-left: 4px solid var(--gold);
      background: #fffaf0;
      padding: 12px 14px;
      margin-top: 18px;
      color: #5f4b1c;
      font-size: 14px;
    }}
    footer {{
      margin: 28px 0 4px;
      color: var(--muted);
      font-size: 12px;
    }}
    a {{ color: var(--brand); }}
    @media (max-width: 900px) {{
      .wrap,
      main {{ padding-left: 14px; padding-right: 14px; }}
      .topbar .wrap {{ align-items: flex-start; flex-direction: column; padding-top: 14px; padding-bottom: 14px; }}
      .overview {{ grid-template-columns: 1fr; }}
      .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .gate-checks {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .action-strip {{ align-items: flex-start; flex-direction: column; }}
      .gate-strip {{ align-items: flex-start; flex-direction: column; }}
      .cooldown-strip {{ grid-template-columns: 1fr; }}
      .digits span {{ height: 76px; font-size: 44px; }}
      .primary-meta {{ grid-template-columns: 1fr; }}
      .section-title {{ display: block; }}
      table {{ font-size: 12px; }}
      th,
      td {{ padding: 10px; }}
    }}
  </style>
</head>
<body>
<div class="topbar">
  <div class="wrap">
    <div>
      <h1>福彩3D每日预测简报</h1>
      <span>数据最新至 {report.latest_date}，第 {report.latest_issue} 期，开奖号码 {report.latest_number}</span>
    </div>
      <span>出手建议: {action.label}，策略: {strategy.active_label}</span>
  </div>
</div>
<main>
  <section class="overview">
    <div class="primary-panel">
      <div class="label">下一期主号码</div>
      <div class="issue">预测期号参考 {report.next_issue_hint}</div>
      <div class="digits">{digit_cards}</div>
      <div class="primary-meta">
        <div><strong>{primary.sum_value}</strong>和值</div>
        <div><strong>{primary.span}</strong>跨度</div>
        <div><strong>{primary.pattern}</strong>形态</div>
      </div>
    </div>
    <div class="brief-panel">
      <div class="label">决策摘要</div>
      <div class="action-strip {_action_tone(action.status)}">
        <div>
          <span>高置信出手过滤</span>
          <strong>{action.label}</strong>
          <em>{action.recommendation}</em>
        </div>
        <b>{action.stake_level}投入</b>
      </div>
      <div class="gate-strip">
        <div>
          <span>策略闸门</span>
          <strong>{gate.label}</strong>
          <em>{gate.summary}</em>
        </div>
        <b>置信 {gate.confidence}</b>
      </div>
      {_cooldown_html(report.primary_cooldown)}
      <ul class="gate-checks">{_gate_checks_html(gate)}</ul>
      <div class="summary-grid">{summary_cards}</div>
      <div class="notice">严格结论: {action.reason} {gate.action} 本页用于提高选号纪律和复盘质量，不能把短期命中当成稳定能力。</div>
    </div>
  </section>

  <div class="section-title">
    <h2>策略自动切换</h2>
    <p>{strategy.summary} {_cache_status_text(strategy.cache_status)}</p>
  </div>
  <div class="table-box">
    <table>
      <thead>
        <tr>
          <th>策略</th><th>当前主号</th><th>全局命中</th><th>z值</th><th>短窗</th><th>年度</th><th>结论</th>
        </tr>
      </thead>
      <tbody>{_strategy_candidates_html(strategy)}</tbody>
    </table>
  </div>

  <div class="section-title">
    <h2>可行动玩法</h2>
    <p>只显示本期适用玩法，隐藏无效项</p>
  </div>
  <div class="table-box">
    <table>
      <thead>
        <tr>
          <th>玩法</th><th>本期内容</th><th>理论命中率</th><th>回测命中率</th><th>近60期</th><th>成本和奖金</th><th>判断</th>
        </tr>
      </thead>
      <tbody>{_play_rows_html(report.play_rows)}</tbody>
    </table>
  </div>

  <div class="section-title">
    <h2>备选号</h2>
    <p>只保留前4个备选，减少干扰</p>
  </div>
  <div class="table-box">
    <table class="compact">
      <thead><tr><th>排名</th><th>号码</th><th>特征</th></tr></thead>
      <tbody>{_alternatives_html(report.alternatives)}</tbody>
    </table>
  </div>

  <footer>
    规则和固定奖金参考 <a href="https://mzj.gz.gov.cn/gzfcw/gczn/fcsd/content/post_8631790.html">广州民政局福彩3D玩法说明</a>。地方派奖和限号销售可能影响实际兑奖，请以购彩地销售规则为准。
  </footer>
</main>
</body>
</html>
"""
