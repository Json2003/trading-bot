#!/usr/bin/env python3
"""Research-only adaptive BTC/ETH volatility-regime momentum model.

Version 3 is deliberately separate from the v2 runner and active paper
configuration.  It uses closed hourly candles, schedules fills for the next
bar open, and keeps the two-asset decision in one portfolio so BTC and ETH
cannot both be entered as identical independent strategies.

The executable is a backtest/research tool only.  A candidate is paper-ready
only when its walk-forward promotion gate is true; this script never changes
the active portfolio configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

try:  # Works both as ``python scripts/...`` and when imported by pytest.
    from scripts.run_momentum_volatility_research import Bar, features, load_bars
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    from run_momentum_volatility_research import Bar, features, load_bars


def _valid(value: float) -> bool:
    return math.isfinite(value)


@dataclass(frozen=True)
class PairBar:
    timestamp: object
    btc: Bar
    eth: Bar


@dataclass(frozen=True)
class V3Config:
    """Frozen research parameters for one v3 candidate."""

    atr_period: int = 14
    regime_window: int = 720
    volume_lookback: int = 20
    fast_window: int = 8
    slow_window: int = 21
    regime_span: int = 200
    breakout_lookback: int = 20
    expansion_lookback: int = 24
    expansion_ratio: float = 1.05
    min_vol_rank: float = 0.35
    reduce_size_rank: float = 0.75
    extreme_vol_rank: float = 0.90
    max_atr_pct: float = 0.06
    extreme_realized_rank: float = 0.95
    reduced_size_multiplier: float = 0.50
    leader_lookback: int = 24
    leader_min_score: float = 0.05
    leader_margin: float = 0.05
    entry_min_body_atr: float = 0.50
    entry_volume_multiplier: float = 1.00
    require_higher_timeframe_trend: bool = True
    higher_timeframe_bars: int = 4
    higher_timeframe_fast_window: int = 5
    higher_timeframe_slow_window: int = 20
    hard_stop_atr: float = 2.0
    trailing_stop_atr: float = 2.5
    time_stop_bars: int = 72
    time_stop_min_return: float = 0.005
    profit_lock_activation_atr: float = 1.00
    profit_lock_floor_atr: float = 0.25
    kill_switch_drawdown: float = 0.02
    cooldown_bars: int = 12
    stable_bars: int = 12
    recovery_size_multiplier: float = 0.50
    recovery_size_bars: int = 24
    recovery_abort_drawdown: float = 0.01
    lifetime_drawdown_limit: float = 0.06

    def as_dict(self) -> dict[str, object]:
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class _PendingEntry:
    symbol: str
    size_multiplier: float
    atr: float
    signal_index: int


@dataclass(frozen=True)
class _PendingExit:
    reason: str


def align_pair(btc: Sequence[Bar], eth: Sequence[Bar]) -> list[PairBar]:
    """Align BTC and ETH by timestamp and reject ambiguous input."""

    btc_by_time = {bar.timestamp: bar for bar in btc}
    eth_by_time = {bar.timestamp: bar for bar in eth}
    if len(btc_by_time) != len(btc) or len(eth_by_time) != len(eth):
        raise ValueError("pair data contains duplicate timestamps")
    timestamps = sorted(set(btc_by_time).intersection(eth_by_time))
    if len(timestamps) < 2:
        raise ValueError("BTC and ETH have fewer than two common timestamps")
    return [PairBar(timestamp, btc_by_time[timestamp], eth_by_time[timestamp]) for timestamp in timestamps]


def _asset_bars(pair: Sequence[PairBar], symbol: str) -> list[Bar]:
    if symbol == "BTC":
        return [item.btc for item in pair]
    if symbol == "ETH":
        return [item.eth for item in pair]
    raise ValueError(f"unsupported pair symbol: {symbol}")


def _prior_highs(bars: Sequence[Bar], lookback: int) -> list[float]:
    result = [math.nan] * len(bars)
    for index in range(lookback + 1, len(bars)):
        # The previous two candles are confirmation candles.  They are not
        # allowed to become part of the breakout reference range.
        prior = bars[index - lookback - 1 : index - 1]
        if len(prior) == lookback:
            result[index] = max(bar.high for bar in prior)
    return result


def build_pair_features(
    pair: Sequence[PairBar],
    config: V3Config,
) -> dict[str, dict[str, list[float]]]:
    """Build causal features for both assets over the common time axis."""

    result: dict[str, dict[str, list[float]]] = {}
    for symbol in ("BTC", "ETH"):
        bars = _asset_bars(pair, symbol)
        base = features(
            bars,
            atr_period=config.atr_period,
            regime_window=config.regime_window,
            volume_lookback=config.volume_lookback,
            higher_timeframe_bars=config.higher_timeframe_bars,
            higher_timeframe_fast_window=config.higher_timeframe_fast_window,
            higher_timeframe_slow_window=config.higher_timeframe_slow_window,
        )
        closes = [bar.close for bar in bars]
        score = [math.nan] * len(bars)
        for index in range(config.leader_lookback, len(bars)):
            realized = base["realized"][index]
            if not _valid(realized) or realized <= 0:
                continue
            change = math.log(closes[index] / closes[index - config.leader_lookback])
            # Risk-adjusted return, using only data available at the signal
            # close.  A positive score is required before selecting a leader.
            score[index] = change / max(realized, base["atr_pct"][index], 1e-9)
        base["prior_high"] = _prior_highs(bars, config.breakout_lookback)
        base["leader_score"] = score
        result[symbol] = base
    return result


def _trend_ok(
    bars: Sequence[Bar],
    f: Mapping[str, list[float]],
    index: int,
    config: V3Config,
) -> bool:
    regime_key = f"ema_regime_{config.regime_span}"
    if regime_key not in f or index < config.expansion_lookback:
        return False
    regime = f[regime_key][index]
    prior_regime = f[regime_key][index - config.expansion_lookback]
    return (
        _valid(f["ema_fast"][index])
        and _valid(f["ema_slow"][index])
        and _valid(regime)
        and _valid(prior_regime)
        and f["ema_fast"][index] > f["ema_slow"][index]
        and bars[index].close > f["ema_slow"][index]
        and bars[index].close > regime
        and regime > prior_regime
    )


def _volatility_state(
    f: Mapping[str, list[float]],
    index: int,
    config: V3Config,
) -> tuple[str, float, float]:
    """Return (state, size multiplier, expansion ratio) at a signal close."""

    atr_pct = f["atr_pct"][index]
    prior_index = index - config.expansion_lookback
    prior_atr_pct = f["atr_pct"][prior_index] if prior_index >= 0 else math.nan
    atr_rank = f["atr_rank"][index]
    realized_rank = f["realized_rank"][index]
    if not all(_valid(value) for value in (atr_pct, prior_atr_pct, atr_rank, realized_rank)):
        return "unknown", 0.0, math.nan
    expansion_ratio = atr_pct / prior_atr_pct if prior_atr_pct > 0 else math.inf
    extreme = (
        atr_rank >= config.extreme_vol_rank
        or realized_rank >= config.extreme_realized_rank
        or atr_pct > config.max_atr_pct
    )
    if extreme:
        return "extreme", 0.0, expansion_ratio
    if atr_rank < config.min_vol_rank:
        return "quiet", 0.0, expansion_ratio
    if expansion_ratio < config.expansion_ratio:
        return "not_expanded", 0.0, expansion_ratio
    multiplier = (
        config.reduced_size_multiplier
        if atr_rank >= config.reduce_size_rank
        else 1.0
    )
    return ("reduced" if multiplier < 1.0 else "normal"), multiplier, expansion_ratio


def _setup(
    bars: Sequence[Bar],
    f: Mapping[str, list[float]],
    index: int,
    config: V3Config,
) -> tuple[bool, float, str]:
    """Evaluate one asset without using the next bar."""

    if index < max(2, config.breakout_lookback + 1) or index + 1 >= len(bars):
        return False, 0.0, "warmup"
    state, size_multiplier, _ = _volatility_state(f, index, config)
    if state not in {"normal", "reduced"}:
        return False, 0.0, state
    if not _trend_ok(bars, f, index, config):
        return False, 0.0, "trend"
    prior_high = f["prior_high"][index]
    if not _valid(prior_high) or bars[index].close <= prior_high:
        return False, 0.0, "breakout"
    first, second = bars[index - 1], bars[index]
    atr = f["atr"][index]
    bullish = first.close > first.open and second.close > second.open
    body = second.close - second.open
    body_ok = _valid(atr) and body >= config.entry_min_body_atr * atr
    median_volume = f["volume_median"][index]
    volume_ok = (
        config.entry_volume_multiplier <= 0
        or (
            _valid(median_volume)
            and second.volume > 0
            and second.volume >= config.entry_volume_multiplier * median_volume
        )
    )
    higher_timeframe_ok = (
        not config.require_higher_timeframe_trend
        or bool(f["higher_timeframe_trend"][index])
    )
    if not bullish or not body_ok or not volume_ok or not higher_timeframe_ok:
        return False, 0.0, "confirmation"
    return True, size_multiplier, state


def _leader(
    feature_map: Mapping[str, Mapping[str, list[float]]],
    index: int,
    config: V3Config,
) -> tuple[str | None, dict[str, float]]:
    scores = {
        symbol: feature_map[symbol]["leader_score"][index]
        for symbol in ("BTC", "ETH")
    }
    valid_scores = {symbol: value for symbol, value in scores.items() if _valid(value)}
    if not valid_scores:
        return None, scores
    ordered = sorted(valid_scores.items(), key=lambda item: item[1], reverse=True)
    winner, winner_score = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else -math.inf
    if winner_score < config.leader_min_score:
        return None, scores
    if len(ordered) > 1 and winner_score - runner_up < config.leader_margin:
        return None, scores
    return winner, scores


def _sell(cash: float, qty: float, price: float, fee: float, slip: float) -> float:
    return cash + qty * price * (1.0 - slip) * (1.0 - fee)


def _buy_fill(cash: float, price: float, desired_notional: float, fee: float, slip: float) -> tuple[float, float, float]:
    notional = min(max(desired_notional, 0.0), cash)
    fill = price * (1.0 + slip)
    if notional <= 0 or fill <= 0:
        return cash, 0.0, fill
    return cash - notional, notional / fill * (1.0 - fee), fill


def run_pair(
    pair: Sequence[PairBar],
    *,
    initial_balance: float,
    order_notional: float,
    fees_bps: float,
    slippage_bps: float,
    config: V3Config = V3Config(),
    start_index: int = 0,
    end_index: int | None = None,
    feature_map: Mapping[str, Mapping[str, list[float]]] | None = None,
) -> dict[str, object]:
    """Run one causal pair portfolio over an inclusive/exclusive slice."""

    if initial_balance <= 0 or order_notional <= 0:
        raise ValueError("initial_balance and order_notional must be positive")
    if not all(
        math.isfinite(value) and value >= 0
        for value in (fees_bps, slippage_bps)
    ):
        raise ValueError("fees_bps and slippage_bps must be finite and non-negative")
    if not pair:
        raise ValueError("pair data must not be empty")
    if not 0 < config.reduced_size_multiplier <= 1:
        raise ValueError("reduced_size_multiplier must be in (0, 1]")
    if not 0 < config.recovery_size_multiplier <= 1:
        raise ValueError("recovery_size_multiplier must be in (0, 1]")
    if not 0 < config.kill_switch_drawdown < config.lifetime_drawdown_limit:
        raise ValueError("kill switch must be below lifetime drawdown limit")
    if config.time_stop_bars <= 0 or config.recovery_size_bars <= 0:
        raise ValueError("time and recovery windows must be positive")
    if end_index is None:
        end_index = len(pair)
    if not 0 <= start_index < end_index <= len(pair):
        raise ValueError("invalid backtest slice")

    feature_map = feature_map or build_pair_features(pair, config)
    btc = _asset_bars(pair, "BTC")
    eth = _asset_bars(pair, "ETH")
    bars_by_symbol = {"BTC": btc, "ETH": eth}
    fee = fees_bps / 10_000.0
    slip = slippage_bps / 10_000.0
    cash = float(initial_balance)
    position_symbol: str | None = None
    qty = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    entry_index = -1
    highest = 0.0
    hard_stop = 0.0
    trailing_stop = 0.0
    profit_stop = 0.0
    profit_lock_active = False
    pending_entry: _PendingEntry | None = None
    pending_exit: _PendingExit | None = None

    lifetime_peak = initial_balance
    risk_peak = initial_balance
    max_drawdown = 0.0
    recovery_baseline = initial_balance
    recovery_until = -1
    halted = False
    blocked_until = -1
    stable_count = 0
    recovery_mode = False
    permanent_halt = False
    permanent_halt_reason: str | None = None

    entries = 0
    exits = 0
    kill_events = 0
    recovery_events = 0
    time_stop_exits = 0
    profit_lock_activations = 0
    profit_lock_exits = 0
    stop_exits = 0
    trend_exits = 0
    size_reductions = 0
    extreme_blocked_entries = 0
    leader_counts = {"BTC": 0, "ETH": 0}
    leader_scores_seen = 0
    entry_reasons: dict[str, int] = {}

    last_index = end_index - 1

    def current_equity(index: int) -> float:
        if position_symbol is None:
            return cash
        return cash + qty * bars_by_symbol[position_symbol][index].close

    def flatten(index: int, *, price: float | None = None) -> None:
        nonlocal cash, qty, position_symbol, pending_entry, pending_exit, exits
        if position_symbol is None or qty <= 0:
            pending_entry = pending_exit = None
            return
        mark = price if price is not None else bars_by_symbol[position_symbol][index].close
        cash = _sell(cash, qty, mark, fee, slip)
        qty = 0.0
        position_symbol = None
        pending_entry = pending_exit = None
        exits += 1

    for index in range(start_index, end_index):
        # Pending signals are formed from the prior close and fill at the next
        # bar open.  A pending order is cancelled by a halt before it can fill.
        if pending_exit is not None and position_symbol is not None:
            bar = bars_by_symbol[position_symbol][index]
            cash = _sell(cash, qty, bar.open, fee, slip)
            reason = pending_exit.reason
            exits += 1
            if reason == "time_stop":
                time_stop_exits += 1
            elif reason == "profit_lock":
                profit_lock_exits += 1
            elif reason == "trend":
                trend_exits += 1
            pending_exit = None
            position_symbol = None
            qty = 0.0

        if pending_entry is not None and position_symbol is None and not halted and not permanent_halt:
            selected = pending_entry
            bar = bars_by_symbol[selected.symbol][index]
            desired = order_notional * selected.size_multiplier
            cash, qty, fill = _buy_fill(cash, bar.open, desired, fee, slip)
            if qty > 0:
                position_symbol = selected.symbol
                entry_price = fill
                entry_atr = selected.atr if _valid(selected.atr) and selected.atr > 0 else fill * 0.02
                entry_index = selected.signal_index
                highest = fill
                hard_stop = fill - config.hard_stop_atr * entry_atr
                trailing_stop = fill - config.trailing_stop_atr * entry_atr
                profit_stop = 0.0
                profit_lock_active = False
                entries += 1
                entry_reasons[selected.symbol] = entry_reasons.get(selected.symbol, 0) + 1
            pending_entry = None

        # Manage an existing position with the current bar.  A newly raised
        # trailing/profit stop applies to the next bar, avoiding a same-bar
        # high-then-low assumption.
        stop_triggered = False
        if position_symbol is not None:
            bars = bars_by_symbol[position_symbol]
            f = feature_map[position_symbol]
            bar = bars[index]
            active_stop = max(hard_stop, trailing_stop, profit_stop)
            if bar.low <= active_stop:
                cash = _sell(cash, qty, active_stop, fee, slip)
                exits += 1
                stop_exits += 1
                if profit_lock_active and active_stop >= profit_stop > 0:
                    profit_lock_exits += 1
                position_symbol = None
                qty = 0.0
                pending_exit = None
                stop_triggered = True
            else:
                highest = max(highest, bar.high)
                atr = f["atr"][index]
                if _valid(atr) and atr > 0:
                    trailing_stop = max(trailing_stop, highest - config.trailing_stop_atr * atr)
                if not profit_lock_active and highest >= entry_price + config.profit_lock_activation_atr * entry_atr:
                    profit_lock_active = True
                    profit_lock_activations += 1
                    profit_stop = entry_price + config.profit_lock_floor_atr * entry_atr
                elif profit_lock_active:
                    profit_stop = max(
                        profit_stop,
                        entry_price + config.profit_lock_floor_atr * entry_atr,
                    )

        equity = current_equity(index)
        lifetime_peak = max(lifetime_peak, equity)
        max_drawdown = max(
            max_drawdown,
            (lifetime_peak - equity) / lifetime_peak if lifetime_peak else 0.0,
        )
        lifetime_dd = (
            (lifetime_peak - equity) / lifetime_peak if lifetime_peak else 0.0
        )

        # The lifetime limit is not reset by the ordinary 2% recovery cycle.
        if lifetime_dd >= config.lifetime_drawdown_limit and not permanent_halt:
            flatten(index)
            permanent_halt = True
            permanent_halt_reason = "lifetime_drawdown_limit"
            halted = True
            recovery_mode = False
            continue

        if recovery_mode and equity <= recovery_baseline * (1.0 - config.recovery_abort_drawdown):
            flatten(index)
            permanent_halt = True
            permanent_halt_reason = "recovery_abort_drawdown"
            halted = True
            recovery_mode = False
            continue

        risk_peak = max(risk_peak, equity)
        risk_dd = (risk_peak - equity) / risk_peak if risk_peak else 0.0
        if risk_dd >= config.kill_switch_drawdown and not halted and not permanent_halt:
            flatten(index)
            kill_events += 1
            halted = True
            recovery_mode = False
            blocked_until = index + config.cooldown_bars
            stable_count = 0
            risk_peak = cash
            continue

        if permanent_halt or index + 1 >= end_index:
            continue

        # Evaluate both assets at the current close.  All arrays are causal;
        # the pending entry/exit is therefore filled no earlier than index+1.
        setups: dict[str, tuple[bool, float, str]] = {
            symbol: _setup(
                bars_by_symbol[symbol], feature_map[symbol], index, config
            )
            for symbol in ("BTC", "ETH")
        }
        leader, scores = _leader(feature_map, index, config)
        if leader is not None:
            leader_counts[leader] += 1
            leader_scores_seen += 1
        for symbol, (_, _, reason) in setups.items():
            if reason == "extreme":
                extreme_blocked_entries += 1

        healthy = any(
            _trend_ok(bars_by_symbol[symbol], feature_map[symbol], index, config)
            and _volatility_state(feature_map[symbol], index, config)[0] in {"normal", "reduced"}
            for symbol in ("BTC", "ETH")
        )
        if halted:
            if index < blocked_until:
                stable_count = 0
                continue
            if healthy:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= config.stable_bars:
                halted = False
                recovery_mode = True
                recovery_events += 1
                recovery_baseline = current_equity(index)
                recovery_until = index + config.recovery_size_bars
                risk_peak = recovery_baseline
            else:
                continue

        if recovery_mode and index >= recovery_until:
            recovery_mode = False

        if stop_triggered:
            continue

        if position_symbol is not None:
            f = feature_map[position_symbol]
            bars = bars_by_symbol[position_symbol]
            bearish = (
                bars[index].close < bars[index].open
                and _valid(f["atr"][index])
                and bars[index].open - bars[index].close >= config.entry_min_body_atr * f["atr"][index]
                and index > 0
                and bars[index].close < bars[index - 1].low
                and bars[index].close < f["ema_fast"][index]
            )
            age = index - entry_index
            weak_trend = not _trend_ok(bars, f, index, config)
            time_stop = age >= config.time_stop_bars and (
                bars[index].close / entry_price - 1.0 < config.time_stop_min_return
            )
            if pending_exit is None and (weak_trend or bearish or time_stop):
                reason = "time_stop" if time_stop else "trend"
                pending_exit = _PendingExit(reason)
            continue

        if halted or permanent_halt or pending_entry is not None:
            continue
        if leader is None:
            continue
        setup_ok, volatility_size, setup_reason = setups[leader]
        if not setup_ok:
            continue
        if volatility_size < 1.0:
            size_reductions += 1
        recovery_size = (
            config.recovery_size_multiplier if recovery_mode else 1.0
        )
        if _valid(feature_map[leader]["atr"][index]) and feature_map[leader]["atr"][index] > 0:
            pending_entry = _PendingEntry(
                symbol=leader,
                size_multiplier=volatility_size * recovery_size,
                atr=feature_map[leader]["atr"][index],
                signal_index=index,
            )
            entry_reasons[setup_reason] = entry_reasons.get(setup_reason, 0) + 1

    if position_symbol is not None:
        flatten(last_index)
    ending_balance = cash
    final_equity = ending_balance
    lifetime_peak = max(lifetime_peak, final_equity)
    max_drawdown = max(
        max_drawdown,
        (lifetime_peak - final_equity) / lifetime_peak if lifetime_peak else 0.0,
    )
    return {
        "start": pair[start_index].timestamp.isoformat(),
        "end": pair[last_index].timestamp.isoformat(),
        "bars": end_index - start_index,
        "initial_balance": initial_balance,
        "ending_balance": ending_balance,
        "pnl": ending_balance - initial_balance,
        "return_pct": (ending_balance / initial_balance - 1.0) * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "lifetime_drawdown_pct": ((lifetime_peak - ending_balance) / lifetime_peak * 100.0) if lifetime_peak else 0.0,
        "entries": entries,
        "exits": exits,
        "trades": entries + exits,
        "kill_switch": kill_events > 0,
        "kill_events": kill_events,
        "recovery_events": recovery_events,
        "permanent_halt": permanent_halt,
        "permanent_halt_reason": permanent_halt_reason,
        "time_stop_exits": time_stop_exits,
        "stop_exits": stop_exits,
        "trend_exits": trend_exits,
        "profit_lock_activations": profit_lock_activations,
        "profit_lock_exits": profit_lock_exits,
        "size_reductions": size_reductions,
        "extreme_blocked_entries": extreme_blocked_entries,
        "leader_counts": leader_counts,
        "leader_scores_seen": leader_scores_seen,
        "entry_reasons": entry_reasons,
        "params": {
            **config.as_dict(),
            "initial_balance": initial_balance,
            "order_notional": order_notional,
            "fees_bps": fees_bps,
            "slippage_bps": slippage_bps,
            "start_index": start_index,
            "end_index": end_index,
        },
    }


CONFIRMATION_HOLDOUT_BARS = 365 * 24
MIN_CONFIRMATION_ENTRIES = 5
MIN_FULL_ENTRIES = 8


def _folds(length: int) -> list[tuple[int, int]]:
    holdout_start = length - CONFIRMATION_HOLDOUT_BARS
    if holdout_start <= 0:
        raise ValueError("data must include a positive one-year confirmation holdout")
    first = int(holdout_start * 0.50)
    second = int(holdout_start * 0.625)
    third = int(holdout_start * 0.75)
    return [(first, second), (second, third), (third, holdout_start)]


def _confirmation_holdout(length: int) -> tuple[int, int]:
    start = length - CONFIRMATION_HOLDOUT_BARS
    if start <= 0:
        raise ValueError("data must include a positive one-year confirmation holdout")
    return start, length


def _result_number(result: Mapping[str, object], key: str) -> float:
    try:
        value = float(result.get(key, math.nan))
    except (TypeError, ValueError):
        return math.nan
    return value


def _result_integer(result: Mapping[str, object], key: str) -> int:
    try:
        value = int(result.get(key, -1))
    except (TypeError, ValueError):
        return -1
    return value


def _median_return(results: Sequence[Mapping[str, object]]) -> float:
    values = [_result_number(result, "return_pct") for result in results]
    if not values or not all(_valid(value) for value in values):
        return math.nan
    return statistics.median(values)


def _promotion_gate(
    full: Mapping[str, object],
    stress_full: Mapping[str, object],
    walk_forward: Sequence[Mapping[str, object]],
    stress_walk_forward: Sequence[Mapping[str, object]],
    confirmation_base: Mapping[str, object],
    confirmation_stress: Mapping[str, object],
) -> dict[str, object]:
    base_median = _median_return(walk_forward)
    stress_median = _median_return(stress_walk_forward)
    reasons: list[str] = []
    if not _valid(base_median) or base_median <= 0:
        reasons.append("base walk-forward median return is not positive and finite")
    if not _valid(stress_median) or stress_median <= 0:
        reasons.append("higher-cost walk-forward median return is not positive and finite")

    required_results = (
        ("full sample", full),
        ("stress full sample", stress_full),
        ("confirmation holdout base", confirmation_base),
        ("confirmation holdout stress", confirmation_stress),
    )
    for label, result in required_results:
        if not isinstance(result, Mapping):
            reasons.append(f"missing {label} result")
            continue
        return_pct = _result_number(result, "return_pct")
        if not _valid(return_pct):
            reasons.append(f"{label} return is not finite")
        elif return_pct <= 0:
            reasons.append(f"{label} return is not positive")
        minimum_entries = (
            MIN_FULL_ENTRIES
            if label in {"full sample", "stress full sample"}
            else MIN_CONFIRMATION_ENTRIES
        )
        if _result_integer(result, "entries") < minimum_entries:
            reasons.append(f"{label} has fewer than {minimum_entries} entries")
        permanent_halt = result.get("permanent_halt")
        if not isinstance(permanent_halt, bool):
            reasons.append(f"{label} permanent_halt flag is invalid")
        elif permanent_halt:
            reasons.append(f"{label} hit a permanent halt")
        kill_events = _result_integer(result, "kill_events")
        if kill_events < 0:
            reasons.append(f"{label} kill_events field is invalid")
        elif kill_events > 1:
            reasons.append(f"{label} has repeated kill-switch events")

    for label, folds in (
        ("base walk-forward", walk_forward),
        ("stress walk-forward", stress_walk_forward),
    ):
        if len(folds) != 3:
            reasons.append(f"{label} must contain exactly three folds")
            continue
        for index, fold in enumerate(folds, start=1):
            if not isinstance(fold, Mapping):
                reasons.append(f"{label} fold {index} is invalid")
                continue
            if not _valid(_result_number(fold, "return_pct")):
                reasons.append(f"{label} fold {index} return is not finite")
            if _result_integer(fold, "entries") < MIN_CONFIRMATION_ENTRIES:
                reasons.append(
                    f"{label} fold {index} has fewer than {MIN_CONFIRMATION_ENTRIES} entries"
                )
            permanent_halt = fold.get("permanent_halt")
            if not isinstance(permanent_halt, bool):
                reasons.append(f"{label} fold {index} permanent_halt flag is invalid")
            elif permanent_halt:
                reasons.append(f"{label} fold {index} hit a permanent halt")
            kill_events = _result_integer(fold, "kill_events")
            if kill_events < 0:
                reasons.append(f"{label} fold {index} kill_events field is invalid")
            elif kill_events > 1:
                reasons.append(f"{label} fold {index} has repeated kill-switch events")

    return {
        "pass": not reasons,
        "base_walk_forward_median_return_pct": base_median,
        "stress_walk_forward_median_return_pct": stress_median,
        "confirmation_holdout_base_return_pct": _result_number(
            confirmation_base, "return_pct"
        ),
        "confirmation_holdout_stress_return_pct": _result_number(
            confirmation_stress, "return_pct"
        ),
        "requirements": {
            "positive_base_median_walk_forward": True,
            "positive_stress_median_walk_forward": True,
            "positive_full_sample_base_and_stress": True,
            "positive_confirmation_holdout_base_and_stress": True,
            "minimum_full_sample_entries": MIN_FULL_ENTRIES,
            "minimum_entries_per_walk_forward_fold": MIN_CONFIRMATION_ENTRIES,
            "minimum_confirmation_holdout_entries": MIN_CONFIRMATION_ENTRIES,
            "no_permanent_halt": True,
            "no_repeated_kill_switch": True,
        },
        "failure_reasons": reasons,
    }


def research(
    btc_path: Path,
    eth_path: Path,
    *,
    initial_balance: float = 75_000.0,
    order_notional: float = 6_000.0,
    fees_bps: float = 10.0,
    slippage_bps: float = 5.0,
    stress_fees_bps: float = 20.0,
    stress_slippage_bps: float = 10.0,
    horizon_initial_balance: float = 5_000.0,
    horizon_order_notional: float | None = None,
) -> dict[str, object]:
    btc = load_bars(btc_path)
    eth = load_bars(eth_path)
    pair = align_pair(btc, eth)
    if horizon_order_notional is None:
        horizon_order_notional = order_notional
    cost_values = (
        fees_bps,
        slippage_bps,
        stress_fees_bps,
        stress_slippage_bps,
    )
    if not all(math.isfinite(value) and value >= 0 for value in cost_values):
        raise ValueError("all cost assumptions must be finite and non-negative")
    if (
        stress_fees_bps < fees_bps
        or stress_slippage_bps < slippage_bps
        or (stress_fees_bps == fees_bps and stress_slippage_bps == slippage_bps)
    ):
        raise ValueError(
            "stress costs must be no lower in either component and higher in at least one"
        )

    candidates: dict[str, V3Config] = {
        "balanced": V3Config(
            expansion_ratio=1.05,
            min_vol_rank=0.25,
            reduce_size_rank=0.75,
            extreme_vol_rank=0.90,
            trailing_stop_atr=2.5,
            time_stop_bars=72,
        ),
        "selective": V3Config(
            expansion_ratio=1.10,
            min_vol_rank=0.35,
            reduce_size_rank=0.75,
            extreme_vol_rank=0.90,
            trailing_stop_atr=2.5,
            time_stop_bars=72,
        ),
        "conservative": V3Config(
            expansion_ratio=1.10,
            min_vol_rank=0.35,
            reduce_size_rank=0.70,
            extreme_vol_rank=0.85,
            trailing_stop_atr=3.0,
            time_stop_bars=96,
            profit_lock_activation_atr=1.25,
        ),
    }
    report: dict[str, object] = {
        "model": "adaptive_momentum_volatility_v3",
        "research_only": True,
        "paper_promotion_required": True,
        "data": {
            "btc_path": str(btc_path),
            "eth_path": str(eth_path),
            "btc_bars": len(btc),
            "eth_bars": len(eth),
            "aligned_bars": len(pair),
            "start": pair[0].timestamp.isoformat(),
            "end": pair[-1].timestamp.isoformat(),
        },
        "costs": {
            "base": {"fees_bps": fees_bps, "slippage_bps": slippage_bps},
            "stress": {"fees_bps": stress_fees_bps, "slippage_bps": stress_slippage_bps},
        },
        "portfolio": {"initial_balance": initial_balance, "order_notional": order_notional},
        "candidate_definitions": {name: config.as_dict() for name, config in candidates.items()},
    }
    folds = _folds(len(pair))
    confirmation_start, confirmation_end = _confirmation_holdout(len(pair))
    candidates_report: dict[str, object] = {}
    for name, config in candidates.items():
        feature_map = build_pair_features(pair, config)
        full = run_pair(
            pair,
            initial_balance=initial_balance,
            order_notional=order_notional,
            fees_bps=fees_bps,
            slippage_bps=slippage_bps,
            config=config,
            feature_map=feature_map,
        )
        stress_full = run_pair(
            pair,
            initial_balance=initial_balance,
            order_notional=order_notional,
            fees_bps=stress_fees_bps,
            slippage_bps=stress_slippage_bps,
            config=config,
            feature_map=feature_map,
        )
        wf = [
            run_pair(
                pair,
                initial_balance=initial_balance,
                order_notional=order_notional,
                fees_bps=fees_bps,
                slippage_bps=slippage_bps,
                config=config,
                start_index=start,
                end_index=end,
                feature_map=feature_map,
            )
            for start, end in folds
        ]
        stress_wf = [
            run_pair(
                pair,
                initial_balance=initial_balance,
                order_notional=order_notional,
                fees_bps=stress_fees_bps,
                slippage_bps=stress_slippage_bps,
                config=config,
                start_index=start,
                end_index=end,
                feature_map=feature_map,
            )
            for start, end in folds
        ]
        confirmation_base = run_pair(
            pair,
            initial_balance=initial_balance,
            order_notional=horizon_order_notional,
            fees_bps=fees_bps,
            slippage_bps=slippage_bps,
            config=config,
            start_index=confirmation_start,
            end_index=confirmation_end,
            feature_map=feature_map,
        )
        confirmation_stress = run_pair(
            pair,
            initial_balance=initial_balance,
            order_notional=horizon_order_notional,
            fees_bps=stress_fees_bps,
            slippage_bps=stress_slippage_bps,
            config=config,
            start_index=confirmation_start,
            end_index=confirmation_end,
            feature_map=feature_map,
        )
        horizon_sizes = {
            "1d": 24,
            "1w": 24 * 7,
            "1m": 24 * 30,
            "1y": 24 * 365,
        }
        horizons: dict[str, object] = {}
        for label, bars_count in horizon_sizes.items():
            start = max(0, len(pair) - bars_count)
            horizons[label] = run_pair(
                pair,
                initial_balance=horizon_initial_balance,
                order_notional=horizon_order_notional,
                fees_bps=fees_bps,
                slippage_bps=slippage_bps,
                config=config,
                start_index=start,
                feature_map=feature_map,
            )
        candidates_report[name] = {
            "full_sample": full,
            "full_sample_stress": stress_full,
            "walk_forward": wf,
            "stress_walk_forward": stress_wf,
            "confirmation_holdout": {
                "base": confirmation_base,
                "stress": confirmation_stress,
                "start_index": confirmation_start,
                "end_index": confirmation_end,
                "bars": confirmation_end - confirmation_start,
            },
            "walk_forward_medians": {
                "base_return_pct": _median_return(wf),
                "stress_return_pct": _median_return(stress_wf),
            },
            "horizons": horizons,
            "promotion_gate": _promotion_gate(
                full,
                stress_full,
                wf,
                stress_wf,
                confirmation_base,
                confirmation_stress,
            ),
        }
    report["walk_forward_folds"] = [
        {"start_index": start, "end_index": end}
        for start, end in folds
    ]
    report["confirmation_holdout"] = {
        "start_index": confirmation_start,
        "end_index": confirmation_end,
        "bars": confirmation_end - confirmation_start,
        "one_year": True,
    }
    report["candidates"] = candidates_report
    report["paper_promotion"] = {
        "passed_candidates": [
            name
            for name, value in candidates_report.items()
            if bool(value["promotion_gate"]["pass"])
        ],
        "active_profile_changed": False,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-path", type=Path, required=True)
    parser.add_argument("--eth-path", type=Path, required=True)
    parser.add_argument("--initial-balance", type=float, default=75_000.0)
    parser.add_argument("--order-notional", type=float, default=6_000.0)
    parser.add_argument("--fees-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--stress-fees-bps", type=float, default=20.0)
    parser.add_argument("--stress-slippage-bps", type=float, default=10.0)
    parser.add_argument("--horizon-initial-balance", type=float, default=5_000.0)
    parser.add_argument("--horizon-order-notional", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = research(
        args.btc_path,
        args.eth_path,
        initial_balance=args.initial_balance,
        order_notional=args.order_notional,
        fees_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
        stress_fees_bps=args.stress_fees_bps,
        stress_slippage_bps=args.stress_slippage_bps,
        horizon_initial_balance=args.horizon_initial_balance,
        horizon_order_notional=args.horizon_order_notional,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
