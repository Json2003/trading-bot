#!/usr/bin/env python3
"""Research a causal momentum-plus-volatility breakout strategy.

This runner intentionally uses only the Python standard library. Signals are
formed at a bar close and orders fill at the next bar open, with an ATR trail,
fees, slippage, and the 2% kill switch. It is research-only and does not touch
the paper operator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_bars(path: Path) -> list[Bar]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        bars = []
        for row in rows:
            raw = row.get("timestamp") or row.get("ts") or row.get("date")
            if raw is None:
                raise ValueError("CSV requires timestamp, ts, or date")
            try:
                ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                value = int(float(raw))
                if value >= 10**14:
                    value //= 1000
                ts = datetime.fromtimestamp(value / 1000, tz=timezone.utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            bars.append(Bar(ts, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row.get("volume", 0))))
    bars.sort(key=lambda item: item.timestamp)
    if not bars:
        raise ValueError("historical CSV is empty")
    if any(a.timestamp == b.timestamp for a, b in zip(bars, bars[1:])):
        raise ValueError("historical CSV contains duplicate timestamps")
    if any(min(b.open, b.high, b.low, b.close) <= 0 or b.high < b.low for b in bars):
        raise ValueError("historical CSV contains invalid OHLC values")
    return bars


def _ema(values: list[float], span: int) -> list[float]:
    alpha = 2 / (span + 1)
    result: list[float] = []
    value = values[0]
    for item in values:
        value = alpha * item + (1 - alpha) * value
        result.append(value)
    return result


def _percentile(value: float, prior: list[float]) -> float:
    if not prior:
        return math.nan
    return sum(item <= value for item in prior) / len(prior)


def _higher_timeframe_flags(
    bars: list[Bar],
    *,
    group_size: int,
    fast_window: int,
    slow_window: int,
) -> list[float]:
    """Return a causal higher-timeframe trend flag for every source bar.

    A higher-timeframe candle is only added after its final source candle has
    closed. Incomplete buckets are discarded when a timestamp gap or bucket
    boundary is encountered, so the flag never uses future candles.
    """

    flags = [0.0] * len(bars)
    completed_closes: list[float] = []
    current_bucket: int | None = None
    current_count = 0
    fast_value: float | None = None
    slow_value: float | None = None
    previous_slow: float | None = None
    current_flag = 0.0
    for index, bar in enumerate(bars):
        bucket = int(bar.timestamp.timestamp()) // (group_size * 3600)
        if bucket != current_bucket:
            current_bucket = bucket
            current_count = 0
        current_count += 1
        if current_count == group_size:
            completed_closes.append(bar.close)
            alpha_fast = 2.0 / (fast_window + 1.0)
            alpha_slow = 2.0 / (slow_window + 1.0)
            previous_slow = slow_value
            fast_value = (
                bar.close
                if fast_value is None
                else alpha_fast * bar.close + (1.0 - alpha_fast) * fast_value
            )
            slow_value = (
                bar.close
                if slow_value is None
                else alpha_slow * bar.close + (1.0 - alpha_slow) * slow_value
            )
            if (
                len(completed_closes) >= slow_window
                and fast_value is not None
                and slow_value is not None
                and previous_slow is not None
            ):
                current_flag = float(
                    fast_value > slow_value
                    and bar.close > slow_value
                    and slow_value > previous_slow
                )
        flags[index] = current_flag
    return flags


def features(
    bars: list[Bar],
    *,
    atr_period: int = 14,
    regime_window: int = 720,
    volume_lookback: int = 20,
    higher_timeframe_bars: int = 4,
    higher_timeframe_fast_window: int = 5,
    higher_timeframe_slow_window: int = 20,
) -> dict[str, list[float]]:
    closes = [bar.close for bar in bars]
    true_ranges: list[float] = []
    returns: list[float] = [math.nan]
    for i, bar in enumerate(bars):
        previous = closes[i - 1] if i else bar.close
        true_ranges.append(max(bar.high - bar.low, abs(bar.high - previous), abs(bar.low - previous)))
        returns.append(math.log(bar.close / previous) if i and previous > 0 else math.nan)
    atr: list[float] = [math.nan] * len(bars)
    atr_pct: list[float] = [math.nan] * len(bars)
    realized: list[float] = [math.nan] * len(bars)
    atr_rank: list[float] = [math.nan] * len(bars)
    realized_rank: list[float] = [math.nan] * len(bars)
    volume_median: list[float] = [math.nan] * len(bars)
    for i in range(len(bars)):
        if i + 1 >= atr_period:
            atr[i] = sum(true_ranges[i - atr_period + 1 : i + 1]) / atr_period
            atr_pct[i] = atr[i] / closes[i]
        if i >= 23:
            sample = [x for x in returns[i - 23 : i + 1] if not math.isnan(x)]
            realized[i] = statistics.pstdev(sample) if len(sample) > 1 else math.nan
        prior_atr = [x for x in atr_pct[max(0, i - regime_window) : i] if not math.isnan(x)]
        prior_realized = [x for x in realized[max(0, i - regime_window) : i] if not math.isnan(x)]
        prior_volumes = [
            bar.volume
            for bar in bars[max(0, i - volume_lookback) : i]
            if bar.volume > 0
        ]
        if prior_volumes:
            volume_median[i] = statistics.median(prior_volumes)
        if atr_pct[i] == atr_pct[i] and len(prior_atr) >= 100:
            atr_rank[i] = _percentile(atr_pct[i], prior_atr)
        if realized[i] == realized[i] and len(prior_realized) >= 100:
            realized_rank[i] = _percentile(realized[i], prior_realized)
    return {
        "ema_fast": _ema(closes, 8),
        "ema_slow": _ema(closes, 21),
        "ema_regime_100": _ema(closes, 100),
        "ema_regime_200": _ema(closes, 200),
        "atr": atr,
        "atr_pct": atr_pct,
        "atr_rank": atr_rank,
        "realized": realized,
        "realized_rank": realized_rank,
        "volume_median": volume_median,
        "higher_timeframe_trend": _higher_timeframe_flags(
            bars,
            group_size=higher_timeframe_bars,
            fast_window=higher_timeframe_fast_window,
            slow_window=higher_timeframe_slow_window,
        ),
    }


def _summary(values: list[float]) -> dict[str, float]:
    clean = sorted(x for x in values if x == x)
    if not clean:
        return {}
    return {f"p{p}": clean[min(len(clean) - 1, int((len(clean) - 1) * p / 100))] for p in (10, 25, 50, 75, 90, 95)}


def run(
    bars: list[Bar],
    *,
    initial_balance: float,
    order_notional: float,
    fees_bps: float,
    slippage_bps: float,
    lower_vol_rank: float = 0.25,
    upper_vol_rank: float = 0.90,
    expansion_ratio: float = 1.05,
    trail_atr: float = 2.5,
    regime_span: int = 100,
    cooldown_bars: int = 12,
    stable_bars: int = 12,
    require_two_bullish_candles: bool = True,
    exit_on_bearish_candle: bool = True,
    entry_min_body_atr: float = 0.5,
    breakout_lookback: int = 20,
    entry_volume_multiplier: float = 1.0,
    require_higher_timeframe_trend: bool = True,
    higher_timeframe_bars: int = 4,
    higher_timeframe_fast_window: int = 5,
    higher_timeframe_slow_window: int = 20,
    bearish_exit_min_body_atr: float = 0.75,
    bearish_exit_requires_breakdown: bool = True,
    hard_stop_atr: float = 2.0,
) -> dict[str, object]:
    if initial_balance <= 0 or order_notional <= 0:
        raise ValueError("initial_balance and order_notional must be positive")
    if entry_min_body_atr < 0 or bearish_exit_min_body_atr < 0:
        raise ValueError("candle body ATR thresholds must be non-negative")
    if breakout_lookback <= 0:
        raise ValueError("breakout_lookback must be positive")
    if entry_volume_multiplier < 0:
        raise ValueError("entry_volume_multiplier must be non-negative")
    if (
        higher_timeframe_bars <= 0
        or higher_timeframe_fast_window <= 0
        or higher_timeframe_slow_window <= 0
        or higher_timeframe_fast_window >= higher_timeframe_slow_window
    ):
        raise ValueError("invalid higher timeframe parameters")
    if hard_stop_atr <= 0 or trail_atr <= 0:
        raise ValueError("hard_stop_atr and trail_atr must be positive")
    f = features(
        bars,
        higher_timeframe_bars=higher_timeframe_bars,
        higher_timeframe_fast_window=higher_timeframe_fast_window,
        higher_timeframe_slow_window=higher_timeframe_slow_window,
    )
    fee = fees_bps / 10_000
    slip = slippage_bps / 10_000
    cash, qty = initial_balance, 0.0
    entry, hard_stop, stop, highest = 0.0, 0.0, 0.0, 0.0
    peak, risk_peak, max_dd = initial_balance, initial_balance, 0.0
    trades, killed, pending_entry, pending_exit = 0, False, False, False
    entry_points, expansion_points, tradable_points = 0, 0, 0
    bullish_confirmation_points, bearish_exit_points = 0, 0
    kill_events, recovery_events = 0, 0
    halted, blocked_until, stable_count = False, -1, 0
    for i, bar in enumerate(bars):
        if pending_exit and qty:
            fill = bar.open * (1 - slip)
            cash += qty * fill * (1 - fee)
            qty, trades, pending_exit = 0.0, trades + 1, False
        if pending_entry and not qty and cash > 0:
            fill = bar.open * (1 + slip)
            notional = min(order_notional, cash)
            qty = notional / fill * (1 - fee)
            cash -= notional
            entry, highest = fill, bar.open
            stop_atr = (
                f["atr"][i - 1]
                if i and f["atr"][i - 1] == f["atr"][i - 1]
                else entry * 0.02
            )
            hard_stop = entry - hard_stop_atr * stop_atr
            stop = entry - trail_atr * stop_atr
            trades, pending_entry = trades + 1, False
        stop_triggered = False
        if qty:
            active_stop = max(hard_stop, stop)
            if bar.low <= active_stop:
                fill = active_stop * (1 - slip)
                cash += qty * fill * (1 - fee)
                qty, trades, pending_exit = 0.0, trades + 1, False
                stop_triggered = True
            else:
                highest = max(highest, bar.high)
                atr = f["atr"][i]
                if atr == atr:
                    stop = max(stop, highest - trail_atr * atr)
        equity = cash + qty * bar.close
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak else 0.0)
        risk_peak = max(risk_peak, equity)
        risk_dd = (risk_peak - equity) / risk_peak if risk_peak else 0.0
        if risk_dd >= 0.02 and not halted:
            if qty:
                fill = bar.close * (1 - slip)
                cash += qty * fill * (1 - fee)
                qty, trades = 0.0, trades + 1
            killed = True
            kill_events += 1
            blocked_until = i + cooldown_bars
            stable_count = 0
            halted = True
            risk_peak = cash
            pending_entry = pending_exit = False
            continue
        if i < 2 or i + 1 >= len(bars):
            continue
        atr_rank = f["atr_rank"][i]
        rv_rank = f["realized_rank"][i]
        atr_now, atr_prior = f["atr_pct"][i], f["atr_pct"][i - 24] if i >= 24 else math.nan
        prior_start = i - breakout_lookback - 1
        prior_end = i - 1
        prior_bars = bars[max(0, prior_start) : prior_end]
        prior_high = max((x.high for x in prior_bars), default=math.nan)
        tradable = (
            atr_rank == atr_rank and rv_rank == rv_rank and lower_vol_rank <= atr_rank <= upper_vol_rank
            and 0.005 <= atr_now <= 0.05 and atr_prior == atr_prior and atr_now / atr_prior >= expansion_ratio
        )
        regime = f[f"ema_regime_{regime_span}"][i]
        regime_prior = f[f"ema_regime_{regime_span}"][i - 24] if i >= 24 else math.nan
        trend = (
            f["ema_fast"][i] > f["ema_slow"][i]
            and bars[i].close > f["ema_slow"][i]
            and regime == regime
            and bars[i].close > regime
            and regime_prior == regime_prior
            and regime > regime_prior
        )
        breakout = prior_high == prior_high and bars[i].close > prior_high
        if tradable:
            tradable_points += 1
        if atr_now == atr_now and atr_prior == atr_prior and atr_now / atr_prior >= expansion_ratio:
            expansion_points += 1
        if trend and breakout:
            entry_points += 1
        bullish_confirmation = (
            bars[i - 1].close > bars[i - 1].open
            and bars[i].close > bars[i].open
        )
        if bullish_confirmation:
            bullish_confirmation_points += 1
        atr_value = f["atr"][i]
        volume_confirmation = (
            f["volume_median"][i] == f["volume_median"][i]
            and bars[i].volume > 0
            and bars[i].volume >= entry_volume_multiplier * f["volume_median"][i]
        )
        body_confirmation = (
            atr_value == atr_value
            and bars[i].close - bars[i].open >= entry_min_body_atr * atr_value
        )
        higher_timeframe_confirmation = bool(f["higher_timeframe_trend"][i])
        bearish_exit = (
            exit_on_bearish_candle
            and bars[i].close < bars[i].open
            and atr_value == atr_value
            and (bars[i].open - bars[i].close) >= bearish_exit_min_body_atr * atr_value
            and (
                not bearish_exit_requires_breakdown
                or (
                    i > 0
                    and bars[i].close < bars[i - 1].low
                    and bars[i].close < f["ema_fast"][i]
                )
            )
        )
        if bearish_exit:
            bearish_exit_points += 1
        if halted and i < blocked_until:
            stable_count = 0
            continue
        if halted and stable_count < stable_bars:
            if trend and tradable:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= stable_bars:
                recovery_events += 1
                halted = False
                risk_peak = cash
        if stop_triggered:
            continue
        entry_confirmation_ok = (
            not require_two_bullish_candles
            or (
                bullish_confirmation
                and body_confirmation
                and breakout
                and volume_confirmation
                and (
                    not require_higher_timeframe_trend
                    or higher_timeframe_confirmation
                )
            )
        )
        if not qty and not halted and trend and breakout and tradable and entry_confirmation_ok:
            pending_entry = True
        if qty and (
            f["ema_fast"][i] <= f["ema_slow"][i]
            or bars[i].close < f["ema_slow"][i]
            or bearish_exit
        ):
            pending_exit = True
    if qty:
        cash += qty * bars[-1].close * (1 - slip) * (1 - fee)
        trades += 1
    return {
        "start": bars[0].timestamp.isoformat(), "end": bars[min(len(bars) - 1, i)].timestamp.isoformat(),
        "bars": min(len(bars), i + 1), "ending_balance": cash, "pnl": cash - initial_balance,
        "return_pct": (cash / initial_balance - 1) * 100, "max_drawdown_pct": max_dd * 100,
        "trades": trades, "kill_switch": killed, "kill_events": kill_events, "recovery_events": recovery_events, "entry_points": entry_points,
        "expansion_points": expansion_points, "tradable_points": tradable_points,
        "bullish_confirmation_points": bullish_confirmation_points,
        "body_confirmation_points": sum(
            1
            for i in range(len(bars))
            if f["atr"][i] == f["atr"][i]
            and bars[i].close - bars[i].open >= entry_min_body_atr * f["atr"][i]
        ),
        "volume_confirmation_points": sum(
            1
            for i in range(len(bars))
            if f["volume_median"][i] == f["volume_median"][i]
            and bars[i].volume > 0
            and bars[i].volume >= entry_volume_multiplier * f["volume_median"][i]
        ),
        "higher_timeframe_points": sum(1 for value in f["higher_timeframe_trend"] if value),
        "bearish_exit_points": bearish_exit_points,
        "params": {
            "lower_vol_rank": lower_vol_rank,
            "upper_vol_rank": upper_vol_rank,
            "expansion_ratio": expansion_ratio,
            "trail_atr": trail_atr,
            "regime_span": regime_span,
            "cooldown_bars": cooldown_bars,
            "stable_bars": stable_bars,
            "order_notional": order_notional,
            "require_two_bullish_candles": require_two_bullish_candles,
            "exit_on_bearish_candle": exit_on_bearish_candle,
            "entry_min_body_atr": entry_min_body_atr,
            "breakout_lookback": breakout_lookback,
            "entry_volume_multiplier": entry_volume_multiplier,
            "require_higher_timeframe_trend": require_higher_timeframe_trend,
            "higher_timeframe_bars": higher_timeframe_bars,
            "higher_timeframe_fast_window": higher_timeframe_fast_window,
            "higher_timeframe_slow_window": higher_timeframe_slow_window,
            "bearish_exit_min_body_atr": bearish_exit_min_body_atr,
            "bearish_exit_requires_breakdown": bearish_exit_requires_breakdown,
            "hard_stop_atr": hard_stop_atr,
        },
    }


def research(
    path: Path,
    initial_balance: float,
    order_notional: float,
    *,
    require_two_bullish_candles: bool = True,
    exit_on_bearish_candle: bool = True,
    entry_min_body_atr: float = 0.5,
    breakout_lookback: int = 20,
    entry_volume_multiplier: float = 1.0,
    require_higher_timeframe_trend: bool = True,
    higher_timeframe_bars: int = 4,
    higher_timeframe_fast_window: int = 5,
    higher_timeframe_slow_window: int = 20,
    bearish_exit_min_body_atr: float = 0.75,
    bearish_exit_requires_breakdown: bool = True,
    hard_stop_atr: float = 2.0,
) -> dict[str, object]:
    bars = load_bars(path)
    f = features(bars)
    report: dict[str, object] = {
        "dataset": str(path), "bars": len(bars), "start": bars[0].timestamp.isoformat(), "end": bars[-1].timestamp.isoformat(),
        "data_points": {"atr_pct": _summary(f["atr_pct"]), "realized_hourly_vol": _summary(f["realized"]), "atr_rank": _summary(f["atr_rank"]), "realized_rank": _summary(f["realized_rank"])},
    }
    candidates = {
        "balanced": {"lower_vol_rank": 0.25, "upper_vol_rank": 0.90, "expansion_ratio": 1.05, "trail_atr": 2.5, "regime_span": 100},
        "strict": {"lower_vol_rank": 0.35, "upper_vol_rank": 0.85, "expansion_ratio": 1.10, "trail_atr": 2.5, "regime_span": 200},
        "wide": {"lower_vol_rank": 0.15, "upper_vol_rank": 0.95, "expansion_ratio": 1.00, "trail_atr": 3.0, "regime_span": 100},
    }
    report["candle_rule"] = {
        "require_two_bullish_candles": require_two_bullish_candles,
        "exit_on_bearish_candle": exit_on_bearish_candle,
        "entry_min_body_atr": entry_min_body_atr,
        "breakout_lookback": breakout_lookback,
        "entry_volume_multiplier": entry_volume_multiplier,
        "require_higher_timeframe_trend": require_higher_timeframe_trend,
        "higher_timeframe_bars": higher_timeframe_bars,
        "higher_timeframe_fast_window": higher_timeframe_fast_window,
        "higher_timeframe_slow_window": higher_timeframe_slow_window,
        "bearish_exit_min_body_atr": bearish_exit_min_body_atr,
        "bearish_exit_requires_breakdown": bearish_exit_requires_breakdown,
        "hard_stop_atr": hard_stop_atr,
    }
    report["candidates"] = {
        name: run(
            bars,
            initial_balance=initial_balance,
            order_notional=order_notional,
            fees_bps=10,
            slippage_bps=5,
            require_two_bullish_candles=require_two_bullish_candles,
            exit_on_bearish_candle=exit_on_bearish_candle,
            entry_min_body_atr=entry_min_body_atr,
            breakout_lookback=breakout_lookback,
            entry_volume_multiplier=entry_volume_multiplier,
            require_higher_timeframe_trend=require_higher_timeframe_trend,
            higher_timeframe_bars=higher_timeframe_bars,
            higher_timeframe_fast_window=higher_timeframe_fast_window,
            higher_timeframe_slow_window=higher_timeframe_slow_window,
            bearish_exit_min_body_atr=bearish_exit_min_body_atr,
            bearish_exit_requires_breakdown=bearish_exit_requires_breakdown,
            hard_stop_atr=hard_stop_atr,
            **params,
        )
        for name, params in candidates.items()
    }
    folds = [(0, int(len(bars) * 0.5), int(len(bars) * 0.5), int(len(bars) * 0.625)), (0, int(len(bars) * 0.625), int(len(bars) * 0.625), int(len(bars) * 0.75)), (0, int(len(bars) * 0.75), int(len(bars) * 0.75), len(bars))]
    report["walk_forward"] = {
        name: [
            run(
                bars[test_start:test_end],
                initial_balance=initial_balance,
                order_notional=order_notional,
                fees_bps=10,
                slippage_bps=5,
                require_two_bullish_candles=require_two_bullish_candles,
                exit_on_bearish_candle=exit_on_bearish_candle,
                entry_min_body_atr=entry_min_body_atr,
                breakout_lookback=breakout_lookback,
                entry_volume_multiplier=entry_volume_multiplier,
                require_higher_timeframe_trend=require_higher_timeframe_trend,
                higher_timeframe_bars=higher_timeframe_bars,
                higher_timeframe_fast_window=higher_timeframe_fast_window,
                higher_timeframe_slow_window=higher_timeframe_slow_window,
                bearish_exit_min_body_atr=bearish_exit_min_body_atr,
                bearish_exit_requires_breakdown=bearish_exit_requires_breakdown,
                hard_stop_atr=hard_stop_atr,
                **params,
            )
            for _, _, test_start, test_end in folds
        ]
        for name, params in candidates.items()
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--initial-balance", type=float, default=75_000)
    parser.add_argument("--order-notional", type=float, default=6_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--disable-candle-confirmation",
        action="store_true",
        help="Research-only comparison run without the v2 candle/volume/HTF confirmation layer",
    )
    args = parser.parse_args()
    v2 = not args.disable_candle_confirmation
    report = research(
        args.path,
        args.initial_balance,
        args.order_notional,
        require_two_bullish_candles=v2,
        exit_on_bearish_candle=v2,
        entry_min_body_atr=0.5 if v2 else 0.0,
        entry_volume_multiplier=1.0 if v2 else 0.0,
        require_higher_timeframe_trend=v2,
        bearish_exit_min_body_atr=0.75 if v2 else 0.0,
        bearish_exit_requires_breakdown=v2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
