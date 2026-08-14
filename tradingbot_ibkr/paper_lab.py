"""Isolated multi-account paper strategy and execution tournament.

Every virtual account receives the same chronological OHLCV sequence but keeps
independent cash, positions, risk limits, fills and trade history. Candidate
logic and execution policies come from fixed allowlists; an operator cannot
submit arbitrary Python or promote a winner to the trading runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from backtest.metrics import max_drawdown, sharpe_ratio

STRATEGY_FAMILIES = frozenset(
    {"ema_momentum", "volume_breakout", "mean_reversion", "trend_pullback"}
)
EXECUTION_POLICIES = frozenset(
    {"next_open_market", "pullback_limit", "breakout_stop"}
)


@dataclass(frozen=True, slots=True)
class ExecutionAssumptions:
    spread_bps: float = 12.0
    slippage_bps: float = 8.0
    commission_per_order: float = 0.0

    def __post_init__(self) -> None:
        values = (self.spread_bps, self.slippage_bps, self.commission_per_order)
        if not all(math.isfinite(float(value)) and value >= 0 for value in values):
            raise ValueError("execution costs must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class StrategyProfile:
    account_id: str
    strategy: str
    execution_policy: str
    params: Mapping[str, float]
    starting_cash: float = 2_000.0
    risk_per_trade: float = 0.01
    max_position_fraction: float = 0.90
    max_daily_loss_fraction: float = 0.03
    stop_atr: float = 1.5
    reward_to_risk: float = 2.0
    max_hold_bars: int = 30

    def __post_init__(self) -> None:
        if not self.account_id.strip():
            raise ValueError("account_id is required")
        if self.strategy not in STRATEGY_FAMILIES:
            raise ValueError(f"unsupported strategy: {self.strategy}")
        if self.execution_policy not in EXECUTION_POLICIES:
            raise ValueError(f"unsupported execution policy: {self.execution_policy}")
        if self.starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        if not 0 < self.risk_per_trade <= 0.05:
            raise ValueError("risk_per_trade must be between 0 and 5%")
        if not 0 < self.max_position_fraction <= 1:
            raise ValueError("max_position_fraction must be between 0 and 1")
        if not 0 < self.max_daily_loss_fraction <= 0.10:
            raise ValueError("max_daily_loss_fraction must be between 0 and 10%")
        if self.stop_atr <= 0 or self.reward_to_risk <= 0:
            raise ValueError("stop and reward multiples must be positive")
        if not 1 <= self.max_hold_bars <= 500:
            raise ValueError("max_hold_bars must be between 1 and 500")


@dataclass(frozen=True, slots=True)
class TradeRecord:
    account_id: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    bars_held: int
    exit_reason: str
    execution_cost: float


@dataclass(frozen=True, slots=True)
class AccountReport:
    account_id: str
    strategy: str
    execution_policy: str
    starting_cash: float
    ending_equity: float
    total_return: float
    max_drawdown: float
    sharpe: float
    profit_factor: float
    win_rate: float
    expectancy: float
    trade_count: int
    rejected_entries: int
    average_execution_cost: float
    score: float
    trades: tuple[TradeRecord, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TournamentReport:
    dataset_rows: int
    evaluation_rows: int
    holdout_fraction: float
    assumptions: ExecutionAssumptions
    accounts: tuple[AccountReport, ...]

    @property
    def leaderboard(self) -> tuple[AccountReport, ...]:
        return tuple(sorted(self.accounts, key=lambda item: item.score, reverse=True))

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_rows": self.dataset_rows,
            "evaluation_rows": self.evaluation_rows,
            "holdout_fraction": self.holdout_fraction,
            "assumptions": asdict(self.assumptions),
            "leaderboard": [item.as_dict() for item in self.leaderboard],
            "promotion_allowed": False,
            "review_required": True,
        }


class PaperStrategyTournament:
    """Replay an identical held-out sequence through isolated virtual accounts."""

    def __init__(
        self,
        profiles: Sequence[StrategyProfile],
        *,
        assumptions: ExecutionAssumptions | None = None,
        holdout_fraction: float = 0.30,
    ) -> None:
        if len(profiles) < 2:
            raise ValueError("at least two strategy profiles are required")
        account_ids = [profile.account_id for profile in profiles]
        if len(account_ids) != len(set(account_ids)):
            raise ValueError("account_id values must be unique")
        if not 0.15 <= holdout_fraction <= 0.70:
            raise ValueError("holdout_fraction must be between 15% and 70%")
        self._profiles = tuple(profiles)
        self._assumptions = assumptions or ExecutionAssumptions()
        self._holdout_fraction = float(holdout_fraction)

    def run(self, data: pd.DataFrame) -> TournamentReport:
        frame = _prepare_data(data)
        evaluation_start = max(30, int(len(frame) * (1.0 - self._holdout_fraction)))
        if evaluation_start >= len(frame) - 10:
            raise ValueError("dataset is too short for the requested holdout")
        reports = tuple(
            self._simulate_account(frame, evaluation_start, profile)
            for profile in self._profiles
        )
        return TournamentReport(
            dataset_rows=len(frame),
            evaluation_rows=len(frame) - evaluation_start,
            holdout_fraction=self._holdout_fraction,
            assumptions=self._assumptions,
            accounts=reports,
        )

    def _simulate_account(
        self,
        frame: pd.DataFrame,
        evaluation_start: int,
        profile: StrategyProfile,
    ) -> AccountReport:
        signal = _build_signal(frame, profile)
        atr = _atr(frame, 14)
        cash = float(profile.starting_cash)
        quantity = 0.0
        entry_price = 0.0
        entry_index = -1
        entry_time = ""
        stop_price = 0.0
        take_profit = 0.0
        entry_cost = 0.0
        trades: list[TradeRecord] = []
        equity_curve: list[float] = []
        rejected_entries = 0
        day_key: str | None = None
        day_start_equity = cash

        for index in range(evaluation_start, len(frame)):
            row = frame.iloc[index]
            timestamp = pd.Timestamp(row["timestamp"])
            current_day = timestamp.date().isoformat()
            mark_price = float(row["close"])
            equity = cash + quantity * mark_price
            if current_day != day_key:
                day_key = current_day
                day_start_equity = equity

            if quantity > 0:
                bars_held = index - entry_index
                exit_reason: str | None = None
                raw_exit = mark_price
                # Conservative intrabar ordering: a stop wins when both stop and
                # target are touched in the same OHLC bar.
                if float(row["low"]) <= stop_price:
                    raw_exit = stop_price
                    exit_reason = "stop"
                elif float(row["high"]) >= take_profit:
                    raw_exit = take_profit
                    exit_reason = "target"
                elif bars_held >= profile.max_hold_bars:
                    exit_reason = "time"
                elif signal.iloc[index - 1] <= 0:
                    exit_reason = "signal"

                if exit_reason is not None:
                    exit_fill, exit_cost = _market_fill(
                        raw_exit,
                        side="sell",
                        quantity=quantity,
                        assumptions=self._assumptions,
                    )
                    proceeds = quantity * exit_fill - self._assumptions.commission_per_order
                    cash += proceeds
                    pnl = proceeds - quantity * entry_price - entry_cost
                    trades.append(
                        TradeRecord(
                            account_id=profile.account_id,
                            entry_time=entry_time,
                            exit_time=timestamp.isoformat(),
                            entry_price=entry_price,
                            exit_price=exit_fill,
                            quantity=quantity,
                            pnl=pnl,
                            return_pct=pnl / max(quantity * entry_price + entry_cost, 1e-9),
                            bars_held=bars_held,
                            exit_reason=exit_reason,
                            execution_cost=entry_cost + exit_cost,
                        )
                    )
                    quantity = 0.0
                    entry_price = 0.0
                    entry_index = -1
                    stop_price = 0.0
                    take_profit = 0.0
                    entry_cost = 0.0
                    equity = cash

            can_enter = quantity == 0 and index > evaluation_start and signal.iloc[index - 1] > 0
            if can_enter:
                daily_loss = max(day_start_equity - equity, 0.0)
                if daily_loss >= day_start_equity * profile.max_daily_loss_fraction:
                    rejected_entries += 1
                else:
                    atr_value = float(atr.iloc[index - 1])
                    entry = _entry_fill(frame, index, profile, atr_value, self._assumptions)
                    if entry is None or not math.isfinite(atr_value) or atr_value <= 0:
                        rejected_entries += 1
                    else:
                        entry_fill, unit_cost = entry
                        stop_distance = atr_value * profile.stop_atr
                        risk_budget = equity * profile.risk_per_trade
                        requested = min(
                            risk_budget / stop_distance,
                            (equity * profile.max_position_fraction) / entry_fill,
                            max(cash - self._assumptions.commission_per_order, 0.0) / entry_fill,
                        )
                        if requested * entry_fill < 25.0:
                            rejected_entries += 1
                        else:
                            total_cost = requested * entry_fill + self._assumptions.commission_per_order
                            if total_cost > cash:
                                requested = max(
                                    (cash - self._assumptions.commission_per_order) / entry_fill,
                                    0.0,
                                )
                                total_cost = requested * entry_fill + self._assumptions.commission_per_order
                            if requested <= 0:
                                rejected_entries += 1
                            else:
                                cash -= total_cost
                                quantity = requested
                                entry_price = entry_fill
                                # unit_cost excludes the fixed per-order fee;
                                # record that fee exactly once for this entry.
                                entry_cost = (
                                    unit_cost * requested
                                    + self._assumptions.commission_per_order
                                )
                                entry_index = index
                                entry_time = timestamp.isoformat()
                                stop_price = max(entry_fill - stop_distance, 0.01)
                                take_profit = entry_fill + stop_distance * profile.reward_to_risk

            equity_curve.append(cash + quantity * mark_price)

        if quantity > 0:
            row = frame.iloc[-1]
            timestamp = pd.Timestamp(row["timestamp"])
            exit_fill, exit_cost = _market_fill(
                float(row["close"]),
                side="sell",
                quantity=quantity,
                assumptions=self._assumptions,
            )
            proceeds = quantity * exit_fill - self._assumptions.commission_per_order
            cash += proceeds
            pnl = proceeds - quantity * entry_price - entry_cost
            trades.append(
                TradeRecord(
                    account_id=profile.account_id,
                    entry_time=entry_time,
                    exit_time=timestamp.isoformat(),
                    entry_price=entry_price,
                    exit_price=exit_fill,
                    quantity=quantity,
                    pnl=pnl,
                    return_pct=pnl / max(quantity * entry_price + entry_cost, 1e-9),
                    bars_held=max(len(frame) - 1 - entry_index, 0),
                    exit_reason="end_of_test",
                    execution_cost=entry_cost + exit_cost,
                )
            )
            if equity_curve:
                equity_curve[-1] = cash

        return _build_report(
            profile,
            ending_equity=cash,
            equity_curve=equity_curve,
            trades=trades,
            rejected_entries=rejected_entries,
        )


def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"dataset missing columns: {sorted(missing)}")
    frame = data.loc[:, required].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().sort_values("timestamp").drop_duplicates("timestamp")
    frame = frame[(frame["close"] > 0) & (frame["high"] >= frame["low"])]
    if len(frame) < 80:
        raise ValueError("dataset requires at least 80 valid rows")
    return frame.reset_index(drop=True)


def _build_signal(frame: pd.DataFrame, profile: StrategyProfile) -> pd.Series:
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    params = profile.params

    if profile.strategy == "ema_momentum":
        fast = max(int(params.get("fast", 8)), 2)
        slow = max(int(params.get("slow", 21)), fast + 1)
        fast_ema = close.ewm(span=fast, adjust=False).mean()
        slow_ema = close.ewm(span=slow, adjust=False).mean()
        return ((fast_ema > slow_ema) & (close > fast_ema)).astype(int)

    if profile.strategy == "volume_breakout":
        lookback = max(int(params.get("lookback", 20)), 5)
        volume_multiple = max(float(params.get("volume_multiple", 1.5)), 1.0)
        prior_high = frame["high"].rolling(lookback).max().shift(1)
        average_volume = volume.rolling(lookback).mean().shift(1)
        return ((close > prior_high) & (volume >= average_volume * volume_multiple)).astype(int)

    if profile.strategy == "mean_reversion":
        lookback = max(int(params.get("lookback", 20)), 5)
        z_entry = max(float(params.get("z_entry", 1.5)), 0.5)
        mean = close.rolling(lookback).mean()
        std = close.rolling(lookback).std().replace(0, np.nan)
        z_score = (close - mean) / std
        return (z_score <= -z_entry).astype(int)

    lookback = max(int(params.get("lookback", 30)), 8)
    pullback_ema = max(int(params.get("pullback_ema", 8)), 3)
    trend = close > close.rolling(lookback).mean()
    ema = close.ewm(span=pullback_ema, adjust=False).mean()
    return (trend & (frame["low"] <= ema) & (close >= ema)).astype(int)


def _atr(frame: pd.DataFrame, period: int) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def _entry_fill(
    frame: pd.DataFrame,
    index: int,
    profile: StrategyProfile,
    atr_value: float,
    assumptions: ExecutionAssumptions,
) -> tuple[float, float] | None:
    row = frame.iloc[index]
    previous = frame.iloc[index - 1]

    def market_entry(raw_price: float) -> tuple[float, float]:
        fill, total_cost = _market_fill(raw_price, "buy", 1.0, assumptions)
        # The caller multiplies the unit cost by quantity and separately
        # records the fixed commission once per order.
        return fill, max(total_cost - assumptions.commission_per_order, 0.0)

    if profile.execution_policy == "next_open_market":
        return market_entry(float(row["open"]))

    if profile.execution_policy == "pullback_limit":
        discount = max(float(profile.params.get("limit_atr", 0.25)), 0.0)
        limit_price = float(previous["close"]) - atr_value * discount
        if float(row["low"]) > limit_price:
            return None
        # A touched limit is still exposed to adverse spread/slippage in this
        # conservative paper model.
        return market_entry(limit_price)

    buffer_atr = max(float(profile.params.get("breakout_atr", 0.10)), 0.0)
    trigger = float(previous["high"]) + atr_value * buffer_atr
    if float(row["high"]) < trigger:
        return None
    raw_fill = max(float(row["open"]), trigger)
    return market_entry(raw_fill)


def _market_fill(
    raw_price: float,
    side: str,
    quantity: float,
    assumptions: ExecutionAssumptions,
) -> tuple[float, float]:
    if (
        not math.isfinite(float(raw_price))
        or raw_price <= 0
        or not math.isfinite(float(quantity))
        or quantity < 0
    ):
        raise ValueError("market fills require finite positive prices and quantity")
    if side not in {"buy", "sell"}:
        raise ValueError(f"unsupported fill side: {side}")
    total_bps = assumptions.spread_bps / 2.0 + assumptions.slippage_bps
    multiplier = (
        1.0 + total_bps / 10_000.0
        if side == "buy"
        else 1.0 - total_bps / 10_000.0
    )
    fill = max(raw_price * multiplier, 0.01)
    execution_cost = abs(fill - raw_price) * quantity + assumptions.commission_per_order
    return fill, execution_cost


def _build_report(
    profile: StrategyProfile,
    *,
    ending_equity: float,
    equity_curve: Sequence[float],
    trades: Sequence[TradeRecord],
    rejected_entries: int,
) -> AccountReport:
    curve = pd.Series(equity_curve or [profile.starting_cash], dtype=float)
    returns = curve.pct_change().fillna(0.0)
    total_return = ending_equity / profile.starting_cash - 1.0
    drawdown = abs(float(max_drawdown(curve)))
    sharpe = float(sharpe_ratio(returns)) if len(returns) > 2 else 0.0
    pnls = [trade.pnl for trade in trades]
    gains = sum(value for value in pnls if value > 0)
    losses = -sum(value for value in pnls if value < 0)
    profit_factor = gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0)
    wins = sum(1 for value in pnls if value > 0)
    win_rate = wins / len(pnls) if pnls else 0.0
    expectancy = sum(pnls) / len(pnls) if pnls else 0.0
    average_cost = (
        sum(trade.execution_cost for trade in trades) / len(trades) if trades else 0.0
    )
    # Raw return cannot dominate the leaderboard. Fragile, high-drawdown and
    # low-sample candidates are penalised heavily.
    sample_penalty = 0.20 if len(trades) < 5 else 0.0
    score = total_return + 0.05 * sharpe - 1.5 * drawdown - sample_penalty
    if not math.isfinite(score):
        score = -1_000.0
    return AccountReport(
        account_id=profile.account_id,
        strategy=profile.strategy,
        execution_policy=profile.execution_policy,
        starting_cash=profile.starting_cash,
        ending_equity=ending_equity,
        total_return=total_return,
        max_drawdown=drawdown,
        sharpe=sharpe,
        profit_factor=profit_factor,
        win_rate=win_rate,
        expectancy=expectancy,
        trade_count=len(trades),
        rejected_entries=rejected_entries,
        average_execution_cost=average_cost,
        score=score,
        trades=tuple(trades),
    )


__all__ = [
    "AccountReport",
    "EXECUTION_POLICIES",
    "ExecutionAssumptions",
    "PaperStrategyTournament",
    "STRATEGY_FAMILIES",
    "StrategyProfile",
    "TournamentReport",
    "TradeRecord",
]
