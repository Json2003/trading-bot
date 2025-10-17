"""Produce a compact markdown summary for portfolio backtests."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


NUMERIC_KEYS = ("equity", "balance", "value", "net_value", "netValue")


def load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        for key in NUMERIC_KEYS:
            if key in value and isinstance(value[key], (int, float)):
                return float(value[key])
    return None


def normalize_equity(equity: object) -> List[float]:
    if isinstance(equity, (int, float)):
        iterable: Iterable[object] = [equity]
    elif isinstance(equity, Sequence) and not isinstance(equity, (str, bytes)):
        iterable = equity
    else:
        iterable = []

    normalized: List[float] = []
    for point in iterable:
        coerced = _coerce_float(point)
        if coerced is not None:
            normalized.append(coerced)
    return normalized


def equity_to_returns(equity: Sequence[float]) -> List[float]:
    return [
        (equity[i] / equity[i - 1] - 1.0)
        for i in range(1, len(equity))
        if equity[i - 1] > 0
    ]


def _normalize_ratio(value: object, *, assume_percent: bool = False) -> float | None:
    coerced = _coerce_float(value)
    if coerced is None:
        return None
    threshold = 1.0 if assume_percent else 2.0
    if abs(coerced) > threshold:
        return coerced / 100.0
    return coerced


def max_drawdown(equity: Sequence[float]) -> float:
    peak, drawdown = 0.0, 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            drawdown = max(drawdown, 1 - value / peak)
    return drawdown


def sharpe(returns: Sequence[float], rf: float = 0.0) -> float:
    if not returns:
        return 0.0
    mu = statistics.mean(returns) - rf
    sd = statistics.pstdev(returns) or 1e-12
    return mu / sd


def sortino(returns: Sequence[float], rf: float = 0.0) -> float:
    if not returns:
        return 0.0
    downside = [r for r in returns if r < 0]
    mu = statistics.mean(returns) - rf
    if not downside:
        return float("inf") if mu > 0 else 0.0
    dd = statistics.pstdev(downside) or 1e-12
    return mu / dd


def profit_factor(trades: Sequence[Mapping[str, object]]) -> float:
    pnls = [_coerce_float(t.get("pnl", 0)) or 0.0 for t in trades]
    gains = sum(p for p in pnls if p > 0)
    losses = sum(-p for p in pnls if p < 0)
    return gains / losses if losses > 0 else float("inf")


@dataclass
class SummaryRow:
    file: str
    period: str
    total_pnl: float | None
    ann_pnl: float | None
    mdd: float | None
    sharpe: float
    sortino: float
    win_rate: float | None
    profit_factor: float | None
    trades: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate backtest JSONs and emit a markdown summary table.",
    )
    parser.add_argument(
        "patterns",
        nargs="*",
        default=["backtest*.json", "report_*.json"],
        help="Glob patterns locating backtest result JSON files.",
    )
    parser.add_argument(
        "--annualization",
        type=int,
        default=252,
        help="Periods per year used for annualized return calculations.",
    )
    parser.add_argument(
        "--risk-free",
        type=float,
        default=0.0,
        help="Annual risk-free rate assumed when computing Sharpe/Sortino.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Decimal precision used for percentage formatting.",
    )
    return parser.parse_args()


def collect_files(patterns: Sequence[str]) -> List[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path().glob(pattern))
    return sorted(set(files))


def load_trades(container: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    results = container.get("results") if isinstance(container.get("results"), Mapping) else {}
    candidates = [
        container.get("trades"),
        results.get("trades") if isinstance(results, Mapping) else None,
        container.get("trade_list"),
        results.get("trade_list") if isinstance(results, Mapping) else None,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, Mapping) and "trades" in candidate:
            candidate = candidate["trades"]
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            return candidate
    return []


def summarize_file(path: Path, *, annualization: int, rf: float) -> SummaryRow | None:
    try:
        data = load_json(path)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] failed to load {path}: {exc}")
        return None

    results_raw = data.get("results")
    results = results_raw if isinstance(results_raw, Mapping) else {}

    equity_raw = None
    for candidate in (
        results.get("equity_curve"),
        data.get("equity_curve"),
        data.get("equity_series"),
        data.get("equity"),
    ):
        if candidate:
            equity_raw = candidate
            break
    if equity_raw is None:
        equity_raw = []
    equity = normalize_equity(equity_raw)
    returns = equity_to_returns(equity) if len(equity) >= 2 else []
    rf_period = (1 + rf) ** (1 / annualization) - 1 if rf else 0.0

    total_pnl = None
    ann_pnl = None
    mdd_value = None
    sharpe_value = sharpe(returns, rf_period) if returns else float("nan")
    sortino_value = sortino(returns, rf_period) if returns else float("nan")

    if len(equity) >= 2 and equity[0]:
        total_pnl = equity[-1] / equity[0] - 1.0
        ann_pnl = annualized_return(returns, annualization)
        mdd_value = max_drawdown(equity)

    metrics_raw = results.get("performance_metrics") if isinstance(results, Mapping) else None
    metrics = metrics_raw if isinstance(metrics_raw, Mapping) else {}
    if total_pnl is None or math.isnan(total_pnl):
        total_pnl = _normalize_ratio(metrics.get("total_return_pct"), assume_percent=True)
    if total_pnl is None:
        total_pnl = _normalize_ratio(metrics.get("total_return"))

    if ann_pnl is None or math.isnan(ann_pnl):
        ann_pnl = _normalize_ratio(metrics.get("annualized_return_pct"), assume_percent=True)
    if ann_pnl is None:
        ann_pnl = _normalize_ratio(metrics.get("annualized_return"))

    if mdd_value is None or math.isnan(mdd_value):
        mdd_value = _normalize_ratio(metrics.get("max_drawdown_pct"), assume_percent=True)
    if mdd_value is None:
        mdd_value = _normalize_ratio(metrics.get("max_drawdown"))

    if math.isnan(sharpe_value):
        sharpe_value = _coerce_float(metrics.get("sharpe_ratio")) or float("nan")
    if math.isnan(sortino_value):
        sortino_value = _coerce_float(metrics.get("sortino_ratio")) or float("nan")

    trades = list(load_trades(data))
    win_rate = (
        sum(1 for trade in trades if (_coerce_float(trade.get("pnl", 0)) or 0.0) > 0) / len(trades)
        if trades
        else None
    )
    if win_rate is None:
        win_rate = _normalize_ratio(metrics.get("win_rate"))
    if win_rate is None:
        win_rate = _normalize_ratio(metrics.get("win_rate_pct"), assume_percent=True)

    profit = profit_factor(trades) if trades else _coerce_float(metrics.get("profit_factor"))
    trade_count = len(trades)
    if not trade_count:
        trade_metric = _coerce_float(metrics.get("total_trades"))
        if trade_metric:
            trade_count = int(trade_metric)

    start = results.get("start") if isinstance(results, Mapping) else data.get("start")
    end = results.get("end") if isinstance(results, Mapping) else data.get("end")
    period = f"{start}→{end}" if start and end else ""

    return SummaryRow(
        file=path.name,
        period=period,
        total_pnl=total_pnl,
        ann_pnl=ann_pnl,
        mdd=mdd_value,
        sharpe=sharpe_value,
        sortino=sortino_value,
        win_rate=win_rate,
        profit_factor=profit,
        trades=trade_count,
    )


def annualized_return(returns: Sequence[float], periods_per_year: int) -> float:
    if not returns:
        return float("nan")
    growth = math.prod(1 + r for r in returns)
    if growth <= 0:
        return float("nan")
    periods = len(returns)
    return growth ** (periods_per_year / periods) - 1


def pct(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if math.isinf(value):
            return "∞%" if value > 0 else "-∞%"
    return f"{100 * value:.{precision}f}%"


def ratio(value: float, precision: int = 2) -> str:
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    return f"{value:.{precision}f}"


def format_markdown(rows: Sequence[SummaryRow], precision: int) -> str:
    lines = ["# Backtest Summary", ""]
    header = (
        "| File | Period | PnL | Ann | MaxDD | Sharpe | Sortino | Win-rate | PF | Trades |"
    )
    divider = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines.extend([header, divider])
    for row in rows:
        profit_factor_value = row.profit_factor
        if profit_factor_value is None:
            pf_str = "n/a"
        elif isinstance(profit_factor_value, float):
            pf_str = ratio(profit_factor_value, precision)
        else:
            pf_str = str(profit_factor_value)
        lines.append(
            "| "
            + " | ".join(
                [
                    row.file,
                    row.period or "",
                    pct(row.total_pnl, precision),
                    pct(row.ann_pnl, precision),
                    pct(row.mdd, precision),
                    ratio(row.sharpe, precision),
                    ratio(row.sortino, precision),
                    pct(row.win_rate, precision) if row.win_rate is not None else "n/a",
                    pf_str,
                    str(row.trades),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    files = collect_files(args.patterns)
    if not files:
        raise SystemExit("No backtest JSON files found")

    rows: list[SummaryRow] = []
    for path in files:
        row = summarize_file(path, annualization=args.annualization, rf=args.risk_free)
        if row:
            rows.append(row)

    if not rows:
        raise SystemExit("No valid backtest reports were parsed")

    print(format_markdown(rows, args.precision))


if __name__ == "__main__":
    main()

