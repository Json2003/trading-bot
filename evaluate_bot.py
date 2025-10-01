"""Command-line tool to evaluate live trading results against a benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TRADE_DATE_CANDIDATES: Sequence[str] = (
    "exit_ts",
    "exit_time",
    "exit_timestamp",
    "close_time",
    "timestamp",
    "ts",
    "datetime",
    "date",
)

TRADE_PNL_CANDIDATES: Sequence[str] = (
    "pnl",
    "net_pnl",
    "profit",
    "pl",
    "pnl_usd",
    "net_profit",
)

TRADE_RETURN_CANDIDATES: Sequence[str] = (
    "return",
    "returns",
    "daily_return",
    "strategy_return",
    "roi",
    "pnl_pct",
    "return_pct",
    "pct_return",
)

BENCHMARK_DATE_CANDIDATES: Sequence[str] = (
    "date",
    "timestamp",
    "ts",
    "datetime",
)

BENCHMARK_RETURN_CANDIDATES: Sequence[str] = (
    "return",
    "returns",
    "daily_return",
    "benchmark_return",
    "pct_change",
    "pct_return",
    "return_pct",
)

BENCHMARK_PRICE_CANDIDATES: Sequence[str] = (
    "close",
    "price",
    "value",
    "nav",
    "benchmark",
)


@dataclass
class TradeRecord:
    timestamp: datetime
    trade_date: date
    pnl: float
    return_: float
    equity_after_trade: float
    raw: Dict[str, str]


@dataclass
class DailyRecord:
    date: date
    strategy_equity: float
    strategy_return: float
    strategy_pnl: float
    benchmark_return: float
    excess_return: float


@dataclass
class EvaluationResults:
    summary: Dict[str, object]
    daily: List[DailyRecord]
    trades: List[TradeRecord]


class EvaluationError(RuntimeError):
    """Raised when the evaluation cannot be completed."""


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise EvaluationError(f"File not found: {path}")
    with path.open("r", newline="", encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise EvaluationError(f"No rows found in {path}")
    return rows


def _find_first_present(columns: Iterable[str], options: Sequence[str]) -> Optional[str]:
    for candidate in options:
        if candidate in columns:
            return candidate
    return None


def _parse_datetime(value: object) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    patterns = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y",
    ]
    for fmt in patterns:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_return(value: object, column_name: str) -> Optional[float]:
    val = _parse_float(value)
    if val is None:
        return None
    name = column_name.lower()
    if "pct" in name:
        return val / 100.0
    if abs(val) > 1.5:
        return val / 100.0
    return val


def load_trade_log(path: Path, initial_capital: float) -> List[TradeRecord]:
    rows = _load_csv_rows(path)
    date_col = _find_first_present(rows[0].keys(), TRADE_DATE_CANDIDATES)
    if not date_col:
        raise EvaluationError("Could not locate a timestamp column in the trade log.")
    pnl_col = _find_first_present(rows[0].keys(), TRADE_PNL_CANDIDATES)
    ret_col = _find_first_present(rows[0].keys(), TRADE_RETURN_CANDIDATES)

    parsed_rows: List[Tuple[datetime, Dict[str, str]]] = []
    for row in rows:
        ts = _parse_datetime(row.get(date_col))
        if ts is None:
            raise EvaluationError(f"Failed to parse timestamp '{row.get(date_col)}' in trade log.")
        parsed_rows.append((ts, row))
    parsed_rows.sort(key=lambda item: item[0])

    trades: List[TradeRecord] = []
    equity = float(initial_capital)
    for ts, row in parsed_rows:
        pnl_value = _parse_float(row.get(pnl_col)) if pnl_col else None
        ret_value = _parse_return(row.get(ret_col), ret_col) if ret_col else None
        if pnl_value is None and ret_value is None:
            raise EvaluationError("Each trade must include either a profit/loss or return column.")
        if ret_value is None:
            ret_value = 0.0 if equity == 0 else pnl_value / equity
        if pnl_value is None:
            pnl_value = equity * ret_value
        equity += pnl_value
        trades.append(
            TradeRecord(
                timestamp=ts,
                trade_date=ts.date(),
                pnl=pnl_value,
                return_=ret_value,
                equity_after_trade=equity,
                raw=row,
            )
        )
    return trades


def load_benchmark(path: Path) -> Dict[date, float]:
    rows = _load_csv_rows(path)
    date_col = _find_first_present(rows[0].keys(), BENCHMARK_DATE_CANDIDATES)
    if not date_col:
        raise EvaluationError("Could not locate a date column in the benchmark data.")
    return_col = _find_first_present(rows[0].keys(), BENCHMARK_RETURN_CANDIDATES)
    price_col = _find_first_present(rows[0].keys(), BENCHMARK_PRICE_CANDIDATES)
    if not return_col and not price_col:
        raise EvaluationError("Benchmark data must include either return or price information.")

    parsed_rows: List[Tuple[date, Dict[str, str]]] = []
    for row in rows:
        ts = _parse_datetime(row.get(date_col))
        if ts is None:
            raise EvaluationError(f"Failed to parse date '{row.get(date_col)}' in benchmark data.")
        parsed_rows.append((ts.date(), row))
    parsed_rows.sort(key=lambda item: item[0])

    returns: Dict[date, float] = {}
    prev_price: Optional[float] = None
    for dt_value, row in parsed_rows:
        if return_col:
            returns[dt_value] = _parse_return(row.get(return_col), return_col) or 0.0
        else:
            price = _parse_float(row.get(price_col)) if price_col else None
            if price is None:
                raise EvaluationError(f"Benchmark price missing for {dt_value}")
            if prev_price is None or prev_price == 0:
                returns[dt_value] = 0.0
            else:
                returns[dt_value] = (price - prev_price) / prev_price
            prev_price = price
    return returns


def _group_trades_by_day(trades: List[TradeRecord]) -> Dict[date, Tuple[float, float]]:
    grouped: Dict[date, Tuple[float, float]] = {}
    daily_totals: Dict[date, float] = {}
    for trade in trades:
        daily_totals[trade.trade_date] = daily_totals.get(trade.trade_date, 0.0) + trade.pnl
        grouped[trade.trade_date] = (daily_totals[trade.trade_date], trade.equity_after_trade)
    return grouped


def build_daily_records(
    trades: List[TradeRecord], benchmark_returns: Dict[date, float], initial_capital: float
) -> List[DailyRecord]:
    if not trades:
        raise EvaluationError("Trade log does not contain any trades.")
    grouped = _group_trades_by_day(trades)
    all_dates = set(grouped.keys()) | set(benchmark_returns.keys())
    first_trade_day = trades[0].trade_date
    all_dates.add(first_trade_day - timedelta(days=1))
    sorted_dates = sorted(all_dates)

    daily_records: List[DailyRecord] = []
    equity = float(initial_capital)
    for current_date in sorted_dates:
        prev_equity = equity
        pnl = 0.0
        if current_date in grouped:
            daily_pnl, daily_equity = grouped[current_date]
            pnl = daily_equity - prev_equity
            equity = daily_equity
        strategy_return = 0.0 if prev_equity == 0 else pnl / prev_equity
        bench_return = benchmark_returns.get(current_date, 0.0)
        daily_records.append(
            DailyRecord(
                date=current_date,
                strategy_equity=equity,
                strategy_return=strategy_return,
                strategy_pnl=pnl,
                benchmark_return=bench_return,
                excess_return=strategy_return - bench_return,
            )
        )
    return daily_records


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return float("nan")
    return sum(items) / len(items)


def _std_dev(values: Iterable[float]) -> float:
    items = list(values)
    if len(items) < 2:
        return 0.0
    mean_value = _mean(items)
    variance = sum((x - mean_value) ** 2 for x in items) / len(items)
    return math.sqrt(variance)


def _max_drawdown(equity_values: Iterable[float]) -> Tuple[float, int]:
    peak = -float("inf")
    max_dd = 0.0
    current_duration = 0
    max_duration = 0
    for value in equity_values:
        if value > peak:
            peak = value
            current_duration = 0
            continue
        if peak == 0:
            continue
        drawdown = value / peak - 1.0
        if drawdown < max_dd:
            max_dd = drawdown
            max_duration = max(max_duration, current_duration + 1)
        current_duration += 1
    return max_dd, max_duration


def _annualised_stats(returns: Iterable[float], periods_per_year: int = 252) -> Tuple[float, float]:
    returns_list = [r for r in returns if not math.isnan(r)]
    if not returns_list:
        return float("nan"), float("nan")
    mean_ret = _mean(returns_list)
    std_ret = _std_dev(returns_list)
    sharpe = float("nan") if std_ret == 0 else math.sqrt(periods_per_year) * mean_ret / std_ret
    downside = [r for r in returns_list if r < 0]
    if not downside:
        sortino = float("nan")
    else:
        downside_std = _std_dev(downside)
        sortino = float("nan") if downside_std == 0 else math.sqrt(periods_per_year) * mean_ret / downside_std
    return sharpe, sortino


def _replace_nan_with_none(values: Dict[str, object]) -> Dict[str, object]:
    cleaned: Dict[str, object] = {}
    for key, value in values.items():
        if isinstance(value, float) and math.isnan(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def summarise(trades: List[TradeRecord], daily: List[DailyRecord], initial_capital: float) -> Dict[str, object]:
    total_trades = len(trades)
    wins = sum(1 for trade in trades if trade.pnl > 0)
    total_pnl = sum(trade.pnl for trade in trades)
    avg_pnl = total_pnl / total_trades if total_trades else float("nan")
    ending_equity = trades[-1].equity_after_trade if trades else initial_capital
    total_return = ending_equity / initial_capital - 1 if initial_capital != 0 else float("nan")
    sharpe, sortino = _annualised_stats([rec.strategy_return for rec in daily])
    max_dd, dd_duration = _max_drawdown([rec.strategy_equity for rec in daily])
    benchmark_total_return = math.prod(1 + rec.benchmark_return for rec in daily) - 1
    summary = {
        "total_trades": total_trades,
        "win_rate": wins / total_trades if total_trades else float("nan"),
        "total_pnl": total_pnl,
        "avg_trade_pnl": avg_pnl,
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_duration_days": dd_duration,
        "ending_equity": ending_equity,
        "benchmark_total_return": benchmark_total_return,
        "alpha": total_return - benchmark_total_return,
    }
    return _replace_nan_with_none(summary)


def _write_daily_csv(path: Path, daily: List[DailyRecord]) -> None:
    fieldnames = [
        "date",
        "strategy_equity",
        "strategy_return",
        "strategy_pnl",
        "benchmark_return",
        "excess_return",
    ]
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in daily:
            writer.writerow(
                {
                    "date": record.date.isoformat(),
                    "strategy_equity": record.strategy_equity,
                    "strategy_return": record.strategy_return,
                    "strategy_pnl": record.strategy_pnl,
                    "benchmark_return": record.benchmark_return,
                    "excess_return": record.excess_return,
                }
            )


def _write_trades_csv(path: Path, trades: List[TradeRecord]) -> None:
    all_fields = set()
    for trade in trades:
        all_fields.update(trade.raw.keys())
    additional_fields = ["timestamp", "trade_date", "pnl", "return", "equity_after_trade"]
    fieldnames = list(all_fields)
    for field in additional_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            row = dict(trade.raw)
            row.update(
                {
                    "timestamp": trade.timestamp.isoformat(),
                    "trade_date": trade.trade_date.isoformat(),
                    "pnl": trade.pnl,
                    "return": trade.return_,
                    "equity_after_trade": trade.equity_after_trade,
                }
            )
            writer.writerow(row)


def evaluate(trades_path: Path, benchmark_path: Path, out_dir: Path, initial_capital: float) -> EvaluationResults:
    trades = load_trade_log(trades_path, initial_capital)
    benchmark = load_benchmark(benchmark_path)
    daily = build_daily_records(trades, benchmark, initial_capital)
    summary = summarise(trades, daily, initial_capital)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf8") as handle:
        json.dump(summary, handle, indent=2)
    _write_daily_csv(out_dir / "daily_metrics.csv", daily)
    _write_trades_csv(out_dir / "trade_metrics.csv", trades)

    return EvaluationResults(summary=summary, daily=daily, trades=trades)


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate live trading results against a benchmark.")
    parser.add_argument("--trades", required=True, type=Path, help="Path to the trade log CSV file.")
    parser.add_argument("--benchmark", required=True, type=Path, help="Path to the benchmark CSV file.")
    parser.add_argument("--out", required=True, type=Path, help="Directory where reports will be written.")
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Starting capital used to translate returns to equity (default: 10000).",
    )
    return parser.parse_args(args=args)


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    args = parse_args(cli_args)
    try:
        results = evaluate(args.trades, args.benchmark, args.out, args.initial_capital)
    except EvaluationError as exc:
        raise SystemExit(str(exc)) from exc

    print("Evaluation summary:")
    for key, value in results.summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":  # pragma: no cover
    main()
