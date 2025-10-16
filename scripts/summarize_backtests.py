import argparse
import csv
import json
import math
import statistics
from collections.abc import Sequence as SequenceABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


NUMERIC_KEYS = ("equity", "balance", "value", "net_value")


def _coerce_float(value):
    """Best-effort conversion of a value that may represent a numeric amount."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in NUMERIC_KEYS:
            if key in value and isinstance(value[key], (int, float)):
                return float(value[key])
    return None


def normalize_equity(equity: Sequence) -> List[float]:
    """Return a list of floats from a variety of equity curve formats."""
    if isinstance(equity, (int, float)):
        iterable = [equity]
    elif isinstance(equity, dict):
        iterable = [equity]
    elif isinstance(equity, (str, bytes)) or not isinstance(equity, SequenceABC):
        iterable = []
    else:
        iterable = equity

    normalized = []
    for point in iterable:
        coerced = _coerce_float(point)
        if coerced is not None:
            normalized.append(coerced)
    return normalized


def equity_to_returns(equity: Sequence[float]) -> List[float]:
    """Convert equity curve to simple period returns."""
    return [(equity[i] / equity[i - 1] - 1.0) for i in range(1, len(equity)) if equity[i - 1] > 0]


def _normalize_ratio(value, assume_percent: bool = False) -> Optional[float]:
    """Return a ratio in decimal form when possible.

    Some backtest exports provide metrics either as decimals (0.12 == 12%)
    or percentages (12.0 == 12%).  To make the summary resilient, we treat
    anything whose absolute value is greater than 2 as a percentage and scale
    it back to decimal form.  When ``assume_percent`` is true we always apply
    this scaling when the value is greater than ``1``.
    """

    coerced = _coerce_float(value)
    if coerced is None:
        return None
    threshold = 1.0 if assume_percent else 2.0
    if abs(coerced) > threshold:
        return coerced / 100.0
    return coerced


def max_drawdown(equity: Sequence[float]) -> float:
    peak, mdd = 0.0, 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            mdd = max(mdd, 1 - value / peak)
    return mdd


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


def profit_factor(trades: Sequence[dict]) -> float:
    def pnl_value(trade: dict) -> float:
        pnl = trade.get("pnl", 0)
        coerced = _coerce_float(pnl)
        return coerced if coerced is not None else 0.0

    pnls = [pnl_value(t) for t in trades]
    gains = sum(p for p in pnls if p > 0)
    losses = sum(-p for p in pnls if p < 0)
    return (gains / losses) if losses > 0 else float("inf")


def annualized_return(returns: Sequence[float], periods_per_year: int) -> float:
    if not returns:
        return 0.0
    try:
        growth = math.prod(1 + r for r in returns)
    except ValueError:
        return float("nan")
    if growth <= 0:
        return float("nan")
    periods = len(returns)
    return growth ** (periods_per_year / periods) - 1


def pct(value: float, precision: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if math.isinf(value):
            return "∞%" if value > 0 else "-∞%"
    return f"{100 * value:.{precision}f}%"


def collect_files(patterns: Iterable[str]) -> List[Path]:
    files = []
    for pattern in patterns:
        files.extend(Path().glob(pattern))
    return sorted(set(files))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize backtest JSON files by computing key performance metrics. "
            "By default the script searches for backtest*.json and report_*.json in the "
            "current directory."
        )
    )
    parser.add_argument(
        "patterns",
        nargs="*",
        default=["backtest*.json", "report_*.json"],
        help="Glob patterns used to locate JSON reports.",
    )
    parser.add_argument(
        "--annualization",
        type=int,
        default=252,
        help="Periods per year used when annualizing returns (default: 252).",
    )
    parser.add_argument(
        "--risk-free",
        type=float,
        default=0.0,
        help="Annual risk-free rate used for Sharpe/Sortino calculations (default: 0).",
    )
    parser.add_argument(
        "--sort",
        choices=[
            "total_pnl",
            "ann_pnl",
            "mdd",
            "sharpe",
            "sortino",
            "win_rate",
            "profit_factor",
            "trades",
        ],
        help="Metric used to sort the summary output.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort metrics in descending order (default: ascending).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write the summary as a CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write a markdown summary.",
    )
    parser.add_argument(
        "--markdown-style",
        choices=["sections", "table"],
        default="sections",
        help=(
            "Controls the markdown layout when --output is provided. "
            "'sections' renders a heading per file (default) while 'table' "
            "emits a compact summary table."
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Decimal precision for percentage values (default: 2).",
    )
    return parser.parse_args()


def summarize_backtests(args: argparse.Namespace) -> List[dict]:
    files = collect_files(args.patterns)
    if not files:
        print("No backtest JSONs found (looking for {})".format(", ".join(args.patterns)))
        raise SystemExit(1)

    rows = []
    rf_daily = (1 + args.risk_free) ** (1 / args.annualization) - 1 if args.risk_free else 0.0

    for path in files:
        try:
            data = load_json(path)
            results = data.get("results", {})
            equity_raw = (
                results.get("equity_curve")
                or data.get("equity_curve")
                or data.get("equity")
                or data.get("equity_series")
                or []
            )
            equity = normalize_equity(equity_raw)
            trades_raw_candidates = [
                results.get("trades"),
                data.get("trades"),
                results.get("trade_list"),
                data.get("trade_list"),
            ]
            trades = []
            for candidate in trades_raw_candidates:
                if candidate is None:
                    continue
                if isinstance(candidate, dict) and "trades" in candidate:
                    candidate = candidate["trades"]
                if isinstance(candidate, (str, bytes)):
                    continue
                if isinstance(candidate, SequenceABC):
                    trades = list(candidate)
                    break
            start = results.get("start") or data.get("start")
            end = results.get("end") or data.get("end")

            metrics = (
                results.get("performance_metrics")
                or data.get("performance_metrics")
                or {}
            )

            if len(equity) >= 2 and equity[0]:
                rets = equity_to_returns(equity)
                total_pnl = equity[-1] / equity[0] - 1.0
                ann_pnl = annualized_return(rets, args.annualization)
                mdd = max_drawdown(equity)
                sh = sharpe(rets, rf_daily)
                so = sortino(rets, rf_daily)
            else:
                rets = []
                total_pnl = ann_pnl = mdd = sh = so = float("nan")

            total_pnl = (
                total_pnl
                if not math.isnan(total_pnl)
                else _normalize_ratio(metrics.get("total_return_pct"), assume_percent=True)
            )
            if total_pnl is None:
                total_pnl = _normalize_ratio(metrics.get("total_return"))

            ann_pnl = (
                ann_pnl
                if not math.isnan(ann_pnl)
                else _normalize_ratio(metrics.get("annualized_return_pct"), assume_percent=True)
            )
            if ann_pnl is None:
                ann_pnl = _normalize_ratio(metrics.get("annualized_return"))

            if math.isnan(mdd):
                mdd = _normalize_ratio(metrics.get("max_drawdown_pct"), assume_percent=True)
            if mdd is None:
                mdd = _normalize_ratio(metrics.get("max_drawdown"))

            if math.isnan(sh):
                sh = _coerce_float(metrics.get("sharpe_ratio"))
            if math.isnan(so):
                so = _coerce_float(metrics.get("sortino_ratio"))
            if sh is None:
                sh = float("nan")
            if so is None:
                so = float("nan")

            def trade_pnl(trade: dict) -> float:
                pnl_value = _coerce_float(trade.get("pnl", 0))
                return pnl_value if pnl_value is not None else 0.0

            wr = (
                sum(1 for t in trades if trade_pnl(t) > 0) / len(trades)
                if trades
                else None
            )
            if wr is None:
                wr = _normalize_ratio(metrics.get("win_rate"))
            if wr is None:
                wr = _normalize_ratio(metrics.get("win_rate_pct"), assume_percent=True)

            pf = profit_factor(trades) if trades else None
            if pf is None:
                pf = _coerce_float(metrics.get("profit_factor"))

            trades_count = len(trades)
            if not trades_count:
                trades_metric = _coerce_float(metrics.get("total_trades"))
                if trades_metric is not None:
                    trades_count = int(trades_metric)
            if not trades_count:
                trades_metric = data.get("trades")
                if isinstance(trades_metric, (int, float)):
                    trades_count = int(trades_metric)

            rows.append(
                {
                    "file": str(path),
                    "period": f"{start} → {end}" if start and end else "",
                    "total_pnl": total_pnl,
                    "ann_pnl": ann_pnl,
                    "mdd": mdd,
                    "sharpe": sh,
                    "sortino": so,
                    "win_rate": wr,
                    "profit_factor": pf,
                    "trades": trades_count,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[ERROR] reading {path}: {exc}")

    if args.sort:

        def sort_key(row: dict):
            value = row.get(args.sort)
            if value is None:
                return (1, 0)
            if isinstance(value, float) and math.isnan(value):
                return (1, 0)
            return (0, value)

        rows.sort(key=sort_key, reverse=args.descending)

    return rows


def format_summary_text(rows: Sequence[dict], precision: int) -> str:
    lines = ["=== Backtest Summary ==="]
    for row in rows:
        period = f" {row['period']}" if row["period"] else ""
        lines.append(f"- {row['file']}{period}")
        lines.append(
            f"  PnL: {pct(row['total_pnl'], precision)}  | Annualized: {pct(row['ann_pnl'], precision)}"
        )
        lines.append(
            f"  MaxDD: {pct(row['mdd'], precision)} | Sharpe: {row['sharpe']:.2f} | Sortino: {row['sortino']:.2f}"
        )
        pf = ""
        if isinstance(row["profit_factor"], float):
            if math.isinf(row["profit_factor"]):
                pf = "| ProfitFactor: ∞"
            elif not math.isnan(row["profit_factor"]):
                pf = f"| ProfitFactor: {row['profit_factor']:.2f}"
        win_rate = pct(row["win_rate"], precision) if isinstance(row["win_rate"], float) else "n/a"
        lines.append(f"  Win-rate: {win_rate} {pf}")
        lines.append(f"  Trades: {row['trades']}")
        lines.append("")
    return "\n".join(lines)


def format_summary_markdown(rows: Sequence[dict], precision: int) -> str:
    lines = ["# Backtest Summary", ""]
    for row in rows:
        period = f" ({row['period']})" if row["period"] else ""
        lines.append(f"## {row['file']}{period}")
        lines.append(
            " | ".join(
                [
                    f"PnL: {pct(row['total_pnl'], precision)}",
                    f"Annualized: {pct(row['ann_pnl'], precision)}",
                    f"MaxDD: {pct(row['mdd'], precision)}",
                    f"Sharpe: {row['sharpe']:.2f}",
                    f"Sortino: {row['sortino']:.2f}",
                ]
            )
        )
        win_rate = (
            pct(row["win_rate"], precision)
            if isinstance(row["win_rate"], float)
            else "n/a"
        )
        if isinstance(row["profit_factor"], float):
            if math.isinf(row["profit_factor"]):
                profit_factor = "∞"
            elif math.isnan(row["profit_factor"]):
                profit_factor = "n/a"
            else:
                profit_factor = f"{row['profit_factor']:.2f}"
        else:
            profit_factor = "n/a"
        lines.append(f"Win-rate: {win_rate} | ProfitFactor: {profit_factor}")
        lines.append(f"Trades: {row['trades']}")
        lines.append("")
    return "\n".join(lines)


def _format_profit_factor(value: Optional[float]) -> str:
    if not isinstance(value, float):
        return "—"
    if math.isinf(value):
        return "∞"
    if math.isnan(value):
        return "—"
    return f"{value:.2f}"


def _format_ratio(value: Optional[float]) -> str:
    if not isinstance(value, float):
        return "n/a"
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    return f"{value:.2f}"


def format_summary_markdown_table(rows: Sequence[dict], precision: int) -> str:
    timestamp = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    lines = ["# Backtest Summary", "", f"_Last updated:_ {timestamp}", ""]
    lines.append(
        "| File | Period | PnL | Ann | MaxDD | Sharpe | Sortino | Win-rate | PF | Trades |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        period = row["period"].replace(" → ", "→") if row["period"] else ""
        total = pct(row["total_pnl"], precision)
        annualized = pct(row["ann_pnl"], precision)
        drawdown = pct(row["mdd"], precision)
        sharpe_ratio = _format_ratio(row["sharpe"])
        sortino_ratio = _format_ratio(row["sortino"])
        win_rate = (
            pct(row["win_rate"], precision)
            if isinstance(row["win_rate"], float)
            else "—"
        )
        profit_factor = _format_profit_factor(row["profit_factor"])
        lines.append(
            " | ".join(
                [
                    f"`{row['file']}`",
                    period,
                    total,
                    annualized,
                    drawdown,
                    sharpe_ratio,
                    sortino_ratio,
                    win_rate,
                    profit_factor,
                    str(row["trades"]),
                ]
            )
        )
    return "\n".join(lines)


def render_summary(rows: Sequence[dict], precision: int) -> str:
    summary = format_summary_text(rows, precision)
    print(summary)
    return summary


def write_csv(rows: Sequence[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "period",
        "total_pnl",
        "ann_pnl",
        "mdd",
        "sharpe",
        "sortino",
        "win_rate",
        "profit_factor",
        "trades",
    ]
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Summary written to {csv_path}")


def main() -> None:
    args = parse_args()
    rows = summarize_backtests(args)
    render_summary(rows, args.precision)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.markdown_style == "table":
            markdown = format_summary_markdown_table(rows, args.precision)
        else:
            markdown = format_summary_markdown(rows, args.precision)
        if not markdown.endswith("\n"):
            markdown += "\n"
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Markdown summary written to {args.output}")
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
