import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable, List, Sequence


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def equity_to_returns(equity: Sequence[float]) -> List[float]:
    """Convert equity curve to simple period returns."""
    return [
        (equity[i] / equity[i - 1] - 1.0)
        for i in range(1, len(equity))
        if equity[i - 1] > 0
    ]


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
    dd = statistics.pstdev(downside) or 1e-12
    mu = statistics.mean(returns) - rf
    return mu / dd


def profit_factor(trades: Sequence[dict]) -> float:
    gains = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
    losses = sum(-t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0)
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
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
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
        "--precision",
        type=int,
        default=2,
        help="Decimal precision for percentage values (default: 2).",
    )
    return parser.parse_args()


def summarize_backtests(args: argparse.Namespace) -> List[dict]:
    files = collect_files(args.patterns)
    if not files:
        print(
            "No backtest JSONs found (looking for {})".format(", ".join(args.patterns))
        )
        raise SystemExit(1)

    rows = []
    rf_daily = (1 + args.risk_free) ** (1 / args.annualization) - 1 if args.risk_free else 0.0

    for path in files:
        try:
            data = load_json(path)
            results = data.get("results", {})
            equity = (
                results.get("equity_curve")
                or data.get("equity_curve")
                or data.get("equity")
                or data.get("equity_series")
                or []
            )
            trades = results.get("trades") or data.get("trades") or []
            start = results.get("start") or data.get("start")
            end = results.get("end") or data.get("end")

            if len(equity) < 2:
                print(f"[WARN] {path}: no equity curve found; skipping.")
                continue

            rets = equity_to_returns(equity)
            total_pnl = equity[-1] / equity[0] - 1.0 if equity[0] else float("nan")
            ann_pnl = annualized_return(rets, args.annualization)
            mdd = max_drawdown(equity)
            sh = sharpe(rets, rf_daily)
            so = sortino(rets, rf_daily)
            wr = (
                sum(1 for t in trades if t.get("pnl", 0) > 0) / len(trades)
                if trades
                else float("nan")
            )
            pf = profit_factor(trades) if trades else float("nan")

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
                    "trades": len(trades),
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


def render_summary(rows: Sequence[dict], precision: int) -> None:
    print("\n=== Backtest Summary ===")
    for row in rows:
        print(f"- {row['file']} {row['period']}")
        print(
            f"  PnL: {pct(row['total_pnl'], precision)}  | Annualized: {pct(row['ann_pnl'], precision)}"
        )
        print(
            f"  MaxDD: {pct(row['mdd'], precision)} | Sharpe: {row['sharpe']:.2f} | Sortino: {row['sortino']:.2f}"
        )
        pf = (
            f"| ProfitFactor: {row['profit_factor']:.2f}"
            if isinstance(row["profit_factor"], float) and not math.isnan(row["profit_factor"])
            else ""
        )
        win_rate = pct(row["win_rate"], precision) if isinstance(row["win_rate"], float) else "n/a"
        print(f"  Win-rate: {win_rate} {pf}")
        print(f"  Trades: {row['trades']}\n")


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
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
