#!/usr/bin/env python3
"""
Backtest Harness CLI

Usage examples:
  python scripts/run_backtest.py --source csv --path data/BTCUSDT-1h.csv
  python scripts/run_backtest.py --source ccxt --exchange binance --symbol "BTC/USDT" --timeframe 1h --since 2023-01-01 --until 2025-08-31

Loads OHLCV into a pandas DataFrame and runs aggressive_strategy_backtest
from tradingbot_ibkr.backtest_ccxt, saving a JSON report and printing a
short summary.
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Optional

# Protect against local shadowing of pandas/requests by files in repo root
# We'll keep REPO_ROOT on path for local package imports, but ensure third-party
# imports resolve from site-packages by temporarily reordering sys.path.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

def import_third_party(mod_name: str):
    """Import a third-party module ensuring repo-local files (e.g., pandas.py) don't shadow it."""
    import importlib, site, sysconfig
    original = sys.path.copy()
    try:
        # Move any repo paths to the end so site-packages are searched first
        repo_paths = {p for p in original if REPO_ROOT in os.path.abspath(p)}
        non_repo = [p for p in original if p not in repo_paths]
        sys.path = non_repo + [p for p in original if p in repo_paths]
        return importlib.import_module(mod_name)
    finally:
        sys.path = original

# Lazy imports inside functions to avoid importing heavy deps on --help

def parse_date(d: str) -> datetime:
    # Accept YYYY-MM-DD or epoch ms/seconds
    d = d.strip()
    if d.isdigit():
        ts = int(d)
        # Heuristic: if it's 13 digits, treat as ms
        if ts > 10_000_000_000:
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    try:
        return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date: {d}. Use YYYY-MM-DD or epoch ms/seconds")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a minimal OHLCV backtest")
    p.add_argument("--source", choices=["csv", "ccxt"], required=True, help="Data source")
    # CSV args
    p.add_argument("--path", help="Path to CSV with columns: ts/open/high/low/close/volume")
    # CCXT args
    p.add_argument("--exchange", default="binance", help="CCXT exchange id, e.g., binance")
    p.add_argument("--symbol", default="BTC/USDT", help="Symbol, e.g., BTC/USDT")
    p.add_argument("--timeframe", default="1h", help="Timeframe, e.g., 1m, 1h, 4h, 1d")
    p.add_argument("--since", type=parse_date, help="Start date YYYY-MM-DD or epoch (s/ms)")
    p.add_argument("--until", type=parse_date, help="End date YYYY-MM-DD or epoch (s/ms)")

    # Strategy knobs (subset to keep simple)
    p.add_argument("--tp", type=float, default=0.004, help="Take profit fraction (0.004=0.4%%)")
    p.add_argument("--sl", type=float, default=0.002, help="Stop loss fraction (0.002=0.2%%)")
    p.add_argument("--hold", type=int, default=12, help="Max holding bars")
    p.add_argument("--fees", type=float, default=0.0, help="Fee per side fraction")
    p.add_argument("--slippage", type=float, default=0.0, help="Slippage fraction")
    p.add_argument("--start-balance", type=float, default=10_000.0, help="Starting balance")

    p.add_argument("--trend", action="store_true", help="Enable EMA trend filter (50/200)")
    p.add_argument("--vol", action="store_true", help="Enable volatility filter")
    p.add_argument("--strategy", default="default", choices=["default", "sma_cross", "enhanced", "breakout"],
                   help="Signal generation strategy to use")

    p.add_argument("--out", default=None, help="Path to save JSON report (default auto)")
    return p


def load_csv(path: str):
    pd = import_third_party('pandas')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # Support ts in ms or ISO; fallback to index if present
    if 'ts' in df.columns:
        try:
            df['ts'] = pd.to_datetime(df['ts'], utc=True)
        except Exception:
            # maybe epoch ms
            df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True, errors='coerce')
        df = df.set_index('ts')
    elif df.index.name:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        raise ValueError("CSV must include 'ts' column or datetime index")

    # Normalize col names
    cols = {c.lower(): c for c in df.columns}
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in cols]
    if missing:
        # try case-insensitive
        lower_cols = {c.lower(): c for c in df.columns}
        if any(c not in lower_cols for c in required):
            raise ValueError(f"CSV missing columns: {required}")
        df = df.rename(columns=lower_cols)
    else:
        df = df.rename(columns=cols)

    # keep essential columns
    keep = ['open','high','low','close','volume']
    present = [c for c in keep if c in df.columns]
    return df[present]


def fetch_ccxt(exchange: str, symbol: str, timeframe: str, since: Optional[datetime], until: Optional[datetime]):
    ccxt = import_third_party('ccxt')
    pd = import_third_party('pandas')

    ex_cls = getattr(ccxt, exchange)
    ex = ex_cls()

    limit = 1000
    all_rows = []
    since_ms = int(since.timestamp()*1000) if since else None
    until_ms = int(until.timestamp()*1000) if until else None

    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        since_ms = batch[-1][0] + 1
        if len(batch) < limit:
            break
        if until_ms and since_ms >= until_ms:
            break

    if not all_rows:
        raise RuntimeError("No OHLCV fetched")

    df = pd.DataFrame(all_rows, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.set_index('ts')
    return df


def run_backtest(args: argparse.Namespace) -> dict:
    # Import here to avoid heavy import on --help
    from tradingbot_ibkr.backtest_ccxt import aggressive_strategy_backtest

    if args.source == 'csv':
        if not args.path:
            raise SystemExit("--path is required for --source csv")
        df = load_csv(args.path)
    else:
        df = fetch_ccxt(args.exchange, args.symbol, args.timeframe, args.since, args.until)

    # Apply signal generation if specified
    if args.strategy != "default":
        from tradingbot_ibkr.signal_generators import get_signal_generator
        signal_func = get_signal_generator(args.strategy)
        signal_df = signal_func(df)
        # Add signals to the main dataframe
        df = df.join(signal_df, how='left')
        print(f"Applied {args.strategy} signal generation strategy")

    stats = aggressive_strategy_backtest(
        df,
        take_profit_pct=args.tp,
        stop_loss_pct=args.sl,
        max_holding_bars=args.hold,
        fee_pct=args.fees,
        slippage_pct=args.slippage,
        starting_balance=args.start_balance,
        trend_filter=args.trend,
        vol_filter=args.vol,
        enable_logging=True,
    )

    # Choose output path
    if args.out:
        out_path = args.out
    else:
        strategy_suffix = f"_{args.strategy}" if args.strategy != "default" else ""
        base = f"backtest_{args.source}_{args.symbol.replace('/','_') if args.source=='ccxt' else os.path.basename(args.path).split('.')[0]}_{args.timeframe if args.source=='ccxt' else 'csv'}{strategy_suffix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        out_path = os.path.join(REPO_ROOT, f"{base}.json")

    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    # Print compact summary
    summary_keys = [
        'total_trades','win_rate_pct','profit_factor','sharpe_ratio','max_drawdown_pct','total_return_pct'
    ]
    compact = {k: stats.get(k) for k in summary_keys if k in stats}
    print("Saved report:", out_path)
    print(json.dumps(compact, indent=2))
    return stats


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run_backtest(args)

if __name__ == '__main__':
    main()
