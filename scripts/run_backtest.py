#!/usr/bin/env python3
"""
Backtest Harness CLI

Usage examples:
  python scripts/run_backtest.py --source csv --path data/BTCUSDT-1h.csv
  python scripts/run_backtest.py --source ccxt --exchange binance --symbol "BTC/USDT" --timeframe 1h --since 2023-01-01 --until 2025-08-31
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Move repo root to end to avoid shadowing site-packages, and drop empty path
sys.path = [p for p in sys.path if p not in ('', REPO_ROOT)] + [REPO_ROOT]

# Purge shadowed modules if already imported
for _name in ("requests", "pandas", "ccxt"):
    _m = sys.modules.get(_name)
    if _m is not None:
        _file = getattr(_m, '__file__', '') or ''
        try:
            if REPO_ROOT in os.path.abspath(_file):
                del sys.modules[_name]
        except Exception:
            pass


def _import_site(mod_name: str):
    import importlib
    # Purge repo-local shadow modules
    mod = sys.modules.get(mod_name)
    if mod is not None:
        mod_file = getattr(mod, '__file__', '') or ''
        try:
            if REPO_ROOT in os.path.abspath(mod_file):
                del sys.modules[mod_name]
        except Exception:
            pass
    original = sys.path.copy()
    try:
        repo_paths = {p for p in original if REPO_ROOT in os.path.abspath(p)}
        non_repo = [p for p in original if p not in repo_paths]
        sys.path = non_repo + [p for p in original if p in repo_paths]
        m = importlib.import_module(mod_name)
        sys.modules[mod_name] = m
        return m
    finally:
        sys.path = original


def parse_date(d: str) -> datetime:
    d = d.strip()
    if d.isdigit():
        ts = int(d)
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
    p.add_argument("--path", help="Path to CSV with columns: ts/open/high/low/close/volume")
    p.add_argument("--exchange", default="binance", help="CCXT exchange id, e.g., binance")
    p.add_argument("--symbol", default="BTC/USDT", help="Symbol, e.g., BTC/USDT")
    p.add_argument("--timeframe", default="1h", help="Timeframe, e.g., 1m, 1h, 4h, 1d")
    p.add_argument("--since", type=parse_date, help="Start date YYYY-MM-DD or epoch (s/ms)")
    p.add_argument("--until", type=parse_date, help="End date YYYY-MM-DD or epoch (s/ms)")
    # Modern engine: strategy + sizing/exits (optional)
    p.add_argument("--strategy", help="Strategy spec module:function, e.g., backtest.strategies.sma_filtered:generate_signals")
    p.add_argument("--strategy_args", default="", help="Comma-separated key=value pairs passed to strategy, e.g. fast=8,slow=21")
    p.add_argument("--tp", type=float, default=0.004, help="Take profit fraction (0.004=0.4%%)")
    p.add_argument("--sl", type=float, default=0.002, help="Stop loss fraction (0.002=0.2%%)")
    # Convenience bps flags (mapped to fraction); if provided, override --tp/--sl
    p.add_argument("--tp_bps", type=float, default=None, help="Take profit in bps, e.g., 40 for 0.40%%")
    p.add_argument("--sl_bps", type=float, default=None, help="Stop loss in bps, e.g., 20 for 0.20%%")
    p.add_argument("--hold", type=int, default=12, help="Max holding bars")
    p.add_argument("--fees", type=float, default=0.0, help="Fee per side fraction")
    p.add_argument("--slippage", type=float, default=0.0, help="Slippage fraction")
    # Convenience bps flags for fees/slippage (mapped to fraction); override --fees/--slippage
    p.add_argument("--fees_bps", type=float, default=None, help="Fees per side in bps, e.g., 10 = 0.10%%")
    p.add_argument("--slip_bps", type=float, default=None, help="Slippage per side in bps, e.g., 5 = 0.05%%")
    p.add_argument("--start-balance", type=float, default=10_000.0, help="Starting balance")
    p.add_argument("--trend", action="store_true", help="Enable EMA trend filter (50/200)")
    p.add_argument("--vol", action="store_true", help="Enable volatility filter")
    # Modern engine exits (bps or ATR), risk/sizing, and trade management
    p.add_argument("--tp_atr_mult", type=float, default=0.0, help="ATR-based TP multiple (0 disables)")
    p.add_argument("--sl_atr_mult", type=float, default=0.0, help="ATR-based SL multiple (0 disables)")
    p.add_argument("--atr_period", type=int, default=14, help="ATR period for ATR-based exits/sizing")
    # dynamic stops
    p.add_argument("--break_even_atr", type=float, default=0.0, help="Move stop to entry after +X*ATR in favor")
    p.add_argument("--trail_atr_mult", type=float, default=0.0, help="Trail stop by X*ATR (0 disables)")
    p.add_argument("--trail_method", choices=["atr","donchian"], default="atr", help="Trailing stop method")
    p.add_argument("--trail_ref", choices=["best","close"], default="best", help="Reference for ATR trailing distance")
    p.add_argument("--donch_mid_n", type=int, default=0, help="Donchian midline window if using --trail_method donchian (0 disables)")
    # partial TP / payday management
    p.add_argument("--tp_r_multiple", type=float, default=0.0, help="Payday at k*R (0 disables)")
    p.add_argument("--partial_tp_frac", type=float, default=0.0, help="Fraction to close at payday (0 disables)")
    p.add_argument("--lock_in_r_after_tp", type=float, default=0.0, help="After payday, lock stop to BE+X*R (0 keeps at BE)")
    # pullback/structure exit
    p.add_argument("--pullback_ema_len", type=int, default=0, help="EMA length for pullback bands (0 disables)")
    p.add_argument("--pullback_atr_mult", type=float, default=0.0, help="ATR band depth around EMA (0 disables)")
    p.add_argument("--pullback_confirm", type=int, default=0, help="Bars to confirm beyond band before exit")
    # momentum/timebox guard
    p.add_argument("--min_rr_by_bars_r", type=float, default=0.0, help="Must reach this R multiple by N bars (0 disables)")
    p.add_argument("--min_rr_by_bars_n", type=int, default=0, help="Bars deadline for min-R guard (0 disables)")
    p.add_argument("--notional", type=float, default=1.0, help="Starting equity units for modern engine")
    p.add_argument("--risk_per_trade", type=float, default=0.005, help="Risk per trade (fraction of equity), e.g. 0.005 = 0.5%%")
    p.add_argument("--max_notional_frac", type=float, default=1.0, help="Cap on position notional fraction of equity")
    p.add_argument("--allow_short", action="store_true", help="Allow short entries if strategy emits -1 signals")
    p.add_argument("--max_bars", type=int, default=0, help="Modern engine: exit after N bars in trade (0=disabled)")
    p.add_argument("--out", default=None, help="Path to save JSON report (default auto)")
    p.add_argument("--out_prefix", default=None, help="If set, save trades/equity/metrics with this prefix (e.g., prefix_trades.csv)")
    return p


def load_csv(path: str):
    pd = _import_site('pandas')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if 'ts' in df.columns:
        try:
            df['ts'] = pd.to_datetime(df['ts'], utc=True)
        except Exception:
            df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True, errors='coerce')
        df = df.set_index('ts')
    elif df.index.name:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        raise ValueError("CSV must include 'ts' column or datetime index")
    cols = {c.lower(): c for c in df.columns}
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in cols]
    if missing:
        lower_cols = {c.lower(): c for c in df.columns}
        if any(c not in lower_cols for c in required):
            raise ValueError(f"CSV missing columns: {required}")
        df = df.rename(columns=lower_cols)
    else:
        df = df.rename(columns=cols)
    keep = ['open','high','low','close','volume']
    present = [c for c in keep if c in df.columns]
    return df[present]


def fetch_ccxt(exchange: str, symbol: str, timeframe: str, since: Optional[datetime], until: Optional[datetime]):
    # Ensure real requests is loaded before ccxt (avoid local requests.py)
    _import_site('requests')
    ccxt = _import_site('ccxt')
    pd = _import_site('pandas')
    ex_cls = getattr(ccxt, exchange)
    ex = ex_cls({'enableRateLimit': True})
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


def _load_aggressive_strategy():
    # Load module directly from file to avoid importing tradingbot_ibkr/__init__.py
    import importlib.util
    mod_path = os.path.join(REPO_ROOT, 'tradingbot_ibkr', 'backtest_ccxt.py')
    spec = importlib.util.spec_from_file_location('backtest_ccxt_local', mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load module from {mod_path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['backtest_ccxt_local'] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, 'aggressive_strategy_backtest')


def run_backtest(args: argparse.Namespace) -> dict:
    # Preload real libs to avoid repo-local stubs interfering
    _import_site('requests')
    _import_site('pandas')
    _import_site('ccxt')
    
    # Decide path: modern engine when strategy/exits/risk flags are used
    def _is_modern(a: argparse.Namespace) -> bool:
        return (
            bool(a.strategy) or
            (a.tp_atr_mult and a.tp_atr_mult > 0.0) or
            (a.sl_atr_mult and a.sl_atr_mult > 0.0) or
            (a.max_bars and a.max_bars > 0) or
            (a.tp_bps is not None and a.tp_bps == 0) or
            (a.sl_bps is not None and a.sl_bps == 0)
        )

    if _is_modern(args):
        # Use modern engine (ExecConfig + run_backtest) with optional strategy
        from backtest.exchange import load_csv as _bx_load_csv, fetch_ccxt as _bx_fetch
        from backtest.engine import ExecConfig as _ExecConfig, run_backtest as _engine_run
        from backtest.metrics import summarize as _summarize

        # Load data
        if args.source == 'csv':
            if not args.path:
                raise SystemExit("--path is required for --source csv")
            df = _bx_load_csv(args.path)
        else:
            if not (args.exchange and args.symbol and args.timeframe and args.since and args.until):
                raise SystemExit("--exchange, --symbol, --timeframe, --since, --until are required for --source ccxt")
            # Reuse the original since/until datetime to string for fetch
            since_str = args.since.strftime('%Y-%m-%d') if hasattr(args.since, 'strftime') else str(args.since)
            until_str = args.until.strftime('%Y-%m-%d') if hasattr(args.until, 'strftime') else str(args.until)
            df = _bx_fetch(args.exchange, args.symbol, args.timeframe, since_str, until_str)

        # Strategy
        spec = args.strategy or "backtest.strategies.sma_filtered:generate_signals"
        import importlib as _il
        mod_name, fn_name = spec.split(":")
        fn = getattr(_il.import_module(mod_name), fn_name)
        kwargs = {}
        if args.strategy_args:
            for pair in args.strategy_args.split(','):
                if not pair or '=' not in pair:
                    continue
                k, v = pair.split('=', 1)
                k = k.strip(); v = v.strip()
                try:
                    v_cast = float(v) if ('.' in v or 'e' in v.lower() or '-' in v) else int(v)
                except ValueError:
                    v_cast = v
                kwargs[k] = v_cast
        strat = (lambda _d, _fn=fn, _kw=kwargs: _fn(_d, **_kw)) if kwargs else fn

        # Exec config
        fees_bps = args.fees_bps if args.fees_bps is not None else 0.0
        slip_bps = args.slip_bps if args.slip_bps is not None else 0.0
        tp_bps = args.tp_bps if args.tp_bps is not None else 0.0
        sl_bps = args.sl_bps if args.sl_bps is not None else 0.0
        cfg = _ExecConfig(
            fees_bps=fees_bps,
            slip_bps=slip_bps,
            tp_bps=tp_bps,
            sl_bps=sl_bps,
            tp_atr_mult=float(args.tp_atr_mult or 0.0),
            sl_atr_mult=float(args.sl_atr_mult or 0.0),
            atr_period=int(args.atr_period or 14),
            break_even_atr_mult=float(args.break_even_atr or 0.0),
            trail_atr_mult=float(args.trail_atr_mult or 0.0),
            trail_method=str(args.trail_method or "atr"),
            trail_ref=str(args.trail_ref or "best"),
            donch_mid_n=int(args.donch_mid_n or 0),
            tp_r_multiple=float(args.tp_r_multiple or 0.0),
            partial_tp_frac=float(args.partial_tp_frac or 0.0),
            lock_in_r_after_tp=float(args.lock_in_r_after_tp or 0.0),
            pullback_ema_len=int(args.pullback_ema_len or 0),
            pullback_atr_mult=float(args.pullback_atr_mult or 0.0),
            pullback_confirm=int(args.pullback_confirm or 0),
            min_rr_by_bars_r=float(args.min_rr_by_bars_r or 0.0),
            min_rr_by_bars_n=int(args.min_rr_by_bars_n or 0),
            notional=float(args.notional or 1.0),
            risk_per_trade=float(args.risk_per_trade or 0.0),
            max_notional_frac=float(args.max_notional_frac or 1.0),
            allow_short=bool(args.allow_short),
            max_bars=int(args.max_bars or 0),
        )

        trades, equity, bar_ret = _engine_run(df, strat, cfg)
        metrics = _summarize(trades, equity, bar_ret)

        # Save artifacts: prefer out_prefix if provided
        try:
            if args.out_prefix:
                pref = args.out_prefix
                outdir = os.path.dirname(pref) or "."
                os.makedirs(outdir, exist_ok=True)
                trades.to_csv(f"{pref}_trades.csv", index=False)
                equity.rename(columns={"equity": "equity"}).to_csv(f"{pref}_equity.csv", index=False)
                with open(f"{pref}_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2, sort_keys=True)
            else:
                os.makedirs("artifacts", exist_ok=True)
                trades.to_csv("artifacts/trades.csv", index=False)
                equity.to_csv("artifacts/equity_curve.csv", index=False)
                with open("artifacts/metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2, sort_keys=True)
        except Exception:
            pass

        print(json.dumps(metrics, indent=2))
        return metrics

    aggressive_strategy_backtest = _load_aggressive_strategy()

    if args.source == 'csv':
        if not args.path:
            raise SystemExit("--path is required for --source csv")
        df = load_csv(args.path)
    else:
        df = fetch_ccxt(args.exchange, args.symbol, args.timeframe, args.since, args.until)

    # Map bps convenience flags to fractions if provided
    tp_frac = args.tp_bps / 10_000.0 if getattr(args, 'tp_bps', None) is not None else args.tp
    sl_frac = args.sl_bps / 10_000.0 if getattr(args, 'sl_bps', None) is not None else args.sl
    fee_frac = args.fees_bps / 10_000.0 if getattr(args, 'fees_bps', None) is not None else args.fees
    slip_frac = args.slip_bps / 10_000.0 if getattr(args, 'slip_bps', None) is not None else args.slippage

    stats = aggressive_strategy_backtest(
        df,
        take_profit_pct=tp_frac,
        stop_loss_pct=sl_frac,
        max_holding_bars=args.hold,
        fee_pct=fee_frac,
        slippage_pct=slip_frac,
        starting_balance=args.start_balance,
        trend_filter=args.trend,
        vol_filter=args.vol,
        enable_logging=True,
    )

    if args.out:
        out_path = args.out
    else:
        base = f"backtest_{args.source}_{args.symbol.replace('/','_') if args.source=='ccxt' else os.path.basename(args.path).split('.')[0]}_{args.timeframe if args.source=='ccxt' else 'csv'}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        out_path = os.path.join(REPO_ROOT, f"{base}.json")

    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

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
