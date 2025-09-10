#!/usr/bin/env python3
import argparse, importlib, os, sys, json
from typing import List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Avoid local shadows and force site-packages precedence
for shadow in ("pandas", "requests", "ccxt"):
    if shadow in sys.modules:
        del sys.modules[shadow]
try:
    import importlib.util as _ilut
    _req_spec = _ilut.find_spec("requests")
    if _req_spec is not None and getattr(_req_spec, "origin", "").endswith("requests.py"):
        raise ImportError("Local requests.py shadowing detected")
except Exception:
    site_paths = [p for p in sys.path if "site-packages" in p]
    non_site = [p for p in sys.path if "site-packages" not in p]
    sys.path[:] = site_paths + non_site
import pandas as pd

from backtest.exchange import fetch_ccxt
from backtest.engine import ExecConfig, run_backtest
from backtest.metrics import summarize


def load_strategy(spec: str):
    mod_name, fn_name = spec.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def parse_kv(csv: str) -> dict:
    out = {}
    if not csv:
        return out
    for pair in csv.split(','):
        if not pair or '=' not in pair:
            continue
        k, v = pair.split('=', 1)
        k = k.strip(); v = v.strip()
        try:
            out[k] = float(v) if '.' in v else int(v)
        except ValueError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exchange', default='kucoin')
    ap.add_argument('--symbols', default='BTC/USDT,ETH/USDT,SOL/USDT')
    ap.add_argument('--timeframe', default='4h')
    ap.add_argument('--extra_timeframe', default=None, help='Optional second timeframe to include (e.g., 1h)')
    ap.add_argument('--since', required=True)
    ap.add_argument('--until', required=True)
    ap.add_argument('--strategy', required=True)
    ap.add_argument('--strategy_args', default='')
    ap.add_argument('--strategy_args_extra', default='', help='Optional strategy args for extra timeframe')
    # engine
    ap.add_argument('--fees_bps', type=float, default=10.0)
    ap.add_argument('--slip_bps', type=float, default=5.0)
    ap.add_argument('--tp_bps', type=float, default=0.0)
    ap.add_argument('--sl_bps', type=float, default=0.0)
    ap.add_argument('--tp_atr_mult', type=float, default=0.0)
    ap.add_argument('--sl_atr_mult', type=float, default=0.0)
    ap.add_argument('--atr_period', type=int, default=14)
    ap.add_argument('--break_even_atr', type=float, default=0.0)
    ap.add_argument('--trail_atr_mult', type=float, default=0.0)
    ap.add_argument('--trail_method', choices=['atr','donchian'], default='atr')
    ap.add_argument('--trail_ref', choices=['best','close'], default='best')
    ap.add_argument('--donch_mid_n', type=int, default=0)
    # partial TP / payday
    ap.add_argument('--tp_r_multiple', type=float, default=0.0)
    ap.add_argument('--partial_tp_frac', type=float, default=0.0)
    ap.add_argument('--lock_in_r_after_tp', type=float, default=0.0)
    # pullback/structure exit
    ap.add_argument('--pullback_ema_len', type=int, default=0)
    ap.add_argument('--pullback_atr_mult', type=float, default=0.0)
    ap.add_argument('--pullback_confirm', type=int, default=0)
    # momentum/timebox
    ap.add_argument('--min_rr_by_bars_r', type=float, default=0.0)
    ap.add_argument('--min_rr_by_bars_n', type=int, default=0)
    ap.add_argument('--notional', type=float, default=1.0)
    ap.add_argument('--risk_per_trade', type=float, default=0.005)
    ap.add_argument('--max_notional_frac', type=float, default=1.0)
    ap.add_argument('--allow_short', action='store_true')
    ap.add_argument('--max_bars', type=int, default=0)
    ap.add_argument('--out_prefix', default='artifacts/portfolio')
    args = ap.parse_args()

    strat_fn = load_strategy(args.strategy)
    s_kwargs = parse_kv(args.strategy_args)
    strat = (lambda df: strat_fn(df, **s_kwargs)) if s_kwargs else strat_fn
    # Extra timeframe strategy kwargs (fallback to base if empty)
    s_kwargs_extra = parse_kv(args.strategy_args_extra) if args.strategy_args_extra else s_kwargs
    strat_extra = (lambda df: strat_fn(df, **s_kwargs_extra)) if s_kwargs_extra else strat_fn

    cfg = ExecConfig(
        fees_bps=args.fees_bps,
        slip_bps=args.slip_bps,
        tp_bps=args.tp_bps,
        sl_bps=args.sl_bps,
        tp_atr_mult=args.tp_atr_mult,
        sl_atr_mult=args.sl_atr_mult,
        atr_period=args.atr_period,
        break_even_atr_mult=args.break_even_atr,
        trail_atr_mult=args.trail_atr_mult,
        trail_method=args.trail_method,
        trail_ref=args.trail_ref,
        donch_mid_n=args.donch_mid_n,
        tp_r_multiple=args.tp_r_multiple,
        partial_tp_frac=args.partial_tp_frac,
        lock_in_r_after_tp=args.lock_in_r_after_tp,
        pullback_ema_len=args.pullback_ema_len,
        pullback_atr_mult=args.pullback_atr_mult,
        pullback_confirm=args.pullback_confirm,
        min_rr_by_bars_r=args.min_rr_by_bars_r,
        min_rr_by_bars_n=args.min_rr_by_bars_n,
        notional=args.notional,
        risk_per_trade=args.risk_per_trade,
        max_notional_frac=args.max_notional_frac,
        allow_short=args.allow_short,
        max_bars=args.max_bars,
    )

    os.makedirs('artifacts', exist_ok=True)

    combined_equity = None
    per_symbol = {}

    syms = [s.strip() for s in args.symbols.split(',') if s.strip()]
    components = []
    # Base timeframe
    for sym in syms:
        df = fetch_ccxt(args.exchange, sym, args.timeframe, args.since, args.until)
        trades, eq, bar_ret = run_backtest(df, strat, cfg)
        m = summarize(trades, eq, bar_ret)
        per_symbol[f"{sym}@{args.timeframe}"] = m
        eq = eq.rename(columns={'equity': f'equity_{sym.replace("/","_")}_{args.timeframe}'})
        components.append(eq)
    # Extra timeframe (optional)
    if args.extra_timeframe:
        for sym in syms:
            df2 = fetch_ccxt(args.exchange, sym, args.extra_timeframe, args.since, args.until)
            trades2, eq2, br2 = run_backtest(df2, strat_extra, cfg)
            m2 = summarize(trades2, eq2, br2)
            per_symbol[f"{sym}@{args.extra_timeframe}"] = m2
            eq2 = eq2.rename(columns={'equity': f'equity_{sym.replace("/","_")}_{args.extra_timeframe}'})
            components.append(eq2)

    # Merge all components by timestamp
    for eq in components:
        if combined_equity is None:
            combined_equity = eq
        else:
            combined_equity = pd.merge_asof(combined_equity.sort_values('timestamp'), eq.sort_values('timestamp'), on='timestamp')
            combined_equity = combined_equity.ffill()

    # Aggregate equity: simple sum of normalized equity curves (assumes same notional each)
    cols = [c for c in combined_equity.columns if c.startswith('equity_')]
    combined_equity['equity'] = combined_equity[cols].mean(axis=1)

    # Recompute metrics on combined equity
    combined_equity['equity_prev'] = combined_equity['equity'].shift(1).fillna(combined_equity['equity'].iloc[0])
    bar_ret = combined_equity['equity'] / combined_equity['equity_prev'] - 1.0
    portfolio_metrics = {
        'symbols': args.symbols,
        'timeframe': args.timeframe,
        'start': combined_equity['timestamp'].iloc[0],
        'end': combined_equity['timestamp'].iloc[-1],
        'total_return': (combined_equity['equity'].iloc[-1] / combined_equity['equity'].iloc[0]) - 1.0,
        'num_components': len(cols),
    }

    # Save artifacts
    combined_equity[['timestamp','equity']].to_csv(f"{args.out_prefix}_equity.csv", index=False)
    with open(f"{args.out_prefix}_metrics.json", 'w') as f:
        json.dump({'per_symbol': per_symbol, 'portfolio': portfolio_metrics}, f, indent=2, default=str)

    print(json.dumps({'portfolio': portfolio_metrics, 'per_symbol': per_symbol}, indent=2, default=str))

if __name__ == '__main__':
    main()
