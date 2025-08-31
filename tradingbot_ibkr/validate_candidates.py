"""Validate selected strategy candidates with walk-forward and Monte-Carlo resampling.

Produces a JSON report per candidate in the repo root named `validation_<label>.json`.
"""
import json
from pathlib import Path
import time
import random
import math

import pandas as pd

from backtest_ccxt import aggressive_strategy_backtest

HERE = Path(__file__).resolve().parent


def load_bars(symbol='BTC/USDT'):
    path = HERE / 'datafiles' / f"{symbol.replace('/','_')}_bars.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=['ts'], index_col='ts')


def max_drawdown_from_equity(equity, starting_balance):
    # equity: list of {'time':..., 'balance':...}
    balances = [e['balance'] for e in equity]
    if not balances:
        return 0.0
    peak = balances[0]
    max_dd = 0.0
    for b in balances:
        if b > peak:
            peak = b
        dd = (peak - b) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd * 100.0


def run_validation(candidate, df, starting_balance=10000.0, mc_runs=1000):
    tp = candidate['tp']
    sl = candidate['sl']
    hold = candidate['hold']
    risk = candidate['risk']
    trail = candidate.get('trail')

    stats = aggressive_strategy_backtest(
        df,
        take_profit_pct=tp,
        stop_loss_pct=sl,
        max_holding_bars=hold,
        fee_pct=0.001,
        slippage_pct=0.0005,
        starting_balance=starting_balance,
        trend_filter=True,
        trailing_stop_pct=trail
    )

    overall_dd = max_drawdown_from_equity(stats.get('equity_curve', []), starting_balance)

    # walk-forward: split df into 3 contiguous OOS chunks and compute OOS metrics
    n = len(df)
    if n < 50:
        wf = []
    else:
        parts = 3
        chunk = n // parts
        wf = []
        for i in range(parts):
            start = i * chunk
            end = (i + 1) * chunk if i < parts - 1 else n
            sub = df.iloc[start:end]
            s = aggressive_strategy_backtest(
                sub,
                take_profit_pct=tp,
                stop_loss_pct=sl,
                max_holding_bars=hold,
                fee_pct=0.001,
                slippage_pct=0.0005,
                starting_balance=starting_balance,
                trend_filter=True,
                trailing_stop_pct=trail
            )
            wf.append({
                'period': i + 1,
                'trades': s.get('trades', 0),
                'win_rate': s.get('win_rate_pct', 0.0),
                'pnl': s.get('pnl', 0.0),
                'max_drawdown_pct': max_drawdown_from_equity(s.get('equity_curve', []), starting_balance)
            })

    # Monte Carlo on trade-level PnL
    trade_list = stats.get('trade_list', [])
    trade_pnls = [t.get('pnl', 0.0) for t in trade_list]
    mc = {
        'runs': mc_runs,
        'samples': 0,
        'prob_dd_gt_30pct': None,
        'worst_dd_pct': None
    }
    if len(trade_pnls) < 5:
        mc['note'] = 'too few trades for reliable MC'
    else:
        exceed = 0
        worst = 0.0
        for _ in range(mc_runs):
            sample = [random.choice(trade_pnls) for _ in range(len(trade_pnls))]
            bal = starting_balance
            peak = bal
            max_dd = 0.0
            for pnl in sample:
                bal += pnl
                if bal > peak:
                    peak = bal
                dd = (peak - bal) / peak if peak>0 else 0.0
                if dd > max_dd:
                    max_dd = dd
            max_dd_pct = max_dd * 100.0
            if max_dd_pct > 30.0:
                exceed += 1
            if max_dd_pct > worst:
                worst = max_dd_pct
        mc['samples'] = mc_runs
        mc['prob_dd_gt_30pct'] = exceed / mc_runs
        mc['worst_dd_pct'] = worst

    report = {
        'candidate': candidate,
        'summary': {
            'trades': stats.get('trades', 0),
            'win_rate': stats.get('win_rate_pct', 0.0),
            'pnl': stats.get('pnl', 0.0),
            'overall_max_drawdown_pct': overall_dd
        },
        'walk_forward': wf,
        'monte_carlo': mc
    }
    return report


def main():
    df = load_bars()
    starting_balance = 10000.0
    candidates = [
        {'label': 'A', 'tp': 0.01, 'sl': 0.005, 'hold': 1, 'risk': 0.5, 'trail': None},
        {'label': 'B', 'tp': 0.01, 'sl': 0.01, 'hold': 1, 'risk': 0.5, 'trail': 0.01},
        # Aggressive candidate for maximum gain and insight
        {'label': 'MAX', 'tp': 0.15, 'sl': 0.15, 'hold': 1, 'risk': 0.5, 'trail': 0.10},
        # High risk, high reward candidate
        {'label': 'HR', 'tp': 0.12, 'sl': 0.12, 'hold': 1, 'risk': 0.5, 'trail': 0.08}
    ]

    out_reports = []
    for c in candidates:
        print('Validating candidate', c['label'])
        start = time.time()
        rep = run_validation(c, df, starting_balance=starting_balance, mc_runs=1000)
        duration = time.time() - start
        out_file = Path(HERE.parent) / f"validation_{c['label']}.json"
        out_file.write_text(json.dumps(rep, indent=2))
        print(f'Wrote {out_file} (t={duration:.1f}s)')
        out_reports.append(str(out_file))

    print('Done. Reports:', out_reports)


if __name__ == '__main__':
    main()
