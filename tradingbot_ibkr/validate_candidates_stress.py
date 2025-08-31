"""Stress validation: stronger stress tests for selected candidates.

Increases fees/slippage, uses block-bootstrap Monte-Carlo, and runs more iterations.
Writes `validation_<label>_stress.json` reports to repo root.
"""
import json
from pathlib import Path
import time
import random

import pandas as pd

from backtest_ccxt import aggressive_strategy_backtest

HERE = Path(__file__).resolve().parent


def load_bars(symbol='BTC/USDT'):
    path = HERE / 'datafiles' / f"{symbol.replace('/','_')}_bars.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=['ts'], index_col='ts')


def max_drawdown_from_equity(equity):
    balances = [e['balance'] for e in equity]
    if not balances:
        return 0.0
    peak = balances[0]
    max_dd = 0.0
    for b in balances:
        if b > peak:
            peak = b
        dd = (peak - b) / peak if peak>0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd * 100.0


def block_bootstrap(trade_pnls, block_size=5, target_length=None):
    # create blocks
    if target_length is None:
        target_length = len(trade_pnls)
    blocks = [trade_pnls[i:i+block_size] for i in range(0, len(trade_pnls), block_size)]
    if not blocks:
        return []
    sample = []
    while len(sample) < target_length:
        b = random.choice(blocks)
        sample.extend(b)
    # trim to target_length
    return sample[:target_length]


def run_stress(candidate, df, starting_balance=10000.0, mc_runs=5000, fee_pct=0.002, slippage_pct=0.001, block_size=5):
    tp = candidate['tp']
    sl = candidate['sl']
    hold = candidate['hold']
    trail = candidate.get('trail')

    stats = aggressive_strategy_backtest(
        df,
        take_profit_pct=tp,
        stop_loss_pct=sl,
        max_holding_bars=hold,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
        starting_balance=starting_balance,
        trend_filter=True,
        trailing_stop_pct=trail
    )

    overall_dd = max_drawdown_from_equity(stats.get('equity_curve', []))

    trade_list = stats.get('trade_list', [])
    trade_pnls = [t.get('pnl', 0.0) for t in trade_list]

    mc = {'runs': mc_runs, 'prob_dd_gt_30pct': None, 'worst_dd_pct': None}
    if len(trade_pnls) < 5:
        mc['note'] = 'too few trades for reliable MC (stress)'
    else:
        exceed = 0
        worst = 0.0
        for _ in range(mc_runs):
            sample = block_bootstrap(trade_pnls, block_size=block_size, target_length=len(trade_pnls))
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
        'monte_carlo_block_bootstrap': mc
    }
    return report


def main():
    df = load_bars()
    candidates = [
        {'label': 'A', 'tp': 0.01, 'sl': 0.005, 'hold': 12, 'risk': 0.005, 'trail': None},
        {'label': 'B', 'tp': 0.01, 'sl': 0.01, 'hold': 12, 'risk': 0.005, 'trail': 0.01}
    ]

    out_files = []
    for c in candidates:
        print('Stress validating', c['label'])
        start = time.time()
        rep = run_stress(c, df, starting_balance=10000.0, mc_runs=5000, fee_pct=0.002, slippage_pct=0.001, block_size=5)
        duration = time.time() - start
        out_file = Path(HERE.parent) / f"validation_{c['label']}_stress.json"
        out_file.write_text(json.dumps(rep, indent=2))
        print(f'Wrote {out_file} (t={duration:.1f}s)')
        out_files.append(str(out_file))

    print('Done. Reports:', out_files)


if __name__ == '__main__':
    main()
