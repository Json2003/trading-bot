"""Expanded grid search optimizer for the aggressive strategy.

Writes ranked results to `opt_results_expanded.json` in the repository root.
"""
import itertools
import json
from pathlib import Path
import pandas as pd
import time

from backtest_ccxt import aggressive_strategy_backtest

HERE = Path(__file__).resolve().parent


def load_bars(symbol='BTC/USDT'):
    path = HERE / 'datafiles' / f"{symbol.replace('/','_')}_bars.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, parse_dates=['ts'], index_col='ts')


def run_grid(symbol='BTC/USDT'):
    df = load_bars(symbol)

    # Expanded / shifted grid
    params = {
        'take_profit_pct': [0.002, 0.004, 0.006, 0.01, 0.02],
        'stop_loss_pct': [0.001, 0.002, 0.003, 0.005, 0.01],
        'max_holding_bars': [3, 6, 12, 24, 48],
        'risk_pct': [0.005, 0.01, 0.02]
    }

    combos = list(itertools.product(
        params['take_profit_pct'],
        params['stop_loss_pct'],
        params['max_holding_bars'],
        params['risk_pct']
    ))

    results = []
    total = len(combos)
    start = time.time()
    for idx, (tp, sl, hold, risk) in enumerate(combos, start=1):
        print(f'[{idx}/{total}] tp={tp} sl={sl} hold={hold} risk={risk}')
        stats = aggressive_strategy_backtest(
            df,
            take_profit_pct=tp,
            stop_loss_pct=sl,
            max_holding_bars=hold,
            fee_pct=0.001,
            slippage_pct=0.0005,
            starting_balance=10000.0
        )
        results.append({
            'params': {'tp': tp, 'sl': sl, 'hold': hold, 'risk': risk},
            'win_rate': stats.get('win_rate_pct', 0.0),
            'pnl': stats.get('pnl', 0.0),
            'trades': stats.get('trades', 0)
        })

    # rank by win_rate then pnl
    results_sorted = sorted(results, key=lambda x: (x['win_rate'], x['pnl']), reverse=True)
    out = Path(HERE.parent) / 'opt_results_expanded.json'
    out.write_text(json.dumps(results_sorted, indent=2))
    elapsed = time.time() - start
    print(f'Saved {out} ({len(results_sorted)} rows) in {elapsed:.1f}s')


if __name__ == '__main__':
    run_grid()
