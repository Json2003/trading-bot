"""Run the expanded grid but with the trend filter (and optional trailing stop) enabled.

Writes results to repo root `opt_results_trend.json`.
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

    params = {
        'take_profit_pct': [0.002, 0.004, 0.006, 0.01],
        'stop_loss_pct': [0.003, 0.005, 0.01],
        'max_holding_bars': [6, 12, 24, 48],
        'risk_pct': [0.005, 0.01],
        'trailing_stop_pct': [None, 0.005, 0.01]
    }

    combos = list(itertools.product(
        params['take_profit_pct'],
        params['stop_loss_pct'],
        params['max_holding_bars'],
        params['risk_pct'],
        params['trailing_stop_pct']
    ))

    results = []
    total = len(combos)
    start = time.time()
    for idx, (tp, sl, hold, risk, trail) in enumerate(combos, start=1):
        print(f'[{idx}/{total}] tp={tp} sl={sl} hold={hold} risk={risk} trail={trail}')
        stats = aggressive_strategy_backtest(
            df,
            take_profit_pct=tp,
            stop_loss_pct=sl,
            max_holding_bars=hold,
            fee_pct=0.001,
            slippage_pct=0.0005,
            starting_balance=10000.0,
            trend_filter=True,
            ema_fast=50,
            ema_slow=200,
            vol_filter=False,
            trailing_stop_pct=trail
        )
        results.append({
            'params': {'tp': tp, 'sl': sl, 'hold': hold, 'risk': risk, 'trail': trail},
            'win_rate': stats.get('win_rate_pct', 0.0),
            'pnl': stats.get('pnl', 0.0),
            'trades': stats.get('trades', 0)
        })

    results_sorted = sorted(results, key=lambda x: (x['win_rate'], x['pnl']), reverse=True)
    out = Path(HERE.parent) / 'opt_results_trend.json'
    out.write_text(json.dumps(results_sorted, indent=2))
    elapsed = time.time() - start
    print(f'Saved {out} ({len(results_sorted)} rows) in {elapsed:.1f}s')


if __name__ == '__main__':
    run_grid()
