"""Grid search optimizer for the aggressive strategy.

Saves ranked results to `opt_results.json`.
"""
import itertools
import json
from pathlib import Path
import pandas as pd

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
        'take_profit_pct': [0.002, 0.004, 0.006],
        'stop_loss_pct': [0.001, 0.002, 0.003],
        'max_holding_bars': [6, 12, 24],
        'risk_pct': [0.01, 0.02]
    }
    combos = list(itertools.product(params['take_profit_pct'], params['stop_loss_pct'], params['max_holding_bars'], params['risk_pct']))
    results = []
    for tp, sl, hold, risk in combos:
        stats = aggressive_strategy_backtest(df, take_profit_pct=tp, stop_loss_pct=sl, max_holding_bars=hold, fee_pct=0.001, slippage_pct=0.0005, starting_balance=10000.0)
        stats['params'] = {'tp': tp, 'sl': sl, 'hold': hold, 'risk': risk}
        results.append({'params': stats['params'], 'win_rate': stats['win_rate_pct'], 'pnl': stats['pnl'], 'trades': stats['trades']})
    # rank by win_rate then pnl
    results_sorted = sorted(results, key=lambda x: (x['win_rate'], x['pnl']), reverse=True)
    out = Path('opt_results.json')
    out.write_text(json.dumps(results_sorted, indent=2))
    print('Saved', out)

if __name__ == '__main__':
    run_grid()
