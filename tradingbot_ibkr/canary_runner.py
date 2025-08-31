"""Canary paper-runner for Candidate B.

Usage:
  python canary_runner.py --once --allocation-pct 0.005 --capital 10000
  python canary_runner.py --install-startup

Writes status to `canary_status.json` in repo root for dashboard/tray to read.
"""
import argparse
import json
import time
import sys
from pathlib import Path

from backtest_ccxt import aggressive_strategy_backtest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATUS_FILE = ROOT / 'canary_status.json'


def create_startup_shortcut(args):
    try:
        import os
        import win32com.client
        startup = Path(os.environ['APPDATA']) / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs' / 'Startup'
        shortcut_path = startup / 'roman_bot_canary.lnk'
        shell = win32com.client.Dispatch('WScript.Shell')
        shortcut = shell.CreateShortcut(str(shortcut_path))
        python = sys.executable
        script = str(HERE / 'canary_runner.py')
        shortcut.Targetpath = python
        shortcut.Arguments = f'"{script}" --background'
        shortcut.WorkingDirectory = str(ROOT)
        shortcut.IconLocation = python
        shortcut.save()
        print('Created startup shortcut:', shortcut_path)
    except Exception as e:
        print('Unable to create startup shortcut:', e)


def run_once(allocation_pct, capital):
    # Candidate B params
    candidate = {'tp': 0.01, 'sl': 0.01, 'hold': 12, 'risk': 0.005, 'trail': 0.01}
    # use allocation of capital for the canary
    allocation_amount = capital * allocation_pct
    df_path = HERE / 'datafiles' / 'BTC_USDT_bars.csv'
    if not df_path.exists():
        raise FileNotFoundError(df_path)
    import pandas as pd
    df = pd.read_csv(df_path, parse_dates=['ts'], index_col='ts')

    stats = aggressive_strategy_backtest(
        df,
        take_profit_pct=candidate['tp'],
        stop_loss_pct=candidate['sl'],
        max_holding_bars=candidate['hold'],
        fee_pct=0.001,
        slippage_pct=0.0005,
        starting_balance=allocation_amount,
        trend_filter=True,
        trailing_stop_pct=candidate['trail']
    )

    status = {
        'timestamp': time.time(),
        'candidate': candidate,
        'allocation_pct': allocation_pct,
        'allocation_amount': allocation_amount,
        'metrics': {
            'trades': stats.get('trades'),
            'win_rate': stats.get('win_rate_pct'),
            'pnl': stats.get('pnl'),
            'overall_max_drawdown_pct': stats.get('equity_curve') and max((max(e['balance'] for e in stats['equity_curve']) - min(e['balance'] for e in stats['equity_curve'])) / max(e['balance'] for e in stats['equity_curve']) * 100.0, 0) or 0
        },
        'last_trades': stats.get('trade_list', [])[-10:]
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2))
    print('Wrote status to', STATUS_FILE)
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--once', action='store_true')
    p.add_argument('--background', action='store_true')
    p.add_argument('--install-startup', action='store_true')
    p.add_argument('--allocation-pct', type=float, default=0.005, help='Fraction of capital to allocate to canary (e.g., 0.005 = 0.5%)')
    p.add_argument('--capital', type=float, default=10000.0)
    p.add_argument('--interval', type=int, default=60)
    args = p.parse_args()

    if args.install_startup:
        create_startup_shortcut(args)
        return

    if args.once:
        run_once(args.allocation_pct, args.capital)
        return

    if args.background:
        print('Starting canary runner in background mode; writing status every', args.interval, 's')
        try:
            while True:
                run_once(args.allocation_pct, args.capital)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print('Exiting canary runner')
            return

    # default: run once
    run_once(args.allocation_pct, args.capital)


if __name__ == '__main__':
    main()
