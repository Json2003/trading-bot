"""Fetch 1 year of historical bars and run the aggressive backtest, save report as JSON."""
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import ccxt
import pandas as pd
import json

load_dotenv()
EXCHANGE = os.getenv('EXCHANGE', 'binance')

def fetch_ohlcv_since(symbol, timeframe='1h', since_ts=None):
    ex = getattr(ccxt, EXCHANGE)()
    all_bars = []
    limit = 1000
    since = since_ts
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        all_bars.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < limit:
            break
    df = pd.DataFrame(all_bars, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df

def run(symbol='BTC/USDT'):
    end = datetime.utcnow()
    start = end - timedelta(days=365)
    since = int(start.timestamp() * 1000)
    print(f'Fetching {symbol} from {start.date()} to {end.date()}')
    df = fetch_ohlcv_since(symbol, timeframe='1h', since_ts=since)
    if df.empty:
        print('No data fetched')
        return

    from backtest_ccxt import aggressive_strategy_backtest
    stats = aggressive_strategy_backtest(df)
    out_path = f'report_{symbol.replace("/","_")}_{start.date()}.json'
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print('Saved report to', out_path)
    print('Summary:')
    print('Trades:', stats['trades'])
    print(f"Win rate: {stats['win_rate_pct']:.2f}%")
    print('Net PnL:', stats['pnl'])

if __name__ == '__main__':
    run()
