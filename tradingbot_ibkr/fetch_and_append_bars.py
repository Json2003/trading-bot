"""Fetch latest 1m OHLCV via ccxt and append to datafiles CSV.

Usage (PowerShell):
  python fetch_and_append_bars.py --symbol BTC/USDT --timeframe 1m
"""
import argparse
import time
from pathlib import Path
import pandas as pd
import ccxt
import os

def fetch_and_append(symbol='BTC/USDT', timeframe='1m', limit=1000, out_dir=Path(__file__).resolve().parent / 'datafiles'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{symbol.replace('/','_')}_bars.csv"

    # load existing
    if fname.exists():
        df_existing = pd.read_csv(fname, parse_dates=['ts'], index_col='ts')
        last_ts = int(df_existing.index[-1].timestamp() * 1000)
    else:
        df_existing = None
        last_ts = None

    ex_name = os.getenv('EXCHANGE', 'binance')
    ex_cls = getattr(ccxt, ex_name)
    ex = ex_cls()

    all_new = []
    since = last_ts + 1 if last_ts is not None else None
    while True:
        try:
            if since:
                data = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            else:
                data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            print('fetch failed:', e)
            break
        if not data:
            break
        # convert to dataframe
        batch = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
        batch['ts'] = pd.to_datetime(batch['ts'], unit='ms')
        # drop any bars <= last_ts
        if last_ts is not None:
            batch = batch[batch['ts'].astype('int64') // 10**6 > last_ts]
        if batch.empty:
            break
        all_new.append(batch)
        since = int(batch['ts'].iloc[-1].timestamp() * 1000) + 1
        # avoid hammering exchange
        time.sleep(0.2)

    if not all_new:
        print('No new bars to append')
        return fname, 0

    new_df = pd.concat(all_new, ignore_index=True)
    new_df.set_index('ts', inplace=True)
    # merge with existing
    if df_existing is not None:
        combined = pd.concat([df_existing, new_df])
        combined = combined[~combined.index.duplicated(keep='first')]
        combined.sort_index(inplace=True)
    else:
        combined = new_df

    combined.to_csv(fname)
    print(f'Appended {len(new_df)} new bars to {fname}')
    return fname, len(new_df)


def main():
    p = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='BTC/USDT')
    parser.add_argument('--timeframe', default='1m')
    parser.add_argument('--limit', type=int, default=1000)
    args = parser.parse_args()
    fetch_and_append(args.symbol, args.timeframe, args.limit)


if __name__ == '__main__':
    main()
