"""Demo script: fetch recent bars, append to data store, run batch training, save candidate model.

Run from the tradingbot_ibkr folder:
    python run_training_demo.py
"""
import os
from dotenv import load_dotenv
import ccxt
import pandas as pd
from pathlib import Path

load_dotenv()
EXCHANGE = os.getenv('EXCHANGE', 'binance')

def fetch_ohlcv(symbol, timeframe='1h', limit=500):
    ex = getattr(ccxt, EXCHANGE)()
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=False)
    return df

def run(symbol='BTC/USDT'):
    print('Demo: fetching bars for', symbol)
    df = fetch_ohlcv(symbol)
    # append to data store and optionally sync with GCS
    from data.store import (
        append_bars,
        load_bars,
        load_bars_from_gcs,
        save_bars_to_gcs,
    )
    # ensure ts column
    if 'ts' not in df.columns:
        df = df.reset_index()
    append_bars(symbol, df)
    gcs_bucket = os.getenv('GCS_BUCKET')
    if gcs_bucket:
        # upload consolidated bars to GCS and reload from there
        local = load_bars(symbol)
        save_bars_to_gcs(symbol, local, gcs_bucket)
        bars = load_bars_from_gcs(symbol, gcs_bucket)
    else:
        bars = load_bars(symbol)
    if bars.empty:
        print('No bars stored, abort')
        return

    # Try running batch training if scikit-learn is available
    try:
        from models.train_batch import train_and_evaluate
        print('Running batch training...')
        res = train_and_evaluate(bars)
        print('Train result:', res)
    except Exception as e:
        print('Batch training skipped (missing dependency or error):', e)

    # candidate model already saved by train_and_evaluate to model_store
    print('Done. Candidate model saved in models/model_store')

if __name__ == '__main__':
    run()
