"""Fetch recent trades from Binance using python-binance Client and write JSON-lines.

Usage (PowerShell):
  # requires BINANCE_API_KEY and BINANCE_API_SECRET in env or a .env file
  python binance_api_trade_fetch.py --symbol BTCUSDT --out-dir ./downloads --limit 1000

Notes:
  - This script uses the python-binance client if installed. It can be run without API keys
    to fetch public recent trades, but some endpoints may require API credentials.
  - The script writes newline-delimited JSON to <out-dir>/<symbol>_trades_YYYYMMDDHHMMSS.jsonl
"""
import argparse
from pathlib import Path
import os
import json
import time
from datetime import datetime

try:
    from binance.client import Client
except Exception:
    Client = None


def create_client():
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    if Client is None:
        raise RuntimeError('python-binance is not installed. Install with pip install python-binance')
    if api_key and api_secret:
        return Client(api_key, api_secret)
    # create unauthenticated client (some endpoints still work)
    return Client()


def fetch_recent_trades(client, symbol='BTCUSDT', limit=1000):
    # wrapper around client.get_recent_trades
    # returns a list of dicts
    return client.get_recent_trades(symbol=symbol, limit=limit)


def main():
    p = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--out-dir', default=str(p / 'downloads'))
    parser.add_argument('--limit', type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = create_client()
    trades = fetch_recent_trades(client, symbol=args.symbol, limit=args.limit)

    ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    fname = out_dir / f"{args.symbol}_trades_{ts}.jsonl"
    with open(fname, 'w', encoding='utf-8') as f:
        for t in trades:
            f.write(json.dumps(t) + '\n')

    print(f'Wrote {len(trades)} trades to {fname}')


if __name__ == '__main__':
    main()
