#!/usr/bin/env python3
"""
Discover symbols from Binance (ccxt) and probe/download archives from data.binance.vision.

Usage (list-only probe):
  python binance_download_all.py --since 2021-01 --until 2021-03 --symbols-regex ".*USDT$" --list-only

When not in list-only, the script shells out to `binance_vision_full_download.py` per symbol.
"""
import argparse
import re
import subprocess
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

def yyyymm_iter(start, end):
    cur = datetime.strptime(start, "%Y-%m")
    last = datetime.strptime(end, "%Y-%m")
    while cur <= last:
        yield cur.strftime("%Y"), cur.strftime("%m")
        cur += relativedelta(months=1)


def month_days(yyyy, mm):
    dt = datetime.strptime(f"{yyyy}-{mm}-01", "%Y-%m-%d")
    nxt = dt + relativedelta(months=1)
    return [f"{d:02d}" for d in range(1, (nxt - dt).days + 1)]


def head_exists(url):
    try:
        import requests
        r = requests.head(url, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def discover_symbols():
    # try using ccxt first
    try:
        import ccxt
        ex = ccxt.binance()
        markets = ex.load_markets()
        syms = list(markets.keys())
        # ccxt gives symbols like 'BTC/USDT' — convert
        syms = [s.replace('/', '').replace('-', '').upper() for s in syms]
        return sorted(set(syms))
    except Exception:
        # fallback common list
        return ['BTCUSDT','ETHUSDT','BNBUSDT','ADAUSDT','XRPUSDT']


def probe_symbol(symbol, start, end):
    BASE = "https://data.binance.vision"
    SPOT_MONTHLY = "/data/spot/monthly/trades/{symbol}/{symbol}-trades-{yyyy}-{mm}.zip"
    SPOT_DAILY   = "/data/spot/daily/trades/{symbol}/{yyyy}/{mm}/{symbol}-trades-{yyyy}-{mm}-{dd}.zip"
    found = []
    for yyyy, mm in yyyymm_iter(start, end):
        murl = BASE + SPOT_MONTHLY.format(symbol=symbol, yyyy=yyyy, mm=mm)
        if head_exists(murl):
            found.append(murl)
            continue
        for dd in month_days(yyyy, mm):
            durl = BASE + SPOT_DAILY.format(symbol=symbol, yyyy=yyyy, mm=mm, dd=dd)
            if head_exists(durl):
                found.append(durl)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default='2017-01')
    ap.add_argument('--until', default=datetime.utcnow().strftime('%Y-%m'))
    ap.add_argument('--symbols-regex', default='.*')
    ap.add_argument('--out', default='./raw_all')
    ap.add_argument('--list-only', action='store_true')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()

    import re
    regex = re.compile(args.symbols_regex)
    syms = discover_symbols()
    syms = [s for s in syms if regex.search(s)]
    print(f'Found {len(syms)} symbols matching regex')
    # probe a few symbols first
    results = {}
    for s in syms[:200]:
        urls = probe_symbol(s, args.since, args.until)
        if urls:
            results[s] = urls
            print(s, '->', len(urls), 'matches (example first 3):')
            for u in urls[:3]:
                print('  ', u)

    total = sum(len(v) for v in results.values())
    print('\nTotal archive URLs found for matched symbols:', total)

    if not args.list_only and results:
        # spawn downloads for matched symbols
        for s in sorted(results.keys()):
            cmd = [sys.executable, 'tradingbot_ibkr/scripts/binance_vision_full_download.py', '--symbol', s, '--since', args.since, '--until', args.until, '--out', args.out, '--threads', '4']
            print('Running:', ' '.join(cmd))
            subprocess.Popen(cmd)


if __name__ == '__main__':
    main()
