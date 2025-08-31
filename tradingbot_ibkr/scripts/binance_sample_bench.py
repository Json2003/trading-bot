#!/usr/bin/env python3
"""
Download a small byte range from a Binance Vision archive to estimate network throughput
and estimate full-download time based on Content-Length.

Usage: python binance_sample_bench.py --symbol BTCUSDT --yyyy 2021 --mm 01 --bytes 33554432
"""
import argparse, time, sys
from pathlib import Path
import requests

BASE = "https://data.binance.vision"
SPOT_MONTHLY = "/data/spot/monthly/trades/{symbol}/{symbol}-trades-{yyyy}-{mm}.zip"

def human(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m}m{s}s" if h else (f"{m}m{s}s" if m else f"{s}s")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--yyyy', required=True)
    ap.add_argument('--mm', required=True)
    ap.add_argument('--bytes', type=int, default=32*1024*1024)
    args = ap.parse_args()

    url = BASE + SPOT_MONTHLY.format(symbol=args.symbol, yyyy=args.yyyy, mm=args.mm)
    print('Probing', url)
    s = requests.Session()
    try:
        h = s.head(url, timeout=20)
        if h.status_code != 200:
            print('HEAD status', h.status_code)
            sys.exit(2)
        total = int(h.headers.get('Content-Length', '0'))
        print('Content-Length bytes=', total)
    except Exception as e:
        print('HEAD failed:', e)
        sys.exit(2)

    rng_end = min(args.bytes - 1, total - 1)
    headers = {'Range': f'bytes=0-{rng_end}'}
    tmp = Path('.').absolute() / f'sample_{args.symbol}_{args.yyyy}_{args.mm}.part'
    print('Downloading sample bytes 0..', rng_end)
    t0 = time.time()
    with s.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        written = 0
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
    t1 = time.time()
    elapsed = t1 - t0
    mb = written / (1024*1024)
    mbps = mb / elapsed if elapsed>0 else 0
    print(f'Downloaded {written} bytes ({mb:.2f} MiB) in {elapsed:.2f}s -> {mbps:.2f} MiB/s')

    if total:
        est_seconds = total / (mb * (1 if mb>0 else 1)) * elapsed if mb>0 else float('inf')
        # more directly estimate via ratio
        est_seconds = total / (mb * 1024*1024) * elapsed if mb>0 else float('inf')
        print('Estimated full download time:', human(est_seconds), f'({est_seconds:.0f}s)')
        total_mb = total / (1024*1024)
        print(f'Total size: {total_mb:.2f} MiB -> estimated avg {mbps:.2f} MiB/s')

    print('Sample file written to', tmp)

if __name__ == '__main__':
    main()
