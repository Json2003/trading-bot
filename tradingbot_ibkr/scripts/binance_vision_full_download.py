#!/usr/bin/env python3
"""
binance_vision_full_download.py — Download historical trade CSVs for one symbol
from Binance's official archive (https://data.binance.vision).

Example:
  python binance_vision_full_download.py --symbol BTCUSDT --since 2017-01 --until 2025-08 --out ./raw_trades
"""

import argparse
import sys
import zipfile
import threading
import queue
import re
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

BASE = "https://data.binance.vision"
SPOT_MONTHLY = "/data/spot/monthly/trades/{symbol}/{symbol}-trades-{yyyy}-{mm}.zip"
SPOT_DAILY   = "/data/spot/daily/trades/{symbol}/{yyyy}/{mm}/{symbol}-trades-{yyyy}-{mm}-{dd}.zip"


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


def url_exists(url, session=None):
    try:
        s = session or requests.Session()
        r = s.head(url, timeout=20)
        if r.status_code == 200:
            return True, int(r.headers.get('Content-Length', '0'))
    except Exception:
        pass
    return False, 0


def make_session():
    s = requests.Session()
    # retry on transient errors
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=(500,502,503,504))
    s.mount('https://', HTTPAdapter(max_retries=retries))
    s.headers.update({'User-Agent': 'binance-vision-downloader/1.0'})
    return s


def download(url, out_path, expected_size, session, pbar, timeout=60):
    tmp = out_path.with_suffix('.part')
    headers = {}
    if tmp.exists():
        pos = tmp.stat().st_size
        if expected_size and pos < expected_size:
            headers['Range'] = f"bytes={pos}-"

    if out_path.exists() and expected_size and out_path.stat().st_size == expected_size:
        # already complete
        pbar.update(expected_size)
        return True

    with session.get(url, stream=True, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        mode = 'ab' if 'Range' in headers else 'wb'
        with open(tmp, mode) as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    if expected_size and tmp.stat().st_size != expected_size:
        return False
    tmp.replace(out_path)
    return True


def unzip(zpath):
    try:
        with zipfile.ZipFile(zpath, 'r') as zf:
            zf.extractall(zpath.parent)
        return True
    except Exception as e:
        tqdm.write(f"Unzip failed ({zpath}): {e}")
        return False


def worker(q, pbar, remove_zip=False, unzip_ok=True):
    session = make_session()
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        url, out_zip, size = item
        try:
            ok = download(url, out_zip, size, session, pbar)
            if ok and unzip_ok:
                if unzip(out_zip) and remove_zip:
                    try:
                        out_zip.unlink()
                    except Exception:
                        pass
        except Exception as e:
            tqdm.write(f"Download failed: {url} -> {e}")
        finally:
            q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out", default="./raw_trades")
    ap.add_argument("--since", default="2017-01")
    ap.add_argument("--until", default=datetime.utcnow().strftime("%Y-%m"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--remove-zip", action='store_true', help='remove zip after successful unzip')
    ap.add_argument("--no-unzip", action='store_true', help='do not unzip archives')
    ap.add_argument("--merge", choices=("csv", "parquet"), help='merge downloaded CSVs into one output file (csv or parquet)')
    ap.add_argument("--merge-only", action='store_true', help='skip downloading and only merge existing CSVs in the out dir')
    ap.add_argument("--verify", action='store_true', help='verify CSV consistency before merging')
    ap.add_argument("--merged-name", default=None, help='filename (without ext) for merged output; defaults to {symbol}_trades')
    args = ap.parse_args()

    out_root = Path(args.out) / args.symbol
    out_root.mkdir(parents=True, exist_ok=True)

    session = make_session()

    def find_tasks():
        tasks = []
        for yyyy, mm in yyyymm_iter(args.since, args.until):
            url = BASE + SPOT_MONTHLY.format(symbol=args.symbol, yyyy=yyyy, mm=mm)
            ok, size = url_exists(url, session=session)
            if ok:
                out_zip = out_root / f"{args.symbol}-trades-{yyyy}-{mm}.zip"
                tasks.append((url, out_zip, size))
            else:
                for dd in month_days(yyyy, mm):
                    durl = BASE + SPOT_DAILY.format(symbol=args.symbol, yyyy=yyyy, mm=mm, dd=dd)
                    dok, dsize = url_exists(durl, session=session)
                    if dok:
                        out_zip = out_root / f"{args.symbol}-trades-{yyyy}-{mm}-{dd}.zip"
                        tasks.append((durl, out_zip, dsize))
        return tasks

    def find_csv_files():
        # find extracted CSVs in the symbol folder
        return sorted(out_root.glob('*.csv'))

    def verify_csvs(paths):
        import pandas as pd
        cols = None
        summary = []
        for p in paths:
            try:
                head = pd.read_csv(p, nrows=1000)
                nrows = sum(1 for _ in open(p, 'r', encoding='utf-8', errors='ignore')) - 1
                sample_cols = tuple(head.columns.tolist())
                if cols is None:
                    cols = sample_cols
                ok_cols = (sample_cols == cols)
                # quick check for a timestamp-like column
                ts_ok = False
                for c in head.columns:
                    if re.search('time|ts|timestamp', c, re.I):
                        try:
                            pd.to_datetime(head[c].iloc[:10])
                            ts_ok = True
                            break
                        except Exception:
                            pass
                summary.append({'file': str(p), 'rows': nrows, 'cols': list(head.columns), 'cols_match': ok_cols, 'ts_parsable': ts_ok})
            except Exception as e:
                summary.append({'file': str(p), 'error': str(e)})
        return cols, summary

    def merge_csvs(paths, out_path, to_parquet=False):
        import pandas as pd
        # read and concatenate in reasonable chunks
        dfs = []
        for p in paths:
            try:
                df = pd.read_csv(p)
                dfs.append(df)
            except Exception as e:
                tqdm.write(f"Skipping {p}: {e}")
        if not dfs:
            raise RuntimeError('no CSVs to merge')
        full = pd.concat(dfs, ignore_index=True)
        # dedupe
        id_col = None
        for candidate in ('id','tradeId','trade_id','tid'):
            if candidate in full.columns:
                id_col = candidate
                break
        if id_col:
            full = full.drop_duplicates(subset=[id_col])
        else:
            # fallback composite
            keys = [c for c in ('time','ts','timestamp','price','qty') if c in full.columns]
            if keys:
                full = full.drop_duplicates(subset=keys)
        if to_parquet:
            try:
                full.to_parquet(out_path)
            except Exception as e:
                raise
        else:
            full.to_csv(out_path, index=False)
        return len(full)

    # Merge-only mode: skip downloads
    if args.merge_only:
        csvs = find_csv_files()
        if not csvs:
            print('No CSVs found to merge in', out_root)
            return
        if args.verify:
            cols, summary = verify_csvs(csvs)
            print('Verification summary:')
            for s in summary:
                print(s)
        merged_basename = args.merged_name or f"{args.symbol}_trades"
        if args.merge == 'parquet':
            outp = out_root / f"{merged_basename}.parquet"
            cnt = merge_csvs(csvs, outp, to_parquet=True)
            print('Wrote', outp, 'rows=', cnt)
        else:
            outp = out_root / f"{merged_basename}.csv"
            cnt = merge_csvs(csvs, outp, to_parquet=False)
            print('Wrote', outp, 'rows=', cnt)
        return

    tasks = find_tasks()

    total = sum(sz for _, _, sz in tasks if sz)
    pbar = tqdm(total=total, unit="B", unit_scale=True, desc=args.symbol)

    q = queue.Queue()
    threads = [threading.Thread(target=worker, args=(q, pbar, args.remove_zip, not args.no_unzip), daemon=True) for _ in range(args.threads)]
    for t in threads:
        t.start()
    for it in tasks:
        q.put(it)
    q.join()
    for _ in threads:
        q.put(None)
    for t in threads:
        t.join()
    pbar.close()

    # optional post-merge/verify
    if args.merge:
        csvs = find_csv_files()
        if not csvs:
            print('No CSVs found to merge in', out_root)
            return
        if args.verify:
            cols, summary = verify_csvs(csvs)
            print('Verification summary:')
            for s in summary:
                print(s)
        merged_basename = args.merged_name or f"{args.symbol}_trades"
        if args.merge == 'parquet':
            outp = out_root / f"{merged_basename}.parquet"
            cnt = merge_csvs(csvs, outp, to_parquet=True)
            print('Wrote', outp, 'rows=', cnt)
        else:
            outp = out_root / f"{merged_basename}.csv"
            cnt = merge_csvs(csvs, outp, to_parquet=False)
            print('Wrote', outp, 'rows=', cnt)


if __name__ == "__main__":
    main()
