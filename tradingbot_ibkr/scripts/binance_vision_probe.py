#!/usr/bin/env python3
"""
Simple probe of data.binance.vision archive URLs for several symbols and date ranges.
Prints found monthly/daily archive URLs.
"""
from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests

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


def head_exists(url):
    try:
        r = requests.head(url, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def probe(symbols, ranges):
    found = {}
    for sym in symbols:
        sfound = []
        for start, end in ranges:
            for yyyy, mm in yyyymm_iter(start, end):
                murl = BASE + SPOT_MONTHLY.format(symbol=sym, yyyy=yyyy, mm=mm)
                if head_exists(murl):
                    sfound.append(murl)
                else:
                    for dd in month_days(yyyy, mm):
                        durl = BASE + SPOT_DAILY.format(symbol=sym, yyyy=yyyy, mm=mm, dd=dd)
                        if head_exists(durl):
                            sfound.append(durl)
        found[sym] = sfound
    return found


if __name__ == '__main__':
    symbols = ['BTCUSDT','ETHUSDT','BNBUSDT','ADAUSDT','BTC_USDT','btcusdt']
    ranges = [('2017-01','2017-03'),('2021-01','2021-03'),('2023-01','2023-03')]
    print('Probing Binance Vision for symbols:', symbols)
    results = probe(symbols, ranges)
    for sym, urls in results.items():
        print('---', sym, '->', len(urls), 'matches')
        for u in urls[:20]:
            print('  ', u)
    # summary
    any_found = sum(len(v) for v in results.values())
    print('\nTotal matches found:', any_found)
