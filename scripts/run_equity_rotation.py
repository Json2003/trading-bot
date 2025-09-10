#!/usr/bin/env python3
"""Minimal equity rotation CLI placeholder.

Accepts the flags used by start_equity.py and writes a small metrics JSON
to artifacts, so the runtime can produce a heartbeat. Replace with your real
rotation logic when ready.
"""
import argparse, os, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--top_n', type=int, default=3)
    ap.add_argument('--value_weight', type=float, default=0.3)
    ap.add_argument('--mom_weight', type=float, default=0.7)
    ap.add_argument('--rebalance_day', type=int, default=1)
    ap.add_argument('--slippage_bps', type=float, default=7.0)
    ap.add_argument('--commission_bps', type=float, default=2.0)
    ap.add_argument('--leverage', type=float, default=1.0)
    ap.add_argument('--out_prefix', required=True)
    ap.add_argument('--tickers', default=None)
    ap.add_argument('--no_sma200', action='store_true')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    metrics = {
        'start': args.start,
        'end': args.end,
        'top_n': args.top_n,
        'value_weight': args.value_weight,
        'mom_weight': args.mom_weight,
        'rebalance_day': args.rebalance_day,
        'slippage_bps': args.slippage_bps,
        'commission_bps': args.commission_bps,
        'leverage': args.leverage,
        'tickers': args.tickers,
        'sma200': not args.no_sma200,
        'total_return': 0.0,
    }
    with open(f"{args.out_prefix}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    # also write a tiny equity heartbeat payload
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
