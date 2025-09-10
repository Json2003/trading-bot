#!/usr/bin/env python3
import os, sys, time, json, argparse, datetime as dt
# Ensure repo modules are importable when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backtest.engine import ExecConfig, run_backtest
from backtest.exchange import fetch_ccxt
import importlib
import pandas as pd

def load_strategy(spec):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)

def now_utc():
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=os.getenv("ENV","paper"))
    ap.add_argument("--artifacts", default=os.getenv("ARTIFACTS_DIR","artifacts"))
    args = ap.parse_args()
    os.makedirs(args.artifacts, exist_ok=True)

    exchange = os.getenv("CRYPTO_EXCHANGE","kucoin")
    symbols  = [s.strip() for s in os.getenv("CRYPTO_SYMBOLS","BTC/USDT").split(",")]
    timeframe= os.getenv("CRYPTO_TIMEFRAME","4h")

    strat_spec = os.getenv("CRYPTO_STRATEGY")
    strat_args = {}
    SARGS = os.getenv("CRYPTO_STRATEGY_ARGS","")
    if SARGS:
        for kv in SARGS.split(","):
            k,v = kv.split("="); v=v.strip()
            if v.replace(".","",1).lstrip("-").isdigit():
                strat_args[k.strip()] = float(v) if "." in v or "-" in v else int(v)
            else:
                strat_args[k.strip()] = v
    strat = load_strategy(strat_spec)
    def signals(df): return strat(df, **strat_args)

    cfg = ExecConfig(
        fees_bps=float(os.getenv("CRYPTO_FEES_BPS",10)),
        slip_bps=float(os.getenv("CRYPTO_SLIP_BPS",5)),
        tp_bps=0, sl_bps=0,
        tp_atr_mult=float(os.getenv("CRYPTO_TP_ATR",3.0)),
        sl_atr_mult=float(os.getenv("CRYPTO_SL_ATR",1.5)),
        atr_period=int(os.getenv("CRYPTO_ATR_PERIOD",14)),
        risk_per_trade=float(os.getenv("CRYPTO_RISK_PER_TRADE",0.005)),
        max_notional_frac=1.0,
        allow_short=bool(int(os.getenv("CRYPTO_ALLOW_SHORT","1"))),
        max_bars=int(os.getenv("CRYPTO_MAX_BARS",12)),
    )

    # Path to superbot active config for hot-loading
    active_cfg_path = os.path.join(args.artifacts, "superbot", "active_config.json")

    while True:
        t0 = now_utc()
        all_trades = []
        eq_rows = []
        # Hot-load superbot active config if present
        try:
            if os.path.exists(active_cfg_path):
                with open(active_cfg_path) as f:
                    active = json.load(f)
                # Update strategy if changed
                new_spec = active.get("strat_path") or strat_spec
                new_args = active.get("strat_args") or strat_args
                new_exit = active.get("exit_args") or {}
                if new_spec != strat_spec or new_args != strat_args:
                    strat_spec = new_spec
                    strat_args = new_args
                    strat = load_strategy(strat_spec)
                    def signals(df): return strat(df, **strat_args)
                # Update execution exits
                cfg.tp_atr_mult = float(new_exit.get("tp_atr_mult", cfg.tp_atr_mult))
                cfg.sl_atr_mult = float(new_exit.get("sl_atr_mult", cfg.sl_atr_mult))
                cfg.max_bars = int(new_exit.get("max_bars", cfg.max_bars))
        except Exception as e:
            # Non-fatal; log minimal and continue
            pass

        for sym in symbols:
            # pull 1800 bars to have full context
            since = (t0 - dt.timedelta(days=365)).strftime("%Y-%m-%d")
            df = fetch_ccxt(exchange, sym, timeframe, since, t0.strftime("%Y-%m-%d"))
            if df.empty: continue
            trades, equity, bar_ret = run_backtest(df, signals, cfg)
            # last trade summary
            meta = {
              "ts": t0.isoformat(), "symbol": sym,
              "env": args.env, "timeframe": timeframe,
              "last_equity": float(equity["equity"].iloc[-1]),
              "num_trades": int(len(trades)),
            }
            all_trades.append({"symbol": sym, "trades": trades.tail(3).to_dict(orient="records"), "meta": meta})
            eq_rows.append({"symbol": sym, "equity": meta["last_equity"]})

        # write heartbeat
        hb_path = os.path.join(args.artifacts, "crypto_heartbeat.json")
        with open(hb_path, "w") as f:
            json.dump({
                "ts": now_utc().isoformat(),
                "env": args.env,
                "equities": eq_rows,
                "last_trades": all_trades,
            }, f, indent=2)

        # TODO: place orders if ENV=live (wire your exchange client)
        # sleep close to one bar length; set 5 minutes for 4h bars
        time.sleep(300)

if __name__ == "__main__":
    main()
