#!/usr/bin/env python3
import os, time, json, subprocess, sys, datetime as dt

def run_once():
    cmd = [
      sys.executable, "scripts/run_equity_rotation.py",
      "--start", os.getenv("EQUITY_START","2015-01-01"),
      "--end",   os.getenv("EQUITY_END","2025-01-01"),
      "--top_n", os.getenv("EQUITY_TOP_N","3"),
      "--value_weight", os.getenv("EQUITY_VALUE_W","0.3"),
      "--mom_weight", os.getenv("EQUITY_MOM_W","0.7"),
      "--rebalance_day", os.getenv("EQUITY_REBAL_DAY","1"),
      "--slippage_bps", os.getenv("EQUITY_SLIPPAGE_BPS","7"),
      "--commission_bps", os.getenv("EQUITY_COMMISSION_BPS","2"),
      "--leverage", os.getenv("EQUITY_LEVERAGE","1.5"),
      "--out_prefix", os.path.join(os.getenv("ARTIFACTS_DIR","artifacts"), "equity_rotation_financials")
    ]
    if os.getenv("EQUITY_SMA200","1") == "0": cmd.append("--no_sma200")
    if os.getenv("EQUITY_TICKERS"):
        cmd.extend(["--tickers", os.getenv("EQUITY_TICKERS")])
    print("Running:", " ".join(cmd)); sys.stdout.flush()
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
    print(out)
    # heartbeat
    with open(os.path.join(os.getenv("ARTIFACTS_DIR","artifacts"), "equity_heartbeat.json"), "w") as f:
        f.write(out)

def main():
    # run once on start
    run_once()
    # then sleep ~1 day; you can switch to cron
    while True:
        time.sleep(24*3600)
        # To avoid hammering Yahoo, only run near month boundaries or weekly:
        if dt.datetime.utcnow().day in (1,2,3,15): run_once()

if __name__ == "__main__":
    main()
