"""Summarize past trades and produce per-model diagnostics.

Reads `tradingbot_ibkr/datafiles/trades.csv` or `tradingbot_ibkr/datafiles/trades.jsonl` if present.
Writes `tradingbot_ibkr/datafiles/self_eval_report.json` and per-model-version CSVs.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import notifier


def read_trade_log(data_dir: Path):
    csv = data_dir / "trades.csv"
    jsonl = data_dir / "trades.jsonl"
    if csv.exists():
        df = pd.read_csv(csv)
        return df
    if jsonl.exists():
        df = pd.read_json(jsonl, lines=True)
        return df
    return None


def compute_trade_metrics(df: pd.DataFrame):
    # Ensure numeric types
    df = df.copy()
    for c in ["entry_price", "exit_price", "size", "pnl"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Basic columns required
    if not {"entry_ts", "exit_ts"}.issubset(df.columns):
        # fallback to index or timestamp
        df["entry_ts"] = pd.to_datetime(df.get("entry_ts", df.index))
        df["exit_ts"] = pd.to_datetime(df.get("exit_ts", df["entry_ts"]))
    else:
        df["entry_ts"] = pd.to_datetime(df["entry_ts"]).dt.tz_localize(None)
        df["exit_ts"] = pd.to_datetime(df["exit_ts"]).dt.tz_localize(None)

    # outcome class
    if "pnl" in df.columns:
        df["outcome"] = np.where(df["pnl"] > 0, "win", np.where(df["pnl"] < 0, "loss", "break_even"))
    else:
        df["outcome"] = "unknown"

    df["hold_time_s"] = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds()
    return df


def summarize(df: pd.DataFrame):
    summary = {}
    summary["n_trades"] = int(len(df))
    if "pnl" in df.columns:
        summary["total_pnl"] = float(df["pnl"].sum())
        summary["avg_pnl"] = float(df["pnl"].mean())
    summary["win_rate"] = float((df["outcome"] == "win").mean()) if len(df) else None
    summary["avg_hold_time_s"] = float(df["hold_time_s"].mean()) if len(df) else None
    return summary


def per_model_tables(df: pd.DataFrame, out_dir: Path):
    if "model_version" not in df.columns:
        return []
    tables = []
    for mv, g in df.groupby("model_version"):
        out = out_dir / f"trades_model_{mv}.csv"
        g.to_csv(out, index=False)
        tables.append(str(out))
    return tables


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="tradingbot_ibkr/datafiles", help="datafiles dir")
    p.add_argument("--out", default="tradingbot_ibkr/datafiles/self_eval_report.json", help="output JSON report")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    df = read_trade_log(data_dir)
    if df is None:
        print("No trade log found. Please write trades to tradingbot_ibkr/datafiles/trades.csv or trades.jsonl with fields: entry_ts, exit_ts, entry_price, exit_price, size, pnl, model_version, entry_reason, exit_reason")
        return

    df = compute_trade_metrics(df)
    report = summarize(df)
    report["per_model_files"] = per_model_tables(df, data_dir)

    # persist historical summaries
    hist_path = Path(data_dir) / 'self_eval_history.json'
    history = []
    if hist_path.exists():
        try:
            history = json.loads(hist_path.read_text())
        except Exception:
            history = []
    entry = {"ts": pd.Timestamp.now().isoformat(), "summary": report}
    history.append(entry)
    hist_path.write_text(json.dumps(history, indent=2))

    # alerting: compare to previous summary (if exists)
    alerts = []
    if len(history) >= 2:
        prev = history[-2]['summary']
        # check win_rate degradation > 5 percentage points
        if prev.get('win_rate') is not None and report.get('win_rate') is not None:
            if (prev['win_rate'] - report['win_rate']) > 0.05:
                alerts.append({"metric": "win_rate", "prev": prev['win_rate'], "now": report['win_rate'], "msg": "win_rate dropped >5pp"})
        # check avg_pnl drop > 20%
        if prev.get('avg_pnl') is not None and report.get('avg_pnl') is not None and prev['avg_pnl'] != 0:
            if (prev['avg_pnl'] - report['avg_pnl']) / abs(prev['avg_pnl']) > 0.2:
                alerts.append({"metric": "avg_pnl", "prev": prev['avg_pnl'], "now": report['avg_pnl'], "msg": "avg_pnl dropped >20%"})

    alerts_path = Path(data_dir) / 'self_eval_alerts.json'
    if alerts:
        alerts_path.write_text(json.dumps(alerts, indent=2))
        print(f"ALERTS written to {alerts_path}")
        try:
            ok = notifier.notify(alerts)
            print('Notifier sent:', ok)
        except Exception as e:
            print('Notifier error:', e)

    with open(args.out, "w", encoding="utf8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote self-eval report to {args.out}")


if __name__ == "__main__":
    main()
