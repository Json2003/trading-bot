"""Label OHLCV bars with event windows from events.json

Produces an annotated CSV with columns: ts, open, high, low, close, volume, event_name, window
where window is one of pre, shock, recovery, or none.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


def load_events(path: Path):
    with open(path, "r", encoding="utf8") as f:
        events = json.load(f)
    # normalize dates
    for e in events:
        e["shock_start"] = pd.to_datetime(e["shock_start"]).tz_localize(None)
        e["shock_end"] = pd.to_datetime(e["shock_end"]).tz_localize(None)
    return events


def annotate(df: pd.DataFrame, events: list[dict]):
    df = df.copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
    elif "timestamp" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp"])
    else:
        # assume index is timestamp
        df = df.reset_index()
        df["ts"] = pd.to_datetime(df.iloc[:, 0])

    df["event_name"] = None
    df["window"] = "none"

    for e in events:
        name = e.get("name")
        start = e["shock_start"] - timedelta(days=int(e.get("pre_days", 30)))
        shock_start = e["shock_start"]
        shock_end = e["shock_end"]
        recovery_end = shock_end + timedelta(days=int(e.get("recovery_days", 180)))

        pre_mask = (df["ts"] >= start) & (df["ts"] < shock_start)
        shock_mask = (df["ts"] >= shock_start) & (df["ts"] <= shock_end)
        recovery_mask = (df["ts"] > shock_end) & (df["ts"] <= recovery_end)

        df.loc[pre_mask, "event_name"] = name
        df.loc[pre_mask, "window"] = "pre"
        df.loc[shock_mask, "event_name"] = name
        df.loc[shock_mask, "window"] = "shock"
        df.loc[recovery_mask, "event_name"] = name
        df.loc[recovery_mask, "window"] = "recovery"

    return df


def main():
    p = argparse.ArgumentParser(description="Annotate bars CSV with event windows")
    p.add_argument("--bars", required=True, help="input OHLCV CSV path")
    p.add_argument("--events", default="tradingbot_ibkr/events.json", help="events.json path")
    p.add_argument("--out", default=None, help="output annotated CSV path")
    args = p.parse_args()

    bars_path = Path(args.bars)
    events_path = Path(args.events)

    df = pd.read_csv(bars_path)
    events = load_events(events_path)
    annotated = annotate(df, events)

    out_path = Path(args.out) if args.out else bars_path.with_name(bars_path.stem + "_annotated.csv")
    annotated.to_csv(out_path, index=False)
    print(f"Wrote annotated CSV to {out_path}")


if __name__ == "__main__":
    main()
