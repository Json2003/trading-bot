"""Run stress tests over annotated bars and produce per-event metrics.

Outputs a JSON report with per-event returns, max drawdown, volatility, and a short summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
try:  # pragma: no cover - optional for tests
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None
try:  # pragma: no cover
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None


def compute_metrics(df: pd.DataFrame):
    if pd is None or np is None:  # pragma: no cover - function unused in tests
        raise RuntimeError("pandas and numpy are required")
    # expects df sorted by ts
    df = df.sort_values("ts").reset_index(drop=True)
    df["ret"] = df["close"].pct_change().fillna(0)
    cum_ret = (1 + df["ret"]).cumprod() - 1
    total_return = cum_ret.iloc[-1]

    peak = (1 + df["ret"]).cumprod().cummax()
    trough = (1 + df["ret"]).cumprod()
    dd = trough / peak - 1
    max_dd = dd.min()
    vol = df["ret"].std() * np.sqrt(252)

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "volatility_annualized": float(vol),
        "n_bars": int(len(df)),
    }


def run_report(annotated_csv: Path, out_json: Path):
    if pd is None:
        raise RuntimeError("pandas is required")
    df = pd.read_csv(annotated_csv)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
    else:
        df["ts"] = pd.to_datetime(df.iloc[:, 0])

    events = [None] + sorted(df["event_name"].dropna().unique().tolist())
    report = {"events": []}

    # overall
    overall = compute_metrics(df)
    report["overall"] = overall

    for ev in df["event_name"].dropna().unique():
        ev_df = df[df["event_name"] == ev]
        # split by window
        for window in ["pre", "shock", "recovery"]:
            w_df = ev_df[ev_df["window"] == window]
            if len(w_df) < 2:
                continue
            m = compute_metrics(w_df)
            report["events"].append({
                "event": ev,
                "window": window,
                "metrics": m,
            })

    with open(out_json, "w", encoding="utf8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote stress report to {out_json}")


def main():
    p = argparse.ArgumentParser(description="Run stress test on annotated bars CSV")
    p.add_argument("--annotated", required=True, help="annotated CSV path produced by event_labeler")
    p.add_argument("--out", default="tradingbot_ibkr/datafiles/stress_report.json", help="output JSON report path")
    args = p.parse_args()

    run_report(Path(args.annotated), Path(args.out))


if __name__ == "__main__":
    main()
