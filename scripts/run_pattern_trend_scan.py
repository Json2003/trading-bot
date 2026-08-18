#!/usr/bin/env python3
"""Research-only mathematical pattern/trend scan.

The pattern universe is fixed before evaluation. Signals use closed candles and
are sampled every holding horizon so forward labels do not overlap.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from statistics import median
from typing import Any

try:
    from scripts.run_momentum_volatility_research import load_bars
    from scripts.run_momentum_volatility_v3 import align_pair
except ModuleNotFoundError:
    from run_momentum_volatility_research import load_bars
    from run_momentum_volatility_v3 import align_pair

BASE_COST, STRESS_COST = 0.0038, 0.0086
HORIZONS, BLOCKS, MIN_PER_BLOCK = (6, 24, 48, 96), 6, 20


def mean(x):
    return sum(x) / len(x) if x else math.nan


def feat(bars):
    c = [float(x.close) for x in bars]
    h = [float(x.high) for x in bars]
    l = [float(x.low) for x in bars]
    r = [(h[i] - l[i]) / c[i] if c[i] else math.nan for i in range(len(c))]
    out = {"c": c, "h": h, "l": l, "r": r}
    for n in (6, 24, 48, 96, 200):
        out[f"m{n}"] = [math.nan if i < n else c[i] / c[i-n] - 1 for i in range(len(c))]
        out[f"avg_r{n}"] = [mean(r[i-n+1:i+1]) if i >= n-1 else math.nan for i in range(len(c))]
        out[f"avg_c{n}"] = [mean(c[i-n+1:i+1]) if i >= n-1 else math.nan for i in range(len(c))]
    out["high48"] = [max(h[i-48:i]) if i >= 49 else math.nan for i in range(len(c))]
    out["low48"] = [min(l[i-48:i]) if i >= 49 else math.nan for i in range(len(c))]
    return out


def ok(*x):
    return all(math.isfinite(float(v)) for v in x)


def choose(name, i, b, e):
    assets = (("BTC", b), ("ETH", e))
    if i < 220:
        return None, 0
    if name == "relative_strength":
        if not ok(b["m24"][i], e["m24"][i]):
            return None, 0
        return ("BTC", 1) if b["m24"][i] > e["m24"][i] + .01 else (("ETH", 1) if e["m24"][i] > b["m24"][i] + .01 else (None, 0))
    eligible = []
    for s, x in assets:
        if not ok(x["c"][i], x["m24"][i], x["m48"][i], x["m96"][i], x["avg_r24"][i]):
            continue
        trend = x["c"][i] > x["avg_c200"][i] and x["m24"][i] > 0
        breakout = ok(x["high48"][i], x["low48"][i]) and (x["c"][i] > x["high48"][i] or x["c"][i] < x["low48"][i])
        expansion = x["r"][i] > 1.25 * x["avg_r24"][i]
        contraction = x["r"][i] < .75 * x["avg_r24"][i]
        if name == "trend_structure" and trend and x["m48"][i] > x["m24"][i]:
            eligible.append((x["m24"][i], s, 1))
        elif name == "breakout" and breakout and trend:
            eligible.append((abs(x["m24"][i]), s, 1 if x["m24"][i] > 0 else -1))
        elif name == "vol_expansion" and expansion and trend:
            eligible.append((x["m24"][i], s, 1))
        elif name == "vol_contraction_break" and contraction and breakout:
            eligible.append((abs(x["m24"][i]), s, 1 if x["m24"][i] > 0 else -1))
        elif name == "momentum_decay" and trend and x["m48"][i] > 0 and x["m24"][i] < x["m48"][i] * .5:
            eligible.append((x["m24"][i], s, 1))
        elif name == "mean_reversion" and x["m24"][i] < -.03:
            eligible.append((abs(x["m24"][i]), s, 1))
        elif name == "exhaustion" and x["m48"][i] > .15 and x["m24"][i] < 0:
            eligible.append((abs(x["m24"][i]), s, -1))
    if not eligible:
        return None, 0
    _, s, direction = max(eligible)
    return s, direction


def scan(btc_path: Path, eth_path: Path):
    pair = align_pair(load_bars(btc_path), load_bars(eth_path))
    pair = pair[-3 * 365 * 24:]
    if len(pair) < 3 * 365 * 24:
        raise ValueError("three years of aligned candles required")
    bars = {"BTC": [x.btc for x in pair], "ETH": [x.eth for x in pair]}
    f = {s: feat(v) for s, v in bars.items()}
    names = ("trend_structure", "breakout", "vol_expansion", "vol_contraction_break",
             "relative_strength", "momentum_decay", "mean_reversion", "exhaustion")
    results = {}
    for name in names:
        results[name] = {}
        for horizon in HORIZONS:
            rows = []
            for i in range(220, len(pair) - horizon, horizon):
                symbol, direction = choose(name, i, f["BTC"], f["ETH"])
                if symbol:
                    x = f[symbol]
                    rows.append((i, direction * (x["c"][i+horizon] / x["c"][i] - 1)))
            blocks = [[] for _ in range(BLOCKS)]
            for i, value in rows:
                blocks[min(BLOCKS - 1, int(i / (len(pair) / BLOCKS)))].append(value)
            means = [mean(x) for x in blocks]
            valid = [x for x in means if math.isfinite(x)]
            med = median(valid) if valid else math.nan
            results[name][str(horizon)] = {
                "observations": len(rows), "non_overlapping": True,
                "block_mean_gross_returns": means,
                "median_block_gross_return": med if math.isfinite(med) else None,
                "base_cost_multiple": med / BASE_COST if math.isfinite(med) else None,
                "stress_cost_multiple": med / STRESS_COST if math.isfinite(med) else None,
                "positive_blocks": sum(x > 0 for x in valid),
                "passes_4x_stress_gate": bool(len(rows) >= BLOCKS * MIN_PER_BLOCK and len(valid) == BLOCKS and all(x > 0 for x in valid) and med / STRESS_COST >= 4),
            }
    passing = [f"{n}:{h}" for n, hs in results.items() for h, v in hs.items() if v["passes_4x_stress_gate"]]
    return {"schema_version": 1, "research_only": True, "orders_placed": False,
            "leverage_enabled": False, "window": {"start": pair[0].timestamp.isoformat(), "end": pair[-1].timestamp.isoformat(), "bars": len(pair), "blocks": BLOCKS},
            "costs": {"base_round_trip": BASE_COST, "stress_round_trip": STRESS_COST},
            "multiple_testing_note": "Pattern universe fixed before evaluation; candidates require all six positive blocks and 4x stress cost multiple.",
            "patterns": results, "passing_patterns": passing,
            "status": "candidate_found" if passing else "no_candidate_passed"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--btc-path", type=Path, required=True)
    p.add_argument("--eth-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = scan(a.btc_path, a.eth_path)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "passing_patterns": report["passing_patterns"]}, indent=2))


if __name__ == "__main__":
    main()
