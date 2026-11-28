"""Count NYSE common-stock opening messages in a public Pillar Trades sample."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path


def audit(path: Path) -> dict:
    raw = path.read_bytes()
    counts = Counter()
    listings = {}
    messages = []
    with gzip.open(io.BytesIO(raw), "rt", encoding="ascii", newline="") as source:
        for row in csv.reader(source):
            if not row:
                continue
            kind = row[0]
            counts[kind] += 1
            if kind == "3":
                market_and_class = tuple(row[3:4] + row[5:7])
                if len(row) != 14 or listings.setdefault(row[2], market_and_class) != market_and_class:
                    raise ValueError(f"inconsistent listing map for {row[2]}")
            elif kind == "220":
                if len(row) != 12:
                    raise ValueError("unexpected Trade message layout")
                if row[9] == "O" or row[11] == "Q":
                    messages.append(row)
            elif kind != "34":
                raise ValueError(f"unsupported message {kind}; inspect corrections before interpreting opens")
    if not counts["3"] or not counts["220"]:
        raise ValueError("sample needs symbol mappings and trade messages")

    common = {symbol for symbol, attributes in listings.items() if attributes == ("1", "N", "C")}
    opening = {}
    official = {}
    for row in messages:
        if row[3] not in listings:
            raise ValueError(f"missing listing mapping for {row[3]}")
        if row[3] not in common:
            continue
        event = (row[3], row[2], Decimal(row[6]), int(row[7]))
        keyed = opening if row[9] == "O" else official
        if event in keyed:
            raise ValueError(f"duplicate opening message for {event}")
        keyed[event] = {"symbol": row[3], "source_time_et": row[2], "trade_id": row[5],
                        "price_usd": row[6], "reported_shares": int(row[7]),
                        "condition_2": row[9], "condition_4": row[11]}
    paired = opening.keys() & official.keys()
    sample_pair = min(paired, key=lambda event: (event[1], event[0])) if paired else None
    times = sorted(row["source_time_et"] for row in opening.values())
    return {
        "sha256": hashlib.sha256(raw).hexdigest(), "file_bytes": len(raw),
        "message_counts": dict(counts), "nyse_listed_common_stock_mappings": len(common),
        "opening_trade_messages": len(opening), "official_open_messages": len(official),
        "opening_stock_count": len({event[0] for event in opening}),
        "official_open_stock_count": len({event[0] for event in official}),
        "opening_trade_messages_after_0930": sum(ts > "09:30:00.000000000" for ts in times),
        "first_opening_trade_time_et": times[0] if times else None,
        "last_opening_trade_time_et": times[-1] if times else None,
        "identical_open_and_official_price_time_size_pairs": len(paired),
        "official_open_messages_without_exact_opening_trade_pair": len(official) - len(paired),
        "official_open_messages_with_zero_trade_id": sum(row["trade_id"] == "0" for row in official.values()),
        "first_exact_pair": {"opening_trade": opening[sample_pair], "official_open": official[sample_pair]}
        if sample_pair else None,
        "execution_verified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trades_file", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.trades_file), sort_keys=True, indent=2))
