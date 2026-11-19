"""Capture and validate public daily-bar receipts for prospective NYSE research."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from research.nyse_accounting import positive, whole
from research.nyse_offline import DataGap, read_json, timestamp

DOLT_API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/stocks/master"
NY = ZoneInfo("America/New_York")
FIELDS = ("open", "high", "low", "close", "volume")


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save_new(path: Path, raw: bytes) -> None:
    """Do not replace an earlier receipt, including during a concurrent retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as temporary:
            temporary.write(raw)
        try:
            os.link(name, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise DataGap(f"immutable output already exists at {path}") from None
    finally:
        Path(name).unlink()


def calendar(plan: dict) -> list[dict]:
    start = date.fromisoformat(plan["history_start"])
    end = date.fromisoformat(plan["calendar_through"])
    closed, early = set(plan["closed_sessions"]), set(plan["early_closes"])
    days = []
    while start <= end:
        if start.weekday() < 5 and start.isoformat() not in closed:
            days.append(start)
        start += timedelta(days=1)
    decision = time.fromisoformat(plan["decision_time_next_session_et"])
    if not days or decision >= time(9, 30):
        raise DataGap("decision cutoff must precede the next 09:30 Eastern open")
    following_day = days[-1] + timedelta(days=1)
    while following_day.weekday() >= 5 or following_day.isoformat() in closed:
        following_day += timedelta(days=1)
    rows = []
    for day, following in zip(days, days[1:] + [following_day]):
        session = day.isoformat()
        rows.append({"session": session, "open_at": datetime.combine(day, time(9, 30), NY).isoformat(),
                     "close_at": datetime.combine(day, time(13 if session in early else 16), NY).isoformat(),
                     "decision_at": datetime.combine(following, decision, NY).isoformat()})
    return rows


def fetch(sql: str) -> dict:
    url = DOLT_API + "?" + urlencode({"q": sql})
    with urlopen(url, timeout=30) as response:
        raw = response.read()
    receipt = {"url": url, "received_at": datetime.now(timezone.utc).isoformat(),
               "sha256": digest(raw), "body": raw.decode()}
    rows(receipt)
    return receipt


def rows(receipt: dict) -> list[dict]:
    raw = receipt["body"].encode()
    if digest(raw) != receipt["sha256"]:
        raise DataGap("source receipt body hash mismatch")
    response = read_json(raw)
    if response.get("query_execution_status") != "Success":
        raise DataGap(f"source query failed: {response.get('query_execution_message')}")
    return response["rows"]


def capture(plan: dict, plan_hash: str, start: str, through: str, directory: Path) -> Path:
    sessions = {row["session"]: row for row in calendar(plan)}
    if start not in sessions or through not in sessions or start > through:
        raise DataGap("capture endpoints must be ordered sessions in the frozen calendar")
    wanted = [s for s in sessions if start <= s <= through]
    if timestamp(sessions[through]["close_at"]) > datetime.now(timezone.utc):
        raise DataGap(f"market session {through} has not closed")
    symbols = plan["symbols"]
    if not symbols or len(set(symbols)) != len(symbols) or any(not re.fullmatch(r"[A-Z]{1,8}", s) for s in symbols):
        raise DataGap("capture plan requires distinct uppercase ticker strings")
    top = fetch("SELECT commit_hash,date,message FROM dolt_log LIMIT 1")
    revision = rows(top)[0]["commit_hash"]
    if not re.fullmatch(r"[0-9a-v]{32}", revision):
        raise DataGap("unexpected Dolt revision format")
    path = directory / f"{start}_{through}_{revision}.json"
    if path.exists():
        packet = read_json(path.read_bytes())
        if packet["plan_sha256"] != plan_hash:
            raise DataGap("existing receipt belongs to a different plan")
        return path
    ticker_sql = ",".join(f"'{s}'" for s in symbols)
    # Point lookups on the provider's (date, act_symbol) index avoid scans.
    def fetch_days(days: list[str]) -> list[dict]:
        day_sql = ",".join(f"'{day}'" for day in days)
        sql = (f"SELECT act_symbol,date,open,high,low,close,volume FROM ohlcv AS OF '{revision}' "
               f"WHERE date IN ({day_sql}) AND act_symbol IN ({ticker_sql}) ORDER BY date,act_symbol")
        try:
            return [fetch(sql)]
        except (TimeoutError, DataGap):
            if len(days) == 1:
                raise
            middle = len(days) // 2
            return fetch_days(days[:middle]) + fetch_days(days[middle:])

    batches = [wanted[i:i + 7] for i in range(0, len(wanted), 7)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = [receipt for group in pool.map(fetch_days, batches) for receipt in group]
    packet = {"schema_version": 1, "plan_sha256": plan_hash, "revision": revision,
              "from": start, "through": through, "head": top, "partitions": receipts}
    # Let the first concurrent capture retain its genuine receipt times.
    try:
        save_new(path, encode(packet))
    except DataGap:
        if read_json(path.read_bytes())["plan_sha256"] != plan_hash:
            raise
    return path


def captured_bars(plan: dict, plan_hash: str, directory: Path) -> tuple[dict, list[dict]]:
    sessions = {row["session"]: row for row in calendar(plan)}
    bars, provenance = {}, []
    for path in sorted(directory.glob("*.json")):
        raw = path.read_bytes()
        packet = read_json(raw)
        if packet.get("schema_version") != 1 or packet["plan_sha256"] != plan_hash:
            raise DataGap(f"receipt {path.name} does not match the frozen plan")
        revision = packet["revision"]
        if rows(packet["head"])[0]["commit_hash"] != revision:
            raise DataGap(f"receipt {path.name} revision mismatch")
        for part in packet["partitions"]:
            received_at = timestamp(part["received_at"])
            if received_at > datetime.now(timezone.utc):
                raise DataGap("source receipt lies in the future")
            for row in rows(part):
                key = row["act_symbol"], row["date"]
                if key[0] not in plan["symbols"] or key[1] not in sessions or not packet["from"] <= key[1] <= packet["through"]:
                    raise DataGap(f"unexpected capture row {key}")
                if received_at < timestamp(sessions[key[1]]["close_at"]):
                    raise DataGap(f"source row precedes the exchange close {key}")
                o, h, low, c, volume = (positive(row[field], f"{key} {field}") for field in FIELDS)
                if not (low <= min(o, c) <= max(o, c) <= h) or volume != whole(volume):
                    raise DataGap(f"inconsistent raw market bar {key}")
                current = {field: str(Decimal(row[field])) for field in FIELDS}
                if key in bars:
                    if any(Decimal(bars[key][field]) != Decimal(current[field]) for field in FIELDS):
                        raise DataGap(f"source revised captured bar {key}; reconcile before replay")
                    if timestamp(bars[key]["available_at"]) <= received_at:
                        continue
                bars[key] = {**current, "available_at": part["received_at"]}
        provenance.append({"file": path.name, "sha256": digest(raw), "revision": revision,
                           "from": packet["from"], "through": packet["through"],
                           "partitions": [{k: r[k] for k in ("url", "received_at", "sha256")}
                                          for r in packet["partitions"]]})
    return bars, provenance


def receipt_index(path: Path) -> dict:
    """Keep source URLs and observed timestamps where the raw files need not be shared."""
    raw = path.read_bytes()
    packet = read_json(raw)
    for part in packet["partitions"]:
        rows(part)
        del part["body"]
    return {"packet_sha256": digest(raw), "packet": packet}


def restore_capture(index: dict, plan_hash: str, directory: Path) -> Path:
    packet = index["packet"]
    if packet["plan_sha256"] != plan_hash:
        raise DataGap("receipt index belongs to a different plan")
    revision = packet["revision"]
    if not re.fullmatch(r"[0-9a-v]{32}", revision):
        raise DataGap("unexpected revision in receipt index")
    start = date.fromisoformat(packet["from"]).isoformat()
    end = date.fromisoformat(packet["through"]).isoformat()
    path = directory / f"{start}_{end}_{revision}.json"
    if path.exists():
        if digest(path.read_bytes()) != index["packet_sha256"]:
            raise DataGap("local capture differs from the recorded receipt")
        return path

    def restore(part: dict) -> dict:
        if not part["url"].startswith(DOLT_API + "?q="):
            raise DataGap("receipt index references an unexpected provider")
        with urlopen(part["url"], timeout=30) as source:
            body = source.read()
        if digest(body) != part["sha256"]:
            raise DataGap("provider response no longer matches the recorded receipt")
        return {**part, "body": body.decode()}

    with ThreadPoolExecutor(max_workers=3) as pool:
        restored = {**packet, "partitions": list(pool.map(restore, packet["partitions"]))}
    raw = encode(restored)
    if digest(raw) != index["packet_sha256"]:
        raise DataGap("reconstructed capture does not match the recorded receipt")
    save_new(path, raw)
    return path
