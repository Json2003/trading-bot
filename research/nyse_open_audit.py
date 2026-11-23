"""Compare pinned daily-bar opens with NYSE participant end-of-day summaries."""
from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path

from research.nyse_accounting import decimal
from research.nyse_capture import calendar, captured_bars, digest
from research.nyse_offline import DataGap, entry_exclusion, read_json, timestamp


OPEN_TOLERANCE_USD = Decimal("0.0101")
PARTICIPANT_MESSAGES = {"ParticipantEODSummary", "FractionalParticipantEODSummary"}
SUMMARY_TIME = re.compile(r"_?(\d{2}):(\d{2}):(\d{2})\.(\d{0,3})")


def participant_opens(raw: bytes, session: date, tickers: set[str]) -> tuple[dict, int]:
    """Select the final participant snapshot, including a missing open in that snapshot."""
    latest = {}
    with io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig", newline="") as source:
        preamble = source.readline().split()
        if len(preamble) < 6:
            raise DataGap("missing CTA summary preamble")
        try:
            file_day = datetime.strptime(f"{preamble[1]} {preamble[2]} {preamble[5]}", "%b %d %Y").date()
        except ValueError as exc:
            raise DataGap("invalid CTA summary preamble date") from exc
        if file_day != session:
            raise DataGap(f"CTA summary is dated {file_day}, expected {session}")

        reader = csv.reader(source, strict=True)
        header = next(reader, [])
        required = ("Msg Cat/Type", "TransTime", "SeqNum", "Symbol", "instrumentType",
                    "partiId", "prevClosePriceDate", "openPrice")
        if len(set(header)) != len(header) or any(field not in header for field in required):
            raise DataGap("CTA summary header is incomplete or ambiguous")
        columns = [header.index(field) for field in required]
        last_column = max(columns)
        rows = 0
        for row in reader:
            if not row:
                continue
            if row[0].startswith("total "):
                if len(row) != 1 or row[0] != f"total {rows}" or not rows or any(reader):
                    raise DataGap("CTA summary trailer does not match the complete data file")
                return latest, rows
            rows += 1
            if len(row) <= last_column:
                raise DataGap("short CTA summary row")
            kind, label, sequence, ticker, instrument, participant, previous_close, opening = (
                row[i] for i in columns
            )
            if kind not in PARTICIPANT_MESSAGES or ticker not in tickers or instrument != "Equity" or participant != "NYSE":
                continue
            match = SUMMARY_TIME.fullmatch(label)
            sequence_digits = sequence.removeprefix("_")
            if not match or not sequence_digits.isascii() or not sequence_digits.isdecimal():
                raise DataGap(f"invalid CTA summary time or sequence for {ticker}")
            hour, minute, second, millis = match.groups()
            try:
                clock = time(int(hour), int(minute), int(second), int(millis or 0) * 1000)
            except ValueError as exc:
                raise DataGap(f"invalid CTA summary clock for {ticker}") from exc
            key = (clock, int(sequence_digits))
            if ticker in latest and key <= latest[ticker]["key"]:
                if key == latest[ticker]["key"]:
                    raise DataGap(f"duplicate CTA participant snapshot for {ticker}")
                continue
            price = decimal(opening, f"CTA open for {ticker}") if opening else Decimal(0)
            if price < 0:
                raise DataGap(f"negative CTA open for {ticker}")
            latest[ticker] = {
                "key": key, "summary_time_label": label, "summary_sequence": sequence,
                "open": price if price and previous_close in {"", "_"} else None,
            }
    raise DataGap("CTA summary is missing its completeness trailer")


def audit_open_prices(plan: dict, plan_hash: str, captures: Path, review_path: Path,
                      cta_path: Path, day: str) -> dict:
    session = date.fromisoformat(day)
    scheduled = next((row for row in calendar(plan) if row["session"] == session.isoformat()), None)
    now = datetime.now(timezone.utc)
    if not scheduled or timestamp(scheduled["close_at"]) > now:
        raise DataGap("opening-price audit needs a completed session in the frozen calendar")
    day = scheduled["session"]
    review_raw = review_path.read_bytes()
    review = read_json(review_raw)
    if review.get("schema_version") != 1 or review.get("plan_sha256") != plan_hash:
        raise DataGap("opening-price audit needs a review for the frozen capture plan")
    reviewed_at = timestamp(review["reviewed_at"])
    if (not timestamp(scheduled["close_at"]) <= reviewed_at <= now
            or (review.get("membership_complete_through") or "") < day
            or review.get("price_basis") != "nominal_ohlc"
            or not review.get("evidence") or not review.get("price_basis_evidence")):
        raise DataGap("security membership and nominal price basis must be reviewed through the audit session")
    securities = review["securities"]
    if (sorted(s["ticker"] for s in securities) != sorted(plan["symbols"])
            or len({s["security_id"] for s in securities}) != len(securities)
            or any(timestamp(s["available_at"]) > reviewed_at for s in securities)):
        raise DataGap("review must identify each frozen ticker by a distinct, observed security class")
    tickers = {s["ticker"]: s["security_id"] for s in securities if not entry_exclusion(s, day)}
    if not tickers:
        raise DataGap("the opening-price audit has no eligible NYSE common stocks")

    bars, provenance = captured_bars(plan, plan_hash, captures)
    raw = cta_path.read_bytes()
    try:
        official, rows = participant_opens(raw, session, set(tickers))
    except csv.Error as exc:
        raise DataGap(f"invalid CTA summary CSV: {exc}") from exc
    comparisons = []
    for ticker, security_id in sorted(tickers.items()):
        vendor = bars.get((ticker, day))
        exchange = official.get(ticker, {})
        daily = Decimal(vendor["open"]) if vendor else None
        venue = exchange.get("open")
        delta = venue - daily if venue is not None and daily is not None else None
        status = "missing_open" if delta is None else "different" if abs(delta) > OPEN_TOLERANCE_USD else "match"
        comparisons.append({
            "ticker": ticker, "security_id": security_id, "status": status,
            "daily_bar_open_usd": str(daily) if daily is not None else None,
            "daily_bar_received_at": vendor["available_at"] if vendor else None,
            "nyse_participant_open_usd": str(venue) if venue is not None else None,
            "cta_summary_time_label": exchange.get("summary_time_label"),
            "cta_summary_sequence": exchange.get("summary_sequence"),
            "venue_minus_daily_usd": str(delta) if delta is not None else None,
            "venue_minus_daily_bps": str((delta / daily * 10000).quantize(Decimal("0.01"))) if delta is not None else None,
        })
    counts = {status: sum(row["status"] == status for row in comparisons)
              for status in ("match", "different", "missing_open")}
    status = "opens_within_tolerance"
    if counts["missing_open"]:
        status = "incomplete_open_comparison"
    elif counts["different"]:
        status = "opens_differ"
    return {
        "schema_version": 1, "status": status, "session": day, "plan_sha256": plan_hash,
        "review_sha256": digest(review_raw), "open_tolerance_usd": str(OPEN_TOLERANCE_USD),
        "cta_summary": {
            "source_url": f"https://ftp.nyse.com/cts_summary_files/CTA.Summary.EODSUM.{session:%Y%m%d}.csv",
            "sha256": digest(raw), "file_bytes": len(raw), "data_rows": rows,
        },
        "daily_bar_receipts": [packet for packet in provenance if packet["from"] <= day <= packet["through"]],
        "counts": counts, "comparisons": comparisons, "executable_fills_verified": False,
        "strategy_trials": 0, "new_independent_observations": 0, "expected_profit": None,
    }
