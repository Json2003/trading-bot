"""Synthetic regression coverage; these fixtures are never strategy evidence."""
import csv
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone

import pytest

from scripts import run_public_flow_volatility_pocket as evaluator
from scripts.restore_public_flow_checkpoint import restore_checkpoint


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-public-flow")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with (root / "completed_minute_flow.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol", "bucket", "completed", "last_trade_price", "best_bid",
                         "best_ask", "buy_notional", "sell_notional",
                         "net_aggressive_notional", "book_imbalance"])
        for minute in range(60 * 1440):
            stamp = evaluator.iso(start + timedelta(minutes=minute))
            price = 100 + minute % 60
            for symbol in ("BTCUSDT", "ETHUSDT"):
                writer.writerow([symbol, stamp, "true", price, price - 0.01,
                                 price + 0.01, 80, 20, 60, 0.2])
    return root


def run(monkeypatch, tmp_path, data_dir):
    monkeypatch.setattr(sys, "argv", ["evaluate", "--data-dir", str(data_dir),
                                     "--output-dir", str(tmp_path), "--require-evaluation"])
    code = evaluator.main()
    return code, json.loads((tmp_path / "report.json").read_text())


def test_exact_60_days_reaches_real_evaluator(archive, tmp_path, monkeypatch):
    code, report = run(monkeypatch, tmp_path, archive)
    assert code == 0
    assert report["status"] == "evaluated"
    assert report["source_duration_days"] == 60
    assert report["duration_days"] == 58
    assert report["evaluation_from"] == "2026-01-03T00:00:00Z"
    assert len(report["blocks"]) == 6
    assert all(b["trade_count"] > 0 for b in report["blocks"])
    assert report["parameters"]["round_trip_cost_bps"] == 86
    assert not report["promotion_allowed"]


@pytest.mark.parametrize("case", ["missing", "short", "gap", "duplicate"])
def test_invalid_archives_fail_closed(case, archive, tmp_path, monkeypatch):
    rows, _ = evaluator.load_rows(archive)
    duplicates = 0
    if case == "missing":
        del rows["ETHUSDT"]
    elif case == "short":
        rows = {symbol: values[:-1] for symbol, values in rows.items()}
    elif case == "gap":
        del rows["ETHUSDT"][100]
    else:
        duplicates = 1
    monkeypatch.setattr(evaluator, "load_rows", lambda _: (rows, duplicates))
    monkeypatch.setattr(evaluator, "build_signals", lambda _: pytest.fail("Invalid archive evaluated"))
    code, report = run(monkeypatch, tmp_path, archive)
    assert code == 1
    assert report["status"] == "skip"
    assert report["segments"] == {}
    assert not report["confirmed"]


def test_development_never_uses_confirmation_exit(archive, tmp_path, monkeypatch):
    rows, _ = evaluator.load_rows(archive)
    # Later data must not extend the frozen screen or provide its exit prices.
    for values in rows.values():
        values.append({**values[-1], "time": values[-1]["time"] + evaluator.MINUTE})
    monkeypatch.setattr(evaluator, "load_rows", lambda _: (rows, 0))
    start = rows["BTCUSDT"][0]["time"]
    boundary = start + timedelta(days=32)
    end = start + timedelta(days=60)

    def signals(usable):
        assert usable[-1]["time"] == end - evaluator.MINUTE
        result = []
        for signal_time in (boundary - timedelta(minutes=32),
                            boundary - timedelta(minutes=31), boundary,
                            end - timedelta(minutes=31)):
            result.append({"signal_time": signal_time,
                           "entry_time": signal_time + evaluator.MINUTE,
                           "exit_time": signal_time + timedelta(minutes=31),
                           "entry": 100, "exit": 101, "direction": 1,
                           "symbol": usable[0]["symbol"]})
        return result

    # Inspect eligible signals before cooldown can conceal a boundary error.
    original_choose = evaluator.choose_non_overlapping
    def choose(signals):
        assert len(signals) == 4  # Two eligible signals per symbol.
        assert all(s["exit_time"] != boundary and s["exit_time"] < end for s in signals)
        return original_choose(signals)

    monkeypatch.setattr(evaluator, "build_signals", signals)
    monkeypatch.setattr(evaluator, "choose_non_overlapping", choose)
    code, report = run(monkeypatch, tmp_path, archive)
    assert code == 0
    assert report["segments"]["development"]["trade_count"] == 1


class FakeBucket:
    name = "synthetic-only"

    def __init__(self, objects):
        self.objects = objects
        self.reads = []

    def blob(self, name):
        self.reads.append(name)
        objects = self.objects
        class Blob:
            def download_as_bytes(self):
                return objects[name]
        return Blob()


def checkpoint_bucket():
    payload = b"synthetic completed minute data\n"
    checkpoint = {"continuity_status": "continuous", "inclusive_segment_id": "segment-000001",
                  "segments_completed": 1, "window_id": "frozen", "data_through_utc": "end"}
    manifest = {"segment_id": "segment-000001", "window_id": "frozen",
                "continuity_status": "initial", "observed_end_exclusive_utc": "end",
                "files": {"completed_minute_flow.csv": {
                    "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}}
    return FakeBucket({
        "prefix/checkpoints/frozen.json": json.dumps(checkpoint).encode(),
        "prefix/segments/segment-000001/segment_manifest.json": json.dumps(manifest).encode(),
        "prefix/segments/segment-000001/completed_minute_flow.csv": payload,
    })


def test_restore_reads_only_pinned_segments(tmp_path):
    bucket = checkpoint_bucket()
    result = restore_checkpoint(bucket, "prefix", "checkpoints/frozen.json", tmp_path)
    assert len(bucket.reads) == 3
    assert result["checkpoint"]["window_id"] == "frozen"
    assert (tmp_path / "segments/segment-000001/completed_minute_flow.csv").exists()


@pytest.mark.parametrize("case", ["checksum", "mutable", "mixed", "gap", "missing"])
def test_restore_rejects_invalid_archive(case, tmp_path):
    bucket = checkpoint_bucket()
    name = "checkpoints/frozen.json"
    if case == "checksum":
        bucket.objects["prefix/segments/segment-000001/completed_minute_flow.csv"] = b"changed"
    elif case == "mutable":
        name = "checkpoint.json"
    elif case == "mixed":
        (tmp_path / "old.csv").write_text("old checkpoint")
    elif case == "gap":
        key = "prefix/checkpoints/frozen.json"
        checkpoint = json.loads(bucket.objects[key])
        checkpoint["continuity_status"] = "contains-explicit-gaps-or-overlaps"
        bucket.objects[key] = json.dumps(checkpoint).encode()
    else:
        del bucket.objects["prefix/segments/segment-000001/completed_minute_flow.csv"]
    with pytest.raises((ValueError, KeyError)):
        restore_checkpoint(bucket, "prefix", name, tmp_path)


@pytest.mark.parametrize("read_only", [True, False])
def test_storage_client_uses_scoped_in_memory_credentials(monkeypatch, read_only):
    from types import ModuleType, SimpleNamespace
    from scripts.public_flow_storage import storage_client

    recorded = {}
    class Credentials:
        project_id = "synthetic-project"

        @classmethod
        def from_service_account_info(cls, info, scopes):
            recorded.update(info=info, scopes=scopes)
            return cls()

    def client(**kwargs):
        recorded.update(kwargs)
        return "synthetic-client"

    google = ModuleType("google")
    cloud = ModuleType("google.cloud")
    oauth2 = ModuleType("google.oauth2")
    cloud.storage = SimpleNamespace(Client=client)
    oauth2.service_account = SimpleNamespace(Credentials=Credentials)
    for name, module in (("google", google), ("google.cloud", cloud), ("google.oauth2", oauth2)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("PUBLIC_FLOW_GCS_CREDENTIALS", '{"synthetic": true}')
    assert storage_client(read_only=read_only) == "synthetic-client"
    suffix = "read_only" if read_only else "read_write"
    assert recorded["scopes"] == [f"https://www.googleapis.com/auth/devstorage.{suffix}"]
    assert recorded["project"] == "synthetic-project"
