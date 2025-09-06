#!/usr/bin/env python3
"""
Compute the total size of Binance public market data hosted on https://data.binance.vision

Features
- Recursively crawls folders under a given prefix (default: data/)
- Sums file sizes; prints human-friendly totals
- Works with both S3-style XML listings and HTML directory pages (fallback)
- Respects pagination (S3 continuation tokens or "Next" links in HTML)
- Optional subset: spot/, futures/um/, futures/cm/, options/

Usage examples
- All data (default):         python binance_vision_size.py
- Spot only:                  python binance_vision_size.py --prefix data/spot/
- USD-M futures only:         python binance_vision_size.py --prefix data/futures/um/
- COIN-M futures only:        python binance_vision_size.py --prefix data/futures/cm/
- Options only:               python binance_vision_size.py --prefix data/options/
"""

import argparse
import time
import math
import sys
import os
from importlib import import_module
from urllib.parse import urljoin, urlencode, urlparse, parse_qs
import xml.etree.ElementTree as ET
from collections import defaultdict, deque

# Ensure the real 'requests' package is used even if a local stub exists.
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir in sys.path:
    sys.path.remove(this_dir)
try:
    requests = import_module("requests")
    from requests.adapters import HTTPAdapter, Retry
except ModuleNotFoundError as e:  # pragma: no cover - environment specific
    raise SystemExit("The 'requests' package is required to run this script.") from e

BASE_URL = "https://data.binance.vision/"

def make_session(timeout=30):
    s = requests.Session()
    retries = Retry(
        total=10,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "binance-vision-size/1.0"})
    s.timeout = timeout
    return s

def human_bytes(n):
    if n is None:
        return "Unknown"
    if n == 0:
        return "0 B"
    units = ["B","KB","MB","GB","TB","PB"]
    p = int(math.floor(math.log(n, 1024)))
    p = min(p, len(units)-1)
    return f"{n / (1024**p):.2f} {units[p]}"

def s3_list_xml(session, prefix, continuation_token=None):
    """
    Try S3-style listing on website endpoint.
    Returns: (files, dirs, is_truncated, next_token)
    files: list of (key, size)
    dirs: list of dir prefixes (ending in '/')
    """
    params = {
        "list-type": "2",
        "delimiter": "/",
        "prefix": prefix
    }
    if continuation_token:
        params["continuation-token"] = continuation_token

    url = BASE_URL + "?" + urlencode(params)
    r = session.get(url)
    if r.status_code != 200:
        return None

    text = r.text
    if "<ListBucketResult" not in text:
        return None  # not XML; caller should try HTML
    root = ET.fromstring(text)

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Directories
    dirs = []
    for cp in root.findall(f".//{ns}CommonPrefixes"):
        p = cp.find(f"{ns}Prefix")
        if p is not None and p.text:
            dirs.append(p.text)

    # Files
    files = []
    for contents in root.findall(f".//{ns}Contents"):
        key = contents.find(f"{ns}Key").text
        size_text = contents.find(f"{ns}Size").text
        try:
            size = int(size_text)
        except:
            size = 0
        # Only count leaf files (not the "directory markers", which usually end with '/')
        if not key.endswith("/"):
            files.append((key, size))

    is_truncated = (root.find(f"{ns}IsTruncated") is not None and root.find(f"{ns}IsTruncated").text == "true")
    next_token_el = root.find(f"{ns}NextContinuationToken")
    next_token = next_token_el.text if next_token_el is not None else None
    return files, dirs, is_truncated, next_token

def html_list(session, url):
    """
    Parse an HTML directory listing page.
    Returns: (files, dirs, next_url)
    files: list of (href, size)
    dirs: list of (href)
    next_url: absolute URL if 'Next' page exists
    """
    # Import BeautifulSoup lazily to avoid a hard dependency when S3 listing suffices.
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError("BeautifulSoup4 is required for HTML parsing fallback") from e

    r = session.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    files = []
    dirs = []
    # Table rows typically contain: Name (a tag), Last modified, Size
    rows = soup.find_all("tr")
    for tr in rows:
        a = tr.find("a")
        if not a: 
            continue
        name = a.get_text(strip=True)
        href = a.get("href")
        if not href:
            continue
        # Skip parent directory entries
        if name.lower().startswith("parent directory"):
            continue
        # Get size column if available
        tds = tr.find_all("td")
        size = None
        if len(tds) >= 3:
            size_text = tds[-1].get_text(strip=True)
            # Often '-' for directories
            if size_text and size_text != "-":
                # The listing size is usually in bytes or human form (e.g., "1.2 MB")
                # Many S3 indexers show raw bytes; if not, try parsing human sizes:
                try:
                    size = int(size_text.replace(",", ""))
                except:
                    # Best-effort parse for human sizes
                    units = {"B":1, "KB":1024, "MB":1024**2, "GB":1024**3, "TB":1024**4}
                    parts = size_text.split()
                    if len(parts) == 2:
                        try:
                            val = float(parts[0])
                            mul = units.get(parts[1].upper(), None)
                            if mul:
                                size = int(val*mul)
                        except:
                            size = None
        abs_href = urljoin(url, href)
        if href.endswith("/"):
            dirs.append(abs_href)
        else:
            files.append((abs_href, size))

    # Find pagination "Next"
    next_url = None
    for a in soup.find_all("a"):
        if a.get_text(strip=True).lower() == "next":
            next_url = urljoin(url, a.get("href"))
            break

    return files, dirs, next_url

def crawl_prefix(session, prefix):
    """
    Recursively collect sizes for all files under 'prefix' using S3 XML first, fallback to HTML.
    Returns total_size (int) and a dict of sums per immediate child folder under the prefix.
    """
    total = 0
    per_bucket = defaultdict(int)  # immediate child segment sums

    # First attempt: S3 XML listing
    queue = deque([(prefix, None)])  # (prefix, continuation_token or None)
    visited_prefixes = set()

    def add_to_buckets(key, size):
        # Example key: data/spot/daily/klines/BTCUSDT/1m/file.zip
        # Bucket by first subdir after given prefix
        if not key.startswith(prefix):
            return
        remainder = key[len(prefix):]
        first = remainder.split("/", 1)[0] if "/" in remainder else remainder
        if first:
            per_bucket[first] += size

    # Try XML crawl
    xml_worked = False
    while queue:
        cur_prefix, token = queue.popleft()
        if (cur_prefix, token) in visited_prefixes:
            continue
        visited_prefixes.add((cur_prefix, token))
        res = s3_list_xml(session, cur_prefix, token)
        if res is None:
            break  # XML not supported; fallback to HTML
        xml_worked = True
        files, dirs, is_truncated, next_token = res
        for key, size in files:
            total += size
            add_to_buckets(key, size)
        if is_truncated and next_token:
            queue.append((cur_prefix, next_token))
        for d in dirs:
            if d.startswith(cur_prefix):
                queue.append((d, None))

    if xml_worked:
        return total, per_bucket

    # Fallback: HTML recursive crawl starting from a listing URL formed with ?prefix=...&delimiter=/
    start_url = BASE_URL + "?" + urlencode({"prefix": prefix, "delimiter": "/"})
    to_visit = deque([start_url])
    visited_urls = set()

    def key_from_file_url(u):
        # Extract the 'prefix=' query param where possible, or path tail
        parsed = urlparse(u)
        qs = parse_qs(parsed.query)
        # On many pages, file links are plain paths like /data/spot/...
        if parsed.path and parsed.path != "/":
            return parsed.path.lstrip("/")
        # If query with 'prefix' present:
        if "prefix" in qs and qs["prefix"]:
            return qs["prefix"][0]
        # Fallback to last part
        return u.replace(BASE_URL, "")

    while to_visit:
        url = to_visit.popleft()
        if url in visited_urls:
            continue
        visited_urls.add(url)
        try:
            files, dirs, next_url = html_list(session, url)
        except Exception as e:
            print(f"[WARN] Failed to parse {url}: {e}", file=sys.stderr)
            continue
        for f_url, size in files:
            if size is None:
                # We don't know; skip counting this file (rare)
                continue
            # Reconstruct a pseudo "key" to bucket correctly
            key = key_from_file_url(f_url)
            total += size
            add_to_buckets(key, size)
        for d_url in dirs:
            to_visit.append(d_url)
        if next_url:
            to_visit.append(next_url)

    return total, per_bucket

def main():
    ap = argparse.ArgumentParser(description="Compute total size of Binance public data (data.binance.vision)")
    ap.add_argument("--prefix", default="data/", help="Prefix to crawl (default: data/). Examples: data/spot/, data/futures/um/, data/futures/cm/, data/options/")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout per request (seconds)")
    args = ap.parse_args()

    if not args.prefix.endswith("/"):
        args.prefix += "/"

    session = make_session(timeout=args.timeout)
    t0 = time.time()
    print(f"Starting crawl: {BASE_URL} prefix={args.prefix}")
    total, buckets = crawl_prefix(session, args.prefix)
    dt = time.time() - t0

    print("\n=== RESULTS ===")
    print(f"Prefix: {args.prefix}")
    print(f"Total size: {total} bytes  ({human_bytes(total)})")
    print("\nBy immediate subfolder under the prefix:")
    for k in sorted(buckets.keys()):
        print(f"  {k:<16} {human_bytes(buckets[k])}  ({buckets[k]} bytes)")
    print(f"\nCompleted in {dt:.1f}s")

if __name__ == "__main__":
    main()
