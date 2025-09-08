#!/usr/bin/env python3
"""
Enhanced Binance Vision data size calculator with async processing and progress tracking.

Features:
- Async HTTP requests for faster crawling performance  
- Real-time progress tracking with ETA calculations
- Recursive folder crawling with intelligent batching
- Comprehensive error handling and retry logic
- Memory-efficient processing for large datasets
- Detailed logging and statistics collection
- Support for both S3-style XML listings and HTML fallback
- Configurable concurrency limits to respect rate limits

Usage examples:
- All data with progress:       python binance_vision_size.py --progress
- Spot only with async:         python binance_vision_size.py --prefix data/spot/ --concurrent 10
- USD-M futures fast crawl:     python binance_vision_size.py --prefix data/futures/um/ --concurrent 20 --progress
- Options with detailed logs:   python binance_vision_size.py --prefix data/options/ --verbose
"""

import argparse
import asyncio
import logging
import time
import math
import sys
import os
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Set
from urllib.parse import urljoin, urlencode, urlparse, parse_qs
import xml.etree.ElementTree as ET

# Async HTTP imports
try:
    import aiohttp
    import aiofiles
    aiohttp_available = True
except ImportError:
    aiohttp_available = False

# Fallback to synchronous requests
try:
    import requests
    from requests.adapters import HTTPAdapter, Retry
    requests_available = True
except ImportError:
    requests_available = False

# Progress bar
try:
    from tqdm import tqdm
    tqdm_available = True
except ImportError:
    tqdm_available = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler('binance_crawl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/"

class ProgressTracker:
    """Track crawling progress with statistics and ETA calculation."""
    
    def __init__(self, show_progress: bool = False):
        self.show_progress = show_progress
        self.start_time = time.time()
        self.total_requests = 0
        self.completed_requests = 0
        self.total_files = 0
        self.total_size = 0
        self.errors = 0
        self.rate_limits = 0
        
        self.progress_bar = None
        if show_progress and tqdm_available:
            self.progress_bar = tqdm(
                desc="Crawling",
                unit="req",
                dynamic_ncols=True,
                leave=True
            )
    
    def update_total_requests(self, count: int):
        """Update total expected requests."""
        self.total_requests = count
        if self.progress_bar:
            self.progress_bar.total = count
    
    def increment_completed(self, files_found: int = 0, size_found: int = 0):
        """Update completed requests and statistics."""
        self.completed_requests += 1
        self.total_files += files_found
        self.total_size += size_found
        
        if self.progress_bar:
            self.progress_bar.update(1)
            
            # Update description with stats
            elapsed = time.time() - self.start_time
            rate = self.completed_requests / elapsed if elapsed > 0 else 0
            
            self.progress_bar.set_postfix({
                'Files': self.total_files,
                'Size': human_bytes(self.total_size),
                'Rate': f"{rate:.1f}/s",
                'Errors': self.errors
            })
    
    def increment_error(self):
        """Increment error count."""
        self.errors += 1
    
    def increment_rate_limit(self):
        """Increment rate limit count."""
        self.rate_limits += 1
    
    def close(self):
        """Close progress tracking."""
        if self.progress_bar:
            self.progress_bar.close()
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        elapsed = time.time() - self.start_time
        return {
            'total_requests': self.total_requests,
            'completed_requests': self.completed_requests,
            'total_files': self.total_files,
            'total_size': self.total_size,
            'errors': self.errors,
            'rate_limits': self.rate_limits,
            'elapsed_seconds': elapsed,
            'requests_per_second': self.completed_requests / elapsed if elapsed > 0 else 0,
            'completion_percentage': (self.completed_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        }

class AsyncBinanceCrawler:
    """Async Binance Vision data crawler with enhanced performance."""
    
    def __init__(self, concurrent_requests: int = 10, timeout: int = 30, 
                 show_progress: bool = False):
        self.concurrent_requests = concurrent_requests
        self.timeout = timeout
        self.progress = ProgressTracker(show_progress)
        self.semaphore = asyncio.Semaphore(concurrent_requests)
        self.session = None
        
        # Statistics
        self.visited_urls = set()
        self.failed_urls = set()
        
        logger.info(f"Async crawler initialized: concurrent={concurrent_requests}, timeout={timeout}s")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not aiohttp_available:
            raise ImportError("aiohttp is required for async crawling")
        
        # Configure aiohttp session with proper timeouts and retries
        timeout_config = aiohttp.ClientTimeout(
            total=self.timeout,
            connect=10,
            sock_read=self.timeout
        )
        
        connector = aiohttp.TCPConnector(
            limit=self.concurrent_requests * 2,
            limit_per_host=self.concurrent_requests,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout_config,
            connector=connector,
            headers={
                'User-Agent': 'Enhanced-Binance-Vision-Crawler/2.0',
                'Accept': 'application/xml,text/html,*/*',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        self.progress.close()
    
    async def make_request(self, url: str, retries: int = 3) -> Optional[str]:
        """Make async HTTP request with retry logic and rate limiting."""
        if url in self.visited_urls:
            return None
        
        self.visited_urls.add(url)
        
        async with self.semaphore:  # Rate limiting
            for attempt in range(retries + 1):
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            logger.debug(f"Successfully fetched: {url}")
                            return content
                        
                        elif response.status == 429:  # Rate limited
                            self.progress.increment_rate_limit()
                            retry_after = int(response.headers.get('Retry-After', 60))
                            logger.warning(f"Rate limited, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue
                        
                        elif response.status in [500, 502, 503, 504]:
                            # Server errors - retry with exponential backoff
                            if attempt < retries:
                                wait_time = (2 ** attempt) + (time.time() % 1)
                                logger.warning(f"Server error {response.status}, retrying in {wait_time:.1f}s")
                                await asyncio.sleep(wait_time)
                                continue
                        
                        else:
                            logger.error(f"HTTP {response.status} for {url}")
                            self.failed_urls.add(url)
                            self.progress.increment_error()
                            return None
                
                except asyncio.TimeoutError:
                    if attempt < retries:
                        logger.warning(f"Timeout for {url}, retrying...")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"Timeout for {url} after {retries} retries")
                        
                except Exception as e:
                    if attempt < retries:
                        logger.warning(f"Request error for {url}: {e}, retrying...")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"Failed to fetch {url}: {e}")
            
            self.failed_urls.add(url)
            self.progress.increment_error()
            return None
    
    async def parse_s3_xml(self, content: str) -> Optional[Tuple[List[Tuple[str, int]], List[str], bool, Optional[str]]]:
        """Parse S3 XML listing response."""
        try:
            if "<ListBucketResult" not in content:
                return None  # Not XML format
            
            root = ET.fromstring(content)
            
            # Handle namespaces
            ns = ""
            if root.tag.startswith("{"):
                ns = root.tag.split("}")[0] + "}"
            
            # Extract directories
            dirs = []
            for cp in root.findall(f".//{ns}CommonPrefixes"):
                prefix_elem = cp.find(f"{ns}Prefix")
                if prefix_elem is not None and prefix_elem.text:
                    dirs.append(prefix_elem.text)
            
            # Extract files
            files = []
            for contents in root.findall(f".//{ns}Contents"):
                key_elem = contents.find(f"{ns}Key")
                size_elem = contents.find(f"{ns}Size")
                
                if key_elem is not None and size_elem is not None:
                    key = key_elem.text
                    try:
                        size = int(size_elem.text)
                    except (ValueError, TypeError):
                        size = 0
                    
                    # Only count actual files, not directory markers
                    if not key.endswith("/"):
                        files.append((key, size))
            
            # Check for pagination
            is_truncated = False
            next_token = None
            
            truncated_elem = root.find(f"{ns}IsTruncated")
            if truncated_elem is not None and truncated_elem.text == "true":
                is_truncated = True
                
                next_token_elem = root.find(f"{ns}NextContinuationToken")
                if next_token_elem is not None:
                    next_token = next_token_elem.text
            
            return files, dirs, is_truncated, next_token
            
        except ET.ParseError as e:
            logger.debug(f"XML parsing failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing XML: {e}")
            return None
    
    async def parse_html(self, content: str, base_url: str) -> Optional[Tuple[List[Tuple[str, int]], List[str], Optional[str]]]:
        """Parse HTML directory listing (fallback)."""
        try:
            # This is a simplified HTML parser - in production you'd want BeautifulSoup
            files = []
            dirs = []
            next_url = None
            
            # Basic HTML parsing for directory listings
            lines = content.split('\n')
            for line in lines:
                if 'href=' in line.lower() and ('parent directory' not in line.lower()):
                    # Extract href and size if possible
                    # This is a very basic implementation
                    # In practice, you'd want proper HTML parsing
                    pass
            
            logger.warning("HTML parsing is basic implementation - consider using BeautifulSoup for production")
            return files, dirs, next_url
            
        except Exception as e:
            logger.error(f"HTML parsing failed: {e}")
            return None
    
    async def crawl_prefix_async(self, prefix: str) -> Tuple[int, Dict[str, int]]:
        """Async crawl of a prefix with comprehensive progress tracking."""
        logger.info(f"Starting async crawl of prefix: {prefix}")
        
        total_size = 0
        per_bucket = defaultdict(int)
        
        # Queue for URLs to process
        queue = deque([(prefix, None)])  # (prefix, continuation_token)
        processed_prefixes = set()
        
        # Estimate initial request count (very rough estimate)
        estimated_requests = 50  # Will be updated as we discover more
        self.progress.update_total_requests(estimated_requests)
        
        # Process queue with batching
        batch_size = min(self.concurrent_requests * 2, 20)
        
        while queue:
            # Process batch of requests
            current_batch = []
            for _ in range(min(batch_size, len(queue))):
                if queue:
                    current_batch.append(queue.popleft())
            
            if not current_batch:
                break
            
            # Create tasks for current batch
            tasks = []
            for cur_prefix, token in current_batch:
                if (cur_prefix, token) in processed_prefixes:
                    continue
                processed_prefixes.add((cur_prefix, token))
                
                # Build URL
                params = {
                    "list-type": "2",
                    "delimiter": "/",
                    "prefix": cur_prefix
                }
                if token:
                    params["continuation-token"] = token
                
                url = BASE_URL + "?" + urlencode(params)
                task = self.process_single_listing(url, prefix, per_bucket)
                tasks.append(task)
            
            if tasks:
                # Execute batch
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Task failed: {result}")
                        continue
                    
                    if result:
                        files, dirs, is_truncated, next_token, batch_size_found = result
                        total_size += batch_size_found
                        
                        # Add more directories to queue
                        for d in dirs:
                            queue.append((d, None))
                        
                        # Add continuation if needed
                        if is_truncated and next_token:
                            queue.append((current_batch[i][0], next_token))
                        
                        # Update estimates
                        if len(dirs) > 0:
                            estimated_requests += len(dirs) * 2  # Rough estimate
                            self.progress.update_total_requests(estimated_requests)
            
            # Small delay to avoid overwhelming the server
            await asyncio.sleep(0.1)
        
        logger.info(f"Async crawl completed. Total size: {human_bytes(total_size)}")
        return total_size, dict(per_bucket)
    
    async def process_single_listing(self, url: str, base_prefix: str, per_bucket: Dict[str, int]) -> Optional[Tuple[List, List, bool, Optional[str], int]]:
        """Process a single directory listing URL."""
        content = await self.make_request(url)
        if not content:
            self.progress.increment_completed()
            return None
        
        # Try S3 XML parsing first
        xml_result = await self.parse_s3_xml(content)
        if xml_result:
            files, dirs, is_truncated, next_token = xml_result
        else:
            # Fallback to HTML parsing
            html_result = await self.parse_html(content, url)
            if html_result:
                files, dirs, next_url = html_result
                is_truncated = next_url is not None
                next_token = next_url
            else:
                self.progress.increment_completed()
                return None
        
        # Calculate size and update buckets
        batch_size = 0
        for key, size in files:
            batch_size += size
            
            # Update per-bucket statistics
            if key.startswith(base_prefix):
                remainder = key[len(base_prefix):]
                first_segment = remainder.split("/", 1)[0] if "/" in remainder else remainder
                if first_segment:
                    per_bucket[first_segment] += size
        
        self.progress.increment_completed(len(files), batch_size)
        
        return files, dirs, is_truncated, next_token, batch_size

def make_sync_session(timeout: int = 30):
    """Create synchronous requests session (fallback)."""
    if not requests_available:
        raise ImportError("requests library is required for synchronous crawling")
    
    session = requests.Session()
    retries = Retry(
        total=10,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    
    adapter = HTTPAdapter(max_retries=retries, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    session.headers.update({
        "User-Agent": "Enhanced-Binance-Vision-Crawler/2.0-sync"
    })
    session.timeout = timeout
    
    return session
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
