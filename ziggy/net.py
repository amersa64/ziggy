"""Polite, cached, rate-limited HTTP.

Every byte this project consumes from the outside world goes through here so
that (a) SEC fair-access limits are respected, (b) re-runs are reproducible from
the on-disk cache rather than from a different snapshot of the internet, and
(c) the raw payloads stay auditable.
"""
from __future__ import annotations

import gzip
import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)


class RateLimiter:
    """Simple thread-safe token bucket expressed as a minimum inter-request gap."""

    def __init__(self, max_per_second: float):
        self.min_interval = 1.0 / float(max_per_second) if max_per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self.min_interval


@dataclass
class FetchResult:
    url: str
    status: int
    content: bytes
    from_cache: bool

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


class Fetcher:
    """HTTP client with on-disk cache, rate limiting and exponential backoff."""

    def __init__(
        self,
        cache_dir: Path,
        user_agent: str,
        max_per_second: float = 4.0,
        max_retries: int = 5,
        backoff_base: float = 1.5,
        timeout: int = 45,
        cache_enabled: bool = True,
        namespace: str = "generic",
    ):
        self.cache_dir = Path(cache_dir) / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(max_per_second)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.cache_enabled = cache_enabled
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )
        self.stats = {"hits": 0, "misses": 0, "errors": 0, "bytes": 0}

    # -- cache -----------------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.gz"

    def cached(self, url: str) -> bytes | None:
        p = self._cache_path(url)
        if self.cache_enabled and p.exists():
            try:
                with gzip.open(p, "rb") as fh:
                    return fh.read()
            except OSError:
                p.unlink(missing_ok=True)
        return None

    def _store(self, url: str, content: bytes) -> None:
        if not self.cache_enabled:
            return
        p = self._cache_path(url)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wb", compresslevel=6) as fh:
            fh.write(content)
        tmp.replace(p)

    # -- fetch -----------------------------------------------------------------
    def get(
        self,
        url: str,
        *,
        allow_404: bool = True,
        headers: dict | None = None,
        force: bool = False,
    ) -> FetchResult:
        if not force:
            hit = self.cached(url)
            if hit is not None:
                self.stats["hits"] += 1
                return FetchResult(url, 200, hit, True)

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.acquire()
            try:
                resp = self.session.get(url, timeout=self.timeout, headers=headers)
            except requests.RequestException as exc:  # network-level failure
                last_exc = exc
                time.sleep(self.backoff_base ** attempt)
                continue

            if resp.status_code == 200:
                self.stats["misses"] += 1
                self.stats["bytes"] += len(resp.content)
                self._store(url, resp.content)
                return FetchResult(url, 200, resp.content, False)
            if resp.status_code == 404 and allow_404:
                # Cache the negative result: a missing EDGAR shard stays missing.
                self._store(url, b"")
                return FetchResult(url, 404, b"", False)
            if resp.status_code in (403, 429, 500, 502, 503, 504):
                sleep_for = self.backoff_base ** attempt
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_for = max(sleep_for, float(retry_after))
                log.warning("HTTP %s for %s, retry in %.1fs", resp.status_code, url, sleep_for)
                time.sleep(sleep_for)
                continue
            self.stats["errors"] += 1
            return FetchResult(url, resp.status_code, b"", False)

        self.stats["errors"] += 1
        log.error("giving up on %s (%s)", url, last_exc)
        return FetchResult(url, 0, b"", False)

    def get_json(self, url: str, **kw):
        import json

        r = self.get(url, **kw)
        if r.status == 200 and r.content:
            try:
                return json.loads(r.content)
            except json.JSONDecodeError:
                log.warning("bad JSON from %s", url)
        return None


def sec_fetcher(cfg, namespace: str = "sec") -> Fetcher:
    h = cfg.http
    return Fetcher(
        cache_dir=cfg.raw_dir / "http_cache",
        user_agent=h["user_agent"],
        max_per_second=float(h["sec_max_requests_per_second"]),
        max_retries=int(h["max_retries"]),
        backoff_base=float(h["backoff_base_seconds"]),
        timeout=int(h["timeout_seconds"]),
        cache_enabled=bool(h.get("cache_enabled", True)),
        namespace=namespace,
    )


def generic_fetcher(cfg, namespace: str = "generic") -> Fetcher:
    h = cfg.http
    return Fetcher(
        cache_dir=cfg.raw_dir / "http_cache",
        user_agent=h["user_agent"],
        max_per_second=float(h["generic_max_requests_per_second"]),
        max_retries=int(h["max_retries"]),
        backoff_base=float(h["backoff_base_seconds"]),
        timeout=int(h["timeout_seconds"]),
        cache_enabled=bool(h.get("cache_enabled", True)),
        namespace=namespace,
    )
