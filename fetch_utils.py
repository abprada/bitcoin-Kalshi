"""HTTP and time-window helpers shared by the fetch scripts."""

import calendar
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


class Client:
    """Unauthenticated GET client: spaces requests and retries rate limits / server errors."""

    def __init__(self, base_url: str, session=None, min_interval=0.5, max_retries=8,
                 sleep=time.sleep, clock=time.monotonic):
        self.base_url = base_url
        self.session = session or requests.Session()
        self.min_interval, self.max_retries = min_interval, max_retries
        self.sleep, self.clock = sleep, clock
        self._last = float("-inf")
        self.n_requests = 0

    def get(self, path: str, **params):
        for attempt in range(self.max_retries + 1):
            wait = self._last + self.min_interval - self.clock()
            if wait > 0:
                self.sleep(wait)
            self._last = self.clock()
            self.n_requests += 1
            try:
                r = self.session.get(self.base_url + path, params=params, timeout=30)
            except (requests.ConnectionError, requests.Timeout):
                r = None
            if r is None or r.status_code == 429 or r.status_code >= 500:
                self.sleep(min(2**attempt, 60))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"GET {path} {params}: still failing after {self.max_retries} retries")


def time_chunks(start_ts: int, end_ts: int, span_minutes: int) -> list[tuple[int, int]]:
    """Split [start_ts, end_ts] into windows spanning at most `span_minutes` minutes.

    Both bounds are inclusive, so the next window starts one minute after the previous one ends.
    """
    chunks, s = [], start_ts
    while s <= end_ts:
        e = min(s + span_minutes * 60, end_ts)
        chunks.append((s, e))
        s = e + 60
    return chunks


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file so an interrupted run never leaves a truncated parquet behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _utc(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def month_windows(start: date, end: date) -> list[tuple[str, int, int]]:
    """(YYYY-MM, first minute start, last minute start) per calendar month, clipped to [start, end]."""
    out, d = [], start
    while d <= end:
        stop = min(date(d.year, d.month, calendar.monthrange(d.year, d.month)[1]), end)
        out.append((f"{d:%Y-%m}", _utc(d), _utc(stop) + 86_400 - 60))
        d = stop + timedelta(days=1)
    return out
