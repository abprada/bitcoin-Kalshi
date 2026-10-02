"""Stitch the BRTI settlement index (60s trailing average, 1 point per second) from Kalshi live data.

Every KXBTCD event serves the 3h of CF Benchmarks' BRTI 60-second trailing average before its
own expiry (GET /live_data/events/{event}). Events expire every hour, including overnight ones
missing from the /events listing, so one event per 3h block (ticker built from the ET expiry
hour) covers the timeline. Hours still missing are filled from overlapping hourly events and
the 15-minute series (KXBTC15M, 1h each). Where windows overlap the values must agree; the
largest disagreement is reported.

This is the quantity that settles every KXBTCD contract (its value at expiry = expiration_value).
It trails spot BRTI by ~30s, so before the final minute exchange spot (00_fetch_spot.py) is the
closer state variable for pricing.

Output (data/raw/brti/):
  {first_day}_{last_day}.parquet  one file per month (the resume unit): ts (UTC, second), index_60s;
                                  months tile without overlap
  brti_60s_1s.parquet             combined
  brti_60s_1m.parquet             minute-end values; joins with the Kalshi and spot candles on ts

Resumable: months already on disk are skipped.
  python 00_fetch_brti.py           # 2025-08-27 .. 2026-10-01
"""

import argparse
import time
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from fetch_utils import Client, month_windows, write_parquet

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
OUT_DIR = Path(__file__).resolve().parent / "data" / "raw" / "brti"
ET = ZoneInfo("America/New_York")
WINDOW_S = 3 * 3600  # KXBTCD live data covers the 3h before expiry


def _et(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, ET)


def kxbtcd_ticker(expiry_ts: int) -> str:
    """Hourly event expiring at expiry_ts: KXBTCD-{yy}{MON}{dd}{HH} in Eastern time."""
    return f"KXBTCD-{_et(expiry_ts):%y%b%d%H}".upper()


def kxbtc15m_ticker(expiry_ts: int) -> str:
    """15-minute event expiring at expiry_ts: KXBTC15M-{yy}{MON}{dd}{HHMM} in Eastern time."""
    return f"KXBTC15M-{_et(expiry_ts):%y%b%d%H%M}".upper()


def index_points(live_data: dict) -> list[tuple[int, float]]:
    """Running 60s averages served with an event, over [expiry - 3h, expiry - 1s].

    The point at maturity is dropped: it is the settlement value (the previous second's average
    rounded to the cent), not an observation of the running average. The event expiring next
    supplies that second, so consecutive events tile without overlap.
    """
    d = live_data["details"]
    if d.get("coin") != "BTC":
        raise ValueError(f"{d.get('event_ticker')}: coin {d.get('coin')!r}, expected 'BTC'")
    return [(p["t"] // 1000, p["v"]) for p in d["timeseries"] if p["t"] != d["maturity_ts_ms"]]


def merge_into(merged: dict[int, float], points: list[tuple[int, float]], start_ts: int, end_ts: int) -> float:
    """Add points within [start_ts, end_ts]; return the largest disagreement with points already present."""
    conflict = 0.0
    for t, v in points:
        if not start_ts <= t <= end_ts:
            continue
        if t in merged:
            conflict = max(conflict, abs(merged[t] - v))
        else:
            merged[t] = v
    return conflict


def incomplete_hours(covered, start_ts: int, end_ts: int) -> list[int]:
    """Hour ends h in (start_ts, end_ts] whose seconds h - 3600 .. h - 1 are not all covered."""
    return [h for h in range(start_ts + 3600, end_ts + 1, 3600)
            if any(t not in covered for t in range(h - 3600, h))]


def block_ends(start_ts: int, end_ts: int) -> list[int]:
    return list(range(start_ts + WINDOW_S, end_ts + 1, WINDOW_S))


# --- network + disk -----------------------------------------------------------------------------


def fetch_live(client: Client, ticker: str) -> list[tuple[int, float]] | None:
    """Index points served with an event, or None if the event never existed (404)."""
    try:
        return index_points(client.get(f"/live_data/events/{ticker}")["live_data"])
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return None
        raise


def fetch_window(client: Client, start_ts: int, end_ts: int) -> tuple[dict[int, float], dict]:
    merged, tried = {}, set()
    stats = {"events": 0, "fallback_events": 0, "max_conflict": 0.0}

    def add(ticker: str) -> None:
        if ticker in tried:
            return
        tried.add(ticker)
        pts = fetch_live(client, ticker)
        if pts:
            stats["events"] += 1
            stats["max_conflict"] = max(stats["max_conflict"], merge_into(merged, pts, start_ts, end_ts - 1))

    for t in block_ends(start_ts, end_ts):
        add(kxbtcd_ticker(t))
    primary = stats["events"]
    for h in incomplete_hours(merged, start_ts, end_ts):
        # hourly events expiring at h, h+1h, h+2h all cover hour h; the 15-min series covers 1h
        for ticker in (kxbtcd_ticker(h), kxbtcd_ticker(h + 3600), kxbtcd_ticker(h + 7200), kxbtc15m_ticker(h)):
            add(ticker)
            if not incomplete_hours(merged, h - 3600, h):
                break
    stats["fallback_events"] = stats["events"] - primary
    stats["requests"] = len(tried)
    return merged, stats


def combine(out: Path) -> None:
    files = sorted(out.glob("*_*.parquet"))
    if not files:
        return
    df = pd.concat(map(pd.read_parquet, files), ignore_index=True).drop_duplicates("ts").sort_values("ts")
    df.to_parquet(out / "brti_60s_1s.parquet", index=False)
    df[df.ts.dt.second == 0].to_parquet(out / "brti_60s_1m.parquet", index=False)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 8, 27), help="first UTC day")
    p.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 1), help="last UTC day")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)
    if a.end >= datetime.now(timezone.utc).date():
        p.error("--end must be before today (UTC): an incomplete window would be cached as done")

    client = Client(BASE_URL, min_interval=0.5)  # shares Kalshi's ~2 requests/s public budget
    day = lambda ts: datetime.fromtimestamp(ts, timezone.utc).date()
    for label, s, e in month_windows(a.start, a.end):
        end = e + 60  # 00:00:00 of the following day; the month covers seconds [s, end - 1]
        path = a.out / f"{day(s)}_{day(e)}.parquet"
        if path.exists():
            continue
        t0 = time.monotonic()
        merged, st = fetch_window(client, s, end)
        df = pd.DataFrame(sorted(merged.items()), columns=["ts", "index_60s"])
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        need = end - s
        print(f"{label}: {len(df):,}/{need:,} seconds ({len(df) / need:.3%}), {need - len(df):,} missing | "
              f"{st['events']} events ({st['fallback_events']} fallback), {st['requests']} requests, "
              f"max overlap disagreement {st['max_conflict']:.6f} | {time.monotonic() - t0:.0f}s", flush=True)
        write_parquet(df, path)

    combine(a.out)
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
