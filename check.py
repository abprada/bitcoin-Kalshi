#!/usr/bin/env python3
"""
Full-history pull of Kalshi equity-related contracts: markets, trades, optional candles.

Kalshi splits data into live and historical tiers, so every pull queries both and
de-duplicates. Uses public market-data endpoints; no API key needed.

Usage
  python kalshi_equity_history.py discover     # writes kalshi_equity/series_candidates.csv
  # -> open the CSV, set keep=1 on the series you want and keep=0 on the rest
  python kalshi_equity_history.py pull --min-volume 1 [--candles 60]
  python kalshi_equity_history.py summarize    # markets / trades / date range per series

Requires: requests, pandas, pyarrow (Python 3.10+). Untested against the live API.
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://external-api.kalshi.com/trade-api/v2"
OUT = Path("kalshi_equity")
CHUNK = 250  # markets per part-file: bounds memory and makes runs resumable
EQUITY_CATEGORIES = {"Financials", "Companies"}  # exact, case-sensitive on Kalshi's side
KEYWORDS = re.compile(
    r"S&P|SPX|Nasdaq|Dow Jones|Russell|\bstocks?\b|\bshares?\b|equit|earnings|\bIPO\b|market cap|\bVIX\b",
    re.IGNORECASE,
)


class Client:
    def __init__(self, max_rps: float = 8.0):
        self.session = requests.Session()
        self.min_gap = 1.0 / max_rps
        self.last = 0.0

    def get(self, path: str, **params) -> dict:
        params = {k: v for k, v in params.items() if v not in (None, "")}
        for attempt in range(8):
            gap = self.min_gap - (time.monotonic() - self.last)
            if gap > 0:
                time.sleep(gap)
            self.last = time.monotonic()
            r = self.session.get(BASE + path, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return {}
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt))  # 429s carry no Retry-After header
                continue
            r.raise_for_status()
        raise RuntimeError(f"gave up on {path} {params}")

    def paginate(self, path: str, key: str, **params):
        cursor = None
        while True:
            data = self.get(path, cursor=cursor, **params)
            yield from data.get(key, [])
            cursor = data.get("cursor")
            if not cursor:
                return


def to_ts(x) -> int | None:
    if x in (None, ""):
        return None
    if isinstance(x, (int, float)):
        return int(x)
    return int(pd.Timestamp(x).timestamp())


def save(rows: list[dict], path: Path) -> None:
    df = pd.json_normalize(rows) if rows else pd.DataFrame({"ticker": pd.Series(dtype=str)})
    df.to_parquet(path, index=False)


def discover(c: Client) -> None:
    rows = []
    for s in c.get("/series", include_volume="true").get("series", []):
        cats = {x for x in (s.get("categories") or []) + [s.get("category")] if x}
        text = f"{s.get('title', '')} {' '.join(s.get('tags') or [])}"
        by_cat, by_kw = bool(cats & EQUITY_CATEGORIES), bool(KEYWORDS.search(text))
        if by_cat or by_kw:
            rows.append({
                "ticker": s["ticker"], "title": s.get("title"), "frequency": s.get("frequency"),
                "categories": "|".join(sorted(cats)), "volume": float(s.get("volume_fp") or 0),
                "match_category": by_cat, "match_keyword": by_kw, "keep": int(by_kw),
            })
    OUT.mkdir(exist_ok=True)
    df = pd.DataFrame(rows).sort_values("volume", ascending=False)
    df.to_csv(OUT / "series_candidates.csv", index=False)
    print(f"{len(df)} candidate series -> {OUT / 'series_candidates.csv'}; edit the keep column")


def markets_for_series(c: Client, series: str) -> list[dict]:
    seen = {}
    for path in ("/historical/markets", "/markets"):  # live last, so it wins on overlap
        for m in c.paginate(path, "markets", series_ticker=series, limit=1000):
            if not m.get("mve_collection_ticker"):  # drop combo/parlay markets
                seen[m["ticker"]] = m
    return sorted(seen.values(), key=lambda m: m["ticker"])


def trades_for_market(c: Client, ticker: str) -> list[dict]:
    seen = {}
    for path in ("/historical/trades", "/markets/trades"):
        for t in c.paginate(path, "trades", ticker=ticker, limit=1000):
            seen[t["trade_id"]] = t
    return list(seen.values())


def candles_for_market(c: Client, series: str, m: dict, interval: int,
                       settled_cutoff: int | None) -> list[dict]:
    start = to_ts(m["open_time"])
    end = to_ts(m.get("settlement_ts") or m["close_time"])
    settled = to_ts(m.get("settlement_ts"))
    hist = f"/historical/markets/{m['ticker']}/candlesticks"
    live = f"/series/{series}/markets/{m['ticker']}/candlesticks"
    archived = settled is not None and settled_cutoff is not None and settled < settled_cutoff
    paths = (hist, live) if archived else (live, hist)
    step = 4000 * interval * 60  # chunk long windows in case the API caps candles per call
    bars = {}
    for lo in range(start, end + 1, step):
        params = dict(start_ts=lo, end_ts=min(lo + step, end), period_interval=interval)
        for path in paths:
            got = c.get(path, **params).get("candlesticks", [])
            if got:
                for b in got:
                    bars[b.get("end_period_ts")] = {**b, "ticker": m["ticker"]}
                break
    return list(bars.values())


def pull(c: Client, min_volume: float, candles: int | None) -> None:
    cand = pd.read_csv(OUT / "series_candidates.csv")
    cutoff = c.get("/historical/cutoff")
    (OUT / "cutoff.json").write_text(json.dumps(cutoff, indent=2))
    settled_cutoff = to_ts(cutoff.get("market_settled_ts"))
    for series in cand.loc[cand["keep"] == 1, "ticker"]:
        sdir = OUT / series
        if (sdir / "_DONE").exists():
            continue
        sdir.mkdir(parents=True, exist_ok=True)
        markets = markets_for_series(c, series)
        save(markets, sdir / "markets.parquet")
        active = [m for m in markets if float(m.get("volume_fp") or 0) >= min_volume]
        print(f"{series}: {len(markets)} markets, {len(active)} with volume >= {min_volume}")
        for k in range(0, len(active), CHUNK):
            part = sdir / f"trades_{k // CHUNK:05d}.parquet"
            if part.exists():  # trades part is written last, so it marks the chunk complete
                continue
            trades, bars = [], []
            for m in active[k:k + CHUNK]:
                trades += trades_for_market(c, m["ticker"])
                if candles:
                    bars += candles_for_market(c, series, m, candles, settled_cutoff)
            if candles:
                save(bars, sdir / f"candles{candles}_{k // CHUNK:05d}.parquet")
            save(trades, part)
            print(f"  markets {k + 1}-{min(k + CHUNK, len(active))}: {len(trades):,} trades")
        (sdir / "_DONE").touch()


def summarize() -> None:
    rows = []
    for sdir in sorted(p for p in OUT.iterdir() if (p / "markets.parquet").exists()):
        m = pd.read_parquet(sdir / "markets.parquet")
        times = []
        for p in sdir.glob("trades_*.parquet"):
            df = pd.read_parquet(p)
            if "created_time" in df:
                times.append(pd.to_datetime(df["created_time"], utc=True))
        t = pd.concat(times) if times else pd.Series(dtype="datetime64[ns, UTC]")
        traded = (m["volume_fp"].astype(float) > 0).sum() if "volume_fp" in m else 0
        rows.append({
            "series": sdir.name, "markets": len(m), "markets_traded": int(traded),
            "trades": len(t), "first_trade": t.min(), "last_trade": t.max(),
            "trading_days": t.dt.tz_convert("America/New_York").dt.date.nunique() if len(t) else 0,
        })
    print(pd.DataFrame(rows).sort_values("trades", ascending=False).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["discover", "pull", "summarize"])
    ap.add_argument("--min-volume", type=float, default=1.0,
                    help="skip markets with lifetime volume below this (most strikes never trade)")
    ap.add_argument("--candles", type=int, choices=[1, 60, 1440],
                    help="also pull candlesticks at this interval in minutes")
    ap.add_argument("--max-rps", type=float, default=8.0)
    a = ap.parse_args()
    c = Client(a.max_rps)
    if a.cmd == "discover":
        discover(c)
    elif a.cmd == "pull":
        pull(c, a.min_volume, a.candles)
    else:
        summarize()


if __name__ == "__main__":
    main()