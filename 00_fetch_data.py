"""Fetch 1-minute bid/ask candles for Kalshi's weekly BTC contracts (Friday 5pm ET expiry).

KXBTCD events expiring Friday 5pm ET open the previous Friday 4pm ET and live 169h; every
strike is "BTC (CF Benchmarks BRTI, 60s average) above K at expiry". Other days' 5pm events
are 25h dailies and are skipped. Public endpoints only, no API key needed.

Output (data/raw/kalshi_btc_weekly/):
  candles/{event}.parquet     one row per strike-minute: YES bid/ask OHLC, volume, open interest
  markets/{event}.parquet     strike ladder: strike, open/close time, how it resolved (result,
                              expiration_value = settlement index, settlement_value, settlement_ts)
  settlement/{event}.parquet  settlement index (BRTI 60s trailing average) per second, last 3h
                              before expiry; its value at expiry is expiration_value
  markets.parquet, events.parquet, settlement.parquet  combined from the per-event files

Each event's resolution is cross-checked (index at expiry = expiration_value; YES iff
expiration_value > floor_strike); mismatches are printed and flagged in resolution_ok.

Resumable: events whose candles file exists are skipped.
  python 00_fetch_data.py                      # 2025-10-01 .. 2026-10-01
  python 00_fetch_data.py --limit 2            # smoke test
"""

import argparse
import math
import re
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from lib.fetch_utils import Client, time_chunks, write_parquet

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXBTCD"
OUT_DIR = Path(__file__).resolve().parent / "data" / "raw" / "kalshi_btc_weekly"
CANDLE_COLS = ["ts", "bid_open", "bid_high", "bid_low", "bid_close",
               "ask_open", "ask_high", "ask_low", "ask_close", "volume", "open_interest"]

EXPIRY_HOUR_ET = "17"
FRIDAY = 4
MAX_CANDLES = 5000  # per candlestick request
MIN_WEEKLY_LIFE_S = 6 * 24 * 3600  # weeklies live 169h, dailies 25h

# Event tickers: KXBTCD-26OCT0217, BTCD-24OCT2317, BTCD-24JUN07-17 -> (date, ET hour)
_TICKER_RE = re.compile(r"-(\d{2}[A-Z]{3}\d{2})-?(\d{2})$")


def event_date(event_ticker: str) -> date:
    return datetime.strptime(_TICKER_RE.search(event_ticker).group(1), "%y%b%d").date()


def is_candidate(event_ticker: str, start: date, end: date) -> bool:
    """Friday 5pm ET expiry within [start, end]. Whether it actually lived a week is checked later."""
    m = _TICKER_RE.search(event_ticker)
    if not m or m.group(2) != EXPIRY_HOUR_ET:
        return False
    d = event_date(event_ticker)
    return d.weekday() == FRIDAY and start <= d <= end


def _num(x) -> float:
    """Kalshi sends prices/counts as strings; empty means not applicable."""
    return float(x) if x not in (None, "") else float("nan")


def market_row(m: dict, source: str) -> dict:
    """One strike of the ladder. `source` is 'live' or 'historical' (decides the candle endpoint)."""
    if m["strike_type"] != "greater":
        raise ValueError(f"{m['ticker']}: strike_type {m['strike_type']!r}, expected 'greater'")
    return {
        "event_ticker": m["event_ticker"],
        "market_ticker": m["ticker"],
        # "$89,000 or above" is floor_strike 88999.99: YES iff settlement value > floor_strike
        "strike": round(m["floor_strike"] + 0.01, 2),
        "floor_strike": m["floor_strike"],
        "open_time": m["open_time"],
        "close_time": m["close_time"],
        "status": m["status"],
        "result": m["result"],
        "expiration_value": _num(m["expiration_value"]),
        "settlement_value": _num(m.get("settlement_value_dollars")),
        "settlement_ts": m.get("settlement_ts"),
        "volume": _num(m["volume_fp"]),
        "open_interest": _num(m["open_interest_fp"]),
        "source": source,
    }


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def classify_event(markets: list[dict]) -> str:
    """'weekly', 'too_short' (a 25h daily that happens to expire Friday), 'unsettled' or 'no_markets'."""
    if not markets:
        return "no_markets"
    if any(m["status"] != "finalized" for m in markets):
        return "unsettled"
    life = max(_ts(m["close_time"]) for m in markets) - min(_ts(m["open_time"]) for m in markets)
    return "weekly" if life >= MIN_WEEKLY_LIFE_S else "too_short"


def candle_rows(candles: list[dict]) -> list[dict]:
    """Flatten 1-min candles to YES bid/ask OHLC + volume + OI.

    Live candles use `close_dollars`/`volume_fp`, archived ones `close`/`volume`; both map here.
    Trade-price OHLC is dropped. Rows are kept raw: no forward-fill, bid 0 / ask 1 = empty book.
    """
    rows = []
    for c in candles:
        live = "volume_fp" in c
        sfx = "_dollars" if live else ""
        row = {"ts": c["end_period_ts"]}
        for side, key in (("bid", "yes_bid"), ("ask", "yes_ask")):
            for f in ("open", "high", "low", "close"):
                row[f"{side}_{f}"] = _num(c[key][f + sfx])
        row["volume"] = _num(c["volume_fp" if live else "volume"])
        row["open_interest"] = _num(c["open_interest_fp" if live else "open_interest"])
        rows.append(row)
    return rows


def settlement_rows(live_data: dict) -> list[dict]:
    """Settlement index from GET /live_data/events/{event}: BRTI 60s trailing average, 1 point per second.

    Kalshi keeps only the last 3 hours before expiry; the value at maturity is the settlement value.
    """
    return [{"ts": p["t"] // 1000, "index_60s": p["v"]} for p in live_data["details"]["timeseries"]]


def resolution_problems(markets: list[dict], live_data: dict) -> list[str]:
    """Cross-check how each strike resolved: settlement value = index at maturity, YES iff value > floor_strike."""
    d = live_data["details"]
    idx = next((p["v"] for p in d["timeseries"] if p["t"] == d["maturity_ts_ms"]), None)
    if idx is None:
        return ["no settlement index value at maturity"]
    problems = []
    missing = [m for m in markets if math.isnan(m["expiration_value"])]
    if missing:
        problems.append(f"expiration_value missing on {len(missing)}/{len(markets)} strikes (index at maturity {idx})")
    for m in markets:
        if not math.isnan(m["expiration_value"]) and abs(m["expiration_value"] - idx) > 0.005:
            problems.append(f"{m['market_ticker']}: expiration_value {m['expiration_value']} != index at maturity {idx}")
        # judged against the index, which exists even when Kalshi leaves expiration_value blank
        if (idx > m["floor_strike"]) != (m["result"] == "yes"):
            problems.append(f"{m['market_ticker']}: result {m['result']!r} but index at maturity "
                            f"{idx} vs floor_strike {m['floor_strike']}")
    return problems


# --- network + disk -----------------------------------------------------------------------------


def _paginate(client: Client, path: str, key: str, **params) -> list[dict]:
    out, cursor = [], ""
    while True:
        d = client.get(path, **params, **({"cursor": cursor} if cursor else {}))
        out += d[key]
        cursor = d.get("cursor")
        if not cursor:
            return out


def fetch_markets(client: Client, event_ticker: str) -> list[dict]:
    """Strike ladder of an event. Markets settled before the historical cutoff live only in /historical."""
    for path, source in (("/markets", "live"), ("/historical/markets", "historical")):
        markets = _paginate(client, path, "markets", event_ticker=event_ticker, limit=1000)
        if markets:
            return [market_row(m, source) for m in markets]
    return []


def fetch_candles(client: Client, market: dict) -> list[dict]:
    """All 1-min candles over this strike's own life (strikes can be listed mid-week)."""
    t = market["market_ticker"]
    path = f"/series/{SERIES}/markets/{t}/candlesticks" if market["source"] == "live" \
        else f"/historical/markets/{t}/candlesticks"
    out = []
    for s, e in time_chunks(_ts(market["open_time"]), _ts(market["close_time"]), MAX_CANDLES):
        out += client.get(path, start_ts=s, end_ts=e, period_interval=1)["candlesticks"]
    return out


def fetch_event(client: Client, event_ticker: str, out: Path) -> tuple[str, int]:
    """Fetch one event's ladder, settlement index and candles to disk. Returns (classification, n_candles)."""
    markets = fetch_markets(client, event_ticker)
    kind = classify_event(markets)
    if kind != "weekly":
        return kind, 0

    live = client.get(f"/live_data/events/{event_ticker}")["live_data"]
    problems = resolution_problems(markets, live)
    for p in problems:
        print(f"  resolution warning: {p}")
    settlement = pd.DataFrame(settlement_rows(live)).assign(event_ticker=event_ticker)
    settlement["ts"] = pd.to_datetime(settlement["ts"], unit="s", utc=True)
    settlement = settlement[["event_ticker", "ts", "index_60s"]]

    frames = []
    for m in markets:
        df = pd.DataFrame(candle_rows(fetch_candles(client, m)), columns=CANDLE_COLS)
        if df.empty and m["volume"] > 0:
            print(f"  warning: {m['market_ticker']} traded {m['volume']:,.0f} contracts but returned no candles")
        frames.append(df.assign(market_ticker=m["market_ticker"], strike=m["strike"]))

    candles = pd.concat(frames, ignore_index=True).assign(event_ticker=event_ticker)
    candles["ts"] = pd.to_datetime(candles["ts"], unit="s", utc=True)
    candles = candles[["event_ticker", "market_ticker", "strike", *CANDLE_COLS]].sort_values(["strike", "ts"])

    mkt = pd.DataFrame(markets).assign(resolution_ok=not problems)
    for col in ("open_time", "close_time", "settlement_ts"):
        mkt[col] = pd.to_datetime(mkt[col], utc=True)

    # Candles last: that file is the "event done" marker checked on resume.
    write_parquet(mkt, out / "markets" / f"{event_ticker}.parquet")
    write_parquet(settlement, out / "settlement" / f"{event_ticker}.parquet")
    write_parquet(candles, out / "candles" / f"{event_ticker}.parquet")
    return kind, len(candles)


def combine(out: Path) -> None:
    """Rebuild markets.parquet, events.parquet and settlement.parquet from the per-event files on disk."""
    files = sorted((out / "markets").glob("*.parquet"))
    if not files:
        return
    markets = pd.concat(map(pd.read_parquet, files), ignore_index=True)
    markets.to_parquet(out / "markets.parquet", index=False)
    events = markets.groupby("event_ticker", as_index=False).agg(
        open_time=("open_time", "min"),
        close_time=("close_time", "max"),
        n_strikes=("market_ticker", "size"),
        expiration_value=("expiration_value", "first"),  # BRTI 60s average at expiry, same for all strikes
        volume=("volume", "sum"),
        resolution_ok=("resolution_ok", "all"),
    )
    events.to_parquet(out / "events.parquet", index=False)
    settlement = pd.concat(map(pd.read_parquet, sorted((out / "settlement").glob("*.parquet"))), ignore_index=True)
    settlement.to_parquet(out / "settlement.parquet", index=False)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 10, 1), help="first expiry date")
    p.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 1), help="last expiry date")
    p.add_argument("--limit", type=int, help="fetch at most N events not yet on disk (for testing)")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)

    # Public reads cost 10 tokens against a bucket refilling ~20/s: ~2 requests/s
    client = Client(BASE_URL, min_interval=0.5)
    events = sorted(
        (t for t in {e["event_ticker"] for e in _paginate(client, "/events", "events", series_ticker=SERIES, limit=200)}
         if is_candidate(t, a.start, a.end)),
        key=event_date,
    )
    todo = [t for t in events if not (a.out / "candles" / f"{t}.parquet").exists()]
    print(f"{len(events)} Friday 5pm events with expiry in {a.start}..{a.end}; {len(events) - len(todo)} already on disk")
    todo = todo[: a.limit] if a.limit else todo

    t_run = time.monotonic()
    for i, t in enumerate(todo, 1):
        t0, n0 = time.monotonic(), client.n_requests
        kind, n = fetch_event(client, t, a.out)
        detail = f"{n:,} candles" if kind == "weekly" else f"skipped ({kind})"
        print(f"[{i}/{len(todo)}] {t}: {detail}  [{client.n_requests - n0} req, {time.monotonic() - t0:.0f}s]", flush=True)

    combine(a.out)
    print(f"done: {client.n_requests} requests in {(time.monotonic() - t_run) / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
