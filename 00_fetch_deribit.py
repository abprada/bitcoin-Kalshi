"""Fetch Deribit BTC option trades for the whole day, plus BTC-PERPETUAL 1-minute bars and funding.

The collaborator's file (data/deribit_btc_options.csv.gz) keeps only two windows a day (07:00-08:00 and
19:30-22:29 UTC) and a few expiries, so it never follows an option through the day. This pulls every BTC option
trade from Deribit's public history API (history.deribit.com keeps expired instruments), with the same fields
(price, mark price, mark iv, index, amount, taker side), so a short-dated option can be valued at any hour.
Public endpoints only, no API key needed.

Output (data/raw/deribit/):
  options/{first_day}_{last_day}.parquet   one file per month (the resume unit): every BTC option trade whose
                                           expiry is at most MAX_TTE_DAYS after the trade
  perp/{first_day}_{last_day}.parquet      BTC-PERPETUAL 1-minute OHLCV (ts = minute END, UTC, like the spot file)
  funding.parquet                          BTC-PERPETUAL hourly funding (interest_1h, interest_8h)
  options.parquet, perp.parquet            combined

Resumable: months already on disk are skipped.
  python 00_fetch_deribit.py                 # 2025-10-01 .. 2026-09-30
"""

import argparse
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from lib.fetch_utils import Client, month_windows, write_parquet

HISTORY_URL = "https://history.deribit.com/api/v2/public"
LIVE_URL = "https://www.deribit.com/api/v2/public"
OUT_DIR = Path(__file__).resolve().parent / "data" / "raw" / "deribit"
MAX_TTE_DAYS = 15          # options expiring later than this after the trade are dropped (not needed here)
PAGE = 1000                # trades per request (API maximum)
PERP_SPAN_MIN = 4999       # 1-minute bars per chart request (API returns at most ~5000)
FUNDING_SPAN_DAYS = 30     # funding history returns at most 744 hourly rows per request

_INSTRUMENT_RE = re.compile(r"^BTC-(\d{1,2}[A-Z]{3}\d{2})-(\d+)-([CP])$")
TRADE_COLS = ["ts", "trade_id", "instrument", "expiry", "strike", "cp", "price_btc", "mark_price_btc", "iv",
              "index_price", "amount", "direction"]


def parse_instrument(name: str) -> tuple[pd.Timestamp, float, str] | None:
    """BTC-27MAR26-114000-C -> (2026-03-27 08:00 UTC, 114000.0, "C"); None for anything else (combos, other)."""
    m = _INSTRUMENT_RE.match(name)
    if not m:
        return None
    expiry = datetime.strptime(m.group(1), "%d%b%y").replace(hour=8, tzinfo=timezone.utc)
    return pd.Timestamp(expiry), float(m.group(2)), m.group(3)


def trade_rows(trades: list[dict]) -> list[dict]:
    """Flatten history trades; keep plain BTC options. Every Deribit option expires at 08:00 UTC."""
    out = []
    for t in trades:
        p = parse_instrument(t["instrument_name"])
        if p is None:
            continue
        out.append({"ts": t["timestamp"], "trade_id": t["trade_id"], "instrument": t["instrument_name"],
                    "expiry": p[0], "strike": p[1], "cp": p[2], "price_btc": t["price"],
                    "mark_price_btc": t["mark_price"], "iv": t.get("iv"), "index_price": t["index_price"],
                    "amount": t["amount"], "direction": t["direction"]})
    return out


def fetch_options(client: Client, start_ts: int, end_ts: int) -> pd.DataFrame:
    """All BTC option trades with start_ts <= timestamp <= end_ts (seconds), paging forward in time.

    Pages restart at the last timestamp seen (not one past it), so fills sharing a millisecond across a page
    boundary are not lost; repeats are dropped by trade id.
    """
    rows, st, end_ms = [], start_ts * 1000, end_ts * 1000 + 59_999
    while True:
        r = client.get("/get_last_trades_by_currency_and_time", currency="BTC", kind="option",
                       start_timestamp=st, end_timestamp=end_ms, count=PAGE, sorting="asc")["result"]
        tr = r["trades"]
        rows += trade_rows(tr)
        if not r["has_more"] or not tr:
            break
        nxt = tr[-1]["timestamp"]
        st = nxt if nxt > st else nxt + 1     # a full page inside one millisecond: step past it
    df = pd.DataFrame(rows, columns=TRADE_COLS).drop_duplicates("trade_id")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
    return df[(df.expiry - df.ts) <= pd.Timedelta(days=MAX_TTE_DAYS)].sort_values("ts").reset_index(drop=True)


def fetch_perp(client: Client, start_ts: int, end_ts: int) -> pd.DataFrame:
    """BTC-PERPETUAL 1-minute OHLCV; `ts` is the minute end, as in 00_fetch_spot.py."""
    frames, s = [], start_ts
    while s <= end_ts:
        e = min(s + PERP_SPAN_MIN * 60, end_ts)
        r = client.get("/get_tradingview_chart_data", instrument_name="BTC-PERPETUAL", resolution="1",
                       start_timestamp=s * 1000, end_timestamp=e * 1000)["result"]
        if r["ticks"]:
            frames.append(pd.DataFrame({"ts": r["ticks"], "open": r["open"], "high": r["high"], "low": r["low"],
                                        "close": r["close"], "volume": r["volume"]}))
        s = e + 60
    df = pd.concat(frames, ignore_index=True).drop_duplicates("ts")
    df = df[(df.ts >= start_ts * 1000) & (df.ts <= end_ts * 1000)]
    df["ts"] = pd.to_datetime(df.ts + 60_000, unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def fetch_funding(client: Client, start_ts: int, end_ts: int) -> pd.DataFrame:
    rows, s = [], start_ts
    while s < end_ts:
        e = min(s + FUNDING_SPAN_DAYS * 86_400, end_ts)
        rows += client.get("/get_funding_rate_history", instrument_name="BTC-PERPETUAL",
                           start_timestamp=s * 1000, end_timestamp=e * 1000)["result"]
        s = e
    df = pd.DataFrame(rows).drop_duplicates("timestamp")
    df["ts"] = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def combine(out: Path) -> None:
    for kind in ("options", "perp"):
        files = sorted((out / kind).glob("*.parquet"))
        if files:
            pd.concat(map(pd.read_parquet, files), ignore_index=True).to_parquet(out / f"{kind}.parquet", index=False)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 10, 1), help="first UTC day")
    p.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 30), help="last UTC day")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--no-options", action="store_true",
                   help="perpetual and funding only (the options come from 00_fetch_deribit_all.py instead)")
    a = p.parse_args(argv)
    if a.end >= datetime.now(timezone.utc).date():
        p.error("--end must be before today (UTC): an incomplete window would be cached as done")

    hist, live = Client(HISTORY_URL, min_interval=0.1), Client(LIVE_URL, min_interval=0.1)
    day = lambda ts: datetime.fromtimestamp(ts, timezone.utc).date()
    for label, s, e in month_windows(a.start, a.end):
        kinds = (("perp", live, fetch_perp),) + (() if a.no_options else (("options", hist, fetch_options),))
        for kind, client, fetch in kinds:
            path = a.out / kind / f"{day(s)}_{day(e)}.parquet"
            if path.exists():
                continue
            t0, n0 = time.monotonic(), client.n_requests
            df = fetch(client, s, e)
            print(f"{kind} {label}: {len(df):,} rows  [{client.n_requests - n0} req, {time.monotonic() - t0:.0f}s]",
                  flush=True)
            write_parquet(df, path)
    s0, e0 = month_windows(a.start, a.end)[0][1], month_windows(a.start, a.end)[-1][2]
    write_parquet(fetch_funding(live, s0, e0 + 60), a.out / "funding.parquet")
    combine(a.out)
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
