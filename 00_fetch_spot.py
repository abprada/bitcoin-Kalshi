"""Fetch 1-minute BTC/USD spot candles from Coinbase and Bitstamp (both CF Benchmarks BRTI constituents).

Trade-based OHLCV: no free source keeps historical bid/ask, and BTC-USD spreads are ~$0.01
(<0.5 bps) anyway. `ts` is the minute END in UTC, the same convention as the Kalshi candles,
so the two join on ts. Coinbase omits minutes without trades; Bitstamp keeps them with zero
volume. Values are stored raw.

Output (data/raw/btc_spot/):
  {exchange}/{first_day}_{last_day}.parquet  one file per exchange-month (the resume unit)
  btcusd_1m.parquet                          combined, long: exchange, ts, open, high, low, close, volume

Resumable: windows already on disk are skipped.
  python 00_fetch_spot.py          # 2025-08-27 .. 2026-10-01 (30 days before the first weekly opened)
"""

import argparse
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from lib.fetch_utils import Client, month_windows, time_chunks, write_parquet

OUT_DIR = Path(__file__).resolve().parent / "data" / "raw" / "btc_spot"
COLS = ["ts", "open", "high", "low", "close", "volume"]


def coinbase_rows(candles: list[list]) -> list[dict]:
    """Coinbase rows are [time, low, high, open, close, volume], newest first, time = minute start."""
    return [{"ts": t + 60, "open": o, "high": h, "low": lo, "close": c, "volume": v}
            for t, lo, h, o, c, v in sorted(candles)]


def bitstamp_rows(ohlc: list[dict]) -> list[dict]:
    """Bitstamp rows are dicts of strings, timestamp = minute start."""
    return [{"ts": int(x["timestamp"]) + 60, **{k: float(x[k]) for k in ("open", "high", "low", "close", "volume")}}
            for x in ohlc]


# --- network + disk -----------------------------------------------------------------------------


def fetch_coinbase(client: Client, start_ts: int, end_ts: int) -> list[dict]:
    rows = []
    for s, e in time_chunks(start_ts, end_ts, 299):  # 300 candles max, both bounds inclusive
        raw = client.get("/products/BTC-USD/candles", granularity=60, start=s, end=e)
        rows += coinbase_rows([c for c in raw if s <= c[0] <= e])
    return rows


def fetch_bitstamp(client: Client, start_ts: int, end_ts: int) -> list[dict]:
    rows = []
    for s, e in time_chunks(start_ts, end_ts, 999):
        # Bitstamp ignores `start` when `end` is also given (it returns the `limit` candles ending
        # at `end`), so send start + limit only and trim to the window here.
        raw = client.get("/api/v2/ohlc/btcusd/", step=60, limit=1000, start=s)["data"]["ohlc"]
        rows += bitstamp_rows([x for x in raw if s <= int(x["timestamp"]) <= e])
    return rows


def combine(out: Path) -> None:
    files = sorted(out.glob("*/*.parquet"))
    if not files:
        return
    df = pd.concat([pd.read_parquet(f).assign(exchange=f.parent.name) for f in files], ignore_index=True)
    df = df.drop_duplicates(["exchange", "ts"]).sort_values(["exchange", "ts"])[["exchange", *COLS]]
    df.to_parquet(out / "btcusd_1m.parquet", index=False)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 8, 27), help="first UTC day")
    p.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 1), help="last UTC day")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)
    if a.end >= datetime.now(timezone.utc).date():
        p.error("--end must be before today (UTC): an incomplete window would be cached as done")

    sources = {
        "coinbase": (Client("https://api.exchange.coinbase.com", min_interval=0.15), fetch_coinbase),  # 10 req/s cap
        "bitstamp": (Client("https://www.bitstamp.net", min_interval=0.1), fetch_bitstamp),
    }
    for name, (client, fetch) in sources.items():
        for label, s, e in month_windows(a.start, a.end):
            day = lambda ts: datetime.fromtimestamp(ts, timezone.utc).date()
            path = a.out / name / f"{day(s)}_{day(e)}.parquet"
            if path.exists():
                continue
            t0, n0 = time.monotonic(), client.n_requests
            df = pd.DataFrame(fetch(client, s, e), columns=COLS)
            df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
            minutes = (e - s) // 60 + 1
            print(f"{name} {label}: {len(df):,}/{minutes:,} minutes ({len(df) / minutes:.1%})  "
                  f"[{client.n_requests - n0} req, {time.monotonic() - t0:.0f}s]", flush=True)
            write_parquet(df, path)

    combine(a.out)
    print(f"done -> {a.out / 'btcusd_1m.parquet'}")


if __name__ == "__main__":
    main()
