"""Fetch 1-minute bid/ask candles for any Kalshi KXBTCD event (dailies included), e.g. for a one-day case study.

00_fetch_data.py fetches only the weeklies. This fetches named events with the same functions and the same file
layout, so the cleaning rules of 02 apply unchanged. Public endpoints only, no API key needed.

Output (data/raw/kalshi_events/):
  candles/{event}.parquet  one row per strike-minute: YES bid/ask OHLC, volume, open interest
  markets/{event}.parquet  strike ladder and how each strike resolved

  python 00_fetch_kalshi_event.py KXBTCD-25OCT0917
"""

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

from lib.fetch_utils import Client, write_parquet

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "raw" / "kalshi_events"
_spec = importlib.util.spec_from_file_location("fetch_data", ROOT / "00_fetch_data.py")
fd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fd)


def fetch(event_ticker: str, client: Client, out: Path = OUT_DIR) -> int:
    markets = fd.fetch_markets(client, event_ticker)
    if not markets:
        raise SystemExit(f"{event_ticker}: no markets")
    frames = []
    for m in markets:
        df = pd.DataFrame(fd.candle_rows(fd.fetch_candles(client, m)), columns=fd.CANDLE_COLS)
        frames.append(df.assign(market_ticker=m["market_ticker"], strike=m["strike"]))
    candles = pd.concat(frames, ignore_index=True).assign(event_ticker=event_ticker)
    candles["ts"] = pd.to_datetime(candles["ts"], unit="s", utc=True)
    candles = candles[["event_ticker", "market_ticker", "strike", *fd.CANDLE_COLS]].sort_values(["strike", "ts"])
    mkt = pd.DataFrame(markets)
    for col in ("open_time", "close_time", "settlement_ts"):
        mkt[col] = pd.to_datetime(mkt[col], utc=True)
    write_parquet(mkt, out / "markets" / f"{event_ticker}.parquet")
    write_parquet(candles, out / "candles" / f"{event_ticker}.parquet")
    return len(candles)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("events", nargs="+")
    a = p.parse_args(argv)
    client = Client(fd.BASE_URL, min_interval=0.5)
    for ev in a.events:
        print(f"{ev}: {fetch(ev, client):,} candles", flush=True)


if __name__ == "__main__":
    main()
