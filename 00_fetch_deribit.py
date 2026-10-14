"""Fetch every Deribit BTC option trade from Deribit's public history API (no key needed).

The collaborator's file (data/deribit_btc_options.csv.gz) has only 07:00-08:00 and 19:30-22:30 UTC and expiries up
to about 8 days; checked against this API on 2026-03-10 it holds every trade inside that filter and about 2% of the
day's volume. This script fetches the whole day, every expiry. Each trade comes with Deribit's mark price, index and
implied vol at that instant, so it serves as a time series of marks as well as of trades. No bid/ask: the API keeps
no historical order book.

Output (data/raw/deribit_trades/):
  {YYYY-MM-DD}.parquet           one file per UTC day, raw API fields (the resume unit)
  btc_options_trades.parquet     combined, in the collaborator's columns (read by `deribit.load_trades`), plus
                                 trade_id, block_trade_id and `window`: "exact_0800UTC" (07:00-08:00),
                                 "us_afternoon" (19:30-22:30, as in the collaborator's file) or "other"

Resumable: days already on disk are skipped.
  python 00_fetch_deribit.py                          # 2026-01-15 .. 2026-09-30, the collaborator's span
  python 00_fetch_deribit.py --start 2026-03-10 --end 2026-03-10
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from lib.fetch_utils import Client, write_parquet

OUT_DIR = Path(__file__).resolve().parent / "data" / "raw" / "deribit_trades"
BASE_URL = "https://history.deribit.com/api/v2"
PAGE = 1000


def fetch_day(client: Client, day: date) -> pd.DataFrame:
    """All BTC option trades with a timestamp in [day 00:00, day + 1 00:00) UTC.

    Pages forward in time from the last timestamp seen (not one past it, so trades sharing the boundary millisecond
    are not skipped) and drops repeats by trade_id.
    """
    start = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    end = start + 86_400_000 - 1
    rows, t0 = [], start
    while True:
        r = client.get("/public/get_last_trades_by_currency_and_time", currency="BTC", kind="option",
                       start_timestamp=t0, end_timestamp=end, count=PAGE, sorting="asc")["result"]
        trades = r["trades"]
        rows += trades
        if not r.get("has_more") or not trades:
            break
        last = trades[-1]["timestamp"]
        t0 = last if last > t0 else t0 + 1      # a full page within one millisecond: step past it
    d = pd.DataFrame(rows)
    return d.drop_duplicates("trade_id").reset_index(drop=True) if len(d) else d


def window_of(ts: pd.Series) -> np.ndarray:
    m = ts.dt.hour * 60 + ts.dt.minute
    return np.select([ts.dt.hour == 7, (m >= 19 * 60 + 30) & (m < 22 * 60 + 30)],
                     ["exact_0800UTC", "us_afternoon"], "other")


def to_collaborator_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """API fields -> the collaborator's columns, so `deribit.load_trades` reads either file."""
    ts = pd.to_datetime(raw.timestamp, unit="ms", utc=True)
    parts = raw.instrument_name.str.split("-", expand=True)            # BTC-13MAR26-70000-C
    expiry = pd.to_datetime(parts[1], format="%d%b%y").dt.tz_localize("UTC") + pd.Timedelta(hours=8)
    return pd.DataFrame({
        "window": window_of(ts), "date": ts.dt.strftime("%Y-%m-%d"), "instrument": raw.instrument_name,
        "expiry_utc": expiry.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"), "strike": parts[2].astype(float), "cp": parts[3],
        "time_utc": ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"), "price_btc": raw.price,
        "index_price": raw.index_price, "price_usd": raw.price * raw.index_price, "amount": raw.amount,
        "direction": raw.direction, "iv": raw.iv, "mark_price_btc": raw.mark_price, "trade_id": raw.trade_id,
        "block_trade_id": raw.get("block_trade_id", pd.Series(pd.NA, index=raw.index)),
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 15))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 30))
    args = ap.parse_args()
    client = Client(BASE_URL, min_interval=0.15)
    day = args.start
    while day <= args.end:
        path = OUT_DIR / f"{day:%Y-%m-%d}.parquet"
        if not path.exists():
            d = fetch_day(client, day)
            write_parquet(d, path)
            print(f"{day}  {len(d):6d} trades  ({client.n_requests} requests so far)", flush=True)
        day += timedelta(days=1)
    files = sorted(p for p in OUT_DIR.glob("????-??-??.parquet"))
    full = pd.concat([to_collaborator_columns(pd.read_parquet(p)) for p in files], ignore_index=True)
    write_parquet(full, OUT_DIR / "btc_options_trades.parquet")
    print(f"combined: {len(full):,} trades, {full.date.nunique()} days -> {OUT_DIR / 'btc_options_trades.parquet'}")


if __name__ == "__main__":
    main()
