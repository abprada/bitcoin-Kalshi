"""Cleaning rules for the Kalshi weekly BTC quotes and exchange spot (used by 02_clean_data.py; the dailies in
lib/term.py go through the same `event_quotes`).

Each function implements one documented rule; 02_clean_data.py applies them, shows the evidence for
each rule and writes the clean dataset.
"""

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
SPREAD_CAP = 0.20          # dollars: wider two-sided quotes give no usable mid
YEAR_SECONDS = 365 * 86_400


def _naive_ns(ts) -> np.ndarray:
    """UTC timestamps (Series, Index or array, tz-aware) as naive datetime64[ns] for fast arithmetic."""
    return pd.DatetimeIndex(ts).tz_convert("UTC").tz_localize(None).to_numpy("datetime64[ns]")


def reconstruct_expiration(settlement: pd.DataFrame, expiry: pd.Series) -> pd.Series:
    """Settlement value per event: BRTI 60s average one second before expiry, rounded to the cent.

    `settlement` has event_ticker, ts, index_60s; `expiry` maps event_ticker -> expiry timestamp.
    Events without an index value at that second get NaN.
    """
    idx = settlement.set_index(["event_ticker", "ts"]).index_60s
    vals = [idx.get((e, t - pd.Timedelta(seconds=1)), np.nan) for e, t in expiry.items()]
    return pd.Series(np.round(np.asarray(vals, dtype=float), 2), index=expiry.index)


def clean_spot(spot: pd.DataFrame) -> pd.DataFrame:
    """One spot price per minute from long exchange candles (exchange, ts, close, volume).

    spot = average of the Coinbase and Bitstamp closes available that minute (Bitstamp alone when
    Coinbase is missing); spot_stale = no trade on any exchange that minute (price carried forward).
    """
    close = spot.pivot(index="ts", columns="exchange", values="close").reindex(columns=["coinbase", "bitstamp"])
    vol = spot.pivot(index="ts", columns="exchange", values="volume").reindex(columns=["coinbase", "bitstamp"])
    out = close.copy()
    out.columns.name = None
    out["spot"] = close.mean(axis=1)
    has_cb, has_bs = close.coinbase.notna(), close.bitstamp.notna()
    out["spot_source"] = np.select([has_cb & has_bs, has_cb], ["both", "coinbase only"], "bitstamp only")
    out["spot_stale"] = ~((vol.coinbase.fillna(0) > 0) | (vol.bitstamp.fillna(0) > 0))
    return out.sort_index()


def build_event_grid(candles: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """Forward-filled 1-minute quote grid for one weekly event.

    Rows run from each strike's first candle to expiry (minutes before a strike's first candle have
    no quote and are dropped). Closing bid/ask and open interest are carried forward; volume is 0
    on minutes without a candle; quote_age_min counts minutes since the strike's last candle.
    `markets` has market_ticker, strike, open_time, close_time; `candles` has market_ticker, ts,
    bid_close, ask_close, volume, open_interest.
    """
    grid = pd.date_range(markets.open_time.min() + pd.Timedelta(minutes=1), markets.close_time.max(), freq="1min")
    gn = _naive_ns(grid)
    n, k = len(grid), len(markets)
    row = pd.Index(markets.market_ticker).get_indexer(candles.market_ticker)
    col = ((_naive_ns(candles.ts) - gn[0]) // np.timedelta64(1, "m")).astype(int)
    vals = {}
    for c in ("bid_close", "ask_close", "volume", "open_interest"):
        vals[c] = np.full((k, n), np.nan, dtype=np.float32)
        vals[c][row, col] = candles[c].to_numpy()
    has = np.zeros((k, n), dtype=bool)
    has[row, col] = True
    last = np.maximum.accumulate(np.where(has, np.arange(n)[None, :], -1), axis=1)
    alive = ((gn[None, :] > _naive_ns(markets.open_time)[:, None])
             & (gn[None, :] <= _naive_ns(markets.close_time)[:, None]))
    r, c = np.nonzero(alive & (last >= 0))
    li = last[r, c]
    return pd.DataFrame({
        "market_ticker": markets.market_ticker.to_numpy()[r],
        "strike": markets.strike.to_numpy()[r],
        "ts": grid[c],
        "bid_raw": vals["bid_close"][r, li],
        "ask_raw": vals["ask_close"][r, li],
        "volume": np.where(has[r, c], vals["volume"][r, c], 0).astype(np.float32),
        "open_interest": vals["open_interest"][r, li],
        "quote_age_min": (c - li).astype(np.int32),
    })


def quote_sides(bid_raw, ask_raw) -> tuple[np.ndarray, np.ndarray]:
    """Kalshi encodes a missing bid as 0 and a missing ask as 1: turn those into NaN."""
    bid_raw, ask_raw = np.asarray(bid_raw), np.asarray(ask_raw)
    return np.where(bid_raw > 0, bid_raw, np.nan), np.where(ask_raw < 1, ask_raw, np.nan)


def mid_and_spread(bid, ask, cap: float = SPREAD_CAP) -> tuple[np.ndarray, np.ndarray]:
    """Spread for valid two-sided quotes (bid < ask); mid only when the spread is at most `cap`.

    Prices are on a 1-cent grid, so the spread is compared in whole cents.
    """
    bid, ask = np.asarray(bid, dtype=float), np.asarray(ask, dtype=float)
    two_sided = bid < ask                      # False for NaN sides, locked and crossed quotes
    spread_c = np.where(two_sided, np.round((ask - bid) * 100), np.nan)
    mid = np.where(two_sided & (spread_c <= round(cap * 100)), (bid + ask) / 2, np.nan)
    return spread_c / 100, mid


def exchange_open(ts) -> np.ndarray:
    """False for minutes inside Kalshi's weekly closure, Thursday 03:00-05:00 ET.

    `ts` are candle minute ends (UTC), so the closed minutes end 03:01 through 05:00 ET.
    """
    et = pd.Series(pd.DatetimeIndex(ts)).dt.tz_convert(ET)
    m = et.dt.hour * 60 + et.dt.minute
    closed = (et.dt.dayofweek == 3) & (m > 180) & (m <= 300)
    return (~closed).to_numpy()


def tau_years(ts, expiry) -> np.ndarray:
    """Time to expiry in years, ACT/365."""
    return (pd.Series(expiry - pd.Series(pd.DatetimeIndex(ts)))).dt.total_seconds().to_numpy() / YEAR_SECONDS


def event_quotes(candles: pd.DataFrame, markets: pd.DataFrame, spot: pd.Series, spot_stale: pd.Series) -> pd.DataFrame:
    """Rules R4-R8 for one event: the forward-filled quote grid with sides, mid and spread, exchange hours, spot
    and moneyness. `candles` and `markets` are one event's raw files; `spot`/`spot_stale` from `clean_spot`."""
    g = build_event_grid(candles, markets)                                  # R4
    bid, ask = quote_sides(g.bid_raw.to_numpy(), g.ask_raw.to_numpy())      # R5
    spread, mid = mid_and_spread(bid, ask)                                  # R6
    is_open = exchange_open(g.ts)                                           # R7
    s = spot.reindex(g.ts).to_numpy()
    return pd.DataFrame({
        # plain strings (parquet dictionary-encodes them): per-file categoricals cannot be read as one folder
        "event_ticker": markets.event_ticker.iloc[0],
        "market_ticker": g.market_ticker.astype(str),
        "strike": g.strike.astype("float32"),
        "ts": g.ts,
        "tau_years": tau_years(g.ts, markets.close_time.max()).astype("float32"),
        "bid": bid.astype("float32"), "ask": ask.astype("float32"),
        "spread": spread.astype("float32"), "mid": mid.astype("float32"),
        "quote_age_min": g.quote_age_min,
        "exchange_open": is_open,
        "usable": ~np.isnan(mid) & is_open,                                 # R8
        "volume": g.volume, "open_interest": g.open_interest,
        "spot": s, "spot_stale": spot_stale.reindex(g.ts).to_numpy(),
        "log_moneyness": np.log(g.strike.to_numpy() / s).astype("float32"),
    })
