"""Bid-ask spreads and volumes of Kalshi's weeklies and the matching Deribit Friday expiries, hour by hour (used by 16).

Kalshi: from 02's clean quotes (every strike, every minute). Spread = ask - bid of the strike nearest spot that minute
(cents, and as a share of its mid); volume = contracts traded, and premium = contracts x mid in dollars (Kalshi's
candles have no trade prices, so the mid stands in for the price paid).

Deribit: the Friday 08:00 expiry with the Kalshi weekly's date, from every trade (00_fetch_deribit_all.py) in its last
7.5 days. There are no quotes, so the spread is the **effective** spread: a taker buy fills at the ask and a sell at the
bid, so the signed gap between the trade price and Deribit's mark is the half-spread actually paid
(`deribit_marks.effective_half_spread`); twice that is the round trip. At the money = within 0.25 ATM standard
deviations of the forward. Volume = BTC traded (one option = 1 BTC) and premium = BTC traded x USD price.

Both markets are summarised per contract and UTC hour, with the hours left to that contract's own expiry.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lib import deribit_marks as dm

ATM_SD = 0.25   # Deribit trades within this many ATM standard deviations of the forward count as at the money


def kalshi_hourly(root: Path) -> pd.DataFrame:
    """Per weekly and UTC hour: median ATM spread (dollars per $1 contract) and relative spread (spread / mid) of the
    strike nearest spot, contracts traded and premium traded (contracts x mid, dollars), hours to expiry."""
    out = []
    for f in sorted((Path(root) / "data" / "clean" / "kalshi_weekly_quotes").glob("*.parquet")):
        q = pd.read_parquet(f, columns=["event_ticker", "strike", "ts", "tau_years", "bid", "ask", "spread", "mid",
                                        "usable", "volume", "log_moneyness"])
        q["hour"] = q.ts.dt.floor("h")
        q["premium"] = q.volume * q.mid
        vol = q.groupby("hour").agg(contracts=("volume", "sum"), premium=("premium", "sum"),
                                    tau_years=("tau_years", "median"))
        u = q[q.usable & (q.ask > q.bid) & (q.mid > 0)]
        atm = u.loc[u.assign(a=u.log_moneyness.abs()).groupby("ts").a.idxmin()]      # nearest strike, each minute
        atm = atm.assign(rel=atm.spread / atm.mid)
        spr = atm.groupby("hour").agg(atm_spread=("spread", "median"), atm_rel_spread=("rel", "median"),
                                      atm_mid=("mid", "median"))
        h = vol.join(spr, how="left").reset_index()
        h["event_ticker"] = q.event_ticker.iloc[0]
        out.append(h)
    k = pd.concat(out, ignore_index=True)
    k["hours_to_expiry"] = k.tau_years * 365 * 24
    return k.drop(columns="tau_years")


def deribit_hourly(root: Path, kalshi: pd.DataFrame) -> pd.DataFrame:
    """Per matching Deribit Friday expiry and UTC hour: median ATM effective round-trip spread (share of the mark
    price, and USD), number of ATM trades, BTC traded and premium traded (USD), hours to Deribit's expiry."""
    expiry_k = (kalshi.hour + pd.to_timedelta(kalshi.hours_to_expiry, unit="h")).dt.round("h")
    fridays = set(expiry_k.dt.normalize() + pd.Timedelta(hours=8))
    t = dm.read_full_trades(root, start=kalshi.hour.min() - pd.Timedelta(days=1), end=kalshi.hour.max(),
                            max_tte_days=7.5)
    t = dm.add_forward(t[t.expiry.isin(fridays)])
    t = dm.effective_half_spread(t)
    t["hour"] = t.ts.dt.floor("h")
    t["premium"] = t.amount * t.price_usd
    t["atm"] = (t.m / t.iv_d).abs() <= ATM_SD
    vol = t.groupby(["expiry", "hour"]).agg(btc=("amount", "sum"), premium=("premium", "sum"),
                                            tau_years=("tau_years", "median"))
    a = t[t.atm & np.isfinite(t.hs_frac)]
    spr = a.groupby(["expiry", "hour"]).agg(atm_rel_spread=("hs_frac", lambda x: 2 * x.median()),
                                            atm_spread_usd=("hs_usd", lambda x: 2 * x.median()),
                                            atm_trades=("hs_frac", "size"))
    d = vol.join(spr, how="left").reset_index()
    d["hours_to_expiry"] = d.tau_years * 365 * 24
    return d.drop(columns="tau_years")


def cached(root: Path) -> dict:
    """{"kalshi": kalshi_hourly, "deribit": deribit_hourly}, saved to data/derived/liquidity_*.parquet on first use
    (delete them to rebuild)."""
    d = Path(root) / "data" / "derived"
    fk, fd = d / "liquidity_kalshi.parquet", d / "liquidity_deribit.parquet"
    if fk.exists() and fd.exists():
        return {"kalshi": pd.read_parquet(fk), "deribit": pd.read_parquet(fd)}
    k = kalshi_hourly(root)
    dd = deribit_hourly(root, k)
    k.to_parquet(fk)
    dd.to_parquet(fd)
    return {"kalshi": k, "deribit": dd}
