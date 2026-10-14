"""Kalshi implied vols from the collaborator's hourly snapshots (data/kalshi_btc_prices.csv.gz; used by 07).

The file has top-of-the-hour bid and ask for every strike of Kalshi's 5pm ET BTC events: the 24-hour dailies (from
January 2026, $250 strikes) and the weeklies. Each snapshot is turned into a smile exactly as in 03 (02's rules 5-7:
bid 0 / ask 1 mean no quote, a mid only when the spread is at most 20c, Thursday 03-05 ET closed; then
`iv.ladder_quotes`' price band, `iv.snapshot_iv`, and `compare.usable_kalshi_snapshots`), with the spot from
02's minute series. The strike ladders are kept too, so Kalshi's straddle can be priced at its bid and ask.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lib import cleaning, compare, iv

YEAR_SECONDS = 365 * 86_400


def load_quotes(root: Path, products=("daily_5pmET",)) -> pd.DataFrame:
    """One row per strike and hourly snapshot, with 02's quote columns (bid, ask, mid, spread, usable) and tau."""
    p = pd.read_csv(Path(root) / "data" / "kalshi_btc_prices.csv.gz")
    p = p[p["product"].isin(products)]
    q = pd.DataFrame({"event_ticker": p.event_ticker.to_numpy(), "strike": p.strike.to_numpy(dtype=float),
                      "ts": pd.to_datetime(p.time_utc, utc=True).to_numpy(),
                      "expiry": pd.to_datetime(p.settle_time_utc, utc=True).to_numpy()})
    q["ts"], q["expiry"] = pd.to_datetime(q.ts, utc=True), pd.to_datetime(q.expiry, utc=True)
    bid, ask = cleaning.quote_sides(p.yes_bid.to_numpy(), p.yes_ask.to_numpy())
    spread, mid = cleaning.mid_and_spread(bid, ask)
    q["bid"], q["ask"], q["spread"], q["mid"] = bid, ask, spread, mid
    q["usable"] = np.isfinite(mid)
    q["exchange_open"] = cleaning.exchange_open(q.ts)
    q["tau_years"] = (q.expiry - q.ts).dt.total_seconds() / YEAR_SECONDS
    q["quote_age_min"] = 0
    life = q.groupby("event_ticker").ts.transform("min")
    q["life_hours"] = (q.expiry - life).dt.total_seconds() / 3600 + 1
    q["kind"] = np.where(q.life_hours > 30, "weekly", "daily")
    return q


def smiles(q: pd.DataFrame, spot: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Smile per (event, snapshot): returns (snapshots, strikes). `spot` is 02's minute spot indexed by ts."""
    q = q.assign(spot=spot.reindex(q.ts).to_numpy())
    q = q[np.isfinite(q.spot)]
    ladder, outer = iv.ladder_quotes(q)
    outer_at = {k: g for k, g in outer.groupby(["event_ticker", "ts"])}
    none = outer.iloc[:0]
    snaps, strikes = [], []
    for (ev, ts), g in ladder.groupby(["event_ticker", "ts"]):
        g = g.sort_values("strike")
        s, tau = float(g.spot.iloc[0]), float(g.tau_years.iloc[0])
        if not iv.ladder_ok(len(g), g.strike.iloc[0], g.strike.iloc[-1], s):
            continue
        o = outer_at.get((ev, ts), none)
        st, snap = iv.snapshot_iv(g.strike, g.mid, 1 / g.spread, s, tau, K_out=o.strike, bid_out=o.bid, ask_out=o.ask)
        snaps.append({"event_ticker": ev, "ts": ts, "expiry": g.expiry.iloc[0], "kind": g.kind.iloc[0], "spot": s,
                      "tau_years": tau, **snap})
        strikes.append(st.assign(event_ticker=ev, ts=ts))
    snaps = pd.DataFrame(snaps)
    snaps["usable"] = compare.usable_kalshi_snapshots(snaps)
    return snaps, pd.concat(strikes, ignore_index=True)


def strip_straddle(ladder: pd.DataFrame, K: float, side: str) -> dict:
    """Kalshi's ATM straddle struck at K, built from its binaries and priced at one side of the book.

    A put pays (K - S)+ = the integral below K of "$1 if S < x" dx: a ladder of "no" contracts below K, each standing
    for the dollars of strike in its cell (half the ladder spacing either side). A call is the ladder of "yes"
    contracts above K. `side` = "mid", "bid" (what selling it brings in: sell yes at the yes bid, sell no at
    1 - yes ask) or "ask" (what buying it costs). Strikes without a usable quote on that side are skipped and their
    dollars reported as `unpriced`. `ladder` has strike, bid, ask, mid for one event and snapshot.
    """
    k = ladder.strike.to_numpy(dtype=float)
    o = np.argsort(k)
    k = k[o]
    bid, ask, mid = (ladder[c].to_numpy(dtype=float)[o] for c in ("bid", "ask", "mid"))
    half = np.median(np.diff(k)) / 2
    lo, hi = k - half, k + half
    w_put = np.clip(np.minimum(hi, K) - lo, 0, None)
    w_call = np.clip(hi - np.maximum(lo, K), 0, None)
    if side == "mid":
        yes, no = mid, 1 - mid
    elif side == "bid":
        yes, no = bid, 1 - ask
    else:
        yes, no = ask, 1 - bid
    ok_c, ok_p = np.isfinite(yes) & (w_call > 0), np.isfinite(no) & (w_put > 0)
    value = float((w_call * yes)[ok_c].sum() + (w_put * no)[ok_p].sum())
    unpriced = float(w_call[~ok_c & (w_call > 0)].sum() + w_put[~ok_p & (w_put > 0)].sum())
    return {"value": value, "unpriced": unpriced}
