"""Backtests of the vol trades: Kalshi binary strips, Deribit straddles, and Kalshi-Deribit dislocations
(used by 08_backtest.py).

Every trade is held to expiry (the Deribit file only has two trading windows a day, 07:00 and 19:00-22:59 UTC, so
intraday holds cannot be marked) except the long weekly leg of the calendar, which is sold back at the daily's expiry.

- **Kalshi strip.** Short vol on Kalshi = one contract on every strike within `half_width` dollars of spot: sell YES
  above spot (at the bid), buy YES below (at the ask). It pays like a short straddle with its wings cut off: each
  strike the price ends beyond costs $1. With strikes `step` dollars apart that is a straddle on 1/step BTC, so P&L
  times `step` is per 1 BTC of straddle. Real settlement (each market's `result`), Kalshi's taker fee per order, or,
  as an optimistic bound, maker fills at mid with a quarter of the fee. Collateral is what Kalshi locks: the price
  paid for each YES bought, 1 - bid for each YES sold (Kalshi does not net positions across strikes).
- **Deribit straddle.** Sold at the money (strike = spot) at Deribit's ATM mark vol, settled on the 30-minute average
  of spot before 08:00 UTC (Deribit's delivery price is the index average over that half hour). Fees per leg are
  0.03% of spot, at most 12.5% of the option's price. We have no Deribit quotes, so the half-spread is an
  assumption, charged as `half_spread_vol` vol points of vega per leg.
- **Dislocation.** Kalshi's ATM vol minus Deribit's (stretched to Kalshi's expiry with the hour-of-week variance
  profile), minus the median of that gap over the previous 30 days at the same horizon: how unusual today's gap is.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lib import cleaning, compare, deribit, iv, term

YEAR_SECONDS = 365 * 86_400
WING_SD = 1.5                  # strip covers spot +/- 1.5 SDs, from BTC's trailing 7-day realized vol (not Kalshi's)
RV_LOOKBACK = pd.Timedelta(days=7)
MAX_QUOTE_AGE_MIN = 60         # a leg's quote must have updated within the last hour
CONTRACTS = 100                # order size per strike, for the fee's rounding
MAKER_FEE_SHARE = 0.25         # Kalshi makers pay a quarter of the taker fee
DERIBIT_HALF_SPREAD_VOL = 0.005  # assumed: Deribit ATM quotes 1 vol point wide
DISLOC_LOOKBACK_DAYS = 30
DISLOC_MIN_OBS = 5
SNAP_TOLERANCE = pd.Timedelta(minutes=30)
QUOTE_COLS = ["strike", "ts", "bid", "ask", "mid", "quote_age_min", "spot"]


# --- data ----------------------------------------------------------------------------------------


def load_spot(root: Path) -> pd.Series:
    return pd.read_parquet(root / "data" / "clean" / "btc_spot_1m.parquet", columns=["ts", "spot"]).set_index("ts").spot


def kalshi_markets(root: Path, product: str) -> pd.DataFrame:
    """Markets of the weeklies or dailies with strike, close time and settled value (1 = YES)."""
    f = {"weekly": root / "data" / "clean" / "kalshi_weekly_markets.parquet",
         "daily": root / "data" / "raw" / "kalshi_btc_daily" / "markets.parquet"}[product]
    m = pd.read_parquet(f)
    m = m[m.result.isin(["yes", "no"])]
    return m.assign(settle=(m.result == "yes").astype(float))


def kalshi_quotes(root: Path, product: str, event: str) -> pd.DataFrame:
    """One event's clean 1-minute quotes: 02's file for weeklies; for dailies, 02's rules applied to the raw candles
    (`cleaning.event_quotes`) and cached in data/clean/kalshi_daily_quotes/."""
    if product == "weekly":
        return pd.read_parquet(root / "data" / "clean" / "kalshi_weekly_quotes" / f"{event}.parquet", columns=QUOTE_COLS)
    cache = root / "data" / "clean" / "kalshi_daily_quotes" / f"{event}.parquet"
    if cache.exists():
        return pd.read_parquet(cache, columns=QUOTE_COLS)
    raw = root / "data" / "raw" / "kalshi_btc_daily"
    m = pd.read_parquet(raw / "markets.parquet", filters=[("event_ticker", "==", event)])
    candles = pd.read_parquet(raw / "candles" / f"{event}.parquet")
    spot = pd.read_parquet(root / "data" / "clean" / "btc_spot_1m.parquet").set_index("ts")
    q = cleaning.event_quotes(candles, m, spot.spot, spot.spot_stale) if len(candles) else pd.DataFrame(columns=QUOTE_COLS)
    cache.parent.mkdir(parents=True, exist_ok=True)
    q.to_parquet(cache, index=False)
    return q[QUOTE_COLS]


def trailing_rv(cum: pd.Series, ts, lookback: pd.Timedelta = RV_LOOKBACK) -> np.ndarray:
    """BTC's realized vol over the `lookback` before each time (known at entry: sets the strip's width)."""
    ts = pd.DatetimeIndex(ts)
    return term.realized_vol(cum, ts - lookback, ts)


# --- Kalshi strip --------------------------------------------------------------------------------


def strip_legs(ladder: pd.DataFrame, spot: float, half_width: float) -> pd.DataFrame:
    """The strikes within `half_width` of spot, each marked `up` (above spot) or not, with its quotes."""
    legs = ladder[(ladder.strike - spot).abs() <= half_width].copy()
    legs["up"] = legs.strike > spot
    return legs


def strip_pnl(legs: pd.DataFrame, value: pd.Series, side: int = -1, maker: bool = False,
              exit_quotes: pd.DataFrame | None = None, contracts: int = CONTRACTS, fee_share: float | None = None) -> dict:
    """P&L of one strip, per contract per strike, in dollars.

    `side` -1 sells vol (sell YES above spot, buy YES below), +1 buys it. `value` maps strike -> settled value
    (1 = YES). If `exit_quotes` (strike, bid, ask) is given, the strip is closed at those quotes instead (taker).
    `fee_share` overrides the share of the taker fee paid (0 with `maker=True` and mid exit quotes: before costs).
    Legs without the quote they need (a bid to sell, an ask to buy, a fresh enough one) are dropped.
    Returns pnl, collateral, fees, number of legs and the strike step.
    """
    buy = legs.up.to_numpy() == (side > 0)                       # buy YES: long vol above, short vol below
    share = fee_share if fee_share is not None else (MAKER_FEE_SHARE if maker else 1)
    rate = compare.KALSHI_FEE_RATE * share
    if maker:
        # mid of the two-sided quote, also where the spread is wider than 02's 20c cap for a "usable" mid
        price = legs.mid.fillna((legs.bid + legs.ask) / 2).to_numpy(dtype=float)
    else:
        price = np.where(buy, legs.ask, legs.bid).astype(float)
    ok = np.isfinite(price) & (legs.quote_age_min.to_numpy() <= MAX_QUOTE_AGE_MIN)
    K = legs.strike.to_numpy()
    if exit_quotes is None:
        v = value.reindex(K).to_numpy(dtype=float)
        exit_fee = np.zeros(len(K))
    else:
        e = exit_quotes.set_index("strike").reindex(K)
        # close: sell what was bought at the bid, buy back what was sold at the ask. No bid means nobody pays for the
        # YES (worth 0), no ask means nobody sells it (worth 1): dropping those legs would hide the losers.
        v = np.where(buy, e.bid.fillna(0.0), e.ask.fillna(1.0)).astype(float)
        exit_fee = compare.kalshi_fee(np.nan_to_num(v), rate=compare.KALSHI_FEE_RATE * (1 if fee_share is None else fee_share),
                                      contracts=contracts)
    ok &= np.isfinite(v)
    if ok.sum() < 2:
        return dict(pnl=np.nan, collateral=np.nan, fees=np.nan, legs=int(ok.sum()), step=np.nan)
    price, v, buy, K, exit_fee = price[ok], v[ok], buy[ok], K[ok], exit_fee[ok]
    fee = compare.kalshi_fee(price, rate=rate, contracts=contracts)
    pnl = np.where(buy, v - price, price - v) - fee - exit_fee
    collateral = np.where(buy, price, 1 - price)
    return dict(pnl=pnl.sum(), collateral=collateral.sum(), fees=(fee + exit_fee).sum(), legs=int(ok.sum()),
                step=float(np.median(np.diff(np.sort(K)))))


def ladder_at(quotes: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    return quotes[quotes.ts == ts]


def mid_exit(quotes: pd.DataFrame) -> pd.DataFrame:
    """Exit quotes at mid (before costs): the mid where there is one, else the middle of the bid (0 if none) and
    the ask (1 if none)."""
    m = quotes.mid.where(quotes.mid.notna(), (quotes.bid.fillna(0.0) + quotes.ask.fillna(1.0)) / 2)
    return quotes.assign(bid=m, ask=m)


# --- Deribit straddle ----------------------------------------------------------------------------


def straddle_price(F, sigma, tau):
    F, sigma = np.asarray(F, dtype=float), np.asarray(sigma, dtype=float)
    return iv.bs_call(F, F, sigma, tau) + iv.bs_put(F, F, sigma, tau)


def deribit_straddle_pnl(F, sigma, tau, S_T, side: int = -1,
                         half_spread_vol: float = DERIBIT_HALF_SPREAD_VOL) -> dict:
    """P&L per 1 BTC of an ATM straddle (strike = F) sold (`side` -1) or bought (+1) at `sigma` and held to expiry,
    after Deribit's fees and an assumed half-spread of `half_spread_vol` vol points per leg."""
    F, sigma, tau, S_T = (np.asarray(x, dtype=float) for x in (F, sigma, tau, S_T))
    prem = straddle_price(F, sigma, tau)
    leg = prem / 2                                                     # at the money, call = put (forward = spot)
    fee = 2 * np.minimum(term.DERIBIT_FEE_RATE * F, term.DERIBIT_FEE_CAP * leg)
    vega = F * np.sqrt(tau) * 0.3989422804014327                       # per leg, at the money
    spread = 2 * half_spread_vol * vega
    gross = side * (np.abs(S_T - F) - prem)
    return dict(gross=gross, pnl=gross - fee - spread, premium=prem, fees=fee, spread_cost=spread)


def deribit_delivery(spot: pd.Series, expiry) -> np.ndarray:
    """Deribit's delivery price, approximated by the mean spot over the 30 minutes up to each 08:00 UTC expiry."""
    out = []
    for e in pd.DatetimeIndex(expiry):
        w = spot.loc[e - pd.Timedelta(minutes=29): e]
        out.append(w.mean() if len(w) >= 20 else np.nan)
    return np.asarray(out)


# --- inference -----------------------------------------------------------------------------------


def summarize(trades: pd.DataFrame, col: str = "pnl_bps", by=None, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per group: trades, weeks, mean, 95% interval for the mean (resampling whole calendar weeks: trades in one
    week share BTC's path), median, share positive, and the mean per quarter.

    `col` is the P&L column; trades need `entry` (UTC time)."""
    by = [] if by is None else ([by] if isinstance(by, str) else list(by))
    rng = np.random.default_rng(seed)
    t = trades.dropna(subset=[col]).assign(week=lambda x: x.entry.dt.tz_convert(None).dt.to_period("W"),
                                           q=lambda x: x.entry.dt.tz_convert(None).dt.to_period("Q").astype(str))
    rows = []
    for key, g in (t.groupby(by, observed=True) if by else [((), t)]):
        weeks = [w[col].to_numpy() for _, w in g.groupby("week")]
        sums, counts = np.array([w.sum() for w in weeks]), np.array([len(w) for w in weeks])
        idx = rng.integers(0, len(weeks), (n_boot, len(weeks)))
        boot = sums[idx].sum(1) / counts[idx].sum(1)
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "trades": len(g), "weeks": len(weeks), "mean": g[col].mean(),
                     "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                     "median": g[col].median(), "win": (g[col] > 0).mean(),
                     **{f"mean_{q}": v for q, v in g.groupby("q")[col].mean().items()}})
    return pd.DataFrame(rows)


# --- outright strips on a schedule ---------------------------------------------------------------


def strip_trades(root: Path, product: str, hours_before, cum: pd.Series, side: int = -1,
                 wing_sd: float = WING_SD) -> pd.DataFrame:
    """One strip per event and entry time `hours_before` its close, held to settlement, taker and maker.

    Returns one row per trade: `event`, `entry`, `hours`, `spot`, `rv_entry`, `half_width`, `pnl_taker` and
    `pnl_maker` (and `pnl_mid`: at mid, no fee, i.e. before costs) per 1 BTC of straddle, the same in basis points of spot (`bps_*`), `ret_*` on collateral, legs."""
    markets = kalshi_markets(root, product)
    rows = []
    for ev, m in markets.groupby("event_ticker"):
        close = m.close_time.max()
        entries = [(close - pd.Timedelta(hours=h)).floor("min") for h in hours_before]
        q = kalshi_quotes(root, product, ev)
        if q.empty:
            continue
        q = q[q.ts.isin(entries)]
        value = m.set_index("strike").settle
        for h, t in zip(hours_before, entries):
            lad = ladder_at(q, t)
            if lad.empty or not np.isfinite(lad.spot.iloc[0]):
                continue
            S = float(lad.spot.iloc[0])
            rv = trailing_rv(cum, [t])[0]
            if not np.isfinite(rv):
                continue
            hw = wing_sd * rv * np.sqrt(h * 3600 / YEAR_SECONDS) * S
            legs = strip_legs(lad, S, hw)
            r = {"event": ev, "entry": t, "hours": h, "spot": S, "rv_entry": rv, "half_width": hw}
            for name, mk, fs in (("taker", False, None), ("maker", True, None), ("mid", True, 0.0)):
                p = strip_pnl(legs, value, side=side, maker=mk, fee_share=fs)
                r |= {f"pnl_{name}": p["pnl"] * p["step"], f"bps_{name}": 1e4 * p["pnl"] * p["step"] / S,
                      f"ret_{name}": p["pnl"] / p["collateral"], f"legs_{name}": p["legs"]}
            rows.append(r)
    return pd.DataFrame(rows)


def calendar_trades(root: Path, cum: pd.Series, hours_before: float = 23, wing_sd: float = WING_SD) -> pd.DataFrame:
    """Kalshi calendar: short the daily strip `hours_before` its close, long the weekly strip on the same dollar range
    at the same minute; the daily settles, the weekly is sold back at the daily's close (taker both ways).

    One row per daily, with each leg's P&L and the total, per 1 BTC of straddle and in bps of spot."""
    dm, wm = kalshi_markets(root, "daily"), kalshi_markets(root, "weekly")
    w_close = wm.groupby("event_ticker").close_time.max()
    rows = []
    for ev, m in dm.groupby("event_ticker"):
        close = m.close_time.max()
        t = (close - pd.Timedelta(hours=hours_before)).floor("min")
        wk = w_close[(w_close > close) & (w_close <= close + pd.Timedelta(days=5))]
        if wk.empty:
            continue
        wev = wk.idxmin()
        dq, wq = kalshi_quotes(root, "daily", ev), kalshi_quotes(root, "weekly", wev)
        if dq.empty:
            continue
        dl, wl, wx = ladder_at(dq, t), ladder_at(wq, t), ladder_at(wq, close.floor("min"))
        if dl.empty or wl.empty or wx.empty or not np.isfinite(dl.spot.iloc[0]):
            continue
        S = float(dl.spot.iloc[0])
        rv = trailing_rv(cum, [t])[0]
        hw = wing_sd * rv * np.sqrt(hours_before * 3600 / YEAR_SECONDS) * S
        dlegs, wlegs = strip_legs(dl, S, hw), strip_legs(wl, S, hw)
        short = strip_pnl(dlegs, m.set_index("strike").settle, side=-1)
        long = strip_pnl(wlegs, pd.Series(dtype=float), side=+1, exit_quotes=wx)
        if not (np.isfinite(short["pnl"]) and np.isfinite(long["pnl"])):
            continue
        # before costs: in and out at mid, no fees
        short0 = strip_pnl(dlegs, m.set_index("strike").settle, side=-1, maker=True, fee_share=0.0)
        long0 = strip_pnl(wlegs, pd.Series(dtype=float), side=+1, maker=True, fee_share=0.0,
                          exit_quotes=mid_exit(wx))
        s, l = short["pnl"] * short["step"], long["pnl"] * long["step"]
        s0, l0 = short0["pnl"] * short0["step"], long0["pnl"] * long0["step"]
        rows.append({"event": ev, "weekly": wev, "entry": t, "spot": S, "short_daily": s, "long_weekly": l,
                     "pnl": s + l, "pnl_bps": 1e4 * (s + l) / S, "short_bps": 1e4 * s / S, "long_bps": 1e4 * l / S,
                     "short_mid_bps": 1e4 * s0 / S, "long_mid_bps": 1e4 * l0 / S, "pnl_mid_bps": 1e4 * (s0 + l0) / S,
                     "weekday": close.tz_convert(cleaning.ET).day_name()})
    return pd.DataFrame(rows)


# --- dislocations --------------------------------------------------------------------------------


def nearest_snapshot(kalshi: pd.DataFrame, ts, tolerance: pd.Timedelta = SNAP_TOLERANCE) -> pd.DataFrame:
    """For each (event, ts) wanted, the Kalshi ATM snapshot of that event nearest in time, within `tolerance`."""
    return pd.merge_asof(ts.sort_values("ts"), kalshi.sort_values("ts").rename(columns={"ts": "k_ts"}),
                         left_on="ts", right_on="k_ts", by="event", direction="nearest", tolerance=tolerance)


def dislocation_panel(kalshi: pd.DataFrame, deribit_obs: pd.DataFrame, hour_profile: pd.Series,
                      product: str) -> pd.DataFrame:
    """Kalshi vs Deribit ATM vol at each Deribit afternoon window, with the dislocation.

    `kalshi`: ATM snapshots (`event`, `ts`, `expiry`, `atm_iv`, unfiltered, with `strikes_per_sd`). `deribit_obs`:
    `term.deribit_atm` rows. Pairing: a weekly (Friday 5pm ET) with Deribit's Friday 08:00 expiry, a daily
    (Monday-Thursday 5pm ET) with the next morning's 08:00 expiry. Deribit's vol is stretched to Kalshi's expiry.
    `gap` (points), `normal` (median gap of the previous 30 days, same whole number of days left), `disloc`.
    """
    d = deribit_obs[deribit_obs.window == "us_afternoon"].rename(columns={"expiry": "deribit_expiry",
                                                                          "atm_iv": "deribit_iv", "ts": "ts"})
    k_exp = kalshi.groupby("event").expiry.max().rename("kalshi_expiry").reset_index()
    if product == "weekly":
        k_exp["deribit_expiry"] = k_exp.kalshi_expiry.dt.normalize() + pd.Timedelta(hours=8)
    else:
        k_exp["deribit_expiry"] = k_exp.kalshi_expiry.dt.normalize() + pd.Timedelta(days=1, hours=8)
    p = d[["deribit_expiry", "ts", "deribit_iv", "n_trades"]].merge(k_exp, on="deribit_expiry")
    p = p[p.ts < p.kalshi_expiry - pd.Timedelta(hours=1)]
    p = nearest_snapshot(kalshi[["event", "ts", "atm_iv", "strikes_per_sd"]], p).dropna(subset=["atm_iv"])
    p = p.rename(columns={"atm_iv": "kalshi_iv"})
    p["ratio"] = deribit.horizon_vol_ratios(p.ts, p.kalshi_expiry, p.deribit_expiry, hour_profile)
    p["deribit_adj"] = p.deribit_iv * p.ratio
    p["gap"] = 100 * (p.kalshi_iv - p.deribit_adj)
    p["kalshi_hours"] = (p.kalshi_expiry - p.k_ts).dt.total_seconds() / 3600
    p["deribit_hours"] = (p.deribit_expiry - p.ts).dt.total_seconds() / 3600
    p["days_left"] = np.round(p.kalshi_hours / 24).astype(int)
    p["product"] = product
    p = p.sort_values("ts").reset_index(drop=True)
    p["normal"] = trailing_normal(p)
    p["disloc"] = p.gap - p.normal
    return p


def trailing_normal(p: pd.DataFrame, days: int = DISLOC_LOOKBACK_DAYS, min_obs: int = DISLOC_MIN_OBS) -> np.ndarray:
    """The normal gap for each row: the gap's usual level at its `days_left` (full-sample median, the term shape),
    plus the median deviation from that shape over the previous `days` days (strictly before the row's day).

    Pooling horizons is needed because a weekly has one observation per days-left, so a 30-day window holds only
    four or five at each. The shape uses the whole sample (mild look-ahead in the shape, none in the level)."""
    shape = p.groupby("days_left").gap.transform("median")
    resid = (p.gap - shape).to_numpy()
    day = p.ts.dt.floor("D").to_numpy()
    out = np.full(len(p), np.nan)
    for i in range(len(p)):
        prior = (day < day[i]) & (day >= day[i] - np.timedelta64(days, "D"))
        if prior.sum() >= min_obs:
            out[i] = shape.iloc[i] + np.median(resid[prior])
    return out


def dislocation_trades(panel: pd.DataFrame, root: Path, spot: pd.Series, cum: pd.Series,
                       product: str, require_disloc: bool = True) -> pd.DataFrame:
    """Every panel row priced as four trades held to expiry, so signals can be applied afterwards:
    Deribit straddle sold (`deribit_short_*`) and Kalshi strip sold, taker and maker (`kalshi_short_*`), in bps of
    spot per 1 BTC of straddle (buying is minus selling, before costs; `kalshi_*_mid` is at mid with no fee). Also
    implied - realized for both markets. Rows without a dislocation (the first 30 days) are dropped unless
    `require_disloc=False`."""
    markets = kalshi_markets(root, product)
    value = {e: g.set_index("strike").settle for e, g in markets.groupby("event_ticker")}
    p = (panel.dropna(subset=["disloc"]) if require_disloc else panel).copy()
    p["S"] = spot.reindex(p.k_ts).to_numpy()
    p["S_T"] = deribit_delivery(spot, p.deribit_expiry)
    tau_d = p.deribit_hours.to_numpy() * 3600 / YEAR_SECONDS
    for side, name in ((-1, "short"), (+1, "long")):
        dr = deribit_straddle_pnl(p.S, p.deribit_iv, tau_d, p.S_T, side=side)
        p[f"deribit_{name}_bps"] = 1e4 * dr["pnl"] / p.S
        p[f"deribit_{name}_gross_bps"] = 1e4 * dr["gross"] / p.S
    p["rv_deribit"] = term.realized_vol(cum, p.ts, p.deribit_expiry)
    p["rv_kalshi"] = term.realized_vol(cum, p.k_ts, p.kalshi_expiry)
    p["deribit_err"] = 100 * (p.deribit_iv - p.rv_deribit)
    p["kalshi_err"] = 100 * (p.kalshi_iv - p.rv_kalshi)
    p["rv_entry"] = trailing_rv(cum, p.k_ts)

    cols = {f"kalshi_{s}_{c}": [] for s in ("short", "long") for c in ("taker", "maker", "mid")}
    for ev, g in p.groupby("event"):
        q = kalshi_quotes(root, product, ev)
        q = q[q.ts.isin(set(g.k_ts))]
        for i, r in g.iterrows():
            lad = ladder_at(q, r.k_ts)
            hw = WING_SD * r.rv_entry * np.sqrt(r.kalshi_hours * 3600 / YEAR_SECONDS) * r.S
            legs = strip_legs(lad, r.S, hw) if not lad.empty else lad
            for s, side in (("short", -1), ("long", +1)):
                for c, mk, fs in (("taker", False, None), ("maker", True, None), ("mid", True, 0.0)):
                    x = (strip_pnl(legs, value[ev], side=side, maker=mk, fee_share=fs) if len(legs)
                         else {"pnl": np.nan, "step": np.nan})
                    cols[f"kalshi_{s}_{c}"].append((i, 1e4 * x["pnl"] * x["step"] / r.S))
    for c, v in cols.items():
        p[f"{c}_bps"] = pd.Series(dict(v))
    p["entry"] = p.k_ts
    return p


def gap_calendar(weekly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Long the one-week Kalshi-Deribit vol spread, short the daily one, entered in the same afternoon window.

    `weekly`, `daily`: `dislocation_trades` rows (with `require_disloc=False`). Weekly spread bought = Kalshi weekly
    strip bought + Deribit Friday straddle sold; daily spread sold = Kalshi daily strip sold + Deribit next-morning
    straddle bought. Each leg held to its own expiry; all four per 1 BTC of straddle. One row per day: the daily
    with about a day left (`days_left` 1) and the weekly that afternoon (dropped on Fridays, when no daily trades).
    P&L columns in bps of spot: `week_spread_*`, `day_spread_*` and `total_*` for `taker` (Kalshi at bid/ask and
    fee, Deribit fees and the assumed half-spread) and `mid` (before costs on both)."""
    key = lambda x: x.assign(day=x.ts.dt.floor("D"))
    w = key(weekly)
    w = w[w.days_left >= 1].sort_values("ts").drop_duplicates("day")
    d = key(daily)
    d = d[d.days_left == 1].sort_values("ts").drop_duplicates("day")
    m = w.merge(d, on="day", suffixes=("_w", "_d"))
    out = pd.DataFrame({"day": m.day, "entry": m.entry_w, "weekly": m.event_w, "daily": m.event_d,
                        "weekly_days_left": m.days_left_w, "gap_w": m.gap_w, "gap_d": m.gap_d,
                        "gap_diff": m.gap_d - m.gap_w})
    for c, kc, dc in (("taker", "taker", ""), ("mid", "mid", "_gross")):
        out[f"k_week_{c}"] = m[f"kalshi_long_{kc}_bps_w"]
        out[f"d_week_{c}"] = m[f"deribit_short{dc}_bps_w"]
        out[f"k_day_{c}"] = m[f"kalshi_short_{kc}_bps_d"]
        out[f"d_day_{c}"] = m[f"deribit_long{dc}_bps_d"]
        out[f"week_spread_{c}"] = out[f"k_week_{c}"] + out[f"d_week_{c}"]
        out[f"day_spread_{c}"] = out[f"k_day_{c}"] + out[f"d_day_{c}"]
        out[f"total_{c}"] = out[f"week_spread_{c}"] + out[f"day_spread_{c}"]
    return out.dropna(subset=["total_taker"]).reset_index(drop=True)


def first_signal(trades: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """One trade per contract: the first row where the signal fires."""
    return trades[mask].sort_values("entry").drop_duplicates("event")


# --- trading the Kalshi-Deribit gap as it widens into expiry --------------------------------------


def straddle_legs(S, K, sigma, tau):
    """Call and put prices and their (equal) vega per 1 BTC, at spot S (forward = spot) and strike K."""
    S, K, sigma, tau = (np.asarray(x, dtype=float) for x in (S, K, sigma, tau))
    c, p = iv.bs_call(S, K, sigma, tau), iv.bs_put(S, K, sigma, tau)
    d1 = (np.log(S / K) + sigma**2 * tau / 2) / (sigma * np.sqrt(tau))
    vega = S * np.sqrt(tau) * np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi)
    return c, p, vega


def deribit_straddle_mtm(K, S0, sigma0, tau0, S1, sigma1, tau1, side: int = -1,
                         half_spread_vol: float = DERIBIT_HALF_SPREAD_VOL) -> dict:
    """A Deribit straddle at strike K opened at (S0, sigma0, tau0) and closed at (S1, sigma1, tau1), per 1 BTC.
    `side` -1 = sold. Fees (0.03% of spot, at most 12.5% of the price, per leg) and the assumed half-spread are
    paid on both trades. Pricing at the ATM vol ignores the smile once spot has moved away from K."""
    c0, p0, v0 = straddle_legs(S0, K, sigma0, tau0)
    c1, p1, v1 = straddle_legs(S1, K, sigma1, tau1)
    gross = side * ((c1 + p1) - (c0 + p0))
    fee = lambda S, c, p: np.minimum(term.DERIBIT_FEE_RATE * S, term.DERIBIT_FEE_CAP * c) + \
        np.minimum(term.DERIBIT_FEE_RATE * S, term.DERIBIT_FEE_CAP * p)
    costs = fee(S0, c0, p0) + fee(S1, c1, p1) + half_spread_vol * 2 * (v0 + v1)
    return dict(gross=gross, pnl=gross - costs, costs=costs)


def fly_value(S, K, w, sigma, tau):
    """Long iron butterfly per 1 BTC: straddle at K minus a strangle at K -/+ w. Pays min(|S_T - K|, w), the payoff of
    a Kalshi strip over the same range. All four options at one vol (`sigma`, Deribit's ATM vol: no smile).
    Returns value, the four option prices (call K, put K, call K+w, put K-w) and their vegas."""
    S, K, w, sigma, tau = (np.asarray(x, dtype=float) for x in (S, K, w, sigma, tau))
    cK, pK, vK = straddle_legs(S, K, sigma, tau)
    cH, _, vH = straddle_legs(S, K + w, sigma, tau)
    _, pL, vL = straddle_legs(S, K - w, sigma, tau)
    return cK + pK - cH - pL, (cK, pK, cH, pL), (vK, vK, vH, vL)


def deribit_fly_mtm(K, w, S0, sigma0, tau0, S1, sigma1, tau1, side: int = -1,
                    half_spread_vol: float = DERIBIT_HALF_SPREAD_VOL) -> dict:
    """The iron butterfly of `fly_value` opened at (S0, sigma0, tau0), closed at (S1, sigma1, tau1), per 1 BTC.
    `side` -1 = sold (straddle sold, wings bought). Fees per option are 0.03% of spot, at most 12.5% of its price;
    as one combo with legs both bought and sold, Deribit waives the cheaper direction's fees. Plus the assumed
    half-spread on every option, in and out."""
    def costs(S, prices, vegas):
        f = [np.minimum(term.DERIBIT_FEE_RATE * S, term.DERIBIT_FEE_CAP * p) for p in prices]
        return np.maximum(f[0] + f[1], f[2] + f[3]) + half_spread_vol * sum(vegas)
    v0, pr0, ve0 = fly_value(S0, K, w, sigma0, tau0)
    v1, pr1, ve1 = fly_value(S1, K, w, sigma1, tau1)
    gross = side * (v1 - v0)
    c = costs(S0, pr0, ve0) + costs(S1, pr1, ve1)
    return dict(gross=gross, pnl=gross - c, costs=c)


def gap_widening_trades(panel: pd.DataFrame, root: Path, spot: pd.Series, cum: pd.Series,
                        entry_days=(7, 6, 5), exit_days: int = 1, wing_sd: float = WING_SD) -> pd.DataFrame:
    """Trade A: buy the Kalshi-Deribit vol spread on one weekly about a week out, sell it back when the gap has
    widened.

    For each Kalshi weekly (Friday 5pm ET) and Deribit's Friday 08:00 UTC expiry of the same week, both read in the
    US afternoon window (`panel` from `dislocation_panel(..., "weekly")`):
    - entry, the first window with `entry_days` left (7, else 6, else 5): buy the Kalshi strip (every strike within
      +/- w = `wing_sd` SDs of spot) and sell the Deribit iron butterfly with the same payoff (straddle at spot,
      wings at spot -/+ w, 1 BTC; `deribit_fly_mtm`), at Deribit's ATM vol;
    - exit, the window with `exit_days` left (Thursday afternoon, Deribit about 11 hours from expiry): sell the same
      Kalshi strikes back and buy the butterfly back at Deribit's ATM vol then.
    `*_mid`: Kalshi in and out at mid with no fee, Deribit with no fee or spread. `*_taker`: Kalshi at bid/ask and
    fee, Deribit fees and the assumed half-spread, both ways. Also `gap_in`, `gap_out` (points, as in the panel)
    and `pred_bps`, the P&L the two gaps imply: vega at exit x gap at exit - vega at entry x gap at entry.
    """
    rows = []
    for ev, g in panel.groupby("event"):
        ent = g[g.days_left.isin(entry_days)].sort_values("k_ts").head(1)
        ext = g[g.days_left == exit_days].sort_values("k_ts").tail(1)
        if ent.empty or ext.empty:
            continue
        e, x = ent.iloc[0], ext.iloc[0]
        q = kalshi_quotes(root, "weekly", ev)
        lad0, lad1 = ladder_at(q, e.k_ts), ladder_at(q, x.k_ts)
        S0, S1 = spot.get(e.k_ts, np.nan), spot.get(x.k_ts, np.nan)
        if lad0.empty or lad1.empty or not (np.isfinite(S0) and np.isfinite(S1)):
            continue
        rv = trailing_rv(cum, [e.k_ts])[0]
        w = wing_sd * rv * np.sqrt(e.kalshi_hours * 3600 / YEAR_SECONDS) * S0
        legs = strip_legs(lad0, S0, w)
        kt = strip_pnl(legs, pd.Series(dtype=float), side=+1, exit_quotes=lad1)
        km = strip_pnl(legs, pd.Series(dtype=float), side=+1, maker=True, fee_share=0.0, exit_quotes=mid_exit(lad1))
        tau = lambda r: (r.deribit_expiry - r.k_ts).total_seconds() / YEAR_SECONDS
        dr = deribit_fly_mtm(S0, w, S0, e.deribit_iv, tau(e), S1, x.deribit_iv, tau(x), side=-1)
        # what the gaps alone are worth: the fly's vega (at Kalshi's horizon) times the gap, at exit minus at entry
        fv = lambda r, S: sum(fly_value(S, S0, w, r.kalshi_iv, r.kalshi_hours * 3600 / YEAR_SECONDS)[2][:2]) - \
            sum(fly_value(S, S0, w, r.kalshi_iv, r.kalshi_hours * 3600 / YEAR_SECONDS)[2][2:])
        pred = fv(x, S1) * x.gap / 100 - fv(e, S0) * e.gap / 100
        bps = lambda v: 1e4 * v / S0
        rows.append({"event": ev, "entry": e.k_ts, "exit": x.k_ts, "days_in": e.days_left, "spot_in": S0,
                     "spot_out": S1, "gap_in": e.gap, "gap_out": x.gap, "legs": kt["legs"],
                     "kalshi_mid": bps(km["pnl"] * km["step"]), "kalshi_taker": bps(kt["pnl"] * kt["step"]),
                     "deribit_mid": bps(dr["gross"]), "deribit_taker": bps(dr["pnl"]), "pred_bps": bps(pred)})
    t = pd.DataFrame(rows)
    t["total_mid"] = t.kalshi_mid + t.deribit_mid
    t["total_taker"] = t.kalshi_taker + t.deribit_taker
    return t


def deribit_atm_at(trades: pd.DataFrame, expiry: pd.Timestamp, t: pd.Timestamp, window_min: int = 30,
                   atm_sd: float = 0.25, min_trades: int = 3) -> float:
    """Deribit's ATM vol for one expiry at time `t`: the median mark vol of its trades within `window_min` minutes
    of `t` and `atm_sd` standard deviations of the money (NaN with fewer than `min_trades`)."""
    g = trades[(trades.expiry == expiry) & ((trades.ts - t).abs() <= pd.Timedelta(minutes=window_min))]
    g = g[g.iv_mark.notna() & ((g.log_moneyness / (g.iv_mark * np.sqrt(g.tau_years))).abs() <= atm_sd)]
    return float(g.iv_mark.median()) if len(g) >= min_trades else np.nan


def last_hours_trades(root: Path, deribit_trades: pd.DataFrame, spot: pd.Series, cum: pd.Series,
                      hours_before: float = 2.0, product: str = "daily", wing_sd: float = WING_SD) -> pd.DataFrame:
    """Trade B: sell Kalshi in its last hours and buy Deribit's next 08:00 UTC straddle, both held to expiry.

    Each Kalshi contract of `product` (dailies Monday-Thursday, weeklies Friday; 5pm ET), entered `hours_before` its
    close (rounded down to the quarter hour): sell the Kalshi strip (+/- `wing_sd` SDs) and buy 1 BTC of Deribit's ATM
    straddle expiring 08:00 UTC the next morning (Saturday for the weekly), at Deribit's ATM vol from that expiry's
    trades within 30 minutes of entry (`deribit_atm_at`; `deribit_trades` from `deribit.load_trades`). Kalshi settles
    on its result, Deribit on the 30-minute average spot before 08:00. Columns as in `gap_widening_trades`."""
    markets = kalshi_markets(root, product)
    tr = deribit_trades[deribit_trades.iv_mark.notna() & (deribit_trades.tau_years > 0)]
    by_exp = {e: g for e, g in tr.groupby("expiry")}
    rows = []
    for ev, m in markets.groupby("event_ticker"):
        close = m.close_time.max()
        t = (close - pd.Timedelta(hours=hours_before)).floor("15min")
        if not (term.DERIBIT_PERIOD[0] <= t < term.DERIBIT_PERIOD[1]):
            continue
        d_exp = close.normalize() + pd.Timedelta(days=1, hours=8)
        sig_d = deribit_atm_at(by_exp[d_exp], d_exp, t) if d_exp in by_exp else np.nan
        q = kalshi_quotes(root, product, ev)
        lad = ladder_at(q, t)
        S = spot.get(t, np.nan)
        if lad.empty or not (np.isfinite(S) and np.isfinite(sig_d)):
            continue
        hours = (close - t).total_seconds() / 3600
        rv = trailing_rv(cum, [t])[0]
        legs = strip_legs(lad, S, wing_sd * rv * np.sqrt(hours * 3600 / YEAR_SECONDS) * S)
        val = m.set_index("strike").settle
        kt = strip_pnl(legs, val, side=-1)
        km = strip_pnl(legs, val, side=-1, maker=True, fee_share=0.0)
        S_T = deribit_delivery(spot, [d_exp])[0]
        dr = deribit_straddle_pnl(S, sig_d, (d_exp - t).total_seconds() / YEAR_SECONDS, S_T, side=+1)
        bps = lambda v: 1e4 * v / S
        rows.append({"event": ev, "product": product, "entry": t, "hours_left": hours, "spot": S, "deribit_iv": sig_d,
                     "legs": kt["legs"], "kalshi_mid": bps(km["pnl"] * km["step"]),
                     "kalshi_taker": bps(kt["pnl"] * kt["step"]), "deribit_mid": bps(dr["gross"]),
                     "deribit_taker": bps(dr["pnl"]), "weekday": close.tz_convert(cleaning.ET).day_name()})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows).dropna(subset=["kalshi_mid"])
    t["total_mid"] = t.kalshi_mid + t.deribit_mid
    t["total_taker"] = t.kalshi_taker + t.deribit_taker
    return t



def _mark_at(trades: pd.DataFrame, strike: float, cp: str, t: pd.Timestamp) -> tuple[float, float, float]:
    """Mark price (USD), index and mark vol of the trade in this option nearest to `t`."""
    g = trades[(trades.strike == strike) & (trades.cp == cp)]
    r = g.loc[(g.ts - t).abs().idxmin()]
    return float(r.mark_price_btc * r.index_price), float(r.index_price), float(r.iv_mark)


def gap_widening_traded(trade_a: pd.DataFrame, deribit_trades: pd.DataFrame, root: Path, spot: pd.Series,
                        cum: pd.Series, inner_max: float = 2_000.0, min_wing: float = 1_000.0,
                        wing_sd: float = WING_SD, window_min: int | None = None) -> pd.DataFrame:
    """Trade A with only contracts that actually traded: no model prices.

    For each week of `trade_a` (entry and exit times from `gap_widening_trades`), Deribit's Friday 08:00 options
    that traded in the US-afternoon window on *both* the entry and the exit day (`deribit_trades` from
    `deribit.load_trades`), or, with `window_min`, within that many minutes of the entry and of the exit time. Sell an iron condor of them: the traded call nearest above spot and put nearest below
    (within `inner_max`), buy the furthest traded call and put out to `wing_sd` SDs (at least `min_wing` beyond the
    inner strikes). Each option is priced at the mark of its own trade nearest the entry or exit minute.
    Kalshi: buy YES on the weekly strikes in (call strike, call wing], sell YES on those in [put wing, put strike):
    the same payoff (a long condor), in and out at Kalshi's quotes. Weeks without such a condor are skipped.

    Per 1 BTC, in dollars and bps of entry spot: `*_mid` (Kalshi at mid, no fee; Deribit at mark, no fee),
    `*_taker` (Kalshi bid/ask and fee; Deribit fees with the combo discount and the assumed half-spread),
    `*_maker` (Kalshi at mid with a quarter fee; Deribit as taker).
    """
    tr = deribit_trades[deribit_trades.mark_price_btc.notna()]
    if window_min is None:
        tr = tr[tr.window == "us_afternoon"]
    rows = []
    for r in trade_a.itertuples():
        exp = (r.exit + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=8)
        if window_min is None:
            e = tr[(tr.expiry == exp) & (tr.ts.dt.normalize() == r.entry.normalize())]
            x = tr[(tr.expiry == exp) & (tr.ts.dt.normalize() == r.exit.normalize())]
        else:
            near = lambda t: (tr.expiry == exp) & ((tr.ts - t).abs() <= pd.Timedelta(minutes=window_min))
            e, x = tr[near(r.entry)], tr[near(r.exit)]
        both = set(zip(e.strike, e.cp)) & set(zip(x.strike, x.cp))
        C = sorted(k for k, c in both if c == "C")
        P = sorted(k for k, c in both if c == "P")
        S0 = r.spot_in
        w = wing_sd * trailing_rv(cum, [r.entry])[0] * np.sqrt((r.exit + pd.Timedelta(days=1) - r.entry).total_seconds()
                                                               / YEAR_SECONDS) * S0
        kc = [k for k in C if S0 <= k <= S0 + inner_max]
        kp = [k for k in P if S0 - inner_max <= k <= S0]
        if not (kc and kp):
            continue
        kc, kp = min(kc), max(kp)
        cw = [k for k in C if kc + min_wing <= k <= S0 + w]
        pw = [k for k in P if S0 - w <= k <= kp - min_wing]
        if not (cw and pw):
            continue
        cw, pw = max(cw), min(pw)
        # Deribit: sell the inner call and put, buy the wings; each at its own mark, in and out
        opts = [(kc, "C", -1), (kp, "P", -1), (cw, "C", +1), (pw, "P", +1)]
        gross = 0.0
        fees_sell, fees_buy, spread = [0.0, 0.0], [0.0, 0.0], 0.0
        for k, cp, side in opts:
            for j, (book, t) in enumerate(((e, r.entry), (x, r.exit))):
                price, idx, vol = _mark_at(book, k, cp, t)
                gross += side * price * (-1 if j == 0 else 1)          # pay/receive at entry, reverse at exit
                fee = min(term.DERIBIT_FEE_RATE * idx, term.DERIBIT_FEE_CAP * price)
                # combo discount: the cheaper direction's fees are waived (at entry the inner legs are sold)
                (fees_sell if (side < 0) == (j == 0) else fees_buy)[j] += fee
                tau = (exp - t).total_seconds() / YEAR_SECONDS
                spread += DERIBIT_HALF_SPREAD_VOL * straddle_legs(idx, k, vol, tau)[2]
        d_cost = sum(max(a, b) for a, b in zip(fees_sell, fees_buy)) + spread
        # Kalshi: the same long condor from the weekly's strikes, in and out at its quotes
        q = kalshi_quotes(root, "weekly", r.event)
        lad0, lad1 = ladder_at(q, r.entry), ladder_at(q, r.exit)
        legs = lad0[((lad0.strike > kc) & (lad0.strike <= cw)) | ((lad0.strike >= pw) & (lad0.strike < kp))].assign(
            up=lambda z: z.strike > S0)
        out = {}
        for name, mk, fs, ex in (("mid", True, 0.0, mid_exit(lad1)), ("taker", False, None, lad1),
                                 ("maker", True, None, mid_exit(lad1))):
            k = strip_pnl(legs, pd.Series(dtype=float), side=+1, maker=mk, fee_share=fs, exit_quotes=ex)
            out[name] = k["pnl"] * k["step"]
        dm = {"mid": gross, "taker": gross - d_cost, "maker": gross - d_cost}
        row = {"event": r.event, "entry": r.entry, "exit": r.exit, "spot_in": S0, "gap_in": r.gap_in,
               "gap_out": r.gap_out, "put_wing": pw, "put": kp, "call": kc, "call_wing": cw, "kalshi_legs": len(legs)}
        for c in ("mid", "taker", "maker"):
            row |= {f"kalshi_{c}_usd": out[c], f"deribit_{c}_usd": dm[c], f"total_{c}_usd": out[c] + dm[c],
                    f"total_{c}": 1e4 * (out[c] + dm[c]) / S0}
        rows.append(row)
    return pd.DataFrame(rows)
