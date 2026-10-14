"""Kalshi vs Deribit implied vol as a time series, lead-lag, and a long/short vol-spread backtest (09_vol_spread.py).

05 paired single Deribit trades with Kalshi in the US afternoon. With Deribit's full-day trades
(00_fetch_deribit.py) Deribit's ATM vol exists every hour (`deribit_marks.smile_grid`), so both markets become hourly
series and can be compared over time:

1. Kalshi: the ATM vol of every hourly snapshot of a daily or weekly event (`kalshi_hourly.smiles`, built as in 03).
2. Deribit at Kalshi's expiry. Kalshi expires at 5pm ET, Deribit at 08:00 UTC. Deribit's total implied variance
   sigma^2 tau of its listed expiries is interpolated to Kalshi's expiry along a "business clock" that lets each
   hour carry its usual share of BTC variance (05's hour-of-week profile, `deribit.hourly_variance`):
   - bracketed: with an expiry before and one after Kalshi's (both within `MAX_BRACKET_DAYS`), linear in
     business time between them. This uses Deribit's own prices for the 13 hours instead of history alone;
   - otherwise the nearest expiry, stretched at its own business-time rate: 05's adjustment.
3. Gap = Kalshi - Deribit (vol points).

Then: who moves first (lead-lag of hourly changes), who closes a gap (error correction), whether Kalshi overreacts
when vol jumps (asymmetry), who forecasts realized vol better, and a long/short trade on the gap.
"""

import numpy as np
import pandas as pd

from lib import deribit, kalshi_hourly

MAX_BRACKET_DAYS = 2.0
HOUR = pd.Timedelta(hours=1)
YEAR_SECONDS = 365 * 86_400


def business_clock(ts, profile: pd.Series) -> np.ndarray:
    """Cumulative expected variance (hour-of-week profile) at each time; differences give expected variance."""
    return deribit._cumulative_variance(pd.DatetimeIndex(ts), profile)


def deribit_at_kalshi(snaps: pd.DataFrame, grid: pd.DataFrame, profile: pd.Series,
                      max_bracket_days: float = MAX_BRACKET_DAYS) -> pd.DataFrame:
    """Deribit's ATM vol interpolated to each Kalshi snapshot's expiry.

    `snaps`: ts, expiry, tau_years (Kalshi). `grid`: `smile_grid` output (expiry, t, atm_iv, tau_years). Adds
    `deribit_iv` (at Kalshi's expiry), `method` ("bracket", "stretch_before", "stretch_after"), `deribit_near`
    (ATM vol of the Deribit expiry closest to Kalshi's, unadjusted), `near_expiry`, `near_hours_diff`.
    """
    by_t = {t: g for t, g in grid.groupby("t")}
    out = []
    for r in snaps.itertuples(index=False):
        g = by_t.get(r.ts)
        if g is None:
            out.append((np.nan, None, np.nan, pd.NaT, np.nan))
            continue
        e = g.expiry.to_numpy()
        v = (g.atm_iv.to_numpy() ** 2) * g.tau_years.to_numpy()
        before, after = e <= r.expiry, e > r.expiry
        lim = pd.Timedelta(days=max_bracket_days)
        ia = np.flatnonzero(before & (e >= r.expiry - lim))
        ib = np.flatnonzero(after & (e <= r.expiry + lim))
        ia = ia[np.argmax(e[ia])] if len(ia) else None
        ib = ib[np.argmin(e[ib])] if len(ib) else None
        W = lambda x: business_clock([x], profile)[0]
        w_t, w_k = W(r.ts), W(r.expiry)
        if ia is not None and ib is not None:
            wa, wb = W(e[ia]), W(e[ib])
            V = v[ia] + (v[ib] - v[ia]) * (w_k - wa) / (wb - wa)
            method = "bracket"
        elif ia is not None or ib is not None:
            i = ia if ia is not None else ib
            V = v[i] * (w_k - w_t) / (W(e[i]) - w_t)
            method = "stretch_before" if ia is not None else "stretch_after"
        else:
            V, method = np.nan, None
        near = np.argmin(np.abs((g.expiry - r.expiry).dt.total_seconds().to_numpy()))
        out.append((np.sqrt(V / r.tau_years) if np.isfinite(V) and V > 0 else np.nan, method,
                    g.atm_iv.iloc[near], g.expiry.iloc[near], (g.expiry.iloc[near] - r.expiry).total_seconds() / 3600))
    o = pd.DataFrame(out, columns=["deribit_iv", "method", "deribit_near", "near_expiry", "near_hours_diff"],
                     index=snaps.index)
    return snaps.join(o)


def realized_vol(spot: pd.Series, start, end) -> np.ndarray:
    """Annualised realized vol from 1-minute log returns between each start and end (spot indexed by minute end)."""
    r2 = np.log(spot).diff().pow(2).fillna(0)
    c = r2.cumsum()
    s0 = c.reindex(pd.DatetimeIndex(start), method="ffill").to_numpy()
    s1 = c.reindex(pd.DatetimeIndex(end), method="ffill").to_numpy()
    yrs = (pd.DatetimeIndex(end) - pd.DatetimeIndex(start)).total_seconds().to_numpy() / YEAR_SECONDS
    return np.sqrt(np.maximum(s1 - s0, 0) / yrs)


def changes(panel: pd.DataFrame, cols, lag_hours: int = 1) -> pd.DataFrame:
    """Within each event, change of `cols` over `lag_hours` (only between snapshots exactly that far apart)."""
    p = panel.sort_values(["event_ticker", "ts"])
    prev = p[["event_ticker", "ts", *cols]].copy()
    prev["ts"] = prev.ts + pd.Timedelta(hours=lag_hours)
    m = p.merge(prev, on=["event_ticker", "ts"], how="left", suffixes=("", "_prev"))
    for c in cols:
        m[f"d_{c}"] = m[c] - m[f"{c}_prev"]
    return m


def ols_clustered(y, X, groups, names=None) -> pd.DataFrame:
    """OLS with standard errors clustered by `groups` (e.g. day). X without a constant; one is added."""
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float).reshape(len(y), -1)])
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X, g = y[ok], X[ok], np.asarray(groups)[ok]
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for gg in pd.unique(g):
        s = X[g == gg].T @ e[g == gg]
        meat += np.outer(s, s)
    G = len(pd.unique(g))
    V = XtX_inv @ meat @ XtX_inv * G / max(G - 1, 1)
    se = np.sqrt(np.diag(V))
    names = ["const", *(names or [f"x{i}" for i in range(1, X.shape[1])])]
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return pd.DataFrame({"coef": b, "se": se, "t": b / se}, index=names).assign(n=len(y), clusters=G, r2=r2)


def kalshi_fee(price, n):
    """Kalshi taker fee for `n` contracts at `price`: 7% x n x P x (1 - P), rounded up to the cent (per strike order)."""
    p, n = np.asarray(price, dtype=float), np.asarray(n, dtype=float)
    return np.ceil(np.round(100 * 0.07 * n * p * (1 - p), 9)) / 100


def kalshi_straddle(ladder: pd.DataFrame, K: float) -> dict:
    """Kalshi's straddle at K from its binary ladder, at bid, mid and ask, and the taker fee to trade it.

    `ladder`: strike, bid, ask, mid (one event and snapshot). Uses `kalshi_hourly.strip_straddle`'s weights (dollars
    of strike per contract), so one strip pays like one straddle on 1 BTC within the listed strikes."""
    out = {s: kalshi_hourly.strip_straddle(ladder, K, s)["value"] for s in ("bid", "mid", "ask")}
    k = np.sort(ladder.strike.to_numpy(dtype=float))
    half = np.median(np.diff(k)) / 2
    lad = ladder.sort_values("strike")
    lo, hi = lad.strike.to_numpy() - half, lad.strike.to_numpy() + half
    w = np.clip(np.minimum(hi, K) - lo, 0, None) + np.clip(hi - np.maximum(lo, K), 0, None)
    px = np.where(lad.strike.to_numpy() >= K, lad.mid.to_numpy(), 1 - lad.mid.to_numpy())
    ok = np.isfinite(px) & (w > 0)
    out["fee"] = float(kalshi_fee(px[ok], w[ok]).sum())
    return out


def leadlag_xcorr(panel: pd.DataFrame, a: str = "atm_iv", b: str = "deribit_iv", lags=range(-4, 5)) -> pd.DataFrame:
    """corr(change in a at t, change in b at t + lag), hourly changes within each event, pooled. A peak at a
    positive lag means a moves first."""
    m = changes(panel, [a, b])
    m = m.set_index(["event_ticker", "ts"])
    da, db = m[f"d_{a}"], m[f"d_{b}"]
    out = []
    for L in lags:
        shifted = db.copy()
        shifted.index = pd.MultiIndex.from_arrays([db.index.get_level_values(0),
                                                   db.index.get_level_values(1) - pd.Timedelta(hours=L)])
        j = pd.concat([da.rename("a"), shifted.rename("b")], axis=1).dropna()
        out.append({"lag_hours": L, "corr": j.a.corr(j.b), "n": len(j)})
    return pd.DataFrame(out)


def forward_changes(panel: pd.DataFrame, cols, h: int) -> pd.DataFrame:
    """Add f{h}_{col}: change of col from t to t + h hours within the same event (NaN if no snapshot then)."""
    nxt = panel[["event_ticker", "ts", *cols]].copy()
    nxt["ts"] = nxt.ts - pd.Timedelta(hours=h)
    m = panel.merge(nxt, on=["event_ticker", "ts"], how="left", suffixes=("", "_next"))
    for c in cols:
        m[f"f{h}_{c}"] = m[f"{c}_next"] - m[c]
        m = m.drop(columns=f"{c}_next")
    return m


# ---------------------------------------------------------------------------------------------------------------
# Long/short backtest
# ---------------------------------------------------------------------------------------------------------------

SD_STRIP = 3.0        # the Kalshi strip covers strikes within this many ATM standard deviations of K


def strip_at(ladder: pd.DataFrame, K: float, width: float) -> pd.DataFrame:
    """The strikes (and contracts per strike) of Kalshi's straddle strip at K: usable two-sided quotes (02's rule 6)
    within `width` dollars of K. Contracts = dollars of strike in each strike's cell; a put cell buys "no", a call
    cell buys "yes". Returns strike, w (contracts), side ("yes"/"no")."""
    lad = ladder[np.isfinite(ladder.mid)].sort_values("strike")
    k = lad.strike.to_numpy(dtype=float)
    if len(k) < 3:
        return pd.DataFrame(columns=["strike", "w", "side"])
    half = np.median(np.diff(k)) / 2
    lo, hi = k - half, k + half
    w_put = np.clip(np.minimum(hi, K) - lo, 0, None)
    w_call = np.clip(hi - np.maximum(lo, K), 0, None)
    near = np.abs(k - K) <= width
    rows = [(kk, wc, "yes") for kk, wc, n in zip(k, w_call, near) if wc > 0 and n]
    rows += [(kk, wp, "no") for kk, wp, n in zip(k, w_put, near) if wp > 0 and n]
    return pd.DataFrame(rows, columns=["strike", "w", "side"])


def price_strip(strip: pd.DataFrame, ladder: pd.DataFrame, side: str) -> tuple[float, float]:
    """Value of a fixed strip on a ladder: (USD value, Kalshi fee to trade it). `side`: "mid", "bid" (we sell) or
    "ask" (we buy). A missing quote is valued at the worst bound for us: selling brings 0 (bid) / buying costs 1."""
    q = ladder.set_index("strike").reindex(strip.strike)
    bid, ask, mid = q.bid.to_numpy(), q.ask.to_numpy(), q.mid.to_numpy()
    yes = strip.side.to_numpy() == "yes"
    if side == "mid":
        m = np.where(np.isfinite(mid), mid, np.where(np.isfinite(bid) & np.isfinite(ask), (bid + ask) / 2,
                                                       np.where(np.isfinite(bid), bid, np.where(np.isfinite(ask), ask, np.nan))))
        px = np.where(yes, m, 1 - m)
    elif side == "bid":   # sell yes at yes bid, sell no at 1 - yes ask
        px = np.where(yes, np.nan_to_num(bid, nan=0.0), 1 - np.nan_to_num(ask, nan=1.0))
    else:                 # buy yes at yes ask, buy no at 1 - yes bid
        px = np.where(yes, np.nan_to_num(ask, nan=1.0), 1 - np.nan_to_num(bid, nan=0.0))
    w = strip.w.to_numpy()
    fee = float(kalshi_fee(np.clip(np.nan_to_num(px), 0, 1), w).sum()) if side != "mid" else 0.0
    return float(np.nansum(w * px)), fee


def _bs_straddle_greeks(F, K, sigma, tau):
    """(delta, vega per vol point) of a Black-Scholes straddle."""
    from lib import deribit_marks as dm
    d = dm.bs_delta(F, K, sigma, tau, True) + dm.bs_delta(F, K, sigma, tau, False)
    return float(d), float(2 * dm.bs_vega(F, K, sigma, tau) / 100)


def strip_width_sd(n_sd: float):
    """Set how far the Kalshi strip reaches, in ATM standard deviations (default SD_STRIP)."""
    global SD_STRIP
    SD_STRIP = n_sd


def backtest(panel: pd.DataFrame, quotes: pd.DataFrame, results: pd.DataFrame, mkt, cs_params, signal: str,
             threshold: float, hold, both_sides: bool = True, hedge: bool = True) -> pd.DataFrame:
    """Kalshi vs Deribit straddle spread, one trade per signal row.

    `panel`: Kalshi snapshots with `event_ticker, ts, expiry, forward, atm_iv, tau_years` and the `signal` column (vol
    points; positive = Kalshi rich). If signal > threshold: SELL Kalshi's straddle strip at K (at its bids, plus
    fees) and BUY `vega_K / vega_D` Deribit straddles at K (model value + half-spread + fee); if signal < -threshold
    and `both_sides`: the reverse. K = the Deribit strike nearest Kalshi's forward.
    `hold`: hours (int) — closed at the Kalshi snapshot `hold` hours later at the opposite side of each book — or
    "expiry": Kalshi's strip settles on its contracts' results, Deribit's straddle on the delivery price of the last
    Deribit expiry before Kalshi's; after it the Kalshi leg is hedged alone.
    The net delta (Black-Scholes, each leg at its own vol) is hedged hourly with the perpetual.
    `quotes`: kalshi_hourly.load_quotes rows; `results`: event_ticker, strike, result_yes; `mkt`:
    calendar_straddle.Market; `cs_params`: calendar_straddle.Params (for spread and fee settings).
    Each leg is hedged with its own perpetual position, so each leg's P&L also stands alone: `k_gross`/`k_cost`
    (the Kalshi leg: trade it on a Deribit signal) and `d_gross`/`d_cost` (the Deribit leg: trade it on a Kalshi
    signal; long Deribit when Kalshi is rich), and `gross`/`costs`/`net` for both legs together (one net hedge).
    USD per 1 BTC Kalshi strip; the Deribit leg is sized to the same vega (`vega_k`, USD per vol point).
    """
    from lib import calendar_straddle as cs

    lad_at = {k: g for k, g in quotes.groupby(["event_ticker", "ts"])}
    iv_at = panel.set_index(["event_ticker", "ts"])[["atm_iv", "forward"]].to_dict("index")
    res = results.set_index(["event_ticker", "strike"]).result_yes
    rows = []
    for r in panel.itertuples(index=False):
        s = getattr(r, signal)
        if not np.isfinite(s) or (s <= threshold and (not both_sides or s >= -threshold)):
            continue
        side_k = -1.0 if s > threshold else 1.0          # Kalshi position: -1 short, +1 long
        t = r.ts
        if hold == "expiry":
            exps = [e for e in mkt.by_expiry if r.expiry - pd.Timedelta(days=1.5) < e <= r.expiry and e > t + HOUR]
            if not exps:
                continue
            E = max(exps)
            t_exit = r.expiry
        else:
            t_exit = t + pd.Timedelta(hours=hold)
            if (r.event_ticker, t_exit) not in lad_at or t_exit > r.expiry - HOUR:
                continue
            exps = [e for e in mkt.by_expiry if e > t_exit and abs((e - r.expiry).total_seconds()) < 1.5 * 86400]
            if not exps:
                continue
            E = min(exps, key=lambda e: abs((e - r.expiry).total_seconds()))
        ks = mkt.strikes.get(E)
        if ks is None:
            continue
        K = float(ks[np.argmin(np.abs(ks - r.forward))])
        d0 = cs.straddle(mkt, E, K, t, 0.0)
        if d0 is None:
            continue
        lad0 = lad_at.get((r.event_ticker, t))
        width = SD_STRIP * r.atm_iv * np.sqrt(r.tau_years) * r.forward
        strip = strip_at(lad0, K, width)
        if len(strip) < 4:
            continue
        k_mid0, _ = price_strip(strip, lad0, "mid")
        k_ex0, k_fee0 = price_strip(strip, lad0, "bid" if side_k < 0 else "ask")
        dK0, vK = _bs_straddle_greeks(r.forward, K, r.atm_iv, r.tau_years)
        n_d = vK / d0[2]
        side_d = -side_k * n_d
        d_cost0 = cs.trade_cost(mkt, cs_params, n_d, d0[0], d0[5], d0[6], d0[4])
        # hedge hour by hour, each leg with its own perpetual position (so each leg's P&L stands alone)
        h_k = h_d = c_k = c_d = c_both = 0.0
        pos_k = pos_d = 0.0
        h, last_dk, last_dd = t, dK0, d0[1]
        while h < t_exit:
            kv = iv_at.get((r.event_ticker, h))
            if kv is not None:
                tau_h = (r.expiry - h).total_seconds() / YEAR_SECONDS
                last_dk = _bs_straddle_greeks(kv["forward"], K, kv["atm_iv"], tau_h)[0]
            if h < E:
                dd = cs.straddle(mkt, E, K, h, cs_params.max_stale_hours)
                if dd is not None:
                    last_dd = dd[1]
            tk = -side_k * last_dk if hedge else 0.0
            td = (-side_d * last_dd if h < E else 0.0) if hedge else 0.0
            px = mkt.perp_at(h)
            c_k += cs_params.perp_fee * abs(tk - pos_k) * px
            c_d += cs_params.perp_fee * abs(td - pos_d) * px
            c_both += cs_params.perp_fee * abs(tk + td - pos_k - pos_d) * px
            pos_k, pos_d = tk, td
            h2 = min(h + HOUR, t_exit)
            move = mkt.perp_at(h2) - px
            h_k += pos_k * move
            h_d += pos_d * move
            h = h2
        px = mkt.perp_at(t_exit)
        c_k += cs_params.perp_fee * abs(pos_k) * px
        c_d += cs_params.perp_fee * abs(pos_d) * px
        c_both += cs_params.perp_fee * abs(pos_k + pos_d) * px
        # exit
        if hold == "expiry":
            pays = res.reindex(pd.MultiIndex.from_arrays([[r.event_ticker] * len(strip), strip.strike])).to_numpy()
            if pd.isna(pays).any():
                continue
            pays = np.where(strip.side.to_numpy() == "yes", pays.astype(bool), ~pays.astype(bool))
            k_mid1 = k_ex1 = float((strip.w.to_numpy() * pays).sum())
            k_fee1 = 0.0
            st = mkt.settle(E)
            if not np.isfinite(st):
                continue
            d_val1, d_cost1 = abs(st - K), 0.0
        else:
            lad1 = lad_at[(r.event_ticker, t_exit)]
            k_mid1, _ = price_strip(strip, lad1, "mid")
            k_ex1, k_fee1 = price_strip(strip, lad1, "ask" if side_k < 0 else "bid")
            d1 = cs.straddle(mkt, E, K, t_exit, cs_params.max_stale_hours)
            if d1 is None:
                continue
            d_val1 = d1[0]
            d_cost1 = cs.trade_cost(mkt, cs_params, n_d, d1[0], d1[5], d1[6], d1[4])
        from lib import deribit_marks as dm
        d_fee = n_d * 2 * float(dm.option_fee(d0[5], d0[0] / 2)) if cs_params.fee else 0.0
        if hold != "expiry" and cs_params.fee:
            d_fee += n_d * 2 * float(dm.option_fee(d1[5], d1[0] / 2))
        k_gross = side_k * (k_mid1 - k_mid0) + h_k
        d_gross = side_d * (d_val1 - d0[0]) + h_d
        k_cost = abs(k_ex0 - k_mid0) + abs(k_ex1 - k_mid1) + k_fee0 + k_fee1 + c_k
        d_cost = d_cost0 + d_cost1 + c_d
        both_cost = k_cost + d_cost - c_k - c_d + c_both
        rows.append(dict(event_ticker=r.event_ticker, ts=t, kind=r.kind, hours_left=r.tau_years * 8760,
                         kalshi_expiry=r.expiry,
                         hold=str(hold), signal=s, side=("short Kalshi" if side_k < 0 else "long Kalshi"), K=K,
                         deribit_expiry=E, n_deribit=n_d, vega_k=vK, kalshi_iv=r.atm_iv, k_value=k_mid0,
                         d_value=d0[0], k_gross=k_gross, d_gross=d_gross, k_cost=k_cost, d_cost=d_cost,
                         k_spread=abs(k_ex0 - k_mid0) + abs(k_ex1 - k_mid1), k_fee=k_fee0 + k_fee1,
                         k_perp=c_k, d_perp=c_d, d_fee=d_fee, d_spread=d_cost0 + d_cost1 - d_fee,
                         gross=k_gross + d_gross, costs=both_cost, net=k_gross + d_gross - both_cost,
                         strip_strikes=len(strip)))
    out = pd.DataFrame(rows)
    if len(out):
        out["day"] = out.ts.dt.normalize()
        for leg, g, c in (("both", "gross", "costs"), ("k", "k_gross", "k_cost"), ("d", "d_gross", "d_cost")):
            out[f"{leg}_gross_vol"] = out[g] / out.vega_k
            out[f"{leg}_net_vol"] = (out[g] - out[c]) / out.vega_k
    return out


def trade_summary(trades: pd.DataFrame, by, cols=("both_net_vol",), n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Per group: trades, days, and for each column its mean with a 95% interval resampling whole days (trades on
    the same day share the market's moves, so they are not independent)."""
    by = [by] if isinstance(by, str) else list(by)
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in trades.groupby(by, observed=True):
        per = g.groupby("day")[list(cols)].agg(["sum", "count"])
        idx = rng.integers(0, len(per), size=(n_boot, len(per)))
        row = {**dict(zip(by, key if isinstance(key, tuple) else (key,))), "trades": len(g), "days": len(per)}
        for c in cols:
            s, n = per[(c, "sum")].to_numpy(), per[(c, "count")].to_numpy()
            boot = s[idx].sum(1) / n[idx].sum(1)
            row.update({c: g[c].mean(), f"{c}_lo": np.quantile(boot, 0.025), f"{c}_hi": np.quantile(boot, 0.975)})
        rows.append(row)
    return pd.DataFrame(rows)
