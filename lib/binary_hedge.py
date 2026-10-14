"""Hedging a long Kalshi binary against a short call spread, walking toward expiry (13_binary_hedging.py).

Position: buy $1,000,000 of Kalshi "BTC above K at 5pm ET" (N = 1,000,000 / price contracts, each paying $1) and,
optionally, sell N / (2h) call spreads struck at K - h and K + h, which pay like the binary except within h of K.
Frictionless: Kalshi at mid, call spreads and BTC at model value / spot, no fees, any size.

Values and Greeks come from one option model: Black-Scholes with Deribit's own smile at Kalshi's expiry (Deribit's
ATM vol interpolated to 5pm ET as in 09, the smile's slope and curvature from Deribit's nearest expiry). Greeks are
taken by finite differences holding the strike vols fixed (sticky strike). Kalshi's leg is marked at Kalshi's
traded mid; at expiry everything settles on Kalshi's settlement value.

Hedging schemes (`simulate`):
- delta-hedge the net position with BTC every `step` minutes (`step=None`: never hedge, Greeks still recorded
  every 30 minutes);
- `delta_mode="smooth"`: hedge with the delta of a call spread whose half-width never falls below
  `smooth_sd` x the expected move to expiry (a wide, tame stand-in for the binary near expiry);
- `stop_minutes`: stop rebalancing that long before expiry (keep the last hedge);
- `band`: rebalance only when the hedge is off by more than `band` (as a share of the binary's notional, N);
- `cap`: never hold more BTC than `cap` x N dollars' worth (the binary can never pay more than N);
- `exit_minutes`: unwind everything that long before expiry (Kalshi at its mid, the spread at model value);
- `offset`: centre the call spread at K + offset instead of K. With offset = +h the spread runs from K to K + 2h
  and never pays more than the binary, so long binary - short spread can only gain at expiry ("one-sided").
"""

import numpy as np
import pandas as pd
from scipy.special import ndtr

from lib import iv

YEAR_MIN = 365 * 24 * 60


def vol_at(K, F, tau, atm, b, c, m_lo, m_hi):
    """Smile vol at strike(s) K: atm + b m + c m^2, m = ln(K/F)/sqrt(tau) clipped to the fitted range."""
    m = np.clip(np.log(np.asarray(K, dtype=float) / F) / np.sqrt(tau), m_lo, m_hi)
    return atm + b * m + c * m**2


def digital(S, K, tau, vols):
    """Black-Scholes digital "$1 if S_T > K" at one vol (forward = S, no rates)."""
    sd = vols * np.sqrt(tau)
    return ndtr((np.log(S / K) - 0.5 * sd**2) / sd)


def call(S, K, tau, vols):
    return iv.bs_call(S, K, vols, tau)


class Smile:
    """Deribit-implied vols at Kalshi's expiry for a fixed set of strikes, frozen at a time (sticky strike)."""

    def __init__(self, S, tau, atm, b, c, m_lo, m_hi):
        self.S, self.tau, self.p = S, tau, (atm, b, c, m_lo, m_hi)

    def vol(self, K):
        return vol_at(K, self.S, max(self.tau, 1e-9), *self.p)


def binary_value(S, K, tau, sm: Smile):
    """Digital price with the smile (skew term included via a tight call spread on the smile)."""
    if tau <= 0:
        return float(S > K)
    e = 1.0
    return float((call(S, K - e, tau, sm.vol(K - e)) - call(S, K + e, tau, sm.vol(K + e))) / (2 * e))


def spread_value(S, K, h, tau, sm: Smile):
    """One call spread K - h / K + h, divided by 2h so that it pays like a binary far from K (USD per unit)."""
    if tau <= 0:
        return float(np.clip(S - (K - h), 0, 2 * h) / (2 * h))
    return float((call(S, K - h, tau, sm.vol(K - h)) - call(S, K + h, tau, sm.vol(K + h))) / (2 * h))


def greeks(f, S, tau, sm: Smile, dS_frac=0.001, dv=0.01):
    """Delta (per $1 of BTC), gamma (per $1^2), vega (per vol point) of value function f(S, tau, smile)."""
    dS = S * dS_frac
    up, mid, dn = f(S + dS, tau, sm), f(S, tau, sm), f(S - dS, tau, sm)
    sm_up = Smile(sm.S, sm.tau, sm.p[0] + dv, *sm.p[1:])
    return dict(value=mid, delta=(up - dn) / (2 * dS), gamma=(up - 2 * mid + dn) / dS**2,
                vega=(f(S, tau, sm_up) - mid) / (dv * 100))


def simulate(path: pd.DataFrame, K: float, settle: float, notional: float = 1e6, h: float | None = 250.0,
             step: int = 30, delta_mode: str = "model", smooth_sd: float = 0.5, stop_minutes: int = 0,
             band: float = 0.0, cap: float | None = None, exit_minutes: int = 0, offset: float = 0.0) -> dict:
    """Hedge a long Kalshi binary (+ short call spreads if `h`) along a minute path to expiry.

    `path`: one row per minute up to expiry with `spot`, `tau` (years), `kalshi_mid` (for strike K), and the smile
    columns `atm, b, c, m_lo, m_hi`. The first row is the trade. Returns `steps` (one row per rebalance with the
    position's Greeks and cumulative P&L by leg) and `summary`.
    """
    p0 = path.iloc[0]
    N = notional / p0.kalshi_mid
    n_sp = N if h else 0.0
    sm0 = Smile(p0.spot, p0.tau, p0.atm, p0.b, p0.c, p0.m_lo, p0.m_hi)
    Ks = K + offset
    sp0 = spread_value(p0.spot, Ks, h, p0.tau, sm0) if h else 0.0
    model_bin0 = binary_value(p0.spot, K, p0.tau, sm0)
    hedge, cash_hedge, rows = 0.0, 0.0, []
    last_S = p0.spot
    T_exp = path.index[-1]
    if exit_minutes:
        path = path.loc[:T_exp - pd.Timedelta(minutes=exit_minutes)]
    times = path.index
    T = times[-1]
    for i, t in enumerate(times):
        r = path.loc[t]
        cash_hedge += hedge * (r.spot - last_S)
        last_S = r.spot
        minutes_left = (T_exp - t).total_seconds() / 60
        on_step = (i % (step or 30) == 0) and minutes_left > stop_minutes and t < T
        sm = Smile(r.spot, r.tau, r.atm, r.b, r.c, r.m_lo, r.m_hi)
        if on_step:
            gb = greeks(lambda S, tau, s: binary_value(S, K, tau, s), r.spot, r.tau, sm)
            gs = greeks(lambda S, tau, s: spread_value(S, Ks, h, tau, s), r.spot, r.tau, sm) if h else \
                dict(value=0.0, delta=0.0, gamma=0.0, vega=0.0)
            if delta_mode == "smooth":
                w = max(h or 0.0, smooth_sd * r.spot * r.atm * np.sqrt(max(r.tau, 1e-9)))
                d_bin = greeks(lambda S, tau, s: spread_value(S, K, w, tau, s), r.spot, r.tau, sm)["delta"]
            else:
                d_bin = gb["delta"]
            net_delta = N * d_bin - n_sp * gs["delta"]
            target = -net_delta
            if cap is not None:
                target = float(np.clip(target, -cap * N / r.spot, cap * N / r.spot))
            if step is not None and (abs(target - hedge) * r.spot > band * N or band == 0):
                hedge = target
            kal_val = r.kalshi_mid if np.isfinite(r.kalshi_mid) else gb["value"]
            rows.append(dict(t=t, spot=r.spot, minutes_left=minutes_left, kalshi_mid=r.kalshi_mid,
                             model_binary=gb["value"], hedge_btc=hedge,
                             bin_delta=N * gb["delta"], sp_delta=-n_sp * gs["delta"],
                             bin_gamma_1pct=0.5 * N * gb["gamma"] * (0.01 * r.spot) ** 2,
                             sp_gamma_1pct=-0.5 * n_sp * gs["gamma"] * (0.01 * r.spot) ** 2,
                             bin_vega=N * gb["vega"], sp_vega=-n_sp * gs["vega"],
                             pnl_kalshi=N * (kal_val - p0.kalshi_mid),
                             pnl_spread=-n_sp * ((gs["value"] if h else 0.0) - sp0),
                             pnl_hedge=cash_hedge))
    if exit_minutes:   # unwind at Kalshi's mid (model value if missing) and the spread's model value
        rl = path.iloc[-1]
        sml = Smile(rl.spot, rl.tau, rl.atm, rl.b, rl.c, rl.m_lo, rl.m_hi)
        pay_bin = float(rl.kalshi_mid) if np.isfinite(rl.kalshi_mid) else binary_value(rl.spot, K, rl.tau, sml)
        pay_sp = spread_value(rl.spot, Ks, h, rl.tau, sml) if h else 0.0
    else:
        pay_bin = float(settle > K)
        pay_sp = float(np.clip(settle - (Ks - h), 0, 2 * h) / (2 * h)) if h else 0.0
    out = dict(pnl_kalshi=N * (pay_bin - p0.kalshi_mid), pnl_spread=-n_sp * (pay_sp - sp0), pnl_hedge=cash_hedge)
    out["total"] = out["pnl_kalshi"] + out["pnl_spread"] + out["pnl_hedge"]
    out["edge_at_entry"] = N * ((sp0 if h else model_bin0) - p0.kalshi_mid)
    out["contracts"] = N
    rows.append(dict(t=T, spot=path.iloc[-1].spot, minutes_left=0.0, kalshi_mid=pay_bin, model_binary=pay_bin,
                     hedge_btc=0.0, pnl_kalshi=out["pnl_kalshi"], pnl_spread=out["pnl_spread"],
                     pnl_hedge=out["pnl_hedge"]))
    steps = pd.DataFrame(rows).set_index("t")
    steps["pnl_total"] = steps.pnl_kalshi + steps.pnl_spread + steps.pnl_hedge
    return dict(steps=steps, summary=out)


def build_day(root, event: str, start, profile: pd.Series) -> tuple[pd.DataFrame, dict, pd.Series]:
    """Minute inputs for one Kalshi event from `start` to its expiry: spot, tau, Deribit smile at Kalshi's expiry
    (30-minute fits, carried forward), and every strike's Kalshi mid (02's rules: missing sides, 20c spread cap).

    Returns (frame with spot/tau/smile columns, {strike: kalshi mid Series}, markets with results).
    """
    from pathlib import Path
    from lib import cleaning, deribit_marks as dm, vol_spread as vs
    root = Path(root)
    mk = pd.read_parquet(root / "data" / "raw" / "kalshi_events" / "markets" / f"{event}.parquet")
    cd = pd.read_parquet(root / "data" / "raw" / "kalshi_events" / "candles" / f"{event}.parquet")
    T = mk.close_time.max()
    start = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
    idx = pd.date_range(start, T, freq="1min")
    spot = cleaning.clean_spot(pd.read_parquet(root / "data" / "raw" / "btc_spot" / "btcusd_1m.parquet")).spot
    fr = pd.DataFrame({"spot": spot.reindex(idx).ffill()}, index=idx)
    fr["tau"] = (T - fr.index).total_seconds() / (365 * 86400)
    d = dm.load_options(root, (start - pd.Timedelta(hours=2)).tz_localize(None), (T + pd.Timedelta(hours=1)).tz_localize(None))
    g = dm.smile_grid(d, freq="30min", half_window="30min")
    times = pd.date_range(start.floor("30min"), T, freq="30min")
    snaps = pd.DataFrame({"ts": times, "expiry": T, "tau_years": (T - times).total_seconds() / (365 * 86400)})
    snaps = snaps[snaps.tau_years > 0]
    x = vs.deribit_at_kalshi(snaps, g, profile).set_index("ts")
    by_t = {t: q for t, q in g.groupby("t")}
    shape = []
    for t, r in x.iterrows():
        q = by_t[t].iloc[np.argmin(np.abs((by_t[t].expiry - T).dt.total_seconds().to_numpy()))]
        shape.append((r.deribit_iv, q.b, q.c, q.m_lo, q.m_hi))
    sm = pd.DataFrame(shape, index=x.index, columns=["atm", "b", "c", "m_lo", "m_hi"])
    fr = fr.join(sm.reindex(idx, method="ffill"))
    mids = {}
    for k, gk in cd.groupby("strike"):
        q = gk.set_index("ts")[["bid_close", "ask_close"]].reindex(idx).ffill()
        bid, ask = cleaning.quote_sides(q.bid_close.to_numpy(), q.ask_close.to_numpy())
        _, mid = cleaning.mid_and_spread(bid, ask)
        mids[float(k)] = pd.Series(mid, index=idx)
    return fr, mids, mk


SCHEMES = {
    "no hedge": dict(h=None, step=None),
    "delta, 30 min": dict(h=None, step=30),
    "delta, 5 min": dict(h=None, step=5),
    "delta, 30 min, stop last hour": dict(h=None, step=30, stop_minutes=60),
    "smoothed delta, 30 min": dict(h=None, step=30, delta_mode="smooth"),
    "call spread $500 wide, static": dict(h=250, step=None),
    "call spread $500 wide + delta, 30 min": dict(h=250, step=30),
    "call spread $1,000 wide + delta, 30 min": dict(h=500, step=30),
    "call spread $2,000 wide + delta, 30 min": dict(h=1000, step=30),
    "call spread $500 wide + delta, 30 min, capped": dict(h=250, step=30, cap=1.0),
    "call spread $500 wide + delta, 30 min, stop last 3h": dict(h=250, step=30, stop_minutes=180),
    "delta, 30 min, capped": dict(h=None, step=30, cap=1.0),
    "call spread $500 wide, unwind 1h before": dict(h=250, step=None, exit_minutes=60),
    "delta, 30 min, unwind 1h before": dict(h=None, step=30, exit_minutes=60),
    "one-sided spread K to K+$500, static": dict(h=250, step=None, offset=250),
    "one-sided spread K-$500 to K, static": dict(h=250, step=None, offset=-250),
}


def cross_day(panel: pd.DataFrame, strikes: pd.DataFrame, spot: pd.Series, settle: pd.Series,
              schemes: dict = SCHEMES, notional: float = 1e6) -> pd.DataFrame:
    """Every Kalshi daily: buy $notional of the strike nearest spot at its first usable hour (Kalshi's mid), hedge
    by each scheme to expiry along the minute BTC path. Smile inputs hourly from `panel` (Deribit at Kalshi's expiry,
    `deribit_sig`, and the nearest expiry's shape `d_b, d_c, d_m_lo, d_m_hi`), carried forward minute by minute;
    Kalshi's leg is marked with the model between hours and settles on the contract's result.

    Returns one row per day and scheme: total P&L, the edge at entry (model value - Kalshi price, times contracts),
    and the hedging error (total - edge), which is what the hedge is responsible for.
    """
    rows = []
    for ev, g in panel[panel.kind == "daily"].sort_values("ts").groupby("event_ticker"):
        g = g.dropna(subset=["deribit_sig", "d_b"])
        if g.empty or ev not in settle.index:
            continue
        r0 = g.iloc[0]
        lad = strikes[(strikes.event_ticker == ev) & (strikes.ts == r0.ts) & strikes.mid.between(0.2, 0.8)]
        if lad.empty:
            continue
        K = float(lad.strike.iloc[np.argmin(np.abs(lad.strike - r0.spot))])
        mid0 = float(lad.loc[lad.strike == K, "mid"].iloc[0])
        idx = pd.date_range(r0.ts, r0.expiry, freq="1min")
        fr = pd.DataFrame({"spot": spot.reindex(idx).ffill()}, index=idx)
        if fr.spot.isna().any():
            continue
        fr["tau"] = (r0.expiry - fr.index).total_seconds() / (365 * 86400)
        sm = g.set_index("ts")[["deribit_sig", "d_b", "d_c", "d_m_lo", "d_m_hi"]]
        sm.columns = ["atm", "b", "c", "m_lo", "m_hi"]
        fr = fr.join(sm.reindex(idx, method="ffill"))
        km = strikes[(strikes.event_ticker == ev) & (strikes.strike == K)].set_index("ts").mid
        fr["kalshi_mid"] = km.reindex(idx)
        fr.iloc[0, fr.columns.get_loc("kalshi_mid")] = mid0
        for name, kw in schemes.items():
            s = simulate(fr, K, float(settle[ev]), notional, **kw)["summary"]
            rows.append(dict(event_ticker=ev, day=r0.expiry.normalize(), scheme=name, K=K, entry=mid0,
                             spot0=r0.spot, settle=float(settle[ev]), hours=(r0.expiry - r0.ts).total_seconds() / 3600,
                             total=s["total"], edge=s["edge_at_entry"], hedge_error=s["total"] - s["edge_at_entry"],
                             dist_settle_sd=np.log(float(settle[ev]) / K) / (r0.deribit_sig * np.sqrt(r0.tau_years))))
    return pd.DataFrame(rows)
