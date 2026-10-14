"""Deribit calendar straddle: long the ~7-day ATM straddle, short the ~1-day one, delta-hedged (08_calendar_straddle.py).

Rules (the base case; every one is a parameter of `Params`):

- Each day at `entry_hour` UTC, buy one ATM straddle (1 call + 1 put, 1 BTC each) on the listed expiry closest to
  `long_days` out, and sell `ratio` ATM straddles on the first expiry at least `min_short_hours` away (the next
  08:00 UTC daily). ATM = the listed strike nearest the expiry's forward. With `ratio = "vega"` the short size makes
  the two vegas equal at entry (about sqrt(7) ~ 2.6 short per long).
- The short leg is bought back `close_short_hours` before its expiry (0 = held to expiry and settled on Deribit's
  delivery price). The long leg is sold `hold_long_hours` after entry (24 = rolled daily). New legs open every day,
  so with longer holds several books overlap.
- The whole book is delta-hedged with BTC-PERPETUAL every `hedge_minutes`, using Black-Scholes deltas at Deribit's
  fitted mark vols. Hedge P&L uses perpetual prices; the hedge pays or receives hourly funding.
- Costs: each option trade pays the half-spread (`spread_mult` x the effective half-spread measured from trades vs
  marks, by time to expiry and distance from the money) plus Deribit's taker fee, 0.03% of the index capped at 12.5%
  of the option price. Perpetual trades pay `perp_fee` of notional. Settlement at expiry is free.

Options are valued every hour from Deribit's fitted mark smile for their expiry (`deribit_marks.smile_grid`, trades
within 30 minutes either side); when an expiry has no fit that hour, its last fit (at most `max_stale_hours` old) is
used with the current index and time left. The index between trades is the last Deribit index printed on a trade.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from lib import deribit_marks as dm

HOUR = pd.Timedelta(hours=1)
HS_TAU_EDGES = [0, 0.25, 1, 2, 4, 8, 16]        # days to expiry, for the half-spread table
HS_SD_EDGES = [0, 0.5, 1, 2, 4, np.inf]          # |sd| from the forward


@dataclass(frozen=True)
class Params:
    entry_hour: int = 8               # UTC hour at which both legs open each day
    long_days: float = 7.0            # target tenor of the long leg ...
    long_range: tuple = (4.0, 11.0)   # ... chosen among listed expiries this many days out
    min_short_hours: float = 12.0     # short leg: first expiry at least this far away
    ratio: object = "vega"            # short straddles per long: "vega" (vega-neutral), "none" (long only) or a number
    long_size: float = 1.0            # long straddles (0 = short leg only)
    close_short_hours: float = 4.0    # buy back the short this long before its expiry (0 = hold to expiry)
    hold_long_hours: float = 24.0     # sell the long this long after entry
    hedge_minutes: int = 60           # delta-hedge interval (0 = never)
    spread_mult: float = 1.0          # x the measured effective half-spread (0 = trade at Deribit's mark)
    fee: bool = True                  # Deribit option taker fees
    perp_fee: float = dm.PERP_TAKER_FEE
    max_stale_hours: float = 3.0
    start: str = "2026-01-01"
    end: str = "2026-10-01"

    def but(self, **kw):
        return replace(self, **kw)


def half_spread_table(d: pd.DataFrame) -> pd.DataFrame:
    """Median effective half-spread in USD and as a share of the mark, by days to expiry and |sd| bucket."""
    e = dm.effective_half_spread(d)
    sd = np.abs(e.k) / (e.iv_d * np.sqrt(e.tau_years))
    e = e.assign(tb=pd.cut(e.tau_years * 365, HS_TAU_EDGES), sb=pd.cut(sd, HS_SD_EDGES))
    t = e.groupby(["tb", "sb"], observed=True).agg(hs_usd=("hs_usd", "median"), hs_frac=("hs_frac", "median"),
                                                     hs_vol=("hs_vol", "median"), n=("hs_usd", "size"))
    return t


def lookup_half_spread(table: pd.DataFrame, tau_days, sd) -> np.ndarray:
    """Half-spread in USD for options at `tau_days` and `sd` (vectorised); floors at zero."""
    tb = pd.cut(np.atleast_1d(tau_days), HS_TAU_EDGES)
    sb = pd.cut(np.abs(np.atleast_1d(sd)), HS_SD_EDGES)
    idx = pd.MultiIndex.from_arrays([tb, sb])
    v = table.hs_usd.reindex(idx).to_numpy(dtype=float)
    return np.clip(np.nan_to_num(v, nan=np.nanmedian(table.hs_usd)), 0, None)


class Market:
    """Hourly Deribit state: fitted smiles per expiry, index, perpetual, funding, listed strikes, delivery prices."""

    def __init__(self, trades: pd.DataFrame, perp: pd.DataFrame, funding: pd.DataFrame, delivery: pd.Series,
                 grid: pd.DataFrame):
        self.by_expiry = {e: g.set_index("t").sort_index() for e, g in grid.groupby("expiry")}
        self.expiries_at = grid.groupby("t").expiry.apply(lambda s: np.sort(s.unique()))
        idx = trades.set_index("ts").index_price.resample("1min", label="right", closed="right").last()
        self.index = idx.ffill()
        p = perp.set_index("ts").close
        self.perp = p.reindex(self.index.index.union(p.index)).ffill()
        f = funding.set_index("ts").interest_1h
        self.funding = f
        self.strikes = trades.groupby("expiry").strike.apply(lambda s: np.sort(s.unique()))
        self.delivery = delivery
        self.hs = half_spread_table(trades)
        self._fit_cache = {}

    def fit(self, expiry, t, max_stale: float):
        """Latest smile fit for `expiry` at or before `t`, at most `max_stale` hours old (None otherwise)."""
        key = (expiry, t, max_stale)
        if key in self._fit_cache:
            return self._fit_cache[key]
        out = None
        g = self.by_expiry.get(expiry)
        if g is not None:
            j = g.index.searchsorted(t, side="right") - 1
            if j >= 0 and (t - g.index[j]) <= pd.Timedelta(hours=max_stale):
                out = g.iloc[j]
        self._fit_cache[key] = out
        return out

    def index_at(self, t):
        return float(self.index.asof(t))

    def perp_at(self, t):
        return float(self.perp.asof(t))

    def settle(self, expiry):
        return float(self.delivery.get(expiry.date(), np.nan))


@dataclass
class Leg:
    expiry: pd.Timestamp
    strike: float
    qty: float                # straddles, + long / - short (1 straddle = 1 call + 1 put, 1 BTC each)
    exit_time: pd.Timestamp   # when it is closed (== expiry: settled)
    role: str
    value: float = np.nan     # last USD value per straddle
    hedge: float = 0.0        # perpetual position hedging this leg (BTC)


def straddle(mkt: Market, leg_expiry, K, t, max_stale):
    """(value, delta, vega per vol point, vol, sd) of one straddle at time t; None when no fit is available."""
    row = mkt.fit(leg_expiry, t, max_stale)
    if row is None:
        return None
    S = mkt.index_at(t)
    tau = (leg_expiry - t).total_seconds() / dm.YEAR_SECONDS
    p, dlt, sig, vg = dm.value(row, np.array([K, K]), np.array([True, False]), S, tau)
    F = S * (1 + row["basis"])
    sd = np.log(K / F) / (sig[0] * np.sqrt(tau))
    return float(p.sum()), float(dlt.sum()), float(vg.sum()), float(sig[0]), float(sd), S, tau


def trade_cost(mkt: Market, p: Params, qty, value_one, S, tau, sd):
    """USD cost of trading |qty| straddles: half-spread on both options plus taker fees (per option, approx. half the
    straddle's value each for the 12.5% cap)."""
    hs = p.spread_mult * 2 * lookup_half_spread(mkt.hs, tau * 365, sd)[0]
    fee = 2 * dm.option_fee(S, value_one / 2) if p.fee else 0.0
    return abs(qty) * (hs + fee)


def run(mkt: Market, p: Params) -> dict:
    """Simulate the strategy hour by hour. Returns `daily` (P&L per UTC day by component), `trades` (one row per leg
    opened, with its entry and exit details) and `hourly` (book value and hedge)."""
    hours = pd.date_range(pd.Timestamp(p.start, tz="UTC"), pd.Timestamp(p.end, tz="UTC"), freq="1h")
    hedge_step = pd.Timedelta(minutes=p.hedge_minutes) if p.hedge_minutes else None
    legs: list[Leg] = []
    rows, trades = [], []
    hedge_pos, last_perp = 0.0, None
    for t in hours:
        pnl = dict(t=t, options=0.0, hedge=0.0, funding=0.0, opt_cost=0.0, perp_cost=0.0, n_legs=len(legs))
        perp_now = mkt.perp_at(t)
        # 1. hedge P&L and funding over the hour just ended, at sub-hour rebalancing if asked
        if last_perp is not None and hedge_step is not None and hedge_step < HOUR and legs:
            hedge_pos, h_pnl, h_cost = _intrahour_hedge(mkt, p, legs, t - HOUR, t, hedge_pos)
            pnl["hedge"] += h_pnl
            pnl["perp_cost"] += h_cost
        elif last_perp is not None:
            pnl["hedge"] += hedge_pos * (perp_now - last_perp)
        if last_perp is not None:
            pnl["funding"] -= hedge_pos * mkt.index_at(t) * float(mkt.funding.get(t, 0.0))
        # 2. revalue and close legs whose exit time has come
        still = []
        for leg in legs:
            if t >= leg.exit_time:
                if leg.exit_time >= leg.expiry:   # settled on the delivery price
                    st = mkt.settle(leg.expiry)
                    v = abs(st - leg.strike) if np.isfinite(st) else leg.value
                    cost = 0.0
                    sd = np.nan
                else:
                    s = straddle(mkt, leg.expiry, leg.strike, t, p.max_stale_hours)
                    v = s[0] if s else leg.value
                    cost = trade_cost(mkt, p, leg.qty, v, s[5], s[6], s[4]) if s else np.nan
                    sd = s[4] if s else np.nan
                pnl["options"] += leg.qty * (v - leg.value)
                pnl["opt_cost"] += np.nan_to_num(cost)
                trades.append(dict(role=leg.role, expiry=leg.expiry, strike=leg.strike, qty=leg.qty,
                                   event="exit", t=t, value=v, cost=cost, sd=sd))
            else:
                s = straddle(mkt, leg.expiry, leg.strike, t, p.max_stale_hours)
                if s is not None:
                    pnl["options"] += leg.qty * (s[0] - leg.value)
                    leg.value = s[0]
                still.append(leg)
        legs = still
        # 3. open the day's legs
        if t.hour == p.entry_hour:
            new = _open_legs(mkt, p, t)
            for leg, cost, info in new:
                legs.append(leg)
                pnl["opt_cost"] += cost
                trades.append(dict(role=leg.role, expiry=leg.expiry, strike=leg.strike, qty=leg.qty, event="entry",
                                   t=t, value=leg.value, cost=cost, **info))
        # 4. rebalance the hedge on the hour
        on_cycle = hedge_step is not None and (hedge_step <= HOUR or ((t.hour - p.entry_hour) * 60) % p.hedge_minutes == 0)
        if on_cycle and legs:
            target = -sum(leg.qty * d for leg in legs if (d := _leg_delta(mkt, p, leg, t)) is not None)
            pnl["perp_cost"] += p.perp_fee * abs(target - hedge_pos) * perp_now
            hedge_pos = target
        elif not legs and hedge_pos != 0:
            pnl["perp_cost"] += p.perp_fee * abs(hedge_pos) * perp_now
            hedge_pos = 0.0
        last_perp = perp_now
        pnl["hedge_pos"] = hedge_pos
        rows.append(pnl)
    hourly = pd.DataFrame(rows).set_index("t")
    hourly["pnl"] = hourly.options + hourly.hedge + hourly.funding - hourly.opt_cost - hourly.perp_cost
    daily = hourly.drop(columns=["n_legs", "hedge_pos"]).resample("1D").sum()
    return dict(daily=daily, hourly=hourly, trades=pd.DataFrame(trades))


def _leg_delta(mkt, p, leg, t):
    s = straddle(mkt, leg.expiry, leg.strike, t, p.max_stale_hours)
    return s[1] if s else None


def _intrahour_hedge(mkt, p, legs, t0, t1, pos):
    """Rebalance every `hedge_minutes` inside (t0, t1], deltas at the latest fit and the minute's index."""
    step = pd.Timedelta(minutes=p.hedge_minutes)
    pnl = cost = 0.0
    prev = mkt.perp_at(t0)
    t = t0 + step
    while t <= t1:
        px = mkt.perp_at(t)
        pnl += pos * (px - prev)
        if t < t1:
            target = -sum(leg.qty * d for leg in legs if (d := _leg_delta(mkt, p, leg, t)) is not None)
            cost += p.perp_fee * abs(target - pos) * px
            pos = target
        prev = px
        t += step
    return pos, pnl, cost


def _atm_strike(mkt, expiry, F):
    ks = mkt.strikes.get(expiry)
    if ks is None or len(ks) == 0:
        return None
    return float(ks[np.argmin(np.abs(ks - F))])


def _open_legs(mkt: Market, p: Params, t):
    """The day's new legs: long straddle on the expiry nearest `long_days`, short `ratio` on the next daily."""
    exps = [e for e in mkt.expiries_at.get(t, []) if mkt.fit(e, t, 0.0) is not None]
    exps = [pd.Timestamp(e) for e in exps]
    days = np.array([(e - t).total_seconds() / 86_400 for e in exps])
    out = []
    short_cands = [e for e, dd in zip(exps, days) if dd * 24 >= p.min_short_hours]
    long_cands = [(abs(dd - p.long_days), e) for e, dd in zip(exps, days) if p.long_range[0] <= dd <= p.long_range[1]]
    if not short_cands or not long_cands:
        return out
    e1, e2 = min(short_cands), min(long_cands)[1]
    S = mkt.index_at(t)
    legs_spec = []
    s2 = None
    if p.long_size:
        k2 = _atm_strike(mkt, e2, S * (1 + mkt.fit(e2, t, 0.0)["basis"]))
        s2 = straddle(mkt, e2, k2, t, 0.0) if k2 else None
        if s2 is None:
            return out
        legs_spec.append(("long", e2, k2, p.long_size, s2, t + pd.Timedelta(hours=p.hold_long_hours)))
    if p.ratio != "none":
        k1 = _atm_strike(mkt, e1, S * (1 + mkt.fit(e1, t, 0.0)["basis"]))
        s1 = straddle(mkt, e1, k1, t, 0.0) if k1 else None
        if s1 is None:
            return []
        if p.ratio == "vega":
            n = (p.long_size or 1.0) * (s2[2] if s2 else straddle_vega_7d(s1)) / s1[2]
        else:
            n = float(p.ratio) * (p.long_size or 1.0)
        exit1 = e1 - pd.Timedelta(hours=p.close_short_hours)
        legs_spec.append(("short", e1, k1, -n, s1, exit1))
    for role, e, k, q, s, ex in legs_spec:
        leg = Leg(expiry=e, strike=k, qty=q, exit_time=min(ex, e), role=role, value=s[0])
        cost = trade_cost(mkt, p, q, s[0], s[5], s[6], s[4])
        info = dict(sd=s[4], vol=s[3], vega=s[2], delta=s[1], days=(e - t).total_seconds() / 86_400, index=s[5])
        out.append((leg, cost, info))
    return out


def straddle_vega_7d(s1):
    """Vega of a 7-day straddle with the short leg's vol, for sizing a short-only book like the base case."""
    value, delta, vega, vol, sd, S, tau = s1
    return float(2 * dm.bs_vega(S, S, vol, 7 / 365) / 100)


def summary(daily: pd.Series, jump_days=()) -> dict:
    """Mean and standard deviation per day, annualised Sharpe (sqrt 365), worst day, max drawdown, hit rate."""
    x = daily.dropna()
    cum = x.cumsum()
    dd = (cum - cum.cummax()).min()
    out = dict(days=len(x), total=x.sum(), mean=x.mean(), sd=x.std(), sharpe=np.sqrt(365) * x.mean() / x.std(),
               worst=x.min(), worst_day=x.idxmin().date() if len(x) else None, max_dd=dd, hit=(x > 0).mean())
    for name, days in dict(jump_days).items():
        out[f"jump_{name}"] = x[x.index.normalize().isin(pd.DatetimeIndex(days))].sum()
    return out
