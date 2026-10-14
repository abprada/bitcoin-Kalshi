"""Build the inputs and run the backtests shown in 09_vol_spread.py and 10_calendar_straddle.py.

Only what the project did not have yet is computed here; everything else is reused (03's ladder method via
lib/iv.py, 05's hour-of-week profile via lib/deribit.py, 02's cleaning rules via lib/cleaning.py):

1. Kalshi smiles from the collaborator's hourly snapshots (dailies and weeklies, Jan-Sep 2026), as in 03.
2. Deribit's hourly mark smile per expiry from full-day trades (00_fetch_deribit.py): centred windows for valuing
   positions, backward-looking windows for trading signals.
3. The hourly Kalshi vs Deribit panel: Deribit's ATM vol interpolated to Kalshi's expiry, the gap, realized vol.
4. Strategy 1 (Kalshi vs Deribit vol spread) trades for several holding periods and two signals.
5. Strategy 2 (Deribit calendar straddle) daily P&L for the base case and its variations.

Needs data/raw/deribit (00_fetch_deribit.py) and data/raw/btc_spot (00_fetch_spot.py).
Writes data/derived/strategies/ (not in git). About 25 minutes.
  python 08_strategy_data.py                  # everything
  python 08_strategy_data.py --only calendar  # one step: inputs, volspread, calendar
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lib import calendar_straddle as cs
from lib import cleaning, deribit, kalshi_hourly
from lib import binary_hedge as bh
from lib import deribit_marks as dm
from lib import dislocation as dl
from lib import vol_spread as vs

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "derived" / "strategies"
H_EDGES = [0, 12, 24, 48, 96, 170]     # Kalshi hours left, for the structural gap

CALENDAR_VARIANTS = {
    # name: (description, Params changes)
    "base": ("long 7d / short 1d vega-neutral, hourly hedge, short closed 4h before expiry, taker costs", {}),
    "gross": ("base, no costs at all", dict(spread_mult=0.0, fee=False, perp_fee=0.0)),
    "maker_hedge": ("base, perpetual hedge at maker fee (0%)", dict(perp_fee=0.0)),
    "hold_short_expiry": ("base, short held to expiry (no buy-back cost)", dict(close_short_hours=0.0)),
    "close_short_1h": ("base, short closed 1h before expiry", dict(close_short_hours=1.0)),
    "close_short_8h": ("base, short closed 8h before expiry", dict(close_short_hours=8.0)),
    "hedge_4h": ("base, hedge every 4h", dict(hedge_minutes=240)),
    "hedge_daily": ("base, hedge once a day at entry", dict(hedge_minutes=1440)),
    "hedge_15m": ("base, hedge every 15 minutes", dict(hedge_minutes=15)),
    "entry_00": ("base, entry 00:00 UTC (the 08:00 daily 8h away is too close: short leg 32h)", dict(entry_hour=0)),
    "entry_12": ("base, entry 12:00 UTC (short leg 20h)", dict(entry_hour=12)),
    "entry_20": ("base, entry 20:00 UTC (short leg 12h, overnight)", dict(entry_hour=20)),
    "ratio_1": ("1 short straddle per long (net long vega)", dict(ratio=1.0)),
    "long_only": ("long 7d straddle only, hedged", dict(ratio="none")),
    "short_only": ("short 1d straddles only (same size as base), hedged", dict(long_size=0.0)),
    "long_3d": ("long leg ~3 days out", dict(long_days=3.0, long_range=(2.0, 4.5))),
    "long_14d": ("long leg ~14 days out", dict(long_days=14.0, long_range=(10.0, 15.0))),
    "hold_long_4d": ("long leg held 4 days (rolled less often)", dict(hold_long_hours=96.0)),
    "spread_2x": ("base, twice the measured option half-spread", dict(spread_mult=2.0)),
    "best_practical": ("short held to expiry, maker hedge every hour", dict(close_short_hours=0.0, perp_fee=0.0)),
    "overnight_expiry": ("entry 20:00 UTC, short (12h, overnight) held to expiry, maker hedge",
                         dict(entry_hour=20, close_short_hours=0.0, perp_fee=0.0)),
    "short_only_expiry": ("short 1d straddles only, held to expiry, maker hedge",
                          dict(long_size=0.0, close_short_hours=0.0, perp_fee=0.0)),
    "short_only_overnight": ("short 12h straddles only (entry 20:00 UTC), held to expiry, maker hedge",
                             dict(long_size=0.0, entry_hour=20, close_short_hours=0.0, perp_fee=0.0)),
}


def step_inputs():
    t0 = time.time()
    spot = cleaning.clean_spot(pd.read_parquet(ROOT / "data" / "raw" / "btc_spot" / "btcusd_1m.parquet")).spot
    profile = deribit.hourly_variance(spot)
    d = dm.load_options(ROOT)
    print(f"Deribit trades: {len(d):,} ({time.time() - t0:.0f}s)")
    grid_c, grid_b = dm.smile_grid(d), dm.smile_grid(d, centred=False)
    grid_c.to_parquet(OUT / "deribit_grid_centred.parquet", index=False)
    grid_b.to_parquet(OUT / "deribit_grid_backward.parquet", index=False)
    cs.half_spread_table(d).reset_index().astype({"tb": str, "sb": str}).to_parquet(OUT / "deribit_half_spreads.parquet")
    dm.delivery_prices(ROOT, d).rename("delivery").rename_axis("date").reset_index().to_parquet(
        OUT / "deribit_delivery.parquet", index=False)
    profile.rename("var_per_hour").rename_axis("hour_of_week").reset_index().to_parquet(OUT / "hour_profile.parquet")

    # Deribit term structure with the realized vol that followed (the PNG's first check)
    g = grid_c.copy()
    g["rv_to_expiry"] = vs.realized_vol(spot, g.t, g.expiry)
    W = lambda x: vs.business_clock(x, profile)
    g["seasonal_var_ratio"] = (W(g.expiry) - W(g.t)) / (profile.mean() * (g.expiry - g.t).dt.total_seconds() / 3600)
    g.to_parquet(OUT / "deribit_term_structure.parquet", index=False)

    # Kalshi hourly smiles (new: the collaborator's hourly file was never turned into vols)
    snap_path = ROOT / "data" / "derived" / "kalshi_hourly_snapshots.parquet"
    if snap_path.exists():
        snaps = pd.read_parquet(snap_path)
    else:
        q = kalshi_hourly.load_quotes(ROOT)
        snaps, strikes = kalshi_hourly.smiles(q[q.ts >= spot.index.min()], spot)
        snaps.to_parquet(snap_path, index=False)
        strikes.to_parquet(ROOT / "data" / "derived" / "kalshi_hourly_strikes.parquet", index=False)
    # every snapshot with an ATM vol: 05's strikes-per-SD rule (`usable`) keeps near expiry only the high-vol
    # moments (with $500 strikes and a few hours left, only vols of 70%+ pass), which fakes a spike in the term
    # structure. It is kept as a precision flag (`fine_grid`) instead of a filter.
    snaps = snaps[snaps.atm_iv.notna() & (snaps.ts >= d.ts.min())].reset_index(drop=True)
    snaps["fine_grid"] = snaps.usable
    snaps["rv_to_expiry"] = vs.realized_vol(spot, snaps.ts, snaps.expiry)

    # the panel: Deribit at Kalshi's expiry, centred (comparison) and backward (signal)
    p = vs.deribit_at_kalshi(snaps, grid_c, profile)
    pb = vs.deribit_at_kalshi(snaps, grid_b, profile)
    p["deribit_sig"], p["method_sig"] = pb.deribit_iv, pb.method
    p["gap"] = 100 * (p.atm_iv - p.deribit_iv)
    p["gap_sig"] = 100 * (p.atm_iv - p.deribit_sig)
    p["gap_near"] = 100 * (p.atm_iv - p.deribit_near)
    p["hours_left"] = p.tau_years * 8760
    p["h_bucket"] = pd.cut(p.hours_left, H_EDGES).astype(str)
    # structural gap: mean gap_sig by kind and hours-left bucket over earlier days only (no look-ahead)
    p["day"] = p.ts.dt.normalize()
    daily = p.groupby(["kind", "h_bucket", "day"]).gap_sig.agg(["sum", "count"]).reset_index()
    daily[["cs", "cc"]] = daily.groupby(["kind", "h_bucket"])[["sum", "count"]].cumsum().values
    daily["past_mean"] = (daily.cs - daily["sum"]) / (daily.cc - daily["count"]).replace(0, np.nan)
    p = p.merge(daily[["kind", "h_bucket", "day", "past_mean"]], on=["kind", "h_bucket", "day"], how="left")
    p["gap_dm"] = p.gap_sig - p.past_mean
    r1 = np.log(spot).diff(60)
    p["abs_ret_1h"] = np.abs(r1.reindex(p.ts).to_numpy())
    p["ret_1h"] = r1.reindex(p.ts).to_numpy()
    p.to_parquet(OUT / "vol_panel.parquet", index=False)
    print(f"inputs done: panel {len(p):,} rows ({time.time() - t0:.0f}s)")
    step_leadlag(d, p, profile)


def step_leadlag(d, p, profile):
    """Lead-lag at 15-minute resolution: Deribit's ATM vol at Kalshi's expiry from trades in the 15 minutes before
    each quarter hour, slid past Kalshi's hourly snapshots; correlation of Kalshi's hourly change (into the hour and
    the next hour) with Deribit's 1-hour change ending e minutes after the snapshot."""
    g = dm.smile_grid(d, freq="15min", half_window="7.5min", centred=False)
    p = p[["event_ticker", "ts", "expiry", "tau_years", "kind", "atm_iv"]].copy()
    D = {}
    for off in range(-120, 76, 15):
        s = p.assign(ts=p.ts + pd.Timedelta(minutes=off), tau_years=p.tau_years - off / 525_600)
        D[off] = 100 * vs.deribit_at_kalshi(s, g, profile).deribit_iv.to_numpy()
    D = pd.DataFrame(D, index=p.index)
    p["K"] = 100 * p.atm_iv
    rows = []
    for kind in ("weekly", "daily"):
        m = p[p.kind == kind]
        nxt = m[["event_ticker", "ts", "K"]].assign(ts=m.ts - pd.Timedelta(hours=1)).rename(columns={"K": "K_next"})
        prv = m[["event_ticker", "ts", "K"]].assign(ts=m.ts + pd.Timedelta(hours=1)).rename(columns={"K": "K_prev"})
        mm = m.merge(nxt, on=["event_ticker", "ts"], how="left").merge(prv, on=["event_ticker", "ts"], how="left")
        mm.index = m.index
        dk_now, dk_next = mm.K - mm.K_prev, mm.K_next - mm.K
        for e in range(-60, 76, 15):
            dd = D.loc[m.index, e] - D.loc[m.index, e - 60]
            j = pd.concat([dk_now.rename("now"), dk_next.rename("next"), dd.rename("d")], axis=1)
            rows.append(dict(kind=kind, deribit_hour_ends=e, corr_with_kalshi_hour_ending_t=j.now.corr(j.d),
                             corr_with_kalshi_next_hour=j.next.corr(j.d), n=int(j[["now", "d"]].dropna().shape[0])))
    pd.DataFrame(rows).to_parquet(OUT / "leadlag_15m.parquet", index=False)


def step_dislocation(mkt):
    """Dislocations (gap minus its recent normal), dislocation trades, and the prediction datasets for 11."""
    t0 = time.time()
    p = pd.read_parquet(OUT / "vol_panel.parquet")
    p = p[p.deribit_sig.notna()].reset_index(drop=True)
    p = dl.add_dislocation(p)
    spot = cleaning.clean_spot(pd.read_parquet(ROOT / "data" / "raw" / "btc_spot" / "btcusd_1m.parquet")).spot
    c = pd.read_csv(ROOT / "data" / "kalshi_btc_contracts.csv.gz",
                    usecols=["event_ticker", "strike", "result", "settlement_value"])
    settle = c.groupby("event_ticker").settlement_value.first()
    p["settle"] = p.event_ticker.map(settle)
    for h in (1, 4, 24):
        p[f"ret_f{h}"] = np.log(spot.reindex(p.ts + pd.Timedelta(hours=h)).to_numpy() / p.spot.to_numpy())
    p["ret_expiry"] = np.log(p.settle / p.spot)

    # Deribit's smile shape (nearest expiry, backward window) at each snapshot, levelled to Deribit's ATM vol at
    # Kalshi's expiry: Deribit's probabilities at Kalshi's strikes and expiry
    gb = pd.read_parquet(OUT / "deribit_grid_backward.parquet")
    by_t = {t: g for t, g in gb.groupby("t")}
    shape = []
    for r in p.itertuples(index=False):
        g = by_t.get(r.ts)
        if g is None:
            shape.append((np.nan,) * 5)
            continue
        j = np.argmin(np.abs((g.expiry - r.expiry).dt.total_seconds().to_numpy()))
        q = g.iloc[j]
        shape.append((q.b, q.c, q.m_lo, q.m_hi, q.basis))
    p[["d_b", "d_c", "d_m_lo", "d_m_hi", "d_basis"]] = shape
    p["d_forward"] = p.spot * (1 + p.d_basis)
    p["k_p_up"] = np.nan
    p["d_p_up"] = dl.digital_from_deribit(p.spot, p.d_forward, p.tau_years, p.deribit_sig, p.d_b, p.d_c,
                                          p.d_m_lo, p.d_m_hi)
    strikes = pd.read_parquet(ROOT / "data" / "derived" / "kalshi_hourly_strikes.parquet")
    strikes = strikes.merge(p[["event_ticker", "ts", "kind", "hours_left", "tau_years", "spot", "d_forward",
                               "deribit_sig", "d_b", "d_c", "d_m_lo", "d_m_hi", "fine_grid", "disl", "day"]],
                            on=["event_ticker", "ts"], how="inner")
    strikes = strikes.merge(c[["event_ticker", "strike", "result"]], on=["event_ticker", "strike"], how="inner")
    strikes = strikes[strikes.result.isin(["yes", "no"]) & strikes.deribit_sig.notna()]
    strikes["y"] = (strikes.result == "yes").astype(float)
    strikes["p_kalshi"] = strikes.D_used
    strikes["p_deribit"] = dl.digital_from_deribit(strikes.strike, strikes.d_forward, strikes.tau_years,
                                                   strikes.deribit_sig, strikes.d_b, strikes.d_c, strikes.d_m_lo,
                                                   strikes.d_m_hi)
    strikes.to_parquet(OUT / "forecast_strikes.parquet", index=False)
    # Kalshi's probability that BTC ends above today's price: its monotone ladder interpolated at spot
    pu = {}
    for (ev, ts), g in strikes.groupby(["event_ticker", "ts"]):
        g = g.sort_values("strike")
        if g.strike.iloc[0] < g.spot.iloc[0] < g.strike.iloc[-1]:
            pu[(ev, ts)] = float(np.interp(g.spot.iloc[0], g.strike, g.D_used))
    p["k_p_up"] = [pu.get(k, np.nan) for k in zip(p.event_ticker, p.ts)]
    p.to_parquet(OUT / "disl_panel.parquet", index=False)
    print(f"  dislocation panel and forecasts ({time.time() - t0:.0f}s)", flush=True)

    q = kalshi_hourly.load_quotes(ROOT)
    q = q[q.ts >= p.ts.min()]
    res = c[c.result.isin(["yes", "no"])].assign(result_yes=lambda x: x.result == "yes")
    cand = p[p.disl.abs() >= 2].reset_index(drop=True)
    parts = []
    for hold in (1, 4, 24, "expiry"):
        b = vs.backtest(cand, q, res, mkt, cs.Params(), "disl", 0.0, hold)
        parts.append(b)
        print(f"  dislocation trades hold={hold}: {len(b):,} ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(OUT / "disl_trades.parquet", index=False)


def step_ab(mkt):
    """Strategies A and B (lib/strategy_ab.py): trades for three 'normal gap' windows and three holding periods."""
    t0 = time.time()
    base = pd.read_parquet(OUT / "vol_panel.parquet")
    base = base[base.deribit_sig.notna()].reset_index(drop=True)
    q = kalshi_hourly.load_quotes(ROOT)
    q = q[q.ts >= base.ts.min()]
    c = pd.read_csv(ROOT / "data" / "kalshi_btc_contracts.csv.gz", usecols=["event_ticker", "strike", "result"])
    res = c[c.result.isin(["yes", "no"])].assign(result_yes=lambda x: x.result == "yes")
    parts = []
    for window in (15, 30, 60):
        p = dl.add_dislocation(base, days=window)
        a = p[(p.kind == "weekly") & (p.disl <= -2)].reset_index(drop=True)
        b = p[(p.kind == "daily") & (p.disl >= 2)].reset_index(drop=True)
        for cand, holds in ((a, (4, 24, "expiry")), (b, (4, "expiry"))):
            for hold in holds:
                tr = vs.backtest(cand, q, res, mkt, cs.Params(), "disl", 0.0, hold)
                parts.append(tr.assign(window=window))
        print(f"  A/B trades, {window}-day normal ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(OUT / "ab_trades.parquet", index=False)


HEDGE_DAY = ("KXBTCD-25OCT0917", [(123000.0, "2025-10-09 00:00"), (121000.0, "2025-10-09 12:00")])


def step_hedge(cross: bool = True):
    """Hedging a long Kalshi binary (13): the 9 Oct 2025 case (data from 00_fetch_kalshi_event.py) and, if `cross`,
    every 2026 Kalshi daily, frictionless, for each hedging scheme in lib/binary_hedge.py."""
    t0 = time.time()
    spot = cleaning.clean_spot(pd.read_parquet(ROOT / "data" / "raw" / "btc_spot" / "btcusd_1m.parquet")).spot
    profile = deribit.hourly_variance(spot)
    event, cases = HEDGE_DAY
    fr, mids, mk = bh.build_day(ROOT, event, cases[0][1], profile)
    settle = float(mk.expiration_value.iloc[0])
    fr.to_parquet(OUT / "hedge_day_inputs.parquet")
    pd.DataFrame(mids).to_parquet(OUT / "hedge_day_kalshi_mids.parquet")
    steps, summ = [], []
    for K, start in cases:
        path = fr.loc[start:].assign(kalshi_mid=mids[K].loc[start:])
        for name, kw in bh.SCHEMES.items():
            r = bh.simulate(path, K, settle, **kw)
            steps.append(r["steps"].reset_index().assign(K=K, scheme=name))
            summ.append({**r["summary"], "K": K, "start": start, "scheme": name, "settle": settle})
    pd.concat(steps, ignore_index=True).to_parquet(OUT / "hedge_day_steps.parquet", index=False)
    pd.DataFrame(summ).to_parquet(OUT / "hedge_day_summary.parquet", index=False)
    print(f"  hedging: 9 Oct 2025 done ({time.time() - t0:.0f}s)", flush=True)
    if not cross:
        return
    panel = pd.read_parquet(OUT / "disl_panel.parquet")
    strikes = pd.read_parquet(ROOT / "data" / "derived" / "kalshi_hourly_strikes.parquet")
    c = pd.read_csv(ROOT / "data" / "kalshi_btc_contracts.csv.gz", usecols=["event_ticker", "settlement_value"])
    settle_all = c.groupby("event_ticker").settlement_value.first()
    bh.cross_day(panel, strikes, spot, settle_all).to_parquet(OUT / "hedge_cross_day.parquet", index=False)
    print(f"  hedging: every 2026 daily done ({time.time() - t0:.0f}s)", flush=True)


def load_market(start=None, end=None):
    d = dm.load_options(ROOT, start, end)
    grid = pd.read_parquet(OUT / "deribit_grid_centred.parquet")
    perp = pd.concat(map(pd.read_parquet, sorted((ROOT / "data" / "raw" / "deribit" / "perp").glob("*.parquet"))))
    fpath = ROOT / "data" / "raw" / "deribit" / "funding.parquet"
    funding = pd.read_parquet(fpath) if fpath.exists() else pd.DataFrame(
        {"ts": pd.DatetimeIndex([], tz="UTC"), "interest_1h": []})
    dl = pd.read_parquet(OUT / "deribit_delivery.parquet")
    delivery = pd.Series(dl.delivery.to_numpy(), index=pd.to_datetime(dl.date).dt.date)
    return cs.Market(d, perp, funding, delivery, grid)


def step_volspread(mkt):
    t0 = time.time()
    p = pd.read_parquet(OUT / "vol_panel.parquet")
    q = kalshi_hourly.load_quotes(ROOT)
    q = q[q.ts >= p.ts.min()]
    c = pd.read_csv(ROOT / "data" / "kalshi_btc_contracts.csv.gz", usecols=["event_ticker", "strike", "result"])
    res = c[c.result.isin(["yes", "no"])].assign(result_yes=lambda x: x.result == "yes")
    parts = []
    for signal in ("gap_sig", "gap_dm"):
        for hold in (1, 4, 24, "expiry"):
            b = vs.backtest(p, q, res, mkt, cs.Params(), signal, 0.0, hold)
            parts.append(b.assign(signal_name=signal))
            print(f"  volspread {signal} hold={hold}: {len(b):,} trades ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(OUT / "volspread_trades.parquet", index=False)


def step_calendar(mkt):
    t0 = time.time()
    daily, trades = [], []
    for period, (start, end) in {"2026": ("2026-01-15", "2026-10-01"), "2025Q4": ("2025-10-02", "2026-01-15")}.items():
        for name, (desc, kw) in CALENDAR_VARIANTS.items():
            if period != "2026" and name not in ("base", "gross", "best_practical", "short_only", "long_only", "short_only_expiry", "overnight_expiry"):
                continue
            r = cs.run(mkt, cs.Params(start=start, end=end, **kw))
            daily.append(r["daily"].reset_index().assign(variant=name, period=period))
            if name in ("base", "best_practical"):
                trades.append(r["trades"].assign(variant=name, period=period))
            print(f"  calendar {period} {name}: mean {r['daily'].pnl.mean():+.0f}/day ({time.time() - t0:.0f}s)",
                  flush=True)
    pd.concat(daily, ignore_index=True).to_parquet(OUT / "calendar_daily.parquet", index=False)
    pd.concat(trades, ignore_index=True).to_parquet(OUT / "calendar_trades.parquet", index=False)
    pd.DataFrame([(k, v[0]) for k, v in CALENDAR_VARIANTS.items()], columns=["variant", "description"]).to_parquet(
        OUT / "calendar_variants.parquet", index=False)


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--only", choices=["inputs", "volspread", "calendar", "dislocation", "ab", "hedge", "hedge_day"])
    args = a.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "inputs"):
        step_inputs()
    if args.only in (None, "hedge"):
        step_hedge()
    if args.only == "hedge_day":
        step_hedge(cross=False)
    if args.only in (None, "volspread", "calendar", "dislocation", "ab"):
        mkt = load_market()
        if args.only in (None, "ab"):
            step_ab(mkt)
        if args.only in (None, "dislocation"):
            step_dislocation(mkt)
        if args.only in (None, "volspread"):
            step_volspread(mkt)
        if args.only in (None, "calendar"):
            step_calendar(mkt)


if __name__ == "__main__":
    main()
