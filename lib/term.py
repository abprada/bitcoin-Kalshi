"""Term structure, implied vs realized vol, and trading costs, for Kalshi and Deribit (used by 07_section1.py).

- ATM vols at every horizon: Kalshi's weeklies (03's snapshots), Kalshi's Monday-Thursday dailies (built here from
  the raw candles with the same cleaning and smile method), and Deribit trades near the money.
- Realized vol to expiry: what BTC actually did between an observation and its contract's expiry, from 5-minute
  log returns, annualized like the implied vols (ACT/365).
- Costs: Kalshi's bid, ask and fee against Deribit's price for the same binary, and the Deribit fees for the
  call spread that would hedge it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lib import cleaning, compare, deribit, iv, pairs

YEAR_SECONDS = 365 * 86_400
RV_STEP_MIN = 5                    # realized vol from 5-minute returns: finer grids pick up exchange noise
SNAPSHOT_EVERY_MIN = 15            # as in 03
ATM_SD = 0.25                      # "at the money": within a quarter of a standard deviation, as in lib/pairs.py
HOUR_EDGES = [1, 6, 12, 24, 48, 72, 120, 170]
HOUR_LABELS = ["1–6h", "6–12h", "12–24h", "1–2d", "2–3d", "3–5d", "5–7d"]
DERIBIT_PERIOD = (pd.Timestamp("2026-01-15", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"))  # Deribit file's span
DERIBIT_FEE_RATE = 0.0003          # Deribit BTC option fee: 0.03% of the underlying per 1-BTC contract ...
DERIBIT_FEE_CAP = 0.125            # ... at most 12.5% of the option's price (Deribit fee schedule, checked Oct 2026)
SHORT_HOURS = (6, 36)              # term-structure spread: the short leg is the nearest expiry 6-36 hours out ...
LONG_HOURS = (96, 200)             # ... the long leg the expiry closest to a week, 4-8 days out


def hour_bucket(hours) -> pd.Categorical:
    return pd.cut(hours, HOUR_EDGES, labels=HOUR_LABELS)


# --- realized vol --------------------------------------------------------------------------------


def cumulative_variance(spot: pd.Series, step_min: int = RV_STEP_MIN) -> pd.Series:
    """Running sum of squared `step_min`-minute log returns, on a regular grid of UTC minute ends.

    `spot` is indexed by UTC minute end; gaps are filled with the last price (no return across the gap).
    The realized variance between two times is the difference of this series at the grid points below them.
    """
    s = spot.dropna()
    s = s.reindex(pd.date_range(s.index.min().ceil(f"{step_min}min"), s.index.max(), freq=f"{step_min}min")).ffill()
    return np.log(s).diff().pow(2).fillna(0.0).cumsum()


def realized_vol(cum: pd.Series, start, end) -> np.ndarray:
    """Annualized realized vol from `start` to `end` (arrays of UTC times), NaN where the grid does not cover both."""
    step = cum.index[1] - cum.index[0]
    t0, t1 = pd.DatetimeIndex(start).floor(step), pd.DatetimeIndex(end).floor(step)
    var = cum.reindex(t1).to_numpy() - cum.reindex(t0).to_numpy()
    years = np.asarray((t1 - t0).total_seconds(), dtype=float) / YEAR_SECONDS
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(years > 0, np.sqrt(var / years), np.nan)


# --- ATM vols per market -------------------------------------------------------------------------


def kalshi_weekly_atm(snapshots: pd.DataFrame, usable_only: bool = True) -> pd.DataFrame:
    """One row per usable 03 snapshot (ATM vol exists, at least 2 strikes per standard deviation).

    With `usable_only=False`, every snapshot with an ATM vol, and `strikes_per_sd` kept as a precision flag: close
    to expiry the grid check keeps only high-vol snapshots (one SD is sigma x sqrt(tau)), which biases vols upward.
    """
    s = snapshots[compare.usable_kalshi_snapshots(snapshots) if usable_only else snapshots.atm_iv.notna()]
    d = pd.DataFrame({"market": "Kalshi weekly", "event": s.event_ticker, "ts": s.ts,
                      "expiry": (s.ts + pd.to_timedelta(s.hours_to_expiry, unit="h")).dt.round("min"),
                      "hours": s.hours_to_expiry, "atm_iv": s.atm_iv})
    if not usable_only:
        d["strikes_per_sd"] = s.strikes_per_sd
    return d.reset_index(drop=True)


def kalshi_daily_atm(root: Path, every_min: int = SNAPSHOT_EVERY_MIN, usable_only: bool = True) -> pd.DataFrame:
    """ATM vol of Kalshi's Monday-Thursday 5pm ET dailies every `every_min` minutes, cached in data/derived/.

    Raw candles from `00_fetch_data.py --product daily`, cleaned with 02's rules (`cleaning.event_quotes`) and
    turned into smiles with 03's method (`pairs.kalshi_smiles`); usable as in `kalshi_weekly_atm`, or every
    snapshot with an ATM vol and its `strikes_per_sd` when `usable_only=False`.
    """
    cache = root / "data" / "derived" / "kalshi_daily_atm_all.parquet"
    raw = root / "data" / "raw" / "kalshi_btc_daily"
    if cache.exists() and cache.stat().st_mtime >= (raw / "markets.parquet").stat().st_mtime:
        return _daily_view(pd.read_parquet(cache), usable_only)
    spot = pd.read_parquet(root / "data" / "clean" / "btc_spot_1m.parquet").set_index("ts")
    markets = pd.read_parquet(raw / "markets.parquet")
    rows = []
    for ev, m in markets.groupby("event_ticker"):
        candles = pd.read_parquet(raw / "candles" / f"{ev}.parquet",
                                  columns=["market_ticker", "ts", "bid_close", "ask_close", "volume", "open_interest"])
        if candles.empty:
            continue
        q = cleaning.event_quotes(candles, m, spot.spot, spot.spot_stale)
        minutes = q.ts[q.ts.dt.minute % every_min == 0].unique()
        for ts, (snap, _) in pairs.kalshi_smiles(q, minutes).items():
            rows.append({"event": ev, "ts": ts, "expiry": m.close_time.max(), "hours": snap["tau_years"] * 8760,
                         "atm_iv": snap["atm_iv"], "strikes_per_sd": snap["strikes_per_sd"]})
    d = pd.DataFrame(rows)
    d = d[d.atm_iv.notna()].reset_index(drop=True)
    d.insert(0, "market", "Kalshi daily")
    cache.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(cache, index=False)
    return _daily_view(d, usable_only)


def _daily_view(d: pd.DataFrame, usable_only: bool) -> pd.DataFrame:
    if not usable_only:
        return d
    return d[compare.usable_kalshi_snapshots(d)].drop(columns="strikes_per_sd").reset_index(drop=True)


def deribit_atm(trades: pd.DataFrame) -> pd.DataFrame:
    """Deribit's ATM vol: the median mark vol of trades within ATM_SD standard deviations of the index, one row per
    expiry, day and trading window. `trades` as from `deribit.load_trades`."""
    t = trades[trades.iv_mark.notna() & (trades.tau_years > 0)]
    t = t[(t.log_moneyness / (t.iv_mark * np.sqrt(t.tau_years))).abs() <= ATM_SD]
    g = (t.groupby(["expiry", "date", "window"])
         .agg(ts=("ts", "median"), atm_iv=("iv_mark", "median"), tau=("tau_years", "median"), n=("iv_mark", "size"))
         .reset_index())
    return pd.DataFrame({"market": "Deribit", "event": g.expiry.dt.strftime("%Y-%m-%d"), "ts": g.ts,
                         "expiry": g.expiry, "hours": g.tau * 8760, "atm_iv": g.atm_iv, "window": g.window,
                         "n_trades": g.n})


def with_realized(atm: pd.DataFrame, cum: pd.Series) -> pd.DataFrame:
    """Add each observation's realized vol to its expiry (`rv`) and its horizon bucket (`h`)."""
    return atm.assign(rv=realized_vol(cum, atm.ts, atm.expiry), h=hour_bucket(atm.hours))


def implied_vs_realized(d: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Per market and horizon: implied and realized vol as root-mean-square (the vol of the average variance, which
    is what a variance seller earns), their difference in vol points, and a 95% interval for it that resamples
    whole expiries (observations of one contract share one realized path, so they are not independent). Also the
    median of implied - realized per observation (`median_diff`, points), which a few crash days move less."""
    rng = np.random.default_rng(seed)
    rows = []
    for (mkt, h), g in d.dropna(subset=["rv", "h"]).groupby(["market", "h"], observed=True):
        per = g.groupby("expiry").agg(iv2=("atm_iv", lambda x: (x**2).sum()), rv2=("rv", lambda x: (x**2).sum()),
                                      n=("rv", "size"))
        iv2, rv2, n = per.iv2.to_numpy(), per.rv2.to_numpy(), per.n.to_numpy()
        prem = lambda i: 100 * (np.sqrt(iv2[i].sum() / n[i].sum()) - np.sqrt(rv2[i].sum() / n[i].sum()))
        boot = [prem(rng.integers(0, len(per), len(per))) for _ in range(n_boot)]
        rows.append({"market": mkt, "h": h, "obs": len(g), "expiries": len(per),
                     "implied": np.sqrt(iv2.sum() / n.sum()), "realized": np.sqrt(rv2.sum() / n.sum()),
                     "premium": prem(np.arange(len(per))), "ci_low": np.percentile(boot, 2.5),
                     "ci_high": np.percentile(boot, 97.5), "median_diff": 100 * (g.atm_iv - g.rv).median(),
                     "implied_higher": (g.atm_iv > g.rv).mean()})
    return pd.DataFrame(rows)


# --- short-dated minus one-week ------------------------------------------------------------------


def kalshi_term_spread(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """At every 15-minute mark when both trade: the nearest daily's ATM vol minus the weekly's (vol points)."""
    d = daily.sort_values("hours").drop_duplicates("ts")
    m = d.merge(weekly, on="ts", suffixes=("_short", "_long"))
    return m.assign(spread=100 * (m.atm_iv_short - m.atm_iv_long))[
        ["ts", "hours_short", "hours_long", "atm_iv_short", "atm_iv_long", "spread"]]


def deribit_term_spread(atm: pd.DataFrame) -> pd.DataFrame:
    """Per day and window: the nearest Deribit expiry 6-36 hours out minus the expiry closest to 7 days, 4-8 days
    out (vol points)."""
    short = atm[atm.hours.between(*SHORT_HOURS)].sort_values("hours").drop_duplicates(["ts", "window"])
    long = (atm[atm.hours.between(*LONG_HOURS)].assign(off=lambda x: (x.hours - 168).abs())
            .sort_values("off").drop_duplicates(["ts", "window"]))
    key = lambda x: x.assign(day=x.ts.dt.date)[["day", "window", "ts", "hours", "atm_iv"]]
    m = key(short).merge(key(long).drop(columns="ts"), on=["day", "window"], suffixes=("_short", "_long"))
    return m.assign(spread=100 * (m.atm_iv_short - m.atm_iv_long))


# --- costs ---------------------------------------------------------------------------------------


def deribit_spread_fee(F, K, sigma, tau, half_width: float = compare.SPREAD_HALF_WIDTH,
                       rate: float = DERIBIT_FEE_RATE, cap: float = DERIBIT_FEE_CAP) -> np.ndarray:
    """Deribit fees, in dollars per $1 binary, for the two-leg spread K -/+ half_width that replicates it.

    The binary is 1 / (2 x half_width) BTC of each leg. Each leg uses the cheaper out-of-the-money option (a call
    spread above the forward, a put spread below: the same payoff by put-call parity), and its fee per BTC is
    rate x F, at most cap x the option's price. Traded as one combo (one leg bought, one sold), Deribit waives the
    fees of the cheaper direction, so only the dearer leg's fee is paid (Deribit "Option Combo Order", checked Oct 2026).
    """
    F, K, sigma = (np.asarray(x, dtype=float) for x in (F, K, sigma))
    fees = [np.minimum(rate * F, cap * np.minimum(iv.bs_call(F, k, sigma, tau), iv.bs_put(F, k, sigma, tau)))
            for k in (K - half_width, K + half_width)]
    return np.maximum(*fees) / (2 * half_width)


def kalshi_edges(matched: pd.DataFrame, contracts: int = 100) -> pd.DataFrame:
    """Kalshi's quotes against Deribit's price for the same binary, after Kalshi's taker fee (cents).

    `matched` from `compare.build_comparison`, with `kalshi_atm`. Adds `gap_c` (Kalshi mid - Deribit), `buy_edge_c`
    (Deribit - ask - fee: buying YES on Kalshi below Deribit's price) and `sell_edge_c` (bid - fee - Deribit),
    `edge_c` (the better of the two), `kalshi_cost_c` (half the spread plus the fee) and `hedge_fee_c` (Deribit
    fees on the replicating spread), plus `sd` and `tte_bucket`.
    """
    m = matched.dropna(subset=["deribit_digital_adj", "bid", "ask", "kalshi_atm"]).copy()
    d = m.deribit_digital_adj
    m["gap_c"] = 100 * (m.mid - d)
    m["buy_edge_c"] = 100 * (d - m.ask - compare.kalshi_fee(m.ask, contracts=contracts))
    m["sell_edge_c"] = 100 * (m.bid - compare.kalshi_fee(m.bid, contracts=contracts) - d)
    m["edge_c"] = np.maximum(m.buy_edge_c, m.sell_edge_c)
    m["kalshi_cost_c"] = 100 * ((m.ask - m.bid) / 2 + compare.kalshi_fee(m.mid, contracts=contracts))
    m["hedge_fee_c"] = 100 * deribit_spread_fee(m.spot, m.strike, m.deribit_vol_adj, m.tau_years)
    m["sd"] = np.log(m.strike / m.spot) / (m.kalshi_atm * np.sqrt(m.tau_years))
    return m
