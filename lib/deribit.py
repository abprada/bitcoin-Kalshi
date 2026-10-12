"""Deribit BTC option trades: loading, validity flags and ATM vols (used by 04_deribit_exploration.py).

Deribit options are inverse: quoted and settled in BTC, paying (S - K)/S BTC for a call. Valued in USD that is
the ordinary payoff S - K, so price_btc * index_price is a standard USD option price and inverts with the usual
Black-Scholes formula. Every expiry is 08:00 UTC. The trades file comes from a collaborator and has no trade id,
so exact duplicate rows cannot be told apart from separate identical fills; they are flagged, not dropped.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lib import iv

YEAR_SECONDS = 365 * 86_400
IV_VALID_PCT = (0.0, 500.0)   # Deribit's iv column uses 0 and 999 as placeholders; keep strictly inside this range
ATM_BAND = 0.01               # |ln(K / index)| below this counts as at the money


def load_trades(path: Path) -> pd.DataFrame:
    """Read the trades CSV and add time to expiry, moneyness, decimal vols and flags.

    Added columns: `ts`, `expiry` (UTC), `tau_years`, `log_moneyness` = ln(K / index), `iv_deribit` (decimal,
    NaN for placeholders), `iv_mark` (our Black-Scholes inversion of the mark price, forward = index),
    `iv_valid`, `exact_dup` (True for every repeat of an identical earlier row).
    """
    d = pd.read_csv(path)
    d["ts"] = pd.to_datetime(d.time_utc, format="ISO8601", utc=True)
    d["expiry"] = pd.to_datetime(d.expiry_utc, utc=True)
    d["tau_years"] = (d.expiry - d.ts).dt.total_seconds() / YEAR_SECONDS
    d["log_moneyness"] = np.log(d.strike / d.index_price)
    d["iv_valid"] = (d.iv > IV_VALID_PCT[0]) & (d.iv < IV_VALID_PCT[1])
    d["iv_deribit"] = d.iv.where(d.iv_valid) / 100
    d["iv_mark"] = mark_iv(d)
    d["exact_dup"] = d.drop(columns=["ts", "expiry"]).duplicated(keep="first")
    return d


def mark_iv(d: pd.DataFrame) -> np.ndarray:
    """Black-Scholes vol of each row's mark price, with the index as the forward (NaN when not invertible)."""
    is_call = (d.cp == "C").to_numpy()
    tau = d.tau_years.to_numpy(dtype=float)
    out = np.full(len(d), np.nan)
    ok = tau > 0
    out[ok] = iv.implied_vol((d.mark_price_btc * d.index_price).to_numpy(dtype=float)[ok],
                             d.index_price.to_numpy(dtype=float)[ok], d.strike.to_numpy(dtype=float)[ok],
                             tau[ok], is_call[ok])
    return out


def hourly_variance(spot: pd.Series) -> pd.Series:
    """Average variance per hour for each hour of the week (0 = Monday 00:00 UTC), from 1-minute prices.

    `spot` is indexed by UTC minute. Squared 1-minute log returns are averaged per hour of the week and scaled
    to one hour, so the profile shows which hours BTC usually moves most.
    """
    r = np.log(spot.dropna()).diff().dropna()
    how = r.index.dayofweek * 24 + r.index.hour
    return (r**2).groupby(how).mean().mul(60).reindex(range(168))


def expected_variance(start: pd.Timestamp, end: pd.Timestamp, profile: pd.Series) -> float:
    """Variance from `start` to `end` implied by an hour-of-week profile, counting partial hours pro rata."""
    total, t = 0.0, start
    while t < end:
        nxt = min(t.floor("h") + pd.Timedelta(hours=1), end)
        total += profile.iloc[t.dayofweek * 24 + t.hour] * (nxt - t).total_seconds() / 3600
        t = nxt
    return total


def horizon_vol_ratio(start: pd.Timestamp, long_end: pd.Timestamp, short_end: pd.Timestamp,
                      profile: pd.Series) -> float:
    """Annualized vol to `long_end` divided by annualized vol to `short_end`, both from `start`, if every hour
    carried its usual variance. Above 1 when the extra hours are busier than average."""
    per_hour = lambda end: expected_variance(start, end, profile) / ((end - start).total_seconds() / 3600)
    return float(np.sqrt(per_hour(long_end) / per_hour(short_end)))


def _cumulative_variance(ts, profile: pd.Series) -> np.ndarray:
    """Variance accumulated from Monday 1970-01-05 00:00 UTC to each time in `ts`, partial hours pro rata."""
    h = (pd.DatetimeIndex(ts) - pd.Timestamp("1970-01-05", tz="UTC")) / pd.Timedelta(hours=1)
    h = np.asarray(h, dtype=float)
    full = np.floor(h).astype(int)
    p = profile.to_numpy(dtype=float)
    cum = np.r_[0.0, np.cumsum(p)]
    return (full // 168) * cum[-1] + cum[full % 168] + p[full % 168] * (h - full)


def horizon_vol_ratios(start, long_end, short_end, profile: pd.Series) -> np.ndarray:
    """`horizon_vol_ratio` for arrays of times at once (same result, vectorized)."""
    t0, t_long, t_short = (_cumulative_variance(x, profile) for x in (start, long_end, short_end))
    hours = lambda a, b: np.asarray((pd.DatetimeIndex(b) - pd.DatetimeIndex(a)) / pd.Timedelta(hours=1), dtype=float)
    return np.sqrt(((t_long - t0) / hours(start, long_end)) / ((t_short - t0) / hours(start, short_end)))


def atm_vol(d: pd.DataFrame, by: list[str], col: str = "iv_deribit", band: float = ATM_BAND,
            min_trades: int = 5) -> pd.DataFrame:
    """Median vol of trades within `band` of the money, per group; groups with fewer than `min_trades` dropped.

    Returns the `by` columns plus `atm_iv`, `n_trades` and `hours_to_expiry` (median over the group's trades).
    """
    x = d[(d.log_moneyness.abs() < band) & d[col].notna() & (d.tau_years > 0)]
    g = x.groupby(by).agg(atm_iv=(col, "median"), n_trades=(col, "size"),
                          hours_to_expiry=("tau_years", lambda t: 8_760 * t.median()))
    return g[g.n_trades >= min_trades].reset_index()
