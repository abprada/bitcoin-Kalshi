"""Deribit -> Kalshi vol factor from the two settlement rules (17_settlement_factor.py).

Deribit's Friday option settles at 08:00 UTC on the average of its BTC index over the last 30 minutes; Kalshi's weekly
at 5pm ET (21:00 UTC in summer, 22:00 in winter) on the average of CF Benchmarks' index over the last 60 seconds.
Each market quotes Black-Scholes vols to its own expiry, so the two vols differ even when both price the same BTC.
Here both are priced on one model of BTC and compared strike by strike:

- BTC: a geometric Brownian motion with no drift whose variance follows the hour-of-week profile
  (`deribit.hourly_variance`), so busy hours carry more. Rates are zero, as in `iv`.
- Settlement = S at the window's start x B, where B = average of S over the window / S at its start, sampled every
  second. S at the window's start is lognormal and B is independent of it, so an option on the settlement is the
  average over simulated B of B x Black-Scholes(strike / B): only the window is simulated (conditional Monte Carlo).
- Each venue's out-of-the-money prices are inverted with its own maturity, as the markets quote them, and the factor
  is R = Kalshi vol / Deribit vol at the same strike. Deribit's vol x R is Deribit on Kalshi's terms.

Model-dependent: a fixed variance schedule, no jumps or stochastic vol. Under it each settlement is close to
lognormal, so R depends on the strike far less than on the time to expiry.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from lib import deribit, iv

YEAR_HOURS = 8_760
DERIBIT_WINDOW_MIN, KALSHI_WINDOW_MIN = 30, 1
MAX_FACTOR_SE = 0.002          # factors with a larger Monte Carlo standard error are not reliable
MIN_PRICE = 1e-8               # cheaper options (x forward) are lost in rounding: their vols are not reliable


def window_steps(end: pd.Timestamp, minutes: float, profile: pd.Series, step_s: int = 1) -> np.ndarray:
    """Expected variance of each `step_s`-second step in the `minutes` before `end` (empty for a zero window)."""
    t = pd.date_range(end - pd.Timedelta(minutes=minutes), end, freq=f"{step_s}s")
    return np.diff(deribit._cumulative_variance(t, profile))


def window_average(step_var, n_paths: int = 20_000, seed: int = 0) -> np.ndarray:
    """Draws of B = average of S over the window / S at its start, S sampled at the end of each step (no drift, so
    E[B] = 1). Antithetic: draws i and i + n_paths // 2 have opposite shocks. With no steps, B = 1."""
    step_var = np.asarray(step_var, dtype=float)
    half = n_paths // 2
    if len(step_var) == 0:
        return np.ones(2 * half)
    rng = np.random.default_rng(seed)
    y, total = np.zeros(2 * half), np.zeros(2 * half)
    for v in step_var:
        z = rng.standard_normal(half)
        y += -0.5 * v + np.sqrt(v) * np.r_[z, -z]
        total += np.exp(y)
    return total / len(step_var)


def option_prices(pre_var: float, B, K) -> dict:
    """Calls and puts on the settlement S_w x B, forward 1: S_w lognormal with mean 1 and log-variance `pre_var` (the
    variance expected before the window), B the window draws, rescaled to mean exactly 1 so put-call parity holds.
    Returns `call`, `put` and their standard errors `call_se`, `put_se` (from the antithetic pairs)."""
    B = np.asarray(B, dtype=float)
    B = B / B.mean()
    K = np.atleast_1d(np.asarray(K, dtype=float))
    call = B[:, None] * iv.bs_call(1.0, K[None, :] / B[:, None], np.sqrt(max(pre_var, 1e-18)), 1.0)
    put = call - (B[:, None] - K[None, :])          # parity path by path: B x (call - put at K / B) = B - K
    half = len(B) // 2

    def se(x):
        if half < 2:
            return np.zeros(x.shape[1])
        return (0.5 * (x[:half] + x[half:2 * half])).std(axis=0, ddof=1) / np.sqrt(half)

    return {"call": call.mean(axis=0), "put": put.mean(axis=0), "call_se": se(call), "put_se": se(put)}


def settlement_vols(pre_var: float, B, K, tau_years: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Black-Scholes vols (forward 1, maturity `tau_years`) of the out-of-the-money options on the settlement (calls
    at K >= 1, puts below), their Monte Carlo standard errors (price error / vega) and the option prices."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    p = option_prices(pre_var, B, K)
    is_call = K >= 1
    price = np.where(is_call, p["call"], p["put"])
    vol = iv.implied_vol(price, 1.0, K, tau_years, is_call)
    s = vol * np.sqrt(tau_years)
    vega = np.exp(-0.5 * ((0.5 * s**2 - np.log(K)) / s) ** 2) / np.sqrt(2 * np.pi) * np.sqrt(tau_years)
    return vol, np.where(is_call, p["call_se"], p["put_se"]) / vega, price


def atm_approx(pre_k, win_k, tau_k, pre_d, win_d, tau_d):
    """Near-the-money factor from variances alone: averaging over a window keeps a third of the window's variance
    (exact for Brownian motion), so each vol is sqrt((pre-window variance + window variance / 3) / maturity)."""
    return np.sqrt(((pre_k + win_k / 3) / tau_k) / ((pre_d + win_d / 3) / tau_d))


def factor_grid(profile: pd.Series, kalshi_expiry: pd.Timestamp, kalshi_hours, sd, deribit_expiry=None,
                deribit_minutes: float = DERIBIT_WINDOW_MIN, kalshi_minutes: float = KALSHI_WINDOW_MIN,
                n_paths: int = 20_000, seed: int = 0, step_s: int = 1) -> pd.DataFrame:
    """R = Kalshi vol / Deribit vol at each of Kalshi's hours to expiry (`kalshi_hours`) and strike (`sd`: standard
    deviations from the money, ln K / (Kalshi's point-settled vol x sqrt(time left)), forward 1).

    `kalshi_expiry` is a Kalshi expiry in UTC (only its hour of the week matters); Deribit's is 08:00 UTC that day
    unless given. Both windows are sampled every `step_s` seconds and reuse the same random draws, so identical
    windows give identical vols. Hours that start inside either settlement window are skipped. One row per hour and
    strike: `kalshi_hours`, `deribit_hours`, `sd`, `strike`, `kalshi_iv`, `deribit_iv`, their standard errors
    `kalshi_se`, `deribit_se`, the option prices `kalshi_price`, `deribit_price`, `factor`, `factor_se`, `atm_approx`
    (the variance formula) and `reliable` (both prices at least MIN_PRICE, both vols found, factor_se <= MAX_FACTOR_SE).
    """
    if deribit_expiry is None:
        deribit_expiry = kalshi_expiry.normalize() + pd.Timedelta(hours=8)
    win_k = kalshi_expiry - pd.Timedelta(minutes=kalshi_minutes)
    win_d = deribit_expiry - pd.Timedelta(minutes=deribit_minutes)
    steps_k = window_steps(kalshi_expiry, kalshi_minutes, profile, step_s)
    steps_d = window_steps(deribit_expiry, deribit_minutes, profile, step_s)
    B_k, B_d = window_average(steps_k, n_paths, seed), window_average(steps_d, n_paths, seed)

    h = np.asarray(kalshi_hours, dtype=float)
    t0 = kalshi_expiry - pd.to_timedelta(h, unit="h")
    keep = np.asarray(t0 <= min(win_k, win_d))
    h, t0 = h[keep], t0[keep]
    cum = lambda ts: deribit._cumulative_variance(ts, profile)
    pre_k, pre_d = cum([win_k])[0] - cum(t0), cum([win_d])[0] - cum(t0)
    tau_k = h / YEAR_HOURS
    tau_d = np.asarray((deribit_expiry - t0) / pd.Timedelta(hours=1), dtype=float) / YEAR_HOURS
    v_k, v_d = steps_k.sum(), steps_d.sum()
    point_vol = np.sqrt((pre_k + v_k) / tau_k)
    sd = np.asarray(sd, dtype=float)

    rows = []
    for i in range(len(h)):
        K = np.exp(sd * point_vol[i] * np.sqrt(tau_k[i]))
        vk, sek, pk = settlement_vols(pre_k[i], B_k, K, tau_k[i])
        vd, sed, pdr = settlement_vols(pre_d[i], B_d, K, tau_d[i])
        rows.append(pd.DataFrame({"kalshi_hours": h[i], "deribit_hours": tau_d[i] * YEAR_HOURS, "sd": sd, "strike": K,
                                  "kalshi_iv": vk, "deribit_iv": vd, "kalshi_se": sek, "deribit_se": sed,
                                  "kalshi_price": pk, "deribit_price": pdr}))
    g = pd.concat(rows, ignore_index=True)
    g["factor"] = g.kalshi_iv / g.deribit_iv
    g["factor_se"] = g.factor * np.hypot(g.kalshi_se / g.kalshi_iv, g.deribit_se / g.deribit_iv)
    g["atm_approx"] = np.repeat(atm_approx(pre_k, v_k, tau_k, pre_d, v_d, tau_d), len(sd))
    g["reliable"] = (g.factor.notna() & (g.factor_se <= MAX_FACTOR_SE)
                     & (g[["kalshi_price", "deribit_price"]].min(axis=1) >= MIN_PRICE))
    return g


def factor_at(grid: pd.DataFrame, kalshi_hours, sd) -> np.ndarray:
    """The factor at each (Kalshi hours to expiry, sd), linear in both between the grid's points; NaN outside the
    grid or next to an unreliable cell."""
    f = grid.assign(factor=grid.factor.where(grid.reliable)).pivot(index="kalshi_hours", columns="sd", values="factor")
    interp = RegularGridInterpolator((f.index.to_numpy(float), f.columns.to_numpy(float)), f.to_numpy(float),
                                     bounds_error=False, fill_value=np.nan)
    return interp(np.column_stack([np.asarray(kalshi_hours, float), np.asarray(sd, float)]))


SD_GRID = np.round(np.arange(-4, 4.01, 0.25), 2)    # strikes for `factor_for`: wider than slide 6's +/-3 SD


def expiry_hours(kalshi_expiry: pd.Timestamp, deribit_expiry=None, deribit_minutes: float = DERIBIT_WINDOW_MIN,
                 last: float = 170) -> np.ndarray:
    """Kalshi hours to expiry for a factor grid: the start of Deribit's window, then every whole hour up to `last`.
    The profile is constant within each clock hour, so the factor is smooth between whole hours."""
    if deribit_expiry is None:
        deribit_expiry = kalshi_expiry.normalize() + pd.Timedelta(hours=8)
    first = (kalshi_expiry - deribit_expiry) / pd.Timedelta(hours=1) + deribit_minutes / 60
    return np.r_[first, np.arange(np.ceil(first + 1e-9), max(last, np.ceil(first + 1e-9)) + 1)]


def factor_for(d: pd.DataFrame, profile: pd.Series, sd_grid=SD_GRID, n_paths: int = 20_000) -> np.ndarray:
    """The factor for each row of `d` (columns `kalshi_expiry`, `kalshi_hours`, `sd`), from one `factor_grid` per
    Kalshi expiry weekday and hour (the profile repeats every week, so that is all a grid depends on), read with
    `factor_at`. NaN where there is no reliable factor, e.g. for a snapshot inside Deribit's settlement window."""
    e = pd.to_datetime(d.kalshi_expiry).reset_index(drop=True)
    hours, sd = d.kalshi_hours.to_numpy(float), d.sd.to_numpy(float)
    out = np.full(len(d), np.nan)
    for _, pos in pd.Series(np.arange(len(d))).groupby([e.dt.dayofweek, e.dt.hour]):
        expiry = e.iloc[pos.iloc[0]]
        grid = factor_grid(profile, expiry, expiry_hours(expiry, last=np.ceil(hours[pos].max())), sd_grid,
                           n_paths=n_paths)
        out[pos] = factor_at(grid, hours[pos], sd[pos])
    return out
