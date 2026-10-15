"""Lead-lag of Kalshi's and Deribit's ATM implied vol at 10-minute resolution, conditioned on how far apart they are
(used by 17_lead_lag.py).

1. Kalshi: the weekly's smile every 10 minutes from 02's clean 1-minute quotes, built as in 03 (`pairs.kalshi_smiles`),
   kept when usable as in 05 (`compare.usable_kalshi_snapshots`). Its ATM vol is the at-the-forward vol.
2. Deribit: the Friday 08:00 expiry with the Kalshi weekly's date; its ATM vol from the trades in the 10 minutes
   *before* each time (so Deribit never sees the future): each trade's mark vol with the smile's shape taken out
   (`deribit_atm`), stretched to Kalshi's 5pm ET expiry with the hour-of-week profile as everywhere else.
3. Changes: dK(t) = K(t) - K(t - 10 min), dD(t) likewise, within one weekly, only between consecutive grid times.
4. Lead-lag: corr(dK(t), dD(t + k)) for k = -60 ... +60 minutes. k > 0: Deribit's change comes after Kalshi's, so a
   peak there means Kalshi moves first; k < 0: Deribit moves first.
5. Conditioning: only times t whose gap (Kalshi - Deribit, vol points) 80 minutes before t, before every change in
   the window, is above X + NORMAL_GAP, so selecting on it cannot select on the changes being correlated.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lib import compare, deribit
from lib import deribit_marks as dm
from lib import pairs

STEP_MIN = 10
NORMAL_GAP = 1.8          # Kalshi's usual premium over Deribit (vol points), as in the deck
MIN_N = 100               # cells with fewer observations are left empty


def kalshi_atm(root: Path, step_min: int = STEP_MIN) -> pd.DataFrame:
    """Kalshi's ATM vol every `step_min` minutes for every weekly: event_ticker, ts, kalshi_expiry, atm_iv, tau_years."""
    rows = []
    for f in sorted((Path(root) / "data" / "clean" / "kalshi_weekly_quotes").glob("*.parquet")):
        q = pd.read_parquet(f, columns=pairs.KALSHI_COLS)
        expiry = (q.ts + pd.to_timedelta(q.tau_years.astype(float) * deribit.YEAR_SECONDS, unit="s")).median().round("h")
        grid = q.ts[q.ts.dt.floor(f"{step_min}min") == q.ts].unique()
        for m, (snap, _) in pairs.kalshi_smiles(q, grid).items():
            rows.append({"event_ticker": q.event_ticker.iloc[0], "ts": m, "kalshi_expiry": expiry,
                         "atm_iv": snap["atm_iv"], "strikes_per_sd": snap["strikes_per_sd"], "tau_years": snap["tau_years"]})
    k = pd.DataFrame(rows)
    return k[compare.usable_kalshi_snapshots(k)].drop(columns="strikes_per_sd").reset_index(drop=True)


SHAPE_WINDOW_MIN = 60     # the smile's skew and curvature come from the previous hour of trades
MAX_SD = 2.0              # trades further from the money than this (ATM standard deviations) are not used


def deribit_atm(t: pd.DataFrame, step_min: int = STEP_MIN) -> pd.DataFrame:
    """Deribit's ATM vol per expiry every `step_min` minutes from the trades in the `step_min` minutes before.

    Every trade carries Deribit's mark vol for its strike (from Deribit's own surface, not the traded price), so each
    one is an ATM estimate once the smile's shape is taken out: atm = iv - b m - c m^2, with b and c from the smile
    fitted to the previous SHAPE_WINDOW_MIN minutes (`smile_grid`, backward window) and m = ln(K / F) / sqrt(tau)
    clipped to the fitted range. The ATM vol is the median of those estimates. A 3-strike fit on 10 minutes of trades
    alone exists only about half the time; this uses any trade, so timing stays at `step_min` minutes.
    Returns expiry, t (window end), deribit_atm (decimal), n_trades.
    """
    shape = dm.smile_grid(t, freq=f"{step_min}min", half_window=f"{SHAPE_WINDOW_MIN / 2:g}min", centred=False)
    x = t.assign(t=t.ts.dt.ceil(f"{step_min}min")).merge(
        shape[["expiry", "t", "atm_iv", "b", "c", "m_lo", "m_hi"]].rename(columns={"atm_iv": "fit_atm"}), on=["expiry", "t"])
    m = x.m.clip(x.m_lo, x.m_hi)
    x["atm_est"] = x.iv_d - x.b * m - x.c * m**2
    x = x[(x.m / x.fit_atm).abs() <= MAX_SD]
    return (x.groupby(["expiry", "t"]).agg(deribit_atm=("atm_est", "median"), n_trades=("atm_est", "size"))
            .reset_index())


def panel(root: Path, hour_profile: pd.Series, step_min: int = STEP_MIN) -> pd.DataFrame:
    """Kalshi and (stretched) Deribit ATM vol on one `step_min` grid, in vol points, with their gap."""
    k = kalshi_atm(root, step_min)
    k["deribit_expiry"] = k.kalshi_expiry.dt.normalize() + pd.Timedelta(hours=8)
    t = dm.read_full_trades(root, start=k.ts.min() - pd.Timedelta(days=1), end=k.ts.max(), max_tte_days=8)
    t = dm.add_forward(t[t.expiry.isin(set(k.deribit_expiry))])
    d = deribit_atm(t, step_min)
    p = k.merge(d.rename(columns={"expiry": "deribit_expiry", "t": "ts"}), on=["deribit_expiry", "ts"], how="left")
    live = p.ts < p.deribit_expiry
    p["ratio"] = np.nan
    p.loc[live, "ratio"] = deribit.horizon_vol_ratios(p.ts[live], p.kalshi_expiry[live], p.deribit_expiry[live], hour_profile)
    p["K"] = 100 * p.atm_iv
    p["D"] = 100 * p.deribit_atm * p.ratio
    p["gap"] = p.K - p.D
    return p.sort_values(["event_ticker", "ts"]).reset_index(drop=True)


def cached_panel(root: Path, hour_profile: pd.Series) -> pd.DataFrame:
    """`panel`, saved to data/derived/leadlag10_panel.parquet on first use (delete it to rebuild)."""
    f = Path(root) / "data" / "derived" / f"leadlag{STEP_MIN}_panel.parquet"
    if f.exists():
        return pd.read_parquet(f)
    p = panel(root, hour_profile)
    p.to_parquet(f)
    return p


def _at(s: pd.Series, events, ts) -> np.ndarray:
    """Values of a (event_ticker, ts)-indexed series at the given keys (NaN where missing)."""
    return s.reindex(pd.MultiIndex.from_arrays([events, ts])).to_numpy()


def conditional_xcorr(p: pd.DataFrame, lags=range(-60, 61, 10), xs=range(-6, 7), normal: float = NORMAL_GAP,
                      step_min: int = STEP_MIN, gap_lead_min: int = 60 + 2 * STEP_MIN) -> pd.DataFrame:
    """corr(dK(t), dD(t + lag)) for each lag (minutes) and threshold X, over times t whose gap `gap_lead_min` minutes
    before t is above X + `normal`. Also the unconditional row (X = None). Columns: x, threshold, lag, corr, n,
    ci_low, ci_high (Fisher-z 95% interval treating observations as independent, so it is too narrow if changes are
    serially correlated), plus the mean gap of the selected times.

    Default: the gap 80 minutes before t, before every change in the window. Measured later (e.g. at the start of
    Kalshi's change, gap_lead_min = step) it shares a time stamp with the end of Deribit's
    change at lag -step, and selecting on a difference of two noisy readings correlates their noise: a spurious
    negative correlation at that lag."""
    step = pd.Timedelta(minutes=step_min)
    idx = p.set_index(["event_ticker", "ts"])
    K, D, G = idx.K, idx.D, idx.gap
    ev, ts = p.event_ticker.to_numpy(), p.ts
    dK = p.K.to_numpy() - _at(K, ev, ts - step)
    gap0 = _at(G, ev, ts - pd.Timedelta(minutes=gap_lead_min))
    dD = {L: _at(D, ev, ts + pd.Timedelta(minutes=L)) - _at(D, ev, ts + pd.Timedelta(minutes=L) - step) for L in lags}
    rows = []
    for x in [None, *xs]:
        sel = np.ones(len(p), bool) if x is None else (gap0 > x + normal)
        for L in lags:
            ok = sel & np.isfinite(dK) & np.isfinite(dD[L])
            n = int(ok.sum())
            r = float(np.corrcoef(dK[ok], dD[L][ok])[0, 1]) if n >= 3 else np.nan
            z, se = np.arctanh(np.clip(r, -0.999999, 0.999999)), 1 / np.sqrt(max(n - 3, 1))
            rows.append({"x": x, "threshold": None if x is None else x + normal, "lag": L, "corr": r, "n": n,
                         "ci_low": float(np.tanh(z - 1.96 * se)), "ci_high": float(np.tanh(z + 1.96 * se)),
                         "mean_gap": float(np.nanmean(gap0[ok])) if n else np.nan})
    return pd.DataFrame(rows)
