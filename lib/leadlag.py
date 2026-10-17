"""Lead-lag of Kalshi's and Deribit's ATM implied vol at 10-minute resolution, conditioned on how far apart they are
(used by 17_lead_lag.py).

1. Kalshi: the weekly's smile every 10 minutes from 02's clean 1-minute quotes, built as in 03 (`pairs.kalshi_smiles`),
   kept when usable as in 05 (`compare.usable_kalshi_snapshots`). Its ATM vol is the at-the-forward vol.
2. Deribit: the Friday 08:00 expiry with the Kalshi weekly's date; its ATM vol from the trades in the 10 minutes
   *before* each time (so Deribit never sees the future): each trade's mark vol with the smile's shape taken out
   (`deribit_atm`), stretched to Kalshi's 5pm ET expiry with the hour-of-week profile as everywhere else.
3. Changes: dK(t) = K(t) - K(t - h), dD(t) likewise, within one weekly, over h = 10, 30 or 60 minutes (sampled every
   10 minutes; 10-minute changes are mostly measurement noise, so longer h shows the co-movement more clearly).
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


def _corr_from_sums(s: np.ndarray) -> np.ndarray:
    """Pearson correlation from summed moments [n, Σa, Σb, Σa², Σb², Σab] along the last axis."""
    n, a, b, aa, bb, ab = np.moveaxis(s, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ab - a * b / n) / np.sqrt((aa - a * a / n) * (bb - b * b / n))


def conditional_xcorr(p: pd.DataFrame, lags=range(-60, 61, 10), xs=range(-6, 7), normal: float = NORMAL_GAP,
                      change_min: int = STEP_MIN, gap_lead_min: int | None = None, n_boot: int = 300,
                      seed: int = 0) -> pd.DataFrame:
    """corr(dK(t), dD(t + lag)) for each lag (minutes) and threshold X, where dK(t) = K(t) - K(t - change_min) and
    likewise dD, over times t whose gap `gap_lead_min` minutes before t is above X + `normal`. Also the unconditional
    row (X = None). Columns: x, threshold, lag, corr, n, ci_low, ci_high, mean_gap.

    Changes over `change_min` minutes are taken every STEP_MIN minutes, so with change_min > STEP_MIN neighbouring
    observations overlap; the 95% interval therefore resamples whole weeklies (`n_boot` times), not observations.

    `gap_lead_min` defaults to the earliest time any change in the window starts, one step before it: max(-min(lags),
    0) + change_min + STEP_MIN. Measured later (e.g. at the start of Kalshi's change) the gap shares a time stamp with
    the end of one of Deribit's changes, and selecting on a difference of two noisy readings correlates their noise: a
    spurious negative correlation at that lag."""
    lags = list(lags)
    if gap_lead_min is None:
        gap_lead_min = max(-min(lags), 0) + change_min + STEP_MIN
    h = pd.Timedelta(minutes=change_min)
    idx = p.set_index(["event_ticker", "ts"])
    K, D, G = idx.K, idx.D, idx.gap
    ev, ts = p.event_ticker.to_numpy(), p.ts
    dK = p.K.to_numpy() - _at(K, ev, ts - h)
    gap0 = _at(G, ev, ts - pd.Timedelta(minutes=gap_lead_min))
    dD = {L: _at(D, ev, ts + pd.Timedelta(minutes=L)) - _at(D, ev, ts + pd.Timedelta(minutes=L) - h) for L in lags}
    codes, events = pd.factorize(ev)
    rng = np.random.default_rng(seed)
    boot = np.stack([np.bincount(rng.integers(0, len(events), len(events)), minlength=len(events))
                     for _ in range(n_boot)])                                     # times each weekly is drawn
    rows = []
    for x in [None, *xs]:
        sel = np.ones(len(p), bool) if x is None else (gap0 > x + normal)
        for L in lags:
            ok = sel & np.isfinite(dK) & np.isfinite(dD[L])
            a, b, c = dK[ok], dD[L][ok], codes[ok]
            per = np.stack([np.bincount(c, w, len(events)) for w in (np.ones_like(a), a, b, a * a, b * b, a * b)], -1)
            r = float(_corr_from_sums(per.sum(0))) if ok.sum() >= 3 else np.nan
            rb = _corr_from_sums(boot @ per) if ok.sum() >= 3 else np.full(n_boot, np.nan)
            rows.append({"x": x, "threshold": None if x is None else x + normal, "lag": L, "corr": r, "n": int(ok.sum()),
                         "ci_low": float(np.nanpercentile(rb, 2.5)) if ok.sum() >= 3 else np.nan,
                         "ci_high": float(np.nanpercentile(rb, 97.5)) if ok.sum() >= 3 else np.nan,
                         "mean_gap": float(np.nanmean(gap0[ok])) if ok.any() else np.nan})
    return pd.DataFrame(rows)


def after_dislocation(p: pd.DataFrame, xs=(0, 2, 4, 6), normal: float = NORMAL_GAP, delay_min: int = 30,
                      horizons=range(10, 241, 10), bucket_hours: float = 12, n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """Event study: when the gap is above X + `normal` at time s, how do Kalshi's ATM vol, Deribit's and the gap move
    from s + `delay_min` (when a trader could act) to s + delay_min + h?

    Each change is taken relative to the average change over the same h from every time with the same time to
    expiry (`bucket_hours` buckets), so the gap's normal widening into expiry is not counted. The reading that
    selects the dislocation (at s) is not part of the change, and the delay lets short-lived measurement noise fade,
    so noise cannot produce the reversion by itself. Intervals: 95%, resampling whole weeklies.
    Returns x, threshold, series (Kalshi / Deribit / gap), minutes (h), change (vol points), lo, hi, n.
    """
    d = pd.Timedelta(minutes=delay_min)
    idx = p.set_index(["event_ticker", "ts"])
    ev, ts = p.event_ticker.to_numpy(), p.ts
    start = ts + d
    gap_sel = p.gap.to_numpy()                                      # the dislocation, seen at s = ts
    tau_h = _at(idx.tau_years, ev, start) * 8760
    bucket = np.floor(tau_h / bucket_hours)
    codes, events = pd.factorize(ev)
    rng = np.random.default_rng(seed)
    boot = np.stack([np.bincount(rng.integers(0, len(events), len(events)), minlength=len(events)) for _ in range(n_boot)])
    rows = []
    for h in horizons:
        end = start + pd.Timedelta(minutes=h)
        for name, col in (("Kalshi", "K"), ("Deribit", "D"), ("gap", "gap")):
            s = idx[col]
            raw = _at(s, ev, end) - _at(s, ev, start)
            ok_all = np.isfinite(raw) & np.isfinite(bucket)
            base = pd.Series(raw[ok_all]).groupby(bucket[ok_all]).mean()
            excess = raw - pd.Series(bucket).map(base).to_numpy()
            for x in xs:
                ok = (gap_sel > x + normal) & np.isfinite(excess)
                s1 = np.bincount(codes[ok], excess[ok], len(events))
                s0 = np.bincount(codes[ok], minlength=len(events)).astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    bs = (boot @ s1) / (boot @ s0)
                rows.append({"x": x, "threshold": x + normal, "series": name, "minutes": h,
                             "change": s1.sum() / s0.sum() if s0.sum() else np.nan,
                             "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5)),
                             "n": int(ok.sum())})
    return pd.DataFrame(rows)
