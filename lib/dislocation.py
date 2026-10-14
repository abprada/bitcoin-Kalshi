"""Kalshi vs Deribit dislocations, and which market predicts BTC better (11_dislocations.py).

A *dislocation* is the Kalshi - Deribit ATM vol gap minus its recent normal: the median gap over the previous 30
days for the same kind of contract (daily or weekly) and the same stretch of time to expiry. Kalshi's premium over
Deribit has a shape (it grows toward expiry) and a trend (it shrank through 2026), so only the departure from that
normal is a candidate mispricing. Everything is measured with Deribit's backward-looking hourly smile, so a signal at
hour t uses only what was known at t.

Prediction tests compare the two markets as forecasts of BTC, all at Kalshi's own expiry:
- probabilities: Kalshi's binary mids against Deribit's smile turned into the same binaries, scored on how each
  contract settled (Brier score, log score);
- direction: each market's probability that BTC ends above today's price, and its forward, against what BTC did;
- size of moves: each market's ATM vol against the vol BTC realized (as in 09).
"""

import numpy as np
import pandas as pd
from scipy.special import ndtr

H_EDGES = [0, 6, 12, 24, 48, 96, 170]
NORMAL_DAYS = 30


def add_dislocation(panel: pd.DataFrame, gap_col: str = "gap_sig", days: int = NORMAL_DAYS) -> pd.DataFrame:
    """Adds `normal_gap` (median of `gap_col` over the previous `days` calendar days, same kind and hours-to-expiry
    bucket, today excluded) and `disl` = gap - normal_gap. Rows without enough history (fewer than 20 past
    observations) get NaN."""
    p = panel.copy()
    p["hb"] = pd.cut(p.hours_left, H_EDGES).astype(str)
    p["day"] = p.ts.dt.normalize()
    out = []
    for (kind, hb), g in p.groupby(["kind", "hb"]):
        by_day = g.groupby("day")[gap_col].apply(list)
        days_idx = by_day.index
        normals, counts = [], []
        for d in days_idx:
            past = by_day[(days_idx < d) & (days_idx >= d - pd.Timedelta(days=days))]
            vals = np.concatenate(past.to_numpy()) if len(past) else np.array([])
            vals = vals[np.isfinite(vals)]
            normals.append(np.median(vals) if len(vals) >= 20 else np.nan)
            counts.append(len(vals))
        out.append(pd.DataFrame({"kind": kind, "hb": hb, "day": days_idx, "normal_gap": normals}))
    n = pd.concat(out, ignore_index=True)
    p = p.merge(n, on=["kind", "hb", "day"], how="left")
    p["disl"] = p[gap_col] - p.normal_gap
    return p


def forward_values(panel: pd.DataFrame, cols, hours) -> pd.DataFrame:
    """For each h in `hours`, the change of each column from t to t + h within the same event (NaN when missing)."""
    p = panel.copy()
    key = p[["event_ticker", "ts"]]
    for h in hours:
        nxt = p[["event_ticker", "ts", *cols]].copy()
        nxt["ts"] = nxt.ts - pd.Timedelta(hours=h)
        m = key.merge(nxt, on=["event_ticker", "ts"], how="left")
        for c in cols:
            p[f"{c}_f{h}"] = m[c].to_numpy() - p[c].to_numpy()
    return p


def digital_from_deribit(K, F, tau, atm, b, c, m_lo, m_hi):
    """Price of "$1 if BTC > K" at Kalshi's expiry from a Deribit smile: vol(m) = atm + b m + c m^2 with
    m = ln(K / F) / sqrt(tau), flat outside [m_lo, m_hi]. Digital = N(d2) - vega x dvol/dK (the skew term), so a
    steeper put wing makes upside binaries cheaper, as it should."""
    K = np.asarray(K, dtype=float)
    m_raw = np.log(K / F) / np.sqrt(tau)
    m = np.clip(m_raw, m_lo, m_hi)
    sig = atm + b * m + c * m**2
    sd = sig * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    d2 = d1 - sd
    inside = (m_raw >= m_lo) & (m_raw <= m_hi)
    dsig_dK = np.where(inside, (b + 2 * c * m) / (K * np.sqrt(tau)), 0.0)
    vega = F * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * np.sqrt(tau)
    return np.clip(ndtr(d2) - vega * dsig_dK, 0.0, 1.0)


def scores(p, y):
    """Brier score and log score (negative log likelihood) per forecast; probabilities clipped to [0.005, 0.995]."""
    p = np.clip(np.asarray(p, dtype=float), 0.005, 0.995)
    y = np.asarray(y, dtype=float)
    return (p - y) ** 2, -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_mean_diff(df: pd.DataFrame, col: str, n_boot: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """Mean of `col` with a 95% interval resampling whole days (`day` column)."""
    rng = np.random.default_rng(seed)
    per = df.groupby("day")[col].agg(["sum", "count"])
    s, n = per["sum"].to_numpy(), per["count"].to_numpy()
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    b = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
