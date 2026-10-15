import numpy as np
import pandas as pd

from lib import leadlag


def synthetic_panel(lead_steps=2, n=4000, seed=0):
    """Kalshi's vol is a random walk; Deribit copies it `lead_steps` grid steps later, plus noise."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-03-02", periods=n, freq="10min", tz="UTC")
    K = 40 + np.cumsum(rng.normal(0, 0.5, n))
    D = np.r_[np.full(lead_steps, K[0]), K[:-lead_steps]] + rng.normal(0, 0.05, n)
    return pd.DataFrame({"event_ticker": "E", "ts": ts, "K": K, "D": D, "gap": K - D})


def test_peak_at_kalshi_lead():
    # Kalshi moves 20 minutes before Deribit: the correlation peaks at lag +20
    r = leadlag.conditional_xcorr(synthetic_panel(lead_steps=2), xs=[])
    r = r[r.x.isna()].set_index("lag")["corr"]
    assert r.idxmax() == 20
    assert r.loc[20] > 0.9 and abs(r.loc[-20]) < 0.1


def test_condition_filters_on_gap_before_the_change():
    p = synthetic_panel()
    r = leadlag.conditional_xcorr(p, lags=[0], xs=[-100, 100], normal=0)
    loose, strict = r[r.x == -100].iloc[0], r[r.x == 100].iloc[0]
    assert loose.n == p.shape[0] - 2          # lag 0, 10-min changes: gap 20 min before t, so the first 2 times have none
    assert strict.n == 0 and np.isnan(strict["corr"])
    late = leadlag.conditional_xcorr(p, lags=[0], xs=[-100], normal=0, gap_lead_min=10)
    assert late[late.x == -100].iloc[0].n == p.shape[0] - 1


def test_longer_changes_and_week_intervals():
    # 60-minute changes over lags -60..+60: the gap is taken 60 + 60 + 10 = 130 minutes before t
    p = pd.concat([synthetic_panel(seed=s).assign(event_ticker=f"E{s}") for s in range(6)], ignore_index=True)
    r = leadlag.conditional_xcorr(p, xs=[-100], normal=0, change_min=60)
    row = r[(r.x == -100) & (r.lag == 20)].iloc[0]
    assert row.n == 6 * (4000 - 13 - 2)       # per weekly: first 13 times have no gap 130 min earlier, last 2 no D 20 min later
    assert row.ci_low <= row["corr"] <= row.ci_high
    u = r[r.x.isna()].set_index("lag")["corr"]
    assert u.idxmax() == 20                   # Kalshi still leads by 20 minutes


def _gap_panel(gaps):
    """One weekly per gap path; Kalshi carries the gap, Deribit is flat at 40."""
    frames = []
    for e, g in enumerate(gaps):
        ts = pd.date_range("2026-03-02", periods=len(g), freq="10min", tz="UTC")
        frames.append(pd.DataFrame({"event_ticker": f"E{e}", "ts": ts, "tau_years": 7 / 365,
                                    "K": 40 + g, "D": 40.0, "gap": g}))
    return pd.concat(frames, ignore_index=True)


def test_after_dislocation_finds_mean_reversion_and_not_without_it():
    rng = np.random.default_rng(1)
    n = 3000
    ar = []
    for _ in range(8):                         # the gap mean-reverts (AR(1) around 0): it closes over time
        g = np.zeros(n)
        for i in range(1, n):
            g[i] = 0.97 * g[i - 1] + rng.normal(0, 1)
        ar.append(g)
    r = leadlag.after_dislocation(_gap_panel(ar), xs=[4], normal=0, delay_min=30, horizons=[120], n_boot=100)
    gap = r[r.series == "gap"].iloc[0]
    assert gap.change < -1 and gap.hi < 0
    assert abs(r[r.series == "Deribit"].iloc[0].change) < 1e-9
    walks = [rng.normal(0, 1, n).cumsum() for _ in range(8)]    # a random-walk gap does not close
    w = leadlag.after_dislocation(_gap_panel(walks), xs=[4], normal=0, delay_min=30, horizons=[120], n_boot=100)
    wg = w[w.series == "gap"].iloc[0]
    assert wg.lo < 0 < wg.hi
