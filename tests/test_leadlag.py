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
    assert loose.n == p.shape[0] - 8          # by default the gap is 80 min before t: the first 8 times have none
    assert strict.n == 0 and np.isnan(strict["corr"])
    late = leadlag.conditional_xcorr(p, lags=[0], xs=[-100], normal=0, gap_lead_min=10)
    assert late[late.x == -100].iloc[0].n == p.shape[0] - 1
