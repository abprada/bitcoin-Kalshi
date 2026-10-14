import numpy as np
import pandas as pd

from lib import term


def test_realized_vol_recovers_the_vol_of_a_simulated_path():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2026-03-02", periods=7 * 1440, freq="1min", tz="UTC")
    sigma = 0.6
    r = rng.normal(0, sigma * np.sqrt(60 / term.YEAR_SECONDS), len(ts))
    cum = term.cumulative_variance(pd.Series(80_000 * np.exp(np.cumsum(r)), index=ts))
    rv = term.realized_vol(cum, [ts[0]], [ts[-1]])
    assert abs(rv[0] - sigma) < 0.02


def test_realized_vol_is_nan_outside_the_data():
    ts = pd.date_range("2026-03-02", periods=100, freq="1min", tz="UTC")
    cum = term.cumulative_variance(pd.Series(np.linspace(80_000, 80_100, 100), index=ts))
    assert np.isnan(term.realized_vol(cum, [ts[0]], [ts[-1] + pd.Timedelta(days=1)])[0])


def test_implied_vs_realized_uses_root_mean_square_vols():
    t = pd.Timestamp("2026-03-02", tz="UTC")
    d = pd.DataFrame({"market": "m", "h": "1–2d", "expiry": [t, t, t + pd.Timedelta(days=7)],
                      "atm_iv": [0.3, 0.4, 0.5], "rv": [0.2, 0.2, 0.2]})
    r = term.implied_vs_realized(d, n_boot=10).iloc[0]
    assert np.isclose(r.implied, np.sqrt((0.09 + 0.16 + 0.25) / 3))
    assert np.isclose(r.realized, 0.2)
    assert r.expiries == 2 and r.implied_higher == 1.0


def test_kalshi_term_spread_uses_the_nearest_daily():
    t = pd.Timestamp("2026-03-03 20:15", tz="UTC")
    daily = pd.DataFrame({"ts": [t, t], "hours": [0.75, 24.75], "atm_iv": [0.9, 0.5]})
    weekly = pd.DataFrame({"ts": [t], "hours": [72.75], "atm_iv": [0.4]})
    s = term.kalshi_term_spread(daily, weekly)
    assert len(s) == 1 and np.isclose(s.spread.iloc[0], 50.0)


def test_deribit_spread_fee_is_capped_by_cheap_options():
    F, tau = 80_000.0, 1 / 365
    near = term.deribit_spread_fee(F, F, 0.5, tau)
    far = term.deribit_spread_fee(F, 1.3 * F, 0.5, tau)
    assert np.isclose(near, 0.0003 * F / 500)       # cap does not bind; combo: only one leg's fee, 4.8c per $1 binary
    assert far < 0.001                             # both legs nearly worthless: the 12.5% cap leaves almost nothing
