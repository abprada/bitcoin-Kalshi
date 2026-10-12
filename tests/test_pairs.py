import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from lib import pairs

T0 = pd.Timestamp("2026-06-29 21:45", tz="UTC")


def test_fills_with_the_same_nearest_minute_are_one_observation():
    ts = [T0 - pd.Timedelta(seconds=9), T0 - pd.Timedelta(seconds=9), T0 + pd.Timedelta(seconds=20),
          T0 + pd.Timedelta(seconds=40)]
    t = pd.DataFrame({"expiry": pd.Timestamp("2026-07-03 08:00", tz="UTC"), "strike": 58_500.0, "cp": ["P", "P", "C", "P"],
                      "ts": ts, "iv_mark": [0.48, 0.50, 0.52, 0.60], "index_price": 60_000.0, "tau_years": 0.01,
                      "amount": [15.0, 15.0, 1.0, 2.0]})
    o = pairs.deribit_observations(t).set_index("minute")
    assert o.index.tolist() == [T0, T0 + pd.Timedelta(minutes=1)]   # 21:45:40 is nearer 21:46
    first = o.loc[T0]
    assert (first.n_trades, first.n_calls, first.amount) == (3, 1, 31.0)
    assert first.deribit_iv == pytest.approx(0.50)
    assert first.max_offset_s == pytest.approx(20.0)


def test_iv_at_strikes_reads_the_ladder_and_never_extrapolates():
    K = np.array([58_000.0, 58_500.0, 59_000.0, 59_500.0])
    vol = np.array([0.55, np.nan, 0.52, 0.51])
    got = pairs.iv_at_strikes([58_000.0, 58_500.0, 59_250.0, 60_000.0], K, vol)
    assert got[0] == pytest.approx(0.55)                                  # on the ladder: its own vol
    w = np.log(58_500 / 58_000) / np.log(59_000 / 58_000)                 # 58,500 has no vol: between its neighbours
    assert got[1] == pytest.approx(0.55 + w * (0.52 - 0.55))
    assert 0.51 < got[2] < 0.52
    assert np.isnan(got[3])                                               # beyond the ladder


def kalshi_minute(ts, F=60_000.0, sig=0.45, hours=72.0, step=500.0):
    K = np.arange(54_000.0, 66_001.0, step)
    tau = hours / 8760
    d2 = (np.log(F / K) - 0.5 * sig**2 * tau) / (sig * np.sqrt(tau))
    mid = np.round(ndtr(d2), 4)
    return pd.DataFrame({"event_ticker": "KXBTCD-TEST", "strike": K, "ts": ts, "tau_years": tau, "bid": mid - 0.005,
                         "ask": mid + 0.005, "mid": mid, "spread": 0.01, "quote_age_min": 0, "spot": F,
                         "exchange_open": True, "usable": True})


def test_kalshi_smiles_recover_a_flat_smile_and_skip_thin_ladders():
    thin = kalshi_minute(T0 + pd.Timedelta(minutes=1))
    thin = thin[thin.strike.between(60_500.0, 62_000.0)]                  # does not span spot
    q = pd.concat([kalshi_minute(T0), thin])
    out = pairs.kalshi_smiles(q, [T0, T0 + pd.Timedelta(minutes=1)])
    assert list(out) == [T0]
    snap, strikes = out[T0]
    assert snap["atm_iv"] == pytest.approx(0.45, abs=0.01)
    near = strikes[strikes.strike.between(57_000.0, 63_000.0)]
    np.testing.assert_allclose(near.iv, 0.45, atol=0.01)


def test_gap_table_counts_days_and_brackets_the_median():
    p = pd.DataFrame({"sd_bucket": [0.0] * 6 + [1.0] * 2, "gap_adj": [1.0, 2.0, 3.0, 1.5, 2.5, 2.0, -1.0, -2.0],
                      "day": ["a", "a", "b", "b", "c", "c", "a", "b"]})
    t = pairs.gap_table(p, "sd_bucket", n_boot=200).set_index("sd_bucket")
    assert (t.loc[0.0, "pairs"], t.loc[0.0, "days"]) == (6, 3)
    assert t.loc[0.0, "median_gap"] == pytest.approx(2.0)
    assert t.loc[0.0, "ci_low"] <= 2.0 <= t.loc[0.0, "ci_high"]
    assert t.loc[1.0, "kalshi_higher"] == 0.0


def test_atm_is_within_a_quarter_standard_deviation():
    p = pd.DataFrame({"sd": [-0.3, -0.25, 0.0, 0.25, 0.3]})
    assert pairs.atm(p).sd.tolist() == [-0.25, 0.0, 0.25]


def test_far_from_expiry_keeps_four_days_or_more():
    p = pd.DataFrame({"kalshi_hours": [30.0, 95.9, 96.0, 160.0]})
    assert pairs.far_from_expiry(p).kalshi_hours.tolist() == [96.0, 160.0]


def test_gap_with_a_flat_profile_is_the_unstretched_gap():
    # if every hour of the week carries the same variance, the 13 extra hours change nothing
    p = pd.DataFrame({"minute": [T0], "kalshi_expiry": [pd.Timestamp("2026-07-03 21:00", tz="UTC")],
                      "deribit_expiry": [pd.Timestamp("2026-07-03 08:00", tz="UTC")],
                      "kalshi_iv": [0.53], "deribit_iv": [0.49]})
    assert pairs.gap_with_profile(p, pd.Series(np.ones(168))).iloc[0] == pytest.approx(4.0)
