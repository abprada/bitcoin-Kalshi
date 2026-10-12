import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr
from scipy.stats import norm

from lib import compare

F, TAU = 60_000.0, 4 / 365


def n_d2(K, sigma):
    d2 = (np.log(F / K) - 0.5 * sigma**2 * TAU) / (sigma * np.sqrt(TAU))
    return ndtr(d2)


def test_flat_smile_digital_is_n_d2():
    k = np.log(np.array([50_000.0, 70_000.0]) / F)
    K = np.array([57_000.0, 60_000.0, 63_000.0])
    # a $500-wide spread is a slightly smoothed binary: off by under 0.05 cents here
    np.testing.assert_allclose(compare.digital_from_smile(K, F, TAU, k, [0.5, 0.5]), n_d2(K, 0.5), atol=5e-4)


def test_skewed_smile_adds_the_skew_term():
    # vol falls with the strike: sigma(K) = 0.5 - 0.4 ln(K / F). Then D = N(d2) - vega * dsigma/dK, larger than N(d2)
    grid = np.linspace(np.log(0.8), np.log(1.2), 401)
    K, slope = 62_000.0, -0.4
    sigma = 0.5 + slope * np.log(K / F)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * TAU) / (sigma * np.sqrt(TAU))
    vega = F * norm.pdf(d1) * np.sqrt(TAU)
    expected = n_d2(K, sigma) - vega * slope / K
    got = compare.digital_from_smile(np.array([K]), F, TAU, grid, 0.5 + slope * grid)[0]
    assert got == pytest.approx(expected, abs=5e-4)
    assert got > n_d2(K, sigma)


def test_no_extrapolation_beyond_traded_strikes():
    k = np.log(np.array([58_000.0, 62_000.0]) / F)
    assert np.isnan(compare.interp_vol(np.log(65_000.0 / F), k, [0.5, 0.5]))
    assert np.isnan(compare.digital_from_smile(np.array([61_900.0]), F, TAU, k, [0.5, 0.5])[0])   # K + 250 is outside


def test_smile_from_trades_takes_medians_per_strike():
    t = pd.DataFrame({"strike": [61_000.0, 61_000.0, 61_000.0, 59_000.0],
                      "log_moneyness": [0.010, 0.012, 0.014, -0.02], "iv_mark": [0.40, 0.42, 0.50, 0.45]})
    s = compare.smile_from_trades(t)
    assert s.strike.tolist() == [59_000.0, 61_000.0]
    assert s.vol.tolist() == pytest.approx([0.45, 0.42])
    assert s.n_trades.tolist() == [1, 3]


def test_kalshi_fee_rounds_up_to_the_cent():
    # 0.07 x 0.5 x 0.5 = 1.75 cents -> 2 cents; 0.07 x 0.1 x 0.9 = 0.63 cents -> 1 cent
    np.testing.assert_allclose(compare.kalshi_fee([0.5, 0.1]), [0.02, 0.01])


def test_nearest_marks_takes_the_closest_call_per_strike():
    ts = pd.Timestamp("2026-06-29 20:30", tz="UTC")
    t = pd.DataFrame({"strike": [61_000.0, 61_000.0, 62_000.0, 61_000.0],
                      "cp": ["C", "C", "C", "P"],
                      "ts": [ts - pd.Timedelta(minutes=10), ts + pd.Timedelta(minutes=2), ts, ts],
                      "mark_price_btc": [0.010, 0.012, 0.005, 0.5], "index_price": [60_000.0] * 4})
    m = compare.nearest_marks(t, ts)
    assert m.loc[61_000.0, "mark_usd"] == pytest.approx(720.0)    # the call 2 minutes away, not the put
    assert m.loc[62_000.0, "mark_usd"] == pytest.approx(300.0)


def test_direct_digital_is_the_call_spread_per_dollar():
    marks = pd.DataFrame({"mark_usd": [1_300.0, 770.0, 550.0]}, index=[60_000.0, 61_000.0, 61_500.0])
    got = compare.direct_digital([60_500.0, 61_000.0], marks)
    assert got[0] == pytest.approx((1_300.0 - 770.0) / 1_000)      # 60,000 and 61,000 both have marks
    assert np.isnan(got[1])                                        # 61,500 has a mark but 60,500 does not


def test_usable_snapshots_ignore_the_forward_basis():
    s = pd.DataFrame({"atm_iv": [0.45, 0.45, np.nan, 0.45],
                      "strikes_per_sd": [3.0, 3.0, 3.0, 1.5],
                      "forward_basis_bps": [5.0, 80.0, 5.0, 5.0]})
    # a large basis is kept; a missing ATM vol or too coarse a strike grid is not
    assert compare.usable_kalshi_snapshots(s).tolist() == [True, True, False, False]
