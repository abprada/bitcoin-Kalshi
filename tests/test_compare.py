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
