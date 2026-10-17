import numpy as np
import pandas as pd

from lib import settlement

KALSHI_EXPIRY = pd.Timestamp("2026-07-10 21:00", tz="UTC")     # a Friday, 5pm EDT
DERIBIT_EXPIRY = pd.Timestamp("2026-07-10 08:00", tz="UTC")    # 13 hours earlier
SD = np.arange(-2.0, 2.01, 0.5)


def flat(sigma=0.6) -> pd.Series:
    """Hour-of-week profile with the same variance every hour: constant vol `sigma`."""
    return pd.Series(sigma**2 / 8760, index=range(168))


def test_window_average_has_the_forward_of_spot():
    steps = settlement.window_steps(DERIBIT_EXPIRY, 30, flat())
    assert len(steps) == 1800 and np.isclose(steps.sum(), 0.36 * 0.5 / 8760)
    B = settlement.window_average(steps, n_paths=20_000)
    half = len(B) // 2
    se = (0.5 * (B[:half] + B[half:])).std(ddof=1) / np.sqrt(half)
    assert abs(B.mean() - 1) < 4 * se


def test_no_window_means_no_averaging():
    assert len(settlement.window_steps(DERIBIT_EXPIRY, 0, flat())) == 0
    assert np.all(settlement.window_average([], n_paths=10) == 1)


def test_put_call_parity_on_the_settlement():
    B = settlement.window_average(settlement.window_steps(DERIBIT_EXPIRY, 30, flat()), n_paths=2_000)
    K = np.exp(np.linspace(-0.1, 0.1, 9))
    p = settlement.option_prices(0.36 * 24 / 8760, B, K)
    np.testing.assert_allclose(p["call"] - p["put"], 1 - K, atol=1e-12)


def test_point_settlement_gives_back_constant_vol_at_every_strike():
    g = settlement.factor_grid(flat(0.6), KALSHI_EXPIRY, [20, 37, 100], SD, deribit_minutes=0, kalshi_minutes=0,
                               n_paths=200)
    np.testing.assert_allclose(g.kalshi_iv, 0.6, atol=1e-8)
    np.testing.assert_allclose(g.deribit_iv, 0.6, atol=1e-8)
    np.testing.assert_allclose(g.factor, 1, atol=1e-8)


def test_identical_expiries_and_windows_give_a_factor_of_one():
    g = settlement.factor_grid(flat(), KALSHI_EXPIRY, [24, 72], SD, deribit_expiry=KALSHI_EXPIRY,
                               deribit_minutes=30, kalshi_minutes=30, n_paths=2_000)
    np.testing.assert_array_equal(g.kalshi_iv, g.deribit_iv)
    assert (g.factor == 1).all()


def test_atm_factor_matches_the_variance_formula():
    # Deribit 24 hours out, Kalshi 13 hours later: sqrt(((37 - 1/90) / 37) / ((24 - 1/3) / 24)) = 1.006866415
    g = settlement.factor_grid(flat(), KALSHI_EXPIRY, [37], SD, n_paths=20_000)
    atm = g[g.sd == 0].iloc[0]
    assert np.isclose(atm.deribit_hours, 24)
    assert np.isclose(atm.atm_approx, 1.006866415, atol=1e-9)
    assert abs(atm.factor - 1.006866415) < 2e-4


def test_busy_extra_hours_raise_the_factor():
    p = flat().copy()
    p.loc[4 * 24 + 8: 4 * 24 + 20] *= 4                     # Friday 08:00-21:00 UTC four times as volatile
    g = settlement.factor_grid(p, KALSHI_EXPIRY, [37], SD, deribit_minutes=0, kalshi_minutes=0, n_paths=200)
    np.testing.assert_allclose(g.factor, np.sqrt((24 + 13 * 4) / 37), rtol=1e-8)


def test_hours_that_start_inside_deribits_window_are_skipped():
    g = settlement.factor_grid(flat(), KALSHI_EXPIRY, [13.25, 13.5, 14], SD, n_paths=200)
    assert sorted(g.kalshi_hours.unique()) == [13.5, 14]


def test_options_too_cheap_to_invert_are_not_reliable():
    # 14 hours before Kalshi's expiry Deribit has 1 hour left: 3 Kalshi SDs out is over 10 Deribit SDs, price ~ 0
    g = settlement.factor_grid(flat(), KALSHI_EXPIRY, [14, 37], [-3.0, 0.0, 3.0], n_paths=2_000)
    g = g.set_index(["kalshi_hours", "sd"])
    assert (g.deribit_price.loc[14, 3.0] < settlement.MIN_PRICE) and not g.reliable.loc[14, 3.0]
    assert not g.reliable.loc[14, -3.0]
    assert g.reliable.loc[37].all() and g.deribit_price.loc[14, 0.0] > settlement.MIN_PRICE


def test_expiry_hours_start_at_deribits_window():
    assert list(settlement.expiry_hours(KALSHI_EXPIRY, last=17)) == [13.5, 14, 15, 16, 17]
    winter = pd.Timestamp("2026-01-16 22:00", tz="UTC")             # 5pm EST: 14 hours after Deribit
    assert list(settlement.expiry_hours(winter, last=16)) == [14.5, 15, 16]


def test_factor_for_rows_in_summer_and_winter():
    winter = pd.Timestamp("2026-01-16 22:00", tz="UTC")
    rows = pd.DataFrame({"kalshi_expiry": [KALSHI_EXPIRY, KALSHI_EXPIRY, winter, KALSHI_EXPIRY],
                         "kalshi_hours": [37.0, 36.75, 38.0, 13.25], "sd": [0.0, 1.3, -0.4, 0.0]})
    f = settlement.factor_for(rows, flat(), n_paths=20_000)
    # constant vol: only the averaging, sqrt(((T_K - 1/90) / T_K) / ((T_D - 1/3) / T_D)) with T in hours
    approx = lambda tk, td: np.sqrt(((tk - 1 / 90) / tk) / ((td - 1 / 3) / td))
    np.testing.assert_allclose(f[:3], [approx(37, 24), approx(36.75, 23.75), approx(38, 24)], atol=3e-4)
    assert np.isnan(f[3])                                             # starts inside Deribit's window


def test_factor_at_interpolates_and_is_empty_outside_the_grid():
    g = settlement.factor_grid(flat(), KALSHI_EXPIRY, [24, 25], SD, deribit_minutes=0, kalshi_minutes=0, n_paths=200)
    f = settlement.factor_at(g, [24.5, 24.5, 30], [0.25, 5.0, 0.0])
    np.testing.assert_allclose(f[0], 1, atol=1e-8)
    assert np.isnan(f[1]) and np.isnan(f[2])
