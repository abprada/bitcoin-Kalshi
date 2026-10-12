import numpy as np
import pytest
from scipy.special import ndtr

from lib import iv


def digital(F, K, sigma, tau):
    """Black-Scholes cash-or-nothing call (r = 0), built independently of lib/iv.py."""
    K = np.asarray(K, dtype=float)
    d2 = (np.log(F / K) - 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    return ndtr(d2)


# --- vanilla Black-Scholes (r = 0, forward = spot) ----------------------------------------------

def test_bs_call_matches_hand_value_at_the_money():
    # d1 = 0.1, d2 = -0.1: C = 100 * (N(0.1) - N(-0.1)) = 7.965567
    assert iv.bs_call(100.0, 100.0, 0.2, 1.0) == pytest.approx(7.965567, abs=1e-6)


def test_bs_put_call_parity():
    F, K = 84_000.0, np.array([80_000.0, 84_000.0, 88_000.0])
    np.testing.assert_allclose(iv.bs_call(F, K, 0.45, 0.01) - iv.bs_put(F, K, 0.45, 0.01), F - K, atol=1e-6)


def test_implied_vol_recovers_hand_value():
    assert iv.implied_vol(np.array([7.965567]), 100.0, np.array([100.0]), 1.0, np.array([True]))[0] == pytest.approx(0.2, abs=1e-6)


def test_implied_vol_round_trips_calls_and_puts():
    F, tau = 84_000.0, 3 / 365
    K = np.array([80_000.0, 83_000.0, 85_000.0, 88_000.0])
    is_call = K >= F
    sig = np.array([0.62, 0.48, 0.41, 0.55])
    price = np.where(is_call, iv.bs_call(F, K, sig, tau), iv.bs_put(F, K, sig, tau))
    np.testing.assert_allclose(iv.implied_vol(price, F, K, tau, is_call), sig, atol=1e-6)


def test_implied_vol_is_nan_outside_no_arbitrage_bounds():
    out = iv.implied_vol(np.array([-1.0, 90_000.0]), 84_000.0, np.array([88_000.0, 88_000.0]), 0.01, np.array([True, True]))
    assert np.isnan(out).all()


def test_implied_vol_is_nan_for_a_worthless_option():
    # a zero price has no positive vol; it must not come back as the bottom of the search range
    out = iv.implied_vol(np.array([0.0, 0.0]), 84_000.0, np.array([95_000.0, 70_000.0]), 1 / 365, np.array([True, False]))
    assert np.isnan(out).all()


# --- binary implied vol (direct inversion of N(d2)) -----------------------------------------------

def test_digital_iv_in_the_money_has_a_unique_root():
    D = digital(105.0, 100.0, 0.5, 0.5)
    assert iv.digital_iv(np.array([D]), 105.0, np.array([100.0]), 0.5)[0] == pytest.approx(0.5, abs=1e-6)


def test_digital_iv_out_of_the_money_takes_the_lower_root():
    # spot below strike: price rises then falls in sigma; the true 30% vol is the lower of two roots
    D = digital(100.0, 105.0, 0.3, 0.1)
    assert iv.digital_iv(np.array([D]), 100.0, np.array([105.0]), 0.1)[0] == pytest.approx(0.3, abs=1e-6)


def test_digital_iv_near_the_money_takes_the_root_on_the_reference_branch():
    # 0.05% OTM: the price peaks at sigma* = sqrt(2*0.0005/0.1) = 0.10, below the true 30% vol, so the
    # true vol is the UPPER root; a reference vol above sigma* must select it
    F, K, tau = 100.0, 100.05, 0.1
    D = digital(F, K, 0.3, tau)
    assert iv.digital_iv(np.array([D]), F, np.array([K]), tau, sigma_ref=0.5)[0] == pytest.approx(0.3, abs=1e-6)
    assert iv.digital_iv(np.array([D]), F, np.array([K]), tau)[0] < 0.10   # without a reference: lower root


def test_digital_iv_is_nan_above_the_maximum_attainable_price():
    # max over sigma of N(d2) at ln(S/K) = ln(100/105) is N(-sqrt(2*0.04879)) = 0.377
    assert np.isnan(iv.digital_iv(np.array([0.45]), 100.0, np.array([105.0]), 0.1)[0])


# --- monotone ladder ------------------------------------------------------------------------------

def test_isotonic_decreasing_pools_violations():
    np.testing.assert_allclose(iv.isotonic_decreasing(np.array([0.9, 0.8, 0.85, 0.5])), [0.9, 0.825, 0.825, 0.5])


def test_isotonic_decreasing_respects_weights():
    out = iv.isotonic_decreasing(np.array([0.9, 0.8, 0.85, 0.5]), np.array([1.0, 3.0, 1.0, 1.0]))
    np.testing.assert_allclose(out, [0.9, 0.8125, 0.8125, 0.5])


def test_isotonic_decreasing_leaves_monotone_ladders_unchanged():
    y = np.array([0.95, 0.7, 0.4, 0.1])
    np.testing.assert_allclose(iv.isotonic_decreasing(y), y)


# --- integration: binaries -> calls and puts --------------------------------------------------------

def test_integrate_ladder_without_tails_ends_at_the_listed_strikes():
    # nothing beyond the outermost strikes: the top call and the bottom put are worth zero
    K, D = np.array([100.0, 110.0, 120.0]), np.array([0.8, 0.5, 0.1])
    out = iv.integrate_ladder(K, D)
    np.testing.assert_allclose(out["call"], [0.65 * 10 + 0.3 * 10, 0.3 * 10, 0.0])
    np.testing.assert_allclose(out["put"], [0.0, 0.35 * 10, 0.35 * 10 + 0.7 * 10])
    assert out["forward"] == pytest.approx(100.0 + 9.5)      # bottom strike + area under the ladder


def test_integrate_ladder_adds_tail_values_to_every_call_and_put():
    K, D = np.array([100.0, 110.0, 120.0]), np.array([0.8, 0.5, 0.1])
    base, out = iv.integrate_ladder(K, D), iv.integrate_ladder(K, D, tail_up=2.0, tail_dn=0.5)
    np.testing.assert_allclose(out["call"], base["call"] + 2.0)
    np.testing.assert_allclose(out["put"], base["put"] + 0.5)
    assert out["forward"] == pytest.approx(base["forward"] + 2.0 - 0.5)


def test_integrate_ladder_satisfies_parity_at_its_implied_forward():
    # C - P = (mean of the ladder's distribution) - K at every strike: that mean is the implied forward
    K = np.array([82_000.0, 83_000.0, 84_000.0, 85_000.0, 86_000.0])
    D = np.array([0.93, 0.78, 0.52, 0.27, 0.09])          # any monotone ladder
    out = iv.integrate_ladder(K, D, tail_up=12.0, tail_dn=7.0)
    np.testing.assert_allclose(out["call"] - out["put"], out["forward"] - K, atol=1e-8)


def test_integrate_ladder_recovers_black_scholes_prices_when_the_ladder_spans_the_distribution():
    F, tau, sig = 84_000.0, 1 / 365, 0.45                  # one sd = $1,979; ladder covers +/- 6 sd
    K = np.arange(72_000.0, 96_001.0, 250.0)
    out = iv.integrate_ladder(K, digital(F, K, sig, tau))
    assert abs(out["forward"] - F) / F * 1e4 < 1.0         # implied forward = spot to within 1 bp
    np.testing.assert_allclose(out["call"], iv.bs_call(F, K, sig, tau), rtol=0.01, atol=1.0)


# --- bounds on the tails from Kalshi's other listed strikes -----------------------------------------

def test_tail_bounds_from_kalshis_outer_quotes():
    # ladder 80-120; Kalshi also lists 60, 70 below and 130, 140 above (NaN = no bid / no ask)
    K, D = np.array([80.0, 100.0, 120.0]), np.array([0.9, 0.5, 0.1])
    K_out = np.array([60.0, 70.0, 130.0, 140.0])
    bid = np.array([0.99, 0.93, 0.04, np.nan])
    ask = np.array([np.nan, 0.97, 0.06, 0.02])
    b = iv.tail_bounds(K, D, K_out, bid, ask)
    # above: D <= running min of asks moving out [0.10, 0.06, 0.02], D >= farthest bids [0.04, 0]
    assert b["up"] == pytest.approx((0.04 * 10 + 0.0 * 10, 0.10 * 10 + 0.06 * 10))
    assert b["K_up"] == 140.0
    # below, for 1 - D: upper bounds 1 - bid [0.10, 0.07, 0.01], lower bounds 1 - ask [0.03, 0]
    assert b["dn"] == pytest.approx((0.03 * 10 + 0.0 * 10, 0.10 * 10 + 0.07 * 10))
    assert b["K_dn"] == 60.0


def test_tail_bounds_without_outer_quotes_cover_nothing():
    K, D = np.array([80.0, 100.0, 120.0]), np.array([0.9, 0.5, 0.1])
    b = iv.tail_bounds(K, D, np.array([]), np.array([]), np.array([]))
    assert b["up"] == (0.0, 0.0) and b["dn"] == (0.0, 0.0)
    assert (b["K_dn"], b["K_up"]) == (80.0, 120.0)


# --- one snapshot end to end ------------------------------------------------------------------------

@pytest.mark.parametrize("step, tol", [(250.0, 0.003), (500.0, 0.01)])
def test_snapshot_iv_recovers_a_flat_45pct_smile(step, tol):
    # no outer quotes: the tails are the edge strikes' own Black-Scholes tails
    F, tau, sig = 84_000.0, 1 / 365, 0.45
    K = np.arange(79_000.0, 89_001.0, step)
    strikes, snap = iv.snapshot_iv(K, digital(F, K, sig, tau), np.ones_like(K), F, tau)
    np.testing.assert_allclose(strikes["iv"], sig, atol=tol)
    # 0.1 vol pt: at the money the binary barely depends on sigma, so the numerically inferred
    # forward (off by cents) moves the direct binary IV slightly there
    np.testing.assert_allclose(strikes["digital_iv"], sig, atol=1e-3)
    assert snap["monotone_fixes"] == 0
    assert snap["tails_clipped"] == 0
    assert snap["atm_iv"] == pytest.approx(sig, abs=tol)
    assert abs(snap["forward_basis_bps"]) < 1.0


def test_snapshot_iv_keeps_a_tail_that_kalshis_outer_quotes_allow():
    F, tau, sig = 84_000.0, 1 / 365, 0.45
    K = np.arange(79_000.0, 89_001.0, 500.0)
    K_out = np.r_[np.arange(75_000.0, 79_000.0, 500.0), np.arange(89_500.0, 93_001.0, 500.0)]
    D_out = digital(F, K_out, sig, tau)
    strikes, snap = iv.snapshot_iv(K, digital(F, K, sig, tau), np.ones_like(K), F, tau,
                                   K_out=K_out, bid_out=0.5 * D_out, ask_out=np.minimum(1.5 * D_out, 1.0))
    assert snap["tails_clipped"] == 0
    np.testing.assert_allclose(strikes["iv"], sig, atol=0.01)


def test_snapshot_iv_clips_the_tail_to_kalshis_outer_asks():
    # Kalshi offers the strikes above the ladder far below the Black-Scholes tail: the call tail is cut
    # to what those asks allow, so the top-edge vol falls, while vols near the money barely move
    F, tau, sig = 84_000.0, 1 / 365, 0.45
    K = np.arange(79_000.0, 89_001.0, 500.0)
    K_out = np.arange(89_500.0, 93_001.0, 500.0)
    ask = 0.1 * digital(F, K_out, sig, tau)
    strikes, snap = iv.snapshot_iv(K, digital(F, K, sig, tau), np.ones_like(K), F, tau,
                                   K_out=K_out, bid_out=np.full_like(K_out, np.nan), ask_out=ask)
    assert snap["tails_clipped"] == 1
    # between the top strike and the next listed one the bound can't beat D(top strike) x spacing,
    # so the cut is at most ~25% of this tail: about 2 vol pts at the top edge
    assert strikes["iv"].iloc[-1] < sig - 0.01
    near = np.abs(K - F) <= F * sig * np.sqrt(tau)
    np.testing.assert_allclose(strikes["iv"][near], sig, atol=0.01)


def test_snapshot_iv_infers_a_forward_away_from_spot_without_breaking_the_wings():
    # Ladder priced off a forward 20 bps above exchange spot: every strike, including far OTM calls
    # worth a few dollars, must still recover the 45% vol, and the basis must be reported.
    S, tau, sig = 84_000.0, 6 / 8760, 0.45
    F_true = S * 1.002
    K = np.arange(81_000.0, 87_001.0, 500.0)
    strikes, snap = iv.snapshot_iv(K, digital(F_true, K, sig, tau), np.ones_like(K), S, tau)
    assert snap["forward_basis_bps"] == pytest.approx(20.0, abs=1.5)
    assert strikes["iv"].notna().all()
    np.testing.assert_allclose(strikes["iv"], sig, atol=0.02)


def test_snapshot_iv_reports_the_probability_beyond_the_ladder():
    # P(S_T < bottom strike) + P(S_T > top strike): the part of the distribution priced by the tails
    K = np.array([82_000.0, 83_000.0, 84_000.0, 85_000.0, 86_000.0])
    D = np.array([0.93, 0.78, 0.52, 0.27, 0.09])
    _, snap = iv.snapshot_iv(K, D, np.ones_like(K), 84_000.0, 1 / 365)
    assert snap["prob_outside"] == pytest.approx(0.07 + 0.09)


def test_snapshot_iv_counts_monotonicity_fixes():
    F, tau = 84_000.0, 1 / 365
    K = np.arange(82_000.0, 86_001.0, 500.0)
    D = digital(F, K, 0.45, tau)
    D[3], D[4] = D[4], D[3]                                  # one inverted pair in the ladder
    _, snap = iv.snapshot_iv(K, D, np.ones_like(K), F, tau)
    assert snap["monotone_fixes"] == 2


# --- winsorizing for the surface ------------------------------------------------------------------

def test_winsorize_by_group_clips_each_group_to_its_own_percentiles():
    import pandas as pd
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, 10.0, 20.0, 30.0, 40.0, 1000.0])
    g = pd.Series(["a"] * 5 + ["b"] * 5)
    # 75th percentile: 4 in group a, 40 in group b (linear interpolation); the minimum is untouched
    out = iv.winsorize_by_group(x, g, lower=0.0, upper=0.75)
    np.testing.assert_allclose(out, [1, 2, 3, 4, 4, 10, 20, 30, 40, 40])


def test_winsorize_by_group_keeps_missing_values_out_of_the_percentiles():
    import pandas as pd
    x = pd.Series([1.0, np.nan, 3.0, 100.0])
    out = iv.winsorize_by_group(x, pd.Series(["a"] * 4), lower=0.0, upper=0.5)   # median of [1, 3, 100] = 3
    np.testing.assert_allclose(out, [1.0, np.nan, 3.0, 3.0])


# --- pooled (shape-normalized) surface ------------------------------------------------------------

def _pool_frame():
    import pandas as pd
    # two snapshots in the same time bin: ATM 40% and 80%, same smile shape relative to ATM
    return pd.DataFrame({
        "snap": [1, 1, 2, 2],
        "hours_to_expiry": [30.0, 30.0, 30.0, 30.0],
        "sd_from_spot": [-0.75, 0.25, -0.75, 0.25],
        "atm_iv": [0.4, 0.4, 0.8, 0.8],
        "iv_surface": [0.44, 0.40, 1.04, 0.80],      # ratios to ATM: 1.1, 1.0 and 1.3, 1.0
    })


def test_pooled_surface_scales_median_shape_by_median_atm():
    # cell (-1, 0]: median ratio 1.2 x median ATM 0.6 = 0.72; cell (0, 0.5]: 1.0 x 0.6 = 0.6
    grid = iv.pooled_surface(_pool_frame(), h_edges=[24, 48], m_edges=[-1, 0, 0.5], min_obs=1, min_snaps=1)
    np.testing.assert_allclose(grid.to_numpy(), [[0.72, 0.60]])


def test_pooled_surface_masks_cells_with_too_few_observations():
    import pandas as pd
    df = pd.concat([_pool_frame(), pd.DataFrame({"snap": [1], "hours_to_expiry": [30.0], "sd_from_spot": [0.75],
                                                 "atm_iv": [0.4], "iv_surface": [0.5]})], ignore_index=True)
    grid = iv.pooled_surface(df, h_edges=[24, 48], m_edges=[-1, 0, 0.5, 1], min_obs=2, min_snaps=2)
    np.testing.assert_allclose(grid.to_numpy(), [[0.72, 0.60, np.nan]])


# --- which quotes make a ladder ---------------------------------------------------------------------

def test_ladder_quotes_keeps_fresh_usable_mids_in_the_price_band():
    import pandas as pd
    q = pd.DataFrame({"strike": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                      "mid": [0.50, 0.50, 0.99, 0.50, 0.50, np.nan],
                      "usable": [True, True, True, False, True, False],
                      "bid": [0.49, 0.49, 0.98, 0.30, 0.49, np.nan], "ask": [0.51] * 5 + [np.nan],
                      "quote_age_min": [0, 20, 0, 0, 0, 0], "exchange_open": [True] * 6,
                      "tau_years": [1 / 365] * 4 + [0.5 / 8760, 1 / 365]})
    ladder, outer = iv.ladder_quotes(q)
    # 2: too old; 5: under an hour to expiry. 3 (99c) and 4 (not usable) bound the tails; 6 has no quote at all
    assert ladder.strike.tolist() == [1.0]
    assert outer.strike.tolist() == [3.0, 4.0]


def test_ladder_ok_needs_enough_strikes_on_both_sides_of_spot():
    assert iv.ladder_ok([5, 4, 5], [90.0, 90.0, 101.0], [110.0, 110.0, 110.0], 100.0).tolist() == [True, False, False]
