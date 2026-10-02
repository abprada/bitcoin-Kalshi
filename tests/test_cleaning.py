from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from lib import cleaning as cl


def utc(*args) -> pd.Timestamp:
    return pd.Timestamp(datetime(*args, tzinfo=timezone.utc))


# --- quote sides, mid and spread ---------------------------------------------------------------

def test_quote_sides_turns_placeholder_prices_into_missing_sides():
    # Kalshi encodes "no bid" as 0 and "no ask" as 1
    bid, ask = cl.quote_sides(np.array([0.0, 0.32, 0.99, 0.0]), np.array([1.0, 0.35, 1.0, 0.05]))
    np.testing.assert_array_equal(bid, [np.nan, 0.32, 0.99, np.nan])
    np.testing.assert_array_equal(ask, [np.nan, 0.35, np.nan, 0.05])


@pytest.mark.parametrize(
    "bid, ask, spread, mid",
    [
        (0.32, 0.35, 0.03, 0.335),        # ordinary two-sided quote
        (0.40, 0.60, 0.20, 0.50),         # exactly at the 20c cap: kept
        (0.40, 0.61, 0.21, np.nan),       # just over the cap: no usable mid
        (0.01, 0.99, 0.98, np.nan),       # placeholder quote
        (0.50, 0.50, np.nan, np.nan),     # locked: not a valid two-sided quote
        (0.55, 0.50, np.nan, np.nan),     # crossed
        (np.nan, 0.05, np.nan, np.nan),   # one-sided
    ],
)
def test_mid_and_spread_apply_two_sided_and_cap_rules(bid, ask, spread, mid):
    s, m = cl.mid_and_spread(np.array([bid], dtype=np.float32), np.array([ask], dtype=np.float32))
    np.testing.assert_allclose(s, [spread], atol=1e-9)
    np.testing.assert_allclose(m, [mid], atol=1e-6)


# --- exchange hours ----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "ts, is_open",
    [
        (utc(2026, 9, 24, 7, 0), True),    # Thu 03:00 EDT: minute 02:59-03:00, still open
        (utc(2026, 9, 24, 7, 1), False),   # Thu 03:01 EDT: closed
        (utc(2026, 9, 24, 9, 0), False),   # Thu 05:00 EDT: last closed minute
        (utc(2026, 9, 24, 9, 1), True),    # Thu 05:01 EDT: reopened
        (utc(2026, 1, 15, 8, 1), False),   # Thu 03:01 EST (winter offset)
        (utc(2026, 1, 15, 10, 1), True),   # Thu 05:01 EST
        (utc(2026, 9, 23, 7, 30), True),   # Wed 03:30 EDT: no closure on other days
    ],
)
def test_exchange_open_flags_thursday_closure_in_eastern_time(ts, is_open):
    assert cl.exchange_open(pd.Series([ts])).tolist() == [is_open]


def test_tau_years_is_act_365():
    expiry = utc(2026, 9, 25, 21)
    np.testing.assert_allclose(cl.tau_years(pd.Series([expiry - pd.Timedelta(days=1), expiry]), expiry),
                               [1 / 365, 0.0])


# --- settlement value ----------------------------------------------------------------------------

def test_reconstruct_expiration_rounds_index_one_second_before_expiry():
    t_a, t_b = utc(2026, 9, 25, 21), utc(2025, 10, 17, 21)
    settlement = pd.DataFrame({
        "event_ticker": ["A", "A", "A", "B"],
        "ts": [t_a - pd.Timedelta(seconds=2), t_a - pd.Timedelta(seconds=1), t_a, t_b - pd.Timedelta(seconds=5)],
        "index_60s": [83880.48299999993, 83880.37249999992, 83880.37, 107047.0],
    })
    out = cl.reconstruct_expiration(settlement, pd.Series({"A": t_a, "B": t_b}))
    assert out["A"] == 83880.37
    assert np.isnan(out["B"])  # no index value one second before B's expiry


# --- spot ----------------------------------------------------------------------------------------

def test_clean_spot_averages_exchanges_and_falls_back_to_bitstamp():
    t = [utc(2026, 5, 8, 1, 17), utc(2026, 5, 8, 1, 18), utc(2026, 5, 8, 1, 19)]
    spot = pd.DataFrame({
        "exchange": ["coinbase", "bitstamp", "bitstamp", "bitstamp"],
        "ts": [t[0], t[0], t[1], t[2]],
        "close": [100.0, 102.0, 101.0, 101.5],
        "volume": [1.0, 2.0, 0.5, 0.0],
    })
    out = cl.clean_spot(spot)
    assert out.spot.tolist() == [101.0, 101.0, 101.5]
    assert out.spot_source.tolist() == ["both", "bitstamp only", "bitstamp only"]
    assert out.spot_stale.tolist() == [False, False, True]  # no trade on any exchange in the last minute


# --- forward-filled quote grid ---------------------------------------------------------------------

def test_build_event_grid_forward_fills_from_first_candle_to_expiry():
    t0 = utc(2026, 9, 25, 20, 54)
    m = lambda i: t0 + pd.Timedelta(minutes=i)   # minute ends t0+1 .. t0+6, expiry at t0+6
    markets = pd.DataFrame({
        "market_ticker": ["A", "B"], "strike": [80000.0, 81000.0],
        "open_time": [t0, m(2)], "close_time": [m(6), m(6)],
    })
    candles = pd.DataFrame({
        "market_ticker": ["A", "A", "B"],
        "ts": [m(1), m(4), m(5)],
        "bid_close": [0.40, 0.45, 0.0], "ask_close": [0.43, 0.47, 0.20],
        "volume": [10.0, 3.0, 1.0], "open_interest": [100.0, 103.0, 1.0],
    })
    g = cl.build_event_grid(candles, markets)
    a, b = g[g.market_ticker == "A"], g[g.market_ticker == "B"]
    assert a.ts.tolist() == [m(i) for i in range(1, 7)]
    np.testing.assert_allclose(a.bid_raw, [0.40, 0.40, 0.40, 0.45, 0.45, 0.45])
    np.testing.assert_allclose(a.volume, [10, 0, 0, 3, 0, 0])
    np.testing.assert_allclose(a.open_interest, [100, 100, 100, 103, 103, 103])
    assert a.quote_age_min.tolist() == [0, 1, 2, 0, 1, 2]
    # B is alive from m(3) but has no quote until its first candle at m(5): earlier minutes are dropped
    assert b.ts.tolist() == [m(5), m(6)]
    np.testing.assert_allclose(b.ask_raw, [0.20, 0.20])
