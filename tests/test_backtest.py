import numpy as np
import pandas as pd

from lib import backtest, compare


def _legs(spot=100_000.0):
    lad = pd.DataFrame({"strike": [99_000.0, 99_500.0, 100_500.0, 101_000.0, 105_000.0],
                        "bid": [0.70, 0.60, 0.38, 0.28, 0.01], "ask": [0.72, 0.62, 0.40, 0.30, 0.02],
                        "quote_age_min": 0})
    lad["mid"] = (lad.bid + lad.ask) / 2
    return backtest.strip_legs(lad, spot, 1_000)


def test_strip_legs_keeps_strikes_within_the_half_width():
    legs = _legs()
    assert list(legs.strike) == [99_000, 99_500, 100_500, 101_000]
    assert list(legs.up) == [False, False, True, True]


def test_short_strip_pays_like_a_capped_short_straddle():
    legs = _legs()
    # BTC ends at 100,200: no strike crossed on either side -> keep every premium
    value = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0], index=[99_000.0, 99_500.0, 100_500.0, 101_000.0, 105_000.0])
    p = backtest.strip_pnl(legs, value, side=-1, contracts=100)
    fee = compare.kalshi_fee(np.array([0.72, 0.62, 0.38, 0.28]), contracts=100)
    # buy YES below at the ask (worth 1), sell YES above at the bid (worth 0)
    expected = (1 - 0.72) + (1 - 0.62) + 0.38 + 0.28 - fee.sum()
    assert np.isclose(p["pnl"], expected)
    assert p["legs"] == 4 and p["step"] == 500
    assert np.isclose(p["collateral"], 0.72 + 0.62 + (1 - 0.38) + (1 - 0.28))
    # BTC ends at 101,500: both upper strikes lost, $1 each
    value_up = pd.Series([1.0, 1.0, 1.0, 1.0, 0.0], index=value.index)
    assert np.isclose(backtest.strip_pnl(legs, value_up, side=-1)["pnl"], expected - 2)


def test_long_strip_is_minus_short_before_costs_at_mid():
    legs = _legs().assign(bid=lambda x: x.mid, ask=lambda x: x.mid)
    value = pd.Series([1.0, 0.0, 0.0, 0.0], index=[99_000.0, 99_500.0, 100_500.0, 101_000.0])
    s = backtest.strip_pnl(legs, value, side=-1)
    l = backtest.strip_pnl(legs, value, side=+1)
    assert np.isclose(s["pnl"] + s["fees"], -(l["pnl"] + l["fees"]))


def test_strip_closed_at_exit_quotes_is_charged_both_ways():
    legs = _legs()
    p = backtest.strip_pnl(legs, pd.Series(dtype=float), side=+1, exit_quotes=legs[["strike", "bid", "ask"]])
    assert p["pnl"] < 0 and np.isclose(p["pnl"], -(legs.ask - legs.bid).sum() - p["fees"])


def test_deribit_short_straddle_keeps_premium_minus_costs_when_btc_ends_at_the_strike():
    r = backtest.deribit_straddle_pnl(100_000.0, 0.5, 7 / 365, 100_000.0, side=-1, half_spread_vol=0.0)
    assert np.isclose(r["gross"], r["premium"]) and np.isclose(r["pnl"], r["premium"] - r["fees"])
    # premium about 0.8 x sigma x sqrt(tau) x F
    assert abs(r["premium"] / (0.8 * 0.5 * np.sqrt(7 / 365) * 100_000) - 1) < 0.01


def test_summarize_resamples_weeks():
    t = pd.DataFrame({"entry": pd.date_range("2026-01-05", periods=20, freq="D", tz="UTC"), "x": np.arange(20.0)})
    s = backtest.summarize(t, "x", n_boot=200).iloc[0]
    assert s.trades == 20 and s.weeks == 3 and np.isclose(s["mean"], 9.5) and s.ci_low < 9.5 < s.ci_high


def test_exit_without_a_bid_counts_the_leg_as_worthless():
    legs = _legs()
    gone = legs[["strike", "bid", "ask"]].assign(bid=np.nan)          # nobody bids for any of them any more
    p = backtest.strip_pnl(legs, pd.Series(dtype=float), side=+1, exit_quotes=gone)
    assert p["legs"] == 4                                             # no leg dropped
    up = legs.up.to_numpy()
    entry = np.where(up, legs.ask, legs.bid)                          # long vol: buy above, sell below
    assert p["pnl"] < -entry[up].sum() + 1e-9                         # bought legs lost their whole price


def test_iron_fly_pays_like_a_kalshi_strip_at_expiry():
    K, w, tau = 100_000.0, 2_000.0, 1e-12
    for S, expected in ((100_000.0, 0.0), (101_000.0, 1_000.0), (98_500.0, 1_500.0), (110_000.0, 2_000.0)):
        v = backtest.fly_value(S, K, w, 0.5, tau)[0]
        assert abs(v - expected) < 1.0


def test_deribit_atm_at_uses_near_the_money_trades_close_in_time():
    t = pd.Timestamp("2026-03-10 18:00", tz="UTC")
    e = pd.Timestamp("2026-03-11 08:00", tz="UTC")
    tr = pd.DataFrame({"expiry": e, "ts": [t, t + pd.Timedelta(minutes=10), t - pd.Timedelta(minutes=20),
                                           t + pd.Timedelta(hours=2), t],
                       "iv_mark": [0.40, 0.42, 0.44, 0.90, 0.80], "tau_years": 14 / 8760,
                       "log_moneyness": [0.0, 0.001, -0.001, 0.0, 0.2]})   # last two: too late, too far out
    assert np.isclose(backtest.deribit_atm_at(tr, e, t), 0.42)
