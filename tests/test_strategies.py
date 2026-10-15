import numpy as np
import pandas as pd
import pytest

from lib import calendar_straddle as cs
from lib import deribit_marks as dm
from lib import iv, kalshi_hourly
from lib import vol_spread as vs


def test_implied_forward_recovers_the_forward_for_calls_and_puts():
    F, tau, sig = 101_000.0, 3 / 365, 0.45
    K = np.array([95_000.0, 100_000.0, 104_000.0])
    for is_call in (True, False):
        price = dm.bs_price(F, K, sig, tau, is_call)
        out = dm.implied_forward(price, K, np.full(3, sig), tau, np.full(3, is_call), np.full(3, 100_000.0))
        np.testing.assert_allclose(out, F, rtol=1e-6)


def _trades_from_smile(a, b, c, tau_days=2.0, F=100_000.0, n_per_strike=3):
    """Synthetic trades whose mark vols lie exactly on iv = a + b m + c m^2, m = ln(K/F)/sqrt(tau)."""
    t0 = pd.Timestamp("2026-03-02 12:00", tz="UTC")
    expiry = t0 + pd.Timedelta(days=tau_days)
    rows = []
    for K in np.arange(94_000, 106_001, 1_000.0):
        for j in range(n_per_strike):
            ts = t0 + pd.Timedelta(minutes=5 * j - 5)
            tau = (expiry - ts).total_seconds() / dm.YEAR_SECONDS
            m = np.log(K / F) / np.sqrt(tau)
            rows.append(dict(ts=ts, expiry=expiry, strike=K, m=m, iv_d=a + b * m + c * m**2, basis=0.0,
                             index_price=F))
    return pd.DataFrame(rows), t0, expiry


def test_smile_grid_recovers_a_quadratic_smile():
    d, t0, expiry = _trades_from_smile(0.45, -0.02, 0.05)
    g = dm.smile_grid(d)
    row = g[(g.t == t0) & (g.expiry == expiry)].iloc[0]
    assert row.atm_iv == pytest.approx(0.45, abs=1e-3)
    assert row.b == pytest.approx(-0.02, abs=1e-3)
    assert row.c == pytest.approx(0.05, abs=2e-3)


def test_backward_grid_only_uses_past_trades():
    d, t0, expiry = _trades_from_smile(0.45, 0.0, 0.0)
    future = d.assign(ts=d.ts + pd.Timedelta(minutes=30), iv_d=0.90)   # later trades at a very different vol
    g = dm.smile_grid(pd.concat([d, future], ignore_index=True), half_window="30min", centred=False)
    row = g[(g.t == t0) & (g.expiry == expiry)].iloc[0]
    assert row.atm_iv == pytest.approx(0.45, abs=1e-6)


def test_effective_half_spread_signs_by_taker_side():
    d = pd.DataFrame({"direction": ["buy", "sell"], "price_usd": [105.0, 95.0], "mark_usd": [100.0, 100.0],
                      "vega": [10.0, 10.0]})
    e = dm.effective_half_spread(d)
    assert e.hs_usd.tolist() == [5.0, 5.0]
    assert e.hs_vol.tolist() == [0.5, 0.5]


def test_option_fee_is_capped_at_12_5_percent_of_price():
    assert dm.option_fee(100_000.0, 1_000.0) == pytest.approx(30.0)       # 0.03% of the index
    assert dm.option_fee(100_000.0, 80.0) == pytest.approx(10.0)          # capped: 12.5% of 80


def test_strip_straddle_pays_like_a_straddle_inside_the_ladder():
    strikes = np.arange(95_000, 105_001, 500.0)
    K, S_T = 100_000.0, 102_300.0
    pays_yes = (S_T > strikes).astype(float)          # each "above k" contract settled
    lad = pd.DataFrame({"strike": strikes, "bid": pays_yes, "ask": pays_yes, "mid": pays_yes})
    v = kalshi_hourly.strip_straddle(lad, K, "mid")["value"]
    assert v == pytest.approx(abs(S_T - K), abs=250)   # within half a strike step


def test_strip_value_matches_black_scholes_straddle():
    F, tau, sig, K = 100_000.0, 2 / 365, 0.45, 100_000.0
    strikes = np.arange(85_000, 115_001, 250.0)
    D = np.asarray(iv.bs_call(F, strikes - 1, sig, tau) - iv.bs_call(F, strikes + 1, sig, tau)) / 2
    lad = pd.DataFrame({"strike": strikes, "bid": D, "ask": D, "mid": D})
    strip = vs.strip_at(lad, K, 15_000)
    value, _ = vs.price_strip(strip, lad, "mid")
    straddle = float(iv.bs_call(F, K, sig, tau) + iv.bs_put(F, K, sig, tau))
    assert value == pytest.approx(straddle, rel=0.01)


def test_price_strip_sides_bracket_mid_and_charge_fees():
    strikes = np.arange(99_000, 101_001, 500.0)
    lad = pd.DataFrame({"strike": strikes, "bid": [0.80, 0.65, 0.48, 0.30, 0.15],
                        "ask": [0.82, 0.67, 0.50, 0.32, 0.17]})
    lad["mid"] = (lad.bid + lad.ask) / 2
    strip = vs.strip_at(lad, 100_000.0, 2_000)
    mid, fee_mid = vs.price_strip(strip, lad, "mid")
    bid, fee_bid = vs.price_strip(strip, lad, "bid")
    ask, fee_ask = vs.price_strip(strip, lad, "ask")
    assert bid < mid < ask
    assert fee_mid == 0 and fee_bid > 0 and fee_ask > 0


def test_kalshi_fee_rounds_up_to_the_cent():
    assert vs.kalshi_fee(0.50, 1) == pytest.approx(0.02)     # 7% x 0.25 = 1.75c -> 2c
    assert vs.kalshi_fee(0.50, 100) == pytest.approx(1.75)


def test_deribit_at_kalshi_brackets_in_business_time():
    profile = pd.Series(1.0, index=range(168))                 # flat: business time = clock time
    t = pd.Timestamp("2026-03-04 12:00", tz="UTC")
    kalshi_exp = pd.Timestamp("2026-03-06 21:00", tz="UTC")
    ea, eb = pd.Timestamp("2026-03-06 08:00", tz="UTC"), pd.Timestamp("2026-03-07 08:00", tz="UTC")
    tau = lambda e: (e - t).total_seconds() / dm.YEAR_SECONDS
    grid = pd.DataFrame({"expiry": [ea, eb], "t": [t, t], "atm_iv": [0.40, 0.50], "tau_years": [tau(ea), tau(eb)]})
    snaps = pd.DataFrame({"ts": [t], "expiry": [kalshi_exp], "tau_years": [tau(kalshi_exp)]})
    out = vs.deribit_at_kalshi(snaps, grid, profile).iloc[0]
    va, vb = 0.40**2 * tau(ea), 0.50**2 * tau(eb)
    v = va + (vb - va) * 13 / 24
    assert out.method == "bracket"
    assert out.deribit_iv == pytest.approx(np.sqrt(v / tau(kalshi_exp)))


def test_deribit_at_kalshi_stretch_matches_05_when_flat():
    profile = pd.Series(1.0, index=range(168))
    t = pd.Timestamp("2026-03-02 20:00", tz="UTC")
    kalshi_exp, ea = pd.Timestamp("2026-03-06 21:00", tz="UTC"), pd.Timestamp("2026-03-06 08:00", tz="UTC")
    tau = lambda e: (e - t).total_seconds() / dm.YEAR_SECONDS
    grid = pd.DataFrame({"expiry": [ea], "t": [t], "atm_iv": [0.40], "tau_years": [tau(ea)]})
    snaps = pd.DataFrame({"ts": [t], "expiry": [kalshi_exp], "tau_years": [tau(kalshi_exp)]})
    out = vs.deribit_at_kalshi(snaps, grid, profile).iloc[0]
    assert out.method == "stretch_before"
    assert out.deribit_iv == pytest.approx(0.40)              # flat profile: same vol per hour


def test_realized_vol_of_constant_minute_moves():
    ts = pd.date_range("2026-03-02", periods=1441, freq="min", tz="UTC")
    spot = pd.Series(100_000 * np.exp(0.001 * (np.arange(1441) % 2)), index=ts)   # +/-0.1% every minute
    rv = vs.realized_vol(spot, [ts[0]], [ts[-1]])[0]
    assert rv == pytest.approx(0.001 * np.sqrt(1440 * 365), rel=1e-6)


def test_half_spread_lookup_falls_back_to_the_median():
    table = pd.DataFrame({"hs_usd": [5.0, 7.0]}, index=pd.MultiIndex.from_arrays(
        [pd.cut([0.5, 0.5], cs.HS_TAU_EDGES), pd.cut([0.2, 0.7], cs.HS_SD_EDGES)]))
    assert cs.lookup_half_spread(table, 0.5, 0.2)[0] == 5.0
    assert cs.lookup_half_spread(table, 10.0, 0.2)[0] == 6.0


def test_digital_from_flat_smile_is_n_d2_and_skew_moves_it():
    from scipy.special import ndtr
    from lib import dislocation as dl
    F, tau, K = 100_000.0, 2 / 365, np.array([98_000.0, 100_000.0, 103_000.0])
    sd = 0.45 * np.sqrt(tau)
    d2 = (np.log(F / K) - 0.5 * sd**2) / sd
    np.testing.assert_allclose(dl.digital_from_deribit(K, F, tau, 0.45, 0.0, 0.0, -5, 5), ndtr(d2), atol=1e-12)
    # vol falling with strike (put skew): -vega x dvol/dK > 0, so "above K" is worth more than N(d2)
    skewed = dl.digital_from_deribit(K, F, tau, 0.45, -0.02, 0.0, -5, 5)
    assert np.all(skewed > ndtr(d2))


def test_dislocation_normal_uses_only_earlier_days():
    from lib import dislocation as dl
    days = pd.date_range("2026-03-01", periods=40, freq="D", tz="UTC")
    rows = [dict(event_ticker=f"E{i}", ts=d + pd.Timedelta(hours=h), hours_left=100.0, kind="weekly",
                 gap_sig=1.0 if i < 39 else 50.0) for i, d in enumerate(days) for h in range(3)]
    p = dl.add_dislocation(pd.DataFrame(rows))
    last = p[p.event_ticker == "E39"]
    assert (last.normal_gap == 1.0).all()            # today's 50s do not enter today's normal
    assert (last.disl == 49.0).all()


def _flat_path(S0=100_000.0, hours=6, vol=0.4, kalshi=0.5, drift=0.0):
    from lib import binary_hedge as bh  # noqa: F401
    idx = pd.date_range("2026-03-02 12:00", periods=hours * 60 + 1, freq="min", tz="UTC")
    T = idx[-1]
    spot = S0 * (1 + drift * np.arange(len(idx)) / len(idx))
    fr = pd.DataFrame({"spot": spot, "tau": (T - idx).total_seconds() / (365 * 86400), "atm": vol, "b": 0.0,
                       "c": 0.0, "m_lo": -9.0, "m_hi": 9.0}, index=idx)
    fr["kalshi_mid"] = np.nan
    fr.iloc[0, fr.columns.get_loc("kalshi_mid")] = kalshi
    return fr


def test_binary_value_matches_n_d2_with_a_flat_smile():
    from scipy.special import ndtr
    from lib import binary_hedge as bh
    S, K, tau, vol = 100_000.0, 101_000.0, 3 / 8760, 0.4
    sd = vol * np.sqrt(tau)
    expected = ndtr((np.log(S / K) - 0.5 * sd**2) / sd)
    assert bh.binary_value(S, K, tau, bh.Smile(S, tau, vol, 0, 0, -9, 9)) == pytest.approx(expected, abs=1e-4)


def test_static_spread_locks_the_edge_when_settling_far_from_the_strike():
    from lib import binary_hedge as bh
    fr = _flat_path()
    for settle in (95_000.0, 105_000.0):
        s = bh.simulate(fr, 100_000.0, settle, h=250, step=None)["summary"]
        assert s["total"] == pytest.approx(s["edge_at_entry"], abs=1e-6)


def test_one_sided_spread_never_loses_at_expiry_relative_to_its_edge():
    from lib import binary_hedge as bh
    fr = _flat_path()
    for settle in np.arange(99_500.0, 100_700.0, 50.0):
        s = bh.simulate(fr, 100_000.0, settle, h=250, step=None, offset=250)["summary"]
        assert s["total"] - s["edge_at_entry"] >= -1e-6


def test_book_keeps_one_position_per_contract_and_metrics_count_it():
    from lib import strategy_ab as ab
    t0 = pd.Timestamp("2026-03-02 10:00", tz="UTC")
    rows = []
    for ev, n in (("E1", 3), ("E2", 2)):
        for i in range(n):
            rows.append(dict(event_ticker=ev, kind="weekly", signal=-7.0, hold="expiry", ts=t0 + pd.Timedelta(hours=i),
                             deribit_expiry=t0 + pd.Timedelta(days=2), kalshi_expiry=t0 + pd.Timedelta(days=2),
                             d_gross=100.0 * (i + 1), d_spread=5.0, d_fee=10.0, d_perp=1.0, vega_k=50.0))
    tr = pd.DataFrame(rows)
    b = ab.book(tr, "A", 6, "expiry", "taker")
    assert list(b.event_ticker) == ["E1", "E2"]
    assert b.pnl_vol.tolist() == pytest.approx([(100 - 16) / 50] * 2)       # the first hour's trade each
    m = ab.metrics(b, "2026-03-01", "2026-03-10")
    assert m["trades"] == 2 and m["hit"] == 1.0


def test_load_options_falls_back_to_the_full_file(tmp_path):
    # Two trades: one 3.5 days from expiry (kept), one 20 days (beyond MAX_TTE_DAYS, dropped)
    cols = ["window", "date", "instrument", "expiry_utc", "strike", "cp", "time_utc", "price_btc", "index_price",
            "price_usd", "amount", "direction", "iv", "mark_price_btc", "trade_id", "block_trade_id"]
    mk = lambda exp: ["other", "2026-03-02", f"BTC-X-90000-C", exp, 90_000.0, "C", "2026-03-02T20:00:00.000000+00:00",
                      0.02, 90_000.0, 1800.0, 0.1, "buy", 45.0, 0.02, "1", None]
    full = tmp_path / "data" / "raw" / "deribit_trades" / "btc_options_trades.parquet"
    full.parent.mkdir(parents=True)
    pd.DataFrame([mk("2026-03-06T08:00:00+00:00"), mk("2026-03-22T08:00:00+00:00")], columns=cols).to_parquet(full)
    d = dm.load_options(tmp_path)
    assert len(d) == 1 and d.tau_years.iloc[0] * 365 == pytest.approx(3.5)
