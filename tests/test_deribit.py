import numpy as np
import pandas as pd
import pytest

from lib import deribit, iv


def write_trades(tmp_path, rows):
    cols = ["window", "date", "instrument", "expiry_utc", "strike", "cp", "time_utc", "price_btc", "index_price",
            "price_usd", "amount", "direction", "iv", "mark_price_btc"]
    path = tmp_path / "trades.csv.gz"
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return path


def row(strike=90_000.0, cp="C", iv_pct=45.0, mark_btc=0.01, amount=0.1, time="2026-03-02T20:00:00+00:00"):
    index = 90_000.0
    return ["us_afternoon", "2026-03-02", f"BTC-06MAR26-{int(strike)}-{cp}", "2026-03-06T08:00:00+00:00", strike, cp,
            time, mark_btc, index, mark_btc * index, amount, "buy", iv_pct, mark_btc]


def test_tau_and_moneyness(tmp_path):
    d = deribit.load_trades(write_trades(tmp_path, [row(strike=99_000.0)]))
    # 2026-03-02 20:00 to 2026-03-06 08:00 UTC is 3.5 days
    assert d.tau_years.iloc[0] == pytest.approx(3.5 / 365)
    assert d.log_moneyness.iloc[0] == pytest.approx(np.log(1.1))


def test_placeholder_vols_are_invalid(tmp_path):
    d = deribit.load_trades(write_trades(tmp_path, [row(iv_pct=0.0), row(iv_pct=999.0), row(iv_pct=45.0)]))
    assert d.iv_valid.tolist() == [False, False, True]
    assert d.iv_deribit.isna().tolist() == [True, True, False]
    assert d.iv_deribit.iloc[2] == pytest.approx(0.45)


def test_mark_iv_recovers_black_scholes_vol(tmp_path):
    F, K, tau, sigma = 90_000.0, 92_000.0, 3.5 / 365, 0.52
    mark_btc = float(iv.bs_call(F, K, sigma, tau)) / F
    put_btc = float(iv.bs_put(F, 88_000.0, sigma, tau)) / F
    d = deribit.load_trades(write_trades(tmp_path, [row(strike=K, mark_btc=mark_btc),
                                                    row(strike=88_000.0, cp="P", mark_btc=put_btc)]))
    np.testing.assert_allclose(d.iv_mark, [sigma, sigma], atol=1e-6)


def test_exact_duplicates_flag_only_repeats(tmp_path):
    d = deribit.load_trades(write_trades(tmp_path, [row(), row(), row(amount=0.2)]))
    assert d.exact_dup.tolist() == [False, True, False]


def test_atm_vol_uses_band_and_minimum(tmp_path):
    rows = [row(strike=90_000.0, iv_pct=v) for v in (40.0, 42.0, 44.0)] + [row(strike=95_000.0, iv_pct=80.0)]
    d = deribit.load_trades(write_trades(tmp_path, rows))
    out = deribit.atm_vol(d, by=["date"], min_trades=3)
    assert out.atm_iv.tolist() == pytest.approx([0.42])        # the 95k trade is 5.4% out of the money
    assert out.n_trades.tolist() == [3]
    assert deribit.atm_vol(d, by=["date"], min_trades=4).empty


def test_hourly_variance_finds_the_busy_hour():
    ts = pd.date_range("2026-03-02 00:00", periods=7 * 24 * 60 * 2, freq="min", tz="UTC")  # two full weeks
    steps = np.where(ts.hour == 14, 0.002, 0.001) * np.where(np.arange(len(ts)) % 2 == 0, 1, -1)
    prof = deribit.hourly_variance(pd.Series(90_000 * np.exp(np.cumsum(steps)), index=ts))
    assert prof.iloc[14] / prof.iloc[13] == pytest.approx(4.0, rel=1e-3)   # twice the move, four times the variance
    assert len(prof) == 168


def test_expected_variance_counts_partial_hours():
    prof = pd.Series(np.ones(168))
    start = pd.Timestamp("2026-03-02 10:30", tz="UTC")
    assert deribit.expected_variance(start, start + pd.Timedelta(hours=2), prof) == pytest.approx(2.0)


def test_horizon_vol_ratio():
    flat = pd.Series(np.ones(168))
    start = pd.Timestamp("2026-03-02 20:00", tz="UTC")              # Monday
    short_end = pd.Timestamp("2026-03-06 08:00", tz="UTC")          # Friday 08:00
    long_end = pd.Timestamp("2026-03-06 22:00", tz="UTC")           # Friday 22:00
    assert deribit.horizon_vol_ratio(start, long_end, short_end, flat) == pytest.approx(1.0)
    busy = flat.copy()
    busy.iloc[4 * 24 + 8: 4 * 24 + 22] = 3.0                        # the 14 extra Friday hours carry 3x variance
    # per hour: (84 + 14 * 3) / 98 vs 84 / 84
    assert deribit.horizon_vol_ratio(start, long_end, short_end, busy) == pytest.approx(np.sqrt(126 / 98))


def test_horizon_vol_ratios_match_the_hour_by_hour_version():
    rng = np.random.default_rng(1)
    prof = pd.Series(rng.uniform(0.5, 3.0, 168))
    start = pd.DatetimeIndex(["2026-03-02 20:17", "2026-03-05 21:44:30", "2026-03-06 23:10"], tz="UTC")
    short_end = pd.DatetimeIndex(["2026-03-06 08:00", "2026-03-06 08:00", "2026-03-13 08:00"], tz="UTC")
    long_end = pd.DatetimeIndex(["2026-03-06 22:00", "2026-03-06 22:00", "2026-03-13 21:00"], tz="UTC")  # 3rd crosses a week
    loop = [deribit.horizon_vol_ratio(a, b, c, prof) for a, b, c in zip(start, long_end, short_end)]
    np.testing.assert_allclose(deribit.horizon_vol_ratios(start, long_end, short_end, prof), loop, rtol=1e-9)
