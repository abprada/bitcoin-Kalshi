import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "fetch_brti", Path(__file__).resolve().parents[1] / "00_fetch_brti.py"
)
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)


def utc(*args) -> int:
    return int(datetime(*args, tzinfo=timezone.utc).timestamp())


@pytest.mark.parametrize(
    "expiry_utc, want",
    [
        (utc(2025, 10, 15, 9), "KXBTCD-25OCT1505"),  # EDT: 09:00Z = 05:00 ET
        (utc(2026, 1, 15, 8), "KXBTCD-26JAN1503"),   # EST: 08:00Z = 03:00 ET
        (utc(2026, 10, 3, 4), "KXBTCD-26OCT0300"),   # midnight ET is hour 00
        (utc(2026, 1, 2, 3), "KXBTCD-26JAN0122"),    # 03:00Z on Jan 2 is still Jan 1 in ET
    ],
)
def test_kxbtcd_ticker_uses_eastern_date_and_hour(expiry_utc, want):
    assert fb.kxbtcd_ticker(expiry_utc) == want


@pytest.mark.parametrize(
    "expiry_utc, want",
    [
        (utc(2025, 10, 15, 11), "KXBTC15M-25OCT150700"),
        (utc(2026, 1, 15, 12, 15), "KXBTC15M-26JAN150715"),
    ],
)
def test_kxbtc15m_ticker_uses_eastern_hhmm(expiry_utc, want):
    assert fb.kxbtc15m_ticker(expiry_utc) == want


def live_data(coin="BTC"):
    """GET /live_data/events/KXBTCD-26SEP2517 -> ["live_data"], series trimmed to its last 3 seconds."""
    return {
        "default_range": "1h",
        "details": {
            "candlesticks": {"15M": [], "1M": []},
            "coin": coin,
            "event_ticker": "KXBTCD-26SEP2517",
            "maturity_ts_ms": 1790370000000,
            "timeseries": [
                {"t": 1790369998000, "v": 83880.48299999993},
                {"t": 1790369999000, "v": 83880.37249999992},
                {"t": 1790370000000, "v": 83880.37},
            ],
        },
        "is_historical": True,
        "range_options": ["15min", "1h", "3h"],
        "type": "crypto",
    }


def test_index_points_drops_the_settlement_value_at_maturity():
    # The point at maturity is the settlement value (the previous second's average, rounded to the
    # cent), not an observation of the running average; the next event supplies that second.
    assert fb.index_points(live_data()) == [(1790369998, 83880.48299999993), (1790369999, 83880.37249999992)]


def test_index_points_rejects_non_btc_payloads():
    with pytest.raises(ValueError):
        fb.index_points(live_data(coin="ETH"))


def test_merge_into_adds_points_inside_window_only():
    merged = {}
    fb.merge_into(merged, [(99, 1.0), (100, 2.0), (200, 3.0), (201, 4.0)], start_ts=100, end_ts=200)
    assert merged == {100: 2.0, 200: 3.0}


def test_merge_into_reports_largest_disagreement_on_overlap():
    merged = {100: 2.0, 101: 3.0}
    conflict = fb.merge_into(merged, [(100, 2.0), (101, 3.25), (102, 4.0)], start_ts=0, end_ts=1000)
    assert conflict == 0.25
    assert merged[102] == 4.0


def test_merge_into_without_overlap_reports_zero():
    assert fb.merge_into({}, [(5, 1.0)], start_ts=0, end_ts=10) == 0.0


def test_incomplete_hours_lists_hour_ends_with_any_missing_second():
    # Hour ending at h needs seconds h-3600 .. h-1 (blocks tile as [T-3h, T-1s])
    start, end = 0, 3 * 3600
    covered = set(range(start, end)) - {3600 + 1800}  # one second missing in the 2nd hour
    assert fb.incomplete_hours(covered, start, end) == [7200]


def test_incomplete_hours_empty_when_fully_covered():
    assert fb.incomplete_hours(set(range(0, 7200)), 0, 7200) == []


def test_incomplete_hours_flags_missing_first_second_of_hour():
    # Seconds 1..3600 present but 0 missing: incomplete, because second 3600 belongs to the next hour
    assert fb.incomplete_hours(set(range(1, 3601)), 0, 3600) == [3600]


def test_block_ends_every_three_hours_through_end():
    day = utc(2026, 3, 8)
    ends = fb.block_ends(day, day + 86_400)
    assert len(ends) == 8
    assert ends[0] == day + 3 * 3600 and ends[-1] == day + 86_400
