import importlib.util
from datetime import date
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "fetch_data", Path(__file__).resolve().parents[1] / "00_fetch_data.py"
)
fd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fd)


@pytest.mark.parametrize(
    "ticker, want",
    [
        ("KXBTCD-26OCT0217", date(2026, 10, 2)),
        ("BTCD-24OCT2317", date(2024, 10, 23)),
        ("BTCD-24JUN07-17", date(2024, 6, 7)),
    ],
)
def test_event_date_parses_all_ticker_formats(ticker, want):
    assert fd.event_date(ticker) == want


START, END = date(2025, 10, 1), date(2026, 10, 1)


@pytest.mark.parametrize(
    "ticker, want",
    [
        ("KXBTCD-26SEP2517", True),   # Friday 5pm, in range
        ("KXBTCD-25OCT0317", True),   # first Friday in range
        ("KXBTCD-26SEP2516", False),  # Friday 4pm
        ("KXBTCD-26SEP2417", False),  # Thursday 5pm
        ("KXBTCD-26OCT0217", False),  # Friday 5pm, after end
        ("KXBTCD-25SEP2617", False),  # Friday 5pm, before start
        ("BTCD-24MAR19-16B", False),  # legacy "additional strikes" oddball
    ],
)
def test_is_candidate_keeps_only_friday_5pm_in_range(ticker, want):
    assert fd.is_candidate(ticker, START, END) is want


def test_is_candidate_range_is_inclusive():
    friday = date(2026, 9, 25)
    assert fd.is_candidate("KXBTCD-26SEP2517", friday, friday) is True


def live_market(**overrides):
    """Real market from GET /markets?event_ticker=KXBTCD-26SEP2517 (rules text trimmed)."""
    m = {
        "can_close_early": True, "close_time": "2026-09-25T21:00:00Z",
        "created_time": "2026-09-18T09:00:31.132567Z", "event_ticker": "KXBTCD-26SEP2517",
        "exchange_index": 2, "expected_expiration_time": "2026-09-25T21:05:00Z",
        "expiration_time": "2026-10-02T21:00:00Z", "expiration_value": "83880.37",
        "floor_strike": 88999.99, "last_price_dollars": "0.0100",
        "latest_expiration_time": "2026-10-02T21:00:00Z", "market_type": "binary",
        "no_ask_dollars": "1.0000", "no_bid_dollars": "0.0000", "no_sub_title": "$89,000 or above",
        "notional_value_dollars": "1.0000", "occurrence_datetime": "2026-09-25T21:05:00Z",
        "open_interest_fp": "25169.94", "open_time": "2026-09-18T20:00:00Z",
        "previous_price_dollars": "0.0100", "previous_yes_ask_dollars": "1.0000",
        "previous_yes_bid_dollars": "0.0000", "price_level_structure": "linear_cent",
        "price_ranges": [{"end": "1.0000", "start": "0.0000", "step": "0.0100"}],
        "result": "no", "rules_primary": "...above 88999.99 at 5 PM EDT on Sep 25, 2026...",
        "rules_secondary": "...", "settlement_bounds_type": "default",
        "settlement_timer_seconds": 60, "settlement_ts": "2026-09-25T21:02:28.209837Z",
        "settlement_value_dollars": "0.0000", "status": "finalized", "strike_type": "greater",
        "subtitle": "$89,000 or above", "ticker": "KXBTCD-26SEP2517-T88999.99",
        "title": "Bitcoin price on Sep 25, 2026?", "updated_time": "2026-09-25T21:02:28.66575Z",
        "volume_24h_fp": "0.00", "volume_fp": "40338.45", "yes_ask_dollars": "1.0000",
        "yes_ask_size_fp": "0.00", "yes_bid_dollars": "0.0000", "yes_bid_size_fp": "0.00",
        "yes_sub_title": "$89,000 or above",
    }
    return {**m, **overrides}


def test_market_row_strike_is_the_or_above_threshold():
    # "$89,000 or above" is encoded as floor_strike 88999.99 (rules: "above 88999.99")
    assert fd.market_row(live_market(), "live")["strike"] == 89000.0


def test_market_row_parses_numeric_strings():
    row = fd.market_row(live_market(), "live")
    assert row["expiration_value"] == 83880.37
    assert row["volume"] == 40338.45
    assert row["open_interest"] == 25169.94
    assert row["settlement_value"] == 0.0


def test_market_row_records_source_and_times():
    row = fd.market_row(live_market(), "historical")
    assert row["source"] == "historical"
    assert row["market_ticker"] == "KXBTCD-26SEP2517-T88999.99"
    assert row["open_time"] == "2026-09-18T20:00:00Z"
    assert row["close_time"] == "2026-09-25T21:00:00Z"


LIVE_CANDLE = {  # GET /series/KXBTCD/markets/KXBTCD-26SEP2517-T83999.99/candlesticks
    "end_period_ts": 1790355600, "open_interest_fp": "106940.24",
    "price": {"close_dollars": "0.3300", "high_dollars": "0.3300", "low_dollars": "0.3200",
              "mean_dollars": "0.3262", "open_dollars": "0.3200", "previous_dollars": "0.3500"},
    "volume_fp": "419.00",
    "yes_ask": {"close_dollars": "0.3300", "high_dollars": "0.3500", "low_dollars": "0.3200", "open_dollars": "0.3500"},
    "yes_bid": {"close_dollars": "0.3200", "high_dollars": "0.3300", "low_dollars": "0.2900", "open_dollars": "0.3300"},
}

HIST_CANDLE = {  # GET /historical/markets/KXBTCD-25JUN1817-T104999.99/candlesticks
    "end_period_ts": 1750190460, "open_interest": "0.00",
    "price": {"close": None, "high": None, "low": None, "mean": None, "open": None, "previous": None},
    "volume": "0.00",
    "yes_ask": {"close": "0.4700", "high": "0.4800", "low": "0.4600", "open": "0.4600"},
    "yes_bid": {"close": "0.4100", "high": "0.4200", "low": "0.0000", "open": "0.0000"},
}


def test_candle_rows_normalizes_live_schema():
    assert fd.candle_rows([LIVE_CANDLE]) == [{
        "ts": 1790355600,
        "bid_open": 0.33, "bid_high": 0.33, "bid_low": 0.29, "bid_close": 0.32,
        "ask_open": 0.35, "ask_high": 0.35, "ask_low": 0.32, "ask_close": 0.33,
        "volume": 419.0, "open_interest": 106940.24,
    }]


def test_candle_rows_normalizes_historical_schema():
    assert fd.candle_rows([HIST_CANDLE]) == [{
        "ts": 1750190460,
        "bid_open": 0.0, "bid_high": 0.42, "bid_low": 0.0, "bid_close": 0.41,
        "ask_open": 0.46, "ask_high": 0.48, "ask_low": 0.46, "ask_close": 0.47,
        "volume": 0.0, "open_interest": 0.0,
    }]


def rows(*specs):
    """Market rows from (open_time, close_time, status) triples."""
    return [fd.market_row(live_market(open_time=o, close_time=c, status=s), "live") for o, c, s in specs]


def test_classify_event_week_long_and_settled_is_weekly():
    rs = rows(("2026-09-18T20:00:00Z", "2026-09-25T21:00:00Z", "finalized"),
              ("2026-09-24T11:30:00Z", "2026-09-25T21:00:00Z", "finalized"))  # strike added late
    assert fd.classify_event(rs) == "weekly"


def test_classify_event_daily_friday_is_too_short():
    # Early-2024 Fridays were 25h dailies
    rs = rows(("2024-04-04T20:00:00Z", "2024-04-05T21:00:00Z", "finalized"))
    assert fd.classify_event(rs) == "too_short"


def test_classify_event_any_unfinalized_strike_is_unsettled():
    rs = rows(("2026-09-25T20:00:00Z", "2026-10-02T21:00:00Z", "finalized"),
              ("2026-09-25T20:00:00Z", "2026-10-02T21:00:00Z", "active"))
    assert fd.classify_event(rs) == "unsettled"


def test_classify_event_without_markets():
    assert fd.classify_event([]) == "no_markets"


def test_market_row_rejects_non_threshold_strikes():
    # The dataset is P(S_T > K); a bucket market would silently break that.
    with pytest.raises(ValueError):
        fd.market_row(live_market(strike_type="between"), "live")


def live_data(maturity_value=83880.37, include_maturity=True):
    """GET /live_data/events/KXBTCD-26SEP2517 -> ["live_data"], series trimmed to its last 3 seconds."""
    series = [
        {"t": 1790369998000, "v": 83880.48299999993},
        {"t": 1790369999000, "v": 83880.37249999992},
    ] + ([{"t": 1790370000000, "v": maturity_value}] if include_maturity else [])
    return {
        "default_range": "1h",
        "details": {
            "candlesticks": {
                "15M": [{"open_ts_ms": 1790359200000, "open": 83768.63083333333, "high": 83995.60166666661,
                         "low": 83768.63083333333, "close": 83986.49983333329}],
                "1M": [{"open_ts_ms": 1790359200000, "open": 83768.63083333333, "high": 83798.68416666656,
                        "low": 83768.63083333333, "close": 83798.68416666656}],
            },
            "coin": "BTC",
            "event_ticker": "KXBTCD-26SEP2517",
            "maturity_ts_ms": 1790370000000,
            "timeseries": series,
        },
        "is_historical": True,
        "range_options": ["15min", "1h", "3h"],
        "type": "crypto",
    }


def test_settlement_rows_converts_ms_timestamps_to_seconds():
    assert fd.settlement_rows(live_data()) == [
        {"ts": 1790369998, "index_60s": 83880.48299999993},
        {"ts": 1790369999, "index_60s": 83880.37249999992},
        {"ts": 1790370000, "index_60s": 83880.37},
    ]


def test_market_row_keeps_settlement_timestamp():
    assert fd.market_row(live_market(), "live")["settlement_ts"] == "2026-09-25T21:02:28.209837Z"


def settled_ladder(**overrides_for_89k):
    """Two strikes of KXBTCD-26SEP2517, which settled at 83880.37: $89k resolved no, $80k yes."""
    return [
        fd.market_row(live_market(**overrides_for_89k), "live"),
        fd.market_row(live_market(ticker="KXBTCD-26SEP2517-T79999.99", floor_strike=79999.99,
                                  result="yes", settlement_value_dollars="1.0000"), "live"),
    ]


def test_resolution_problems_none_when_consistent():
    assert fd.resolution_problems(settled_ladder(), live_data()) == []


def test_resolution_problems_flags_result_contradicting_settlement_value():
    problems = fd.resolution_problems(settled_ladder(result="yes"), live_data())
    assert len(problems) == 1
    assert "KXBTCD-26SEP2517-T88999.99" in problems[0]


def test_resolution_problems_flags_index_not_matching_settlement_value():
    assert len(fd.resolution_problems(settled_ladder(), live_data(maturity_value=83881.00))) == 2


def test_resolution_problems_flags_missing_index_at_maturity():
    assert len(fd.resolution_problems(settled_ladder(), live_data(include_maturity=False))) == 1


@pytest.mark.parametrize("result, n_problems", [("no", 0), ("yes", 1)])
def test_resolution_problems_above_is_strict(result, n_problems):
    # Settlement exactly at floor_strike is not "above" -> must resolve no
    ladder = [fd.market_row(live_market(floor_strike=83880.37, result=result), "live")]
    assert len(fd.resolution_problems(ladder, live_data())) == n_problems


def test_resolution_problems_flags_missing_expiration_value_once():
    # KXBTCD-25OCT1717 settled below its ladder and Kalshi left expiration_value blank;
    # NaN comparisons are always False, so this must be checked explicitly.
    ladder = [fd.market_row(live_market(expiration_value=""), "live"),
              fd.market_row(live_market(ticker="KXBTCD-26SEP2517-T79999.99", floor_strike=79999.99,
                                        result="yes", expiration_value=""), "live")]
    problems = fd.resolution_problems(ladder, live_data())
    assert len(problems) == 1
    assert "expiration_value" in problems[0]


def test_resolution_problems_checks_results_against_index_when_value_missing():
    # $89k strike resolving yes contradicts an index of 83880.37 even without expiration_value
    ladder = [fd.market_row(live_market(expiration_value="", result="yes"), "live")]
    assert len(fd.resolution_problems(ladder, live_data())) == 2
