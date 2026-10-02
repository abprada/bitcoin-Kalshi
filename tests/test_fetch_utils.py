from datetime import date, datetime, timezone

import pytest
import requests

import fetch_utils as fu


def test_time_chunks_splits_a_week_without_gaps_or_overlap():
    # Kalshi: 169h weekly = 10140 minutes, max 5000 candles per request.
    # Bounds are inclusive, so each chunk starts one minute after the previous ends.
    assert fu.time_chunks(0, 608_400, 5000) == [(0, 300_000), (300_060, 600_060), (600_120, 608_400)]


def test_time_chunks_keeps_short_range_whole():
    assert fu.time_chunks(1_000_000, 1_090_000, 5000) == [(1_000_000, 1_090_000)]


def test_time_chunks_exactly_at_cap_is_one_request():
    assert fu.time_chunks(0, 300_000, 5000) == [(0, 300_000)]


def test_time_chunks_small_span():
    # Coinbase: 300 candles per request, inclusive both ends -> 299-minute span
    assert fu.time_chunks(0, 36_000, 299) == [(0, 17_940), (18_000, 35_940), (36_000, 36_000)]


def response(status, body=b"{}"):
    r = requests.Response()
    r.status_code, r._content, r.url = status, body, "https://example/x"
    return r


class FakeSession:
    """Stands in for the network: replays responses (or raises exceptions) in order."""

    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def client(session, **kw):
    sleeps = []
    c = fu.Client("https://api.example", session=session, sleep=sleeps.append, clock=lambda: 0.0,
                  min_interval=0.0, **kw)
    return c, sleeps


def test_get_retries_rate_limits_with_backoff_then_returns_json():
    s = FakeSession(response(429), response(429), response(200, b'{"ok": 1}'))
    c, sleeps = client(s)
    assert c.get("/events", series_ticker="KXBTCD") == {"ok": 1}
    assert len(s.calls) == 3
    assert sleeps == [1, 2]
    assert s.calls[0] == ("https://api.example/events", {"series_ticker": "KXBTCD"})


def test_get_returns_json_arrays():
    # Coinbase candles come back as a bare list
    c, _ = client(FakeSession(response(200, b"[[1, 2]]")))
    assert c.get("/candles") == [[1, 2]]


def test_get_retries_server_errors_and_dropped_connections():
    s = FakeSession(response(503), requests.ConnectionError("reset"), response(200, b'{"ok": 1}'))
    c, _ = client(s)
    assert c.get("/x") == {"ok": 1}
    assert len(s.calls) == 3


def test_get_fails_fast_on_client_errors():
    s = FakeSession(response(404), response(200))
    c, _ = client(s)
    with pytest.raises(requests.HTTPError):
        c.get("/x")
    assert len(s.calls) == 1


def test_get_gives_up_after_max_retries():
    s = FakeSession(response(503), response(503), response(503), response(200))
    c, _ = client(s, max_retries=2)
    with pytest.raises(RuntimeError):
        c.get("/x")
    assert len(s.calls) == 3


def test_get_spaces_requests_by_min_interval():
    sleeps = []
    c = fu.Client("https://api.example", session=FakeSession(response(200), response(200)),
                  sleep=sleeps.append, clock=lambda: 100.0, min_interval=0.5)
    c.get("/a")
    c.get("/b")
    assert sleeps == [0.5]


def test_write_parquet_creates_dirs_and_leaves_no_temp_file(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"ts": [1, 2], "close": [83880.37, 83881.0]})
    path = tmp_path / "coinbase" / "2026-09-01_2026-09-30.parquet"
    fu.write_parquet(df, path)
    assert pd.read_parquet(path).equals(df)
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def utc(*args) -> int:
    return int(datetime(*args, tzinfo=timezone.utc).timestamp())


def test_month_windows_clips_first_and_last_month_to_range():
    w = fu.month_windows(date(2025, 8, 27), date(2026, 10, 1))
    assert len(w) == 15
    assert w[0] == ("2025-08", utc(2025, 8, 27), utc(2025, 8, 31, 23, 59))
    assert w[-1] == ("2026-10", utc(2026, 10, 1), utc(2026, 10, 1, 23, 59))


def test_month_windows_full_month_covers_every_minute_start():
    w = dict((label, (s, e)) for label, s, e in fu.month_windows(date(2025, 8, 27), date(2026, 10, 1)))
    assert w["2026-02"] == (utc(2026, 2, 1), utc(2026, 2, 28, 23, 59))


def test_month_windows_single_day():
    assert fu.month_windows(date(2026, 3, 8), date(2026, 3, 8)) == [("2026-03", utc(2026, 3, 8), utc(2026, 3, 8, 23, 59))]
