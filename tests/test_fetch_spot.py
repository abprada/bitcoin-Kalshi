import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "fetch_spot", Path(__file__).resolve().parents[1] / "00_fetch_spot.py"
)
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)


def test_coinbase_rows_maps_low_high_open_close_order_and_stamps_minute_end():
    # GET /products/BTC-USD/candles: [time, low, high, open, close, volume], time = minute start
    raw = [[1790369940, 83865.21, 83885.06, 83874.12, 83867.63, 0.7102114]]
    assert fs.coinbase_rows(raw) == [{
        "ts": 1790370000,
        "open": 83874.12, "high": 83885.06, "low": 83865.21, "close": 83867.63, "volume": 0.7102114,
    }]


def test_coinbase_rows_sorts_newest_first_input_ascending():
    raw = [[1790369940, 83865.21, 83885.06, 83874.12, 83867.63, 0.7102114],
           [1790369880, 83845.73, 83874.13, 83848.3, 83874.12, 0.7461523]]
    assert [r["ts"] for r in fs.coinbase_rows(raw)] == [1790369940, 1790370000]


def test_bitstamp_rows_parses_strings_and_stamps_minute_end():
    # GET /api/v2/ohlc/btcusd/: timestamp = minute start, all values strings
    raw = [{"timestamp": "1790310060", "open": "84099.30", "high": "84120.53", "low": "84074.47",
            "close": "84120.53", "volume": "0.89317423"}]
    assert fs.bitstamp_rows(raw) == [{
        "ts": 1790310120,
        "open": 84099.30, "high": 84120.53, "low": 84074.47, "close": 84120.53, "volume": 0.89317423,
    }]
