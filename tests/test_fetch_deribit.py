import importlib

import pandas as pd

fetch = importlib.import_module("00_fetch_deribit")


def test_api_rows_become_the_collaborators_columns():
    raw = pd.DataFrame({"timestamp": [1773100800000 + 7 * 3_600_000 + 5_000, 1773100800000 + 20 * 3_600_000],
                        "instrument_name": ["BTC-13MAR26-70000-C", "BTC-27MAR26-60000-P"], "price": [0.01, 0.02],
                        "index_price": [70_000.0, 68_000.0], "amount": [1.0, 0.5], "direction": ["buy", "sell"],
                        "iv": [50.0, 60.0], "mark_price": [0.011, 0.019], "trade_id": ["1", "2"]})
    d = fetch.to_collaborator_columns(raw)
    assert list(d.window) == ["exact_0800UTC", "us_afternoon"]
    assert d.expiry_utc.iloc[0] == "2026-03-13T08:00:00+00:00" and d.strike.iloc[1] == 60_000 and d.cp.iloc[1] == "P"
    assert d.price_usd.iloc[0] == 700.0 and d.mark_price_btc.iloc[1] == 0.019
