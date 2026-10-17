# Local data

Data files are not distributed with the source. Run fetchers from the repository root;
each script documents its schema, date defaults, and resume behavior. Public endpoint
availability can change, and a successful unit test does not verify historical API coverage.

| Path | Producer / requirement |
| --- | --- |
| `raw/kalshi_btc_weekly/` | `00_fetch_data.py` |
| `raw/kalshi_btc_daily/` | `00_fetch_data.py --product daily` |
| `raw/kalshi_events/` | `00_fetch_kalshi_event.py EVENT_TICKER` |
| `raw/btc_spot/` | `00_fetch_spot.py` |
| `raw/brti/` | Optional settlement index history from `00_fetch_brti.py` |
| `raw/deribit_trades/` | All expiries and full-day trades from `00_fetch_deribit_all.py` |
| `raw/deribit/` | Short-dated trades, perpetual bars, and funding from `00_fetch_deribit.py`; use `--no-options` if all-expiry trades exist |
| `clean/` | `02_clean_data.py` |
| `derived/` | IV/comparison notebooks and `08_strategy_data.py` |

## External inputs

These files were supplied separately for the original research. They are not generated
by the current pipeline. Obtain suitable data and permission to use it independently.

- `deribit_btc_options.csv.gz`: optional, restricted-window trade sample used as a fallback
  when `raw/deribit_trades/btc_options_trades.parquet` is absent. Prefer the full-day fetcher.
  Required fields are defined by `lib.deribit.load_trades`.
- `kalshi_btc_prices.csv.gz`: hourly snapshots used by `lib.kalshi_hourly` and the strategy
  pipeline. Fields include `product`, `event_ticker`, `strike`, `time_utc`,
  `settle_time_utc`, `yes_bid`, and `yes_ask`. The fetchers' minute candles are not a drop-in
  replacement for this file.
- `kalshi_btc_contracts.csv.gz`: contract metadata used by the strategy pipeline, including
  `event_ticker`, `strike`, `result`, and `settlement_value`.
- `deribit_delivery_prices.csv`: official settlement prices with `date` and `delivery_price`
  columns, read by `lib.deribit_marks.delivery_prices`. Earlier missing dates are filled
  using the trade-index averaging method described in that function.

The former WRDS/IBIT options input is unused by the current Python pipeline and is excluded.
Raw data and intermediate datasets remain local; charts and summary tables in `outputs/` are tracked.
Removing a file from the current tree
does not erase it from earlier Git commits.
