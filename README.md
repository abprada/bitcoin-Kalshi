# Bitcoin: Kalshi and Deribit

Research on Bitcoin binary prediction markets and vanilla options: implied volatility,
settlement conventions, liquidity, and historical trading strategies. The analysis compares
Kalshi BTC contracts with Deribit BTC options using Python and interactive marimo notebooks.

## Setup

Use Python 3.12 and run commands from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

For analysis only, install `requirements.txt`. Tests use synthetic data and mocked HTTP
responses; they do not require downloaded datasets or API credentials.

## Workflow

The numbered files are the research pipeline. Keep them at the repository root because
notebooks resolve data paths relative to their own location. Fetchers use public endpoints
and resume from files already on disk. Check `--help` to select dates before downloading;
defaults cover the original research period and can produce large datasets.

```sh
python 00_fetch_data.py --limit 2
python 00_fetch_spot.py
python 00_fetch_deribit_all.py
marimo edit 02_clean_data.py
marimo edit 03_iv_surface.py
marimo edit 05_kalshi_vs_deribit.py
```

Run the cleaning notebook before the IV notebook, then the comparison notebook before
opening the results. The two-event fetch above is a small starting sample; omit `--limit`
for the full weekly history. Different date ranges and samples will change the results.
The fetchers' date arguments differ: Kalshi filters expiry dates; spot and Deribit filter
observation dates. See [data requirements](data/README.md) for the additional inputs needed
by the strategy notebooks.

| Files | Purpose |
| --- | --- |
| `00_fetch_*.py` | Download Kalshi quotes/settlements, spot, and Deribit trades |
| `01_data_exploration.py` | Inspect raw Kalshi and spot data |
| `02_clean_data.py` | Clean quotes and align spot observations |
| `03_iv_surface.py` | Estimate binary implied-volatility surfaces |
| `04_deribit_exploration.py`–`06_results_summary.py` | Explore and compare the two venues |
| `08_strategy_data.py` | Build strategy inputs and run historical simulations |
| `09_vol_spread.py`–`13_binary_hedging.py` | Inspect strategy results and hedging mechanics |
| `14_backtests.py` | Additional backtests |
| `15_deribit_iv_surface.py` | Deribit volatility surfaces |
| `16_presentation_figures.py` | Presentation charts |
| `17_settlement_factor.py` | Model settlement-window and time-of-week adjustments |
| `liquidity_summary.py` | Generate liquidity summaries |
| `lib/` | Shared pricing, cleaning, fetching, and strategy functions |
| `tests/` | Automated tests |

Numbering retains the research sequence; there is no step 07. Open a notebook with
`marimo edit FILE.py`, or run it with `python FILE.py`. Build strategy inputs with
`python 08_strategy_data.py` before running notebooks 09–13. Fetch the perpetual/funding
inputs with `python 00_fetch_deribit.py --no-options` when full option trades are already
available. The single-event hedging example also needs
`python 00_fetch_kalshi_event.py KXBTCD-25OCT0917`.

## Data and outputs

Datasets, caches, and generated charts stay local and are ignored by Git. No datasets are
included in a fresh checkout. Some analyses depend on externally supplied hourly snapshots
and delivery prices that the fetchers do not reconstruct; these are documented in
[data/README.md](data/README.md). Generated charts are written under `outputs/`.

The [IV methodology](docs/iv_surface_method.md) explains the estimation procedure.
Historical strategy results depend on quote quality, observation timing, settlement
assumptions, and modeled execution costs; they are research simulations.

## Development

```sh
python -m pytest -q
python -m compileall -q lib tests
```

Keep reusable calculations in `lib/` and presentation code in the notebooks. Tests should
use small synthetic examples and mock network requests. Do not commit datasets, credentials,
or generated exports.
