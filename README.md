# Bitcoin: Kalshi and Deribit

Research on Bitcoin binary prediction markets and vanilla options: implied volatility,
settlement conventions, liquidity, and historical trading strategies. The analysis compares
Kalshi BTC contracts with Deribit BTC options using Python and interactive marimo notebooks.

## Results

The October 2026 presentation, *IM_Lab_VF*, finds that Kalshi prices more short-dated
Bitcoin volatility near the money, while Deribit prices more downside volatility.
Trading costs absorb most of the apparent edge. The figures below summarize the
presentation's reported results, rather than a new run of the pipeline.

### Implied volatility

The comparison uses **26,651 clean snapshots across 51 Kalshi weekly contracts**.
Deribit volatility is adjusted to Kalshi's later expiry using Bitcoin's hour-of-week
variance profile. A volatility point means one percentage point of annualized volatility.

- **Kalshi's ATM premium grows near expiry:** approximately **+0.2 volatility points**
  one week out, **+3 points** with one day left, and **+1.2 points** overall.
- **Deribit has a richer downside smile:** implied volatility is **4.6–6.7 points higher**
  at strikes more than two standard deviations below the forward.
- **Large gaps narrow within hours:** a gap of at least **5.8 points** narrows by roughly
  **1 point over two hours**, beyond normal drift. Both markets adjust; the analysis
  finds no clear lead by either venue.

See the [ATM spread by maturity](outputs/presentation/4b_atm_spread_by_maturity.png),
[smile comparison](outputs/presentation/5a_smile.png), and
[volatility surfaces](outputs/presentation/6e_surfaces_both_3d.html).

### Trading strategies

**Daily spread and calendar trades.** Selling the Kalshi daily strip against a
strike-matched Deribit iron butterfly earns a small gross premium. Adding a weekly leg
does not improve the result. The corrected backtests in slides 16–17 report:

| Strategy | Before costs | Kalshi maker scenario | Taker scenario |
| --- | ---: | ---: | ---: |
| Daily spread only | +8.0 bps | −9.5 bps | −23.2 bps |
| Calendar, weekly leg closed with daily | +7.5 bps | −23.3 bps | −64.8 bps |
| Calendar, weekly leg held to expiry | +2.5 bps | −23.1 bps | −52.6 bps |

Values are the presentation's mean basis-point results per daily trade, not annualized
returns. The daily spread's gross 95% confidence interval is **+0.4 to +15.7 bps**;
both calendar intervals include zero. All three variants lose after modeled costs.

**Dislocation trades.** The signal measures the volatility gap relative to its trailing
30-day median for the same contract type and time to expiry. Positions are sized to
$1,000 of vega per volatility point and delta-hedged hourly with the BTC perpetual.
The sample runs from January 15 to October 1, 2026, including a 30-day warm-up.

| Strategy | Trades | Before costs | Maker scenario | Taker scenario |
| --- | ---: | ---: | ---: | ---: |
| A: sell a Deribit weekly straddle when the dislocation is ≤ −6 points | 12 | +12.1 | +10.9 | +6.3 |
| B: sell a Kalshi daily strip when the dislocation is ≥ +3 points | 106 | +6.4 | +6.0 | −2.3 |

Results are mean P&L per trade in volatility-point equivalents at the stated vega
sizing. Strategy A remains positive as a taker in a small sample; strategy B is positive
only under the maker assumptions.

### Limits of the evidence

These are in-sample historical simulations. Strategy A has only 12 trades, concentrated
mostly in January–April; thresholds were selected using robustness grids. The maker
scenario assumes fills at mid, reduced Kalshi fees, and no hedge cost, making it an
optimistic bound rather than demonstrated execution. Thin liquidity, missing Deribit
order-book quotes, settlement differences, and Black–Scholes assumptions also affect
the comparison. The presentation calls for out-of-sample testing and validation of
actual maker fills before interpreting these results as an executable trading edge.

*Source: IM_Lab_VF.pptx, October 2026, slides 5, 10–12, 16–20, 25 and 30.
Calendar results follow the corrected slide tables; older speaker notes contain
superseded figures.*

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
| `17_lead_lag.py` | Lead–lag analysis and adjustment after dislocations |
| `18_trades.py` | Consolidated trade results before and after costs |
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

Input datasets and caches stay local and are ignored by Git. No input datasets are
included in a fresh checkout. Some analyses depend on externally supplied hourly snapshots
and delivery prices that the fetchers do not reconstruct; these are documented in
[data/README.md](data/README.md). Generated charts and summary tables are tracked under `outputs/`.

The [IV methodology](docs/iv_surface_method.md) explains the estimation procedure.
Historical strategy results depend on quote quality, observation timing, settlement
assumptions, and modeled execution costs; they are research simulations.

## Development

```sh
python -m pytest -q
python -m compileall -q lib tests
```

Keep reusable calculations in `lib/` and presentation code in the notebooks. Tests should
use small synthetic examples and mock network requests. Do not commit input datasets or
credentials. Generated research outputs belong in `outputs/`.

## License

The project is available under the [MIT License](LICENSE).
