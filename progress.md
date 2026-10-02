# Progress

*As of 2026-10-02*

## Goal
Price Kalshi's weekly Bitcoin contracts ("BTC above $K at Friday 5pm ET") with Black-Scholes and build implied-volatility surfaces.

## Data collected (`data/raw/`, not in git)

| Dataset | Script | What it is |
|---|---|---|
| Kalshi weekly contracts | `00_fetch_data.py` | 51 weekly contracts (Oct 2025 – Sep 2026), 2,736 strikes: 1-minute bid/ask per strike, and how each strike resolved |
| BTC spot | `00_fetch_spot.py` | 1-minute BTC/USD prices from Coinbase and Bitstamp |
| Settlement index (BRTI) | `00_fetch_brti.py` | The index the contracts settle on, minute by minute |

Shared code lives in `lib/` (download helpers, cleaning rules, implied-vol math), covered by 100 tests in `tests/`. A collaborator's files in `data/` (Deribit options, WRDS IBIT options, Kalshi hourly data) are kept as they are; their Kalshi prices match ours exactly where the two overlap.

## What we learned (`01_data_exploration.py`)
- The data is consistent: no duplicates or broken prices, and every strike's result matches its settlement value.
- Kalshi only records a 1-minute candle when a quote changes. Forward-filling the last quote is safe (checked on every gap).
- Kalshi closes every Thursday 03:00–05:00 ET. Quotes carried through the closure are stale.
- Some "two-sided" quotes are placeholders (e.g. bid 0.01 / ask 0.99), so a usable mid needs a spread cap.
- Liquidity grew a lot over the year, with a jump in January 2026.
- Exchange spot agrees with the settlement index to about 1.5 bps. It is the right spot price for pricing, because the settlement index is a 60-second average that lags the market.
- In 2 of the 51 weeks, BTC settled outside the range of strikes offered.

## Clean dataset (`02_clean_data.py`, written to `data/clean/`)
Rules (documented in the notebook, implemented and tested in `lib/cleaning.py`):
1. Fill the one missing settlement value (Oct 17, 2025) from the settlement index.
2. Flag the 2 weeks where BTC settled outside the strikes offered.
3. Spot = average of Coinbase and Bitstamp (Bitstamp alone when Coinbase is missing).
4. Forward-fill each strike's last quote, minute by minute, and record how old it is.
5. Treat a bid of 0 or an ask of 1 as "no quote".
6. Use a mid price only when the spread is 20¢ or less.
7. Flag Thursday 03:00–05:00 ET as exchange closed.
8. Mark as `usable` the quotes with a mid while the exchange is open.

Result: 26.4M strike-minutes across 51 weeklies, of which 14.6M are usable (1.6M within 1% of spot). Rebuild with `python 02_clean_data.py` (about 25 seconds).

## Implied-volatility surface (`03_iv_surface.py`, written to `data/derived/`)
Each Kalshi strike is a binary call. Inverting each binary price directly is unreliable near the money, so instead:
1. Every 15 minutes, take each weekly's ladder of usable binary prices and make it monotone across strikes.
2. Integrate the ladder into call prices (strikes above the forward) and put prices (below), with Black-Scholes tails beyond the outer strikes.
3. Take the forward from put-call parity, and invert each option for an ordinary Black-Scholes implied vol: one vol per strike.

The method is in `lib/iv.py`, with tests. Result: 32,377 smiles and 878,470 strike-level vols; 82% of snapshots pass the quality checks (fewer within 6 hours of expiry, where the $500 strike spacing becomes coarse). Median at-the-money vol is 42%, in line with BTC's realized vol, rising from about 40% a week out to about 50% in the last hours. Smiles have expensive wings, steepest near expiry, with puts richer than calls.

## Figures (`outputs/`, in git)
Written by `03_iv_surface.py` together with its data. Charts use **moneyness = strike / spot**.
- `figures/iv_surface_heatmap.png`: median implied vol by moneyness and time to expiry
- `figures/atm_term_structure.png`: at-the-money vol by hours to expiry
- `figures/atm_vs_realized.png`: Kalshi at-the-money vol vs BTC realized vol, daily
- `figures/forward_basis.png` and `figures/clean_by_ttm.png`: quality checks
- `figures/smile_<weekly>_72h.png`, `_24h`, `_6h`: example smiles for the latest weekly
- `iv_surface_3d.html`: the surface in 3D (moneyness × hours to expiry × implied vol); open it in a browser to rotate and zoom

## Next
1. Decide what the surface is used for (comparison with option markets, testing Kalshi's pricing, or forecasting volatility).
