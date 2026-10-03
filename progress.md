# Progress

*As of 2026-10-02*

## The idea in one paragraph

[Kalshi](https://kalshi.com) is a website where people trade on yes/no questions. Every week it lists questions like *"Will Bitcoin be above $84,000 at 5pm ET this Friday?"*, one for each price level ("strike") in steps of $500. Each contract pays $1 if the answer is yes and $0 if no, so its price is the crowd's probability: a contract trading at 30¢ means "about a 30% chance". We turn these prices into **implied volatility**, the standard number traders use to say how much they expect a price to move. Then we map how it changes with the strike and with the time left until Friday. That map is the **implied-volatility surface**.

## Key words

| Word | Meaning |
|---|---|
| Strike | The price level in the question, e.g. $84,000 in "above $84,000?" |
| Spot | Bitcoin's price right now on exchanges (we use the average of Coinbase and Bitstamp) |
| Volatility | How much the price jumps around, stated per year. 40% means a typical move of about ±40% over a year, which works out to about ±5.5% over one week |
| Realized volatility | Volatility measured from how much Bitcoin actually moved |
| Implied volatility | The volatility that makes the standard pricing formula (Black-Scholes) match the market's price. It is the market's *expectation* of future moves |
| Moneyness | Strike ÷ spot. 1.00 is a strike at today's price; 1.05 is 5% above it |
| Smile | Implied volatility across strikes at one moment. It is usually lowest near today's price and higher far away, so it looks like a smile |
| Basis point (bp) | 0.01%. Used for small differences |

## How the project is organised

Run the numbered scripts in order. Each one reads the previous one's output.

| Step | File | What it does | Output |
|---|---|---|---|
| 0 | `00_fetch_data.py` | Downloads Kalshi's weekly Bitcoin contracts: minute-by-minute bid and ask for every strike, and how each one settled | `data/raw/` |
| 0 | `00_fetch_spot.py` | Downloads Bitcoin's price every minute from Coinbase and Bitstamp | `data/raw/` |
| 0 | `00_fetch_brti.py` | Downloads the index Kalshi uses to settle the contracts (BRTI) | `data/raw/` |
| 1 | `01_data_exploration.py` | Notebook that checks the data: gaps, odd prices, missing values | charts only |
| 2 | `02_clean_data.py` | Applies the cleaning rules below and saves a clean dataset (about 25 seconds) | `data/clean/` |
| 3 | `03_iv_surface.py` | Turns the clean prices into implied volatilities and draws the surface (about 2 minutes) | `data/derived/`, `outputs/` |

Other folders:
- `lib/`: the shared code (downloading, cleaning rules, volatility maths).
- `tests/`: 112 automatic checks that the code does what it should. Run with `pytest`.
- `outputs/`: the figures (saved in git).
- `docs/iv_surface_method.md`: a one-page note on the method for the team.
- `data/`: the data. `raw/`, `clean/` and `derived/` are not in git, because they can be rebuilt with the scripts. The other files in `data/` are a collaborator's (Deribit and IBIT options, Kalshi hourly data) and are kept as they are.
- `Literature/`: reference papers.

The notebooks are [marimo](https://marimo.io) files: open one with `marimo edit 03_iv_surface.py`, or run it as a plain script with `python 03_iv_surface.py`.

## The data

- **51 weekly contracts**, October 2025 to September 2026, each living one week (Friday 5pm to Friday 5pm ET), with about 54 strikes each (2,736 in total).
- **Bitcoin spot every minute.** Coinbase and Bitstamp agree with Kalshi's settlement index to within about 0.015%.

What the checks found (`01`):
- The data is consistent: every contract settled the way its strike and the settlement price say it should.
- Kalshi only records a new row when a quote changes, so a quote stays valid until the next row.
- Kalshi closes every Thursday 3–5am ET, so quotes during that window are stale.
- Some quotes are placeholders, such as "buy at 1¢, sell at 99¢". They tell us nothing.
- Trading got much busier over the year, with a jump in January 2026.

## Cleaning rules (`02`)

1. Fill the one missing settlement price (Oct 17, 2025) from the settlement index.
2. Flag the 2 weeks where Bitcoin finished outside all the strikes offered.
3. Spot = the average of Coinbase and Bitstamp.
4. Carry each strike's last quote forward minute by minute, and record how old it is.
5. A bid of 0 or an ask of $1 means "no quote".
6. Only use the middle price (halfway between bid and ask) when the gap between them is 20¢ or less.
7. Mark Thursday 3–5am ET as closed.

Result: 26.4 million strike-minutes, of which 14.6 million have a usable price.

## How we get the implied volatility (`03`)

Every 15 minutes we take the current ladder of contract prices and do the following.

**1. Fix the ladder.** "Above $86k" can never be more likely than "above $84k", so prices must fall as the strike rises. The few that don't get smoothed out.

**2. Add up the ladder to price ordinary options.** A *call* option with strike $84,000 pays $1 for every dollar Bitcoin ends above $84,000. That is the same as owning one "$1 if above $84,000" contract, plus one "$1 if above $84,001", and so on. So the call's price is the sum of the contract prices above $84,000. Below the current price we do the same for *put* options, which pay for every dollar Bitcoin ends below the strike.

**3. Handle the ends of the ladder (the "tails").** The usable ladder stops somewhere, and about 5–6% of the probability lies beyond its last strike. Kalshi tells us *how much* probability is out there, but not *how far out*, and that distance changes the option prices.
- We assume the standard bell-curve shape (Black-Scholes), using the volatility of the last contract on the ladder.
- We then check it against Kalshi's other strikes further out. Their prices are too thin for the ladder, but they still set a floor (bids) and a ceiling (asks). If the assumed tail breaks those limits, we cut it back to them. This happens 9% of the time, almost always making the tail bigger: Kalshi's traders expect big moves more than the bell curve does.
- **Only Kalshi prices go in, never Bitcoin's own volatility.**
- Leaving the tails out entirely would make every volatility too low (by about 1.5–2 points at today's price and 5–7 points far away), because it ignores the chance of big moves.

**4. Find Kalshi's expected Friday price.** The ladder implies an average Friday price. It sits about 0.05% above the exchange price, which is about what one week of interest at Kalshi's 3.5% rate adds. We use it instead of the exchange price because it is what Kalshi's traders are actually pricing.

**5. Solve for volatility.** The Black-Scholes formula turns (Friday price, strike, time left, volatility) into an option price. We know the option price, so we solve backwards for the volatility. That gives one implied volatility per strike.

**Quality check ("clean").** A snapshot counts as clean if:
- we can compute a volatility at today's price;
- strikes are close enough together compared with the expected move (at least 2 strikes per typical move);
- Kalshi's expected Friday price is within 0.25% of the exchange price.

82% of snapshots pass.

**Why not solve each contract directly?** We tried this first ("method 1"). Near today's price a contract's price barely depends on volatility: 1¢ of noise changes the answer by about 17 points, and for 1 in 3 contracts there is no answer at all. Even averaged over thousands of snapshots it comes out 6–11 points too low. That is not noise: it is a systematic error from the smile's shape getting mixed in. Adding up the ladder first ("method 2", what we use) avoids both problems.

**Making the surface.** To draw a smooth surface we:
- use only contracts priced between 5¢ and 95¢ (at 2¢ one tick is half the price);
- cap extreme volatilities at the 1st and 99th percentile;
- average the *shape* of the smile over many snapshots, then scale it by the typical volatility at that time to Friday.

## Results

- **32,377 smiles** (one every 15 minutes for each weekly) and **878,470 strike-level volatilities**.
- **Typical volatility at today's price is 42%.** It rises as Friday approaches: about 39% with 5–7 days left and about 60% in the last 12 hours.
- **Far strikes are more expensive** (the smile shape), especially close to Friday. Downside strikes are more expensive than upside ones: traders pay more to protect against crashes.
- **Compared with Bitcoin's actual moves:**
  - day by day, Kalshi's volatility averages 45.5% against 41.0% realized, with a correlation of 0.73;
  - measured against how much Bitcoin moved over the rest of each contract's life, it is about the same in a typical week.
- **Near Friday the surface has no far strikes.** By then those contracts are almost certain (1¢ or 99¢) and carry no information about volatility.

## Figures (`outputs/`)

- `figures/iv_surface_heatmap.png`: volatility by moneyness and time to Friday
- `figures/atm_term_structure.png`: volatility at today's price by hours left
- `figures/atm_vs_realized.png`: Kalshi's volatility against Bitcoin's realized volatility, day by day
- `figures/smile_<weekly>_72h.png`, `_24h`, `_6h`: example smiles for the latest week
- `figures/forward_basis.png`, `figures/clean_by_ttm.png`: quality checks
- 3D surfaces you can rotate in a browser: `iv_surface_3d.html` (full week), `iv_surface_3d_1d7d.html` (1–7 days), `iv_surface_3d_1d.html` (last 24 hours)

## Known limits

- In the last 6 hours the $500 strike spacing is too coarse compared with how little Bitcoin can still move, so those volatilities are rough.
- October 2025 has few usable strikes, so most of each smile there comes from the tails.
- The tail shape is an assumption, kept within Kalshi's own limits. Kalshi's far-strike quotes only rule out extreme cases.

## Next

1. Decide what the surface is for: comparing with the options market, testing whether Kalshi prices are rational, or forecasting volatility.
2. Compare Kalshi's volatility at today's price with Deribit's, the main Bitcoin options exchange (the collaborator's Deribit data is in `data/`).
