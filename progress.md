# Progress

*As of 2026-10-06*

## The idea in one paragraph

[Kalshi](https://kalshi.com) is a website where people trade on yes/no questions. Every week it lists questions like *"Will Bitcoin be above $84,000 at 5pm ET this Friday?"*, one for each price level ("strike") in steps of $500. Each contract pays $1 if the answer is yes and $0 if no, so its price is the crowd's probability: a contract trading at 30¢ means "about a 30% chance". We turn these prices into **implied volatility**, the standard number traders use to say how much they expect a price to move. Then we map how it changes with the strike and with the time left until Friday. That map is the **implied-volatility surface**. Finally we compare Kalshi with [Deribit](https://www.deribit.com), the main Bitcoin options exchange, to see whether the two markets agree (handout question 2b).

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
| Call spread | Buy a call at one strike and sell a call at a higher strike. A narrow one ($500 apart) pays like a Kalshi yes/no contract, which is how Deribit prices are turned into Kalshi prices |
| Mark price | Deribit's own estimate of an option's fair value at a moment. Not a trade; Deribit uses it to value positions |
| Vol point | One percentage point of volatility: 43% to 45% is 2 points |

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
| 4 | `04_deribit_exploration.py` | Notebook that explains and checks the Deribit data, with a first daily comparison | charts only |
| 5 | `05_kalshi_vs_deribit.py` | Compares Kalshi with Deribit, in prices and in volatilities (about 15 seconds) | charts only |
| 6 | `06_results_summary.py` | **Short summary of the results for the supervisors**: question, answer, method, four figures | charts only |

Other folders:
- `lib/`: the shared code (downloading, cleaning rules, volatility maths, Deribit loading, the Kalshi–Deribit comparison).
- `tests/`: 127 automatic checks that the code does what it should. Run with `pytest`.
- `outputs/`: the figures (saved in git).
- `docs/iv_surface_method.md`: a one-page note on the method for the team.
- `data/`: the data. `raw/`, `clean/` and `derived/` are not in git, because they can be rebuilt with the scripts. The other files in `data/` are a collaborator's (Deribit and IBIT options, Kalshi hourly data) and are kept as they are.
- `Literature/`: reference papers.

The notebooks are [marimo](https://marimo.io) files: read one as a page with `marimo run 05_kalshi_vs_deribit.py` (code hidden), edit it with `marimo edit`, or run it as a plain script with `python 05_kalshi_vs_deribit.py`.

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

**Making the surface.** Smile and surface horizontal axes use `(K / current spot − 1) / (snapshot ATM IV × √tau_years)`: zero is current spot, and ±1 is one ATM-implied standard deviation over the remaining life. Surface bins run from −3σ to +3σ in steps of 0.25σ. To draw a smooth surface we:
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
- **Near Friday the surface has no far strikes.** In dollar terms the usable strike range contracts because distant contracts become almost certain (1¢ or 99¢). The standardized plots adjust for that shrinking expected move; sparse cells remain blank.

## Figures (`outputs/`)

- `figures/iv_surface_heatmap.png`: volatility by ATM standard deviations from current spot and time to Friday
- `figures/atm_term_structure.png`: volatility at today's price by hours left
- `figures/atm_vs_realized.png`: Kalshi's volatility against Bitcoin's realized volatility, day by day
- `figures/smile_<weekly>_72h.png`, `_24h`, `_6h`: example smiles for the latest week
- `figures/forward_basis.png`, `figures/clean_by_ttm.png`: quality checks
- 3D surfaces you can rotate in a browser: `iv_surface_3d.html` (full week), `iv_surface_3d_1d7d.html` (1–7 days), `iv_surface_3d_1d.html` (last 24 hours)

## Deribit (`04`)

The collaborator's file has **205,896 Deribit option trades** from January 15 to September 30, 2026 (259 days).

- Deribit options are ordinary calls and puts, paid in Bitcoin. Price in Bitcoin × Bitcoin's price gives a normal dollar price, so the usual formula works.
- Every Deribit option expires at **08:00 UTC**. Deribit lists one expiry every day plus Fridays further ahead, so several are live at once. The file only keeps trades in two windows each day (the hour before 08:00 UTC, and 19:30–22:29 UTC around Kalshi's 5pm ET expiry) and only expiries a few days away.
- A Friday Deribit option ends **13 hours before** the Kalshi weekly with the same date. This matters a lot (below).
- Data issues: 0 and 999 used as placeholder volatilities; 5,927 identical rows (kept, as there is no trade id). Our own volatility from Deribit's prices matches Deribit's to within 0.13 points.
- The IBIT options file covers Nov 2024 – Aug 2025, which **does not overlap** our Kalshi data.

## Kalshi vs Deribit (`05`)

**How.** The two markets sell different things, so we put them in the same units two ways:
1. **Prices:** a Deribit call spread from K − $250 to K + $250, divided by $500, pays like Kalshi's "above K" contract. Its cost is Deribit's price for Kalshi's contract, in cents.
2. **Volatilities:** both markets turned into implied volatility.

Each Kalshi snapshot (every 15 minutes, 19:30–22:29 UTC) is paired with Deribit's mark prices from within 15 minutes, in the Friday expiry with the same date: **2,118 snapshots on 249 days**. Deribit's volatilities are interpolated to Kalshi's strikes (never beyond the strikes Deribit traded).

Which Kalshi snapshots are used: those with a volatility at today's price and enough strikes, as in `03`'s quality check, but **without** the "expected Friday price within 0.25% of the exchange price" rule. That rule dropped 14% of snapshots, mostly 3–7 days before expiry; the large gaps neither follow nor predict Bitcoin's moves, and Kalshi's volatilities are computed against Kalshi's own expected price, so they are not distorted by it. Keeping them changes no result (with the rule: 1,861 snapshots, 1.13 points at today's price; without: 1.09). Bad individual quotes are still removed strike by strike (gap between bid and ask over 20¢, or a quote older than 15 minutes).

**The 13-hour gap.** Deribit has no option ending when Kalshi's does, so we stretch Deribit's volatility over Kalshi's extra hours. Those hours are Friday's US session, which historically moves 1.6 times as much as an average hour of the week, so Deribit's volatility is raised by about 3% of itself (1.03×). Results are shown with and without this step, because it is an assumption.

**Results.**
- **Volatility at today's price:** Kalshi is 2.9 points above Deribit as traded, **1.1 points** after the 13-hour adjustment, higher on about two days in three. (The first daily comparison in `04` found about 4 points; about half of that was the 13 hours.)
- **Shape:** far below today's price, Kalshi is about 3 points *lower* than Deribit. Deribit charges more for crash protection.
- **Prices:** the typical difference is **1.5¢**, about Kalshi's own bid–ask spread (2¢). Kalshi is slightly cheaper at and just below today's price, slightly dearer above it.
- **Money on the table?** Deribit's price is outside Kalshi's bid–ask 60% of the time. After Kalshi's fee an edge is left on 30% of quotes, typically 1.4¢. Copying a Kalshi contract on Deribit costs two option trades; with an assumed 5¢ of Deribit costs only 3% of quotes still show an edge, mostly in the last two days, where the Deribit leg expires 13 hours early (risky, not free money).
- **Without the 13-hour adjustment** Kalshi looks about 3 vol points too high at today's price, and its prices look low below today's price and high above it: exactly what 13 extra hours of possible movement produce. The adjustment removes most of this.
- **Check without any model:** Deribit's own dollar prices at real strikes K − 500 and K + 500 (2,107 cases) give the same picture as the Black-Scholes version before the adjustment, so Black-Scholes is not driving the result.
- **Answer to 2b so far:** the two markets broadly agree once put in the same units and the 13-hour difference is handled; what is left is mostly smaller than trading costs. Possible reasons for the rest: different users (Deribit does not serve US residents), Kalshi's full collateral and fees, its 1¢ tick and $500 strikes.

## Known limits

- In the last 6 hours the $500 strike spacing is too coarse compared with how little Bitcoin can still move, so those volatilities are rough.
- October 2025 has few usable strikes, so most of each smile there comes from the tails.
- The tail shape is an assumption, kept within Kalshi's own limits. Kalshi's far-strike quotes only rule out extreme cases.
- The 13-hour adjustment uses Bitcoin's past hour-by-hour movement, not what Deribit's prices say about that Friday.
- We have Deribit trades and marks, not its bid and ask, so Deribit's real trading costs are assumed, not measured.
- Kalshi's fee formula in `lib/compare.py` (7% × P × (1 − P), rounded up to the cent) still needs checking against Kalshi's current fee schedule.

## Next

1. Handle the 13-hour gap with market prices instead of history: interpolate between Deribit's Friday 08:00 and Saturday 08:00 UTC expiries, which bracket Kalshi's 5pm ET.
2. Robustness of `05`: matching window length, trade prices instead of marks, the morning window.
3. Get Deribit bid and ask quotes, to measure its trading costs properly.
4. Handout question 1: write up the theory (Kalshi contract logistics, Black-Scholes value N(d₂), the call-spread link).
5. Handout question 3: check whether Kalshi's probabilities come true as often as they say (calibration against how contracts settled), and compare with Deribit's.
