# Progress

*As of 2026-10-12*

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
| 0 | `00_fetch_deribit.py` | Downloads every Deribit BTC option trade (expiries up to 15 days), the BTC perpetual every minute, and funding (about 50 minutes) | `data/raw/deribit/` |
| 1 | `01_data_exploration.py` | Notebook that checks the data: gaps, odd prices, missing values | charts only |
| 2 | `02_clean_data.py` | Applies the cleaning rules below and saves a clean dataset (about 25 seconds) | `data/clean/` |
| 3 | `03_iv_surface.py` | Turns the clean prices into implied volatilities and draws the surface (about 2 minutes) | `data/derived/`, `outputs/` |
| 4 | `04_deribit_exploration.py` | Notebook that explains and checks the Deribit data, with a first daily comparison | charts only |
| 5 | `05_kalshi_vs_deribit.py` | Compares Kalshi's and Deribit's implied volatilities, trade by trade (about 2 minutes) | charts only |
| 6 | `06_results_summary.py` | **Short summary of the results for the supervisors**: question, answer, method, four figures | charts only |
| 8 | `08_strategy_data.py` | Builds the inputs of the two trading backtests and runs them (about 20 minutes) | `data/derived/strategies/` |
| 9 | `09_vol_spread.py` | Strategy 1: Kalshi vs Deribit vol, hour by hour; who leads; a long/short trade on the gap | `outputs/figures/09_*` |
| 10 | `10_calendar_straddle.py` | Strategy 2: long the 7-day Deribit straddle, short the 1-day one, delta-hedged | `outputs/figures/10_*` |
| 0 | `00_fetch_kalshi_event.py` | Downloads 1-minute quotes for any named Kalshi event (e.g. the 9 Oct 2025 daily) | `data/raw/kalshi_events/` |
| 12 | `12_strategies_ab.py` | The two dislocation strategies (A, B) as portfolios: execution variants, robustness, out of sample, and the dislocation as a signal | `outputs/figures/12_*` |
| 13 | `13_binary_hedging.py` | Hedging a long Kalshi binary with a short call spread: Greeks through 9 Oct 2025, P&L with half-hourly hedges, and every 2026 daily | `outputs/figures/13_*` |
| 11 | `11_dislocations.py` | Large Kalshi-vs-Deribit dislocations: who adjusts, mean-reversion trades, and which market predicts BTC better | `outputs/figures/11_*` |

Other folders:
- `lib/`: the shared code (downloading, cleaning rules, volatility maths, Deribit loading, the Kalshi–Deribit comparison in `pairs.py`; `compare.py` holds the earlier snapshot method and the call-spread prices, no longer shown in the notebooks).
- `tests/`: automatic checks that the code does what it should. Run with `pytest`.
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

Kalshi vs Deribit (written by running `python 05_kalshi_vs_deribit.py` and `python 06_results_summary.py`):
- `figures/05_13h_hour_profile.png`: how much Bitcoin usually moves each hour up to Kalshi's expiry, with the 13 extra hours
- `figures/05_vol_gap_across_smile.png`, `figures/05_smile_shape.png`: the gap across the smile and the two average smile shapes
- `figures/05_daily_atm_vol.png`: volatility at today's price, day by day, both markets
- `figures/05_iv_surfaces.png`, `figures/05_gap_surface.png`: the two surfaces and their gap, by days to expiry and distance from today's price
- `figures/05_far_from_expiry_gap.png`: the gap across the smile with 4–7 days left, with and without the 13-hour adjustment
- `figures/06_fig1_daily_atm_vol.png` to `figures/06_fig4_gap_surface.png`: the four figures of the results summary
- `05_iv_surfaces_3d.html`: both surfaces in 3D
- `figures/smile_in_sd.png`, `figures/trades_A_one_afternoon.png` to `trades_D_per_day.png`: the average smile from the pairs, and when Deribit trades compared with Kalshi (one-off figures, not written by a notebook)

## Deribit (`04`)

The collaborator's file has **205,896 Deribit option trades** from January 15 to September 30, 2026 (259 days).

- Deribit options are ordinary calls and puts, paid in Bitcoin. Price in Bitcoin × Bitcoin's price gives a normal dollar price, so the usual formula works.
- Every Deribit option expires at **08:00 UTC**. Deribit lists one expiry every day plus Fridays further ahead, so several are live at once. The file only keeps trades in two windows each day (the hour before 08:00 UTC, and 19:30–22:29 UTC around Kalshi's 5pm ET expiry) and only expiries a few days away.
- A Friday Deribit option ends **13 hours before** the Kalshi weekly with the same date. This matters a lot (below).
- Data issues: 0 and 999 used as placeholder volatilities; 5,927 identical rows (kept, as there is no trade id). Our own volatility from Deribit's prices matches Deribit's to within 0.13 points.
- The IBIT options file covers Nov 2024 – Aug 2025, which **does not overlap** our Kalshi data.

## Kalshi vs Deribit (`05`)

**The timing problem.** Kalshi has every strike's bid and ask every minute. Our Deribit data are only trades, between 19:30 and 22:29 UTC, at whatever strikes people happen to trade. A comparison is only fair if both prices are from the same moment.

**Volatilities: trade by trade (`lib/pairs.py`).** Each Deribit trade is the anchor, and Kalshi is read at that moment:
1. **Deribit's volatility** from the trade's mark price (Deribit's fair value at that instant; the traded price lands on the bid or the ask), using Deribit's Bitcoin index and the time left to Deribit's expiry. Pieces of one order (same strike, same minute) count once.
2. **Kalshi's volatility** at the same strike, from Kalshi's ladder at the nearest minute (**at most 30 seconds** from the trade), built exactly as in `03`. The strike is on Kalshi's ladder 3 times in 4; otherwise it is read between the two neighbouring Kalshi strikes. Nothing is interpolated on the Deribit side.
3. **The 13-hour gap** (below), then **gap = Kalshi − Deribit**, pooled by distance from today's price in standard deviations and by time to expiry. The 95% intervals resample whole days, because pairs on the same day are not independent.

This gives **15,142 pairs on 251 days** (37 weeklies), a median of 44 per afternoon. Deribit only trades a few strikes at a time, so the result is the **average smile over the year**, not a full Deribit smile at any one moment. Kalshi's ladder must pass `03`'s checks (fresh quotes, bid–ask gap at most 20¢, at least 5 strikes around today's price, a volatility at today's price, at least 2 strikes per standard deviation), but **not** the rule that its expected Friday price be within 0.25% of the exchange price: that rule dropped 14% of snapshots, mostly 3–7 days before expiry, with gaps that neither follow nor predict Bitcoin's moves, and Kalshi's volatilities are computed against Kalshi's own expected price anyway.

**Only volatilities.** The notebooks compare implied volatilities only. An earlier price comparison in cents (Deribit call spreads against Kalshi's contracts, typical gap 1.5¢, about Kalshi's bid–ask spread) has been dropped from them; its code stays in `lib/compare.py`.

**The 13-hour gap.** Deribit has no option ending when Kalshi's does, so we stretch Deribit's volatility over Kalshi's extra hours. Those hours are Friday's US session, which historically moves 1.6 times as much as an average hour of the week, so Deribit's volatility is raised by 2–14% of itself (×1.02 a week out, up to ×1.14 a day out). Results are shown with and without this step, because it is an assumption.

**Results.**
- **Volatility at today's price:** Kalshi is 3.8 points above Deribit as traded, **1.8 points** after the 13-hour adjustment (95% interval 1.3 to 2.2), higher in 74% of pairs and on 76% of days. Day by day the two move together (correlation 0.95).
- **The adjustment does not remove everything:** after it, the gap at today's price still grows from +0.4 points a week out to +3.0 with 1–2 days left. Either Kalshi really prices the last days higher, or the 13 hours carry more movement than history gives them.
- **Shape:** Deribit's smile is steeper. Two standard deviations below today's price Kalshi is 5.4 points *lower* than Deribit (Deribit charges more for crash protection); two above, 1.2 lower, not clearly different from zero.
- **Compared with the earlier snapshot method** (1.1 points at today's price): the snapshot method drew straight lines between the few strikes Deribit traded in 30 minutes. Deribit's smile is curved, so those lines overstated Deribit's volatility at today's price and understated the gap. Timing itself made little difference.
- **Far from expiry (4–7 days left, 5,789 pairs on 144 days),** the 13 hours are about 11% of the horizon and the adjustment only ×1.02–1.03. There Kalshi is **+1.8 points** above Deribit at today's price with no adjustment (95% interval 1.3 to 2.2) and **+0.7** with it (0.3 to 1.2): higher either way. Deribit's steeper downside holds even with no adjustment (−2.9 points two standard deviations below). Across three assumptions about how busy the extra hours are, the result a week out moves by 2.6 points, against 8.8 one to two days out, so this is the comparison that least depends on the adjustment.
- **Answer to 2b so far:** in implied volatility the two markets broadly agree once compared at the same moment and the 13-hour difference is handled. Kalshi is a little richer at today's price and flatter in the wings. Possible reasons: different users (Deribit does not serve US residents), Kalshi's full collateral and fees, its 1¢ tick and $500 strikes.

## Two trading strategies (`08`, `09`, `10`)

**New data.** The collaborator's Deribit file only has two windows a day, so it never follows an option through
the day. `00_fetch_deribit.py` downloads every Deribit BTC option trade from Deribit's public history: 2.7 million
trades, October 2025 to September 2026, about 13 times the collaborator's file. Every one of the collaborator's
trades (in these expiries) is in it with the same mark and vol. Every trade carries Deribit's mark, so the trades in
one expiry over an hour trace Deribit's smile, and any option can be valued at any hour (`lib/deribit_marks.py`).
Deribit's bid-ask spread is measured as the gap between what takers paid and the mark: about $5–8 per
at-the-money option. The collaborator's hourly Kalshi file is now turned into vols too, dailies included
(`lib/kalshi_hourly.py`, same method as `03`). The 13-hour gap is handled with Deribit's own prices when Deribit
has an expiry on each side of Kalshi's (41% of hours: "Next 1" below), otherwise with `05`'s stretch.

**Strategy 1: Kalshi vs Deribit vol (`09`).**
- Kalshi is richer than Deribit at the same expiry: about +1 vol point with 4–7 days left, +2.5–3 with 1–4 days,
  +3–4 for the 24-hour dailies, more in the last hours. This reproduces `05` (+1.9 in its afternoon window) and the
  growth toward expiry also appears where Deribit prices both sides of Kalshi's expiry, so it is not the stretch.
- Kalshi does jump higher when vol jumps: it follows Deribit's vol up about twice as strongly as down, and after a
  vol spike the gap widens and stays wide for a day.
- Neither market leads by 15 minutes or more; hour by hour they move together. When they drift apart, Kalshi does
  most of the moving back.
- Kalshi's extra vol is partly right: BTC then moves more than Deribit expected.
- The trade loses after costs in every version (1, 4, 24 hours, to expiry; thresholds; two signals). A Kalshi
  straddle is a strip of 15–30 binaries; crossing their spreads and paying Kalshi's fee costs about 15–19 vol points
  round trip, against gaps of 1–4. Trading only Deribit on Kalshi's signal costs 2–4 points and earns under 1.

**Strategy 2: Deribit calendar straddle (`10`).**
- The "first check" fails: Deribit's daily options carry no short-dated premium. Their vol falls into expiry
  (about 37% with 1–4 days left, 33% in the last 12 hours, which are the quiet night before 08:00 UTC) and is at or
  below what BTC then realizes. Kalshi's rises into its 5pm ET expiry.
- The base case loses about $70 a day before costs and $700 after (Sharpe −1.6 and −16; costs: option fees and
  spreads about $370 a day, hourly hedging about $270). Deribit's fee, 0.03% of BTC per option, is about 4% of a
  daily option's price each way.
- None of 23 variations makes money after costs. The only clear gross profit is selling the overnight daily (enter
  20:00 UTC, hold to 08:00): about +$80–100 a day before costs, also in Oct–Dec 2025, but fees take all of it.
- Jump weeks (early February, early June, late August) and intraday whipsaws are where the short gamma loses.

## Corrections and dislocations (`11`)

**Corrections (from review).**
- *The near-expiry vol spike is mostly a filter.* 05's rule (at least 2 strikes per standard deviation) passes
  near expiry only when vol is high: on the hourly data, weekly snapshots with 0–3 hours left show 96% median ATM
  vol if filtered, 58% if not. `08`–`11` now use every snapshot and keep the rule as a flag (`fine_grid`).
  **03's term structure (the 88% spike), its clean flag and the dark near-expiry rows of its heatmap, and the
  "about 60% in the last 12 hours" line above, still need the same fix.** Unfiltered, Kalshi's dailies still
  price their last 6 hours richly (49% against 44% realized).
- *Kalshi's premium shrinks through the year:* median daily gap to Deribit about 7 points in Q1 2026, 2.5 in Q2–Q3.
- *Deribit's short end has a weekend effect:* Saturday and Sunday dailies trade at about 29% and 23% vol against
  35–37% on weekdays. `10` now uses Tuesday–Friday expiries for the term structure and shows the P&L by weekday
  (weekday-only, the calendar still loses about $140 a day before costs).

**Dislocations** = the gap minus its normal (median over the previous 30 days, same contract type and time to
expiry).
- Kalshi unusually *cheap* vs Deribit: Deribit spiked and Kalshi did not follow; Deribit falls back (about 4 points
  in a day) and had overpriced BTC's later moves. Kalshi unusually *rich*: the gap barely closes, and Deribit had
  underpriced BTC's later moves (weeklies).
- Kalshi's dislocation predicts Deribit's later error (t = −2.8, clustered by week); Deribit's own vol spike does
  not (after Deribit's biggest jumps BTC keeps moving, unless Kalshi stayed calm).
- Trades: selling Deribit's straddle on large Kalshi-cheap dislocations (≥ 6 points), held to expiry, earns about
  +6 vol points per trade after taker costs, but on 12 weeks only. Selling Kalshi's rich dailies earns about +6
  only as a maker and has faded to about zero by Q3. Nothing with short holds survives.
- Predicting BTC: Deribit's probabilities score marginally better than Kalshi's (Brier 0.1013 vs 0.1018); Kalshi
  overprices big up-moves (longshot bias, question 1); neither predicts direction; Deribit forecasts realized vol
  better, Kalshi's dislocation adds to it.

## Strategies A and B, and hedging a binary (`12`, `13`)

- **A** (sell Deribit's weekly straddle when Kalshi is ≥ 6 vol points cheaper than its normal gap, hold to
  expiry): +6 vol points per trade after taker costs, +11 as a maker (Sharpe about 2 and 3), but 12 positions,
  nearly all February–April; after April the signal fired 3 times. Thresholds below 6 lose in every window.
- **B** (sell Kalshi's daily strip when it is ≥ 3 points richer than normal): +6 vol points per trade if sold at
  mid as a maker, negative if it crosses the spread; positive in every quarter as a maker, shrinking.
- **What holds up is the signal:** sorted by dislocation, Deribit's vol error against realized falls from about +2
  points (Kalshi cheapest) to −5 (Kalshi richest), in both halves of the year.
- **Hedging a long Kalshi binary** (professor's question): the short call spread is a static hedge; delta-hedging
  the binary alone fails near expiry (gamma ~ 1 / time left; ±$500k error on $1M across 2026 dailies). On 9 Oct
  2025, $1M of the $123,000 strike (Kalshi 1.4¢ cheap) hedged with the $500 spread returns about the edge
  (+$24k); the binary alone, delta-hedged half-hourly, about −$27k after falling to −$120k, with hedges up to
  1,200 BTC on the pin strike. What is left is pin risk; across 2026 the best hedge is the static spread unwound an
  hour before expiry (error ±$72k vs ±$237k held to expiry and ±$500k for the delta-hedged binary).

## Known limits

- In the last 6 hours the $500 strike spacing is too coarse compared with how little Bitcoin can still move, so those volatilities are rough.
- October 2025 has few usable strikes, so most of each smile there comes from the tails.
- The tail shape is an assumption, kept within Kalshi's own limits. Kalshi's far-strike quotes only rule out extreme cases.
- The 13-hour adjustment uses Bitcoin's past hour-by-hour movement, not what Deribit's prices say about that Friday.
- We have Deribit trades and marks, not its bid and ask, so Deribit's real trading costs are assumed, not measured.
- Kalshi's fee formula in `lib/compare.py` (7% × P × (1 − P), rounded up to the cent) still needs checking against Kalshi's current fee schedule.

## Next

0. The strategies: Kalshi as a *maker* (posting the strip instead of crossing spreads) is the one untested change
   that could make strategy 1 pay; Kalshi's 1-minute quotes (`00_fetch_data.py`, `02`) would push the lead-lag test
   below 15 minutes.

1. Handle the 13-hour gap with market prices instead of history: interpolate between Deribit's Friday 08:00 and Saturday 08:00 UTC expiries, which bracket Kalshi's 5pm ET. *Done hourly in `08`/`09` (`vol_spread.deribit_at_kalshi`); not yet applied to `05`'s trade-by-trade pairs.*
2. Robustness of `05`: matching window length, trade prices instead of marks, the morning window.
3. Get Deribit option-chain quotes (bid, ask and mark for every option, e.g. Tardis.dev): a full Deribit smile every minute for the comparison.
4. Handout question 1: write up the theory (Kalshi contract logistics, Black-Scholes value N(d₂), the call-spread link).
5. Handout question 3: check whether Kalshi's probabilities come true as often as they say (calibration against how contracts settled), and compare with Deribit's.
