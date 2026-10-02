import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="02 Clean data")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from lib import cleaning as cl
    from lib.fetch_utils import write_parquet

    return cl, mo, np, pd, write_parquet


@app.cell
def _(mo):
    mo.md(r"""
    # 02 · Clean dataset

    Builds the clean inputs for Black-Scholes implied-volatility surfaces from the raw data in
    `data/raw/`, applying the rules agreed in `01_data_exploration.py`. Every rule is listed below,
    implemented in `lib/cleaning.py` (unit-tested in `tests/test_cleaning.py`), and checked in its own
    section with the evidence that justifies it.

    | | Rule | Section |
    |---|---|---|
    | R1 | A missing settlement value is reconstructed as the BRTI 60-second average one second before expiry, rounded to the cent | 1 |
    | R2 | Events whose settlement fell outside the strike ladder (every strike resolved the same way) are flagged `ladder_outside` | 1 |
    | R3 | Spot = average of the Coinbase and Bitstamp closes, Bitstamp alone where Coinbase is missing; `spot_stale` when no exchange traded that minute | 2 |
    | R4 | Each strike's quotes are forward-filled minute by minute from its first candle to expiry; volume is 0 on filled minutes; `quote_age_min` = minutes since its last candle; minutes before the first candle are dropped | 3 |
    | R5 | A bid of 0 means no bid and an ask of 1 means no ask | 3 |
    | R6 | `spread` exists for valid two-sided quotes (bid < ask); `mid` only when the spread is at most 20¢ | 3 |
    | R7 | `exchange_open` is False during Kalshi's weekly closure, Thursday 03:00–05:00 ET (minute ends 03:01–05:00) | 3 |
    | R8 | `usable` = a mid exists and the exchange is open: the quotes an implied-vol fit should use. No quote-age cap is applied; `quote_age_min` is kept for filtering in the analysis | 3 |

    **Outputs** (`data/clean/`, not in git): `kalshi_weekly_events.parquet`, `kalshi_weekly_markets.parquet`,
    `btc_spot_1m.parquet`, and `kalshi_weekly_quotes/{event}.parquet` (one file per weekly; read the folder
    with `pd.read_parquet`). Column definitions are in section 5.
    """)
    return


@app.cell
def _(mo):
    write_button = mo.ui.run_button(label="Write clean dataset to data/clean/", kind="success")
    mo.vstack([
        mo.md("""
        **Writing.** Opening this notebook only builds and checks the clean data in memory. Files are written
        when you press the button, or when the notebook runs as a script: `python 02_clean_data.py`.
        """),
        write_button,
    ])
    return (write_button,)


@app.cell
def _(mo, pd, write_button):
    WRITE = write_button.value or not mo.running_in_notebook()
    ROOT = mo.notebook_dir()
    RAW, CLEAN = ROOT / "data" / "raw", ROOT / "data" / "clean"
    KALSHI_DIR = RAW / "kalshi_btc_weekly"

    events = pd.read_parquet(KALSHI_DIR / "events.parquet")
    markets = pd.read_parquet(KALSHI_DIR / "markets.parquet")
    settlement = pd.read_parquet(KALSHI_DIR / "settlement.parquet")
    spot_raw = pd.read_parquet(RAW / "btc_spot" / "btcusd_1m.parquet")
    return CLEAN, KALSHI_DIR, WRITE, events, markets, settlement, spot_raw


@app.cell
def _(CLEAN, WRITE, cl, events, markets, mo, np, settlement, write_parquet):
    # R1: reconstruct missing settlement values from the BRTI settlement index
    settle_rec = cl.reconstruct_expiration(settlement, events.set_index("event_ticker").close_time)
    _published = events.set_index("event_ticker").expiration_value
    _known = _published.notna()
    _reproduced = int(np.isclose(settle_rec[_known], _published[_known], atol=0.005).sum())

    _source = lambda s: np.where(s.isna(), "reconstructed from BRTI", "published")
    events_clean = events.assign(expiration_source=_source(events.expiration_value),
                                 expiration_value=events.expiration_value.fillna(events.event_ticker.map(settle_rec)))

    # R2: flag events where every strike resolved the same way (settlement outside the ladder)
    _ladder = markets.groupby("event_ticker").agg(ladder_low=("strike", "min"), ladder_high=("strike", "max"),
                                                  _floor_min=("floor_strike", "min"), _floor_max=("floor_strike", "max"))
    events_clean = events_clean.join(_ladder, on="event_ticker")
    events_clean["ladder_outside"] = ((events_clean.expiration_value <= events_clean._floor_min)
                                      | (events_clean.expiration_value > events_clean._floor_max))
    events_clean = events_clean.drop(columns=["_floor_min", "_floor_max"])

    markets_clean = markets.assign(
        expiration_source=_source(markets.expiration_value),
        expiration_value=markets.expiration_value.fillna(markets.event_ticker.map(settle_rec)),
    ).merge(events_clean[["event_ticker", "ladder_outside"]], on="event_ticker")
    _consistent = (markets_clean.expiration_value > markets_clean.floor_strike) == (markets_clean.result == "yes")

    if WRITE:
        write_parquet(events_clean, CLEAN / "kalshi_weekly_events.parquet")
        write_parquet(markets_clean, CLEAN / "kalshi_weekly_markets.parquet")

    mo.vstack([
        mo.md(f"""
        ## 1 · Contracts and settlement (R1, R2)

        **R1 evidence.** The rule reproduces **{_reproduced} of {int(_known.sum())}** published settlement values
        exactly. **Change:** {int((~_known).sum())} event had no published value and is filled from the rule
        ({int(markets.expiration_value.isna().sum())} strikes); `expiration_source` records it. After filling,
        **{int(_consistent.sum()):,} of {len(markets_clean):,}** strikes resolved consistently with
        settlement value > floor strike.

        **R2.** {int(events_clean.ladder_outside.sum())} of {len(events_clean)} weeklies settled outside their
        strike ladder; they have no tail to difference and are flagged rather than dropped.
        """),
        mo.ui.table(events_clean.loc[(events_clean.expiration_source != "published") | events_clean.ladder_outside,
                                     ["event_ticker", "close_time", "expiration_value", "expiration_source",
                                      "ladder_low", "ladder_high", "ladder_outside"]],
                    selection=None, show_download=False),
    ])
    return


@app.cell
def _(CLEAN, WRITE, cl, mo, pd, spot_raw, write_parquet):
    # R3: one spot price per minute
    spot_clean = cl.clean_spot(spot_raw)
    spot_clean.index.name = "ts"
    _full = pd.date_range(spot_clean.index.min(), spot_clean.index.max(), freq="1min")
    _gaps = len(_full) - int(spot_clean.spot.notna().sum())

    if WRITE:
        write_parquet(spot_clean.reset_index(), CLEAN / "btc_spot_1m.parquet")

    mo.vstack([
        mo.md(f"""
        ## 2 · Spot (R3)

        {len(spot_clean):,} minutes ({spot_clean.index.min():%Y-%m-%d} → {spot_clean.index.max():%Y-%m-%d}),
        **{_gaps} minutes without a price**. Coinbase and Bitstamp agree to a median of about 0.1 bps
        (section 6 of the exploration), so averaging them is safe; Bitstamp alone covers Coinbase's outages.
        `spot_stale` marks the {int(spot_clean.spot_stale.sum()):,} minutes in which no exchange traded
        (the price is the last trade, carried forward).
        """),
        mo.ui.table(spot_clean.spot_source.value_counts().rename_axis("spot_source").reset_index(name="minutes"),
                    selection=None, show_download=False),
    ])
    return (spot_clean,)


@app.cell
def _(
    CLEAN,
    KALSHI_DIR,
    WRITE,
    cl,
    markets,
    np,
    pd,
    spot_clean,
    write_parquet,
):
    # R4-R8: forward-filled quote grid per weekly, with sides, mid/spread, exchange hours, spot and moneyness
    _spot, _stale = spot_clean.spot, spot_clean.spot_stale
    _rows, _checks = [], []
    for _ev, _m in markets.groupby("event_ticker"):
        _cdl = pd.read_parquet(KALSHI_DIR / "candles" / f"{_ev}.parquet",
                               columns=["market_ticker", "ts", "bid_close", "ask_close", "volume", "open_interest"])
        _g = cl.build_event_grid(_cdl, _m)                                      # R4
        _bid, _ask = cl.quote_sides(_g.bid_raw.to_numpy(), _g.ask_raw.to_numpy())  # R5
        _spread, _mid = cl.mid_and_spread(_bid, _ask)                           # R6
        _open = cl.exchange_open(_g.ts)                                         # R7
        _s = _spot.reindex(_g.ts).to_numpy()
        _q = pd.DataFrame({
            # plain strings (parquet dictionary-encodes them): per-file categoricals cannot be read as one folder
            "event_ticker": _ev,
            "market_ticker": _g.market_ticker.astype(str),
            "strike": _g.strike.astype("float32"),
            "ts": _g.ts,
            "tau_years": cl.tau_years(_g.ts, _m.close_time.max()).astype("float32"),
            "bid": _bid.astype("float32"), "ask": _ask.astype("float32"),
            "spread": _spread.astype("float32"), "mid": _mid.astype("float32"),
            "quote_age_min": _g.quote_age_min,
            "exchange_open": _open,
            "usable": ~np.isnan(_mid) & _open,                                  # R8
            "volume": _g.volume, "open_interest": _g.open_interest,
            "spot": _s, "spot_stale": _stale.reindex(_g.ts).to_numpy(),
            "log_moneyness": np.log(_g.strike.to_numpy() / _s).astype("float32"),
        })
        if WRITE:
            write_parquet(_q, CLEAN / "kalshi_weekly_quotes" / f"{_ev}.parquet")

        # independent checks of the grid against the raw candles
        _first = _cdl.groupby("market_ticker").ts.min()
        _expected_rows = int(((_m.close_time.max() - _first) // pd.Timedelta(minutes=1) + 1).sum())
        _at = _q.merge(_cdl, on=["market_ticker", "ts"], how="inner")
        _rows.append({
            "event_ticker": _ev, "rows": len(_q), "usable": int(_q.usable.sum()),
            "near_usable": int((_q.usable & (_q.log_moneyness.abs() < 0.01)).sum()),
            "closed_rows": int((~_q.exchange_open).sum()),
        })
        _checks.append({
            "rows ≠ first candle → expiry": len(_q) - _expected_rows,
            "candle minutes whose carried quote ≠ raw close": int((~np.isclose(np.nan_to_num(_at.bid, nan=0.0), _at.bid_close)
                                                                   | ~np.isclose(np.nan_to_num(_at.ask, nan=1.0), _at.ask_close)).sum()),
            "rows without spot": int(np.isnan(_s).sum()),
            "mids with spread > 20¢": int((_q.mid.notna() & (_q.spread > 0.2001)).sum()),
            "negative time to expiry": int((_q.tau_years < 0).sum()),
        })
        quote_sample = _q[_q.usable & (_q.log_moneyness.abs() < 0.01)].tail(8)
    quote_summary = pd.DataFrame(_rows)
    quote_checks = pd.DataFrame(_checks).sum().rename("failures").rename_axis("check").reset_index()
    return quote_checks, quote_sample, quote_summary


@app.cell
def _(mo, quote_checks, quote_sample, quote_summary):
    _tot = quote_summary.rows.sum()
    mo.vstack([
        mo.md(f"""
        ## 3 · Quotes (R4–R8)

        **R4 forward-fill.** In the exploration, every gap of two minutes or more between a strike's candles
        reopened exactly at the previous closing quote, so carrying the last quote forward reconstructs
        the quote path. Rows run from each strike's first candle to expiry: **{_tot:,}** strike-minutes
        across {len(quote_summary)} weeklies.

        **R5–R6 sides, spread and mid.** Bid 0 / ask 1 are Kalshi's encodings for an empty side. Placeholder
        quotes such as bid 0.01 / ask 0.99 are two-sided only on paper (their mid ~0.50 is not a probability),
        so a mid requires a spread of at most 20¢.

        **R7 exchange hours.** Kalshi closes every Thursday 03:00–05:00 ET: tight quotes are pulled, nothing
        trades, BTC keeps moving. {int(quote_summary.closed_rows.sum()):,} strike-minutes fall in the closure.

        **R8 usable.** **{int(quote_summary.usable.sum()):,}** strike-minutes ({quote_summary.usable.sum() / _tot:.0%})
        have a usable mid with the exchange open; {int(quote_summary.near_usable.sum()):,} of them are within 1% of
        spot.

        **Checks against the raw data** (all should be 0):
        """),
        mo.hstack([mo.ui.table(quote_checks, selection=None, show_download=False)], justify="start"),
        mo.md("Per weekly:"),
        mo.ui.table(quote_summary.assign(usable_share=(quote_summary.usable / quote_summary.rows).map("{:.1%}".format)),
                    selection=None, show_download=False, page_size=10),
        mo.md("Sample of usable near-the-money rows (last weekly):"),
        mo.ui.table(quote_sample, selection=None, show_download=False),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · What is not in the clean data

    - **Interest rates and implied volatility**: computed in the pricing step, not here.
    - **A quote-age cap**: none is applied. Near the money, 0.8% of strike-minutes carry a quote older than
      an hour; filter on `quote_age_min` in the analysis if needed.
    - **Minutes before a strike's first candle**: no quote exists to fill (560 strike-minutes in total).

    ## 5 · Column definitions

    **`kalshi_weekly_quotes/{event}.parquet`**: one row per strike per minute (UTC minute ends).

    | Column | Meaning |
    |---|---|
    | `event_ticker`, `market_ticker` | Weekly event and strike |
    | `strike` | K, the "or above" threshold in USD (YES pays if the settlement value is above K − 0.01) |
    | `ts` | Minute end (UTC); joins with spot on `ts` |
    | `tau_years` | Time to expiry, ACT/365 |
    | `bid`, `ask` | Best YES bid / ask at the minute end in dollars; NaN when that side is empty (R5) |
    | `spread` | ask − bid for valid two-sided quotes (R6) |
    | `mid` | (bid + ask) / 2 when the spread is at most 20¢ (R6) |
    | `quote_age_min` | Minutes since the strike's last candle; 0 = updated this minute (R4) |
    | `exchange_open` | False during the Thursday 03:00–05:00 ET closure (R7) |
    | `usable` | `mid` present and exchange open (R8) |
    | `volume` | Contracts traded in the minute (0 on forward-filled minutes) |
    | `open_interest` | Contracts outstanding (carried forward) |
    | `spot`, `spot_stale` | Exchange spot at the minute end and its staleness flag (R3) |
    | `log_moneyness` | ln(strike / spot) |

    **`kalshi_weekly_markets.parquet`**: one row per strike, with `result`, `expiration_value` (filled per R1),
    `expiration_source` and `ladder_outside`. **`kalshi_weekly_events.parquet`**: one row per weekly, with the
    same settlement fields and the ladder range. **`btc_spot_1m.parquet`**: `ts`, `coinbase`, `bitstamp`,
    `spot`, `spot_source`, `spot_stale`.
    """)
    return


@app.cell
def _(CLEAN, WRITE, mo):
    _files = sorted(CLEAN.rglob("*.parquet")) if CLEAN.exists() else []
    _size = sum(f.stat().st_size for f in _files) / 1e6
    mo.md(
        f"**Written this run:** {'yes' if WRITE else 'no (press the button above, or run as a script)'}. "
        f"`data/clean/` currently holds {len(_files)} files, {_size:,.0f} MB."
    )
    return


if __name__ == "__main__":
    app.run()
