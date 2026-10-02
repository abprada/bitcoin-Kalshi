import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="01 Data exploration")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    return alt, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # 01 · Data exploration

    Before any modelling: what is in the two datasets we keep, what is missing, and what needs
    cleaning.

    - **Kalshi weekly BTC** (`data/raw/kalshi_btc_weekly/`): every weekly KXBTCD contract
      (Friday 5pm ET expiry) from Oct 2025 to Sep 2026. Each strike is a binary
      "BTC settles above K", quoted as 1-minute YES bid/ask candles, plus how each strike resolved
      and the settlement index for the last 3h before expiry.
    - **BTC spot** (`data/raw/btc_spot/`): 1-minute BTC/USD candles from Coinbase and Bitstamp,
      two of the exchanges in the CF Benchmarks BRTI the contracts settle on.

    All timestamps are UTC minute ends, so Kalshi and spot join on `ts`. Shares of quote states are
    **time-weighted**: Kalshi only emits a candle when something changes, so each candle is weighted
    by how long its closing quote stood.
    """)
    return


@app.cell
def _(alt):
    # Chart styling: reference palette (slot 1 blue, slot 2 orange), recessive gray chrome.
    INK, INK2, MUTED, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
    SLOT1, SLOT2 = "#2a78d6", "#eb6834"
    EXCHANGE_SCALE = alt.Scale(domain=["coinbase", "bitstamp"], range=[SLOT1, SLOT2])
    W, H = 640, 200


    @alt.theme.register("research_light", enable=True)
    def _research_light() -> alt.theme.ThemeConfig:
        return {
            "config": {
                "background": "#fcfcfb",
                "font": "system-ui, -apple-system, 'Segoe UI', sans-serif",
                "view": {"stroke": None},
                "axis": {"gridColor": "#e1e0d9", "domainColor": CONTEXT, "tickColor": CONTEXT,
                         "labelColor": MUTED, "titleColor": INK2, "titleFontWeight": "normal"},
                "legend": {"labelColor": INK2, "titleColor": INK2, "orient": "top"},
                "axisX": {"labelAngle": 0},
                "title": {"color": INK, "anchor": "start", "fontWeight": "normal", "fontSize": 14},
                "bar": {"color": SLOT1, "cornerRadiusEnd": 2},
                "line": {"color": SLOT1, "strokeWidth": 2},
                "point": {"color": SLOT1, "size": 64, "filled": True},
            }
        }

    return CONTEXT, EXCHANGE_SCALE, H, MUTED, SLOT1, SLOT2, W


@app.cell
def _(mo, pd):
    DATA = mo.notebook_dir() / "data" / "raw"
    KALSHI_DIR, SPOT_DIR = DATA / "kalshi_btc_weekly", DATA / "btc_spot"

    events = pd.read_parquet(KALSHI_DIR / "events.parquet")
    markets = pd.read_parquet(KALSHI_DIR / "markets.parquet")
    settlement = pd.read_parquet(KALSHI_DIR / "settlement.parquet")
    spot = pd.read_parquet(SPOT_DIR / "btcusd_1m.parquet")

    _floats = ["strike", "bid_open", "bid_high", "bid_low", "bid_close",
               "ask_open", "ask_high", "ask_low", "ask_close", "volume", "open_interest"]
    # 11.8M rows: float32 (cast per file) and categorical tickers keep it to well under 1 GB
    candles = pd.concat([pd.read_parquet(_f).astype({_c: "float32" for _c in _floats})
                         for _f in sorted((KALSHI_DIR / "candles").glob("*.parquet"))], ignore_index=True)
    candles = candles.astype({"event_ticker": "category", "market_ticker": "category"})
    return candles, events, markets, settlement, spot


@app.cell
def _(candles, events, markets, mo, pd, settlement, spot):
    def _span(s):
        return f"{s.min():%Y-%m-%d %H:%M} → {s.max():%Y-%m-%d %H:%M}"


    tables = {"events": (events, "close_time"), "markets": (markets, "close_time"),
              "candles": (candles, "ts"), "settlement": (settlement, "ts"), "spot": (spot, "ts")}
    inventory = pd.DataFrame([
        {"table": _name, "rows": len(_df), "columns": _df.shape[1], "time span (UTC)": _span(_df[_col]),
         "memory (MB)": round(_df.memory_usage(deep=True).sum() / 1e6, 1)}
        for _name, (_df, _col) in tables.items()
    ])
    mo.vstack([mo.md("## 1 · Inventory"), mo.ui.table(inventory, selection=None, show_download=False)])
    return (tables,)


@app.cell
def _(mo, pd, tables):
    nan_report = pd.concat([
        _df.isna().sum().rename("nan").rename_axis("column").reset_index().assign(table=_name, rows=len(_df))
        for _name, (_df, _col) in tables.items()
    ])
    nan_report["share"] = (nan_report["nan"] / nan_report["rows"]).map("{:.4%}".format)
    nan_nonzero = nan_report.loc[nan_report["nan"] > 0, ["table", "column", "nan", "share"]]
    mo.vstack([
        mo.md(f"""
        ## 2 · NaNs

        {len(nan_report)} columns checked across the five tables; **{len(nan_nonzero)} contain any NaN**.
        Every other column is complete. Note that missing *quotes* in Kalshi are not NaN: an empty side
        is encoded as bid 0 / ask 1 (section 4), and missing *minutes* are absent rows (sections 4, 6).
        """),
        mo.ui.table(nan_nonzero, selection=None, show_download=False),
    ])
    return


@app.cell
def _(H, W, alt, events, markets, mo):
    ev = events.assign(
        life_h=(events.close_time - events.open_time).dt.total_seconds() / 3600,
        expiry_et=events.close_time.dt.tz_convert("America/New_York").dt.strftime("%a %Y-%m-%d %H:%M"),
    )
    life_counts = ev.life_h.value_counts().rename_axis("life (hours)").reset_index(name="events")
    volume_chart = (
        alt.Chart(ev).mark_bar()
        .encode(
            x=alt.X("close_time:T", title="expiry"),
            y=alt.Y("volume:Q", title="contracts traded"),
            tooltip=["event_ticker:N", "expiry_et:N", alt.Tooltip("volume:Q", format=",.0f"), "n_strikes:Q"],
        )
        .properties(title="Contracts traded per weekly contract", width=W, height=H)
    )
    mo.vstack([
        mo.md(f"""
        ## 3 · Kalshi contracts

        **{len(ev)} weekly events**, {len(markets):,} strikes. All expire Friday 5pm ET:
        **{(ev.close_time.dt.tz_convert('America/New_York').dt.weekday.eq(4) & ev.close_time.dt.tz_convert('America/New_York').dt.hour.eq(17)).all()}**.
        Lifetimes are 169h except across the two DST changes (168h / 170h of real time; still
        Friday 4pm ET → Friday 5pm ET). Jan 9 2026 had no weekly (its Friday event lived 21h) and is
        excluded by construction. Strikes per event: {ev.n_strikes.min()}–{ev.n_strikes.max()}
        (median {ev.n_strikes.median():.0f}).
        """),
        mo.hstack([mo.ui.table(life_counts, selection=None, show_download=False)], justify="start"),
        volume_chart,
        mo.md(f"Volume per weekly ranges from {ev.volume.min():,.0f} to {ev.volume.max():,.0f} contracts "
              f"({ev.volume.max() / ev.volume.min():.0f}×). The growth is not smooth: weekly volume roughly "
              "triples between the Jan 2 and Jan 16 2026 contracts (0.98M → 2.84M; there was no Jan 9 weekly). "
              "Treat that as a structural break: any time-series comparison should control for it or "
              "split the sample there."),
    ])
    return (ev,)


@app.cell
def _(CONTEXT, MUTED, SLOT1, W, alt, markets, np, settlement):
    _terminal = settlement.sort_values("ts").groupby("event_ticker").index_60s.last()
    ladder = (markets.groupby("event_ticker")
              .agg(low=("strike", "min"), high=("strike", "max"), n_strikes=("strike", "size"),
                   settle=("expiration_value", "first"), expiry=("close_time", "max"))
              .reset_index())
    ladder["settle_source"] = np.where(ladder.settle.isna(), "settlement index (expiration_value missing)",
                                       "expiration_value")
    ladder["settle"] = ladder.settle.fillna(ladder.event_ticker.map(_terminal))
    ladder["position"] = np.select([ladder.settle < ladder.low, ladder.settle > ladder.high],
                                   ["below ladder", "above ladder"], "inside ladder")
    ladder["outside"] = ladder.position != "inside ladder"

    _base = alt.Chart(ladder).encode(x=alt.X("expiry:T", title="expiry"))
    _tip = ["event_ticker:N", "position:N", alt.Tooltip("settle:Q", format=",.2f", title="settled at"),
            alt.Tooltip("low:Q", format=",.0f", title="lowest strike"),
            alt.Tooltip("high:Q", format=",.0f", title="highest strike"), "settle_source:N"]
    ladder_chart = (
        _base.mark_rule(color=CONTEXT, strokeWidth=5).encode(
            y=alt.Y("low:Q", title="BTC (USD)", scale=alt.Scale(zero=False)), y2="high:Q", tooltip=_tip)
        + _base.mark_point(size=50).encode(
            y="settle:Q", color=alt.condition("datum.outside", alt.value(SLOT1), alt.value(MUTED)), tooltip=_tip)
        + _base.transform_filter("datum.outside").mark_text(align="left", dx=8, color=SLOT1).encode(
            y="settle:Q", text="position:N")
    ).properties(title="Strike ladder (gray range) vs settlement value (dot)", width=W, height=260)
    ladder_chart
    return (ladder,)


@app.cell
def _(events, ladder, markets, mo, np):
    _gaps = (markets.sort_values(["event_ticker", "strike"]).groupby("event_ticker").strike
             .agg(lambda s: ", ".join(f"${g:,.0f}" for g in sorted(set(np.round(np.diff(s.to_numpy()), 2))))))
    strike_spacing = _gaps.value_counts().rename_axis("gaps between adjacent strikes").reset_index(name="events")
    resolution_flags = events.loc[~events.resolution_ok, ["event_ticker", "expiration_value", "resolution_ok"]]
    mo.vstack([
        mo.md(f"""
        **Ladder coverage.** In {int(ladder.outside.sum())} of {len(ladder)} weeks the settlement value fell
        outside the strike ladder, so every strike resolved the same way and that week has no tail to
        difference. **Strike spacing** is $500 everywhere; {int(strike_spacing.loc[strike_spacing.iloc[:, 0] != "$500", "events"].sum())}
        events also have strikes added during the week at finer spacing ($100 / $250).
        **Resolution checks** (fetch time): every strike's result matches settlement value >
        floor strike. {len(resolution_flags)} event is flagged, because Kalshi published no
        `expiration_value` for it; its results are still consistent with the settlement index.
        """),
        mo.hstack([mo.ui.table(strike_spacing, selection=None, show_download=False),
                   mo.ui.table(resolution_flags, selection=None, show_download=False)], justify="start", gap=2),
    ])
    return


@app.cell
def _(candles, markets, np, pd):
    # Per-candle derived arrays (numpy for speed on 11.8M rows)
    ticker_codes = candles.market_ticker.cat.codes.to_numpy()
    ticker_cats = candles.market_ticker.cat.categories
    _ts = candles.ts.dt.tz_convert(None).to_numpy("datetime64[ns]")
    _close = markets.set_index("market_ticker").close_time.dt.tz_convert(None).reindex(ticker_cats).to_numpy("datetime64[ns]")
    _open = markets.set_index("market_ticker").open_time.dt.tz_convert(None).reindex(ticker_cats).to_numpy("datetime64[ns]")
    _bid, _ask = candles.bid_close.to_numpy(), candles.ask_close.to_numpy()

    tau_h = (_close[ticker_codes] - _ts) / np.timedelta64(1, "h")
    # minutes each candle's closing quote stands: until the strike's next candle, or until expiry
    same_next = np.r_[ticker_codes[1:] == ticker_codes[:-1], False]
    _next_ts = np.r_[_ts[1:], _ts[-1:]]
    dur_min = np.where(same_next, (_next_ts - _ts), (_close[ticker_codes] - _ts)) / np.timedelta64(1, "m")
    outside_life = (_ts < _open[ticker_codes]) | (_ts > _close[ticker_codes])

    STATES = ["two-sided", "bid only", "ask only", "empty book", "crossed/locked"]
    _hb, _ha = _bid > 0, _ask < 1
    state = np.select([_hb & _ha & (_bid < _ask), _hb & ~_ha, ~_hb & _ha, ~_hb & ~_ha], [0, 1, 2, 3], 4).astype(np.int8)
    two_sided = state == 0
    mid = np.where(two_sided, (_bid + _ask) / 2, np.nan)
    spread_c = np.where(two_sided, np.round((_ask - _bid) * 100), np.nan)
    TAU_LABELS = ["< 5 min", "5–60 min", "1–6 h", "6–24 h", "1–3 d", "3–7 d"]
    tau_bucket = pd.cut(tau_h, [-0.01, 1 / 12, 1, 6, 24, 72, 171], labels=TAU_LABELS)
    return (
        STATES,
        TAU_LABELS,
        dur_min,
        mid,
        outside_life,
        same_next,
        spread_c,
        state,
        tau_bucket,
        tau_h,
        ticker_cats,
        ticker_codes,
        two_sided,
    )


@app.cell
def _(H, STATES, TAU_LABELS, W, alt, dur_min, mo, pd, state, tau_bucket):
    _w = pd.DataFrame({"state": state, "dur": dur_min, "tau": tau_bucket})
    state_share = (_w.groupby("state").dur.sum() / _w.dur.sum()).rename(index=dict(enumerate(STATES)))
    state_table = pd.DataFrame({"quote state (at minute close)": state_share.index,
                                "share of strike-minutes": state_share.map("{:.2%}".format).values,
                                "candles": pd.Series(state).value_counts().reindex(range(5), fill_value=0).values})
    two_sided_by_tau = (_w.assign(ts_dur=_w.dur * (_w.state == 0)).groupby("tau", observed=False)[["ts_dur", "dur"]].sum()
                        .pipe(lambda d: (d.ts_dur / d.dur).rename("share").reset_index()))
    two_sided_chart = (
        alt.Chart(two_sided_by_tau).mark_bar()
        .encode(x=alt.X("tau:N", sort=TAU_LABELS, title="time to expiry"),
                y=alt.Y("share:Q", title="share two-sided", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                tooltip=["tau:N", alt.Tooltip("share:Q", format=".1%")])
        .properties(title="Share of strike-minutes with a two-sided quote", width=W, height=H)
    )
    mo.vstack([
        mo.md("""
        ## 4 · Kalshi quotes

        Each candle's closing quote is classified: **two-sided** (0 < bid < ask < 1), **bid only**
        (ask = 1, nobody offering), **ask only** (bid = 0), **empty book** (bid 0 / ask 1), or
        **crossed/locked** (bid ≥ ask). Only two-sided quotes give a usable mid. Far-from-the-money
        strikes sit one-sided for long stretches, so the two-sided share is the first filter for any
        pricing comparison.
        """),
        mo.hstack([mo.ui.table(state_table, selection=None, show_download=False)], justify="start"),
        two_sided_chart,
        mo.md("The two-sided share falls toward expiry because most of the ~50 strikes are already "
              "decided by then (bid 0.99 with nobody offering, or an offer at 0.01 with nobody bidding). "
              "This is the ladder collapsing, not missing data; only the strikes near the money stay "
              "two-sided, which is where the pricing comparison has to focus."),
    ])
    return (state_share,)


@app.cell
def _(
    H,
    W,
    alt,
    candles,
    dur_min,
    mid,
    mo,
    np,
    pd,
    spread_c,
    tau_h,
    two_sided,
):
    def _wmedian(values, weights):
        _o = np.argsort(values)
        _cw = np.cumsum(weights[_o])
        return values[_o][np.searchsorted(_cw, _cw[-1] / 2)]


    _two = pd.DataFrame({"spread": spread_c[two_sided], "mid": mid[two_sided], "dur": dur_min[two_sided]})
    _two["spread_bin"] = np.where(_two.spread >= 21, "21+", _two.spread.clip(upper=21).astype(int).astype(str))
    SPREAD_ORDER = [str(i) for i in range(1, 21)] + ["21+"]
    spread_dist = (_two.groupby("spread_bin").dur.sum() / _two.dur.sum()).rename("share").reset_index()
    spread_chart = (
        alt.Chart(spread_dist).mark_bar()
        .encode(x=alt.X("spread_bin:N", sort=SPREAD_ORDER, title="bid-ask spread (cents)"),
                y=alt.Y("share:Q", title="share of two-sided minutes", axis=alt.Axis(format="%")),
                tooltip=["spread_bin:N", alt.Tooltip("share:Q", format=".2%")])
        .properties(title="Spread distribution (two-sided quotes, time-weighted)", width=W, height=H)
    )
    _two["mid_bin"] = (np.floor(_two.mid * 20) / 20 + 0.025).round(3)
    spread_by_mid = (_two.assign(sd=_two.spread * _two.dur).groupby("mid_bin")[["sd", "dur"]].sum()
                     .pipe(lambda d: (d.sd / d.dur).rename("mean_spread").reset_index()))
    mid_dist = (_two.groupby("mid_bin").dur.sum() / _two.dur.sum()).rename("share").reset_index()
    spread_mid_chart = (
        alt.Chart(spread_by_mid).mark_line(point=True)
        .encode(x=alt.X("mid_bin:Q", title="mid price (probability)", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("mean_spread:Q", title="mean spread (cents)"),
                tooltip=[alt.Tooltip("mid_bin:Q", format=".3f"), alt.Tooltip("mean_spread:Q", format=".2f")])
        .properties(title="Spread by mid price", width=W // 2 - 20, height=H)
    )
    _bid2, _ask2 = candles.bid_close.to_numpy()[two_sided], candles.ask_close.to_numpy()[two_sided]
    _wide = _two.spread.to_numpy() >= 50
    _wide_ph = _two.dur[_wide & np.isclose(_bid2, 0.01) & np.isclose(_ask2, 0.99)].sum() / _two.dur[_wide].sum()
    _tau_wide = tau_h[two_sided][_wide]
    wide_table = pd.DataFrame({
        "spread at least": [f"{t}¢" for t in (10, 20, 50, 90)],
        "share of two-sided minutes": [f"{_two.dur[_two.spread >= t].sum() / _two.dur.sum():.2%}" for t in (10, 20, 50, 90)],
    })
    mid_chart = (
        alt.Chart(mid_dist).mark_bar()
        .encode(x=alt.X("mid_bin:Q", bin=alt.Bin(step=0.05), title="mid price (probability)", axis=alt.Axis(format=".1f")),
                y=alt.Y("share:Q", title="share of two-sided minutes", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("mid_bin:Q", format=".3f"), alt.Tooltip("share:Q", format=".2%")])
        .properties(title="Where two-sided mids sit", width=W // 2 - 20, height=H)
    )
    mo.vstack([
        spread_chart,
        mo.md(f"Time-weighted median two-sided spread: **{_wmedian(_two.spread.to_numpy(), _two.dur.to_numpy()):.0f}¢**; "
              f"{(spread_dist.loc[spread_dist.spread_bin == '1', 'share'].sum()):.1%} of two-sided minutes are at the 1¢ minimum tick."),
        mo.hstack([mid_chart, spread_mid_chart], justify="start"),
        mo.md(f"""
        **Placeholder quotes.** The spike in spread at mids near 0.50 comes from quotes that are two-sided
        only on paper: {_two.dur[_two.spread >= 20].sum() / _two.dur.sum():.1%} of two-sided minutes have a
        spread of 20¢ or more, and the single most common wide quote is **bid 0.01 / ask 0.99**
        ({_wide_ph:.0%} of the time spent at spreads ≥ 50¢). They cluster early in the week (median
        {np.median(_tau_wide):.0f}h to expiry). Their mid (~0.50) is not a probability estimate, so a
        usable mid needs a spread cap as well as two sides.
        """),
        mo.hstack([mo.ui.table(wide_table, selection=None, show_download=False)], justify="start"),
    ])
    return


@app.cell
def _(
    H,
    W,
    alt,
    candles,
    dur_min,
    markets,
    mo,
    np,
    pd,
    same_next,
    ticker_cats,
    ticker_codes,
):
    _gap_bins = [0, 1, 2, 5, 15, 60, 240, 1440, 20000]
    _gap_labels = ["1 min", "2 min", "3–5 min", "6–15 min", "16–60 min", "1–4 h", "4–24 h", "> 1 day"]
    _gaps_only = dur_min[same_next]
    gap_dist = (pd.Series(pd.cut(_gaps_only, _gap_bins, labels=_gap_labels)).value_counts(normalize=True)
                .reindex(_gap_labels).rename("share").rename_axis("gap").reset_index())
    gap_chart = (
        alt.Chart(gap_dist).mark_bar()
        .encode(x=alt.X("gap:N", sort=_gap_labels, title="time until the strike's next candle"),
                y=alt.Y("share:Q", title="share of candles", axis=alt.Axis(format="%")),
                tooltip=["gap:N", alt.Tooltip("share:Q", format=".2%")])
        .properties(title="Candle sparsity: how long a quote stands before it changes", width=W, height=H)
    )
    _life_min = (markets.close_time - markets.open_time).dt.total_seconds() / 60
    fill = (candles.groupby("market_ticker", observed=True).size()
            / _life_min.set_axis(markets.market_ticker).reindex(ticker_cats[np.unique(ticker_codes)]).to_numpy())
    mo.vstack([
        mo.md(f"""
        **Sparsity.** A candle exists only for minutes where the quote or trading changed. Across strikes,
        candles cover a median **{fill.median():.0%}** of each strike's lifetime minutes
        (10th–90th percentile {fill.quantile(.1):.0%}–{fill.quantile(.9):.0%}). The rest are minutes where
        the previous quote still stood, so the quotes must be **forward-filled per strike** on a 1-minute
        grid before joining with spot.
        """),
        gap_chart,
    ])
    return (fill,)


@app.cell
def _(candles, dur_min, mo, np, outside_life, pd, state):
    _bo, _bh, _bl, _bc = (candles[c].to_numpy() for c in ("bid_open", "bid_high", "bid_low", "bid_close"))
    _ao, _ah, _al, _ac = (candles[c].to_numpy() for c in ("ask_open", "ask_high", "ask_low", "ask_close"))
    _px = np.c_[_bo, _bh, _bl, _bc, _ao, _ah, _al, _ac]
    candle_checks = pd.DataFrame([
        ("bid OHLC out of order (low ≤ open, close ≤ high)", int(((_bl > np.minimum(_bo, _bc)) | (_bh < np.maximum(_bo, _bc))).sum())),
        ("ask OHLC out of order", int(((_al > np.minimum(_ao, _ac)) | (_ah < np.maximum(_ao, _ac))).sum())),
        ("any price outside [0, 1]", int(((_px < 0) | (_px > 1)).any(axis=1).sum())),
        ("bid/ask close not on the 1¢ grid", int((~np.isclose(np.c_[_bc, _ac] * 100, np.round(np.c_[_bc, _ac] * 100), atol=1e-3)).any(axis=1).sum())),
        ("crossed/locked closing quote (bid ≥ ask, both present)", int((state == 4).sum())),
        ("duplicate (strike, minute)", int(candles.duplicated(["market_ticker", "ts"]).sum())),
        ("candle outside the strike's life", int(outside_life.sum())),
        ("timestamps not increasing within a strike", int((dur_min < 0).sum())),
        ("negative volume or open interest", int(((candles.volume < 0) | (candles.open_interest < 0)).sum())),
    ], columns=["check", "candles failing"])
    mo.vstack([mo.md("**Integrity checks on the 11.8M candles**"),
               mo.ui.table(candle_checks, selection=None, show_download=False)])
    return


@app.cell
def _(H, TAU_LABELS, W, alt, candles, mo, pd, tau_bucket):
    volume_by_tau = (pd.DataFrame({"tau": tau_bucket, "volume": candles.volume.to_numpy()})
                     .groupby("tau", observed=False).volume.sum().pipe(lambda s: (s / s.sum()).rename("share").reset_index()))
    volume_tau_chart = (
        alt.Chart(volume_by_tau).mark_bar()
        .encode(x=alt.X("tau:N", sort=TAU_LABELS, title="time to expiry"),
                y=alt.Y("share:Q", title="share of all volume", axis=alt.Axis(format="%")),
                tooltip=["tau:N", alt.Tooltip("share:Q", format=".1%")])
        .properties(title="When weekly contracts trade", width=W, height=H)
    )
    mo.vstack([
        volume_tau_chart,
        mo.md(f"{(candles.volume > 0).mean():.1%} of candles carry volume. "
              f"{volume_by_tau.loc[volume_by_tau.tau.isin(['< 5 min', '5–60 min']), 'share'].sum():.0%} of volume trades "
              "in the final hour, where spot timing matters most for pricing."),
    ])
    return


@app.cell
def _(mo, pd, settlement):
    _sizes = settlement.groupby("event_ticker").size()
    _gaps_s = (settlement.sort_values(["event_ticker", "ts"]).groupby("event_ticker").ts.diff().dt.total_seconds() - 1).clip(lower=0)
    settlement_gaps = (pd.DataFrame({"points": _sizes, "missing seconds": 10801 - _sizes,
                                     "longest gap (s)": _gaps_s.groupby(settlement.event_ticker).max().reindex(_sizes.index)})
                       .query("`missing seconds` > 0").reset_index())
    mo.vstack([
        mo.md(f"""
        ## 5 · Kalshi settlement index (last 3h per weekly)

        BRTI 60-second trailing average, one point per second, 18:00/19:00 → 21:00/22:00 UTC on each
        expiry Friday (10,801 points when complete). {len(_sizes) - len(settlement_gaps)} of {len(_sizes)}
        events are complete; the rest have short holes in Kalshi's own record (none at the expiry second).
        """),
        mo.ui.table(settlement_gaps, selection=None, show_download=False),
    ])
    return


@app.cell
def _(EXCHANGE_SCALE, H, W, alt, mo, pd, spot):
    spot_close = spot.pivot(index="ts", columns="exchange", values="close")
    spot_close = spot_close.reindex(pd.date_range(spot_close.index.min(), spot_close.index.max(), freq="1min"))
    missing_daily = (spot_close.isna().resample("D").sum().rename_axis("day").reset_index()
                     .melt(id_vars="day", var_name="exchange", value_name="missing_minutes"))
    missing_chart = (
        alt.Chart(missing_daily).mark_line(point=alt.OverlayMarkDef(size=30))
        .encode(x=alt.X("day:T", title=None), y=alt.Y("missing_minutes:Q", title="missing minutes per day"),
                color=alt.Color("exchange:N", scale=EXCHANGE_SCALE, title=None),
                tooltip=[alt.Tooltip("day:T", format="%Y-%m-%d"), "exchange:N", "missing_minutes:Q"])
        .properties(title="Spot coverage: minutes with no candle, per exchange per day", width=W, height=H)
    )
    _both_missing = int(spot_close.isna().all(axis=1).sum())
    mo.vstack([
        mo.md(f"""
        ## 6 · BTC spot

        {len(spot_close):,} minutes from {spot_close.index.min():%Y-%m-%d} to {spot_close.index.max():%Y-%m-%d}.
        Coinbase is missing {int(spot_close.coinbase.isna().sum()):,} minutes (two ~6h outages:
        2025-10-25 and 2026-05-08, plus short holes), Bitstamp {int(spot_close.bitstamp.isna().sum()):,}.
        Minutes missing from **both**: **{_both_missing}**.
        """),
        missing_chart,
    ])
    return (spot_close,)


@app.cell
def _(mo, np, pd, spot):
    _o, _h, _l, _c, _v = (spot[c].to_numpy() for c in ("open", "high", "low", "close", "volume"))
    spot_checks = pd.DataFrame([
        ("OHLC out of order", int(((_l > np.minimum(_o, _c)) | (_h < np.maximum(_o, _c))).sum())),
        ("non-positive price", int(((np.c_[_o, _h, _l, _c]) <= 0).any(axis=1).sum())),
        ("duplicate (exchange, minute)", int(spot.duplicated(["exchange", "ts"]).sum())),
        ("Coinbase zero-volume minutes", int(((spot.exchange == "coinbase") & (spot.volume == 0)).sum())),
        ("Bitstamp zero-volume minutes (price carried, stale)", int(((spot.exchange == "bitstamp") & (spot.volume == 0)).sum())),
    ], columns=["check", "rows"])
    mo.vstack([mo.md("**Integrity checks on spot candles**"), mo.ui.table(spot_checks, selection=None, show_download=False)])
    return


@app.cell
def _(H, W, alt, mo, np, pd, spot_close):
    diff_bps = (1e4 * (spot_close.coinbase - spot_close.bitstamp) / spot_close.bitstamp).dropna()
    _clip = 10
    diff_hist = (pd.cut(diff_bps.clip(-_clip, _clip), np.arange(-_clip, _clip + 0.5, 0.5), include_lowest=True)
                 .value_counts(normalize=True).sort_index())
    diff_hist = pd.DataFrame({"bps": [iv.mid for iv in diff_hist.index], "share": diff_hist.to_numpy()})
    diff_hist_chart = (
        alt.Chart(diff_hist).mark_bar()
        .encode(x=alt.X("bps:Q", bin=alt.Bin(step=0.5), title="Coinbase − Bitstamp close (bps; tails clipped at ±10)"),
                y=alt.Y("share:Q", title="share of minutes", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("bps:Q", format=".2f"), alt.Tooltip("share:Q", format=".2%")])
        .properties(title="Cross-exchange price difference", width=W // 2 - 20, height=H)
    )
    diff_daily = diff_bps.abs().resample("D").quantile(0.99).rename("p99_abs_bps").rename_axis("day").reset_index()
    diff_daily_chart = (
        alt.Chart(diff_daily).mark_line()
        .encode(x=alt.X("day:T", title=None), y=alt.Y("p99_abs_bps:Q", title="99th pct |difference| (bps)"),
                tooltip=[alt.Tooltip("day:T", format="%Y-%m-%d"), alt.Tooltip("p99_abs_bps:Q", format=".1f")])
        .properties(title="Daily 99th percentile of |difference|", width=W // 2 - 20, height=H)
    )
    mo.vstack([
        mo.md(f"""
        **Do the two exchanges agree?** Median difference {diff_bps.median():+.2f} bps, standard deviation
        {diff_bps.std():.2f} bps; {(diff_bps.abs() > 5).mean():.2%} of minutes differ by more than 5 bps and
        {int((diff_bps.abs() > 25).sum())} minutes by more than 25 bps (mostly the Oct 10 2025 crash, when
        Coinbase traded up to ~2% below Bitstamp).
        """),
        mo.hstack([diff_hist_chart, diff_daily_chart], justify="start"),
    ])
    return (diff_bps,)


@app.cell
def _(H, W, alt, mo, np, pd, spot_close):
    spot_px = spot_close.mean(axis=1)  # average of the exchanges available each minute
    ret_bps = (1e4 * np.log(spot_px).diff()).dropna()
    _edges = np.arange(-100, 102, 2)
    _counts, _ = np.histogram(ret_bps.clip(-100, 100), bins=_edges)
    ret_hist = pd.DataFrame({"bps": (_edges[:-1] + 1), "minutes": _counts}).query("minutes > 0")
    ret_chart = (
        alt.Chart(ret_hist).mark_line(point=alt.OverlayMarkDef(size=20))
        .encode(x=alt.X("bps:Q", title="1-minute log return (bps; tails clipped at ±100)"),
                y=alt.Y("minutes:Q", title="minutes (log scale)", scale=alt.Scale(type="log")),
                tooltip=["bps:Q", alt.Tooltip("minutes:Q", format=",")])
        .properties(title="1-minute return distribution", width=W, height=H)
    )
    top_moves = (ret_bps.abs().nlargest(8).index.to_frame(index=False, name="ts")
                 .assign(return_bps=lambda d: ret_bps.loc[d.ts].round(1).to_numpy(),
                         coinbase=lambda d: spot_close.coinbase.reindex(d.ts).round(0).to_numpy(),
                         bitstamp=lambda d: spot_close.bitstamp.reindex(d.ts).round(0).to_numpy()))
    mo.vstack([
        mo.md(f"""
        **Returns.** 1-minute log returns of the exchange average: sd {ret_bps.std():.2f} bps, excess kurtosis
        **{ret_bps.kurt():.0f}** (a normal distribution has 0), so tails are far fatter than Black-Scholes
        assumes at this horizon. Largest single-minute moves:
        """),
        ret_chart,
        mo.ui.table(top_moves, selection=None, show_download=False),
    ])
    return (spot_px,)


@app.cell
def _(H, W, alt, mo, np, spot_px):
    _px15 = spot_px[spot_px.index.minute % 15 == 0]
    _r15 = np.log(_px15).diff().dropna()
    _daily = (_r15 ** 2).resample("D").agg(["sum", "count"])
    _daily = _daily[_daily["count"] >= 90]  # drop partial days (96 fifteen-minute returns in a full day)
    daily_rv = np.sqrt(_daily["sum"] * 365).rename("rv").rename_axis("day").reset_index()
    rv_chart = (
        alt.Chart(daily_rv).mark_line()
        .encode(x=alt.X("day:T", title=None), y=alt.Y("rv:Q", title="annualized realized vol", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("day:T", format="%Y-%m-%d"), alt.Tooltip("rv:Q", format=".1%")])
        .properties(title="Daily realized volatility (15-minute returns)", width=W, height=H)
    )
    mo.vstack([
        mo.md(f"Full-sample realized vol from 15-minute returns: **{np.sqrt((_r15 ** 2).mean() * 365 * 96):.1%}**. "
              f"Daily values range from {daily_rv.rv.min():.0%} to {daily_rv.rv.max():.0%}."),
        rv_chart,
    ])
    return


@app.cell
def _(candles, mo, pd, spot_close):
    _kalshi_minutes = pd.DatetimeIndex(candles.ts.drop_duplicates())
    _avail = spot_close.reindex(_kalshi_minutes).notna()
    join_coverage = pd.Series({
        "both exchanges": (_avail.coinbase & _avail.bitstamp).mean(),
        "Bitstamp only": (~_avail.coinbase & _avail.bitstamp).mean(),
        "Coinbase only": (_avail.coinbase & ~_avail.bitstamp).mean(),
        "no spot": (~_avail.coinbase & ~_avail.bitstamp).mean(),
    }).map("{:.3%}".format).rename_axis("spot available").reset_index(name="share of Kalshi candle minutes")
    mo.vstack([
        mo.md(f"## 7 · Joining Kalshi to spot\n\n{len(_kalshi_minutes):,} distinct minutes carry at least one Kalshi candle."),
        mo.hstack([mo.ui.table(join_coverage, selection=None, show_download=False)], justify="start"),
    ])
    return


@app.cell
def _(
    diff_bps,
    dur_min,
    ev,
    fill,
    ladder,
    markets,
    mo,
    pd,
    spot,
    spot_close,
    spread_c,
    state,
    state_share,
    two_sided,
):
    cleaning_plan = pd.DataFrame([
        ("Kalshi", "Empty or one-sided quotes (bid 0 / ask 1)",
         f"{1 - state_share['two-sided']:.1%} of strike-minutes",
         "Treat bid 0 as no bid and ask 1 as no ask; compute mids only from two-sided quotes"),
        ("Kalshi", "Sparse candles", f"median coverage {fill.median():.0%} of lifetime minutes",
         "Forward-fill each strike's quotes on a 1-minute grid within its own life"),
        ("Kalshi", "Placeholder two-sided quotes (e.g. 0.01 / 0.99)",
     f"{dur_min[two_sided & (spread_c >= 20)].sum() / dur_min[two_sided].sum():.1%} of two-sided minutes have spread ≥ 20¢",
     "Require spread ≤ 20¢ (threshold to confirm) for a usable mid"),
    ("Kalshi", "Crossed/locked closing quotes", f"{int((state == 4).sum()):,} candles",
         "Drop as invalid quotes"),
        ("Kalshi", "Missing expiration_value", f"{int(markets.expiration_value.isna().sum())} strikes (1 event)",
         "Use the settlement index at expiry for KXBTCD-25OCT1717"),
        ("Kalshi", "Settlement outside the strike ladder", f"{int(ladder.outside.sum())} events",
         "Flag; exclude from tail and density work"),
        ("Kalshi", f"Liquidity varies {ev.volume.max() / ev.volume.min():.0f}× across weeks", f"{ev.volume.min():,.0f} to {ev.volume.max():,.0f} contracts per week",
         "Control for it (or split the sample) in any time-series comparison"),
        ("Spot", "Coinbase missing minutes", f"{int(spot_close.coinbase.isna().sum()):,} minutes",
         "Average both exchanges; use Bitstamp alone where Coinbase is missing"),
        ("Spot", "Bitstamp zero-volume minutes", f"{int(((spot.exchange == 'bitstamp') & (spot.volume == 0)).sum()):,} minutes",
         "Price is carried forward; flag as stale"),
        ("Spot", "Cross-exchange dislocation", f"{int((diff_bps.abs() > 25).sum())} minutes > 25 bps apart",
         "Flag (Oct 10 2025 crash); check results with and without"),
    ], columns=["dataset", "issue", "size", "proposed cleaning rule"])
    mo.vstack([
        mo.md("""
        ## 8 · Proposed cleaning rules

        Nothing is modified here. These rules follow from the checks above and are the input to the
        cleaning step, implemented in `02_clean_data.py`.
        """),
        mo.ui.table(cleaning_plan, selection=None, show_download=False),
    ])
    return


@app.cell
def _(candles, dur_min, mo, np, pd, same_next):
    # Is a missing minute "no change"? Then the next candle must open where the previous one closed.
    _bo_next = np.r_[candles.bid_open.to_numpy()[1:], np.nan]
    _ao_next = np.r_[candles.ask_open.to_numpy()[1:], np.nan]
    _cont = (np.isclose(_bo_next, candles.bid_close.to_numpy())
             & np.isclose(_ao_next, candles.ask_close.to_numpy()))[same_next]
    _gap = pd.cut(dur_min[same_next], [0, 1, 5, 60, np.inf],
                  labels=["1 min (consecutive)", "2–5 min", "6–60 min", "over 1 h"])
    ffill_check = (pd.Series(_cont).groupby(_gap, observed=False).agg(["mean", "size"])
                   .rename_axis("time until the strike's next candle").reset_index()
                   .rename(columns={"mean": "next open = previous close", "size": "candle pairs"}))
    ffill_check["next open = previous close"] = ffill_check["next open = previous close"].map("{:.2%}".format)
    mo.vstack([
        mo.md(r"""
        ## 9 · Candle sparsity and forward-fill

        Kalshi emits a 1-minute candle only when a strike's quote or trading changes, so most
        strike-minutes have no row (section 4). Before forward-filling, two questions: **is a missing
        minute really "nothing changed"**, and **where is the data thin** (across contracts, by time to
        maturity, by distance from the money)?

        **Validity of forward-fill.** If nothing changed during a gap, the next candle must *open* at
        the quote the previous candle *closed* at. Checked on every consecutive pair of candles of the
        same strike:
        """),
        mo.hstack([mo.ui.table(ffill_check, selection=None, show_download=False)], justify="start"),
        mo.md("Across every gap of two minutes or more, the next candle opens exactly where the previous "
              "one closed: **missing minutes contain no unobserved quote changes**, so forward-filling "
              "reconstructs the quote path exactly. (Consecutive minutes match 99.9%; the rest are quotes "
              "that changed exactly on the minute boundary.)"),
    ])
    return


@app.cell
def _(candles, markets, mo, np, pd, spot_close):
    # Full strike × minute grid per weekly: which alive minutes have a candle, how far the strike is
    # from spot (average exchange close), and how old the quote would be after forward-filling.
    _px = spot_close.mean(axis=1).ffill()
    _px.index = _px.index.tz_convert(None)
    _cts = candles.ts.dt.tz_convert(None).to_numpy("datetime64[ns]")
    _evc = candles.event_ticker.to_numpy()
    MNY_LABELS = ["within 1%", "1–5%", "beyond 5%"]  # |strike / spot - 1|
    AGE_LABELS = ["0 (candle this minute)", "1–4 min", "5–14 min", "15–59 min", "1–4 h", "4 h +"]
    _age_edges = [0, 1, 5, 15, 60, 240, np.inf]
    _rows = []
    quote_age_counts = np.zeros((len(MNY_LABELS), len(AGE_LABELS)))
    unfillable_minutes = 0
    for _ev, _g in markets.groupby("event_ticker"):
        _close = _g.close_time.max().tz_convert(None).to_datetime64()
        _grid = pd.date_range(_g.open_time.min().tz_convert(None) + pd.Timedelta(minutes=1), _close,
                              freq="1min").to_numpy("datetime64[ns]")
        _n, _k = len(_grid), _g.strike.to_numpy()
        _alive = _grid[None, :] > _g.open_time.dt.tz_convert(None).to_numpy("datetime64[ns]")[:, None]
        _sel = _evc == _ev
        _r = candles.market_ticker[_sel].map({_t: _i for _i, _t in enumerate(_g.market_ticker)}).to_numpy().astype(int)
        _c = ((_cts[_sel] - _grid[0]) // np.timedelta64(1, "m")).astype(int)
        _has = np.zeros((len(_k), _n), bool)
        _has[_r, _c] = True
        _mb = np.digitize(np.abs(_k[:, None] / _px.reindex(_grid).to_numpy()[None, :] - 1), [0.01, 0.05])
        _last = np.maximum.accumulate(np.where(_has, np.arange(_n)[None, :], -1), axis=1)
        _fillable = _alive & (_last >= 0)
        unfillable_minutes += int((_alive & (_last < 0)).sum())
        for _m in range(len(MNY_LABELS)):
            quote_age_counts[_m] += np.histogram((np.arange(_n)[None, :] - _last)[_fillable & (_mb == _m)], bins=_age_edges)[0]
        _h = np.minimum(((_close - _grid) // np.timedelta64(1, "m")) // 60, 168).astype(int)  # hours to expiry
        _key = (_h[None, :] * 3 + _mb)[_alive]
        _rows.append(pd.DataFrame({
            "event_ticker": _ev, "expiry": pd.Timestamp(_close, tz="UTC"),
            "ttm_h": np.repeat(np.arange(169), 3), "mny": np.tile(MNY_LABELS, 169),
            "alive": np.bincount(_key, minlength=169 * 3),
            "candles": np.bincount(_key, weights=_has[_alive], minlength=169 * 3),
        }))
    sparsity_grid = pd.concat(_rows, ignore_index=True).query("alive > 0")
    mo.md(f"Grid built: **{int(sparsity_grid.alive.sum()):,}** alive strike-minutes across {sparsity_grid.event_ticker.nunique()} "
          f"weeklies, of which **{int(sparsity_grid.candles.sum()):,}** have a candle "
          f"({sparsity_grid.candles.sum() / sparsity_grid.alive.sum():.0%}).")
    return (
        AGE_LABELS,
        MNY_LABELS,
        quote_age_counts,
        sparsity_grid,
        unfillable_minutes,
    )


@app.cell
def _(H, MNY_LABELS, SLOT1, SLOT2, W, alt, mo, np, sparsity_grid):
    _scale = alt.Scale(domain=MNY_LABELS, range=[SLOT1, SLOT2, "#1baf7a"])
    _by_ttm = sparsity_grid.groupby(["ttm_h", "mny"])[["alive", "candles"]].sum().reset_index()
    _by_ttm["coverage"] = _by_ttm.candles / _by_ttm.alive
    _by_ev = sparsity_grid.groupby(["expiry", "event_ticker", "mny"])[["alive", "candles"]].sum().reset_index()
    _by_ev["coverage"] = _by_ev.candles / _by_ev.alive
    _color = alt.Color("mny:N", scale=_scale, sort=MNY_LABELS, title="strike vs spot")
    cov_ttm_chart = (
        alt.Chart(_by_ttm).mark_line()
        .encode(x=alt.X("ttm_h:Q", title="hours to expiry", scale=alt.Scale(domain=[168, 0])),
                y=alt.Y("coverage:Q", title="minutes with a candle", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                color=_color,
                tooltip=["ttm_h:Q", "mny:N", alt.Tooltip("coverage:Q", format=".1%"), alt.Tooltip("alive:Q", format=",", title="strike-minutes")])
        .properties(title="Candle coverage by time to expiry (all 51 weeklies pooled)", width=W, height=H)
    )
    cov_contract_chart = (
        alt.Chart(_by_ev).mark_line(point=alt.OverlayMarkDef(size=30))
        .encode(x=alt.X("expiry:T", title="expiry"),
                y=alt.Y("coverage:Q", title="minutes with a candle", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                color=_color,
                tooltip=["event_ticker:N", "mny:N", alt.Tooltip("coverage:Q", format=".1%")])
        .properties(title="Candle coverage by contract (whole life of each weekly)", width=W, height=H)
    )
    _cov = lambda d: d.candles.sum() / d.alive.sum()
    _near = sparsity_grid[sparsity_grid.mny == MNY_LABELS[0]]
    _far = sparsity_grid[sparsity_grid.mny == MNY_LABELS[2]]
    _exp = np.sort(sparsity_grid.expiry.unique())
    mo.vstack([
        mo.md(f"""
        **Where the data is thin.** Coverage is mostly a question of **distance from the money**:
        strikes within 1% of spot have a candle in {_cov(_near):.0%} of their minutes, strikes beyond 5%
        in {_cov(_far):.0%} (their quote sits at 0.01/0.99-type levels and rarely moves).
        """),
        cov_ttm_chart,
        mo.md(f"""
        **By time to maturity**, near-the-money coverage rises from {_cov(_near[_near.ttm_h >= 72]):.0%}
        more than 3 days out to {_cov(_near[_near.ttm_h < 24]):.0%} in the final day, where most volume
        trades. Far strikes stay around {_cov(_far):.0%} throughout.

        Two patterns are not data problems. The collapse to ~0% about 36–38 hours before expiry is the
        **weekly exchange closure**: Kalshi's published trading hours split Thursday into 00:00–03:00 and
        05:00–24:00 ET, so nothing is quoted Thursday 03:00–05:00 ET. The 24-hour waves are the **daily
        cycle of market making**, peaking around midday ET (≈101, 77, 53 and 29 hours before the Friday
        5pm expiry).
        """),
        cov_contract_chart,
        mo.md(f"""
        **Across contracts**, near-the-money coverage climbs from {_cov(_near[_near.expiry.isin(_exp[:4])]):.0%}
        in the first four weeklies (Oct 2025) to {_cov(_near[_near.expiry.isin(_exp[-4:])]):.0%} in the last
        four (Sep 2026), tracking the rise in liquidity (section 3).
        """),
    ])
    return


@app.cell
def _(CONTEXT, H, W, alt, candles, events, mo, np, pd, spot_close):
    # The ~36-38h dip: Kalshi's trading hours exclude Thursday 03:00-05:00 ET.
    _ETZ = "America/New_York"
    _et = candles.ts.dt.tz_convert(_ETZ)
    _min = (_et.dt.hour * 60 + _et.dt.minute).to_numpy()
    _sel = (_et.dt.dayofweek == 3).to_numpy() & (_min > 60) & (_min <= 420)   # Thursday 01:00-07:00 ET
    _t = pd.DataFrame({"date": _et[_sel].dt.date.to_numpy(), "m": _min[_sel], "volume": candles.volume.to_numpy()[_sel]})
    _n_thu = _t.date.nunique()

    # activity profile, average Thursday (candle minute ends)
    _prof = _t[_t.m.between(150, 330)].groupby("m").size().reindex(range(150, 331), fill_value=0) / _n_thu
    closure_profile = pd.DataFrame({"time": pd.Timestamp("2000-01-06") + pd.to_timedelta(_prof.index, unit="m"),
                                    "candles": _prof.to_numpy()})
    _shade = pd.DataFrame({"start": [pd.Timestamp("2000-01-06 03:00")], "end": [pd.Timestamp("2000-01-06 05:00")]})
    closure_chart = (
        alt.Chart(_shade).mark_rect(color=CONTEXT, opacity=0.35).encode(x="start:T", x2="end:T")
        + alt.Chart(closure_profile).mark_line().encode(
            x=alt.X("time:T", title="Thursday, ET (candle minute end)", axis=alt.Axis(format="%H:%M")),
            y=alt.Y("candles:Q", title="candles per minute (average Thursday)"),
            tooltip=[alt.Tooltip("time:T", format="%H:%M"), alt.Tooltip("candles:Q", format=".1f")])
    ).properties(title="Kalshi activity around the Thursday 03:00–05:00 ET closure (gray)", width=W, height=H)

    # every week?
    _t["window"] = pd.cut(_t.m, [60, 180, 300, 420], labels=["01:00–03:00", "03:00–05:00", "05:00–07:00"])
    closure_weeks = _t.groupby(["date", "window"], observed=False).size().unstack()
    closure_weeks["traded 03:00–05:00"] = _t[_t.window == "03:00–05:00"].groupby("date").volume.sum().reindex(closure_weeks.index, fill_value=0)
    _ratio = closure_weeks["03:00–05:00"] / closure_weeks["01:00–03:00"]
    _odd = closure_weeks[(_ratio > 0.2) | (closure_weeks["01:00–03:00"] < 0.25 * closure_weeks["01:00–03:00"].median())]

    # near-the-money quotes and BTC across the closure
    _px = spot_close.mean(axis=1).ffill()
    _c = candles.loc[_sel, ["event_ticker", "market_ticker", "strike", "ts", "bid_close", "ask_close"]]
    _targets = {"02:59": 179, "03:05": 185, "04:30": 270, "05:01": 301, "05:10": 310}
    _rows, _moves = [], []
    for _e, _close in events.set_index("event_ticker").close_time.items():
        _thu0 = (_close.tz_convert(_ETZ) - pd.Timedelta(days=1)).normalize()
        _ce = _c[_c.event_ticker == _e]
        _at = {_k: (_thu0 + pd.Timedelta(minutes=_v)).tz_convert("UTC") for _k, _v in _targets.items()}
        for _k, _tt in _at.items():
            _last = _ce[_ce.ts <= _tt].groupby("market_ticker", observed=True).last()
            _near = (_last.strike / _px.get(_tt) - 1).abs() < 0.01
            _b, _a = _last.bid_close[_near], _last.ask_close[_near]
            _two = (_b > 0) & (_a < 1)
            _rows.append({"time (ET)": _k, "spread_c": float(((_a - _b)[_two] * 100).median()) if _two.any() else np.nan})
        _moves.append(abs(np.log(_px.get(_at["05:01"]) / _px.get(_at["02:59"]))) * 1e4)
    closure_quotes = (pd.DataFrame(_rows).groupby("time (ET)", sort=False).spread_c.median()
                      .rename("median near-the-money spread (¢)").reset_index())
    closure_btc_move_bps = pd.Series(_moves)
    _sig = 0.43 * np.sqrt(36 / 8760)  # sigma*sqrt(tau) for an ATM weekly ~36h from expiry
    mo.vstack([
        mo.md(f"""
        **The dip ~36–38 hours before expiry is the weekly exchange closure.** Kalshi's trading hours exclude
        Thursday 03:00–05:00 ET, which is exactly 38 → 36 hours before the Friday 5pm expiry (the same clock
        time in summer and winter, since both are set in ET).
        """),
        closure_chart,
        mo.md(f"""
        - **Every week**: on all {_n_thu} Thursdays, activity in the closed window falls to a median
          {_ratio.median():.0%} of the two hours before ({closure_weeks['03:00–05:00'].median():.0f} vs
          {closure_weeks['01:00–03:00'].median():.0f} candles).
        - **Sharp edges**: a burst at 03:00–03:01 as market makers cancel their quotes, near silence until
          05:00, and a second burst at 05:01 when they re-post.
        - **No trading**: {closure_weeks['traded 03:00–05:00'].sum():,.0f} contracts changed hands in the closed
          window across all weeks (vs {_t[_t.window == '01:00–03:00'].volume.sum():,.0f} in the two hours
          before), all of them on one Thursday.
        - **What is left is wide and stale**: near-the-money spreads go from ~3¢ before the closure to
          ~20¢ during it, while BTC keeps trading and moves a median {closure_btc_move_bps.median():.0f} bps
          (90th percentile {closure_btc_move_bps.quantile(.9):.0f}, max {closure_btc_move_bps.max():.0f}) between
          02:59 and 05:01. For an at-the-money weekly that is roughly
          {100 * 0.3989 * closure_btc_move_bps.median() * 1e-4 / _sig:.0f}¢ of fair value at the median.
        """),
        mo.hstack([mo.ui.table(closure_quotes, selection=None, show_download=False),
                   mo.ui.table(_odd.reset_index().astype({"date": str}), selection=None, show_download=False)],
                  justify="start", gap=2),
        mo.md("""
        The two irregular Thursdays (right): **2026-05-21**, when quoting and trading resumed around 03:30
        (an early reopening), and **2026-02-12**, when activity had already collapsed from midnight (an
        incident; it explains the dip for the Feb 13 weekly in the coverage-by-contract chart).
        """),
    ])
    return


@app.cell
def _(
    AGE_LABELS,
    MNY_LABELS,
    mo,
    pd,
    quote_age_counts,
    sparsity_grid,
    unfillable_minutes,
):
    quote_age_table = pd.DataFrame(quote_age_counts / quote_age_counts.sum(axis=1, keepdims=True),
                                   index=MNY_LABELS, columns=AGE_LABELS)
    _stale_near = quote_age_table.loc[MNY_LABELS[0], ["1–4 h", "4 h +"]].sum()
    mo.vstack([
        mo.md("""
        **How old do forward-filled quotes get?** Minutes since the strike's last candle, for every alive
        strike-minute after forward-filling (share of strike-minutes per row):
        """),
        mo.ui.table(quote_age_table.map("{:.2%}".format).rename_axis("strike vs spot").reset_index(),
                    selection=None, show_download=False),
        mo.md(f"""
        Near the money, {quote_age_table.loc[MNY_LABELS[0], AGE_LABELS[0]]:.0%} of minutes have a fresh candle
        and only {_stale_near:.1%} carry a quote older than an hour. Only **{unfillable_minutes:,}** of
        {int(sparsity_grid.alive.sum()):,} alive strike-minutes come before a strike's first candle and cannot
        be filled.

        **Cleaning rule for sparse candles** (implemented in `02_clean_data.py`):

        1. Build each strike's 1-minute grid over its own life (open → expiry).
        2. Forward-fill the last closing bid/ask into minutes without a candle (validated above);
           filled minutes get volume 0 and flat OHLC equal to the carried quote.
        3. Leave minutes before a strike's first candle empty ({unfillable_minutes:,} minutes).
        4. Keep a `quote_age_min` column (minutes since the strike's last candle) so stale quotes can be
           filtered in analysis rather than silently trusted. Whether to cap the age (e.g. at 60 min) is
           a modelling choice; near the money it would affect {_stale_near:.1%} of minutes.
        5. Flag minutes when the exchange is closed, Thursday 03:00–05:00 ET (candle minute ends
           03:01–05:00): tight quotes are pulled at 03:00, nothing trades, and BTC keeps moving, so a quote
           carried through the closure is neither tradable nor current. Those minutes are excluded from
           pricing comparisons. The two irregular Thursdays are handled by the same flag plus
           `quote_age_min`: 2026-05-21 loses ~30 minutes of early-reopened trading, and the 2026-02-12
           overnight collapse shows up as stale quotes.
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10 · Cleaning

    The cleaning now lives in **`02_clean_data.py`**, which applies the rules from sections 8 and 9
    (with a 20¢ spread cap for usable mids), documents each rule with its evidence, and writes the clean
    dataset to `data/clean/`. The rules are implemented and unit-tested in `lib/cleaning.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
