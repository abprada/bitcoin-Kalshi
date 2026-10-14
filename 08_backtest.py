import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="08 Backtests: Kalshi and Deribit vol trades")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from lib import backtest, deribit, term

    ROOT = mo.notebook_dir()
    return ROOT, backtest, deribit, mo, np, pd, term


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 08 · Backtests

    Every trade is held to expiry, one per contract and entry rule (the Deribit file has two windows a day, so
    intraday holds cannot be marked). Code in `lib/backtest.py`; P&L in **basis points of spot per 1 BTC of
    straddle notional**, so Kalshi strips and Deribit straddles are in the same units. 95% intervals resample whole
    calendar weeks. For scale: an ATM straddle at 45% vol is worth about 75 bps with 3 hours left, 290 bps with one
    day and 500 bps with a week.

    - **Short vol on Kalshi** = one contract per strike within ±1.5 SD of spot (SD from BTC's trailing 7-day realized
      vol): sell YES above spot, buy YES below. A short straddle with the wings cut off. Taker: bid/ask and Kalshi's
      fee per 100-lot order. Maker: fills at mid, a quarter of the fee, an optimistic bound.
    - **Short vol on Deribit** = ATM straddle at Deribit's mark vol, settled on the 30-minute average spot before
      08:00 UTC; Deribit fees plus an *assumed* half-spread of 0.5 vol points per leg (we have no Deribit quotes).
    """)
    return


@app.cell(hide_code=True)
def _(ROOT, backtest, deribit, pd, term):
    spot = backtest.load_spot(ROOT)
    cum = term.cumulative_variance(spot)
    hour_profile = deribit.hourly_variance(spot)
    snaps = term.kalshi_weekly_atm(pd.read_parquet(ROOT / "data" / "derived" / "iv_snapshots.parquet"), usable_only=False)
    daily_atm = term.kalshi_daily_atm(ROOT, usable_only=False)
    # every Deribit BTC option trade (00_fetch_deribit.py) if fetched, else the collaborator's two-window file
    _full = ROOT / "data" / "raw" / "deribit_trades" / "btc_options_trades.parquet"
    deribit_trades = deribit.load_trades(_full if _full.exists() else ROOT / "data" / "deribit_btc_options.csv.gz")
    # ATM vols per expiry, day and window: only the collaborator's two windows, so 1-4 and 6 compare like for like
    deribit_obs = term.with_realized(term.deribit_atm(deribit_trades[deribit_trades.window != "other"]), cum)
    return cum, daily_atm, deribit_obs, deribit_trades, hour_profile, snaps, spot


@app.cell(hide_code=True)
def _(compare_ivr, mo):
    mo.vstack([mo.md(r"""
    ## Check 1 · The near-expiry filter

    03's "at least 2 strikes per SD" keeps a snapshot close to expiry only when vol is high (one SD is σ√τ), so it
    biases vol *levels* upward (`level_*`, medians). Implied minus realized is barely affected, and with all
    snapshots the 1–6 hour premium is estimated on twice the expiries: it holds.
    """), mo.ui.table(compare_ivr, selection=None, pagination=False, show_download=False)])
    return


@app.cell(hide_code=True)
def _(cum, daily_atm, deribit_obs, pd, snaps, term):
    _lo, _hi = term.DERIBIT_PERIOD

    def _ivr(d):
        d = d[(d.ts >= _lo) & (d.ts < _hi) & d.h.notna()]
        r = term.implied_vs_realized(d.drop(columns="strikes_per_sd", errors="ignore"))
        return r[(r.expiries >= 8) & (r.obs >= 30)].set_index(["market", "h"])

    _parts = []
    for _name, _all in (("Kalshi weekly", snaps), ("Kalshi daily", daily_atm)):
        _all = term.with_realized(_all, cum)
        _filt = _all[_all.strikes_per_sd >= 2]
        _a, _f = _ivr(_all), _ivr(_filt)
        _parts.append(pd.DataFrame({
            "level_filtered": _filt.groupby("h", observed=True).atm_iv.median(),
            "level_all": _all.groupby("h", observed=True).atm_iv.median(),
            "premium_filtered": _f.premium.droplevel(0), "expiries_filtered": _f.expiries.droplevel(0),
            "premium_all": _a.premium.droplevel(0), "ci_all": _a.ci_low.droplevel(0).round(1).astype(str) + " to " + _a.ci_high.droplevel(0).round(1).astype(str),
            "expiries_all": _a.expiries.droplevel(0)}).reset_index().assign(market=_name))
    compare_ivr = pd.concat(_parts).dropna(subset=["premium_all"]).round(3)

    _d = deribit_obs.assign(wd=deribit_obs.expiry.dt.dayofweek)
    deribit_weekday = pd.DataFrame({
        "nearest − one-week, all expiries": term.deribit_term_spread(_d).groupby("day").spread.median().describe(),
        "nearest − one-week, Tue–Fri nearest": term.deribit_term_spread(_d[_d.wd.between(1, 4) | (_d.hours > 96)])
        .groupby("day").spread.median().describe()}).round(2)
    deribit_levels = (_d[_d.window == "us_afternoon"].groupby(["wd", "h"], observed=True).atm_iv.median().unstack()
                      .rename(index=dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))).round(3))
    return compare_ivr, deribit_levels, deribit_weekday


@app.cell(hide_code=True)
def _(deribit_levels, deribit_weekday, mo):
    mo.vstack([mo.md(r"""
    ## Check 2 · Deribit's weekend expiries

    Saturday and Sunday expiries trade 10–20 points below weekdays at short horizons. Restricted to Tuesday–Friday
    nearest expiries (the ones that pair with Kalshi's Monday–Thursday dailies), Deribit's "nearest is below the
    one-week" disappears: it was mostly the weekend.
    """), mo.ui.table(deribit_weekday.reset_index(), selection=None, pagination=False, show_download=False),
        mo.md("Deribit ATM vol (median, US afternoon) by expiry weekday and time left:"),
        mo.ui.table(deribit_levels.reset_index(), selection=None, pagination=False, show_download=False)])
    return


@app.cell(hide_code=True)
def _(ROOT, backtest, cum):
    strips_daily = backtest.strip_trades(ROOT, "daily", [23, 12, 6, 3, 1], cum)
    strips_weekly = backtest.strip_trades(ROOT, "weekly", [167, 120, 72, 48, 24, 12, 6, 3, 1], cum)
    calendar = backtest.calendar_trades(ROOT, cum)
    return calendar, strips_daily, strips_weekly


@app.cell(hide_code=True)
def _(backtest, calendar, mo, pd, strips_daily, strips_weekly):
    def _tab(t):
        a = backtest.summarize(t, "bps_taker", by="hours")
        b = backtest.summarize(t, "bps_maker", by="hours")[["hours", "mean", "ci_low", "ci_high"]]
        return a.merge(b, on="hours", suffixes=("", "_maker")).round(1)

    _cal = pd.concat([backtest.summarize(calendar, c).assign(leg=c) for c in ("pnl_bps", "short_bps", "long_bps")]).round(1)
    mo.vstack([
        mo.md(r"""
        ## Strategy 1 · Short Kalshi vol at a fixed time before expiry

        One strip per contract, entered `hours` before the close and held to settlement (bps of spot, taker; the
        `_maker` columns are the optimistic mid-fill case). Only the last hours pay: 3 hours out on both contracts.
        Further out the spread and fee on 10–25 legs eat the premium.
        """),
        mo.md("**Dailies**"), mo.ui.table(_tab(strips_daily), selection=None, pagination=False, show_download=False),
        mo.md("**Weeklies**"), mo.ui.table(_tab(strips_weekly), selection=None, pagination=False, show_download=False),
        mo.md(r"""
        ## Strategy 2 · Kalshi calendar: short the daily, long the weekly

        23 hours before the daily's close: sell the daily strip, buy the weekly strip on the same dollar range, sell
        the weekly back at the daily's close. The weekly leg crosses the spread twice in a day; it loses on 90% of days.
        """),
        mo.ui.table(_cal, selection=None, pagination=False, show_download=False),
        mo.ui.table(backtest.summarize(calendar, "pnl_bps", by="weekday").round(1), selection=None, pagination=False,
                    show_download=False),
    ])
    return


@app.cell(hide_code=True)
def _(ROOT, backtest, cum, daily_atm, deribit_obs, hour_profile, snaps, spot):
    disloc = {}
    for _prod, _k in (("weekly", snaps), ("daily", daily_atm)):
        _p = backtest.dislocation_panel(_k, deribit_obs, hour_profile, _prod)
        disloc[_prod] = backtest.dislocation_trades(_p, ROOT, spot, cum, _prod)
    return (disloc,)


@app.cell(hide_code=True)
def _(backtest, disloc, np, pd):
    _rng = np.random.default_rng(1)

    def _placebo(t, f, col, n=5000):
        """Share of random same-days-left draws (one per signal trade) whose mean beats the signal's."""
        by = {d: g[col].dropna().to_numpy() for d, g in t.groupby("days_left")}
        sims = np.array([np.mean([_rng.choice(by[d]) for d in f.days_left]) for _ in range(n)])
        return (sims >= f[col].mean()).mean()

    _rows = []
    for _prod, _t in disloc.items():
        for _thr in (3, 6):
            for _sig, _mask in ((f"Kalshi cheap ≤ −{_thr}", _t.disloc <= -_thr), (f"Kalshi rich ≥ +{_thr}", _t.disloc >= _thr),
                                ("every contract (first obs)", _t.disloc.notna())):
                _f = backtest.first_signal(_t, _mask)
                for _col in ("deribit_short_bps", "deribit_long_bps", "kalshi_short_taker_bps", "kalshi_short_maker_bps"):
                    _s = backtest.summarize(_f, _col, n_boot=1000).iloc[0]
                    _rows.append({"product": _prod, "signal": _sig, "trade": _col.removesuffix("_bps"),
                                  **_s.drop(["median"]).to_dict(),
                                  "placebo_p": np.nan if _sig.startswith("every") else _placebo(_t, _f, _col)})
    disloc_table = pd.DataFrame(_rows).drop_duplicates(["product", "signal", "trade"])
    disloc_table = disloc_table[[c for c in disloc_table if c not in ("mean_2025Q4",)]].round(3)
    return (disloc_table,)


@app.cell(hide_code=True)
def _(disloc_table, mo):
    mo.vstack([mo.md(r"""
    ## Strategy 3 · Kalshi–Deribit dislocations

    Gap = Kalshi ATM vol − Deribit's (stretched to Kalshi's expiry), at each Deribit afternoon window. Normal = the
    gap's usual level at that many days left plus its median deviation over the previous 30 days. Dislocation =
    gap − normal. Weeklies pair with Deribit's Friday expiry, dailies with the next morning's (Tuesday–Friday). One
    trade per contract: the first time the signal fires. `placebo_p`: share of random trades at the same days left
    (one per signal trade) that did at least as well; it controls for the horizon mix. About 30 cells are tested, so
    expect one or two p < 0.05 by chance.
    """), mo.ui.table(disloc_table, selection=None, pagination=False, show_download=False)])
    return


@app.cell(hide_code=True)
def _():
    import altair as alt

    # reference palette, fixed order: slot 1 blue = before costs, slot 3 aqua = maker, slot 2 orange = taker
    INK, INK2, MUTED, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
    SLOT1, SLOT2, SLOT3 = "#2a78d6", "#eb6834", "#1baf7a"
    COSTS = ["before costs", "maker (mid, ¼ fee)", "taker"]
    COST_SCALE = alt.Scale(domain=COSTS, range=[SLOT1, SLOT3, SLOT2])

    @alt.theme.register("backtest_light", enable=True)
    def _theme() -> alt.theme.ThemeConfig:
        return {"config": {
            "background": "#fcfcfb", "font": "system-ui, -apple-system, 'Segoe UI', sans-serif",
            "view": {"stroke": None},
            "axis": {"gridColor": "#e1e0d9", "domainColor": CONTEXT, "tickColor": CONTEXT, "labelColor": MUTED,
                     "titleColor": INK2, "titleFontWeight": "normal"},
            "axisX": {"labelAngle": 0, "grid": False},
            "legend": {"labelColor": INK2, "titleColor": INK2, "orient": "top"},
            "header": {"labelColor": INK2, "titleColor": INK2},
            "title": {"color": INK, "anchor": "start", "fontWeight": "normal", "fontSize": 14,
                      "subtitleColor": MUTED, "subtitleFontSize": 11},
            "bar": {"cornerRadiusEnd": 4},
        }}

    def zero_rule():
        return alt.Chart().mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y=alt.datum(0))

    return COSTS, COST_SCALE, INK2, MUTED, SLOT1, SLOT2, alt, zero_rule


@app.cell(hide_code=True)
def _(COSTS, COST_SCALE, alt, backtest, pd, strips_daily, strips_weekly, zero_rule):
    _rows = []
    for _prod, _t, _hours in (("Kalshi daily", strips_daily, [23, 12, 6, 3, 1]),
                              ("Kalshi weekly", strips_weekly, [72, 24, 12, 6, 3, 1])):
        for _col, _lab in (("bps_mid", COSTS[0]), ("bps_maker", COSTS[1]), ("bps_taker", COSTS[2])):
            _s = backtest.summarize(_t[_t.hours.isin(_hours)], _col, by="hours", n_boot=1000)
            _rows.append(_s.assign(product=_prod, cost=_lab))
    _d = pd.concat(_rows)
    _d["entry"] = _d.hours.map(lambda h: f"{h}h")
    _order = ["72h", "24h", "23h", "12h", "6h", "3h", "1h"]
    _off = alt.XOffset("cost:N", sort=COSTS)
    _col = alt.Color("cost:N", title=None, scale=COST_SCALE, sort=COSTS)

    def _panel(prod, ytitle):
        x = alt.X("entry:N", sort=_order, title="entered this long before expiry, held to settlement")
        base = alt.Chart(_d[_d["product"] == prod])
        return alt.layer(
            base.mark_bar(size=12).encode(
                x=x, xOffset=_off, y=alt.Y("mean:Q", title=ytitle, scale=alt.Scale(domain=[-100, 60])), color=_col,
                tooltip=["product:N", "entry:N", "cost:N", alt.Tooltip("mean:Q", format="+.1f"),
                         alt.Tooltip("ci_low:Q", format="+.1f"), alt.Tooltip("ci_high:Q", format="+.1f"), "trades:Q"]),
            base.mark_rule(color="#52514e", strokeWidth=1.2).encode(x=x, xOffset=_off, y="ci_low:Q", y2="ci_high:Q"),
            base.mark_rule(color="#898781", strokeDash=[4, 3]).encode(y=alt.datum(0)),
        ).properties(width=320, height=240, title={"text": prod, "fontSize": 12, "color": "#52514e"})

    fig_strips = alt.hconcat(_panel("Kalshi daily", "mean P&L (bps of spot per 1 BTC straddle)"),
                             _panel("Kalshi weekly", None)).properties(title={
        "text": "Short Kalshi vol: only the last hours pay, and costs take most of it",
        "subtitle": "Mean per trade, 95% interval (resampling weeks). Earlier entries pay the spread and fee on 10–25 strikes for a premium that isn't there."})
    fig_strips
    return (fig_strips,)


@app.cell(hide_code=True)
def _(COSTS, COST_SCALE, alt, backtest, calendar, pd, zero_rule):
    _rows = []
    for _leg, _mid, _tk in (("short daily", "short_mid_bps", "short_bps"), ("long weekly", "long_mid_bps", "long_bps"),
                            ("calendar total", "pnl_mid_bps", "pnl_bps")):
        for _c, _lab in ((_mid, COSTS[0]), (_tk, COSTS[2])):
            _rows.append(backtest.summarize(calendar, _c, n_boot=1000).assign(leg=_leg, cost=_lab))
    _d = pd.concat(_rows)
    _legs = ["short daily", "long weekly", "calendar total"]
    _x = alt.X("leg:N", sort=_legs, title=None)
    _off = alt.XOffset("cost:N", sort=COSTS)
    _base = alt.Chart()
    fig_calendar = alt.layer(
        _base.mark_bar(size=26).encode(x=_x, xOffset=_off, y=alt.Y("mean:Q", title="mean P&L per day (bps of spot per 1 BTC straddle)"),
                                       color=alt.Color("cost:N", title=None, sort=[COSTS[0], COSTS[2]],
                                                       scale=alt.Scale(domain=[COSTS[0], COSTS[2]], range=["#2a78d6", "#eb6834"])),
                                       tooltip=["leg:N", "cost:N", alt.Tooltip("mean:Q", format="+.1f"), "trades:Q"]),
        _base.mark_rule(color="#52514e", strokeWidth=1.2).encode(x=_x, xOffset=_off, y="ci_low:Q", y2="ci_high:Q"),
        _base.mark_text(dy=-8, color="#52514e", fontSize=11).encode(
            x=_x, xOffset=_off, y=alt.Y("hi:Q"), text=alt.Text("mean:Q", format="+.0f")).transform_calculate(hi="max(datum.ci_high, 0)"),
        zero_rule(), data=_d,
    ).properties(width=460, height=260, title={
        "text": "Kalshi calendar: buying the weekly for one day costs more than the daily earns",
        "subtitle": [f"Short the daily 23h before its close, long the weekly on the same strikes, sold back at the daily's close; {len(calendar)} days.",
                     "The weekly round trip crosses the spread twice on every strike."]})
    fig_calendar
    return (fig_calendar,)


@app.cell(hide_code=True)
def _(SLOT1, SLOT2, alt, backtest, disloc, pd, zero_rule):
    _w, _dl = disloc["weekly"], disloc["daily"]
    _specs = [("A · Kalshi weekly cheap ≤ −3 → sell Deribit", _w, _w.disloc <= -3, "deribit_short_bps", "taker"),
              ("B · Kalshi daily rich ≥ +3 → sell Kalshi", _dl, _dl.disloc >= 3, "kalshi_short_taker_bps", "taker"),
              ("B · Kalshi daily rich ≥ +3 → sell Kalshi", _dl, _dl.disloc >= 3, "kalshi_short_maker_bps", "maker (mid, ¼ fee)")]
    _rows = []
    for _name, _t, _m, _c, _cost in _specs:
        _f = backtest.first_signal(_t, _m).dropna(subset=[_c])
        _f = _f.assign(q=_f.entry.dt.tz_convert(None).dt.to_period("Q").astype(str))
        _g = _f.groupby("q")[_c].agg(mean="mean", n="size").reset_index()
        _rows.append(_g.assign(strategy=_name, cost=_cost))
    _d = pd.concat(_rows)
    _d["label"] = _d.n.astype(str) + " trades"
    _d["ypos"] = _d["mean"].clip(lower=0)
    _x = alt.X("q:N", title=None)
    _off = alt.XOffset("cost:N", sort=["taker", "maker (mid, ¼ fee)"])
    _col = alt.Color("cost:N", title=None, scale=alt.Scale(domain=["taker", "maker (mid, ¼ fee)"], range=[SLOT2, "#1baf7a"]))
    _base = alt.Chart()
    fig_fade = alt.layer(
        _base.mark_bar(size=22).encode(x=_x, xOffset=_off, y=alt.Y("mean:Q", title="mean P&L per trade (bps of spot per 1 BTC straddle)"),
                                       color=_col, tooltip=["strategy:N", "q:N", "cost:N", alt.Tooltip("mean:Q", format="+.1f"), "n:Q"]),
        _base.mark_text(dy=-6, color="#52514e", fontSize=10).encode(x=_x, xOffset=_off, y="ypos:Q", text="n:Q"),
        zero_rule(),
    ).properties(width=300, height=240).facet(column=alt.Column("strategy:N", title=None), data=_d).properties(title={
        "text": "Dislocation trades: A is one quarter's story; B is not clearly positive at taker costs",
        "subtitle": ["One trade per contract, mean per quarter; numbers over bars are trade counts. A has 16 trades in all, 3 at the −6 threshold.",
                     "B over the whole sample: taker +8 bps (95% interval −18 to +35), maker +25 (−1 to +49)."]})
    fig_fade
    return (fig_fade,)


@app.cell(hide_code=True)
def _(INK2, MUTED, SLOT1, SLOT2, alt, disloc, np, pd):
    _rng = np.random.default_rng(0)
    _rows = []
    for _prod, _t in disloc.items():
        _t = _t.dropna(subset=["rv_deribit", "deribit_iv", "kalshi_iv"]).assign(
            k=lambda x: x.kalshi_iv / x.ratio, week=lambda x: x.entry.dt.tz_convert(None).dt.to_period("W").astype(str))
        _fit = lambda d: np.linalg.lstsq(np.c_[np.ones(len(d)), d.deribit_iv, d.k], d.rv_deribit, rcond=None)[0]
        _wk = [g for _, g in _t.groupby("week")]
        _b = _fit(_t)
        _bs = np.array([_fit(pd.concat([_wk[i] for i in _rng.integers(0, len(_wk), len(_wk))])) for _ in range(1000)])
        for _j, _name in ((1, "Deribit implied vol"), (2, "Kalshi implied vol")):
            _rows.append({"product": f"{ {'weekly': 'weeklies', 'daily': 'dailies'}[_prod]} ({len(_t)} obs)", "variable": _name, "coef": _b[_j],
                          "lo": np.percentile(_bs[:, _j], 2.5), "hi": np.percentile(_bs[:, _j], 97.5)})
    _d = pd.DataFrame(_rows)
    _y = alt.Y("variable:N", title=None, sort=["Deribit implied vol", "Kalshi implied vol"])
    _col = alt.Color("variable:N", title=None, legend=None,
                     scale=alt.Scale(domain=["Deribit implied vol", "Kalshi implied vol"], range=[SLOT2, SLOT1]))
    _base = alt.Chart()
    fig_signal = alt.layer(
        _base.mark_rule(strokeWidth=2).encode(y=_y, x=alt.X("lo:Q", title="coefficient on realized vol (95% interval, resampling weeks)"),
                                              x2="hi:Q", color=_col),
        _base.mark_point(filled=True, size=90, stroke="#fcfcfb", strokeWidth=2).encode(
            y=_y, x="coef:Q", color=_col, tooltip=["product:N", "variable:N", alt.Tooltip("coef:Q", format=".2f"),
                                                   alt.Tooltip("lo:Q", format=".2f"), alt.Tooltip("hi:Q", format=".2f")]),
        _base.mark_text(align="left", dx=8, dy=-10, color=INK2, fontSize=11).encode(y=_y, x="coef:Q", text=alt.Text("coef:Q", format=".2f")),
        alt.Chart().mark_rule(color=MUTED, strokeDash=[4, 3]).encode(x=alt.datum(0)),
    ).properties(width=300, height=90).facet(row=alt.Row("product:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
                                             data=_d).properties(title={
        "text": "Kalshi's vol adds nothing to Deribit's in predicting what BTC then does",
        "subtitle": ["Realized vol to Deribit's expiry regressed on both markets' ATM implied vols.",
                     "'Dislocation predicts Deribit's error' can arise from noise in Deribit's vol alone: it enters both sides."]})
    fig_signal
    return (fig_signal,)


@app.cell(hide_code=True)
def _(MUTED, SLOT1, SLOT2, alt, pd, term):
    # The friend's Oct 9 2025 case: $1M buys 1.75M contracts on the $123,000 strike (about 57c), edge 1.4c each,
    # hedged with the $122,750 / $123,250 call spread: 1.75M / $500 = 3,500 BTC per leg.
    _n, _p, _edge, _F = 1_750_000, 1_000_000 / 1_750_000, 0.014, 123_000.0
    _kalshi_fee = 0.07 * _n * _p * (1 - _p)
    _btc_per_leg = _n / 500
    # 0.03% of spot per BTC (the 12.5%-of-price cap does not bind for these near-the-money calls); as one combo
    # (one leg bought, one sold) Deribit waives the cheaper leg's fee, so one leg's fee is paid
    _deribit_fee = _btc_per_leg * term.DERIBIT_FEE_RATE * _F
    _d = pd.DataFrame({"item": ["edge locked in", "Kalshi taker fee", "Deribit fee, opening the hedge (combo)"],
                       "usd": [_edge * _n, _kalshi_fee, _deribit_fee], "kind": ["edge", "cost", "cost"]})
    _y = alt.Y("item:N", sort=list(_d.item), title=None)
    fig_hedge = alt.layer(
        alt.Chart(_d).mark_bar(size=22).encode(
            y=_y, x=alt.X("usd:Q", title="US dollars", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("kind:N", legend=None, scale=alt.Scale(domain=["edge", "cost"], range=[SLOT1, SLOT2])),
            tooltip=["item:N", alt.Tooltip("usd:Q", format="$,.0f")]),
        alt.Chart(_d).mark_text(align="left", dx=6, color="#52514e", fontSize=11).encode(y=_y, x="usd:Q", text=alt.Text("usd:Q", format="$,.0f")),
    ).properties(width=440, height=120, title={
        "text": "The Oct 9 binary hedge: fees are larger than the edge",
        "subtitle": ["$1M in Kalshi's $123,000 binary (1.4¢ cheap) hedged with a $500 call spread on Deribit (3,500 BTC per leg).",
                     "Fees at Kalshi's and Deribit's schedules; before any spread, and before Deribit's 08:00 vs Kalshi's 5pm expiry mismatch."]})
    fig_hedge
    return (fig_hedge,)


@app.cell(hide_code=True)
def _(ROOT, backtest, cum, daily_atm, deribit_obs, hour_profile, snaps, spot):
    # all panel rows, including the first 30 days (no dislocation needed here)
    _legs = {}
    for _prod, _k in (("weekly", snaps), ("daily", daily_atm)):
        _p = backtest.dislocation_panel(_k, deribit_obs, hour_profile, _prod)
        _legs[_prod] = backtest.dislocation_trades(_p, ROOT, spot, cum, _prod, require_disloc=False)
    gap_cal = backtest.gap_calendar(_legs["weekly"], _legs["daily"])
    return (gap_cal,)


@app.cell(hide_code=True)
def _(COSTS, COST_SCALE, alt, backtest, gap_cal, mo, pd):
    _names = {"k_week": "buy Kalshi weekly", "d_week": "sell Deribit Friday", "week_spread": "= weekly spread bought",
              "k_day": "sell Kalshi daily", "d_day": "buy Deribit next-morning", "day_spread": "= daily spread sold",
              "total": "total"}
    _rows = []
    for _leg, _lab in _names.items():
        for _c, _cl in (("mid", COSTS[0]), ("taker", COSTS[2])):
            _rows.append(backtest.summarize(gap_cal, f"{_leg}_{_c}", n_boot=1000).assign(leg=_lab, cost=_cl))
    gap_cal_table = pd.concat(_rows)[["leg", "cost", "trades", "mean", "ci_low", "ci_high", "win"]].round(1)
    _x = alt.X("leg:N", sort=list(_names.values()), title=None, axis=alt.Axis(labelAngle=-20, labelLimit=160))
    _off = alt.XOffset("cost:N", sort=[COSTS[0], COSTS[2]])
    _base = alt.Chart()
    fig_gap_calendar = alt.layer(
        _base.mark_bar(size=18).encode(
            x=_x, xOffset=_off, y=alt.Y("mean:Q", title="mean P&L per day (bps of spot per 1 BTC straddle)"),
            color=alt.Color("cost:N", title=None, sort=[COSTS[0], COSTS[2]],
                            scale=alt.Scale(domain=[COSTS[0], COSTS[2]], range=["#2a78d6", "#eb6834"])),
            tooltip=["leg:N", "cost:N", alt.Tooltip("mean:Q", format="+.1f"), alt.Tooltip("ci_low:Q", format="+.1f"),
                     alt.Tooltip("ci_high:Q", format="+.1f"), "trades:Q"]),
        _base.mark_rule(color="#52514e", strokeWidth=1.2).encode(x=_x, xOffset=_off, y="ci_low:Q", y2="ci_high:Q"),
        _base.mark_rule(color="#898781", strokeDash=[4, 3]).encode(y=alt.datum(0)),
        data=gap_cal_table,
    ).properties(width=560, height=260, title={
        "text": "Long the weekly Kalshi−Deribit spread, short the daily one: no edge to start with",
        "subtitle": [f"Same afternoon window, {len(gap_cal)} days; each leg held to its own expiry. At matched times the daily gap "
                     f"({gap_cal.gap_d.mean():+.1f} pts) is no bigger than the weekly's ({gap_cal.gap_w.mean():+.1f}).",
                     "Before costs about zero; Kalshi's spreads and fees and Deribit's fees make it a clear loss."]})
    mo.vstack([mo.md(r"""
    ## Strategy 4 · Long the one-week Kalshi−Deribit spread, short the daily spread

    Each afternoon window: buy Kalshi's weekly strip and sell Deribit's Friday straddle (the one-week spread), sell
    Kalshi's daily strip (about a day left) and buy Deribit's next-morning straddle (the daily spread). Each leg held
    to its own expiry, all per 1 BTC of straddle. Deribit before costs = no fee, no spread; taker = fees plus the
    assumed half-spread.
    """), fig_gap_calendar, mo.ui.table(gap_cal_table, selection=None, pagination=False, show_download=False)])
    return (fig_gap_calendar,)


@app.cell(hide_code=True)
def _(COSTS, alt, calendar, gap_cal, pd):
    def _cum(t, mid, taker, name):
        t = t.sort_values("entry")
        return pd.concat([pd.DataFrame({"date": t.entry, "cum": t[c].cumsum(), "cost": lab, "panel": name})
                          for c, lab in ((mid, COSTS[0]), (taker, "after costs (taker)"))])

    _d = pd.concat([
        _cum(calendar, "pnl_mid_bps", "pnl_bps",
             f"Kalshi calendar: short daily, long weekly ({len(calendar)} days)"),
        _cum(gap_cal, "total_mid", "total_taker",
             f"Kalshi−Deribit spread calendar: long weekly spread, short daily ({len(gap_cal)} days)")])
    _dom = [COSTS[0], "after costs (taker)"]
    _col = alt.Color("cost:N", title=None, sort=_dom, scale=alt.Scale(domain=_dom, range=["#2a78d6", "#eb6834"]))
    _end = _d.sort_values("date").groupby(["panel", "cost"]).tail(1)

    def _panel(name, ytitle):
        d, e = _d[_d.panel == name], _end[_end.panel == name]
        x = alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y"))
        return alt.layer(
            alt.Chart(d).mark_rule(color="#898781", strokeDash=[4, 3]).encode(y=alt.datum(0)),
            alt.Chart(d).mark_line(strokeWidth=2, interpolate="step-after").encode(
                x=x, y=alt.Y("cum:Q", title=ytitle), color=_col,
                tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), "cost:N", alt.Tooltip("cum:Q", format="+,.0f")]),
            alt.Chart(e).mark_point(filled=True, size=70, stroke="#fcfcfb", strokeWidth=2).encode(x=x, y="cum:Q", color=_col),
            alt.Chart(e).mark_text(align="left", dx=8, fontSize=11, color="#52514e").encode(
                x=x, y="cum:Q", text=alt.Text("cum:Q", format="+,.0f")),
        ).properties(width=360, height=250, title={"text": name, "fontSize": 12, "color": "#52514e"})

    _p = list(dict.fromkeys(_d.panel))
    fig_calendar_cum = alt.hconcat(_panel(_p[0], "cumulative P&L (bps of spot, 1 BTC straddle per leg)"),
                                   _panel(_p[1], None)).resolve_scale(y="independent").properties(title={
        "text": "Both calendars are flat before costs; trading costs turn them into steady losses",
        "subtitle": "Cumulative P&L, one trade per day, each held as in the backtest. Before costs: in and out at mid, no fees. "
                    "After costs: Kalshi's bid/ask and fee; Deribit's fees and an assumed half-spread."})
    fig_calendar_cum
    return (fig_calendar_cum,)


@app.cell(hide_code=True)
def _(alt, cum, deribit_obs, pd, snaps, term):
    # implied vs realized by time to expiry, each market against its own expiry, root-mean-square, Tue-Fri Deribit
    _lo, _hi = term.DERIBIT_PERIOD
    _k = term.with_realized(snaps.drop(columns="strikes_per_sd"), cum)
    _d = deribit_obs[deribit_obs.expiry.dt.dayofweek.between(1, 4)]
    _obs = pd.concat([_k, _d.assign(market="Deribit")])
    _obs = _obs[(_obs.ts >= _lo) & (_obs.ts < _hi) & _obs.h.notna()]
    _r = term.implied_vs_realized(_obs)
    _r = _r[(_r.expiries >= 8) & (_r.obs >= 20)]
    _mid = dict(zip(term.HOUR_LABELS, [(a + b) / 2 for a, b in zip(term.HOUR_EDGES[:-1], term.HOUR_EDGES[1:])]))
    _long = pd.concat([
        _r.assign(series=_r.market.map({"Kalshi weekly": "Kalshi implied", "Deribit": "Deribit implied"}), vol=_r.implied),
        _r.assign(series=_r.market.map({"Kalshi weekly": "BTC realized to Kalshi's expiry",
                                        "Deribit": "BTC realized to Deribit's expiry"}), vol=_r.realized)])
    _long["hours"] = _long.h.astype(str).map(_mid)
    _dom = ["Kalshi implied", "BTC realized to Kalshi's expiry", "Deribit implied", "BTC realized to Deribit's expiry"]
    _enc = dict(
        x=alt.X("hours:Q", title="hours to expiry", scale=alt.Scale(domain=[170, 0])),
        y=alt.Y("vol:Q", title="annualized vol (root-mean-square)", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None, sort=_dom,
                        scale=alt.Scale(domain=_dom, range=["#2a78d6", "#2a78d6", "#eb6834", "#eb6834"])),
        strokeDash=alt.StrokeDash("series:N", sort=_dom, legend=None,
                                  scale=alt.Scale(domain=_dom, range=[[1, 0], [5, 3], [1, 0], [5, 3]])),
        detail="series:N")
    _tip = ["series:N", "h:N", alt.Tooltip("vol:Q", format=".1%"), "expiries:Q", "obs:Q"]
    fig_term_fixed = alt.layer(
        alt.Chart(_long).mark_line(strokeWidth=2).encode(**_enc),
        alt.Chart(_long).mark_point(filled=True, size=60, stroke="#fcfcfb", strokeWidth=1.5).encode(
            **{k: v for k, v in _enc.items() if k != "strokeDash"}, tooltip=_tip),
    ).properties(width=620, height=280, title={
        "text": "Kalshi charges more than BTC delivers in its last hours; Deribit doesn't",
        "subtitle": ["Each market against BTC's realized vol to its own expiry (solid: implied, dashed: realized). "
                     "Root-mean-square per bucket; Deribit Tue–Fri expiries; Jan 15 – Sep 30, 2026.",
                     "Kalshi's last hours are the busy US afternoon, Deribit's the quiet Asian night: both markets price that."]})
    term_fixed_table = _r[["market", "h", "expiries", "obs", "implied", "realized", "premium", "ci_low", "ci_high"]].round(3)
    fig_term_fixed
    return fig_term_fixed, term_fixed_table


@app.cell(hide_code=True)
def _(ROOT, backtest, cum, deribit_obs, deribit_trades, hour_profile, pd, snaps, spot):
    gap_A = backtest.gap_widening_trades(backtest.dislocation_panel(snaps, deribit_obs, hour_profile, "weekly"),
                                         ROOT, spot, cum)
    # B on a grid: dailies and weeklies, entered 6, 3 and 2 hours before Kalshi's close (Deribit priced at that time)
    gap_B_grid = pd.concat([backtest.last_hours_trades(ROOT, deribit_trades, spot, cum, hours_before=_h, product=_p)
                            .assign(hours_before=_h) for _p in ("daily", "weekly") for _h in (6, 3, 2)],
                           ignore_index=True)
    gap_B = gap_B_grid[(gap_B_grid["product"] == "daily") & (gap_B_grid.hours_before == 2)]
    return gap_A, gap_B, gap_B_grid


@app.cell(hide_code=True)
def _(COSTS, alt, backtest, gap_A, gap_B, mo, pd):
    _A_name = "A · buy the weekly gap a week out, sell it back Thursday"
    _B_name = "B · last 2 hours: sell Kalshi daily, buy Deribit next morning"
    _rows = []
    for _name, _t in ((_A_name, gap_A), (_B_name, gap_B)):
        for _leg, _lab in (("kalshi", "Kalshi leg"), ("deribit", "Deribit leg"), ("total", "total")):
            for _c, _cl in (("mid", COSTS[0]), ("taker", COSTS[2])):
                _rows.append(backtest.summarize(_t, f"{_leg}_{_c}", n_boot=1000).assign(trade=_name, leg=_lab, cost=_cl))
    gap_trades_table = pd.concat(_rows)[["trade", "leg", "cost", "trades", "mean", "ci_low", "ci_high", "win"]].round(1)

    _off = alt.XOffset("cost:N", sort=[COSTS[0], COSTS[2]])
    _col = alt.Color("cost:N", title=None, sort=[COSTS[0], COSTS[2]],
                     scale=alt.Scale(domain=[COSTS[0], COSTS[2]], range=["#2a78d6", "#eb6834"]))

    def _panel(name, ytitle):
        d = gap_trades_table[gap_trades_table.trade == name]
        x = alt.X("leg:N", sort=["Kalshi leg", "Deribit leg", "total"], title=None)
        return alt.layer(
            alt.Chart(d).mark_bar(size=22).encode(
                x=x, xOffset=_off, y=alt.Y("mean:Q", title=ytitle, scale=alt.Scale(domain=[-320, 160])), color=_col,
                tooltip=["leg:N", "cost:N", alt.Tooltip("mean:Q", format="+.1f"), alt.Tooltip("ci_low:Q", format="+.1f"),
                         alt.Tooltip("ci_high:Q", format="+.1f"), "trades:Q"]),
            alt.Chart(d).mark_rule(color="#52514e", strokeWidth=1.2).encode(x=x, xOffset=_off, y="ci_low:Q", y2="ci_high:Q"),
            alt.Chart(d).mark_rule(color="#898781", strokeDash=[4, 3]).encode(y=alt.datum(0)),
        ).properties(width=270, height=240, title={"text": name, "fontSize": 12, "color": "#52514e"})

    _A = gap_A.assign(widening=gap_A.gap_out - gap_A.gap_in)
    _sc = alt.layer(
        alt.Chart(_A).mark_rule(color="#898781", strokeDash=[4, 3]).encode(y=alt.datum(0)),
        alt.Chart(_A).mark_point(filled=True, size=60, color="#2a78d6", stroke="#fcfcfb", strokeWidth=1.5).encode(
            x=alt.X("widening:Q", title="gap at exit − gap at entry (vol points)"),
            y=alt.Y("total_mid:Q", title="A before costs (bps)"),
            tooltip=["event:N", alt.Tooltip("gap_in:Q", format="+.1f"), alt.Tooltip("gap_out:Q", format="+.1f"),
                     alt.Tooltip("total_mid:Q", format="+.0f")]),
        alt.Chart(_A).transform_regression("widening", "total_mid").mark_line(color="#2a78d6", strokeWidth=2).encode(
            x="widening:Q", y="total_mid:Q"),
    ).properties(width=230, height=240, title={"text": "A pays when the gap widens", "fontSize": 12, "color": "#52514e"})
    fig_gap_trades = alt.hconcat(
        _panel(_A_name, "mean P&L per trade (bps of spot, 1 BTC)"), _panel(_B_name, None), _sc,
    ).properties(title={
        "text": "Trading the gap as it widens: A makes money before costs, Kalshi's spreads take it all",
        "subtitle": [f"A: {len(gap_A)} weeklies, the ATM gap goes from {gap_A.gap_in.mean():+.1f} to {gap_A.gap_out.mean():+.1f} "
                     f"points; B: {len(gap_B)} dailies. 95% intervals resample weeks.",
                     "B's Deribit leg mostly covers the night after Kalshi has expired, so it adds cost and noise, not a hedge."]})
    mo.vstack([mo.md(r"""
    ## Strategy 5 · Trading the Kalshi−Deribit gap as it widens into expiry

    Built from Section 1: the ATM gap is about 0 a week out and +4 to +5 points with a day left (A), and largest
    in the last hours (B). Holding the legs to expiry (Strategy 4) pays the entry gap and never collects the
    widening; A sells the spread back instead.

    - **A.** Each Kalshi weekly (Friday 5pm ET) with Deribit's Friday 08:00 UTC expiry. First US-afternoon window
      with 7 days left (else 6, 5): buy the Kalshi strip (strikes within ±1.5 SD of spot) and sell the Deribit iron
      butterfly with the same payoff (straddle at spot, wings at the strip's edges, 1 BTC), at Deribit's ATM vol.
      Thursday's afternoon window (Kalshi about 25 hours left, Deribit about 11): sell the Kalshi strikes back and buy
      the butterfly back. Deribit's options are priced at its ATM vol (no smile); its costs are fees (combo
      discount) plus the assumed half-spread, in and out.
    - **B.** Each Monday–Thursday daily at 19:15 UTC (1h45 left in summer, 2h45 in winter): sell the Kalshi strip,
      buy 1 BTC of Deribit's ATM straddle expiring 08:00 UTC the next morning; both to expiry.
    """), fig_gap_trades, mo.ui.table(gap_trades_table, selection=None, pagination=False, show_download=False)])
    return (fig_gap_trades,)


@app.cell(hide_code=True)
def _(ROOT, backtest, cum, deribit_trades, gap_A, spot):
    # options that traded within 90 minutes of the entry and of the exit (the full trade file covers every hour)
    gap_A_traded = backtest.gap_widening_traded(gap_A, deribit_trades, ROOT, spot, cum, window_min=90)
    return (gap_A_traded,)


@app.cell(hide_code=True)
def _(alt, gap_A_traded, mo, pd):
    _t = gap_A_traded.sort_values("exit")
    _labels = {"mid": "before costs", "maker": "maker on Kalshi, after costs", "taker": "after costs (taker)"}
    _d = pd.concat([pd.DataFrame({"date": pd.concat([pd.Series([_t.entry.min()]), _t.exit]).dt.tz_convert(None),
                                  "cum": [0.0] + list(_t[f"total_{c}_usd"].cumsum()), "series": lab})
                    for c, lab in _labels.items()])
    _dom = list(_labels.values())
    fig_gap_A_cum = alt.layer(
        alt.Chart(_d).mark_rule(color="#898781", strokeDash=[4, 3]).encode(y=alt.datum(0)),
        alt.Chart(_d).mark_line(strokeWidth=2, interpolate="step-after").encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", tickCount="month")),
            y=alt.Y("cum:Q", title="cumulative P&L ($ per 1 BTC position)", axis=alt.Axis(format=",.0f")),
            color=alt.Color("series:N", title=None, sort=_dom,
                            scale=alt.Scale(domain=_dom, range=["#2a78d6", "#1baf7a", "#eb6834"])),
            tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), "series:N", alt.Tooltip("cum:Q", format="+,.0f")]),
    ).properties(width=640, height=300, title={
        "text": "Long the weekly Kalshi−Deribit gap, sold back Thursday: positive before costs, not after",
        "subtitle": [f"{len(_t)} weeks with a Deribit iron condor whose four options traded within 90 minutes of both entry and exit; "
                     "Kalshi's weekly strikes over the same ranges. Each Deribit option at its own trade's mark; Kalshi at its quotes.",
                     "Maker: Kalshi at mid with a quarter of the fee, Deribit fees and the assumed half-spread. "
                     "Taker: Kalshi at bid/ask with the full fee."]})
    mo.vstack([mo.md(r"""
    **Trade A with only contracts that actually traded.** Deribit's Friday 08:00 options that traded in the US
    afternoon window on both the entry day (about 7 days out) and the exit day (Thursday): sell the traded call nearest
    above spot and put nearest below (within $2,000), buy the furthest traded call and put out to 1.5 SD (at least
    $1,000 beyond). Kalshi: buy YES on the weekly's strikes between the call and its wing, sell YES between the put
    wing and the put: the same payoff. Weeks without such a condor are left out.
    """), fig_gap_A_cum,
        mo.ui.table(gap_A_traded.drop(columns=[c for c in gap_A_traded if c.startswith("total_") and not c.endswith("usd")])
                    .round(1), selection=None, pagination=False, show_download=False)])
    return (fig_gap_A_cum,)


@app.cell(hide_code=True)
def _(COSTS, alt, backtest, gap_B_grid, mo, pd):
    _rows = []
    for (_p, _h), _t in gap_B_grid.groupby(["product", "hours_before"]):
        for _leg, _lab in (("kalshi", "Kalshi leg alone"), ("total", "Kalshi + Deribit hedge")):
            for _c, _cl in (("mid", COSTS[0]), ("taker", COSTS[2])):
                _rows.append(backtest.summarize(_t, f"{_leg}_{_c}", n_boot=1000).assign(
                    product=f"Kalshi {_p}", entry=f"{_h}h before close", leg=_lab, cost=_cl))
    gap_B_table = pd.concat(_rows)[["product", "entry", "leg", "cost", "trades", "mean", "ci_low", "ci_high", "win"]].round(1)
    _entries = ["6h before close", "3h before close", "2h before close"]
    _col = alt.Color("cost:N", title=None, sort=[COSTS[0], COSTS[2]],
                     scale=alt.Scale(domain=[COSTS[0], COSTS[2]], range=["#2a78d6", "#eb6834"]))

    def _panel(product, leg, ytitle):
        d = gap_B_table[(gap_B_table["product"] == product) & (gap_B_table.leg == leg)]
        x = alt.X("entry:N", sort=_entries, title=None)
        off = alt.XOffset("cost:N", sort=[COSTS[0], COSTS[2]])
        return alt.layer(
            alt.Chart(d).mark_bar(size=20).encode(
                x=x, xOffset=off, y=alt.Y("mean:Q", title=ytitle, scale=alt.Scale(domain=[-120, 60])), color=_col,
                tooltip=["product:N", "entry:N", "leg:N", "cost:N", alt.Tooltip("mean:Q", format="+.1f"),
                         alt.Tooltip("ci_low:Q", format="+.1f"), alt.Tooltip("ci_high:Q", format="+.1f"), "trades:Q"]),
            alt.Chart(d).mark_rule(color="#52514e", strokeWidth=1.2).encode(x=x, xOffset=off, y="ci_low:Q", y2="ci_high:Q"),
            alt.Chart(d).mark_rule(color="#898781", strokeDash=[4, 3]).encode(y=alt.datum(0)),
        ).properties(width=250, height=200, title={"text": f"{product}: {leg}", "fontSize": 12, "color": "#52514e"})

    _y = "mean P&L per trade (bps of spot, 1 BTC)"
    fig_gap_B_grid = alt.vconcat(
        alt.hconcat(_panel("Kalshi daily", "Kalshi leg alone", _y), _panel("Kalshi daily", "Kalshi + Deribit hedge", None)),
        alt.hconcat(_panel("Kalshi weekly", "Kalshi leg alone", _y), _panel("Kalshi weekly", "Kalshi + Deribit hedge", None)),
    ).properties(title={
        "text": "Selling Kalshi's last hours pays on its own; adding Deribit's next-morning straddle does not help",
        "subtitle": ["Sell the Kalshi strip (±1.5 SD) 6, 3 or 2 hours before its 5pm ET close; hedge: buy 1 BTC of Deribit's ATM "
                     "straddle expiring 08:00 UTC next morning (Saturday for the weekly), priced from its trades within 30 minutes.",
                     "Mean per trade, 95% interval (resampling weeks). Every Deribit BTC option trade, Jan 15 – Sep 30, 2026."]})
    mo.vstack([mo.md(r"""
    ### Trade B on the full Deribit data

    With every Deribit trade, the hedge can be priced at the moment the Kalshi leg is sold. Deribit still has no
    option that ends at Kalshi's 5pm close: the nearest expiry is 08:00 UTC the next morning, so the hedge's life is
    mostly the night after Kalshi has settled.
    """), fig_gap_B_grid, mo.ui.table(gap_B_table, selection=None, pagination=False, show_download=False)])
    return fig_gap_B_grid, gap_B_table


@app.cell(hide_code=True)
def _(alt, gap_B_grid, pd):
    _sel = gap_B_grid[gap_B_grid.hours_before == 3]
    _series = {"kalshi_mid": "Kalshi alone, before costs", "kalshi_taker": "Kalshi alone, after costs (taker)",
               "total_mid": "with Deribit hedge, before costs", "total_taker": "with Deribit hedge, after costs"}
    _parts = []
    for _p, _t in _sel.groupby("product"):
        _t = _t.sort_values("entry")
        for _c, _lab in _series.items():
            _parts.append(pd.DataFrame({"date": _t.entry.dt.tz_convert(None), "series": _lab, "product": f"Kalshi {_p}",
                                        "cum": (_t[_c] * _t.spot / 1e4).cumsum()}))
    _d = pd.concat(_parts)
    _dom = list(_series.values())
    _enc = dict(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", tickCount="month")),
        y=alt.Y("cum:Q", title="cumulative P&L ($ per 1 BTC position)", axis=alt.Axis(format=",.0f")),
        color=alt.Color("series:N", title=None, sort=_dom, scale=alt.Scale(domain=_dom, range=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])),
        tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), "series:N", alt.Tooltip("cum:Q", format="+,.0f")])

    def _panel(product):
        d = _d[_d["product"] == product]
        return alt.layer(
            alt.Chart(d).mark_rule(color="#898781", strokeDash=[4, 3]).encode(y=alt.datum(0)),
            alt.Chart(d).mark_line(strokeWidth=2, interpolate="step-after").encode(**_enc),
        ).properties(width=380, height=260, title={"text": f"{product} ({d.date.nunique()} trades)", "fontSize": 12, "color": "#52514e"})

    fig_gap_B_cum = alt.hconcat(_panel("Kalshi daily"), _panel("Kalshi weekly")).resolve_scale(y="independent").properties(title={
        "text": "Selling Kalshi 3 hours before its close: cumulative P&L, with and without a Deribit hedge",
        "subtitle": "Kalshi strip at its quotes (after costs: bid/ask and fee); Deribit next-morning ATM straddle at its mark "
                    "(after costs: fees and an assumed half-spread of 0.5 vol points per leg)."})
    fig_gap_B_cum
    return (fig_gap_B_cum,)


@app.cell(hide_code=True)
def _(
    ROOT,
    fig_calendar,
    fig_calendar_cum,
    fig_fade,
    fig_gap_A_cum,
    fig_gap_B_cum,
    fig_gap_B_grid,
    fig_gap_calendar,
    fig_gap_trades,
    fig_hedge,
    fig_signal,
    fig_strips,
    fig_term_fixed,
    mo,
):
    _out = ROOT / "outputs" / "backtest"
    _out.mkdir(parents=True, exist_ok=True)
    _figs = {"1_short_kalshi_by_entry": fig_strips, "2_calendar_legs": fig_calendar, "3_dislocation_fade": fig_fade,
             "4_signal_regression": fig_signal, "5_hedge_fees_vs_edge": fig_hedge,
             "6_gap_calendar": fig_gap_calendar, "7_calendars_cumulative": fig_calendar_cum,
             "8_implied_vs_realized_own_expiry": fig_term_fixed,
             "9_gap_widening_trades": fig_gap_trades,
             "10_gap_A_traded_cumulative": fig_gap_A_cum,
             "11_last_hours_grid": fig_gap_B_grid, "12_last_hours_cumulative": fig_gap_B_cum}
    for _name, _chart in _figs.items():
        _chart.save(_out / f"{_name}.png", scale_factor=2)
    mo.md(f"**Figures:** {len(_figs)} PNGs written to `outputs/backtest/`.")
    return


if __name__ == "__main__":
    app.run()
