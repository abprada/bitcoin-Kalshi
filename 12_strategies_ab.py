import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="12 Dislocation strategies A and B")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    from lib import strategy_ab as ab
    from lib import vol_spread as vs

    return ab, alt, mo, np, pd, vs


@app.cell(hide_code=True)
def _(alt):
    # Chart styling as in 05/06. Execution modes in a fixed order: maker (blue), half spread (aqua), taker (orange).
    INK, INK2, MUTED, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
    SLOT1, SLOT2, SLOT3, SLOT4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    W, H = 320, 230


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
                "header": {"labelColor": INK2, "labelFontSize": 12},
                "line": {"color": SLOT1, "strokeWidth": 2},
                "point": {"color": SLOT1, "size": 64, "filled": True},
            }
        }

    MODE_COLORS = {"maker": SLOT1, "half spread": SLOT3, "taker": SLOT2, "gross": "#898781"}
    return H, MODE_COLORS, MUTED, SLOT1, SLOT2, SLOT3, W


@app.cell(hide_code=True)
def _(ab, mo, pd):
    ROOT = mo.notebook_dir()
    D = ROOT / "data" / "derived" / "strategies"
    FIG = ROOT / "outputs" / "figures"
    trades = pd.read_parquet(D / "ab_trades.parquet")
    panel = pd.read_parquet(D / "disl_panel.parquet")
    START, END, SPLIT = "2026-01-15", "2026-10-01", "2026-05-01"
    # the two rules as presented: 30-day normal, held to expiry; thresholds from section 2's grid
    RULES = {"A": dict(window=30, threshold=6, hold="expiry"), "B": dict(window=30, threshold=3, hold="expiry")}
    MODES = ["gross", "maker", "half spread", "taker"]


    def rule_book(st, mode, **kw):
        r = {**RULES[st], **kw}
        return ab.book(trades[trades.window == r["window"]], st, r["threshold"], r["hold"], mode)


    headline = pd.DataFrame([{"strategy": st, "execution": m, **ab.metrics(rule_book(st, m), START, END)}
                             for st in RULES for m in MODES])
    return END, FIG, MODES, RULES, SPLIT, START, headline, panel, rule_book, trades


@app.cell(hide_code=True)
def _(headline, mo):
    _h = headline.set_index(["strategy", "execution"])
    _a, _b = _h.loc["A"], _h.loc["B"]
    mo.md(f"""
    # 12 · Two dislocation strategies, backtested: A and B

    *Kalshi vs Deribit, January – September 2026. Trades from `08_strategy_data.py` (`lib/strategy_ab.py`,
    `lib/vol_spread.py`); this notebook only reads them. Dislocation = Kalshi's ATM vol minus Deribit's at the same
    expiry, minus its median over the previous 30 days for the same contract type and time to expiry (11).*

    | | **A · sell Deribit vol when Kalshi is unusually cheap** | **B · sell Kalshi vol when Kalshi is unusually rich** |
    |---|---|---|
    | Market | Deribit weekly straddle (the Friday expiry) | Kalshi daily straddle (strip of binaries) |
    | Signal | weekly dislocation ≤ −6 vol points | daily dislocation ≥ +3 vol points |
    | Trade | sell the ATM straddle, delta-hedge hourly with the perpetual, hold to Deribit's expiry | sell the strip at its quotes, delta-hedge hourly, hold to Kalshi's settlement |
    | Positions | one per weekly contract, $1,000 of vega per vol point | one per daily contract, $1,000 of vega per vol point |
    | Taker | **{_a.loc['taker', 'trades']:.0f} trades, {_a.loc['taker', 'avg_vol']:+.1f} vol pts/trade, Sharpe {_a.loc['taker', 'sharpe']:.1f}**, total ${_a.loc['taker', 'total']:,.0f} | **{_b.loc['taker', 'trades']:.0f} trades, {_b.loc['taker', 'avg_vol']:+.1f} vol pts/trade, Sharpe {_b.loc['taker', 'sharpe']:.1f}**, total ${_b.loc['taker', 'total']:,.0f} |
    | Maker | {_a.loc['maker', 'avg_vol']:+.1f} vol pts/trade, Sharpe {_a.loc['maker', 'sharpe']:.1f}, total ${_a.loc['maker', 'total']:,.0f} | {_b.loc['maker', 'avg_vol']:+.1f} vol pts/trade, Sharpe {_b.loc['maker', 'sharpe']:.1f}, total ${_b.loc['maker', 'total']:,.0f} |
    | Out of sample (May–Sep) | only 3 signals: not testable | positive only as a maker (+2.3 vol pts/trade) |
    | **Verdict** | **works in Jan–Apr, then the signal stops firing** | **a market-making edge, not a taker trade; fading** |

    **What we would present: the dislocation as a *signal*, not a strategy (section 5).** Sorted into fifths by
    dislocation, Deribit's vol error against what BTC then realized falls in a straight line, from about +2 vol
    points (Deribit too high) when Kalshi is cheapest to about −5 (Deribit too low) when Kalshi is richest, for
    weeklies and dailies alike, and with the same sign before and after May. Kalshi tells you when Deribit's vol is
    wrong; the trades above are too few or too costly to collect it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How the backtest works

    - **Signal, no look-ahead.** Each hour, Kalshi's ATM vol (from its quoted ladder, as in 03) minus Deribit's ATM
      vol at Kalshi's exact expiry (from Deribit trades in the *previous* hour only), minus that gap's own median
      over the previous 30 days. A position opens the first hour a contract's dislocation crosses the threshold.
    - **Trade.** Sell the ATM straddle (Deribit) or the binary strip that pays like one (Kalshi: "no" contracts below
      the strike, "yes" above, out to 3 standard deviations). Delta-hedged every hour with the BTC perpetual, held
      to expiry: Deribit settles on its delivery price, Kalshi on its contracts' results.
    - **Size and P&L.** Every position is $1,000 of vega per vol point (about 10 weekly ATM straddles on Deribit,
      or about 25 one-BTC Kalshi strips). P&L is booked on the day the position settles, in dollars, and in vol
      points of vega per trade. Sharpe uses daily P&L over every calendar day (×√365).
    - **Execution.** *Taker*: cross every spread, full fees (Kalshi 7% × P × (1 − P) per contract; Deribit 0.03%
      of BTC per option), hedge at the perpetual's 0.05% taker fee. *Maker*: fill at mid; Kalshi charges a quarter
      of its taker fee (an assumption), Deribit's 0.03% applies to makers too, hedge free. *Half spread*: fills
      halfway, full fees, taker hedge. Maker is an upper bound: it assumes every resting order fills at mid.
    """)
    return


@app.cell(hide_code=True)
def _(END, FIG, H, MODE_COLORS, RULES, START, W, alt, mo, pd, rule_book):
    _parts = []
    for _st, _title in [("A", "A · sell Deribit vol (Kalshi cheap, weeklies)"),
                        ("B", "B · sell Kalshi vol (Kalshi rich, dailies)")]:
        for _m in ["maker", "half spread", "taker"]:
            _b = rule_book(_st, _m)
            _days = pd.date_range(START, END, freq="D", tz="UTC", inclusive="left")
            _cum = _b.groupby("close").pnl.sum().reindex(_days, fill_value=0.0).cumsum()
            _parts.append(pd.DataFrame({"day": _days, "cum": _cum.to_numpy(), "execution": _m, "strategy": _title}))
    _eq = pd.concat(_parts)
    fig_eq = (
        alt.Chart(_eq).mark_line(strokeWidth=1.8, interpolate="step-after").encode(
            x=alt.X("day:T", title=None, axis=alt.Axis(format="%b")),
            y=alt.Y("cum:Q", title="cumulative P&L ($, $1,000 vega per trade)", axis=alt.Axis(format="~s")),
            color=alt.Color("execution:N", title=None, scale=alt.Scale(domain=list(MODE_COLORS)[:3],
                                                                       range=list(MODE_COLORS.values())[:3])),
            tooltip=[alt.Tooltip("day:T", format="%Y-%m-%d"), "execution:N", alt.Tooltip("cum:Q", format=",.0f")])
        .properties(width=W, height=H)
        .facet(column=alt.Column("strategy:N", title=None), spacing=30)
        .resolve_scale(y="independent")
        .properties(title="Figure 1 · Equity curves by execution")
    )
    fig_eq.save(FIG / "12_fig1_equity_curves.png", scale_factor=2)
    mo.vstack([
        mo.md("""
        ## 1 · Performance

        Figure 1: cumulative P&L of each rule under three executions. **A** makes its money in a handful of
        February–April weeks and then goes quiet: large Kalshi-cheap dislocations stopped occurring after April.
        **B** is a straight line as a maker and a straight line *down* as a taker: the edge is roughly the size of
        Kalshi's spread and fee.
        """),
        fig_eq,
    ])
    return


@app.cell(hide_code=True)
def _(headline, mo):
    _t = headline.rename(columns={"trades": "trades", "hit": "hit rate", "avg_vol": "avg vol pts/trade",
                                  "t": "t-stat", "total": "total $", "sharpe": "Sharpe", "max_dd": "max drawdown $",
                                  "worst": "worst trade $", "best": "best trade $"})
    mo.vstack([
        mo.md("**All metrics, both rules, every execution** (15 Jan – 30 Sep 2026)"),
        mo.ui.table(_t.round(2), selection=None, show_download=True, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(END, FIG, H, START, W, ab, alt, mo, pd, trades):
    # Robustness: average P&L per trade over the parameter grid (A at taker cost, B at maker cost), held to expiry
    _rows = []
    for _st, _mode in [("A", "taker"), ("B", "maker")]:
        for _w in (15, 30, 60):
            for _thr in (3, 4, 5, 6, 8, 10):
                _m = ab.metrics(ab.book(trades[trades.window == _w], _st, _thr, "expiry", _mode), START, END)
                _rows.append({"strategy": f"{_st} ({_mode})", "normal window": f"{_w} days", "threshold": _thr,
                              "avg": _m["avg_vol"], "trades": _m["trades"], "sharpe": _m["sharpe"]})
    grid = pd.DataFrame(_rows)
    _base = alt.Chart(grid).encode(
        x=alt.X("threshold:O", title="|dislocation| threshold (vol points)"),
        y=alt.Y("normal window:N", title=None, sort=["15 days", "30 days", "60 days"]))
    fig_grid = (
        (_base.mark_rect().encode(color=alt.Color("avg:Q", title="vol pts/trade",
                                                  scale=alt.Scale(scheme="redblue", domainMid=0, domain=[-10, 10],
                                                                  clamp=True)),
                                  tooltip=["strategy:N", "normal window:N", "threshold:O",
                                           alt.Tooltip("avg:Q", format="+.1f"), "trades:Q",
                                           alt.Tooltip("sharpe:Q", format=".2f")])
         + _base.mark_text(fontSize=11).encode(
             text=alt.Text("label:N"),
             color=alt.condition("abs(datum.avg) > 7", alt.value("#ffffff"), alt.value("#0b0b0b")))
         .transform_calculate(label="format(datum.avg, '+.1f') + ' (' + datum.trades + ')'"))
        .properties(width=W, height=110)
        .facet(column=alt.Column("strategy:N", title=None), spacing=30)
        .properties(title="Figure 2 · Average P&L per trade across rules (number of trades in brackets)")
    )
    fig_grid.save(FIG / "12_fig2_robustness.png", scale_factor=2)
    # Other holding periods
    _hold = []
    for _st, _holds in [("A", ["4", "24", "expiry"]), ("B", ["4", "expiry"])]:
        for _h in _holds:
            for _mode in ["maker", "taker"]:
                _m = ab.metrics(ab.book(trades[trades.window == 30], _st, 6 if _st == "A" else 3, _h, _mode), START, END)
                _hold.append({"strategy": _st, "held": f"{_h} h" if _h != "expiry" else "to expiry", "execution": _mode,
                              "trades": _m["trades"], "avg vol pts/trade": _m["avg_vol"], "Sharpe": _m["sharpe"]})
    mo.vstack([
        mo.md("""
        ## 2 · Is it the parameters?

        Figure 2 moves the two rule parameters: the dislocation threshold and the window that defines "normal".
        **A** (at taker cost) is negative for every threshold below 6 and positive from 6 up, in all three windows,
        but with 1–13 trades per cell. **B** (at maker cost) is positive almost everywhere, best at low thresholds
        where it trades most. Holding for 4 or 24 hours instead of to expiry does not work for either (table).
        """),
        fig_grid,
        mo.ui.table(pd.DataFrame(_hold).round(2), selection=None, show_download=False, pagination=False),
    ])
    return (grid,)


@app.cell(hide_code=True)
def _(MODES, RULES, SPLIT, ab, mo, pd, rule_book, trades):
    _q = []
    for _st in RULES:
        for _m in ["maker", "half spread", "taker"]:
            _b = rule_book(_st, _m)
            _g = _b.groupby(_b.close.dt.quarter)
            for _qq, _x in _g:
                _q.append({"strategy": _st, "execution": _m, "quarter": f"Q{_qq}", "trades": len(_x),
                           "avg vol pts/trade": _x.pnl_vol.mean(), "total $": _x.pnl.sum()})
    _qt = pd.DataFrame(_q).pivot_table(index=["strategy", "execution"], columns="quarter",
                                       values=["trades", "avg vol pts/trade"]).round(1)
    _qt.columns = [f"{a} {b}" for a, b in _qt.columns]
    _oos = []
    for _st, _mode in [("A", "taker"), ("A", "maker"), ("B", "maker"), ("B", "half spread"), ("B", "taker")]:
        _o = ab.out_of_sample(trades, _st, "expiry", _mode, RULES[_st]["window"], split=SPLIT)
        _t = _o["table"].set_index("threshold")
        _fixed = RULES[_st]["threshold"]
        _oos.append({"strategy": _st, "execution": _mode, "threshold picked on Jan–Apr": _o["chosen"],
                     "its Jan–Apr Sharpe": _t.loc[_o["chosen"], "in_sharpe"],
                     "its May–Sep trades": _t.loc[_o["chosen"], "out_trades"],
                     "its May–Sep vol pts/trade": _t.loc[_o["chosen"], "out_avg_vol"],
                     f"rule threshold, May–Sep trades": _t.loc[_fixed, "out_trades"],
                     f"rule threshold, May–Sep vol pts/trade": _t.loc[_fixed, "out_avg_vol"],
                     f"rule threshold, May–Sep Sharpe": _t.loc[_fixed, "out_sharpe"]})
    mo.vstack([
        mo.md("""
        ## 3 · Over time and out of sample

        **By quarter.** A's trades are almost all in Q1–Q2. B as a maker earns in every quarter, less after Q1, in
        line with Kalshi's shrinking premium (11, section 0); as a taker it loses in every quarter.

        **Out of sample.** Pick each rule's threshold on January–April (best Sharpe, at least 4 trades) and run it on
        May–September. A picks 6 and then barely trades (3 positions). B's in-sample favourite overfits and loses;
        the simple 3-point rule keeps a positive edge only as a maker.
        """),
        mo.ui.table(_qt.reset_index(), selection=None, show_download=False, pagination=False),
        mo.ui.table(pd.DataFrame(_oos).round(2), selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(FIG, H, SLOT1, SLOT2, W, alt, mo, np, panel, pd, vs):
    # The signal: Deribit's (and Kalshi's) implied vol minus the vol BTC then realized, by dislocation quintile
    _p = panel.dropna(subset=["rv_to_expiry", "disl"]).copy()
    _p["Deribit"] = 100 * (_p.deribit_sig - _p.rv_to_expiry)
    _p["Kalshi"] = 100 * (_p.atm_iv - _p.rv_to_expiry)
    _rows, _stats = [], []
    for _kind, _lab in [("weekly", "weeklies"), ("daily", "dailies")]:
        _x = _p[_p.kind == _kind].copy()
        _x["q"] = pd.qcut(_x.disl, 5, labels=["1 Kalshi cheapest", "2", "3", "4", "5 Kalshi richest"])
        _g = _x.groupby("q", observed=True)[["Deribit", "Kalshi"]].mean()
        for _qq, _r in _g.iterrows():
            for _mkt in ["Deribit", "Kalshi"]:
                _rows.append({"contracts": _lab, "quintile": _qq, "market": _mkt, "error": _r[_mkt]})
        for _per, (_a, _b) in {"Jan–Apr": ("2026-01-01", "2026-05-01"), "May–Sep": ("2026-05-01", "2026-10-01"),
                               "all": ("2026-01-01", "2026-10-01")}.items():
            _y = _x[(_x.ts >= _a) & (_x.ts < _b)]
            _r = vs.ols_clustered(_y.Deribit, _y[["disl"]], _y.event_ticker, ["disl"])
            _e = _y.groupby("event_ticker")[["disl", "Deribit"]].mean()
            _stats.append({"contracts": _lab, "period": _per, "hours": len(_y), "contracts traded": _y.event_ticker.nunique(),
                           "Deribit error per point of dislocation": _r.coef["disl"], "t (clustered by contract)": _r.t["disl"],
                           "rank correlation, one point per contract": _e.disl.corr(_e.Deribit, method="spearman")})
    _sig = pd.DataFrame(_rows)
    fig_sig = (
        alt.Chart(_sig).mark_bar(width=16, cornerRadiusEnd=3).encode(
            x=alt.X("quintile:N", title="dislocation fifth (Kalshi vs Deribit, vs its normal)"),
            xOffset=alt.XOffset("market:N", sort=["Deribit", "Kalshi"]),
            y=alt.Y("error:Q", title="implied − realized vol (points)"),
            color=alt.Color("market:N", title=None, scale=alt.Scale(domain=["Deribit", "Kalshi"], range=[SLOT2, SLOT1])),
            tooltip=["contracts:N", "quintile:N", "market:N", alt.Tooltip("error:Q", format="+.1f")])
        .properties(width=W, height=H)
        .facet(column=alt.Column("contracts:N", title=None, sort=["weeklies", "dailies"]), spacing=30)
        .resolve_scale(y="independent")
        .properties(title="Figure 3 · The signal: when Kalshi is rich vs its normal, Deribit underprices BTC's moves")
    )
    fig_sig.save(FIG / "12_fig3_signal.png", scale_factor=2)
    mo.vstack([
        mo.md("""
        ## 4 · What survives: the dislocation as a signal

        Each hour is sorted into fifths by its dislocation. For each fifth, Figure 3 shows how far Deribit's (orange)
        and Kalshi's (blue) ATM vol were from the vol BTC then realized to Kalshi's expiry. Deribit's error falls in
        a straight line across the fifths: when Kalshi is unusually cheap, Deribit was too high; when Kalshi is
        unusually rich, Deribit was too low by about 5 points. The slope has the same sign in January–April and
        May–September, and holds with one observation per contract (table).

        On the dailies, the richest fifth is also where Kalshi itself overshoots (+12 points): the truth sits in
        between, which is why selling Kalshi there (B) pays before costs.

        **How a desk would use it.** Not as a stand-alone trade but as a filter on Deribit vol positions: lean
        short Deribit vol when Kalshi is cheap versus its normal, avoid selling (or buy) when Kalshi is rich. It is
        free to observe, updates every hour, and is not just Deribit's own momentum (11, section 2).
        """),
        fig_sig,
        mo.ui.table(pd.DataFrame(_stats).round(3), selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Bottom line and limits

    - **A** (sell Deribit vol when Kalshi is unusually cheap) earns about 6 vol points per trade after taker costs
      and 11 as a maker, Sharpe about 2–3, but on 12 positions concentrated in February–April; the signal fired
      only 3 times after April. Promising, not proven.
    - **B** (sell Kalshi's rich dailies) is a market-making edge: about +6 vol points per trade if the strip can be
      sold at mid, negative if it has to cross the spread. Its size has shrunk every quarter.
    - **The robust finding is the signal**: Kalshi's dislocation predicts Deribit's subsequent vol error, in both
      directions, in both halves of the year.

    **Limits.** One year, hourly Kalshi data; the maker case assumes fills at mid and a Kalshi maker fee of a
    quarter of the taker fee; positions are $1,000 of vega each, which on Kalshi's dailies means about 100,000
    contracts across the strip (small next to typical daily volume of about 2 million contracts per event, but
    concentrated near the money); no margin or funding cost of the short option positions is charged.
    """)
    return


if __name__ == "__main__":
    app.run()
