import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="18 Trades: calendar spreads and the dislocation trade")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    from lib import backtest, deribit, leadlag, pairs, term
    from lib import strategy_ab as ab

    return ab, alt, backtest, deribit, leadlag, mo, np, pairs, pd, term


@app.cell
def _(mo):
    mo.md(r"""
    # 18 · The trades: calendar spreads and the dislocation trade

    Every trade idea in one place: what it buys and sells, why, and what it made. Nothing is re-estimated here: the
    trades come from the backtests of `10` (Deribit calendar), `12` (dislocation strategies), `14` (Kalshi and
    Kalshi−Deribit calendars) and `17` (what happens after a dislocation), with the same rules and costs.

    **Before and after costs.** *Before costs*: every trade at mid prices, no fees. *Taker*: crossing every bid-ask
    spread and paying full fees (Kalshi 7% × P × (1 − P) per contract; Deribit 0.03% of BTC per option; the
    perpetual hedge 0.05%). *Maker* (where shown): resting orders filled at mid, Kalshi at a quarter of its taker fee,
    an optimistic upper bound.

    **Units differ by family, as in their source notebooks.** Kalshi and Kalshi−Deribit calendars: **bps of spot per
    1 BTC of straddle** per trade (an ATM straddle is worth about 290 bps with a day left, 500 with a week). Deribit
    calendar: **dollars per day** per long straddle. Dislocation strategies: **dollars**, each position sized to
    **$1,000 of vega per vol point**.
    """)
    return


@app.cell
def _(alt):
    INK, INK2, MUTED, CONTEXT, GRID = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e1e0d9"
    W, H = 640, 240
    # Execution in a fixed order and colour, as in 12: before costs (gray), maker (blue), taker (orange)
    MODES = alt.Scale(domain=["before costs", "maker", "taker"], range=["#898781", "#2a78d6", "#eb6834"])


    @alt.theme.register("research_light", enable=True)
    def _research_light() -> alt.theme.ThemeConfig:
        return {
            "config": {
                "background": "#fcfcfb",
                "font": "system-ui, -apple-system, 'Segoe UI', sans-serif",
                "view": {"stroke": None},
                "axis": {"gridColor": GRID, "domainColor": CONTEXT, "tickColor": CONTEXT, "labelColor": MUTED,
                         "titleColor": INK2, "titleFontWeight": "normal", "labelFontSize": 12, "titleFontSize": 13},
                "legend": {"labelColor": INK2, "titleColor": INK2, "orient": "top", "labelFontSize": 12},
                "axisX": {"labelAngle": 0},
                "title": {"color": INK, "anchor": "start", "fontWeight": "normal", "fontSize": 15,
                          "subtitleColor": INK2, "subtitleFontSize": 12},
                "line": {"strokeWidth": 2.5},
            }
        }

    return H, INK2, MODES, MUTED, W


@app.cell
def _(H, MODES, MUTED, W, alt, pd):
    def cum_chart(df, date_col, cols, title, subtitle, unit):
        """Cumulative P&L of one trade under each execution mode; `cols` maps mode label -> P&L column."""
        parts = []
        for label, col in cols.items():
            s = df[[date_col, col]].dropna().sort_values(date_col)
            parts.append(pd.DataFrame({"date": pd.to_datetime(s[date_col]), "cum": s[col].cumsum().to_numpy(),
                                       "pnl": s[col].to_numpy(), "mode": label}))
        d = pd.concat(parts, ignore_index=True)
        # Carry each line flat to the last date, so the final trade is a step, not a tick at the edge
        last = d.sort_values("date").groupby("mode").tail(1).assign(date=d.date.max() + (d.date.max() - d.date.min()) * 0.03, pnl=float("nan"))
        d = pd.concat([d, last], ignore_index=True)
        end = d.sort_values("date").groupby("mode").tail(1).assign(t=lambda x: x["mode"] + " " + x.cum.map(lambda v: f"{v:+,.0f}"))
        x = alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y"))
        return (
            alt.Chart(d).mark_line(interpolate="step-after").encode(
                x=x, y=alt.Y("cum:Q", title=f"cumulative P&L ({unit})"),
                color=alt.Color("mode:N", title=None, scale=MODES, legend=None),
                tooltip=["mode:N", alt.Tooltip("date:T", format="%Y-%m-%d"), alt.Tooltip("pnl:Q", title="this trade", format="+,.1f"),
                         alt.Tooltip("cum:Q", title="cumulative", format="+,.0f")])
            + alt.Chart(end).mark_text(align="left", dx=6, fontSize=12, fontWeight="bold").encode(
                x=x, y="cum:Q", text="t:N", color=alt.Color("mode:N", scale=MODES, legend=None))
            + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
        ).properties(width=W - 120, height=H, padding={"right": 130}, title=alt.TitleParams(title, subtitle=subtitle))

    return (cum_chart,)


@app.cell
def _(backtest, deribit, mo, pd, term):
    # Notebook 14's inputs and trades (same functions and arguments)
    ROOT = mo.notebook_dir()
    spot = backtest.load_spot(ROOT)
    cum = term.cumulative_variance(spot)
    hour_profile = deribit.hourly_variance(spot)
    snaps = term.kalshi_weekly_atm(pd.read_parquet(ROOT / "data" / "derived" / "iv_snapshots.parquet"), usable_only=False)
    daily_atm = term.kalshi_daily_atm(ROOT, usable_only=False)
    deribit_trades = deribit.load_trades(deribit.trades_path(ROOT))
    deribit_obs = term.with_realized(term.deribit_atm(deribit_trades[deribit_trades.window != "other"]), cum)

    kalshi_cal = backtest.calendar_trades(ROOT, cum)
    _legs = {}
    for _prod, _k in (("weekly", snaps), ("daily", daily_atm)):
        _p = backtest.dislocation_panel(_k, deribit_obs, hour_profile, _prod)
        _legs[_prod] = backtest.dislocation_trades(_p, ROOT, spot, cum, _prod, require_disloc=False)
    gap_cal = backtest.gap_calendar(_legs["weekly"], _legs["daily"])
    gap_A = backtest.gap_widening_trades(backtest.dislocation_panel(snaps, deribit_obs, hour_profile, "weekly"),
                                         ROOT, spot, cum)
    gap_A_traded = backtest.gap_widening_traded(gap_A, deribit_trades, ROOT, spot, cum, window_min=90)
    return ROOT, gap_A, gap_A_traded, gap_cal, kalshi_cal


@app.cell
def _(ROOT, ab, pd):
    # Notebook 10's Deribit calendar (daily P&L from 08) and notebook 12's dislocation strategies
    _D = ROOT / "data" / "derived" / "strategies"
    cal_daily = pd.read_parquet(_D / "calendar_daily.parquet").assign(gross=lambda d: d.options + d.hedge + d.funding)
    _tr = pd.read_parquet(_D / "ab_trades.parquet")
    AB_RULES = {"A": dict(threshold=6, hold="expiry"), "B": dict(threshold=3, hold="expiry")}
    AB_START, AB_END = "2026-01-15", "2026-10-01"
    ab_books = {(st, mode): ab.book(_tr[_tr.window == 30], st, r["threshold"], r["hold"], mode)
                for st, r in AB_RULES.items() for mode in ("gross", "maker", "taker")}
    return AB_END, AB_RULES, AB_START, ab_books, cal_daily


@app.cell
def _(AB_END, AB_START, ab, ab_books, backtest, cal_daily, gap_A, gap_A_traded, gap_cal, kalshi_cal, mo, np, pd):
    # Overview: one row per trade, average per trade (or per day) before and after costs, with 95% intervals
    def _s(df, col):
        r = backtest.summarize(df, col, n_boot=1000).iloc[0]
        return f"{r['mean']:+.1f} [{r.ci_low:+.1f}, {r.ci_high:+.1f}]", int(r.trades)

    def _ab(st, mode):
        b = ab_books[(st, mode)]
        m = ab.metrics(b, AB_START, AB_END)
        return f"{m['avg_vol']:+.1f} vol pts ({m['trades']} trades)"

    _cd = cal_daily[(cal_daily.period == "2026") & (cal_daily.variant == "base")]
    _rows = [
        ("Calendar", "1 · Kalshi calendar (deck Idea 1)", "short Kalshi daily strip, long Kalshi weekly strip; 23 h before the daily's close, out at its close",
         *_s(kalshi_cal, "pnl_mid_bps"), _s(kalshi_cal, "pnl_bps")[0], "bps / trade", "loses: the weekly leg crosses the spread twice in a day"),
        ("Calendar", "2 · Deribit calendar straddle", "long 1 ATM straddle ~7 days out, short ~2.6 one-day straddles (vega-neutral), delta-hedged hourly",
         f"{_cd.gross.mean():+.0f} $/day", len(_cd), f"{_cd.pnl.mean():+.0f} $/day", "$ / day", "loses even before costs: Deribit's dailies carry no last-day premium"),
        ("Calendar, on the spread", "3 · Gap calendar", "long the 1-week Kalshi−Deribit spread, short the 1-day spread; each leg to expiry",
         *_s(gap_cal, "total_mid"), _s(gap_cal, "total_taker")[0], "bps / trade", "no edge before costs"),
        ("Calendar, on the spread", "4 · Gap widening", "buy the Kalshi−Deribit spread a week out, sell it back Thursday with a day left",
         *_s(gap_A, "total_mid"), _s(gap_A, "total_taker")[0], "bps / trade", "earns the gap's widening before costs; Kalshi's spreads and fees take it all"),
        ("Dislocation", "5 · A: sell Deribit when Kalshi is unusually cheap", "weekly dislocation ≤ −6 points: sell Deribit's weekly ATM straddle, hedged, to expiry",
         _ab("A", "gross"), len(ab_books[("A", "gross")]), _ab("A", "taker"), "vol pts / trade", "worked Jan–Apr 2026, then the signal stopped firing"),
        ("Dislocation", "6 · B: sell Kalshi when Kalshi is unusually rich", "daily dislocation ≥ +3 points: sell Kalshi's daily strip, hedged, to settlement",
         _ab("B", "gross"), len(ab_books[("B", "gross")]), _ab("B", "taker"), "vol pts / trade", "maker-only edge; loses as a taker"),
    ]
    overview = pd.DataFrame(_rows, columns=["family", "trade", "what it does", "before costs (95% interval)", "trades / days",
                                            "after costs (taker)", "unit", "verdict"])
    mo.vstack([
        mo.md("## Overview"),
        mo.ui.table(overview, selection=None, show_download=False, pagination=False),
        mo.md("""
        The **calendar** trades all bet on the same thing: very short-dated vol is rich, especially Kalshi's in its last
        day. Versions 1–2 trade it within one market; 3–4 trade the **Kalshi−Deribit spread** across maturities. The
        **dislocation** trades bet that whichever market is unusually rich relative to the other is overpricing vol.
        """),
    ])
    return (overview,)


@app.cell
def _(cum_chart, kalshi_cal, mo):
    kalshi_cal_chart = cum_chart(
        kalshi_cal, "entry", {"before costs": "pnl_mid_bps", "taker": "pnl_bps"},
        "1 · Kalshi calendar: short the daily, long the weekly",
        f"{len(kalshi_cal)} trades, one per Kalshi daily; bps of spot per 1 BTC of straddle.", "bps")
    mo.vstack([
        mo.md(r"""
        ## Calendar spreads

        ### 1 · Kalshi calendar: short the daily, long the weekly (deck Idea 1)

        - **Entry**, 23 hours before a Kalshi daily closes: **sell the daily's strip** (the binaries that pay like an
          at-the-money straddle) and **buy the weekly's strip** over the same dollar range.
        - **Exit**: the daily settles at 5pm ET; the weekly is sold back at that moment.
        - **The bet**: Kalshi's last-day vol is rich, and the weekly hedges the vega so only that premium is left.
        - **Why it fails**: the weekly is bought and sold within a day, crossing Kalshi's spread twice on every strike.
        """),
        kalshi_cal_chart,
    ])
    return (kalshi_cal_chart,)


@app.cell
def _(cal_daily, cum_chart, mo, pd):
    _b = cal_daily[(cal_daily.period == "2026") & (cal_daily.variant == "base")]
    _o = cal_daily[(cal_daily.period == "2026") & (cal_daily.variant == "overnight_expiry")]
    _d = pd.DataFrame({"t": _b.t.to_numpy(), "gross": _b.gross.to_numpy(), "net": _b.pnl.to_numpy()})
    deribit_cal_chart = cum_chart(
        _d, "t", {"before costs": "gross", "taker": "net"},
        "2 · Deribit calendar straddle: long the week, short the day",
        f"{len(_b)} days, Jan 15 – Sep 30 2026; dollars per long straddle (1 BTC call + 1 BTC put).", "$")
    mo.vstack([
        mo.md(f"""
        ### 2 · Deribit calendar straddle: long the week, short the day (notebook `10`)

        - **Every day at 08:00 UTC**: **buy 1 ATM straddle** on the Deribit expiry about 7 days out and **sell about 2.6
          ATM straddles** on the next daily expiry, so the two vegas match. **Delta-hedge every hour** with the BTC
          perpetual.
        - **Exit**: buy the short leg back 4 hours before it expires; sell the long leg after a day; repeat.
        - **The bet**: the same last-day premium, on Deribit.
        - **Why it fails**: Deribit's daily options carry **no** last-day premium; their vol *falls* into the 08:00
          expiry, which comes after the quiet night. It makes {_b.gross.mean():+.0f} $ a day before costs and
          {_b.pnl.mean():+.0f} $ after. The one positive variant sells only the overnight hours ({_o.gross.mean():+.0f} $
          a day before costs, {_o.pnl.mean():+.0f} $ after).
        """),
        deribit_cal_chart,
    ])
    return (deribit_cal_chart,)


@app.cell
def _(cum_chart, gap_cal, mo):
    gap_cal_chart = cum_chart(
        gap_cal, "day", {"before costs": "total_mid", "taker": "total_taker"},
        "3 · Gap calendar: long the weekly Kalshi−Deribit spread, short the daily spread",
        f"{len(gap_cal)} days; bps of spot per 1 BTC of straddle on each leg.", "bps")
    mo.vstack([
        mo.md(f"""
        ### 3 · Gap calendar: trade the Kalshi−Deribit spread across maturities (notebook `14`, Strategy 4)

        - **Long the 1-week spread**: buy Kalshi's weekly strip, sell Deribit's Friday straddle.
        - **Short the 1-day spread**: sell Kalshi's daily strip (about a day left), buy Deribit's next-morning straddle.
        - **Hold**: each leg to its own expiry. Entered each US afternoon.
        - **The bet**: the gap is wider at short maturities (at matched times the daily gap averages
          **{gap_cal.gap_d.mean():+.1f}** points, the weekly's **{gap_cal.gap_w.mean():+.1f}**), so sell the expensive
          short-dated spread and buy the cheap long-dated one.
        - **Why it fails**: held to expiry, each leg just earns its own implied-minus-realized vol; nothing forces the
          daily spread to collapse, and four legs of costs follow.
        """),
        gap_cal_chart,
    ])
    return (gap_cal_chart,)


@app.cell
def _(alt, cum_chart, gap_A, gap_A_traded, mo):
    gap_A_chart = cum_chart(
        gap_A, "exit", {"before costs": "total_mid", "taker": "total_taker"},
        "4 · Gap widening: buy the Kalshi−Deribit spread a week out, sell it back with a day left",
        f"{len(gap_A)} weeks; bps of spot per 1 BTC.", "bps")
    gap_A_traded_chart = cum_chart(
        gap_A_traded, "exit", {"before costs": "total_mid", "maker": "total_maker", "taker": "total_taker"},
        "4b · The same, using only Deribit strikes that actually traded",
        f"{len(gap_A_traded)} weeks with a tradable condor on both days; bps of spot per 1 BTC.", "bps")
    mo.vstack([
        mo.md(f"""
        ### 4 · Gap widening: trade the spread as it widens into expiry (notebook `14`, Strategy 5A)

        - **Entry**, the first US afternoon with 7 days left (else 6, 5): **buy the Kalshi weekly strip** (strikes within
          ±1.5 standard deviations) and **sell the Deribit butterfly with the same payoff**: long the Kalshi−Deribit
          spread while it is narrow.
        - **Exit**, Thursday afternoon (Kalshi about 25 hours left): sell both back.
        - **The bet**: the ATM gap is about 0 a week out and about +3 points a day out (notebook `16`), so buying the
          spread early and selling it late collects the widening.
        - **Result**: positive before costs, the only calendar that is, but each of the ~25–35 Kalshi strikes crosses
          the spread twice, and that costs far more than 2–3 points of widening are worth.
        """),
        gap_A_chart,
        gap_A_traded_chart,
    ])
    return gap_A_chart, gap_A_traded_chart


@app.cell
def _(AB_END, AB_RULES, AB_START, ab, ab_books, cum_chart, mo, pd):
    def _frame(st):
        parts = {m: ab_books[(st, m)][["close", "pnl"]].rename(columns={"pnl": m}) for m in ("gross", "maker", "taker")}
        out = parts["gross"].merge(parts["maker"], on="close", how="outer").merge(parts["taker"], on="close", how="outer")
        return out.sort_values("close")

    _mA = {m: ab.metrics(ab_books[("A", m)], AB_START, AB_END) for m in ("gross", "maker", "taker")}
    _mB = {m: ab.metrics(ab_books[("B", m)], AB_START, AB_END) for m in ("gross", "maker", "taker")}
    disl_A_chart = cum_chart(_frame("A"), "close", {"before costs": "gross", "maker": "maker", "taker": "taker"},
                             "5 · A: sell Deribit when Kalshi is unusually cheap",
                             f"{_mA['gross']['trades']} positions, {AB_START} – {AB_END}; dollars at $1,000 of vega per vol point.", "$")
    disl_B_chart = cum_chart(_frame("B"), "close", {"before costs": "gross", "maker": "maker", "taker": "taker"},
                             "6 · B: sell Kalshi when Kalshi is unusually rich",
                             f"{_mB['gross']['trades']} positions; dollars at $1,000 of vega per vol point.", "$")
    mo.vstack([
        mo.md(f"""
        ## The dislocation trade (notebooks `11`–`12`)

        **The signal**, every hour: Kalshi's ATM vol minus Deribit's ATM vol at Kalshi's expiry (Deribit from the
        previous hour's trades only, so no look-ahead), minus that gap's own median over the previous 30 days for the same
        contract type and time to expiry (its *normal* premium, about 1.8 points). What is left is the **dislocation**:
        how much richer (+) or cheaper (−) Kalshi is than usual.

        | | **A** | **B** |
        |---|---|---|
        | Trigger | weekly dislocation ≤ −{AB_RULES['A']['threshold']} points (Kalshi unusually cheap, so Deribit unusually rich) | daily dislocation ≥ +{AB_RULES['B']['threshold']} points (Kalshi unusually rich) |
        | Trade | sell Deribit's weekly ATM straddle | sell Kalshi's daily strip (pays like a short straddle) |
        | Then | delta-hedge hourly with the perpetual, hold to Deribit's expiry | delta-hedge hourly, hold to Kalshi's settlement |
        | Size | $1,000 of vega per vol point | $1,000 of vega per vol point |
        | Before costs | {_mA['gross']['avg_vol']:+.1f} vol pts / trade, total ${_mA['gross']['total']:,.0f} | {_mB['gross']['avg_vol']:+.1f} vol pts / trade, total ${_mB['gross']['total']:,.0f} |
        | Maker | {_mA['maker']['avg_vol']:+.1f}, total ${_mA['maker']['total']:,.0f}, Sharpe {_mA['maker']['sharpe']:.1f} | {_mB['maker']['avg_vol']:+.1f}, total ${_mB['maker']['total']:,.0f}, Sharpe {_mB['maker']['sharpe']:.1f} |
        | Taker | {_mA['taker']['avg_vol']:+.1f}, total ${_mA['taker']['total']:,.0f}, Sharpe {_mA['taker']['sharpe']:.1f} | {_mB['taker']['avg_vol']:+.1f}, total ${_mB['taker']['total']:,.0f}, Sharpe {_mB['taker']['sharpe']:.1f} |

        **The bet**: whichever market is unusually rich relative to the other is overpricing vol; sell it and collect
        the excess as BTC realizes less. Each strategy sells **one** market and holds it to expiry: it earns that leg's
        implied-minus-realized vol, not the convergence of the gap.
        """),
        disl_A_chart,
        disl_B_chart,
    ])
    return disl_A_chart, disl_B_chart


@app.cell
def _(H, MUTED, ROOT, W, alt, leadlag, mo, pairs, pd):
    # Why the dislocation idea has a basis: after a dislocation the gap closes (notebook 17)
    _panel = leadlag.cached_panel(ROOT, pairs.cached_pairs(ROOT)["hour_profile"])
    _ev = leadlag.after_dislocation(_panel, xs=(2, 4, 6), horizons=range(10, 241, 10), n_boot=200)
    _zero = pd.DataFrame([{"x": x, "series": "gap", "minutes": 0, "change": 0.0, "lo": 0.0, "hi": 0.0} for x in (2, 4, 6)])
    _g = pd.concat([_zero, _ev[_ev.series == "gap"]]).assign(hours=lambda d: d.minutes / 60)
    _lbl = {2: "3.8+ points", 4: "5.8+ points", 6: "7.8+ points"}
    _g["Kalshi above Deribit"] = _g.x.map(_lbl)
    _sc = alt.Scale(domain=list(_lbl.values()), range=["#86b6ef", "#2a78d6", "#0d366b"])
    _x = alt.X("hours:Q", title="hours after acting (30 minutes after the dislocation is seen)", scale=alt.Scale(domain=[0, 4], nice=False), axis=alt.Axis(values=[0, 1, 2, 3, 4]))
    gap_closes_chart = (
        alt.Chart(_g).mark_area(opacity=0.12).encode(x=_x, y="lo:Q", y2="hi:Q", color=alt.Color("Kalshi above Deribit:N", scale=_sc, legend=None))
        + alt.Chart(_g).mark_line().encode(x=_x, y=alt.Y("change:Q", title="change in the gap beyond normal drift (vol pts)"),
                                           color=alt.Color("Kalshi above Deribit:N", scale=_sc, legend=None))
        + alt.Chart(_g[_g.minutes == _g.minutes.max()]).mark_text(align="left", dx=6, fontSize=12, fontWeight="bold").encode(
            x=_x, y="change:Q", text="Kalshi above Deribit:N", color=alt.Color("Kalshi above Deribit:N", scale=_sc, legend=None))
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    ).properties(width=W - 120, height=H, padding={"right": 110}, title=alt.TitleParams(
        "After a dislocation the Kalshi−Deribit gap closes, more for bigger gaps",
        subtitle="Event study from notebook 17; shaded: 95% interval resampling weeks."))
    _e4 = _ev[(_ev.x == 4) & (_ev.minutes == 120)].set_index("series")
    mo.vstack([
        mo.md(f"""
        ### Why the dislocation idea has a basis

        When Kalshi is unusually far above Deribit, the gap does close (`17`): after a 5.8+ point gap it narrows by
        **{-_e4.loc['gap', 'change']:.1f}** points within 2 hours, with Kalshi's vol falling
        ({_e4.loc['Kalshi', 'change']:+.2f}) and Deribit's rising ({_e4.loc['Deribit', 'change']:+.2f}). But 1–2 points is
        well under 1¢ on a $1 Kalshi contract near the money, against a 1–2¢ quoted spread and up to 1.75¢ of fees,
        which is why B only pays with maker fills and trading the gap outright loses after costs.
        """),
        gap_closes_chart,
    ])
    return (gap_closes_chart,)


@app.cell
def _(
    ROOT,
    deribit_cal_chart,
    disl_A_chart,
    disl_B_chart,
    gap_A_chart,
    gap_A_traded_chart,
    gap_cal_chart,
    gap_closes_chart,
    kalshi_cal_chart,
    mo,
):
    # Save the figures when run as a script (python 18_trades.py); opening the notebook writes nothing
    if not mo.running_in_notebook():
        _out = ROOT / "outputs" / "figures"
        _out.mkdir(parents=True, exist_ok=True)
        for _name, _chart in [("18_1_kalshi_calendar", kalshi_cal_chart), ("18_2_deribit_calendar", deribit_cal_chart),
                              ("18_3_gap_calendar", gap_cal_chart), ("18_4_gap_widening", gap_A_chart),
                              ("18_4b_gap_widening_traded", gap_A_traded_chart), ("18_5_dislocation_A", disl_A_chart),
                              ("18_6_dislocation_B", disl_B_chart), ("18_7_gap_closes", gap_closes_chart)]:
            _chart.save(_out / f"{_name}.png", scale_factor=2)
    return


if __name__ == "__main__":
    app.run()
