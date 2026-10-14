import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="09 Kalshi vs Deribit vol spread")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    from lib import vol_spread as vs

    return alt, mo, np, pd, vs


@app.cell(hide_code=True)
def _(alt):
    # Chart styling as in 05/06 (slot 1 blue, slot 2 orange, slot 3 aqua).
    INK, INK2, MUTED, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
    SLOT1, SLOT2, SLOT3 = "#2a78d6", "#eb6834", "#1baf7a"
    W, H = 660, 240


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
                "line": {"color": SLOT1, "strokeWidth": 2},
                "point": {"color": SLOT1, "size": 64, "filled": True},
            }
        }

    return H, MUTED, SLOT1, SLOT2, SLOT3, W


@app.cell(hide_code=True)
def _(mo, np, pd):
    ROOT = mo.notebook_dir()
    D = ROOT / "data" / "derived" / "strategies"
    FIG = ROOT / "outputs" / "figures"
    panel = pd.read_parquet(D / "vol_panel.parquet")
    panel = panel[panel.deribit_iv.notna()].copy()
    panel["K"], panel["D"], panel["RV"] = 100 * panel.atm_iv, 100 * panel.deribit_iv, 100 * panel.rv_to_expiry
    leadlag = pd.read_parquet(D / "leadlag_15m.parquet")
    trades = pd.read_parquet(D / "volspread_trades.parquet")
    trades["abs_signal"] = trades.signal.abs()
    trades["deribit_side"] = np.where(trades.side == "short Kalshi", "long Deribit", "short Deribit")
    HOLD_ORDER = ["1", "4", "24", "expiry"]
    return D, FIG, HOLD_ORDER, leadlag, panel, trades


@app.cell(hide_code=True)
def _(mo, panel, trades):
    _w = panel[panel.kind == "weekly"]
    _aft = _w[_w.ts.dt.hour.isin([20, 21, 22])]
    _d = panel[panel.kind == "daily"]
    head = dict(n=len(panel), days=panel.day.nunique(), weeks=_w.event_ticker.nunique(), dailies=_d.event_ticker.nunique(),
                aft=_aft.gap.median(), aft_far=_aft[_aft.hours_left >= 96].gap.median(),
                w_far=_w[_w.hours_left >= 96].gap.median(), w_mid=_w[(_w.hours_left > 24) & (_w.hours_left < 96)].gap.median(),
                d_gap=_d.gap.median(), corr=panel.groupby("day")[["K", "D"]].median().corr().iloc[0, 1],
                n_trades=len(trades))
    mo.md(f"""
    # 09 · Kalshi vs Deribit: a long/short vol trade, and who leads

    *January – September 2026. Built by `08_strategy_data.py` (`lib/vol_spread.py`); this notebook only reads its
    results.*

    **The idea.** 05 found that Kalshi and Deribit roughly agree at the money, with Kalshi a little richer, and the
    daily chart (06, Figure 1) suggests Kalshi jumps higher when vol jumps. If Kalshi overshoots, sell Kalshi's vol
    and buy Deribit's, and wait for them to come back together; or, if one market moves first, trade the other.

    **Answers.**

    1. **Kalshi is richer, and more so close to expiry.** With Deribit's vol interpolated to Kalshi's exact expiry
       every hour, Kalshi's weekly is {head['w_far']:+.1f} vol points above Deribit with 4–7 days left,
       {head['w_mid']:+.1f} with 1–4 days left, and Kalshi's 24-hour dailies {head['d_gap']:+.1f}. In 05's afternoon
       window this reproduces 05's trade-by-trade result ({head['aft']:+.1f} here vs +1.8 there).
    2. **Yes, Kalshi overreacts to vol spikes, and stays up.** Kalshi follows Deribit's vol up about twice as
       strongly as down, and after a jump in Deribit's vol the gap widens and stays wide for a day (section 2).
    3. **Neither market leads by a tradable margin.** Hour by hour the two move together; the timing difference is
       shorter than our data resolve (15 minutes). When they diverge, it is mostly Kalshi that moves back (section 3).
    4. **The gap is real but cannot be traded at a profit.** Kalshi's straddle is a strip of $1 binaries; buying and
       selling it back costs about 9–14 vol points in spreads and fees (median), against gaps of 1–4 points. Trading
       only Deribit on Kalshi's signal costs 2–4 points and earns well under 1 (section 5). No version, at any
       holding period or threshold, makes money after costs with any confidence. (Larger, rarer dislocations are
       studied in 11.)

    **Data.** {head['n']:,} hourly Kalshi snapshots ({head['weeks']} weeklies, {head['dailies']} dailies,
    {head['days']} days) paired with Deribit's fitted ATM vol at the same hour; {head['n_trades']:,} simulated trades.
    Daily correlation of the two ATM vols: {head['corr']:.2f}.
    """)
    return (head,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method

    **Kalshi, every hour.** The collaborator's file has the bid and ask of every strike at the top of each hour for
    Kalshi's 5pm ET events: the weeklies and, from January 2026, the 24-hour dailies ($250 strikes). Each snapshot is
    turned into a smile exactly as in 03 (02's quote rules, `iv.snapshot_iv`), which had not been done for this
    file before (`lib/kalshi_hourly.py`). Every snapshot with an ATM vol is kept: 05's strikes-per-SD rule would keep
    only high-vol moments near expiry (it passes 17–28% of snapshots in the last 3 hours), which inflates Kalshi's
    vol there; it is kept as a precision flag (`fine_grid`) instead.

    **Deribit, every hour, at Kalshi's expiry.** From all Deribit option trades (`00_fetch_deribit.py`), Deribit's
    mark smile is fitted per expiry and hour (`lib/deribit_marks.py`, as in 10). Deribit expires at 08:00 UTC and
    Kalshi at 5pm ET, so Deribit's total implied variance σ²τ is interpolated to Kalshi's expiry along a clock in
    which each hour carries its usual share of BTC variance (05's hour-of-week profile):

    - **bracketed** (37% of snapshots): between the Deribit expiry before and the one after Kalshi's, so Deribit's
      own prices price the 13 hours; this is the market-based fix listed as "Next 1" in `progress.md`;
    - otherwise **stretched** from the nearest expiry at its own rate: 05's adjustment.

    Gap = Kalshi ATM − Deribit ATM at Kalshi's expiry, in vol points. For *comparisons* Deribit uses trades within
    ±30 minutes of the hour; for *trading signals* only the hour before, so a signal never sees the future.

    **Trades** (`vol_spread.backtest`). At each hourly snapshot with a signal: if Kalshi is rich, sell Kalshi's
    straddle and buy Deribit's, the reverse if Kalshi is cheap. Kalshi's straddle is the strip of binaries that pays
    like one: "no" contracts below the strike, "yes" above, each sized to the dollars of strike it covers, out to 3
    standard deviations (`tests/test_strategies.py` checks it prices a Black-Scholes straddle within 1%). It trades
    at Kalshi's real bids and asks plus Kalshi's fee (7% × P × (1 − P) per contract, rounded up per strike). The
    Deribit straddle is sized to the same vega, valued on Deribit's smile, and pays the measured half-spread and the
    0.03% fee. Each leg is delta-hedged hourly with the perpetual (0.05% fee). Closed after 1, 4 or 24 hours at
    the opposite side of each book, or held to expiry (Kalshi settles on its contracts' results, Deribit on its
    delivery price). Because each leg is hedged on its own, the same run also gives the **Deribit leg alone on
    Kalshi's signal** and the **Kalshi leg alone on Deribit's**. P&L is in vol points of the Kalshi straddle's vega.
    """)
    return


@app.cell(hide_code=True)
def _(FIG, H, SLOT1, SLOT2, W, alt, mo, panel, pd, vs):
    _edges = [0, 6, 12, 18, 24, 48, 72, 96, 120, 170]
    _p = panel.assign(hb=pd.cut(panel.hours_left, _edges))
    _s = vs.trade_summary(_p, ["kind", "hb"], cols=("gap",))
    _s["hours"] = [(i.left + i.right) / 2 for i in _s.hb]
    _s["contract"] = _s.kind.map({"daily": "Kalshi daily (24h)", "weekly": "Kalshi weekly"})
    _s = _s.drop(columns="hb")
    _dom = ["Kalshi weekly", "Kalshi daily (24h)"]
    _x = alt.X("hours:Q", title="hours to Kalshi's expiry", scale=alt.Scale(reverse=True))
    _c = alt.Color("contract:N", title=None, scale=alt.Scale(domain=_dom, range=[SLOT1, SLOT2]))
    fig_gap = (
        alt.Chart(_s).mark_area(opacity=0.15).encode(x=_x, y="gap_lo:Q", y2="gap_hi:Q", color=_c)
        + alt.Chart(_s).mark_line(point=True).encode(
            x=_x, y=alt.Y("gap:Q", title="Kalshi − Deribit (vol points, mean)"), color=_c,
            tooltip=["contract:N", alt.Tooltip("hours:Q", format=".0f"), alt.Tooltip("gap:Q", format="+.1f"),
                     "trades:Q", "days:Q"])
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#898781", strokeDash=[4, 3]).encode(y="y:Q")
    ).properties(title="Figure 1 · Kalshi's ATM vol minus Deribit's at the same expiry, by hours left (95% interval)",
                 width=W, height=H)
    fig_gap.save(FIG / "09_fig1_gap_by_hours.png", scale_factor=2)
    _t = (panel.groupby(["kind", "h_bucket"]).agg(snapshots=("gap", "size"), days=("day", "nunique"),
                                                  median_gap=("gap", "median"), kalshi_higher=("gap", lambda x: (x > 0).mean()),
                                                  kalshi=("K", "median"), deribit=("D", "median"), realized=("RV", "median"))
          .round(2).reset_index())
    _m = panel.groupby(["kind", "method"]).gap.agg(["size", "median"]).round(2).reset_index()
    mo.vstack([
        mo.md("""
        ## 1 · How rich is Kalshi, hour by hour?

        The gap is small a week out and grows toward expiry, for weeklies and dailies alike. 05 saw the same growth
        and could not tell whether it came from its 13-hour stretch; here the bracketed snapshots, priced by
        Deribit's own expiries on both sides of Kalshi's, show it too (table below), so it is a feature of Kalshi's
        prices, not of the adjustment.
        """),
        fig_gap,
        mo.ui.table(_t, selection=None, show_download=False, pagination=False),
        mo.accordion({"Gap by interpolation method": mo.ui.table(_m, selection=None, show_download=False, pagination=False)}),
    ])
    return


@app.cell(hide_code=True)
def _(FIG, H, SLOT1, SLOT2, W, alt, mo, np, panel, pd, vs):
    # Asymmetry: does Kalshi follow Deribit's vol up more than down?
    _rows = []
    for _kind in ["weekly", "daily"]:
        _x = panel[panel.kind == _kind]
        for _h in [1, 4]:
            _m = vs.changes(_x, ["K", "D"], _h)
            _r = vs.ols_clustered(_m.d_K, np.column_stack([_m.d_D.clip(lower=0), _m.d_D.clip(upper=0)]), _m.day,
                                  ["up", "down"])
            _rows.append({"contract": _kind, "change over": f"{_h}h", "Kalshi per point Deribit rises": _r.coef["up"],
                          "(s.e.)": _r.se["up"], "Kalshi per point Deribit falls": _r.coef["down"], "(s.e.) ": _r.se["down"],
                          "observations": int(_r.n.iloc[0])})
        _r = vs.ols_clustered(_x.gap, np.column_stack([_x.D, 100 * _x.abs_ret_1h]), _x.day, ["level", "move"])
        _rows[-1]["gap per 1% BTC move in the last hour"] = _r.coef["move"]
        _rows[-2]["gap per 1% BTC move in the last hour"] = _r.coef["move"]
    asym = pd.DataFrame(_rows)
    # Event study: gap around the largest 4-hour rises in Deribit's vol (top 5%, one per contract per 12 hours)
    _ev_rows = []
    for _kind in ["weekly", "daily"]:
        _x = panel[panel.kind == _kind]
        _m = vs.changes(_x, ["D"], 4)
        _ev = _m[_m.d_D >= _m.d_D.quantile(0.95)][["event_ticker", "ts"]].sort_values(["event_ticker", "ts"])
        _ev = _ev[_ev.groupby("event_ticker").ts.diff().fillna(pd.Timedelta(days=9)) > pd.Timedelta(hours=12)]
        _base = _x.set_index(["event_ticker", "ts"]).gap
        for _off in range(-8, 25, 2):
            _v = _base.reindex(pd.MultiIndex.from_arrays([_ev.event_ticker, _ev.ts + pd.Timedelta(hours=_off)]))
            _ev_rows.append({"contract": _kind, "hours": _off, "gap": _v.mean(), "n": int(_v.notna().sum()),
                             "events": len(_ev)})
    _es = pd.DataFrame(_ev_rows)
    _es = _es[_es.n >= 20]
    _es["contract"] = _es.contract.map({"daily": "Kalshi daily (24h)", "weekly": "Kalshi weekly"})
    fig_event = (
        alt.Chart(_es).mark_line(point=True).encode(
            x=alt.X("hours:Q", title="hours after the end of a 4-hour jump in Deribit's vol"),
            y=alt.Y("gap:Q", title="Kalshi − Deribit (vol points, mean)"),
            color=alt.Color("contract:N", title=None, scale=alt.Scale(domain=["Kalshi weekly", "Kalshi daily (24h)"],
                                                                      range=[SLOT1, SLOT2])),
            tooltip=["contract:N", "hours:Q", alt.Tooltip("gap:Q", format="+.1f"), "n:Q"])
        + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#898781", strokeDash=[4, 3]).encode(x="x:Q")
    ).properties(title="Figure 2 · The gap around the largest jumps in Deribit's vol", width=W, height=H)
    fig_event.save(FIG / "09_fig2_gap_after_vol_jumps.png", scale_factor=2)
    _w = asym[(asym.contract == "weekly") & (asym["change over"] == "1h")].iloc[0]
    _d = asym[(asym.contract == "daily") & (asym["change over"] == "1h")].iloc[0]
    mo.vstack([
        mo.md(f"""
        ## 2 · Does Kalshi jump higher when vol jumps?

        **Yes.** Regressing Kalshi's hourly change in ATM vol on Deribit's, split into rises and falls: when Deribit
        rises one point Kalshi rises {_w['Kalshi per point Deribit rises']:.2f} (weekly) and
        {_d['Kalshi per point Deribit rises']:.2f} (daily); when Deribit falls one point Kalshi falls only
        {_w['Kalshi per point Deribit falls']:.2f} and {_d['Kalshi per point Deribit falls']:.2f}. Kalshi's vol
        ratchets up. The gap also widens after big BTC moves (last column) and when vol is high. Standard errors
        cluster by day.

        Figure 2 follows the gap around the top 5% of 4-hour rises in Deribit's vol. In the jump itself Kalshi lags
        a little (the weekly gap dips to about 1.7 points at 0), then keeps climbing while Deribit levels off: the
        weekly gap runs at 3.5–5 points for the next day, about a point above its level before the jump, and the
        daily gap roughly doubles (fewer events, mostly February and June). This is the pattern in 06's Figure 1.
        """),
        mo.ui.table(asym.round(3), selection=None, show_download=False, pagination=False),
        fig_event,
    ])
    return (asym,)


@app.cell(hide_code=True)
def _(FIG, H, SLOT1, SLOT2, W, alt, leadlag, mo, panel, pd, vs):
    _ll = leadlag.assign(contract=leadlag.kind.map({"daily": "Kalshi daily (24h)", "weekly": "Kalshi weekly"}))
    _long = pd.concat([
        _ll.assign(which="Kalshi's change into the hour", corr=_ll.corr_with_kalshi_hour_ending_t),
        _ll.assign(which="Kalshi's change in the next hour", corr=_ll.corr_with_kalshi_next_hour)])
    fig_ll = (
        alt.Chart(_long).mark_line(point=True).encode(
            x=alt.X("deribit_hour_ends:Q", title="Deribit's 1-hour change ends this many minutes after Kalshi's snapshot"),
            y=alt.Y("corr:Q", title="correlation"),
            color=alt.Color("contract:N", title=None, scale=alt.Scale(domain=["Kalshi weekly", "Kalshi daily (24h)"],
                                                                      range=[SLOT1, SLOT2])),
            strokeDash=alt.StrokeDash("which:N", title=None),
            tooltip=["contract:N", "which:N", "deribit_hour_ends:Q", alt.Tooltip("corr:Q", format=".2f"), "n:Q"])
        + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#898781", strokeDash=[4, 3]).encode(x="x:Q")
    ).properties(title="Figure 3 · Kalshi's hourly vol change against Deribit's, shifted in 15-minute steps",
                 width=W, height=H)
    fig_ll.save(FIG / "09_fig3_lead_lag.png", scale_factor=2)
    # Who closes a gap? Change over the next h hours regressed on today's gap.
    _rows = []
    for _kind, _hs in [("weekly", [1, 4, 24]), ("daily", [1, 4])]:
        _x = panel[panel.kind == _kind]
        for _h in _hs:
            _f = vs.forward_changes(_x, ["K", "D", "gap"], _h)
            _rk = vs.ols_clustered(_f[f"f{_h}_K"], _f[["gap"]], _f.day, ["gap"]).loc["gap"]
            _rd = vs.ols_clustered(_f[f"f{_h}_D"], _f[["gap"]], _f.day, ["gap"]).loc["gap"]
            _rows.append({"contract": _kind, "hours ahead": _h, "Kalshi moves (per point of gap)": _rk.coef,
                          "t (Kalshi)": _rk.t, "Deribit moves (per point of gap)": _rd.coef, "t (Deribit)": _rd.t,
                          "observations": int(_rk.n)})
    ec = pd.DataFrame(_rows)
    mo.vstack([
        mo.md("""
        ## 3 · Who moves first?

        Kalshi is observed once an hour, at the top of the hour. Deribit's ATM vol is fitted from trades in 15-minute
        windows, so Deribit's 1-hour change can be slid past Kalshi's in 15-minute steps (Figure 3). Dashed lines:
        Kalshi's change into the hour; solid: Kalshi's change in the following hour.

        - Kalshi's move matches Deribit's move over the **same hour** (peak correlation 0.34–0.40 with Deribit's hour
          ending 0–15 minutes after the snapshot) and almost nothing outside it.
        - Deribit's last hour barely predicts Kalshi's next hour (solid lines at 0: 0.06–0.10), and Kalshi's change
          barely predicts Deribit's next 15–60 minutes (correlation at most 0.07).

        So any lead is shorter than our resolution. An earlier version of this test with ±30-minute Deribit windows
        showed "Deribit leads by an hour", and one with past-only windows "Kalshi leads by an hour": both were the
        window, not the market. Kalshi's 1-minute quotes (05's data) would resolve it further.

        **When the two diverge, Kalshi does most of the moving back** (table: next-hours change regressed on today's
        gap). For weeklies, each point of gap is followed by Kalshi falling 0.15 points within the hour (part of
        that is noise in Kalshi's hourly quote coming back) while Deribit barely moves (+0.01); over a day the
        adjustment is shared (Kalshi −0.21, Deribit +0.23). The gap mean-reverts, but toward its positive average, not toward zero.
        """),
        fig_ll,
        mo.ui.table(ec.round(3), selection=None, show_download=False, pagination=False),
    ])
    return (ec,)


@app.cell(hide_code=True)
def _(mo, np, panel, pd, vs):
    _rows = []
    for (_kind, _hb), _x in panel.dropna(subset=["RV"]).groupby(["kind", "h_bucket"]):
        _r = vs.ols_clustered(_x.RV, np.column_stack([_x.K, _x.D]), _x.day, ["K", "D"])
        _rows.append({"contract": _kind, "hours left": _hb, "snapshots": len(_x),
                      "Kalshi − realized": (_x.K - _x.RV).mean(), "Deribit − realized": (_x.D - _x.RV).mean(),
                      "RMSE Kalshi": np.sqrt(((_x.K - _x.RV) ** 2).mean()),
                      "RMSE Deribit": np.sqrt(((_x.D - _x.RV) ** 2).mean()),
                      "weight on Kalshi": _r.coef["K"], "t": _r.t["K"], "weight on Deribit": _r.coef["D"], "t ": _r.t["D"]})
    fc = pd.DataFrame(_rows)
    _slopes = {}
    for _kind in ["weekly", "daily"]:
        _x = panel[panel.kind == _kind].dropna(subset=["RV"])
        _slopes[_kind] = vs.ols_clustered(_x.RV - _x.D, _x[["gap"]], _x.day, ["gap"]).loc["gap"]
    mo.vstack([
        mo.md(f"""
        ## 4 · Is Kalshi's richness wrong?

        If Kalshi's extra vol were pure overreaction, BTC should then move about as Deribit predicts. It does not.
        Against the vol BTC realized from each hour to Kalshi's expiry (1-minute returns), **Deribit is too low on
        average** (by up to 3 points: 2026 realized about 44%), **Kalshi is closer**, except in its last 12 hours
        (where it overshoots by 6–8). In
        a regression of realized vol on both, both carry information, and when Kalshi is richer than Deribit, BTC
        moves more than Deribit expected: the realized-minus-Deribit vol rises
        {_slopes['daily'].coef:.2f} points per point of gap for dailies (t = {_slopes['daily'].t:.1f}) and
        {_slopes['weekly'].coef:.2f} for weeklies (t = {_slopes['weekly'].t:.1f}).

        So "Kalshi richer on upticks" is partly Kalshi being right that the move is not over. That is good news for
        Kalshi as a forecast (handout question 3) and bad news for a trade that sells Kalshi's richness.
        """),
        mo.ui.table(fc.round(2), selection=None, show_download=False, pagination=False),
    ])
    return (fc,)


@app.cell(hide_code=True)
def _(FIG, HOLD_ORDER, H, MUTED, SLOT1, SLOT2, SLOT3, W, alt, mo, pd, trades, vs):
    _base = trades[(trades.signal_name == "gap_sig")]
    _legs = {"both": "both legs (sell rich, buy cheap)", "d": "Deribit only, on Kalshi's signal",
             "k": "Kalshi only, on Deribit's signal"}
    _parts = []
    for _leg, _name in _legs.items():
        _s = vs.trade_summary(_base, ["kind", "hold"], cols=(f"{_leg}_gross_vol", f"{_leg}_net_vol"))
        _s = _s.rename(columns=lambda c: c.replace(f"{_leg}_", ""))
        _parts.append(_s.assign(trade=_name))
    res = pd.concat(_parts, ignore_index=True)
    res["hold"] = pd.Categorical(res.hold, HOLD_ORDER)
    _plot = res.melt(id_vars=["kind", "hold", "trade"], value_vars=["gross_vol", "net_vol"], var_name="what",
                     value_name="vol")
    _plot["what"] = _plot.what.map({"gross_vol": "before costs", "net_vol": "after costs"})
    _plot["contract"] = _plot.kind.map({"daily": "dailies", "weekly": "weeklies"})
    _plot["hold"] = _plot.hold.astype(str)
    fig_trades = (
        alt.Chart(_plot).mark_point(filled=True, size=70).encode(
            x=alt.X("vol:Q", title="average P&L per trade (vol points of Kalshi vega)"),
            y=alt.Y("hold:N", title="held (hours)", sort=HOLD_ORDER),
            color=alt.Color("trade:N", title=None, scale=alt.Scale(domain=list(_legs.values()),
                                                                   range=[SLOT1, SLOT2, SLOT3])),
            shape=alt.Shape("what:N", title=None, scale=alt.Scale(domain=["before costs", "after costs"],
                                                                  range=["circle", "triangle-down"])),
            row=alt.Row("contract:N", title=None),
            tooltip=["contract:N", "trade:N", "hold:N", "what:N", alt.Tooltip("vol:Q", format="+.2f")])
        .properties(width=W - 80, height=110,
                    title="Figure 4 · Every Kalshi-vs-Deribit trade, before and after costs (signal: gap > 0)")
    )
    fig_trades.save(FIG / "09_fig4_trades.png", scale_factor=2)
    # thresholds and the demeaned signal, for the cheapest leg (Deribit only)
    _thr = []
    for _sig, _sname in [("gap_sig", "gap"), ("gap_dm", "gap minus its past average")]:
        for _t in [0, 3, 6, 10]:
            _x = trades[(trades.signal_name == _sig) & (trades.abs_signal > _t)]
            _s = vs.trade_summary(_x, ["kind", "hold"], cols=("d_net_vol", "k_net_vol", "both_net_vol"))
            _s["weeks or days"] = _x.groupby(["kind", "hold"]).event_ticker.nunique().to_numpy()
            _thr.append(_s.assign(signal=_sname, threshold=_t))
    thr = pd.concat(_thr, ignore_index=True)
    _pos = thr[(thr.d_net_vol_lo > 0) | (thr.k_net_vol_lo > 0) | (thr.both_net_vol_lo > 0)]
    _cost = trades[trades.signal_name == "gap_sig"].assign(
        k_cost_vol=lambda d: d.k_cost / d.vega_k, d_cost_vol=lambda d: d.d_cost / d.vega_k)
    _ct = _cost.groupby(["kind", "hold"])[["k_cost_vol", "d_cost_vol"]].median().round(1).reset_index()
    mo.vstack([
        mo.md(f"""
        ## 5 · The trades

        Figure 4: average P&L per trade, signal = the gap (Kalshi rich → sell Kalshi / buy Deribit; Kalshi cheap →
        the reverse), every hour it is available. Circles before costs, triangles after.

        - **Before costs, the spread trade makes money**, a little: 0.1–0.7 vol points over 1–4 hours, 1.2–4.2 held
          to expiry, more for bigger gaps. The gap does come back.
        - **After costs everything loses.** The Kalshi straddle is 15–30 strikes of $1 binaries, 250–500 contracts
          each; crossing each 1–2 cent spread and paying Kalshi's fee costs about 9–14 vol points per round trip
          (median, table below), several times the gap. On Kalshi's most vol-sensitive contracts, one vol point is
          worth about 0.6 cents.
        - **Deribit only, on Kalshi's signal**, is the cheap version (2–4 vol points round trip: Deribit's fee is
          0.03% of BTC per option). Its edge is small and not significant: buying Deribit when Kalshi is rich earns
          +0.1 points over an hour for weeklies (95% interval −0.04 to +0.24), +0.25 over 4 hours.
        - **Thresholds and the demeaned signal** (table below, sortable): bigger gaps raise the gross edge, but no
          cell has a 95% interval above zero after costs{" except: " + ", ".join(f"{r.kind} {r.hold}h ({r.signal} > {r.threshold})" for r in _pos.itertuples()) if len(_pos) else ""}.
          The positive point estimates at gaps above 10 points come from 1–17 distinct weeks, essentially the
          February and June spikes.
        """),
        fig_trades,
        mo.md("**Median round-trip cost, vol points of Kalshi vega**"),
        mo.ui.table(_ct.rename(columns={"k_cost_vol": "Kalshi leg", "d_cost_vol": "Deribit leg"}), selection=None,
                    show_download=False, pagination=False),
        mo.md("**By threshold on |signal| (net, vol points, 95% interval resampling days)**"),
        mo.ui.table(thr.round(2), selection=None, show_download=True, pagination=False),
    ])
    return res, thr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Conclusion and limits

    **Verdict: the vol spread is real and has a clear shape, but it is not a trade.** Kalshi's ATM vol sits 1–4
    points above Deribit's at the same expiry, more so close to expiry and after vol spikes, because Kalshi's vol
    ratchets up with Deribit's and comes down slowly. The two markets move together within the hour; neither leads
    by enough to trade. Kalshi's extra vol partly anticipates bigger BTC moves rather than being pure overreaction.
    And the costs are decisive: a Kalshi straddle is a strip of binaries whose spreads and fees are worth about
    15–19 vol points round trip. Using Kalshi only as a signal for Deribit cuts costs to 2–4 points but the edge is
    below 1.

    **What could change it:** Kalshi as a *maker* (posting the strip instead of crossing spreads; Kalshi's maker
    fees are lower or zero on many markets), larger gaps than 2026 produced, or a cheaper Deribit execution.

    **Limits.** Kalshi is observed hourly (the collaborator's snapshots), so lead-lag below 15 minutes and trades
    timed within the hour are out of reach; Deribit is valued from its mark smile, not quotes; the Kalshi strip
    stops at 3 standard deviations (beyond it the Deribit leg is unhedged); Kalshi's fee formula is the one in
    `lib/compare.py`, still to be checked against the current schedule.
    """)
    return


if __name__ == "__main__":
    app.run()
