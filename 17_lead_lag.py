import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="17 Lead-lag")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    from lib import leadlag, pairs

    return alt, leadlag, mo, np, pairs, pd


@app.cell
def _(mo):
    mo.md(r"""
    # 17 · Lead-lag: who moves first, and does it change when the markets are dislocated?

    Context for the dislocation trade. When Kalshi's vol is unusually far above Deribit's, does one market's move
    show up in the other's a few minutes later?

    **The measure.** The change in each market's at-the-money implied vol over a window of h minutes (60 by default;
    30 and 10 in the selector below), sampled every 10 minutes: dK(t) for Kalshi's weekly, dD(t) for Deribit's Friday
    expiry with the same date. Then the correlation between dK(t) and dD(t + k), for k from −60 to +60 minutes in
    10-minute steps (`lib/leadlag.py`).

    - **k > 0**: Deribit's change comes *after* Kalshi's. A peak here means **Kalshi moves first**.
    - **k < 0**: Deribit's change comes first. A peak here means **Deribit moves first**.
    - **k = 0**: both change over the same window.

    **The condition.** Only times when the gap, Kalshi − Deribit in vol points, is above **X + 1.8** (1.8 points is
    Kalshi's usual premium), for X from −6 to +6. X = 0 keeps the times Kalshi is richer than usual; X = +4 keeps
    those where it is 4 points richer than usual. The gap is measured **before every change in the window** (60 + h
    + 10 minutes before t), so selecting on it cannot select on the changes being correlated (the check at the end
    shows why this matters).

    **How each vol is measured.** Kalshi: its smile every 10 minutes from the 1-minute quotes, as in `03`. Deribit:
    every trade in the 10 minutes *before* each time (so it never uses trades from the future), each trade's mark vol
    turned into an ATM estimate with the previous hour's smile shape, median taken; stretched to Kalshi's 5pm ET expiry
    as in `05`/`16`.

    It writes nothing when opened; run as a script it saves its figures to `outputs/figures/` (prefixed `17_`).
    """)
    return


@app.cell
def _(alt):
    INK, INK2, MUTED, CONTEXT, GRID = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e1e0d9"
    W, H = 640, 260


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

    return H, INK, INK2, MUTED, W


@app.cell
def _(leadlag, mo, pairs):
    ROOT = mo.notebook_dir()
    _hp = pairs.cached_pairs(ROOT)["hour_profile"]
    panel = leadlag.cached_panel(ROOT, _hp)
    _live = panel.ts < panel.deribit_expiry
    _both = panel.K.notna() & panel.D.notna()
    mo.md(f"""
    ## The data

    **{len(panel):,}** usable Kalshi snapshots every 10 minutes in **{panel.event_ticker.nunique()}** weeklies
    ({panel.ts.min():%b %Y} – {panel.ts.max():%b %Y}). {_live.sum():,} fall before Deribit's Friday 08:00 expiry, and
    **{_both.sum() / _live.sum():.0%}** of those have a Deribit ATM estimate from the previous 10 minutes of trades (a
    median of {panel.n_trades.median():.0f} trades). The gap averages **{panel.gap.mean():+.1f}** points (median
    {panel.gap.median():+.1f}), against the 1.8 used as "usual".
    """)
    return ROOT, panel


@app.cell
def _(mo):
    horizon = mo.ui.dropdown({"60 minutes": 60, "30 minutes": 30, "10 minutes": 10}, value="60 minutes",
                             label="Change window h")
    return (horizon,)


@app.cell
def _(horizon, leadlag, mo, panel, pd):
    H_MIN = horizon.value
    xc = leadlag.conditional_xcorr(panel, change_min=H_MIN)
    # Why the window matters: 10-minute changes are mostly measurement noise, longer ones carry the common moves
    _rows = []
    for _h in (10, 30, 60, 120, 240):
        _r = leadlag.conditional_xcorr(panel, lags=[0], xs=[], change_min=_h, n_boot=50).iloc[0]
        _rows.append({"change window": f"{_h} min", "same-time correlation": round(_r["corr"], 2), "observations": f"{_r.n:,}"})
    mo.vstack([
        mo.md(f"""
        ## The change window

        Ten-minute changes in implied vol are mostly measurement noise: Kalshi's quotes move in 1¢ ticks and go stale, and
        Deribit's estimate rests on a handful of trades, so each market's own 10-minute changes partly reverse the next
        10 minutes. Noise in two markets does not correlate, while real moves add up over time, so the same-time
        correlation rises with the window (below). The lead-lag uses **{H_MIN}-minute** changes, still sliding k in
        10-minute steps; neighbouring observations then overlap, so the 95% intervals resample whole weeklies.
        """),
        horizon,
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_download=False, pagination=False),
    ])
    return H_MIN, xc


@app.cell
def _(H, H_MIN, INK2, W, alt, leadlag, mo, pd, xc):
    # Heatmap: correlation by lag (columns) and threshold X (rows); thin cells empty
    _c = xc[xc.x.notna() & (xc.n >= leadlag.MIN_N)].copy()
    _c["X"] = _c.x.astype(int).map(lambda v: f"{v:+d}" if v else "0")
    _order = [f"{v:+d}" if v else "0" for v in range(6, -7, -1)]
    _tip = [alt.Tooltip("X:N", title="X"), alt.Tooltip("threshold:Q", title="gap above", format="+.1f"),
            alt.Tooltip("lag:Q", title="lag (min)"), alt.Tooltip("corr:Q", format="+.2f"),
            alt.Tooltip("ci_low:Q", title="95% low", format="+.2f"), alt.Tooltip("ci_high:Q", title="95% high", format="+.2f"),
            alt.Tooltip("n:Q", title="observations", format=",")]
    _lim = float(_c["corr"].abs().max())
    _base = alt.Chart(_c).encode(
        x=alt.X("lag:O", title="k: Deribit's change k minutes after Kalshi's (negative = Deribit first)"),
        y=alt.Y("X:N", title="X (gap above X + 1.8 points)", sort=_order), tooltip=_tip)
    heat_chart = (
        _base.mark_rect(stroke="#fcfcfb", strokeWidth=1.5).encode(color=alt.Color(
            "corr:Q", title="correlation", scale=alt.Scale(domain=[-_lim, 0, _lim], range=["#c23b3b", "#f0efec", "#1f5fae"],
                                                          interpolate="rgb")))
        + _base.mark_text(fontSize=10).encode(text=alt.Text("corr:Q", format="+.2f"), color=alt.condition(
            f"abs(datum.corr) > {_lim * 0.6:.3f}", alt.value("white"), alt.value("#0b0b0b")))
    ).properties(width=W, height=H + 80, title=alt.TitleParams(
        f"Correlation of Kalshi's {H_MIN}-minute vol change with Deribit's, k minutes later",
        subtitle=f"Rows: only times Kalshi − Deribit is above X + 1.8 points; empty cells have fewer than {leadlag.MIN_N} observations."))

    _u = xc[xc.x.isna()].set_index("lag")
    _z = xc[(xc.lag == 0) & xc.x.notna()].set_index("x")["corr"]
    _c6 = xc[(xc.lag == 0) & (xc.x == 6)].iloc[0]
    _u0 = xc[(xc.lag == 0) & xc.x.isna()].iloc[0]
    _x6 = xc[xc.x == 6].set_index("lag")["corr"]
    _l6, _r6 = _x6[_x6.index < 0].mean(), _x6[_x6.index > 0].mean()
    mo.vstack([
        mo.md(f"""
        ## Who moves first

        **Unconditionally** (all times), the correlation peaks at **k = {(lambda v: f"{v:+d}" if v else "0")(int(_u["corr"].idxmax()))} minutes**
        ({_u["corr"].max():+.2f}, against {_u["corr"].loc[0]:+.2f} at k = 0), and the profile is close to symmetric: {_u["corr"].loc[10]:+.2f} at +10 (Kalshi first)
        against {_u["corr"].loc[-10]:+.2f} at −10 (Deribit first), {_u["corr"].loc[30]:+.2f} against
        {_u["corr"].loc[-30]:+.2f} at ±30. Neither market leads: they mostly move together.

        **When Kalshi is rich, the two decouple.** The same-time correlation is {_u["corr"].loc[0]:+.2f} at all times,
        {_z.loc[0]:+.2f} at X = 0, {_z.loc[2]:+.2f} at X = +2, **{_z.loc[4]:+.2f}** at X = +4 and **{_z.loc[6]:+.2f}** at
        X = +6 (95% interval {_c6.ci_low:+.2f} to {_c6.ci_high:+.2f}, against {_u0.ci_low:+.2f} to {_u0.ci_high:+.2f} at all
        times). The profile flattens rather than shifting to one side: in a large dislocation the two markets' vols move
        largely independently, so the gap is not closed by one market quickly copying the other.
        At X = +6 the Deribit-first side averages {_l6:+.2f} and the Kalshi-first side {_r6:+.2f}: a hint that Kalshi then
        follows Deribit, on {int(_c6.n):,} observations, so not conclusive.
        """),
        heat_chart,
    ])
    return (heat_chart,)


@app.cell
def _(H, INK2, MUTED, W, alt, leadlag, mo, pd, xc):
    # Lines: correlation by lag for a few thresholds, with Fisher-z 95% bands
    _pick = {None: "all times", 0: "X = 0 (gap > 1.8)", 2: "X = +2 (gap > 3.8)", 4: "X = +4 (gap > 5.8)"}
    _l = xc[xc.x.isin([k for k in _pick if k is not None]) | xc.x.isna()].copy()
    _l["which"] = [_pick[None] if pd.isna(v) else _pick[int(v)] for v in _l.x]
    _l = _l[_l.n >= leadlag.MIN_N]
    _dom = list(_pick.values())
    _col = alt.Color("which:N", title=None, sort=_dom, scale=alt.Scale(domain=_dom, range=["#898781", "#86b6ef", "#2a78d6", "#0d366b"]))
    _x = alt.X("lag:Q", title="k (minutes): Deribit's change after Kalshi's", axis=alt.Axis(values=list(range(-60, 61, 10))))
    lines_chart = (
        alt.Chart(_l).mark_area(opacity=0.12).encode(x=_x, y="ci_low:Q", y2="ci_high:Q", color=_col)
        + alt.Chart(_l).mark_line(point=alt.OverlayMarkDef(size=45)).encode(
            x=_x, y=alt.Y("corr:Q", title="correlation"), color=_col,
            tooltip=["which:N", "lag:Q", alt.Tooltip("corr:Q", format="+.2f"), alt.Tooltip("n:Q", format=",")])
        + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(x="x:Q")
    ).properties(width=W, height=H, title=alt.TitleParams(
        "Lead-lag profile for selected thresholds",
        subtitle="Shaded: 95% interval, resampling whole weeklies."))

    # Observations kept by each threshold
    _nn = xc[(xc.lag == 0) & xc.x.notna()].assign(X=lambda d: d.x.astype(int))
    n_chart = alt.Chart(_nn).mark_bar(color="#2a78d6", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("X:O", title="X"), y=alt.Y("n:Q", title="10-minute changes kept"),
        tooltip=["X:O", alt.Tooltip("n:Q", format=","), alt.Tooltip("mean_gap:Q", title="mean gap", format="+.1f")]
    ).properties(width=W, height=H - 60, title="How many observations each threshold keeps")

    _peaks = (xc[xc.x.notna() & (xc.n >= leadlag.MIN_N)].sort_values("corr").groupby("x").tail(1)
              .sort_values("x")[["x", "threshold", "lag", "corr", "n"]])
    _peaks = _peaks.assign(**{"peak at k (min)": _peaks.lag, "peak corr": _peaks["corr"].round(2),
                              "X": _peaks.x.astype(int), "gap above": _peaks.threshold.round(1)})
    mo.vstack([
        lines_chart,
        n_chart,
        mo.md("**Where each threshold's correlation peaks** (positive k: Kalshi moves first):"),
        mo.ui.table(_peaks[["X", "gap above", "peak at k (min)", "peak corr", "n"]], selection=None,
                    show_download=False, pagination=False),
        mo.md(f"""
        <span style='color:{INK2}'>**Reading it.** Correlations of 10-minute vol changes are small even when the markets
        track each other over hours, because most 10-minute moves are quote and fit noise. The question is the *shape*:
        a peak to the right of 0 means Deribit follows Kalshi, to the left that Kalshi follows Deribit. High thresholds
        keep few observations, so their curves are noisier; check `n` before reading a peak.</span>
        """),
    ])
    return lines_chart, n_chart


@app.cell
def _(H, INK2, MUTED, W, alt, leadlag, mo, np, panel, pd):
    # Event study: after the gap is seen above X + 1.8, the gap and each leg from 30 minutes later (when a trader
    # could act) over the next 4 hours, relative to the normal drift at the same time to expiry
    DELAY = 30
    ev = leadlag.after_dislocation(panel, delay_min=DELAY)
    _zero = pd.DataFrame([{"x": x, "series": s, "minutes": 0, "change": 0.0, "lo": 0.0, "hi": 0.0}
                          for x in ev.x.unique() for s in ev.series.unique()])
    _e = pd.concat([_zero, ev], ignore_index=True).assign(hours=lambda d: d.minutes / 60)
    _lbl = {0: "1.8+ points (X = 0)", 2: "3.8+ points (X = +2)", 4: "5.8+ points (X = +4)", 6: "7.8+ points (X = +6)"}
    _e["dislocation"] = _e.x.map(_lbl)
    _dom = list(_lbl.values())
    _blues = alt.Scale(domain=_dom, range=["#9ec5f4", "#5598e7", "#256abf", "#0d366b"])   # ordered: light = small gap
    _x = alt.X("hours:Q", title="hours after acting (30 minutes after the dislocation is seen)",
               scale=alt.Scale(domain=[0, 4], nice=False), axis=alt.Axis(values=[0, 1, 2, 3, 4]))
    _g = _e[_e.series == "gap"]
    _end = _g[_g.minutes == _g.minutes.max()]
    gap_path_chart = (
        alt.Chart(_g).mark_area(opacity=0.12).encode(x=_x, y="lo:Q", y2="hi:Q", color=alt.Color("dislocation:N", scale=_blues, legend=None))
        + alt.Chart(_g).mark_line(strokeWidth=3).encode(
            x=_x, y=alt.Y("change:Q", title="change in Kalshi − Deribit beyond normal drift (vol points)"),
            color=alt.Color("dislocation:N", scale=_blues, legend=None),
            tooltip=["dislocation:N", alt.Tooltip("hours:Q", format=".2f"), alt.Tooltip("change:Q", format="+.2f"),
                     alt.Tooltip("lo:Q", title="95% low", format="+.2f"), alt.Tooltip("hi:Q", title="95% high", format="+.2f"),
                     alt.Tooltip("n:Q", format=",")])
        + alt.Chart(_end).mark_text(align="left", dx=6, fontSize=12, fontWeight="bold").encode(
            x=_x, y="change:Q", text="dislocation:N", color=alt.Color("dislocation:N", scale=_blues, legend=None))
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    ).properties(width=W - 130, height=H + 40, padding={"right": 150}, title=alt.TitleParams(
        "After a dislocation, the gap closes, and the bigger the gap the more it closes",
        subtitle="Lines: how far Kalshi was above Deribit when seen (Kalshi's usual premium is 1.8). Shaded: 95% interval, resampling weeks."))

    _l = _e[(_e.series != "gap") & (_e.x == 4)]
    _legs = alt.Scale(domain=["Kalshi", "Deribit"], range=["#2a78d6", "#eb6834"])
    _lend = _l[_l.minutes == _l.minutes.max()].assign(t=lambda d: d.series + " " + d.change.map("{:+.1f}".format))
    legs_chart = (
        alt.Chart(_l).mark_area(opacity=0.15).encode(x=_x, y="lo:Q", y2="hi:Q", color=alt.Color("series:N", scale=_legs, legend=None))
        + alt.Chart(_l).mark_line(strokeWidth=3).encode(
            x=_x, y=alt.Y("change:Q", title="change in ATM vol beyond normal drift (vol points)"),
            color=alt.Color("series:N", scale=_legs, legend=None),
            tooltip=["series:N", alt.Tooltip("hours:Q", format=".2f"), alt.Tooltip("change:Q", format="+.2f"),
                     alt.Tooltip("lo:Q", title="95% low", format="+.2f"), alt.Tooltip("hi:Q", title="95% high", format="+.2f")])
        + alt.Chart(_lend).mark_text(align="left", dx=6, fontSize=13, fontWeight="bold").encode(
            x=_x, y="change:Q", text="t:N", color=alt.Color("series:N", scale=_legs, legend=None))
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    ).properties(width=W - 130, height=H + 40, padding={"right": 110}, title=alt.TitleParams(
        "Both legs close it: Kalshi falls and Deribit rises",
        subtitle="Kalshi 5.8+ points above Deribit when seen (X = +4). Shaded: 95% interval, resampling weeks."))

    _at4 = lambda s, x, m: ev[(ev.series == s) & (ev.x == x) & (ev.minutes == m)].iloc[0]
    _g6, _g4, _g2 = _at4("gap", 6, 240), _at4("gap", 4, 240), _at4("gap", 2, 240)
    _k4, _d4 = _at4("Kalshi", 4, 240), _at4("Deribit", 4, 240)
    _g4_2h = _at4("gap", 4, 120)
    _k4_2h, _d4_2h = _at4("Kalshi", 4, 120), _at4("Deribit", 4, 120)
    mo.vstack([
        mo.md(f"""
        ## After a dislocation: the gap closes

        This is the evidence for the dislocation trade. Correlations of changes (above) show whether the markets move
        *together*; the trade needs the gap to *shrink*. So: when Kalshi is unusually far above Deribit, what happens
        next? Changes are measured from {DELAY} minutes after the dislocation is seen (time to act, and for short-lived
        measurement noise to fade; the reading that flags the dislocation is never part of the change), and against
        the normal change at the same time to expiry, so the gap's usual widening into expiry does not count.

        - **The gap closes.** Over the next 4 hours it narrows by **{-_g2.change:.1f}** points after a 3.8+ point gap,
          **{-_g4.change:.1f}** after 5.8+ ({-_g4_2h.change:.1f} within 2 hours) and **{-_g6.change:.1f}** after 7.8+;
          every 95% interval is below zero.
        - **Both legs move.** After a 5.8+ point gap, within 2 hours Kalshi's vol falls **{-_k4_2h.change:.2f}** points
          (95% interval {_k4_2h.lo:+.2f} to {_k4_2h.hi:+.2f}) and Deribit's rises **{_d4_2h.change:.2f}**
          ({_d4_2h.lo:+.2f} to {_d4_2h.hi:+.2f}): the convergence is shared, not Deribit catching up alone. By 4 hours
          ({_k4.change:+.1f} and {_d4.change:+.1f}) each leg's own interval is wide enough to touch zero, while the gap's
          stays well below it.
        - **It is slow.** The closing builds over hours, while within the hour the two markets move largely
          independently (the decoupling above), so the trade earns from mean reversion, not from one market copying
          the other.
        - **It is small next to costs.** A 1–2 point convergence is worth well under 1¢ on a $1 Kalshi contract near
          the money, against a quoted spread of 1–2¢ and a fee of up to 1.75¢; the trade needs maker fills (`12`).
        """),
        gap_path_chart,
        legs_chart,
    ])
    return ev, gap_path_chart, legs_chart


@app.cell
def _(H, MUTED, W, alt, leadlag, mo, panel, pd):
    # Check, on 10-minute changes where it is clearest: why the gap is measured before every change in the window.
    # Measured at the start of Kalshi's change instead, it shares its time stamp with the end of Deribit's change at
    # k = -10, and selecting on a difference of two noisy readings correlates their noise: a spurious negative
    # correlation that deepens with the threshold.
    _late = leadlag.conditional_xcorr(panel, lags=[-10], change_min=10, gap_lead_min=10, n_boot=50)
    _clean = leadlag.conditional_xcorr(panel, lags=[-10], change_min=10, n_boot=50)      # gap 80 minutes before t
    _cmp = pd.concat([
        _late[_late.x.notna()].assign(timing="gap at the start of Kalshi's change (biased)"),
        _clean[_clean.x.notna()].assign(timing="gap 80 minutes earlier (used)"),
    ])
    _dom = ["gap at the start of Kalshi's change (biased)", "gap 80 minutes earlier (used)"]
    check_chart = (
        alt.Chart(_cmp).mark_line(point=alt.OverlayMarkDef(size=45)).encode(
            x=alt.X("x:Q", title="X (gap above X + 1.8 points)"), y=alt.Y("corr:Q", title="correlation at k = −10"),
            color=alt.Color("timing:N", title=None, sort=_dom, legend=alt.Legend(labelLimit=0), scale=alt.Scale(domain=_dom, range=["#c23b3b", "#2a78d6"])),
            tooltip=["timing:N", "x:Q", alt.Tooltip("corr:Q", format="+.3f"), alt.Tooltip("n:Q", format=",")])
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    ).properties(width=W, height=H - 40, title="Check (10-minute changes): the k = −10 correlation under the two gap timings")
    _b, _g = _late.set_index("x")["corr"], _clean.set_index("x")["corr"]
    mo.vstack([
        mo.md(f"""
        ## Check: when the gap is measured matters

        With 10-minute changes, measured at the start of Kalshi's change, the gap shares its time stamp with the *end* of Deribit's change at
        k = −10. Selecting times where a difference of two noisy readings is large makes those readings' noise
        correlated, which shows up as a negative correlation at k = −10 that deepens with the threshold
        ({_b.loc[0]:+.3f} at X = 0, {_b.loc[4]:+.3f} at X = +4, {_b.loc[6]:+.3f} at X = +6). It would read as "when
        Kalshi is rich, a Deribit rise is followed by a Kalshi fall", but it is produced by the selection: measured 80
        minutes earlier, before every change, the same cells are {_g.loc[0]:+.3f}, {_g.loc[4]:+.3f} and {_g.loc[6]:+.3f}.
        """),
        check_chart,
    ])
    return (check_chart,)


@app.cell
def _(ROOT, check_chart, gap_path_chart, heat_chart, legs_chart, lines_chart, mo, n_chart):
    # Save the figures when run as a script (python 17_lead_lag.py); opening the notebook writes nothing
    if not mo.running_in_notebook():
        _out = ROOT / "outputs" / "figures"
        _out.mkdir(parents=True, exist_ok=True)
        for _name, _chart in [("17_leadlag_heatmap", heat_chart), ("17_leadlag_lines", lines_chart),
                              ("17_leadlag_observations", n_chart), ("17_leadlag_timing_check", check_chart),
                              ("17_after_dislocation_gap", gap_path_chart), ("17_after_dislocation_legs", legs_chart)]:
            _chart.save(_out / f"{_name}.png", scale_factor=2)
    return


if __name__ == "__main__":
    app.run()
