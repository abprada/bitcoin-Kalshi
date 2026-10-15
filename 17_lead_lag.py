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

    **The measure.** Every 10 minutes, the change in each market's at-the-money implied vol: dK(t) for Kalshi's
    weekly, dD(t) for Deribit's Friday expiry with the same date. Then the correlation between dK(t) and
    dD(t + k), for k from −60 to +60 minutes in 10-minute steps (`lib/leadlag.py`).

    - **k > 0**: Deribit's change comes *after* Kalshi's. A peak here means **Kalshi moves first**.
    - **k < 0**: Deribit's change comes first. A peak here means **Deribit moves first**.
    - **k = 0**: both change in the same 10 minutes.

    **The condition.** Only times when the gap, Kalshi − Deribit in vol points, is above **X + 1.8** (1.8 points is
    Kalshi's usual premium), for X from −6 to +6. X = 0 keeps the times Kalshi is richer than usual; X = +4 keeps
    those where it is 4 points richer than usual. The gap is measured **80 minutes before t**, before every change in
    the window, so selecting on it cannot select on the changes being correlated (the check at the end shows why
    this matters).

    **How each vol is measured.** Kalshi: its smile every 10 minutes from the 1-minute quotes, as in `03`. Deribit: its
    smile fitted to the trades in the 10 minutes *before* each time, so it never uses trades from the future, and
    stretched to Kalshi's 5pm ET expiry as in `05`/`16`.

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
    xc = leadlag.conditional_xcorr(panel)
    _live = panel.ts < panel.deribit_expiry
    _both = panel.K.notna() & panel.D.notna()
    mo.md(f"""
    ## The data

    **{len(panel):,}** usable Kalshi snapshots every 10 minutes in **{panel.event_ticker.nunique()}** weeklies
    ({panel.ts.min():%b %Y} – {panel.ts.max():%b %Y}). {_live.sum():,} fall before Deribit's Friday 08:00 expiry, and
    **{_both.sum() / _live.sum():.0%}** of those have a Deribit smile from the previous 10 minutes of trades (a median
    of {panel.n_trades.median():.0f} trades). The gap averages **{panel.gap.mean():+.1f}** points (median
    {panel.gap.median():+.1f}), against the 1.8 used as "usual".
    """)
    return ROOT, panel, xc


@app.cell
def _(H, INK2, W, alt, leadlag, mo, pd, xc):
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
        "Correlation of Kalshi's 10-minute vol change with Deribit's k minutes later",
        subtitle=f"Rows: only times Kalshi − Deribit is above X + 1.8 points; empty cells have fewer than {leadlag.MIN_N} observations."))

    _u = xc[xc.x.isna()].set_index("lag")
    _z = xc[(xc.lag == 0) & xc.x.notna()].set_index("x")["corr"]
    _n4 = int(xc[(xc.lag == 0) & (xc.x == 4)].n.iloc[0])
    mo.vstack([
        mo.md(f"""
        ## Who moves first

        **Unconditionally** (all times), the correlation peaks at **k = {(lambda v: f"{v:+d}" if v else "0")(int(_u["corr"].idxmax()))} minutes**
        ({_u["corr"].max():+.2f}): the two markets mostly move in the same 10 minutes, with a little more at +10
        (Kalshi first, {_u["corr"].loc[10]:+.2f}) than at −10 (Deribit first, {_u["corr"].loc[-10]:+.2f}).

        **When Kalshi is rich, the two decouple.** The same-10-minute correlation falls from {_u["corr"].loc[0]:+.2f}
        (all times) to {_z.loc[0]:+.2f} at X = 0, {_z.loc[2]:+.2f} at X = +2 and {_z.loc[4]:+.2f} at X = +4. No lag picks up
        what the contemporaneous correlation loses: neither market reliably follows the other within an hour, so a
        dislocation is not closed by one market catching up with the other's moves. At high thresholds the cells rest on
        a few thousand changes (95% band about ±{1.96 / _n4 ** 0.5:.2f} at X = +4), so single cells are not significant.
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
        subtitle="Shaded: 95% interval treating 10-minute changes as independent (too narrow if they are autocorrelated)."))

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
def _(H, MUTED, W, alt, leadlag, mo, panel, pd, xc):
    # Check: why the gap is measured 80 minutes before t. Measured at the start of Kalshi's change instead, it shares
    # its time stamp with the end of Deribit's change at k = -10, and selecting on a difference of two noisy readings
    # correlates their noise: a spurious negative correlation that deepens with the threshold.
    _late = leadlag.conditional_xcorr(panel, lags=[-10], gap_lead_min=leadlag.STEP_MIN)
    _cmp = pd.concat([
        _late[_late.x.notna()].assign(timing="gap at the start of Kalshi's change (biased)"),
        xc[(xc.lag == -10) & xc.x.notna()].assign(timing="gap 80 minutes earlier (used)"),
    ])
    _dom = ["gap at the start of Kalshi's change (biased)", "gap 80 minutes earlier (used)"]
    check_chart = (
        alt.Chart(_cmp).mark_line(point=alt.OverlayMarkDef(size=45)).encode(
            x=alt.X("x:Q", title="X (gap above X + 1.8 points)"), y=alt.Y("corr:Q", title="correlation at k = −10"),
            color=alt.Color("timing:N", title=None, sort=_dom, legend=alt.Legend(labelLimit=0), scale=alt.Scale(domain=_dom, range=["#c23b3b", "#2a78d6"])),
            tooltip=["timing:N", "x:Q", alt.Tooltip("corr:Q", format="+.3f"), alt.Tooltip("n:Q", format=",")])
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    ).properties(width=W, height=H - 40, title="Check: the k = −10 correlation under the two gap timings")
    _b, _g = _late.set_index("x")["corr"], xc[(xc.lag == -10)].set_index("x")["corr"]
    mo.vstack([
        mo.md(f"""
        ## Check: when the gap is measured matters

        Measured at the start of Kalshi's change, the gap shares its time stamp with the *end* of Deribit's change at
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
def _(ROOT, check_chart, heat_chart, lines_chart, mo, n_chart):
    # Save the figures when run as a script (python 17_lead_lag.py); opening the notebook writes nothing
    if not mo.running_in_notebook():
        _out = ROOT / "outputs" / "figures"
        _out.mkdir(parents=True, exist_ok=True)
        for _name, _chart in [("17_leadlag_heatmap", heat_chart), ("17_leadlag_lines", lines_chart),
                              ("17_leadlag_observations", n_chart), ("17_leadlag_timing_check", check_chart)]:
            _chart.save(_out / f"{_name}.png", scale_factor=2)
    return


if __name__ == "__main__":
    app.run()
