import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="15 Deribit implied-vol surface")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import plotly.graph_objects as go

    from lib import deribit, iv
    from lib import deribit_marks as dm

    return alt, deribit, dm, go, iv, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # 15 · Deribit implied-vol surface

    The Deribit counterpart of `03`'s Kalshi surface: implied vol across strikes and time to expiry, built the same
    way and drawn on the same axes, so the two can be read side by side.

    **Data.** Every Deribit BTC option **trade**, all day and every expiry, 15 Nov 2024 – 30 Sep 2026 (the span of
    our Kalshi data; `data/raw/deribit_trades/`, from `00_fetch_deribit_all.py`). Without that fetch the notebook
    falls back to the collaborator's file, which has two windows a day and expiries up to about 8.5 days.
    There are no quotes, but every trade carries Deribit's **mark vol** for its instrument at that instant:
    Deribit's own fair-value vol, the one it values positions with. Trades over a short window therefore trace
    Deribit's mark smile.

    **Method** (`lib/deribit_marks.py`, as in `08`–`14`):

    1. **Forward.** Deribit's vol is quoted against its forward for that expiry. Black-Scholes with the mark price
       and mark vol pins the forward each trade was marked against; its median per window is the forward.
    2. **Snapshots.** Every hour, the trades within ±30 minutes. Per expiry and strike: the median mark vol.
    3. **Smile.** Per expiry and snapshot, a quadratic in log-moneyness per root year, fitted to strikes within
       2.5 ATM standard deviations (at least three strikes, forward inside their range). Its constant is the
       **ATM vol**.
    4. **Surface.** As in `03` (`iv.pooled_surface`): each strike's mark vol divided by its own smile's ATM vol,
       median per cell (the typical smile *shape*), times the median ATM vol of that time-to-expiry bin. Pooling
       shapes rather than raw vols keeps the volatile days, when far strikes trade, from inflating the wings.

    **Coordinates.** Horizontal: distance from the forward in ATM standard deviations,
    `(K / F − 1) / (ATM IV × √τ)`; 0 is the money, ±1 one ATM-implied standard deviation over the option's
    remaining life. `03` measures from spot. Within a week Deribit's forward is within about 0.01% of its index,
    so there the two are the same axis; further out the forward carries the futures basis (a median of roughly 0.1%
    at 1–4 weeks, 0.4% at 1–3 months, 2% at 6–12 months in March 2026), and measuring from the forward is the
    standard convention. Vertical: time to expiry. Vols are annualised.

    It writes nothing when opened; run as a script (`python 15_deribit_iv_surface.py`) it saves its figures to
    `outputs/figures/` (prefixed `15_`) and the 3D surfaces to `outputs/15_deribit_iv_surface_3d.html`.
    """)
    return


@app.cell
def _(alt):
    # Chart styling: reference palette (slot 2 orange = Deribit, as in 04/05), recessive gray chrome.
    INK, INK2, MUTED, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
    SLOT1, SLOT2 = "#2a78d6", "#eb6834"
    W, H = 640, 220


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
                "line": {"color": SLOT2, "strokeWidth": 2},
                "point": {"color": SLOT2, "size": 40, "filled": True},
            }
        }

    return CONTEXT, H, INK2, SLOT2, W


@app.cell
def _(deribit, dm, mo, np, pd):
    ROOT = mo.notebook_dir()
    HOURS_PER_YEAR = 365 * 24

    _path = deribit.trades_path(ROOT)    # every trade if fetched, else the collaborator's two-window file
    trades = dm.add_forward(dm.read_full_trades(ROOT) if deribit.is_full(_path) else deribit.load_trades(_path))
    smiles = dm.smile_grid(trades)                         # one row per (expiry, hourly snapshot)
    per_k, snap_info = dm.window_strikes(trades)           # the per-strike mark vols those smiles were fitted to

    # Strike level: each strike's median mark vol in its snapshot, with that smile's ATM vol and forward
    strikes = per_k.merge(smiles[["expiry", "t", "atm_iv", "tau_years"]], on=["expiry", "t"])
    strikes = strikes.join(snap_info[["basis", "index"]], on=["expiry", "t"])
    strikes["fwd"] = strikes["index"] * (1 + strikes.basis)
    strikes["hours_to_expiry"] = strikes.tau_years * HOURS_PER_YEAR
    strikes["sd_from_fwd"] = (strikes.strike / strikes.fwd - 1) / (strikes.atm_iv * np.sqrt(strikes.tau_years))
    strikes["snap"] = strikes.expiry.astype(str) + " @ " + strikes.t.astype(str)
    smiles["hours_to_expiry"] = smiles.tau_years * HOURS_PER_YEAR

    _n_snap = smiles.t.nunique()
    mo.md(f"""
    ## 1 · What the data give

    **{len(trades):,}** trades with a usable mark vol and forward ({trades.ts.min():%d %b %Y} – {trades.ts.max():%d %b %Y},
    **{trades.expiry.nunique()}** expiries). On the hourly grid they give **{_n_snap:,}** snapshots and
    **{len(smiles):,}** fitted smiles, about {len(smiles) / _n_snap:.1f} live expiries per snapshot, from
    **{len(strikes):,}** strike-level mark vols (median {smiles.n_strikes.median():.0f} strikes per smile).

    Median ATM vol **{smiles.atm_iv.median():.1%}** (interquartile {smiles.atm_iv.quantile(.25):.1%} –
    {smiles.atm_iv.quantile(.75):.1%}). The fitted skew term is negative in {(smiles.b < 0).mean():.0%} of smiles
    (puts richer than calls) and the curvature positive in {(smiles.c > 0).mean():.0%} (wings priced above
    the money). Forward basis to the index: median {smiles.basis.median():.3%}.
    """)
    return ROOT, smiles, strikes


@app.cell
def _(mo, smiles):
    _dates = sorted(smiles.t.dt.strftime("%Y-%m-%d").unique())
    date_select = mo.ui.dropdown(_dates, value=_dates[len(_dates) // 2], label="Date (UTC)")
    return (date_select,)


@app.cell
def _(date_select, mo, smiles):
    _times = sorted(smiles[smiles.t.dt.strftime("%Y-%m-%d") == date_select.value].t.dt.strftime("%H:%M").unique())
    time_select = mo.ui.dropdown(_times, value=_times[-1], label="Snapshot (UTC)")
    return (time_select,)


@app.cell
def _(CONTEXT, H, INK2, W, alt, date_select, dm, mo, np, pd, smiles, strikes, time_select):
    _t = pd.Timestamp(f"{date_select.value} {time_select.value}", tz="UTC")
    _sm = smiles[smiles.t == _t].sort_values("expiry")
    _pts = strikes[(strikes.t == _t) & strikes.sd_from_fwd.abs().le(3)].copy()

    def _label(e, h):
        return f"{e:%a %d %b} ({h / 24:.1f}d)"

    _labels = {e: _label(e, h) for e, h in zip(_sm.expiry, _sm.hours_to_expiry)}
    _pts["expiry_label"] = _pts.expiry.map(_labels)
    # Fitted smiles over the strikes each was fitted to, on the same standardized axis
    _lines = []
    for _, _r in _sm.iterrows():
        _sd = np.sqrt(_r.tau_years)
        _m = np.linspace(_r.m_lo, _r.m_hi, 60)
        _F = _r["index"] * (1 + _r.basis)
        _K = _F * np.exp(_m * _sd)
        _lines.append(pd.DataFrame({"expiry_label": _labels[_r.expiry],
                                    "sd_from_fwd": (_K / _F - 1) / (_r.atm_iv * _sd),
                                    "iv": dm.smile_iv(_r, _K, _F)}))
    _lines = pd.concat(_lines, ignore_index=True)
    _lines = _lines[_lines.sd_from_fwd.abs() <= 3]
    # Expiries are ordered, so one hue from light (nearest) to dark (furthest)
    _order = list(_labels.values())
    _ramp = ["#fbd3bf", "#f6a47c", "#eb6834", "#b8431a", "#7a2a0e", "#4a1907"][-len(_order):] if len(_order) <= 6 else None
    _color = alt.Color("expiry_label:N", sort=_order, title="Deribit expiry (time left)",
                       scale=alt.Scale(domain=_order, range=_ramp) if _ramp else alt.Scale(scheme="oranges"))
    _x = alt.X("sd_from_fwd:Q", title="distance from forward (ATM standard deviations)", scale=alt.Scale(domain=[-3, 3]))
    _y = alt.Y("iv:Q", title="implied vol", axis=alt.Axis(format=".0%"), scale=alt.Scale(zero=False))
    _zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=CONTEXT, strokeDash=[3, 3]).encode(x="x:Q")
    snapshot_chart = (
        _zero
        + alt.Chart(_lines).mark_line().encode(_x, _y, _color)
        + alt.Chart(_pts).mark_point(filled=True, size=45, opacity=0.9).encode(
            _x, _y, _color, tooltip=[alt.Tooltip("expiry_label:N", title="expiry"), alt.Tooltip("strike:Q", format=",.0f"),
                                     alt.Tooltip("iv:Q", title="mark vol", format=".1%"), alt.Tooltip("n:Q", title="trades")])
    ).properties(title=f"Deribit smiles at {_t:%d %b %Y %H:%M} UTC, every live expiry", width=W, height=H + 80)

    _ts = _sm.assign(expiry_label=_sm.expiry.map(_labels), days=_sm.hours_to_expiry / 24)
    term_now_chart = (
        alt.Chart(_ts).mark_line(point=True).encode(
            alt.X("days:Q", title="days to expiry"), alt.Y("atm_iv:Q", title="ATM vol", axis=alt.Axis(format=".0%"),
                                                           scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("expiry_label:N", title="expiry"), alt.Tooltip("atm_iv:Q", format=".1%"),
                     alt.Tooltip("n_strikes:Q", title="strikes"), alt.Tooltip("n_trades:Q", title="trades")])
    ).properties(title="ATM term structure at this snapshot", width=W, height=H - 60)

    mo.vstack([
        mo.md("""
        ## 2 · Smile explorer

        Every Deribit expiry live at one snapshot. **Points**: each strike's median mark vol over the ±30-minute
        window (hover for strike and trade count). **Lines**: the fitted quadratic over the strikes it was fitted to.
        Snapshots are hourly; an expiry shows when at least three of its strikes traded in the window around it, so
        quiet long-dated expiries come and go from hour to hour.
        """),
        mo.hstack([date_select, time_select], justify="start"),
        snapshot_chart,
        term_now_chart,
        mo.md(f"<span style='color:{INK2}'>Only expiries with a valid fit are shown; strikes beyond ±3 ATM standard deviations are left off.</span>"),
    ])
    return


@app.cell
def _(H, INK2, W, alt, iv, mo, np, pd, smiles, strikes):
    # Pooled surface (method step 4): typical smile shape x typical ATM vol, as in 03
    # Time bins in hours: fine near expiry, then days, weeks, months, out to the longest expiry traded
    _all_edges = [1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 72, 96, 120, 144, 168, 240, 336, 504, 720, 1080, 1440, 2160,
                  4380, 8760]
    H_EDGES = [h for h in _all_edges if h < smiles.hours_to_expiry.max()]
    H_EDGES.append(next(h for h in _all_edges if h >= smiles.hours_to_expiry.max()))
    SD_EDGES = np.arange(-3.0, 3.001, 0.25)
    MIN_OBS, MIN_SNAPS = 30, 20
    surf_in = strikes.assign(sd_from_spot=strikes.sd_from_fwd, iv_surface=strikes.iv)   # pooled_surface's column names


    def deribit_surface(h_edges, m_edges):
        return iv.pooled_surface(surf_in, h_edges, m_edges, min_obs=MIN_OBS, min_snaps=MIN_SNAPS)


    def _span(h):
        return f"{round(h):g}h" if h < 24 else f"{h / 24:g}d"   # round: pd.cut nudges the first edge below 1


    _grid = deribit_surface(H_EDGES, SD_EDGES)
    surface = _grid.stack().rename("median").dropna().reset_index()
    surface["z_mid"] = surface.m.map(lambda i: round(i.mid, 3)).astype(float)
    surface["z_lo"], surface["z_hi"] = surface.z_mid - 0.125, surface.z_mid + 0.125
    surface["tau"] = surface.h.map(lambda i: f"{_span(i.left)}–{_span(i.right)}")
    _tau_order = [f"{_span(a)}–{_span(b)}" for a, b in zip(H_EDGES[:-1], H_EDGES[1:])][::-1]
    surface = surface.drop(columns=["m", "h"])
    surface_chart = (
        alt.Chart(surface).mark_rect()
        .encode(alt.X("z_lo:Q", title="distance from forward (ATM standard deviations)", scale=alt.Scale(domain=[-3, 3])),
                alt.X2("z_hi:Q"),
                alt.Y("tau:O", title="time to expiry", sort=_tau_order),
                alt.Color("median:Q", title="implied vol", legend=alt.Legend(format=".0%", orient="right"),
                          scale=alt.Scale(scheme="oranges", domain=[0.3, 0.7], clamp=True)),
                tooltip=[alt.Tooltip("tau:O", title="time to expiry"), alt.Tooltip("z_mid:Q", title="ATM SDs"),
                         alt.Tooltip("median:Q", title="implied vol", format=".1%")])
        .properties(title="Deribit implied-vol surface: typical smile shape x typical ATM vol", width=W,
                    height=18 * (len(H_EDGES) - 1))
    )

    # ATM term structure: median and interquartile range of the fitted ATM vol per time bin
    _s = smiles.assign(h=pd.cut(smiles.hours_to_expiry, H_EDGES, include_lowest=True)).dropna(subset=["h"])
    term = _s.groupby("h", observed=True).atm_iv.describe()[["25%", "50%", "75%", "count"]].reset_index()
    term["hours"] = term.h.map(lambda i: i.mid).astype(float)
    term = term.drop(columns="h")
    _ticks = [h for h in [1, 3, 6, 12, 24, 72, 168, 720, 2160, 8760] if h <= H_EDGES[-1]]
    _x = alt.X("hours:Q", title="time to expiry (log scale)", scale=alt.Scale(type="log", domain=[1, H_EDGES[-1]]),
               axis=alt.Axis(values=_ticks, labelExpr="datum.value < 24 ? datum.value + 'h' : "
                                                      "datum.value < 720 ? datum.value / 24 + 'd' : "
                                                      "round(datum.value / 730) + 'mo'"))
    term_chart = (
        alt.Chart(term).mark_area(opacity=0.25, color="#eb6834").encode(_x, alt.Y("25%:Q"), alt.Y2("75%:Q"))
        + alt.Chart(term).mark_line(point=True).encode(
            _x, alt.Y("50%:Q", title="ATM vol", axis=alt.Axis(format=".0%"), scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("hours:Q", format=".1f"), alt.Tooltip("50%:Q", title="median", format=".1%"),
                     alt.Tooltip("25%:Q", format=".1%"), alt.Tooltip("75%:Q", format=".1%"),
                     alt.Tooltip("count:Q", title="smiles")])
    ).properties(title="Deribit ATM vol by time to expiry (median, shaded: interquartile range)", width=W, height=H)

    _cell = lambda h, z: surface.loc[(surface.tau == h) & np.isclose(surface.z_mid, z), "median"]
    _row = "1d–1.5d"
    _p, _c, _a = (_cell(_row, z) for z in (-1.875, 1.875, 0.125))
    _skew_txt = (f"At {_row} to expiry the cell around −1.9 SDs is at **{_p.iloc[0]:.1%}**, around +1.9 SDs "
                 f"**{_c.iloc[0]:.1%}**, against **{_a.iloc[0]:.1%}** near the money."
                 if len(_p) and len(_c) and len(_a) else "")
    EMPTY_NOTE = """**Reading the rows.** The **4–5 day row stands out** (higher at every strike). Beyond four days only Friday
    expiries trade (Deribit lists dailies about four days ahead), and that row sees them on Sunday and Monday, when
    about 5% of the remaining life is weekend hours, against 22–30% in the 5–7 day rows. BTC moves more on weekdays
    (`04`, section 6), so an annualised vol over a weekday-only stretch is higher; that is the likely reason. The
    empty top-left corner is the far put wing of the longest expiries, where few strikes trade."""
    mo.vstack([
        mo.md(f"""
        ## 3 · The pooled surface

        Rows are time to expiry, columns quarter-standard-deviation strike bins. Each cell is the median of
        (strike mark vol ÷ its smile's ATM vol) times the median ATM vol of the row; cells need at least
        {MIN_OBS} strike-snapshots from {MIN_SNAPS} snapshots, and thinner cells are left empty. The colour scale is
        capped at 30–70% so the body of the surface stays readable. {_skew_txt}
        """),
        surface_chart,
        term_chart,
        mo.md(f"<span style='color:{INK2}'>{EMPTY_NOTE}</span>"),
    ])
    return H_EDGES, SD_EDGES, deribit_surface, surface_chart, term_chart


@app.cell
def _(H_EDGES, INK2, SD_EDGES, deribit_surface, go, mo):
    def surface_figure(g, title, log_time=True):
        """Plotly 3D surface of a `pooled_surface` grid, styled as 03's (Deribit orange instead of Kalshi blue).
        Time on a log axis with explicit labels unless `log_time` is False."""
        ticks = (dict(type="log", tickvals=[1, 3, 12, 24, 72, 168, 720, 2160, 8760],
                      ticktext=["1h", "3h", "12h", "1d", "3d", "1w", "1mo", "3mo", "1y"]) if log_time else
                 dict(type="linear", tickvals=[24, 48, 72, 96, 120, 144, 168],
                      ticktext=["1d", "2d", "3d", "4d", "5d", "6d", "7d"]))
        fig = go.Figure(go.Surface(
            x=[i.mid for i in g.columns], y=[i.mid for i in g.index], z=g.to_numpy(),
            colorscale=[[0, "#fde3d6"], [1, "#7a2a0e"]], cmin=0.3, cmax=0.7,
            colorbar=dict(title="IV", tickformat=".0%"),
            hovertemplate="ATM standard deviations %{x:.2f}<br>hours to expiry %{y:.1f}<br>implied vol %{z:.1%}<extra></extra>"))
        fig.update_layout(
            title=title, height=600, margin=dict(l=0, r=0, t=50, b=0), paper_bgcolor="#fcfcfb",
            font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2),
            scene=dict(xaxis_title="distance from forward (ATM standard deviations)", yaxis=dict(title="time to expiry", **ticks),
                       zaxis=dict(title="implied vol", tickformat=".0%"), camera=dict(eye=dict(x=1.7, y=-1.5, z=0.8))))
        return fig


    surface_3d = surface_figure(deribit_surface(H_EDGES, SD_EDGES), "Deribit implied-vol surface, every expiry")
    surface_3d_week = surface_figure(deribit_surface([24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168], SD_EDGES),
                                     "Deribit implied-vol surface, 1 to 7 days to expiry (Kalshi's weekly horizon)",
                                     log_time=False)
    mo.vstack([
        mo.md("""
        ## 4 · 3D surfaces

        Drag to rotate, scroll to zoom, hover for values. Same construction as the heatmap. The first covers every
        expiry with time on a log axis, so the last hours and the longest expiries are both visible; the second covers
        1 to 7 days, the life of a Kalshi weekly, on a linear axis in 12-hour rows. Holes are cells with too few
        observations.
        """),
        surface_3d,
        surface_3d_week,
    ])
    return surface_3d, surface_3d_week


@app.cell
def _(ROOT, mo, surface_3d, surface_3d_week, surface_chart, term_chart):
    # Save the figures when run as a script (python 15_deribit_iv_surface.py); opening the notebook writes nothing
    if not mo.running_in_notebook():
        _out = ROOT / "outputs" / "figures"
        _out.mkdir(parents=True, exist_ok=True)
        for _name, _chart in [("deribit_iv_surface", surface_chart), ("deribit_atm_term_structure", term_chart)]:
            _chart.save(_out / f"15_{_name}.png", scale_factor=2)
        (ROOT / "outputs" / "15_deribit_iv_surface_3d.html").write_text(
            "<!doctype html><html><head><meta charset='utf-8'><title>Deribit implied-vol surface</title></head><body>"
            + surface_3d.to_html(include_plotlyjs="cdn", full_html=False)
            + surface_3d_week.to_html(include_plotlyjs=False, full_html=False) + "</body></html>")
    return


if __name__ == "__main__":
    app.run()
