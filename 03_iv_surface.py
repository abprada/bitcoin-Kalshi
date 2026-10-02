import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 03 · Black-Scholes implied-volatility surface

    Each Kalshi strike is a **binary call**: its price D(K) is the market's probability that BTC settles
    above K. Inverting each binary price for a volatility directly is unreliable near the money (often no
    solution, and 1¢ of noise moves the answer by ~17 vol points), and it mixes the skew into the level.
    Instead this notebook uses the fact that a binary is the strike-derivative of a call, D(K) = −dC/dK:

    1. **Snapshots**: one ladder per weekly every 15 minutes, using usable mids with a quote at most 15
       minutes old, priced between 2¢ and 98¢, at least 1 hour before expiry, with at least 5 strikes
       spanning spot.
    2. **Monotone ladder**: binary prices must fall as the strike rises; violations are removed by a
       weighted isotonic fit (weight 1/spread) and counted.
    3. **Tails**: beyond the outermost strikes the distribution follows Black-Scholes at that edge strike's
       binary-implied vol, which gives closed-form tail prices.
    4. **Integrate**: calls C(K) = ∫ above K of D, puts P(K) = ∫ below K of (1 − D), each plus its tail.
    5. **Forward from put-call parity**: C − P + K is the same at every strike, the mean of the ladder's
       distribution. That implied forward is used for the inversion; its gap to exchange spot is reported
       as the **forward basis** (bps).
    6. **Invert** out-of-the-money calls (K ≥ forward) and puts (K < forward) for ordinary Black-Scholes
       implied vols, with zero rates (Kalshi pays interest on collateral; over at most a week discounting
       is below 0.1¢). **One vanilla implied vol per usable strike.**

    For comparison each strike also carries the **direct binary IV** (inverting N(d₂), taking the root on
    the same branch as the snapshot's ATM vol). Snapshot diagnostics: monotonicity fixes, forward basis,
    tail vols, and **strikes per standard deviation**, which drops near expiry as $500 strike spacing
    becomes coarse relative to the distribution. The method is implemented in `lib/iv.py` and tested in
    `tests/test_iv.py` (including exact recovery of a flat smile and of a forward away from spot).

    **Outputs** (`data/derived/`, not in git): `iv_strikes.parquet` (one implied vol per strike per snapshot)
    and `iv_snapshots.parquet` (one row per smile, with its diagnostics). Column definitions are in the
    **Outputs** section at the end of this notebook.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import plotly.graph_objects as go

    from lib import iv
    from lib.fetch_utils import write_parquet

    return alt, go, iv, mo, np, pd, write_parquet


@app.cell
def _(alt):
    # Chart styling: reference palette (slot 1 blue, slot 2 orange), recessive gray chrome.
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
                "axisX": {"labelAngle": 0},
                "legend": {"labelColor": INK2, "titleColor": INK2, "orient": "top"},
                "title": {"color": INK, "anchor": "start", "fontWeight": "normal", "fontSize": 14},
                "line": {"color": SLOT1, "strokeWidth": 2},
                "point": {"color": SLOT1, "size": 64, "filled": True},
            }
        }

    return CONTEXT, H, INK2, MUTED, SLOT1, SLOT2, W


@app.cell
def _(mo):
    write_button = mo.ui.run_button(label="Write IV surface to data/derived/", kind="success")
    mo.vstack([
        mo.md("**Writing.** Opening this notebook builds everything in memory. Files are written when you press "
              "the button, or when the notebook runs as a script: `python 03_iv_surface.py`."),
        write_button,
    ])
    return (write_button,)


@app.cell
def _(mo, pd, write_button):
    WRITE = write_button.value or not mo.running_in_notebook()
    ROOT = mo.notebook_dir()
    CLEAN, DERIVED = ROOT / "data" / "clean", ROOT / "data" / "derived"

    SNAPSHOT_EVERY_MIN = 15
    MAX_QUOTE_AGE_MIN = 15
    PRICE_BAND = (0.02, 0.98)
    MIN_HOURS_TO_EXPIRY = 1.0
    MIN_STRIKES = 5

    # read weekly by weekly and keep only snapshot minutes, so memory stays small
    _cols = ["event_ticker", "market_ticker", "strike", "ts", "tau_years", "mid", "spread", "quote_age_min", "spot", "usable"]
    _frames = []
    for _f in sorted((CLEAN / "kalshi_weekly_quotes").glob("*.parquet")):
        _d = pd.read_parquet(_f, columns=_cols)
        _d = _d[_d.usable & (_d.ts.dt.minute % SNAPSHOT_EVERY_MIN == 0) & (_d.quote_age_min <= MAX_QUOTE_AGE_MIN)
                & _d.mid.between(*PRICE_BAND) & (_d.tau_years * 8760 >= MIN_HOURS_TO_EXPIRY)]
        _frames.append(_d.drop(columns="usable"))
    quotes = pd.concat(_frames, ignore_index=True)

    _s = quotes.groupby(["event_ticker", "ts"]).agg(n=("strike", "size"), spot=("spot", "first"),
                                                   lo=("strike", "min"), hi=("strike", "max"))
    _spans = (_s.lo < _s.spot) & (_s.hi > _s.spot)
    snapshots = _s[(_s.n >= MIN_STRIKES) & _spans].reset_index()
    mo.md(f"""
    **Inputs.** {len(quotes):,} filtered quotes in {len(_s):,} candidate snapshots. Kept **{len(snapshots):,}**
    snapshots ({len(snapshots) / len(_s):.0%}); dropped {int((_s.n < MIN_STRIKES).sum()):,} with fewer than
    {MIN_STRIKES} usable strikes and {int(((_s.n >= MIN_STRIKES) & ~_spans).sum()):,} whose strikes do not span spot.
    Median usable strikes per kept snapshot: {snapshots.n.median():.0f}.
    """)
    return CLEAN, DERIVED, ROOT, WRITE, quotes, snapshots


@app.cell
def _(DERIVED, WRITE, iv, mo, np, pd, quotes, snapshots, write_parquet):
    # One integrated smile per snapshot (iv.snapshot_iv), collected into a strike table and a snapshot table
    _q = quotes.merge(snapshots[["event_ticker", "ts"]], on=["event_ticker", "ts"]).sort_values(["event_ticker", "ts", "strike"])
    _ev, _ts = _q.event_ticker.to_numpy(), _q.ts.to_numpy()
    _starts = np.flatnonzero(np.r_[True, (_ev[1:] != _ev[:-1]) | (_ts[1:] != _ts[:-1])])
    _ends = np.r_[_starts[1:], len(_q)]
    _K, _M = _q.strike.to_numpy(float), _q.mid.to_numpy(float)
    _Wt, _S, _T = 1 / _q.spread.to_numpy(float), _q.spot.to_numpy(float), _q.tau_years.to_numpy(float)
    _tickers = _q.market_ticker.to_numpy()

    _strike_parts, _snap_rows = [], []
    for _i in mo.status.progress_bar(range(len(_starts)), title="Building smiles",
                                     subtitle=f"{len(_starts):,} snapshots, one per weekly per 15 minutes"):
        _a, _b = _starts[_i], _ends[_i]
        _st, _sn = iv.snapshot_iv(_K[_a:_b], _M[_a:_b], _Wt[_a:_b], _S[_a], _T[_a])
        _st["snap"] = _i
        _st["market_ticker"] = _tickers[_a:_b][np.argsort(_K[_a:_b])]
        _strike_parts.append(_st)
        _snap_rows.append({"snap": _i, "event_ticker": _ev[_a], "ts": _ts[_a], "spot": _S[_a], "tau_years": _T[_a],
                           "n_empty_iv": int(_st.iv.isna().sum()), **_sn})

    iv_snapshots = pd.DataFrame(_snap_rows)
    iv_snapshots["ts"] = pd.to_datetime(iv_snapshots.ts, utc=True)
    iv_snapshots["hours_to_expiry"] = iv_snapshots.tau_years * 8760
    # a snapshot is "clean" if its ATM vol exists, its strike grid resolves the distribution (>= 2 strikes per
    # standard deviation) and Kalshi's implied forward is within 25 bps of exchange spot
    iv_snapshots["clean"] = (iv_snapshots.atm_iv.notna() & (iv_snapshots.strikes_per_sd >= 2)
                             & (iv_snapshots.forward_basis_bps.abs() <= 25))
    iv_strikes = pd.concat(_strike_parts, ignore_index=True).merge(
        iv_snapshots[["snap", "event_ticker", "ts", "spot", "tau_years", "hours_to_expiry", "forward", "atm_iv", "clean"]], on="snap")
    iv_strikes["moneyness"] = iv_strikes.strike / iv_strikes.spot  # strike / exchange spot: the axis used in the charts
    iv_strikes["log_moneyness"] = np.log(iv_strikes.strike / iv_strikes.forward)
    iv_strikes["sd_from_forward"] = iv_strikes.log_moneyness / (iv_strikes.atm_iv * np.sqrt(iv_strikes.tau_years))

    if WRITE:
        write_parquet(iv_strikes, DERIVED / "iv_strikes.parquet")
        write_parquet(iv_snapshots, DERIVED / "iv_snapshots.parquet")
    mo.md(f"Built **{len(iv_snapshots):,}** smiles with **{int(iv_strikes.iv.notna().sum()):,}** strike-level implied vols; "
          f"**{iv_snapshots.clean.mean():.0%}** of snapshots are clean. Written to `data/derived/`: {'yes' if WRITE else 'no'}.")
    return iv_snapshots, iv_strikes


@app.cell
def _(H, SLOT1, W, alt, iv_snapshots, iv_strikes, mo, np, pd):
    _s = iv_snapshots
    _fail = pd.DataFrame([
        ("ATM vol could not be interpolated", int(_s.atm_iv.isna().sum())),
        ("fewer than 2 strikes per standard deviation", int((_s.strikes_per_sd < 2).sum())),
        ("forward basis beyond ±25 bps", int((_s.forward_basis_bps.abs() > 25).sum())),
    ], columns=["reason a snapshot is not clean (can overlap)", "snapshots"])

    _hb = pd.cut(_s.hours_to_expiry, [1, 3, 6, 12, 24, 48, 72, 120, 170],
                 labels=["1–3h", "3–6h", "6–12h", "12–24h", "1–2d", "2–3d", "3–5d", "5–7d"])
    _by_h = (_s.assign(h=_hb).groupby("h", observed=False)
             .agg(snapshots=("snap", "size"), clean=("clean", "mean"), strikes_per_sd=("strikes_per_sd", "median"),
                  basis_bps=("forward_basis_bps", "median"),
                  basis_abs_p90=("forward_basis_bps", lambda x: x.abs().quantile(0.9)))
             .reset_index())
    H_ORDER = list(_hb.cat.categories)
    clean_chart = (
        alt.Chart(_by_h).mark_bar(color=SLOT1)
        .encode(x=alt.X("h:N", sort=H_ORDER, title="time to expiry"),
                y=alt.Y("clean:Q", title="share of snapshots clean", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                tooltip=["h:N", alt.Tooltip("clean:Q", format=".1%"), alt.Tooltip("strikes_per_sd:Q", format=".1f"),
                         alt.Tooltip("basis_abs_p90:Q", format=".1f", title="90th pct |basis| (bps)"), "snapshots:Q"])
        .properties(title="Clean snapshots by time to expiry", width=W, height=H)
    )
    _b = _s.forward_basis_bps.clip(-60, 60)
    _hist = pd.cut(_b, np.arange(-60, 62.5, 2.5), include_lowest=True).value_counts(normalize=True).sort_index()
    basis_chart = (
        alt.Chart(pd.DataFrame({"bps": [i.mid for i in _hist.index], "share": _hist.to_numpy()})).mark_bar(color=SLOT1)
        .encode(x=alt.X("bps:Q", bin=alt.Bin(step=2.5), title="Kalshi-implied forward minus exchange spot (bps; clipped at ±60)"),
                y=alt.Y("share:Q", title="share of snapshots", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("bps:Q", format=".1f"), alt.Tooltip("share:Q", format=".2%")])
        .properties(title="Forward basis", width=W, height=H)
    )

    # direct binary IV vs integrated vanilla IV, by distance from the forward
    _k = iv_strikes[iv_strikes.clean]
    _mb = pd.cut(_k.log_moneyness.abs(), [0, 0.005, 0.02, 0.05, 1], labels=["< 0.5%", "0.5–2%", "2–5%", "> 5%"], include_lowest=True)
    direct_vs_integrated = (_k.assign(m=_mb, diff=(_k.digital_iv - _k.iv).abs())
                            .groupby("m", observed=False)
                            .agg(strikes=("iv", "size"), direct_missing=("digital_iv", lambda x: x.isna().mean()),
                                 median_abs_gap_vol_pts=("diff", lambda x: 100 * x.median()))
                            .reset_index().rename(columns={"m": "distance from forward"}))
    direct_vs_integrated["direct_missing"] = direct_vs_integrated.direct_missing.map("{:.1%}".format)
    direct_vs_integrated["median_abs_gap_vol_pts"] = direct_vs_integrated.median_abs_gap_vol_pts.round(1)

    mo.vstack([
        mo.md(f"""
        ## Checks

        **Clean snapshots: {_s.clean.mean():.0%}.** Every filtered strike received an implied vol
        ({int(iv_strikes.iv.notna().sum()):,} of {len(iv_strikes):,}). Monotonicity fixes were needed in
        {(_s.monotone_fixes > 0).mean():.0%} of snapshots (median {_s.monotone_fixes[_s.monotone_fixes > 0].median():.0f}
        strikes when needed).
        """),
        mo.hstack([mo.ui.table(_fail, selection=None, show_download=False)], justify="start"),
        clean_chart,
        mo.md(f"""
        **Forward basis.** Median {_s.forward_basis_bps.median():+.1f} bps, 5th–95th percentile
        {_s.forward_basis_bps.quantile(0.05):+.1f} to {_s.forward_basis_bps.quantile(0.95):+.1f} bps.
        """),
        basis_chart,
        mo.md("**Direct binary IV vs integrated vanilla IV** (clean snapshots): how far the naive per-strike inversion "
              "is from the integrated vol, and how often it has no solution at all."),
        mo.hstack([mo.ui.table(direct_vs_integrated, selection=None, show_download=False)], justify="start"),
    ])
    return basis_chart, clean_chart


@app.cell
def _(iv_snapshots, mo):
    _weeks = iv_snapshots.groupby("event_ticker").ts.max().sort_values().index.tolist()  # by expiry
    week_select = mo.ui.dropdown(options=_weeks, value=_weeks[-1], label="Weekly")
    hours_slider = mo.ui.slider(1, 168, step=1, value=24, label="Hours to expiry", show_value=True)
    mo.vstack([mo.md("## Smile explorer\n\nPick a weekly and a time to expiry; the nearest snapshot is shown."),
               mo.hstack([week_select, hours_slider], justify="start", gap=2)])
    return hours_slider, week_select


@app.cell
def _(CONTEXT, MUTED, SLOT1, W, alt, iv_strikes, pd):
    def make_smile_chart(row):
        """Integrated vanilla IV (blue) and direct binary IV (gray) across strikes for one snapshot row."""
        k = iv_strikes[iv_strikes.snap == row.snap]
        long = pd.concat([k.assign(method="integrated vanilla IV", vol=k.iv), k.assign(method="direct binary IV", vol=k.digital_iv)])
        color = alt.Color("method:N", scale=alt.Scale(domain=["integrated vanilla IV", "direct binary IV"], range=[SLOT1, MUTED]), title=None)
        tip = [alt.Tooltip("strike:Q", format=",.0f"), alt.Tooltip("mid:Q", format=".3f", title="binary mid"),
               alt.Tooltip("iv:Q", format=".1%", title="integrated IV"), alt.Tooltip("digital_iv:Q", format=".1%", title="direct binary IV"),
               alt.Tooltip("option:N")]
        base = alt.Chart(long).encode(x=alt.X("moneyness:Q", title="moneyness (strike / spot)", scale=alt.Scale(zero=False), axis=alt.Axis(format=".2f")),
                                      y=alt.Y("vol:Q", title="implied vol", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
                                      color=color, tooltip=tip)
        chart = (
            base.transform_filter("datum.method == 'integrated vanilla IV'").mark_line(point=True)
            + base.transform_filter("datum.method == 'direct binary IV'").mark_point(filled=True, size=40, opacity=0.8)
            + alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(color=CONTEXT).encode(x="x:Q")
        ).properties(width=W, height=280, title=(
            f"{row.event_ticker} · {row.hours_to_expiry:.1f}h to expiry · ATM {row.atm_iv:.1%} · "
            f"forward basis {row.forward_basis_bps:+.1f} bps · {'clean' if row.clean else 'not clean'}"))
        caption = (f"{int(row.n_strikes)} usable strikes · {int(row.monotone_fixes)} monotonicity fixes · "
                   f"{row.strikes_per_sd:.1f} strikes per standard deviation · tail vols {row.sigma_low:.0%} / {row.sigma_high:.0%} · "
                   f"spot {row.spot:,.0f}, implied forward {row.forward:,.0f}")
        return chart, caption

    return (make_smile_chart,)


@app.cell
def _(hours_slider, iv_snapshots, make_smile_chart, mo, week_select):
    _s = iv_snapshots[iv_snapshots.event_ticker == week_select.value]
    _row = _s.iloc[int((_s.hours_to_expiry - hours_slider.value).abs().argmin())]
    smile_chart, _caption = make_smile_chart(_row)
    mo.vstack([smile_chart, mo.md(_caption)])
    return


@app.cell
def _(CLEAN, H, SLOT1, SLOT2, W, alt, iv_snapshots, iv_strikes, mo, np, pd):
    # Pooled surface: median implied vol by moneyness (strike / spot) x time to expiry (clean snapshots)
    _c = iv_strikes[iv_strikes.clean & iv_strikes.iv.notna()]
    TAU_LABELS = ["1–3h", "3–6h", "6–12h", "12–24h", "1–2d", "2–3d", "3–5d", "5–7d"]
    _tau = pd.cut(_c.hours_to_expiry, [1, 3, 6, 12, 24, 48, 72, 120, 170], labels=TAU_LABELS)
    _z = pd.cut(_c.moneyness, np.arange(0.85, 1.1501, 0.01))
    surface = (_c.assign(tau=_tau, z=_z).groupby(["tau", "z"], observed=True).iv.agg(["median", "size"])
               .reset_index().query("size >= 30"))
    surface["z_mid"] = surface.z.map(lambda i: round(i.mid, 3)).astype(float)
    surface["z_lo"], surface["z_hi"] = surface.z_mid - 0.005, surface.z_mid + 0.005
    surface = surface.drop(columns="z").astype({"tau": str})  # intervals/categoricals do not serialize to the chart
    surface_chart = (
        alt.Chart(surface).mark_rect()
        .encode(x=alt.X("z_lo:Q", title="moneyness (strike / spot)",
                        scale=alt.Scale(domain=[0.85, 1.15]), axis=alt.Axis(values=[0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15], format=".2f")),
                x2="z_hi:Q",
                y=alt.Y("tau:N", sort=TAU_LABELS[::-1], title="time to expiry"),
                color=alt.Color("median:Q", title="median IV",
                                scale=alt.Scale(range=["#cde2fb", "#0d366b"], domain=[0.35, 0.9], clamp=True),
                                legend=alt.Legend(format="%")),
                tooltip=["tau:N", alt.Tooltip("z_mid:Q", title="moneyness", format=".3f"), alt.Tooltip("median:Q", format=".1%"),
                         alt.Tooltip("size:Q", format=",", title="strike-snapshots")])
        .properties(title="Implied-vol surface: median across all clean snapshots", width=W, height=260)
    )

    _t = iv_snapshots[iv_snapshots.clean].assign(h=lambda d: np.floor(d.hours_to_expiry).astype(int))
    term = (_t.groupby("h").atm_iv.agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), n="size")
            .reset_index().query("n >= 20"))
    term_chart = (
        alt.Chart(term).mark_area(color=SLOT1, opacity=0.18).encode(x=alt.X("h:Q", title="hours to expiry", scale=alt.Scale(domain=[168, 0])),
                                                                    y=alt.Y("q25:Q", title="ATM implied vol", axis=alt.Axis(format="%")), y2="q75:Q")
        + alt.Chart(term).mark_line().encode(x="h:Q", y="median:Q",
                                             tooltip=["h:Q", alt.Tooltip("median:Q", format=".1%"), alt.Tooltip("q25:Q", format=".1%"),
                                                      alt.Tooltip("q75:Q", format=".1%"), "n:Q"])
    ).properties(title="ATM implied vol by time to expiry (median, interquartile band)", width=W, height=H)

    # sanity: Kalshi ATM vol vs BTC realized vol, per day (realized from 15-minute exchange returns)
    _px = pd.read_parquet(CLEAN / "btc_spot_1m.parquet", columns=["ts", "spot"]).set_index("ts").spot
    _r = np.log(_px[_px.index.minute % 15 == 0]).diff().dropna()
    _d = (_r ** 2).resample("D").agg(["sum", "count"])
    _rv = np.sqrt(_d.loc[_d["count"] >= 90, "sum"] * 365)
    _atm = iv_snapshots[iv_snapshots.clean].set_index("ts").atm_iv.resample("D").median()
    _both = pd.DataFrame({"Kalshi ATM implied vol": _atm, "BTC realized vol (same day)": _rv}).dropna()
    _long = _both.rename_axis("day").reset_index().melt(id_vars="day", var_name="series", value_name="vol")
    rv_chart = (
        alt.Chart(_long).mark_line(strokeWidth=1.6)
        .encode(x=alt.X("day:T", title=None), y=alt.Y("vol:Q", title="annualized vol", axis=alt.Axis(format="%")),
                color=alt.Color("series:N", scale=alt.Scale(domain=list(_both.columns), range=[SLOT1, SLOT2]), title=None),
                tooltip=[alt.Tooltip("day:T", format="%Y-%m-%d"), "series:N", alt.Tooltip("vol:Q", format=".1%")])
        .properties(title="Kalshi ATM implied vol vs realized vol, daily", width=W, height=H)
    )
    _corr = _both.corr().iloc[0, 1]
    mo.vstack([
        mo.md("## Surface"),
        surface_chart,
        mo.md(f"""
        Reading across a row gives the average smile at that time to expiry; reading down a column shows how
        vol at a given moneyness changes as expiry approaches. Cells need at least 30
        strike-snapshots, and near expiry the strikes only span a narrow band of moneyness, so those rows are narrower; the color scale is capped at 35–90% so the body of the surface stays readable
        (the wings 3–6 hours before expiry exceed 100%).
        """),
        term_chart,
        mo.md(f"""
        Median ATM vol is {term.loc[term.h >= 120, 'median'].median():.0%} with 5–7 days to go and
        {term.loc[term.h < 12, 'median'].median():.0%} in the last 12 hours.
        """),
        rv_chart,
        mo.md(f"""
        **Sanity check.** Across {len(_both)} days, Kalshi's ATM implied vol averages {_both.iloc[:, 0].mean():.0%}
        against {_both.iloc[:, 1].mean():.0%} realized, with a daily correlation of {_corr:.2f}.
        """),
    ])
    return rv_chart, surface_chart, term_chart


@app.cell
def _(INK2, go, iv_strikes, mo, np, pd):
    # Rotatable 3D surface: median IV over moneyness (strike / spot) x hours to expiry (clean snapshots)
    MONEYNESS_EDGES = np.arange(0.85, 1.1501, 0.01)
    _c = iv_strikes[iv_strikes.clean & iv_strikes.iv.notna()]
    _h_edges = [1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 72, 96, 120, 144, 168.5]
    _g = (_c.assign(h=pd.cut(_c.hours_to_expiry, _h_edges), z=pd.cut(_c.moneyness, MONEYNESS_EDGES))
          .groupby(["h", "z"], observed=False).iv.agg(["median", "size"]))
    _grid = _g["median"].where(_g["size"] >= 30).unstack("z")


    def surface_figure(x, y, z, title):
        """Plotly 3D surface of implied vol; hours to expiry on a log axis so the last hours are visible."""
        fig = go.Figure(go.Surface(
            x=x, y=y, z=z, colorscale=[[0, "#cde2fb"], [1, "#0d366b"]], cmin=0.35, cmax=0.9,
            colorbar=dict(title="IV", tickformat=".0%"),
            hovertemplate="moneyness %{x:.3f}<br>hours to expiry %{y:.1f}<br>implied vol %{z:.1%}<extra></extra>"))
        fig.update_layout(
            title=title, height=600, margin=dict(l=0, r=0, t=50, b=0), paper_bgcolor="#fcfcfb",
            font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2),
            scene=dict(xaxis_title="moneyness (strike / spot)", yaxis=dict(title="hours to expiry", type="log"),
                       zaxis=dict(title="implied vol", tickformat=".0%"), camera=dict(eye=dict(x=1.7, y=-1.5, z=0.8))))
        return fig


    surface_3d = surface_figure([i.mid for i in _grid.columns], [i.mid for i in _grid.index], _grid.to_numpy(),
                                "Implied-vol surface: median across all clean snapshots")
    mo.vstack([
        mo.md("## 3D surface\n\nDrag to rotate, scroll to zoom, hover for values. Hours to expiry are on a log axis so the "
              "final hours are visible; cells need at least 30 strike-snapshots."),
        surface_3d,
    ])
    return MONEYNESS_EDGES, surface_3d, surface_figure


@app.cell
def _(MONEYNESS_EDGES, iv_strikes, np, pd, surface_figure, week_select):
    # The selected weekly's own surface over its life (follows the weekly dropdown above)
    _w = iv_strikes[(iv_strikes.event_ticker == week_select.value) & iv_strikes.iv.notna()]
    _g = (_w.assign(h=pd.cut(_w.hours_to_expiry, np.arange(0, 172, 2)), z=pd.cut(_w.moneyness, MONEYNESS_EDGES))
          .groupby(["h", "z"], observed=False).iv.median().unstack("z"))
    week_surface_3d = surface_figure([i.mid for i in _g.columns], [i.mid for i in _g.index], _g.to_numpy(),
                                     f"{week_select.value}: implied vol over the contract's life (2-hour bins)")
    week_surface_3d
    return


@app.cell
def _(
    ROOT,
    WRITE,
    basis_chart,
    clean_chart,
    iv_snapshots,
    make_smile_chart,
    mo,
    rv_chart,
    surface_3d,
    surface_chart,
    term_chart,
):
    # Figures: PNGs of the charts above (2x resolution) and the 3D surface as a standalone HTML page
    OUTPUTS = ROOT / "outputs"
    FIGURES = OUTPUTS / "figures"
    _last = iv_snapshots[iv_snapshots.event_ticker == iv_snapshots.loc[iv_snapshots.ts.idxmax(), "event_ticker"]]
    _smiles = {h: _last.iloc[int((_last.hours_to_expiry - h).abs().argmin())] for h in (72, 24, 6)}
    figure_files = []
    if WRITE:
        FIGURES.mkdir(parents=True, exist_ok=True)
        for _name, _chart in {"iv_surface_heatmap": surface_chart, "atm_term_structure": term_chart,
                              "atm_vs_realized": rv_chart, "forward_basis": basis_chart, "clean_by_ttm": clean_chart}.items():
            _chart.save(FIGURES / f"{_name}.png", scale_factor=2)
            figure_files.append(f"figures/{_name}.png")
        for _h, _r in _smiles.items():
            _f = f"smile_{_r.event_ticker}_{_h}h.png"
            make_smile_chart(_r)[0].save(FIGURES / _f, scale_factor=2)
            figure_files.append(f"figures/{_f}")
        surface_3d.write_html(OUTPUTS / "iv_surface_3d.html", include_plotlyjs="cdn")
        figure_files.append("iv_surface_3d.html")
    mo.md(f"**Figures** (`outputs/`, committed to git): {len(figure_files)} files written this run."
          if WRITE else "**Figures** are written to `outputs/` with the data (button at the top, or `python 03_iv_surface.py`).")
    return


@app.cell
def _(WRITE, iv_snapshots, iv_strikes, mo):
    mo.md(f"""
    ## Outputs

    **`data/derived/iv_strikes.parquet`**: one row per strike per snapshot ({len(iv_strikes):,} rows).

    | Column | Meaning |
    |---|---|
    | `snap`, `event_ticker`, `market_ticker`, `ts` | Snapshot id, weekly, strike ticker, snapshot time (UTC) |
    | `tau_years`, `hours_to_expiry` | Time to expiry (ACT/365) |
    | `strike`, `mid` | Strike and the binary's mid price |
    | `D_used` | Binary price after the monotone fit |
    | `option`, `price` | Out-of-the-money option integrated from the ladder (call if K ≥ forward, else put) and its price (USD) |
    | `iv` | **Vanilla Black-Scholes implied vol** from that option |
    | `digital_iv` | Direct binary implied vol (comparison only; unreliable near the money) |
    | `forward`, `atm_iv`, `clean` | Snapshot's implied forward, ATM vol and clean flag |
    | `moneyness` | strike / exchange spot (the axis used in the charts) |
    | `log_moneyness`, `sd_from_forward` | ln(K / implied forward), and the same in standard deviations (ATM vol · √τ) |

    **`data/derived/iv_snapshots.parquet`**: one row per snapshot ({len(iv_snapshots):,} rows): `spot`, `forward`,
    `forward_basis_bps`, `n_strikes`, `n_empty_iv`, `monotone_fixes`, `sigma_low` / `sigma_high` (tail vols),
    `atm_iv`, `strikes_per_sd`, `clean` (ATM vol exists, ≥ 2 strikes per standard deviation, |basis| ≤ 25 bps).

    **`outputs/`** (committed to git): `figures/iv_surface_heatmap.png`, `atm_term_structure.png`,
    `atm_vs_realized.png`, `forward_basis.png`, `clean_by_ttm.png`, three example smiles for the latest weekly
    (`smile_<weekly>_72h.png`, `_24h`, `_6h`), and `iv_surface_3d.html`, the rotatable 3D surface (open it in a
    browser; it loads the plotting library from the web).

    **Written this run:** {"yes" if WRITE else "no (press the button at the top, or run `python 03_iv_surface.py`)"}.
    """)
    return


if __name__ == "__main__":
    app.run()
