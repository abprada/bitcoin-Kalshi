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
    3. **Tails, anchored to Kalshi**: Kalshi says how much probability lies beyond the ladder
       (`prob_outside`) but not how far out, which is what the tails are worth. Each tail is priced with
       Black-Scholes at that edge strike's own binary-implied vol, then clipped to the bounds set by
       Kalshi's other listed strikes (one-sided, too wide, or outside the price band): since D only falls
       as the strike rises, bids bound it from below and asks from above. Beyond the last strike Kalshi
       lists, the Black-Scholes tail alone (negligible). Only Kalshi prices enter, never BTC volatility.
    4. **Integrate**: calls C(K) = ∫ above K of D, puts P(K) = ∫ below K of (1 − D), each plus its tail.
    5. **Forward from put-call parity**: C − P + K is the same at every strike, the mean of the ladder's
       distribution. That implied forward is used for the inversion; its gap to exchange spot is reported
       as the **forward basis** (bps).
    6. **Invert** out-of-the-money calls (K ≥ forward) and puts (K < forward) for ordinary Black-Scholes
       implied vols, with zero rates (Kalshi pays interest on collateral; over at most a week discounting
       is below 0.1¢). **One vanilla implied vol per usable strike.**
    7. **Surface inputs**: the surfaces (heatmap and 3D) use binary prices between 5¢ and 95¢ only (at 2–5¢
       one tick is 20–50% of the price, so the vol mostly reflects the tick size) and winsorize vols at the
       1st/99th percentile within each time-to-expiry bucket (`iv_surface`). The per-strike `iv` stays raw.
    8. **Pooling**: the surfaces pool the smile's *shape* across snapshots: each strike's vol divided by its own
       snapshot's ATM vol, median per cell, times the median ATM vol at that time to expiry. Pooling raw vols
       would mix volatility regimes (near expiry, far strikes only have usable quotes on very volatile days, so
       raw medians spike at the edges). Cells need at least 100 strike-snapshots from 20 snapshots.

    For comparison each strike also carries the **direct binary IV** (inverting N(d₂), taking the root on
    the same branch as the snapshot's ATM vol). Snapshot diagnostics: monotonicity fixes, forward basis,
    probability beyond the outermost strikes, tail vols and whether Kalshi's quotes clipped a tail, and
    **strikes per standard deviation**, which drops near expiry as $500 strike spacing
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
    _cols = ["event_ticker", "market_ticker", "strike", "ts", "tau_years", "bid", "ask", "mid", "spread", "quote_age_min",
             "spot", "exchange_open", "usable"]
    _frames, _outer = [], []
    for _f in sorted((CLEAN / "kalshi_weekly_quotes").glob("*.parquet")):
        _d = pd.read_parquet(_f, columns=_cols)
        _d = _d[_d.exchange_open & (_d.ts.dt.minute % SNAPSHOT_EVERY_MIN == 0) & (_d.quote_age_min <= MAX_QUOTE_AGE_MIN)
                & (_d.tau_years * 8760 >= MIN_HOURS_TO_EXPIRY)]
        _ladder = _d.usable & _d.mid.between(*PRICE_BAND)
        _frames.append(_d[_ladder].drop(columns=["bid", "ask", "exchange_open", "usable"]))
        # every other quoted strike (one-sided, too wide, or outside the price band) bounds the tails (method step 3)
        _outer.append(_d.loc[~_ladder & (_d.bid.notna() | _d.ask.notna()), ["event_ticker", "ts", "strike", "bid", "ask"]])
    quotes = pd.concat(_frames, ignore_index=True)
    outer_quotes = pd.concat(_outer, ignore_index=True)

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
    return CLEAN, DERIVED, ROOT, WRITE, outer_quotes, quotes, snapshots


@app.cell
def _(DERIVED, WRITE, iv, mo, np, outer_quotes, pd, quotes, snapshots, write_parquet):
    # One integrated smile per snapshot (iv.snapshot_iv), collected into a strike table and a snapshot table
    _q = quotes.merge(snapshots[["event_ticker", "ts"]], on=["event_ticker", "ts"]).sort_values(["event_ticker", "ts", "strike"])
    _ev, _ts = _q.event_ticker.to_numpy(), _q.ts.to_numpy()
    _starts = np.flatnonzero(np.r_[True, (_ev[1:] != _ev[:-1]) | (_ts[1:] != _ts[:-1])])
    _ends = np.r_[_starts[1:], len(_q)]
    _K, _M = _q.strike.to_numpy(float), _q.mid.to_numpy(float)
    _Wt, _S, _T = 1 / _q.spread.to_numpy(float), _q.spot.to_numpy(float), _q.tau_years.to_numpy(float)
    _tickers = _q.market_ticker.to_numpy()
    # Kalshi's other listed strikes at each snapshot, for the tail bounds
    _o = outer_quotes.merge(snapshots[["event_ticker", "ts"]], on=["event_ticker", "ts"]).sort_values(["event_ticker", "ts", "strike"])
    _oev, _ots = _o.event_ticker.to_numpy(), _o.ts.to_numpy()
    _os = np.flatnonzero(np.r_[True, (_oev[1:] != _oev[:-1]) | (_ots[1:] != _ots[:-1])]) if len(_o) else np.array([], dtype=int)
    _outer_at = {(_oev[_x], _ots[_x]): (_x, _y) for _x, _y in zip(_os, np.r_[_os[1:], len(_o)])}
    _OK, _OB, _OA = _o.strike.to_numpy(float), _o.bid.to_numpy(float), _o.ask.to_numpy(float)

    _strike_parts, _snap_rows = [], []
    for _i in mo.status.progress_bar(range(len(_starts)), title="Building smiles",
                                     subtitle=f"{len(_starts):,} snapshots, one per weekly per 15 minutes"):
        _a, _b = _starts[_i], _ends[_i]
        _x, _y = _outer_at.get((_ev[_a], _ts[_a]), (0, 0))
        _st, _sn = iv.snapshot_iv(_K[_a:_b], _M[_a:_b], _Wt[_a:_b], _S[_a], _T[_a],
                                  K_out=_OK[_x:_y], bid_out=_OB[_x:_y], ask_out=_OA[_x:_y])
        _st["snap"] = _i
        _st["market_ticker"] = _tickers[_a:_b][np.argsort(_K[_a:_b])]
        _strike_parts.append(_st)
        _snap_rows.append({"snap": _i, "event_ticker": _ev[_a], "ts": _ts[_a], "spot": _S[_a], "tau_years": _T[_a],
                           "n_empty_iv": int(_st.iv.isna().sum()), **_sn})

    iv_snapshots = pd.DataFrame(_snap_rows)
    iv_snapshots["ts"] = pd.to_datetime(iv_snapshots.ts, utc=True)
    iv_snapshots["hours_to_expiry"] = (iv_snapshots.tau_years * 8760).round(4)  # float32 tau: snapshots sit on 15-minute marks
    # a snapshot is "clean" if its ATM vol exists, its strike grid resolves the distribution (>= 2 strikes per
    # standard deviation) and Kalshi's implied forward is within 25 bps of exchange spot
    iv_snapshots["clean"] = (iv_snapshots.atm_iv.notna() & (iv_snapshots.strikes_per_sd >= 2)
                             & (iv_snapshots.forward_basis_bps.abs() <= 25))
    iv_strikes = pd.concat(_strike_parts, ignore_index=True).merge(
        iv_snapshots[["snap", "event_ticker", "ts", "spot", "tau_years", "hours_to_expiry", "forward", "atm_iv", "clean"]], on="snap")
    iv_strikes["moneyness"] = iv_strikes.strike / iv_strikes.spot  # strike / exchange spot: the axis used in the charts
    iv_strikes["log_moneyness"] = np.log(iv_strikes.strike / iv_strikes.forward)
    iv_strikes["sd_from_forward"] = iv_strikes.log_moneyness / (iv_strikes.atm_iv * np.sqrt(iv_strikes.tau_years))
    # Surface input (step 7): clean snapshots, binary prices 5-95c only, vols winsorized at the 1st/99th percentile
    # within each time-to-expiry bucket. The per-strike `iv` column stays raw.
    _surface_in = iv_strikes.clean & iv_strikes.mid.between(0.05, 0.95)
    iv_strikes["iv_surface"] = iv.winsorize_by_group(
        iv_strikes.iv.where(_surface_in), pd.cut(iv_strikes.hours_to_expiry, [1, 3, 6, 12, 24, 48, 72, 120, 170], include_lowest=True), 0.01, 0.99)

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

    _hb = pd.cut(_s.hours_to_expiry, [1, 3, 6, 12, 24, 48, 72, 120, 170], include_lowest=True,
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
                   f"{row.strikes_per_sd:.1f} strikes per standard deviation · {row.prob_outside:.0%} of probability in the tails · tail vols {row.sigma_low:.0%} / {row.sigma_high:.0%} · "
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
def _(CLEAN, H, SLOT1, SLOT2, W, alt, iv, iv_snapshots, iv_strikes, mo, np, pd):
    # Pooled surface (step 8): typical smile shape x typical ATM vol, by moneyness (strike / spot) x time to expiry
    TAU_LABELS = ["1–3h", "3–6h", "6–12h", "12–24h", "1–2d", "2–3d", "3–5d", "5–7d"]
    _grid = iv.pooled_surface(iv_strikes, [1, 3, 6, 12, 24, 48, 72, 120, 170], np.arange(0.85, 1.1501, 0.01),
                              min_obs=100, min_snaps=20)
    _grid.index = pd.Index(TAU_LABELS, name="tau")
    surface = _grid.stack().rename("median").dropna().reset_index()
    surface["z_mid"] = surface.m.map(lambda i: round(i.mid, 3)).astype(float)
    surface["z_lo"], surface["z_hi"] = surface.z_mid - 0.005, surface.z_mid + 0.005
    surface = surface.drop(columns="m").astype({"tau": str})  # intervals/categoricals do not serialize to the chart
    surface_chart = (
        alt.Chart(surface).mark_rect()
        .encode(x=alt.X("z_lo:Q", title="moneyness (strike / spot)",
                        scale=alt.Scale(domain=[0.85, 1.15]), axis=alt.Axis(values=[0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15], format=".2f")),
                x2="z_hi:Q",
                y=alt.Y("tau:N", sort=TAU_LABELS[::-1], title="time to expiry"),
                color=alt.Color("median:Q", title="IV",
                                scale=alt.Scale(range=["#cde2fb", "#0d366b"], domain=[0.35, 0.9], clamp=True),
                                legend=alt.Legend(format="%")),
                tooltip=["tau:N", alt.Tooltip("z_mid:Q", title="moneyness", format=".3f"), alt.Tooltip("median:Q", format=".1%", title="IV")])
        .properties(title="Implied-vol surface: typical smile shape x typical ATM vol", width=W, height=260)
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

    # Same-day sample standard deviation of 15-minute log returns, annualized
    # using 96 intervals/day and 365 calendar days/year (not a rolling window).
    _px = pd.read_parquet(CLEAN / "btc_spot_1m.parquet", columns=["ts", "spot"]).set_index("ts").spot
    _r = np.log(_px[_px.index.minute % 15 == 0]).diff().dropna()
    _d = _r.resample("D").agg(["std", "count"])
    _rv = _d.loc[_d["count"] >= 90, "std"] * np.sqrt(96 * 365)
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
        vol at a given moneyness changes as expiry approaches. Each cell is the median smile shape (each strike's vol
        relative to its own snapshot's ATM vol) times the typical ATM vol at that time to expiry, from at least 100
        strike-snapshots across 20 snapshots (method step 8), and near expiry the strikes only span a narrow band of moneyness, so those rows are narrower; the color scale is capped at 35–90% so the body of the surface stays readable.
        Surfaces use binary prices between 5¢ and 95¢ and vols winsorized at the 1st/99th percentile per
        time-to-expiry bucket (method step 7).
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
        Realized volatility is the sample standard deviation of that UTC day's 15-minute BTC log returns,
        multiplied by √(96 × 365). Days require at least 90 returns; there is no rolling window.
        """),
    ])
    return rv_chart, surface_chart, term_chart


@app.cell
def _(INK2, go, iv, iv_strikes, mo, np):
    # Rotatable 3D surfaces (step 8): typical smile shape x typical ATM vol, moneyness (strike / spot) x time to expiry
    MONEYNESS_EDGES = np.arange(0.85, 1.1501, 0.01)


    def surface_figure(x, y, z, title, log_time=True):
        """Plotly 3D surface of implied vol. Time to expiry on a log axis (so the last hours are visible) unless
        `log_time` is False; explicit tick labels, since plotly's default log labels ("100, 5, 2, 10") read as a
        broken axis."""
        ticks = (dict(type="log", tickvals=[1, 2, 3, 6, 12, 24, 48, 72, 120, 168],
                      ticktext=["1h", "2h", "3h", "6h", "12h", "1d", "2d", "3d", "5d", "7d"]) if log_time else
                 dict(type="linear", tickvals=[24, 48, 72, 96, 120, 144, 168],
                      ticktext=["1d", "2d", "3d", "4d", "5d", "6d", "7d"]))
        fig = go.Figure(go.Surface(
            x=x, y=y, z=z, colorscale=[[0, "#cde2fb"], [1, "#0d366b"]], cmin=0.35, cmax=0.9,
            colorbar=dict(title="IV", tickformat=".0%"),
            hovertemplate="moneyness %{x:.3f}<br>hours to expiry %{y:.1f}<br>implied vol %{z:.1%}<extra></extra>"))
        fig.update_layout(
            title=title, height=600, margin=dict(l=0, r=0, t=50, b=0), paper_bgcolor="#fcfcfb",
            font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2),
            scene=dict(xaxis_title="moneyness (strike / spot)", yaxis=dict(title="time to expiry", **ticks),
                       zaxis=dict(title="implied vol", tickformat=".0%"), camera=dict(eye=dict(x=1.7, y=-1.5, z=0.8))))
        return fig


    def _pooled_figure(h_edges, m_edges, title, log_time=True):
        _g = iv.pooled_surface(iv_strikes, h_edges, m_edges, min_obs=100, min_snaps=20)
        return surface_figure([i.mid for i in _g.columns], [i.mid for i in _g.index], _g.to_numpy(), title, log_time)


    surface_3d = _pooled_figure([1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 72, 96, 120, 144, 168.5], MONEYNESS_EDGES,
                                "Implied-vol surface, full week")
    surface_3d_1d7d = _pooled_figure([24, 30, 36, 42, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168.5], MONEYNESS_EDGES,
                                     "Implied-vol surface, 1 to 7 days to expiry", log_time=False)
    surface_3d_1d = _pooled_figure([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 24], np.arange(0.95, 1.0501, 0.005),
                                   "Implied-vol surface, last 24 hours")
    mo.vstack([
        mo.md("""
        ## 3D surfaces

        Drag to rotate, scroll to zoom, hover for values. Each surface is the **typical smile shape** (each strike's
        vol divided by its own snapshot's ATM vol, median per cell) times the **typical ATM vol** at that time to
        expiry (method step 8); cells need at least 100 strike-snapshots from 20 snapshots. Three views: the **full
        week** (time on a log axis so the last hours are visible), **1 to 7 days** (linear time axis), and the **last
        24 hours** (finer time bins, moneyness 0.95–1.05; the final hours have too few clean snapshots to show).
        """),
        surface_3d,
        surface_3d_1d7d,
        surface_3d_1d,
    ])
    return MONEYNESS_EDGES, surface_3d, surface_3d_1d, surface_3d_1d7d, surface_figure


@app.cell
def _(MONEYNESS_EDGES, iv_strikes, np, pd, surface_figure, week_select):
    # The selected weekly's own surface over its life (follows the weekly dropdown above)
    _w = iv_strikes[(iv_strikes.event_ticker == week_select.value) & iv_strikes.iv_surface.notna()]
    _g = (_w.assign(h=pd.cut(_w.hours_to_expiry, np.arange(0, 172, 2)), z=pd.cut(_w.moneyness, MONEYNESS_EDGES))
          .groupby(["h", "z"], observed=False).iv_surface.median().unstack("z"))
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
    surface_3d_1d,
    surface_3d_1d7d,
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
        surface_3d_1d7d.write_html(OUTPUTS / "iv_surface_3d_1d7d.html", include_plotlyjs="cdn")
        figure_files.append("iv_surface_3d_1d7d.html")
        surface_3d_1d.write_html(OUTPUTS / "iv_surface_3d_1d.html", include_plotlyjs="cdn")
        figure_files.append("iv_surface_3d_1d.html")
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
    | `iv_surface` | `iv` as used in the surfaces: clean snapshots, binary price 5–95¢, winsorized 1st/99th percentile per time-to-expiry bucket (else empty) |
    | `digital_iv` | Direct binary implied vol (comparison only; unreliable near the money) |
    | `forward`, `atm_iv`, `clean` | Snapshot's implied forward, ATM vol and clean flag |
    | `moneyness` | strike / exchange spot (the axis used in the charts) |
    | `log_moneyness`, `sd_from_forward` | ln(K / implied forward), and the same in standard deviations (ATM vol · √τ) |

    **`data/derived/iv_snapshots.parquet`**: one row per snapshot ({len(iv_snapshots):,} rows): `spot`, `forward`,
    `forward_basis_bps`, `n_strikes`, `n_empty_iv`, `monotone_fixes`, `prob_outside` (probability beyond the outermost strikes, priced by the tails),
    `sigma_low` / `sigma_high` (tail vols), `tails_clipped` (0–2 tails cut to Kalshi's outer-strike bounds),
    `atm_iv`, `strikes_per_sd`, `clean` (ATM vol exists, ≥ 2 strikes per standard deviation, |basis| ≤ 25 bps).

    **`outputs/`** (committed to git): `figures/iv_surface_heatmap.png`, `atm_term_structure.png`,
    `atm_vs_realized.png`, `forward_basis.png`, `clean_by_ttm.png`, three example smiles for the latest weekly
    (`smile_<weekly>_72h.png`, `_24h`, `_6h`), and the rotatable 3D surfaces `iv_surface_3d.html` (full week),
    `iv_surface_3d_1d7d.html` (1 to 7 days) and `iv_surface_3d_1d.html` (last 24 hours); open them in a browser (they load the plotting library from the web).

    **Written this run:** {"yes" if WRITE else "no (press the button at the top, or run `python 03_iv_surface.py`)"}.
    """)
    return


if __name__ == "__main__":
    app.run()
