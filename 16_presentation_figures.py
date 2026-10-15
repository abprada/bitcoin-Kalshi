import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="16 Presentation figures")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import plotly.graph_objects as go

    from lib import deribit, iv, pairs

    return alt, deribit, go, iv, mo, np, pairs, pd


@app.cell
def _(mo):
    mo.md(r"""
    # 16 · Presentation figures: Kalshi vs Deribit

    The figures for the Kalshi vs Deribit slides, in slide order, each with the numbers to quote. Everything comes
    from the same **pairs** as `05`: every Deribit trade in a Friday expiry, matched with Kalshi's weekly ladder at
    the same strike and minute (`lib/pairs.py`). The pairs are built once and cached in `data/derived/pairs*.parquet`
    (about 15 minutes); delete those files to rebuild after new data.

    Run as a script (`python 16_presentation_figures.py`) it saves every chart to `outputs/presentation/`: a PNG at
    2× for the slides and, for the surfaces, HTML files that rotate in a browser. Opening it writes nothing.
    """)
    return


@app.cell
def _(alt):
    # Reference palette (as in 04/05): Kalshi = slot 1 blue, Deribit = slot 2 orange; spreads in ink; gray chrome.
    INK, INK2, MUTED, CONTEXT, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e1e0d9", "#fcfcfb"
    KALSHI, DERIBIT = "#2a78d6", "#eb6834"
    W, H = 760, 340
    MARKETS = alt.Scale(domain=["Kalshi", "Deribit (stretched)", "Deribit (as traded)"], range=[KALSHI, DERIBIT, DERIBIT])


    @alt.theme.register("slides", enable=True)
    def _slides() -> alt.theme.ThemeConfig:
        return {
            "config": {
                "background": SURFACE,
                "font": "system-ui, -apple-system, 'Segoe UI', sans-serif",
                "view": {"stroke": None},
                "axis": {"gridColor": GRID, "domainColor": CONTEXT, "tickColor": CONTEXT, "labelColor": INK2,
                         "titleColor": INK2, "titleFontWeight": "normal", "labelFontSize": 13, "titleFontSize": 14},
                "legend": {"labelColor": INK2, "titleColor": INK2, "orient": "top", "labelFontSize": 13,
                           "symbolStrokeWidth": 3},
                "axisX": {"labelAngle": 0},
                "title": {"color": INK, "anchor": "start", "fontWeight": 600, "fontSize": 18,
                          "subtitleColor": INK2, "subtitleFontSize": 13, "subtitlePadding": 6, "offset": 12},
                "line": {"strokeWidth": 2.5},
                "point": {"size": 70, "filled": True},
            }
        }

    return CONTEXT, DERIBIT, H, INK, INK2, KALSHI, MARKETS, MUTED, W


@app.cell
def _(mo, pairs, pd):
    ROOT = mo.notebook_dir()
    _r = pairs.cached_pairs(ROOT)
    paired, hour_profile, offsets = _r["pairs"], _r["hour_profile"], _r["trade_offsets"]
    atm = pairs.atm(paired)
    _ts = pd.read_parquet(ROOT / "data" / "clean" / "btc_spot_1m.parquet", columns=["ts"]).ts   # hour_profile's source
    SPOT_SPAN = f"{_ts.min():%b %Y} – {_ts.max():%b %Y}"
    mo.md(f"""
    **Data:** {len(paired):,} pairs on {paired.day.nunique()} days in {paired.event_ticker.nunique()} Kalshi weeklies
    ({paired.day.min():%b %Y} – {paired.day.max():%b %Y}), {len(atm):,} of them at the money (within 0.25 standard
    deviations), built from {len(offsets):,} Deribit trades.
    """)
    return ROOT, SPOT_SPAN, atm, hour_profile, offsets, paired


@app.cell
def _(CONTEXT, H, INK2, KALSHI, MUTED, SPOT_SPAN, W, alt, hour_profile, mo, np, pd):
    # Slide 1: BTC's typical vol in every hour of the week, with the hours only Kalshi's contract covers
    _days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    _v = pd.DataFrame({"how": range(168), "vol": np.sqrt(hour_profile.to_numpy() * 24 * 365)})
    _v["mid"] = _v.how + 0.5                                  # each bar fills its hour, [h, h + 1)
    _v["label"] = [f"{_days[h // 24]} {h % 24:02d}:00–{h % 24 + 1:02d}:00 UTC" for h in _v.how]
    _fri = 4 * 24
    _v["period"] = np.where(_v.how.between(_fri + 8, _fri + 20), "Only in Kalshi's contract: Fri 08:00 UTC to 5pm ET",
                            "Rest of the week")
    WEEK_AVG = float(np.sqrt(hour_profile.mean() * 24 * 365))
    _extra = float(np.sqrt(hour_profile.iloc[_fri + 8:_fri + 21].mean() * 24 * 365))
    _wkend = float(np.sqrt(hour_profile.iloc[5 * 24:].mean() * 24 * 365))
    _x = alt.X("mid:Q", title=None, scale=alt.Scale(domain=[0, 168], nice=False),
               axis=alt.Axis(values=list(range(0, 168, 24)), labelExpr=f"{_days}[datum.value / 24] + ' 00:00'", grid=False))
    _marks = pd.DataFrame({"mid": [_fri + 8, _fri + 21], "text": ["Deribit expires\nFri 08:00 UTC", "Kalshi expires\nFri 5pm ET"],
                           "side": ["right", "left"]})
    week_vol_chart = (
        alt.Chart(_v).mark_bar(width=3.2, cornerRadiusTopLeft=1.5, cornerRadiusTopRight=1.5).encode(
            x=_x, y=alt.Y("vol:Q", title="BTC realized vol in that hour (annualized)", axis=alt.Axis(format="%")),
            color=alt.Color("period:N", title=None, legend=alt.Legend(labelLimit=0), scale=alt.Scale(
                domain=["Rest of the week", "Only in Kalshi's contract: Fri 08:00 UTC to 5pm ET"], range=[CONTEXT, KALSHI])),
            tooltip=[alt.Tooltip("label:N", title="hour"), alt.Tooltip("vol:Q", title="realized vol", format=".1%")])
        + alt.Chart(pd.DataFrame({"y": [WEEK_AVG]})).mark_rule(color=MUTED, strokeDash=[5, 4], strokeWidth=1.5).encode(y="y:Q")
        + alt.Chart(pd.DataFrame({"y": [WEEK_AVG], "t": [f"week average {WEEK_AVG:.0%}"]})).mark_text(
            align="left", dx=4, dy=-8, color=INK2, fontSize=12).encode(x=alt.value(0), y="y:Q", text="t:N")
        + alt.Chart(_marks).mark_rule(color=INK2, strokeWidth=1.2).encode(x="mid:Q")
        + alt.Chart(_marks[_marks.side == "right"]).mark_text(align="right", dx=-5, color=INK2, fontSize=12, lineBreak="\n").encode(
            x="mid:Q", y=alt.value(14), text="text:N")
        + alt.Chart(_marks[_marks.side == "left"]).mark_text(align="left", dx=5, color=INK2, fontSize=12, lineBreak="\n").encode(
            x="mid:Q", y=alt.value(14), text="text:N")
    ).properties(width=W, height=H, title=alt.TitleParams(
        "Kalshi's contract runs 13 hours longer, through some of the busiest hours of the week",
        subtitle=f"Average over {SPOT_SPAN}, 1-minute Coinbase/Bitstamp prices. "
                 f"The extra hours: {_extra:.0%}; weekend: {_wkend:.0%}; week average: {WEEK_AVG:.0%}."))
    mo.vstack([
        mo.md(f"""
        ## Slide 1 · Different expiries

        Both contracts are named for the same Friday, but Deribit's options expire at **08:00 UTC** and Kalshi's at
        **5pm ET** (21:00 UTC in summer, 22:00 in winter). Each bar is how much BTC moves, on average, in that hour of
        the week, quoted as an annual vol. Kalshi's extra Friday hours (blue) run at about **{_extra:.0%}**, against a
        week average of {WEEK_AVG:.0%} and {_wkend:.0%} at weekends, so the two prices are not for the same risk.
        """),
        week_vol_chart,
    ])
    return WEEK_AVG, week_vol_chart


@app.cell
def _(DERIBIT, H, INK2, W, alt, mo, paired, pd):
    # Slide 2: the stretch factor that puts Deribit's vol on Kalshi's horizon, by time to Kalshi's expiry
    _p = paired.assign(days=paired.kalshi_hours / 24)
    _p = _p[_p.days.between(0.5, 7)]
    _b = (_p.groupby(pd.cut(_p.days, [x / 4 for x in range(2, 29)]), observed=True).horizon_ratio
          .describe(percentiles=[.1, .5, .9])[["10%", "50%", "90%", "count"]].reset_index())
    _b["days"] = _b.days.map(lambda i: i.mid).astype(float)
    _b = _b[_b["count"] >= 200]
    _x = alt.X("days:Q", title="days to Kalshi's expiry", scale=alt.Scale(domain=[0.5, 7], nice=False))
    stretch_chart = (
        alt.Chart(_b).mark_area(color=DERIBIT, opacity=0.18).encode(x=_x, y="10%:Q", y2="90%:Q")
        + alt.Chart(_b).mark_line(color=DERIBIT, point=alt.OverlayMarkDef(size=50, color=DERIBIT)).encode(
            x=_x, y=alt.Y("50%:Q", title="Deribit vol is multiplied by", scale=alt.Scale(zero=False),
                          axis=alt.Axis(format=".2f", labelExpr="'×' + format(datum.value, '.2f')")),
            tooltip=[alt.Tooltip("days:Q", title="days left", format=".2f"), alt.Tooltip("50%:Q", title="median factor", format=".3f"),
                     alt.Tooltip("10%:Q", format=".3f"), alt.Tooltip("90%:Q", format=".3f"), alt.Tooltip("count:Q", title="pairs", format=",")])
    ).properties(width=W, height=H, title=alt.TitleParams(
        "Stretching Deribit's vol to Kalshi's expiry matters most near expiry",
        subtitle="Median factor per quarter-day (shaded: 10th–90th percentile of the pairs). "
                 "Variance adds up hour by hour, each hour weighted by the profile on slide 1."))
    _week = _b[_b.days >= 6]["50%"].median()
    _day = _b[_b.days <= 1.5]["50%"].median()
    mo.vstack([
        mo.md(f"""
        ## Slide 2 · How Deribit's vol is stretched

        Deribit has no option that expires when Kalshi's does, so its vol is moved to Kalshi's horizon:

        1. Deribit's vol σ over its own life gives its expected **variance to Friday 08:00 UTC**: σ² × T_Deribit.
        2. Add the variance BTC usually has in the **13 extra hours**, in proportion to slide 1's profile (how busy
           those hours are relative to the hours already in Deribit's life).
        3. Convert the total back to an annual vol over **Kalshi's** life: σ_stretched = σ × √(expected variance to
           5pm ET ÷ expected variance to 08:00 UTC) × √(T_Deribit ÷ T_Kalshi).

        The factor is about **×{_week:.2f}** a week out, where 13 hours are a small share of the life, and
        **×{_day:.2f}** with about a day left. The results that follow are shown on Kalshi's clock with Deribit
        stretched (`lib/deribit.py`, `horizon_vol_ratios`).
        """),
        stretch_chart,
    ])
    return (stretch_chart,)


@app.cell
def _(DERIBIT, H, INK2, W, alt, mo, np, offsets, pd):
    # Slide 3: Deribit is trades, Kalshi is a quote every minute: how far each trade is from the quote it is read against
    _edges = np.arange(-30, 31, 2)
    _h = pd.cut(offsets, _edges, include_lowest=True).value_counts(normalize=True, sort=False)
    _d = pd.DataFrame({"lo": [i.left for i in _h.index], "hi": [i.right for i in _h.index], "share": _h.to_numpy()})
    _d["lo"] = _d.lo.clip(lower=-30)
    _abs = offsets.abs()
    offset_chart = (
        alt.Chart(_d).mark_rect(color=DERIBIT, cornerRadiusTopLeft=3, cornerRadiusTopRight=3, stroke="#fcfcfb", strokeWidth=2).encode(
            x=alt.X("lo:Q", title="Deribit trade time − Kalshi quote time (seconds; negative = trade first)",
                    scale=alt.Scale(domain=[-30, 30], nice=False), axis=alt.Axis(values=list(range(-30, 31, 10)))),
            x2="hi:Q", y=alt.Y("share:Q", title="share of Deribit trades (2-second bins)", axis=alt.Axis(format=".1%"),
                               scale=alt.Scale(zero=True)), y2=alt.datum(0),
            tooltip=[alt.Tooltip("lo:Q", title="from (s)"), alt.Tooltip("hi:Q", title="to (s)"),
                     alt.Tooltip("share:Q", title="share", format=".1%")])
        + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK2, strokeDash=[4, 3]).encode(x="x:Q")
    ).properties(width=W, height=H, title=alt.TitleParams(
        "Every Deribit trade is compared with Kalshi's quote at most 30 seconds away",
        subtitle=f"{len(offsets):,} Deribit trades; median distance {_abs.median():.0f} s, "
                 f"{(_abs <= 10).mean():.0%} within 10 s. Kalshi's quote is its bid/ask at the nearest minute end."))
    mo.vstack([
        mo.md(f"""
        ## Slide 3 · Trades vs continuous quotes

        **Kalshi** posts a bid and ask on every strike, every minute, so its smile exists at any moment. **Deribit**
        data are individual **trades**: a strike is only observed when someone trades it. So each Deribit trade sets
        the moment and Kalshi is read at the nearest minute end. By construction no trade is more than 30 seconds
        from its Kalshi quote; the median is **{_abs.median():.0f} seconds**. Trades are spread across the minute,
        with a peak in the first few seconds after the minute mark. Deribit's vol is taken from the trade's **mark price** (Deribit's fair value
        at that instant), not the traded price, which sits on the bid or the ask.
        """),
        offset_chart,
    ])
    return (offset_chart,)


@app.cell
def _(ROOT, deribit, hour_profile, pd):
    # Every hour, not only the paired moments: each clean hourly Kalshi snapshot (03, every weekly) against Deribit's
    # fitted ATM vol for the Friday expiry with the same date at the same hour (08's hourly smile grid), stretched to
    # Kalshi's expiry exactly as in the pairs. Each week-hour counts once.
    _s = pd.read_parquet(ROOT / "data" / "derived" / "iv_snapshots.parquet")
    _k = _s[_s.clean & (_s.ts.dt.minute == 0)].copy()
    _k["kalshi_expiry"] = (_k.ts + pd.to_timedelta(_k.hours_to_expiry, unit="h")).dt.round("h")
    _k["deribit_expiry"] = _k.kalshi_expiry.dt.normalize() + pd.Timedelta(hours=8)
    _g = pd.read_parquet(ROOT / "data" / "derived" / "strategies" / "deribit_grid_centred.parquet",
                         columns=["expiry", "t", "atm_iv"])
    hourly = _k.merge(_g.rename(columns={"expiry": "deribit_expiry", "t": "ts", "atm_iv": "deribit_atm"}),
                      on=["deribit_expiry", "ts"])
    hourly["deribit_adj"] = hourly.deribit_atm * deribit.horizon_vol_ratios(
        hourly.ts, hourly.kalshi_expiry, hourly.deribit_expiry, hour_profile)
    hourly["gap_adj"] = 100 * (hourly.atm_iv - hourly.deribit_adj)
    hourly["days"] = hourly.hours_to_expiry / 24
    n_hourly_kalshi = len(_k)
    return hourly, n_hourly_kalshi


@app.cell
def _(DERIBIT, H, INK, KALSHI, MUTED, W, alt, atm, hourly, mo, n_hourly_kalshi, pairs, pd):
    # Slide 4: ATM vol by time to Kalshi's expiry, Kalshi vs Deribit stretched, and their spread.
    # Main version: every hour (smooth, each week-hour counts once). Backup: the paired trades, which weight each
    # moment by how many Deribit trades it had, so busy (volatile) days dominate and the levels jump between bins.
    DAY_EDGES = [x / 2 for x in range(1, 15)]                   # half-day bins, 0.5 to 7 days
    _scale = alt.Scale(domain=["Kalshi", "Deribit (stretched)"], range=[KALSHI, DERIBIT])
    _x = alt.X("days:Q", title="days to Kalshi's expiry", scale=alt.Scale(domain=[0.5, 7], nice=False))


    def term_charts(d, kalshi, deribit_, unit, min_n, n_label, title, subtitle):
        """Levels (median per half-day bin) and spread (median, 95% interval resampling whole `unit`s)."""
        d = d.assign(bin=pd.cut(d.days, DAY_EDGES), day=d[unit]).dropna(subset=["bin"])
        lv = (d.groupby("bin", observed=True)
              .agg(**{"Kalshi": (kalshi, "median"), "Deribit (stretched)": (deribit_, "median"), "n": (kalshi, "size")})
              .reset_index())
        lv = lv[lv.n >= min_n]
        lv["days"] = lv.bin.map(lambda i: i.mid).astype(float)
        long = lv.drop(columns="bin").melt(id_vars=["days", "n"], var_name="market", value_name="vol")
        levels = alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=55)).encode(
            x=_x, y=alt.Y("vol:Q", title="implied vol at the money (median)", axis=alt.Axis(format="%"),
                          scale=alt.Scale(zero=False)),
            color=alt.Color("market:N", title=None, scale=_scale),
            tooltip=["market:N", alt.Tooltip("days:Q", title="days left"), alt.Tooltip("vol:Q", format=".1%"),
                     alt.Tooltip("n:Q", title=n_label, format=",")]
        ).properties(width=W, height=H, title=alt.TitleParams(title, subtitle=subtitle))
        g = pairs.gap_table(d, "bin", col="gap_adj").rename(columns={"days": "n_units"})
        g["days"] = g.bin.map(lambda i: i.mid).astype(float)
        g = g[g.pairs >= min_n].drop(columns="bin")          # intervals cannot go into a chart spec
        spread = (
            alt.Chart(g).mark_area(color=MUTED, opacity=0.25).encode(x=_x, y="ci_low:Q", y2="ci_high:Q")
            + alt.Chart(g).mark_line(color=INK, point=alt.OverlayMarkDef(size=55, color=INK)).encode(
                x=_x, y=alt.Y("median_gap:Q", title="Kalshi − Deribit (vol points)"),
                tooltip=[alt.Tooltip("days:Q", title="days left"), alt.Tooltip("median_gap:Q", title="spread", format="+.2f"),
                         alt.Tooltip("ci_low:Q", title="95% low", format="+.2f"),
                         alt.Tooltip("ci_high:Q", title="95% high", format="+.2f"), alt.Tooltip("pairs:Q", title=n_label, format=",")])
            + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
        ).properties(width=W, height=H - 60, title=alt.TitleParams(
            "The ATM spread: Kalshi − Deribit (stretched)",
            subtitle=f"Median per half-day bin; shaded: 95% interval resampling whole {'weeks' if unit == 'event_ticker' else 'days'}."))
        overall = pairs.gap_table(d.assign(all=1), "all", col="gap_adj").iloc[0]
        return levels, spread, g, overall


    atm_term_chart, atm_spread_chart, _g, _all = term_charts(
        hourly, "atm_iv", "deribit_adj", "event_ticker", 100, "hours",
        "At the money, Kalshi prices more volatility than Deribit, most of all near expiry",
        f"Every hour: Kalshi's ATM vol vs Deribit's fitted ATM vol for the same Friday, stretched to Kalshi's expiry; "
        f"{len(hourly):,} hours in {hourly.event_ticker.nunique()} weeks.")
    atm_term_pairs_chart, atm_spread_pairs_chart, _gp, _allp = term_charts(
        atm.assign(days=atm.kalshi_hours / 24), "kalshi_iv", "deribit_iv_adj", "day", 200, "pairs",
        "Backup: the same comparison on the paired trades only",
        "At-the-money pairs (within 0.25 SD); each Deribit trade counts once, so busy days weigh more.")
    _far, _near = _g[_g.days >= 5].median_gap.median(), _g[_g.days <= 1.5].median_gap.median()
    mo.vstack([
        mo.md(f"""
        ## Slide 4 · ATM implied vol by time to maturity

        **Every hour of every week**, not just the moments when Deribit traded: Kalshi's ATM vol from each clean hourly
        snapshot ({len(hourly):,} of {n_hourly_kalshi:,} have a Deribit Friday expiry still live), against Deribit's
        fitted ATM vol for the same Friday at the same hour, stretched to Kalshi's expiry. Overall Kalshi is
        **{_all.median_gap:+.1f} points** above Deribit (95% interval {_all.ci_low:+.1f} to {_all.ci_high:+.1f},
        resampling whole weeks), higher in {_all.kalshi_higher:.0%} of hours. The spread is about **{_near:+.1f}**
        points with about a day left and **{_far:+.1f}** with 5–7 days left.
        """),
        atm_term_chart,
        atm_spread_chart,
        mo.md(f"""
        **Why the paired version (below) is noisier.** A pair exists only when someone trades a near-the-money Deribit
        strike, and every trade counts once, so busy days, which are the volatile days, dominate whichever bins they
        fall in and the *levels* jump from bin to bin. The *spread* is far steadier because each pair compares the two
        markets at the same minute. It tells the same story: {_allp.median_gap:+.1f} points overall at the money.
        """),
        atm_term_pairs_chart,
        atm_spread_pairs_chart,
    ])
    return atm_spread_chart, atm_spread_pairs_chart, atm_term_chart, atm_term_pairs_chart


@app.cell
def _(H, INK, MARKETS, MUTED, W, alt, mo, paired, pairs, pd):
    # Slide 5: the smile, implied vol against strike in standard deviations, both markets, and their spread
    _keep = pairs.gap_table(paired, "sd_bucket", col="gap_adj")
    _keep = _keep[_keep.pairs >= 100]
    _sds = _keep.sd_bucket.astype(float)
    # Levels as shapes (each vol over Kalshi's ATM vol at that minute, median per bucket), scaled by the median
    # Kalshi ATM vol, so calm and volatile weeks pool without distorting the wings (as in 05 and 03)
    _atm_level = float(pairs.atm(paired).kalshi_atm.median())
    _s = (paired.assign(Kalshi=paired.kalshi_iv / paired.kalshi_atm,
                        **{"Deribit (stretched)": paired.deribit_iv_adj / paired.kalshi_atm})
          .groupby("sd_bucket", observed=True)[["Kalshi", "Deribit (stretched)"]].median() * _atm_level)
    _s = _s.reset_index().assign(sd=lambda d: d.sd_bucket.astype(float))
    _s = _s[_s.sd.isin(_sds)].drop(columns="sd_bucket").melt(id_vars="sd", var_name="market", value_name="vol")
    _x = alt.X("sd:Q", title="strike: standard deviations from the money (negative = below spot)",
               scale=alt.Scale(domain=[-3, 3], nice=False), axis=alt.Axis(values=list(range(-3, 4))))
    smile_chart = (
        alt.Chart(_s).mark_line(point=alt.OverlayMarkDef(size=55)).encode(
            x=_x, y=alt.Y("vol:Q", title="implied vol", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
            color=alt.Color("market:N", title=None, scale=alt.Scale(domain=["Kalshi", "Deribit (stretched)"],
                                                                    range=[MARKETS.range[0], MARKETS.range[1]])),
            tooltip=["market:N", alt.Tooltip("sd:Q", title="SD from the money"), alt.Tooltip("vol:Q", format=".1%")])
    ).properties(width=W, height=H, title=alt.TitleParams(
        "Deribit prices crashes much higher; Kalshi is richer around the money",
        subtitle=f"Typical smile shape of each market (vol ÷ Kalshi's ATM vol at that minute, median per half-SD bucket) "
                 f"× the median ATM vol, {_atm_level:.0%}. Deribit stretched to Kalshi's expiry."))
    _keep["sd"] = _sds
    smile_spread_chart = (
        alt.Chart(_keep).mark_area(color=MUTED, opacity=0.25).encode(x=_x, y="ci_low:Q", y2="ci_high:Q")
        + alt.Chart(_keep).mark_line(color=INK, point=alt.OverlayMarkDef(size=55, color=INK)).encode(
            x=_x, y=alt.Y("median_gap:Q", title="Kalshi − Deribit (vol points)"),
            tooltip=[alt.Tooltip("sd:Q", title="SD"), alt.Tooltip("median_gap:Q", title="spread", format="+.2f"),
                     alt.Tooltip("ci_low:Q", title="95% low", format="+.2f"), alt.Tooltip("ci_high:Q", title="95% high", format="+.2f"),
                     alt.Tooltip("pairs:Q", format=","), alt.Tooltip("days:Q", title="days with pairs")])
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    ).properties(width=W, height=H - 60, title=alt.TitleParams(
        "The spread across strikes: Kalshi − Deribit (stretched)",
        subtitle="Median per half-SD bucket; shaded: 95% interval resampling whole days; buckets with 100+ pairs."))
    _k = _keep.set_index("sd")
    _c = _k.loc[0.0] if 0.0 in _k.index else None
    mo.vstack([
        mo.md(f"""
        ## Slide 5 · Implied vol against strike

        The horizontal axis is the strike's distance from the money in standard deviations of BTC's move to expiry,
        ln(K / F) ÷ (Kalshi ATM vol × √time left), where F is Kalshi's forward, within a few hundredths of a percent of
        spot over a week. Kalshi is **{_c.median_gap:+.1f}** points above Deribit at the money but
        **{_k.loc[-2.0].median_gap:+.1f}** two standard deviations below and **{_k.loc[2.0].median_gap:+.1f}** two above.
        Kalshi is richer around the money; Deribit is clearly richer on the downside (crash protection), while on the
        upside the two are within a couple of points and the far call wing rests on few pairs.
        """),
        smile_chart,
        smile_spread_chart,
    ])
    return smile_chart, smile_spread_chart


@app.cell
def _(INK2, W, alt, go, mo, paired, pairs):
    # Slide 6: the surfaces, on the same pairs and axes; rotatable 3D (HTML) plus flat heatmaps for a static slide
    surf = pairs.surface(paired)
    surf["days"] = ((surf.hours_lo + surf.hours_hi) / 48).round().astype(int)
    _sd = sorted(surf.sd.unique())
    _days = sorted(surf.days.unique())
    _grid = lambda c: surf.pivot(index="days", columns="sd", values=c).reindex(index=_days, columns=_sd).to_numpy()
    _blues, _oranges = [[0, "#cde2fb"], [1, "#0d366b"]], [[0, "#fde3d6"], [1, "#7a2a0e"]]
    _zmin, _zmax = float(surf[["kalshi", "deribit"]].min().min()), float(surf[["kalshi", "deribit"]].max().max())
    _font = dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=13)
    _vol_z = dict(title="implied vol", tickformat=".0%", range=[_zmin * 0.95, _zmax * 1.02])

    def _fig(traces, title, zaxis=_vol_z):
        f = go.Figure(traces)
        f.update_layout(title=dict(text=title, font=dict(size=18)), height=620, margin=dict(l=0, r=0, t=60, b=0),
                        paper_bgcolor="#fcfcfb", font=_font,
                        scene=dict(xaxis_title="standard deviations from the money", yaxis_title="days to Kalshi's expiry",
                                   zaxis=zaxis, camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8))))
        return f

    def _surface(col, name, scale, opacity=1.0, bar=True):
        return go.Surface(x=_sd, y=_days, z=_grid(col), name=name, colorscale=scale, cmin=_zmin, cmax=_zmax,
                          opacity=opacity, showscale=bar, colorbar=dict(title="IV", tickformat=".0%"),
                          hovertemplate=name + "<br>%{x:+.1f} SD<br>%{y} days left<br>vol %{z:.1%}<extra></extra>")

    surface_kalshi = _fig([_surface("kalshi", "Kalshi", _blues)], "Kalshi implied-vol surface")
    surface_deribit = _fig([_surface("deribit", "Deribit (stretched)", _oranges)], "Deribit implied-vol surface (stretched to Kalshi's expiry)")
    surface_both = _fig([_surface("kalshi", "Kalshi", _blues, 0.95, False),
                         _surface("deribit", "Deribit (stretched)", _oranges, 0.7, False)],
                        "Kalshi (blue) and Deribit (orange) on the same axes")
    _gmax = float(surf.gap.abs().quantile(0.95))
    surface_gap = _fig([go.Surface(x=_sd, y=_days, z=surf.pivot(index="days", columns="sd", values="gap")
                                   .reindex(index=_days, columns=_sd).to_numpy(),
                                   colorscale=[[0, "#c4501f"], [0.5, "#f0efec"], [1, "#1f5fae"]], cmin=-_gmax, cmax=_gmax,
                                   colorbar=dict(title="points"),
                                   hovertemplate="%{x:+.1f} SD<br>%{y} days left<br>Kalshi − Deribit %{z:+.1f} points<extra></extra>")],
                       "Spread surface: Kalshi − Deribit, vol points (blue: Kalshi higher; orange: Deribit higher)",
                       zaxis=dict(title="Kalshi − Deribit (points)", tickformat="+.0f"))

    # Flat versions for a static slide
    _x = alt.X("sd:O", title="standard deviations from the money", axis=alt.Axis(labelAngle=0))
    _y = alt.Y("days:O", title="days to Kalshi's expiry", sort="descending")
    _tip = [alt.Tooltip("days:O", title="days left"), alt.Tooltip("sd:Q", title="SD"),
            alt.Tooltip("kalshi:Q", format=".1%", title="Kalshi"), alt.Tooltip("deribit:Q", format=".1%", title="Deribit"),
            alt.Tooltip("gap:Q", format="+.1f", title="spread (points)"), alt.Tooltip("pairs:Q", format=",")]

    def _heat(col, title, rng):
        b = alt.Chart(surf).encode(x=_x, y=_y, tooltip=_tip)
        return (b.mark_rect(stroke="#fcfcfb", strokeWidth=1.5).encode(color=alt.Color(
                    f"{col}:Q", title="implied vol", legend=alt.Legend(format=".0%", orient="right"),
                    scale=alt.Scale(range=rng, domain=[_zmin, _zmax])))
                + b.mark_text(fontSize=11).encode(text=alt.Text(f"{col}:Q", format=".0%"), color=alt.condition(
                    f"datum.{col} > {(_zmin + _zmax) / 2:.3f}", alt.value("white"), alt.value("#0b0b0b")))
                ).properties(title=title, width=W // 2 - 40, height=260)

    surfaces_heat_chart = alt.hconcat(_heat("kalshi", "Kalshi", ["#cde2fb", "#0d366b"]),
                                      _heat("deribit", "Deribit (stretched)", ["#fde3d6", "#7a2a0e"])
                                      ).resolve_scale(color="independent").properties(title=alt.TitleParams(
                                          "Implied-vol surfaces on the same pairs",
                                          subtitle="Median per cell of each vol ÷ Kalshi's ATM vol, × the row's median ATM vol; cells with 40+ pairs."))
    _gb = alt.Chart(surf).encode(x=_x, y=_y, tooltip=_tip)
    gap_heat_chart = (
        _gb.mark_rect(stroke="#fcfcfb", strokeWidth=1.5).encode(color=alt.Color(
            "gap:Q", title="points", scale=alt.Scale(domain=[-_gmax, 0, _gmax], range=["#c4501f", "#f0efec", "#1f5fae"],
                                                     interpolate="rgb", clamp=True)))
        + _gb.mark_text(fontSize=11).encode(text=alt.Text("gap:Q", format="+.1f"), color=alt.condition(
            f"abs(datum.gap) > {_gmax * 0.6:.2f}", alt.value("white"), alt.value("#0b0b0b")))
    ).properties(width=W, height=260, title=alt.TitleParams(
        "Spread surface: Kalshi − Deribit (stretched), vol points",
        subtitle="Blue: Kalshi higher; orange: Deribit higher. Median of the pairs' own spreads per cell."))
    mo.vstack([
        mo.md("""
        ## Slide 6 · Implied-vol surfaces

        Built on exactly the same pairs, so the two surfaces are directly comparable: rows are days to Kalshi's expiry,
        columns half-SD strike buckets. Drag to rotate, scroll to zoom, hover for values. Saved as HTML files that open
        in any browser, for a live demo or a link from the slide; flat heatmaps below for a static slide.
        """),
        mo.hstack([surface_kalshi, surface_deribit], widths="equal"),
        surface_both,
        surface_gap,
        surfaces_heat_chart,
        gap_heat_chart,
    ])
    return (gap_heat_chart, surface_both, surface_deribit, surface_gap, surface_kalshi, surfaces_heat_chart)


@app.cell
def _(INK2, ROOT, W, alt, go, iv, mo, np, pd):
    # Slide 6 extra: Kalshi's surface from ALL its data (03: every clean 15-minute smile, every strike), not only the
    # strikes and moments where Deribit traded. Same construction as 03's surface (iv.pooled_surface).
    _k = pd.read_parquet(ROOT / "data" / "derived" / "iv_strikes.parquet",
                         columns=["snap", "strike", "spot", "atm_iv", "tau_years", "hours_to_expiry", "iv_surface"])
    _k = _k[_k.iv_surface.notna()]
    _k["sd_from_spot"] = (_k.strike / _k.spot - 1) / (_k.atm_iv * np.sqrt(_k.tau_years))   # as 03 defines it
    KALSHI_H_EDGES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 72, 96, 120, 144, 168.5]
    _sd_edges = np.arange(-3.0, 3.001, 0.25)
    _grid = iv.pooled_surface(_k, KALSHI_H_EDGES, _sd_edges, min_obs=100, min_snaps=20)
    _x, _y = [i.mid for i in _grid.columns], [i.mid for i in _grid.index]
    surface_kalshi_all = go.Figure(go.Surface(
        x=_x, y=_y, z=_grid.to_numpy(), colorscale=[[0, "#cde2fb"], [1, "#0d366b"]], cmin=0.35, cmax=0.9,
        colorbar=dict(title="IV", tickformat=".0%"),
        hovertemplate="%{x:+.2f} SD from spot<br>%{y:.1f} hours to expiry<br>implied vol %{z:.1%}<extra></extra>"))
    surface_kalshi_all.update_layout(
        title=dict(text="Kalshi implied-vol surface, all data: 1 hour to 7 days", font=dict(size=18)), height=620,
        margin=dict(l=0, r=0, t=60, b=0), paper_bgcolor="#fcfcfb",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=13),
        scene=dict(xaxis_title="standard deviations from spot",
                   yaxis=dict(title="time to expiry", type="log", tickvals=[1, 2, 3, 6, 12, 24, 48, 72, 120, 168],
                              ticktext=["1h", "2h", "3h", "6h", "12h", "1d", "2d", "3d", "5d", "7d"]),
                   zaxis=dict(title="implied vol", tickformat=".0%"), camera=dict(eye=dict(x=1.7, y=-1.5, z=0.8))))

    _span = lambda h: f"{h:g}h" if h < 24 else f"{h / 24:g}d"
    _cells = _grid.stack().rename("vol").dropna().reset_index()
    _cells["sd"] = _cells.m.map(lambda i: round(i.mid, 3)).astype(float)
    _cells["tau"] = _cells.h.map(lambda i: f"{_span(round(i.left))}–{_span(round(i.right))}")
    _order = [f"{_span(round(a))}–{_span(round(b))}" for a, b in zip(KALSHI_H_EDGES[:-1], KALSHI_H_EDGES[1:])][::-1]
    _cells = _cells.drop(columns=["m", "h"]).assign(sd_lo=lambda d: d.sd - 0.125, sd_hi=lambda d: d.sd + 0.125)
    kalshi_all_heat_chart = (
        alt.Chart(_cells).mark_rect(stroke="#fcfcfb", strokeWidth=1).encode(
            x=alt.X("sd_lo:Q", title="standard deviations from spot", scale=alt.Scale(domain=[-3, 3], nice=False),
                    axis=alt.Axis(values=[x / 2 for x in range(-6, 7)])),
            x2="sd_hi:Q", y=alt.Y("tau:O", title="time to Kalshi's expiry", sort=_order),
            color=alt.Color("vol:Q", title="implied vol", legend=alt.Legend(format=".0%", orient="right"),
                            scale=alt.Scale(range=["#cde2fb", "#0d366b"], domain=[0.35, 0.9], clamp=True)),
            tooltip=[alt.Tooltip("tau:O", title="time to expiry"), alt.Tooltip("sd:Q", title="SD from spot"),
                     alt.Tooltip("vol:Q", title="implied vol", format=".1%")])
    ).properties(width=W, height=380, title=alt.TitleParams(
        "Kalshi's surface from all of its data: vol rises toward expiry and in both wings",
        subtitle=f"Every clean 15-minute smile ({_k.snap.nunique():,} snapshots, {len(_k):,} strike vols); typical smile "
                 f"shape × typical ATM vol per row, as in 03; colour capped at 35–90%."))
    _atm = _grid.loc[:, [c for c in _grid.columns if c.left == 0.0 or c.right == 0.0]].mean(axis=1).dropna()
    _first = _atm.index[0]                                   # the shortest time bin with enough data
    mo.vstack([
        mo.md(f"""
        ### Slide 6 extra · Kalshi's surface from all of its data

        Slide 6's surfaces use only the pairs, so Kalshi is read just where Deribit traded. Here Kalshi uses
        **everything**: every clean smile, one every 15 minutes, every strike, {_k.snap.nunique():,} snapshots in all
        (`03_iv_surface.py`). Time is on a log axis so the final day, where Kalshi's vol rises sharply, is visible:
        about **{_atm.iloc[0]:.0%}** at the money with {_first.left:g}–{_first.right:g} hours left, against
        **{_atm.iloc[-1]:.0%}** a week out. Blank cells are where Kalshi's prices are too coarse to give a vol (as in
        `03`, only contracts priced 5–95¢ are used): the last few hours, and strikes beyond about ±2 standard deviations.
        This surface is not on Deribit's terms; for a side-by-side, use slide 6.
        """),
        surface_kalshi_all,
        kalshi_all_heat_chart,
    ])
    return kalshi_all_heat_chart, surface_kalshi_all


@app.cell
def _(
    ROOT,
    atm_spread_chart,
    atm_spread_pairs_chart,
    atm_term_chart,
    atm_term_pairs_chart,
    gap_heat_chart,
    kalshi_all_heat_chart,
    mo,
    offset_chart,
    smile_chart,
    smile_spread_chart,
    stretch_chart,
    surface_both,
    surface_deribit,
    surface_gap,
    surface_kalshi,
    surface_kalshi_all,
    surfaces_heat_chart,
    week_vol_chart,
):
    # Save everything when run as a script (python 16_presentation_figures.py); opening the notebook writes nothing
    if not mo.running_in_notebook():
        _out = ROOT / "outputs" / "presentation"
        _out.mkdir(parents=True, exist_ok=True)
        for _name, _chart in [("1_week_vol_profile", week_vol_chart), ("2_stretch_factor", stretch_chart),
                              ("3_trade_to_quote_distance", offset_chart), ("4a_atm_vol_by_maturity", atm_term_chart),
                              ("4b_atm_spread_by_maturity", atm_spread_chart),
                              ("4c_backup_atm_vol_pairs_only", atm_term_pairs_chart),
                              ("4d_backup_atm_spread_pairs_only", atm_spread_pairs_chart), ("5a_smile", smile_chart),
                              ("5b_smile_spread", smile_spread_chart), ("6a_surfaces_heatmap", surfaces_heat_chart),
                              ("6b_spread_surface_heatmap", gap_heat_chart),
                              ("6g_kalshi_all_data_surface_heatmap", kalshi_all_heat_chart)]:
            _chart.save(_out / f"{_name}.png", scale_factor=2)
            _chart.save(_out / f"{_name}.html")
        for _name, _fig in [("6c_surface_kalshi_3d", surface_kalshi), ("6d_surface_deribit_3d", surface_deribit),
                            ("6e_surfaces_both_3d", surface_both), ("6f_spread_surface_3d", surface_gap),
                            ("6h_kalshi_all_data_surface_3d", surface_kalshi_all)]:
            _fig.write_html(_out / f"{_name}.html", include_plotlyjs=True)   # self-contained: works offline
    return


if __name__ == "__main__":
    app.run()
