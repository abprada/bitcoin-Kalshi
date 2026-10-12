import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="05 Kalshi vs Deribit")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import plotly.graph_objects as go

    from lib import compare, deribit, pairs

    return alt, compare, deribit, go, mo, np, pairs, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 05 · Kalshi vs Deribit

    Handout Q2b: **are Kalshi's prices consistent with option prices?** Kalshi sells "$1 if BTC ends above K";
    Deribit sells calls and puts. Both can be turned into **implied volatility**, the market's expected size of
    Bitcoin's moves, and that is the comparison here.

    Kalshi has a full ladder of quotes every minute; Deribit only has trades, at scattered strikes and times. So every
    Deribit trade is the anchor: its vol from Deribit's mark price, against Kalshi's vol **at the same strike, in the
    same minute** (at most 30 seconds apart). No Deribit smile is built and nothing is interpolated on the Deribit side.

    **The expiry mismatch.** Deribit's Friday option expires at 08:00 UTC, Kalshi's at 5pm ET, about 13 hours later.
    Deribit's vol is stretched to Kalshi's horizon with the hour-of-week variance profile (section 3), and the whole
    comparison uses that stretch. Because it is a model, results are also shown without it, and **section 5 compares
    the two markets far from expiry** (4–7 days left), where the 13 hours are a small share of the horizon and the
    conclusion hardly depends on the stretch.

    **The appendix at the end defines every number exactly**: how a standard deviation from the money is computed,
    what "at the money" means, the buckets, the medians and the 95% intervals, and the 13-hour factor.

    The code is in `lib/pairs.py` and `lib/deribit.py`, tested in `tests/`.
    Needs `02_clean_data.py` and `03_iv_surface.py` to have run. Opening it writes nothing; running it as a script (`python 05_kalshi_vs_deribit.py`) saves its figures to `outputs/figures/` (prefixed `05_`) and the 3D surfaces to `outputs/05_iv_surfaces_3d.html`. Takes about two minutes.
    """)
    return


@app.cell(hide_code=True)
def _(alt):
    # Chart styling: reference palette (slot 1 blue = Kalshi / adjusted, slot 2 orange = Deribit / raw).
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
                "line": {"color": SLOT1, "strokeWidth": 2},
                "point": {"color": SLOT1, "size": 64, "filled": True},
            }
        }

    return H, MUTED, SLOT1, SLOT2, W


@app.cell(hide_code=True)
def _(compare, mo, pairs):
    # Every Deribit trade paired with Kalshi in the same minute (lib/pairs.py, about 90 seconds), shared with 06.
    # The earlier snapshot method (lib/compare.py) only for section 6, which compares the two.
    ROOT = mo.notebook_dir()
    _r = pairs.build_pairs(ROOT)
    paired, funnel, hour_profile, deribit_trades = _r["pairs"], _r["funnel"], _r["hour_profile"], _r["trades"]
    matched = compare.build_comparison(ROOT)["matched"]
    return ROOT, deribit_trades, funnel, hour_profile, matched, paired


@app.cell(hide_code=True)
def _(funnel, mo, pairs, paired, pd):
    _per_day = paired.groupby("day").size()
    _atm_day = pairs.atm(paired).groupby("day").size().reindex(_per_day.index, fill_value=0)
    _sources = pd.DataFrame([
        ("Kalshi", "bid and ask of every strike ($250–500 apart), every minute, all week",
         "a full smile is available at any minute"),
        ("Deribit", "individual trades only, 19:30–22:29 UTC, January–September 2026",
         "a strike appears only when someone trades it"),
    ], columns=["market", "what we observe", "consequence"])
    mo.vstack([
        mo.md(f"""
        ## 1 · Why we pair trade by trade

        To compare the two markets, their prices must be taken **at the same moment**: otherwise BTC, and the
        markets, may have moved in between, and part of any gap is only timing. The two data sets are very different:
        """),
        mo.ui.table(_sources, selection=None, show_download=False, pagination=False),
        mo.md(f"""
        So the sparse side, Deribit, sets the moments, and the dense side, Kalshi, is read there:

        - **Deribit:** each trade's **mark price** (Deribit's own fair value of that option at that instant, not the
          traded price, which lands on the bid or the ask) is inverted with Black-Scholes, using Deribit's BTC index
          and the time left to Deribit's expiry. Fills at the same strike in the same minute are usually pieces of one
          order, so they count once (their median vol).
        - **Kalshi:** the ladder at the minute end nearest the trade (at most
          {paired.max_offset_s.max():.0f} seconds away), turned into a smile exactly as in `03`, read at the Deribit
          strike. That strike is on Kalshi's ladder {paired.on_ladder.mean():.0%} of the time; otherwise its vol is
          interpolated between the two neighbouring Kalshi strikes ($250–500 apart), never beyond the ladder.
        - Kalshi's ladder must pass `03`'s checks (fresh quotes, spread at most 20¢, at least 5 strikes around spot, an
          ATM vol, at least 2 strikes per standard deviation). Its expected Friday price is **not** required to be
          close to spot.

        **How much survives each step:**
        """),
        mo.ui.table(funnel.assign(count=funnel["count"].map("{:,}".format)), selection=None, show_download=False,
                    pagination=False),
        mo.md(f"""
        **{len(paired):,} pairs on {paired.day.nunique()} days** in {paired.event_ticker.nunique()} weeklies: a median of
        **{_per_day.median():.0f} per afternoon** (10th–90th percentile {_per_day.quantile(.1):.0f}–{_per_day.quantile(.9):.0f}),
        of which {_atm_day.median():.0f} at the money. Most of the loss is Deribit trading Friday expiries more than a
        week away, before Kalshi lists that week. Deribit only trades a few strikes at a time, so this gives **the
        average smile over the year, not a full smile at any one moment**; that would need Deribit's quote data.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(ROOT, deribit_trades, mo, pairs, paired, pd):
    # Worked example: a Deribit 58,500 put traded Monday June 29, 2026, 21:44:51 UTC; both contracts expire Fri July 3
    _m, _K = pd.Timestamp("2026-06-29 21:45", tz="UTC"), 58_500.0
    _p = paired[(paired.minute == _m) & (paired.strike == _K)].iloc[0]
    _fills = deribit_trades[(deribit_trades.expiry == _p.deribit_expiry) & (deribit_trades.strike == _K)
                            & ((deribit_trades.ts - _m).abs() <= pd.Timedelta(seconds=30))]
    _q = pd.read_parquet(ROOT / "data" / "clean" / "kalshi_weekly_quotes" / f"{_p.event_ticker}.parquet",
                         columns=pairs.KALSHI_COLS)
    _snap, _strikes = pairs.kalshi_smiles(_q, [_m])[_m]
    _ladder = _strikes[_strikes.strike.between(_K - 1_000, _K + 1_000)]
    ladder_table = pd.DataFrame({"strike": _ladder.strike.map("{:,.0f}".format),
                                 "Kalshi price (above)": (100 * _ladder.mid).round(0).astype(int).astype(str) + "¢",
                                 "Kalshi implied vol": (100 * _ladder.iv).round(2).astype(str) + "%"})
    _mark_usd = float((_fills.mark_price_btc * _fills.index_price).median())
    pair_steps = pd.DataFrame([
        ("1 · Deribit trade", f"{_K:,.0f} put, {_fills.ts.min():%H:%M:%S} UTC, {len(_fills)} fills "
                              f"({' + '.join(f'{a:g}' for a in _fills.amount)} contracts) = one observation"),
        ("2 · Deribit vol", f"mark {_fills.mark_price_btc.median():.6f} BTC × index ${_p.index_price:,.2f} = "
                            f"${_mark_usd:,.0f}; {_p.tau_d * 8760:.1f} h to Deribit's expiry → {_p.deribit_iv:.2%}"),
        ("3 · Kalshi at the same moment", f"ladder at {_m:%H:%M} UTC ({_p.max_offset_s:.0f} s from the trade), BTC "
                                          f"${_p.spot:,.0f}, {_p.kalshi_hours:.1f} h to Kalshi's expiry"),
        ("4 · Kalshi vol at the strike", f"{_K:,.0f} is on Kalshi's ladder → {_p.kalshi_iv:.2%}"),
        ("5 · Stretch Deribit to Kalshi's expiry", f"× {_p.horizon_ratio:.4f} → {_p.deribit_iv_adj:.2%}"),
        ("6 · Gap", f"{_p.gap_raw:+.2f} points as traded, {_p.gap_adj:+.2f} stretched; the strike is "
                    f"{abs(_p.sd):.2f} standard deviations {'below' if _p.sd < 0 else 'above'} Kalshi's forward"),
    ], columns=["step", "this pair"])
    mo.vstack([
        mo.md("""
        ## 2 · One pair by hand

        Monday June 29, 2026, BTC near $60,400. Both contracts expire Friday July 3: Deribit at 08:00 UTC, Kalshi at
        5pm ET (21:00 UTC).
        """),
        mo.ui.table(pair_steps, selection=None, show_download=False, pagination=False),
        mo.md("Kalshi's smile at that minute, around the strike:"),
        mo.ui.table(ladder_table, selection=None, show_download=False, pagination=False),
        mo.md(f"""
        So at this moment Kalshi priced this downside strike {_p.gap_adj:.1f} vol points richer than Deribit. The
        rest of the notebook does the same for all {len(paired):,} pairs and pools them.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(H, SLOT1, SLOT2, W, alt, deribit, hour_profile, mo, np, paired, pd):
    # Walk through the 13-hour adjustment on section 2's pair (Monday June 29, 21:45 UTC, the 58,500 strike)
    _p = paired[(paired.minute == pd.Timestamp("2026-06-29 21:45", tz="UTC")) & (paired.strike == 58_500.0)].iloc[0]
    _ts, _kexp, _dexp = _p.minute, _p.kalshi_expiry, _p.deribit_expiry
    _h_k, _h_d = (_kexp - _ts) / pd.Timedelta(hours=1), (_dexp - _ts) / pd.Timedelta(hours=1)

    # hour by hour, how busy each hour usually is (1 = an average hour of the week)
    _avg = hour_profile.mean()
    _hours = pd.date_range(_ts.floor("h"), _kexp, freq="h", inclusive="left")
    _blocks = pd.DataFrame({"hour": _hours,
                            "busy": hour_profile.iloc[_hours.dayofweek * 24 + _hours.hour].to_numpy() / _avg})
    _blocks["period"] = np.where(_blocks.hour < _dexp.floor("h"), "both markets (to Deribit's expiry)",
                                 "Kalshi only: the 13 extra hours")
    timeline_chart = (
        alt.Chart(_blocks).mark_bar(width=4)
        .encode(x=alt.X("hour:T", title="time (UTC)", scale=alt.Scale(type="utc"), axis=alt.Axis(format="%a %H:%M", labelAngle=0, tickCount=8)),
                y=alt.Y("busy:Q", title="how much BTC usually moves (× average hour)"),
                color=alt.Color("period:N", title=None, scale=alt.Scale(
                    domain=["both markets (to Deribit's expiry)", "Kalshi only: the 13 extra hours"], range=[SLOT1, SLOT2])),
                tooltip=[alt.Tooltip("utcdayhoursminutes(hour):T", title="hour (UTC)"), alt.Tooltip("busy:Q", format=".2f"), "period:N"])
        .properties(title="Every hour from Monday 21:45 UTC to Kalshi's expiry, by how much BTC usually moves then",
                    width=W, height=H)
    )

    # the numbers
    _units_d = deribit.expected_variance(_ts, _dexp, hour_profile) / _avg
    _units_k = deribit.expected_variance(_ts, _kexp, hour_profile) / _avg
    _ratio = deribit.horizon_vol_ratio(_ts, _kexp, _dexp, hour_profile)
    _sig_d = float(_p.deribit_iv)                                  # Deribit's vol for this trade
    _move_d = _sig_d * np.sqrt(_h_d / 8_760)
    _move_k = _move_d * np.sqrt(_units_k / _units_d)
    _sig_k = _move_k / np.sqrt(_h_k / 8_760)
    steps = pd.DataFrame([
        ("Clock hours from now to expiry", f"{_h_d:.1f}", f"{_h_k:.1f}"),
        ("Worth in 'average hours' of BTC movement", f"{_units_d:.1f}", f"{_units_k:.1f}"),
        ("Typical move by expiry (1 standard deviation)", f"±{_move_d:.2%}  (Deribit's {_sig_d:.1%} × √({_h_d:.1f} h / 1 yr))",
         f"±{_move_k:.2%}  (= ±{_move_d:.2%} × √({_units_k:.1f} / {_units_d:.1f}))"),
        ("Same move quoted as a yearly vol", f"{_sig_d:.1%}", f"{_sig_k:.1%}  (= ±{_move_k:.2%} ÷ √({_h_k:.1f} h / 1 yr))"),
    ], columns=["", "Deribit, to Fri 08:00 UTC", "Stretched to Kalshi, to Fri 5pm ET"])
    mo.vstack([
        mo.md(f"""
        ## 3 · How the 13 extra hours are handled

        **The problem.** Deribit's Friday option asks "will BTC be above K at **08:00 UTC** Friday?". Kalshi's asks
        the same about **5pm ET** (21:00 UTC in summer). Those are different questions: BTC has 13 more hours to move
        before Kalshi settles, so Kalshi's range of outcomes is wider. Deribit has no option that ends when Kalshi's
        does, so there is no Deribit *price* for Kalshi's question.

        **The idea.** Prices cannot be moved to another expiry, but volatility can, because **variance adds up over
        time**: the variance of BTC's move to Friday 5pm is the variance to Friday 08:00 plus the variance of the 13
        hours after it. So we take the movement Deribit expects up to its expiry, add the movement the 13 extra hours
        usually bring, and convert the total back to a yearly vol for Kalshi's horizon.

        **The catch.** The 13 extra hours are not ordinary hours. They are Friday's US trading day, when BTC tends to
        move a lot (orange). Below, each bar is one hour from the example pair to Kalshi's expiry, and its height
        is how much BTC moved in that hour of the week on average over our year of minute data (`lib/deribit.py`,
        `hourly_variance`). Weekend and late-night hours are quiet; US daytime hours are busy.
        """),
        timeline_chart,
        mo.md(f"""
        **Step by step for section 2's pair** (Monday June 29, 21:45 UTC, the {_p.strike:,.0f} strike):

        1. Count each period in "average hours" of movement, weighting every hour by its bar above. Deribit's
           {_h_d:.1f} clock hours are worth **{_units_d:.1f}** average hours. Kalshi's {_h_k:.1f} are worth
           **{_units_k:.1f}**: the 13 extra hours count as **{_units_k - _units_d:.1f}**, because they are busy ones.
        2. Deribit's vol of {_sig_d:.1%} over {_h_d:.1f} hours means a typical move of **±{_move_d:.2%}** by
           Friday 08:00.
        3. Variance grows with the average hours, so by Kalshi's expiry the typical move is
           ±{_move_d:.2%} × √({_units_k:.1f} / {_units_d:.1f}) = **±{_move_k:.2%}**.
        4. Quoted per year over Kalshi's {_h_k:.1f} hours, that is **{_sig_k:.1%}**: Deribit's vol × {_ratio:.3f}.
           This is the "stretched" Deribit vol compared with Kalshi (`horizon_vol_ratio`). Each pair in section 4 gets its own factor, for its own minute.
        """),
        mo.ui.table(steps, selection=None, show_download=False, pagination=False),
        mo.md(f"""
        The yearly vol goes up, not just the move, because the extra hours are busier than the hours already in
        Deribit's period ({(_units_k - _units_d) / (_h_k - _h_d):.2f} average hours per clock hour, against
        {_units_d / _h_d:.2f}). If they were just as busy, the move would grow but the yearly vol would stay at
        {_sig_d:.1%}.

        Across all pairs the factor ranges from {paired.horizon_ratio.min():.3f} (a week out, when 13 hours are a small
        share) to {paired.horizon_ratio.max():.3f} (a day out, when they are a large share).

        **What this assumes:** Deribit's view of how volatile the week is carries over to the 13 extra hours, in the
        same proportion as history (busy hours stay 1.6× as busy). If traders expect an unusually calm or wild
        Friday, this is off. That is why the results below are also shown without it, and why section 5 looks at the
        pairs far from expiry, where it matters least. A more market-based alternative is to interpolate between
        Deribit's Friday 08:00 and Saturday 08:00 expiries, which bracket Kalshi's.
        """),
    ])
    return (timeline_chart,)


@app.cell(hide_code=True)
def _(H, MUTED, SLOT1, SLOT2, W, alt, mo, pairs, paired, pd):
    _atm = pairs.atm(paired)
    _head = pd.concat([pairs.gap_table(_atm.assign(all=1), "all", col=c).assign(version=v)
                       for c, v in [("gap_raw", "Deribit as traded"), ("gap_adj", "Deribit stretched to Kalshi's expiry")]],
                      ignore_index=True)
    headline = pd.DataFrame({
        "Deribit vol used": _head.version,
        "Kalshi − Deribit (median, vol points)": _head.median_gap.map("{:+.2f}".format),
        "95% interval": [f"{lo:+.2f} to {hi:+.2f}" for lo, hi in zip(_head.ci_low, _head.ci_high)],
        "Kalshi higher": _head.kalshi_higher.map("{:.0%}".format),
        "pairs / days": [f"{n:,} / {d}" for n, d in zip(_head.pairs, _head.days)],
    })
    _raw, _adj = _head.iloc[0], _head.iloc[1]

    # gap across the smile, with day-resampled 95% intervals
    _t = pd.concat([pairs.gap_table(paired, "sd_bucket", col=c).assign(version=v)
                    for c, v in [("gap_raw", "Deribit as traded"), ("gap_adj", "Deribit stretched to Kalshi's expiry")]])
    _t["sd"] = _t.sd_bucket.astype(float)
    _t = _t[_t.pairs >= 100]
    _color = alt.Color("version:N", title=None, scale=alt.Scale(
        domain=["Deribit as traded", "Deribit stretched to Kalshi's expiry"], range=[SLOT2, SLOT1]))
    _x = alt.X("sd:Q", title="strike, in standard deviations from the money (negative = below)")
    gap_chart = (
        alt.Chart(_t).mark_area(opacity=0.15).encode(x=_x, y="ci_low:Q", y2="ci_high:Q", color=_color)
        + alt.Chart(_t).mark_line(point=True).encode(
            x=_x, y=alt.Y("median_gap:Q", title="Kalshi − Deribit (vol points, median)"), color=_color,
            tooltip=["version:N", alt.Tooltip("sd:Q", title="SD"), alt.Tooltip("median_gap:Q", format="+.2f"),
                     alt.Tooltip("ci_low:Q", format="+.2f"), alt.Tooltip("ci_high:Q", format="+.2f"), "pairs:Q", "days:Q"])
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    ).properties(title="Vol gap across the smile (bands: 95% intervals, resampling days)", width=W, height=H + 40)
    _g = _t[_t.version == "Deribit stretched to Kalshi's expiry"].set_index("sd")

    # average smile shape: both vols divided by Kalshi's ATM vol at that moment, so volatile and calm weeks pool
    _s = (paired.assign(Kalshi=paired.kalshi_iv / paired.kalshi_atm,
                        Deribit=paired.deribit_iv_adj / paired.kalshi_atm)
          .groupby("sd_bucket", observed=True)[["Kalshi", "Deribit"]].median()
          .reset_index().melt(id_vars="sd_bucket", var_name="market", value_name="ratio"))
    _s = _s[_s.sd_bucket.astype(float).isin(_t.sd.unique())].assign(sd=lambda d: d.sd_bucket.astype(float))
    _s["market"] = _s.market.map({"Kalshi": "Kalshi", "Deribit": "Deribit (stretched)"})
    shape_chart = (
        alt.Chart(_s).mark_line(point=True).encode(
            x=_x, y=alt.Y("ratio:Q", title="implied vol ÷ Kalshi's ATM vol", scale=alt.Scale(zero=False)),
            color=alt.Color("market:N", title=None, scale=alt.Scale(domain=["Kalshi", "Deribit (stretched)"],
                                                                    range=[SLOT1, SLOT2])),
            tooltip=["market:N", alt.Tooltip("sd:Q", title="SD"), alt.Tooltip("ratio:Q", format=".3f")])
    ).properties(title="Average smile shape, Kalshi vs Deribit", width=W, height=H)

    # at the money by time to expiry
    _bt = pd.concat([pairs.gap_table(_atm, "tte_bucket", col=c).assign(v=c) for c in ["gap_raw", "gap_adj"]])
    by_tte = pd.DataFrame({
        "Kalshi time to expiry": _bt[_bt.v == "gap_raw"].tte_bucket.astype(str).to_numpy(),
        "pairs": _bt[_bt.v == "gap_raw"].pairs.to_numpy(),
        "gap as traded": _bt[_bt.v == "gap_raw"].median_gap.map("{:+.1f}".format).to_numpy(),
        "gap stretched": _bt[_bt.v == "gap_adj"].median_gap.map("{:+.1f}".format).to_numpy(),
        "95% interval (stretched)": [f"{lo:+.1f} to {hi:+.1f}" for lo, hi in
                                     zip(_bt[_bt.v == "gap_adj"].ci_low, _bt[_bt.v == "gap_adj"].ci_high)],
    })
    _bta = _bt[_bt.v == "gap_adj"].set_index("tte_bucket").median_gap
    _week, _last = _bta.get("5–7 days", float("nan")), _bta.get("1–2 days", float("nan"))

    # gap by distance and time to expiry; cells with fewer than 50 pairs left empty
    _cell = paired.groupby(["tte_bucket", "sd_bucket"], observed=True).gap_adj.agg(["median", "size"])
    grid = (_cell["median"].where(_cell["size"] >= 50).round(1).unstack("sd_bucket")
            .reindex(columns=[c for c in pairs.SD_CENTRES if abs(c) <= 2.5]))
    grid.columns = [f"{c:+.1f} SD" for c in grid.columns]
    grid = grid.reset_index().rename(columns={"tte_bucket": "time to expiry"}).astype({"time to expiry": str})

    _dn, _up = _g.loc[-2.0], _g.loc[2.0]
    mo.vstack([
        mo.md(f"""
        ## 4 · Comparing vols

        For every pair: Kalshi's vol minus Deribit's vol at the same strike and minute. Positive means Kalshi prices
        more movement. Each row of the table pools the pairs at the money (within a quarter of a standard deviation
        of Kalshi's forward). The intervals resample whole days, because pairs on the same day share the market's
        mood and are not independent.
        """),
        mo.ui.table(headline, selection=None, show_download=False, pagination=False),
        mo.md(f"""
        - **At the money, Kalshi is {_adj.median_gap:+.1f} points above Deribit** after the expiry adjustment
          ({_raw.median_gap:+.1f} as traded), and higher in {_adj.kalshi_higher:.0%} of pairs.
        - **Deribit's smile is steeper.** Moving away from the money, the gap turns negative: two standard deviations
          below, Kalshi is **{_dn.median_gap:+.1f}** points (95% interval {_dn.ci_low:+.1f} to {_dn.ci_high:+.1f});
          two above, {_up.median_gap:+.1f} ({_up.ci_low:+.1f} to {_up.ci_high:+.1f}). Deribit charges more for large
          moves, above all for crashes.
        """),
        gap_chart,
        mo.md("""
        **The same in shapes.** Both vols divided by Kalshi's ATM vol at that moment, so calm and volatile weeks pool
        (as for the surface in `03`). Deribit's curve sits lower near the money and higher in both wings.
        """),
        shape_chart,
        mo.md(f"""
        **At the money, by time to expiry.** The adjustment matters most near expiry, where 13 hours are a larger
        share of what is left. It removes most of the pattern, **but not all of it**: the stretched gap still grows
        from {_week:+.1f} points with 5–7 days left to {_last:+.1f} with 1–2 days left. Either Kalshi really prices the last days
        richer, or the 13 hours carry more movement than the historical profile gives them (for instance because the
        market expects news on Friday). A week out, where the adjustment is smallest, the two markets are close.
        """),
        mo.ui.table(by_tte, selection=None, show_download=False, pagination=False),
        mo.md("**The full grid** (gap after the adjustment, median per cell; empty with fewer than 50 pairs):"),
        mo.ui.table(grid, selection=None, show_download=False, pagination=False),
    ])
    return gap_chart, shape_chart


@app.cell(hide_code=True)
def _(H, SLOT1, SLOT2, W, alt, mo, pairs, paired, pd):
    _d = (pairs.atm(paired).groupby("day")[["kalshi_iv", "deribit_iv_adj"]].median().reset_index()
          .rename(columns={"kalshi_iv": "Kalshi", "deribit_iv_adj": "Deribit (stretched)"})
          .assign(day=lambda d: pd.to_datetime(d.day)))
    _corr = _d[["Kalshi", "Deribit (stretched)"]].corr().iloc[0, 1]
    _long = _d.melt(id_vars="day", var_name="market", value_name="vol")
    daily_chart = (
        alt.Chart(_long).mark_line(strokeWidth=1.6)
        .encode(x=alt.X("day:T", title=None, axis=alt.Axis(format="%b %Y")),
                y=alt.Y("vol:Q", title="implied vol at the money", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
                color=alt.Color("market:N", title=None, scale=alt.Scale(domain=["Kalshi", "Deribit (stretched)"],
                                                                        range=[SLOT1, SLOT2])),
                tooltip=[alt.Tooltip("day:T", format="%Y-%m-%d"), "market:N", alt.Tooltip("vol:Q", format=".1%")])
        .properties(title="At-the-money vol, daily median of the pairs", width=W, height=H)
    )
    _above = (_d["Kalshi"] > _d["Deribit (stretched)"]).mean()
    mo.vstack([
        mo.md(f"""
        **Day by day** ({len(_d)} days with pairs at the money), the two markets' vols rise and fall together
        (correlation **{_corr:.2f}**), with Kalshi above on {_above:.0%} of days.
        """),
        daily_chart,
    ])
    return (daily_chart,)


@app.cell(hide_code=True)
def _(W, alt, go, mo, pairs, paired, pd):
    # The surfaces: Kalshi, Deribit (stretched) and their gap, by days to Kalshi's expiry x standard deviations
    surf = pairs.surface(paired)
    surf["to_expiry"] = ((surf.hours_lo + surf.hours_hi) / 48).round().astype(int)
    surf["row"] = surf.to_expiry.map(lambda d: f"{d} day" + ("s" if d > 1 else ""))
    _rows = [f"{d} day" + ("s" if d > 1 else "") for d in sorted(surf.to_expiry.unique())]
    _x = alt.X("sd:O", title="standard deviations from the money", axis=alt.Axis(labelAngle=0))
    _y = alt.Y("row:O", title="time to Kalshi's expiry", sort=_rows)
    _tip = [alt.Tooltip("row:N", title="to expiry"), alt.Tooltip("sd:Q", title="SD"),
            alt.Tooltip("kalshi:Q", format=".1%", title="Kalshi"), alt.Tooltip("deribit:Q", format=".1%", title="Deribit"),
            alt.Tooltip("gap:Q", format="+.1f", title="gap (points)"), "pairs:Q", alt.Tooltip("days:Q", title="days with pairs")]
    _vol_scale = alt.Scale(range=["#cde2fb", "#0d366b"], domain=[0.35, 0.80], clamp=True)   # as in 03

    def _level(col, title):
        base = alt.Chart(surf).encode(x=_x, y=_y, tooltip=_tip)
        return (base.mark_rect().encode(color=alt.Color(f"{col}:Q", title="implied vol", scale=_vol_scale,
                                                        legend=alt.Legend(format="%")))
                + base.mark_text(fontSize=9).encode(
                    text=alt.Text(f"{col}:Q", format=".0%"),
                    color=alt.condition(f"datum.{col} > 0.6", alt.value("white"), alt.value("#0b0b0b")))
                ).properties(title=title, width=W // 2 - 20, height=200)

    levels_chart = alt.hconcat(_level("kalshi", "Kalshi"), _level("deribit", "Deribit (stretched)")).resolve_scale(color="shared")
    _gap_scale = alt.Scale(domain=[-8, 0, 8], range=["#c4501f", "#f2f1ec", "#1f5fae"], clamp=True,
                           interpolate="rgb")   # orange < 0 < blue, straight blends (no detour through green)
    _base = alt.Chart(surf).encode(x=_x, y=_y, tooltip=_tip)
    gap_surface_chart = (
        _base.mark_rect().encode(color=alt.Color("gap:Q", title="Kalshi − Deribit (points)", scale=_gap_scale))
        + _base.mark_text(fontSize=10).encode(
            text=alt.Text("gap:Q", format="+.1f"),
            color=alt.condition("abs(datum.gap) > 5", alt.value("white"), alt.value("#0b0b0b")))
    ).properties(title="Gap surface: Kalshi − Deribit, stretched (blue: Kalshi higher; orange: Deribit higher)",
                 width=W, height=230)

    # rotatable 3D: both surfaces on the same axes
    _sd = sorted(surf.sd.unique())
    _days = sorted(surf.to_expiry.unique())
    _grid = lambda c: surf.pivot(index="to_expiry", columns="sd", values=c).reindex(index=_days, columns=_sd).to_numpy()
    surface_3d = go.Figure([
        go.Surface(x=_sd, y=_days, z=_grid("kalshi"), name="Kalshi", colorscale=[[0, "#cde2fb"], [1, "#0d366b"]],
                   showscale=False, opacity=0.95,
                   hovertemplate="Kalshi<br>SD %{x}<br>%{y} days<br>vol %{z:.1%}<extra></extra>"),
        go.Surface(x=_sd, y=_days, z=_grid("deribit"), name="Deribit", colorscale=[[0, "#fbd9c8"], [1, "#b8461b"]],
                   showscale=False, opacity=0.75,
                   hovertemplate="Deribit (stretched)<br>SD %{x}<br>%{y} days<br>vol %{z:.1%}<extra></extra>"),
    ])
    surface_3d.update_layout(
        title="Kalshi (blue) and Deribit stretched (orange) implied-vol surfaces: drag to rotate",
        scene=dict(xaxis_title="standard deviations from the money", yaxis_title="days to Kalshi's expiry",
                   zaxis=dict(title="implied vol", tickformat=".0%")),
        height=560, margin=dict(l=0, r=0, t=40, b=0))

    _g = surf.set_index(["to_expiry", "sd"]).gap
    _centre = surf[surf.sd.between(0.0, 1.0)]
    _down = surf[surf.sd <= -1.5]
    mo.vstack([
        mo.md(f"""
        ### The surfaces: all distances and expiries at once

        The same pairs, laid out by distance from the money (columns, half a standard deviation wide) and time to
        Kalshi's expiry (rows). Deribit trades only in the US afternoon, so every pair is within about two hours of a
        whole number of days before Kalshi's Friday 5pm expiry, and each row is one day. As for `03`'s surface, each
        vol is divided by Kalshi's ATM vol at that minute, the median is taken per cell, and the result is scaled back
        by the row's typical ATM vol, so calm and volatile weeks do not distort the edges. Both markets are pooled on
        exactly the same pairs; Deribit is stretched to Kalshi's expiry. Cells with fewer than 40 pairs are empty.
        """),
        levels_chart,
        mo.md(f"""
        **The gap surface** is the median of the pairs' own gaps in each cell (vol points), so it is the comparison
        itself, not the difference of the two pictures above:

        - From the money to one standard deviation above, **Kalshi is higher in {(_centre.gap > 0).sum()} of
          {len(_centre)} cells** (from {_centre.gap.min():+.1f} to {_centre.gap.max():+.1f} points).
        - From 1.5 standard deviations below, **Deribit is higher in {(_down.gap < 0).sum()} of {len(_down)} cells**
          (from {_down.gap.max():+.1f} to {_down.gap.min():+.1f} points): its smile is steeper on the downside at
          every horizon.
        - **With one day left both wings swing hard** toward Deribit ({_g.get((1, -2.0), float('nan')):+.1f} at
          −2 standard deviations, {_g.get((1, 2.0), float('nan')):+.1f} at +2): this is where the 13 extra hours are
          the largest share of what is left, so it is the row that depends most on the stretch.
        """),
        gap_surface_chart,
        mo.md("**Both surfaces in 3D** (drag to rotate, hover for values). Gaps in a surface are cells with too few pairs."),
        surface_3d,
    ])
    return gap_surface_chart, levels_chart, surf, surface_3d


@app.cell(hide_code=True)
def _(H, MUTED, SLOT1, SLOT2, W, alt, hour_profile, mo, pairs, paired, pd):
    _far = pairs.far_from_expiry(paired)
    _atm_far = pairs.atm(_far)
    _versions = [("gap_raw", "no stretch: Deribit as traded"), ("gap_adj", "stretched, as in section 4")]
    _head = pd.concat([pairs.gap_table(_atm_far.assign(all=1), "all", col=c).assign(version=v) for c, v in _versions],
                      ignore_index=True)
    far_headline = pd.DataFrame({
        "Deribit vol used": _head.version,
        "Kalshi − Deribit at the money (median, vol points)": _head.median_gap.map("{:+.2f}".format),
        "95% interval": [f"{lo:+.2f} to {hi:+.2f}" for lo, hi in zip(_head.ci_low, _head.ci_high)],
        "Kalshi higher": _head.kalshi_higher.map("{:.0%}".format),
        "pairs / days": [f"{n:,} / {d}" for n, d in zip(_head.pairs, _head.days)],
    })
    _raw, _adj = _head.iloc[0], _head.iloc[1]

    # gap across the smile, far from expiry, both versions
    _t = pd.concat([pairs.gap_table(_far, "sd_bucket", col=c).assign(version=v) for c, v in _versions])
    _t["sd"] = _t.sd_bucket.astype(float)
    _t = _t[_t.pairs >= 100]
    _color = alt.Color("version:N", title=None, scale=alt.Scale(domain=[v for _, v in _versions], range=[SLOT2, SLOT1]))
    _x = alt.X("sd:Q", title="strike, in standard deviations from the money (negative = below)")
    far_gap_chart = (
        alt.Chart(_t).mark_area(opacity=0.15).encode(x=_x, y="ci_low:Q", y2="ci_high:Q", color=_color)
        + alt.Chart(_t).mark_line(point=True).encode(
            x=_x, y=alt.Y("median_gap:Q", title="Kalshi − Deribit (vol points, median)"), color=_color,
            tooltip=["version:N", alt.Tooltip("sd:Q", title="SD"), alt.Tooltip("median_gap:Q", format="+.2f"),
                     alt.Tooltip("ci_low:Q", format="+.2f"), alt.Tooltip("ci_high:Q", format="+.2f"), "pairs:Q", "days:Q"])
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    ).properties(title="Vol gap across the smile, 4–7 days to expiry (bands: 95% intervals, resampling days)",
                 width=W, height=H + 40)
    _dn = _t[_t.version == _versions[0][1]].set_index("sd").loc[-2.0]

    # how much does each horizon depend on the stretch? the ATM gap under three assumptions about the extra hours
    _busy = hour_profile.copy()
    _busy.iloc[[4 * 24 + h for h in range(8, 23)]] *= 1.5          # Friday 08:00-23:00 UTC, 1.5x its usual variance
    _a = pairs.atm(paired)
    _s = (pd.DataFrame({"time to expiry": _a.tte_bucket, "no stretch": _a.gap_raw, "historical stretch": _a.gap_adj,
                        "Friday's extra hours 1.5× busier than history": pairs.gap_with_profile(_a, _busy)})
          .groupby("time to expiry", observed=True).median())
    _spread = _s.max(axis=1) - _s.min(axis=1)
    sensitivity = _s.round(1).assign(**{"range across the three": _spread.round(1)}).reset_index().astype({"time to expiry": str})

    _d = _atm_far.groupby("day")[["kalshi_iv", "deribit_iv"]].median()
    mo.vstack([
        mo.md(f"""
        ## 5 · Far from expiry: a comparison that hardly needs the stretch

        The 13 extra hours matter less the further away the expiry is. With **4–7 days left** (at least
        {pairs.FAR_FROM_EXPIRY_HOURS} hours), they are about {13 / _far.kalshi_hours.median():.0%} of Kalshi's horizon,
        and the stretch moves Deribit's vol by only ×{_far.horizon_ratio.min():.3f} to ×{_far.horizon_ratio.max():.3f}.
        Here the two markets can be compared almost directly: {len(_far):,} pairs on {_far.day.nunique()} days, of
        which {len(_atm_far):,} at the money.
        """),
        mo.ui.table(far_headline, selection=None, show_download=False, pagination=False),
        mo.md(f"""
        - **At the money, Kalshi is above Deribit either way**: {_raw.median_gap:+.1f} points with no stretch at all,
          {_adj.median_gap:+.1f} with it. No stretch treats Kalshi's extra Friday hours as no busier than the rest of
          Deribit's horizon; history says they are busier, so the true gap is most likely between the two.
        - **Deribit's steeper downside does not depend on the stretch**: two standard deviations below the money,
          Kalshi is {_dn.median_gap:+.1f} points with no stretch (95% interval {_dn.ci_low:+.1f} to {_dn.ci_high:+.1f}).
        - **Day by day the two move together**: correlation {_d.corr().iloc[0, 1]:.2f} with no stretch, over {len(_d)} days.
        """),
        far_gap_chart,
        mo.md(f"""
        **Why far from expiry is the cleanest comparison.** The table redoes the at-the-money gap under three
        assumptions about Kalshi's extra Friday hours: as busy as the rest of the horizon (no stretch), as busy as
        history says (the stretch used everywhere else), and half as busy again. A week out, the three readings span
        {_spread.get('5–7 days', float('nan')):.1f} points; one to two days out, {_spread.get('1–2 days', float('nan')):.1f}.
        So the results far from expiry are the ones that least depend on how the 13 hours are handled. The pattern near
        expiry (section 4) could be Kalshi pricing the last days richer, or the market expecting a busier Friday than
        history: these pairs alone cannot tell the two apart.
        """),
        mo.ui.table(sensitivity, selection=None, show_download=False, pagination=False),
    ])
    return (far_gap_chart,)


@app.cell(hide_code=True)
def _(matched, mo, pairs, paired, pd):
    _old = matched[matched.log_moneyness.abs() < 0.005].dropna(subset=["vol_gap_adj"])
    _new = pairs.atm(paired)
    _rows = pd.DataFrame([
        ("Moment", "Kalshi every 15 minutes", "every Deribit trade"),
        ("Time between the two prices", "up to 15 minutes (median about 7.5)", f"at most {paired.max_offset_s.max():.0f} seconds"),
        ("Deribit at Kalshi's strike", "straight lines between the few strikes Deribit traded in 30 minutes",
         "Deribit's own strike, no interpolation"),
        ("Kalshi at Deribit's strike", "Kalshi's own strike", "Kalshi's own strike, or between neighbours $250–500 apart"),
        ("At-the-money comparisons", f"{len(_old):,} on {_old.ts.dt.date.nunique()} days", f"{len(_new):,} on {_new.day.nunique()} days"),
        ("Gap at the money, stretched", f"{_old.vol_gap_adj.median():+.2f} points", f"{_new.gap_adj.median():+.2f} points"),
        ("Gap at the money, as traded", f"{_old.vol_gap_raw.median():+.2f} points", f"{_new.gap_raw.median():+.2f} points"),
    ], columns=["", "before: snapshots, Deribit within ±15 min", "now: pairs"])
    mo.vstack([
        mo.md("""
        ## 6 · What changed from the snapshot method

        The earlier version of this notebook compared vols at Kalshi snapshots, building a Deribit smile from the trades
        within 15 minutes. The pairs give a larger gap at the money. The main reason is interpolation, not timing:
        Deribit's smile is curved, and a straight line between two traded strikes on either side of the money lies
        **above** the curve, so the snapshot method overstated Deribit's ATM vol and understated the gap. (Narrowing the
        snapshot window from 15 to 2 minutes left the gap unchanged once the set of strikes was held fixed.)
        """),
        mo.ui.table(_rows, selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo, pairs, paired):
    _a = pairs.atm(paired)
    _t = pairs.gap_table(_a.assign(all=1), "all").iloc[0]
    _by = pairs.gap_table(paired, "sd_bucket").assign(sd=lambda d: d.sd_bucket.astype(float)).set_index("sd")
    _tt = _a.groupby("tte_bucket", observed=True).gap_adj.median()
    _week, _last, _dn = _tt.get("5–7 days", float("nan")), _tt.get("1–2 days", float("nan")), _by.loc[-2.0].median_gap
    _far = pairs.atm(pairs.far_from_expiry(paired))
    _far_raw, _far_adj = _far.gap_raw.median(), _far.gap_adj.median()
    mo.md(f"""
    ## 7 · Main takeaways

    **Q2b, are Kalshi's prices consistent with option prices? In implied vol, broadly yes, with a richer centre and
    flatter wings.**

    1. **Vol at the money: Kalshi about {_t.median_gap:.1f} points above Deribit** (95% interval {_t.ci_low:+.1f} to
       {_t.ci_high:+.1f}), from {len(paired):,} pairs taken at most 30 seconds apart, after aligning expiries. Kalshi is
       higher in {_t.kalshi_higher:.0%} of pairs, and the two move together day by day.
       **Far from expiry (4–7 days), where the expiry adjustment barely matters, Kalshi is {_far_adj:+.1f} points
       above with it and {_far_raw:+.1f} without it**: higher either way.
    2. **Deribit's smile is steeper.** Two standard deviations below the money Kalshi is
       {_dn:+.1f} points: Deribit traders pay more for crash protection (and, less clearly,
       for large rises).
    3. **The expiry mismatch matters, and the adjustment is not the last word.** Without it the gap at the money is
       {_a.gap_raw.median():+.1f} points. With it, the gap still grows toward expiry (from {_week:+.1f} a week out to
       {_last:+.1f} with 1–2 days left), so results close to expiry depend on the adjustment much more than results
       far from it (section 5). A market-based alternative, interpolating between Deribit's Friday and Saturday 08:00
       expiries, would test it.
    4. **Why gaps can persist at all:** different users (US retail on Kalshi; professional and offshore on Deribit,
       which does not serve US residents), so little capital moves between them; Kalshi's full collateral and fees;
       Kalshi's 1¢ tick and $500 strikes.

    **Caveats.** The pairs only exist where Deribit trades, so the far wings rest on few pairs. Deribit marks are
    Deribit's model values, not executable prices. The 13-hour adjustment uses BTC's past hour-by-hour movement, not
    what the market expected that Friday.

    **Next.** Deribit option-chain quotes (Tardis.dev or similar) would give Deribit's full smile every minute. Test the
    expiry adjustment with Deribit's Saturday expiry. Extend to probabilities for Q3 (Kalshi vs Deribit vs realized
    outcomes).
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, pairs, paired, pd):
    # Every definition, with the numbers of section 2's pair (Monday June 29, 21:45 UTC, the 58,500 strike)
    _p = paired[(paired.minute == pd.Timestamp("2026-06-29 21:45", tz="UTC")) & (paired.strike == 58_500.0)].iloc[0]
    _move = _p.kalshi_atm * np.sqrt(_p.tau_k)
    mo.md(rf"""
    ## Appendix · How every number is computed

    The definitions behind every table and chart above, each with the numbers of the pair from section 2
    (Monday June 29, 21:45 UTC, the {_p.strike:,.0f} strike).

    **Deribit's vol for one observation.**
    - Deribit's options are priced in bitcoin; mark price × Deribit's BTC index = the dollar price
      ({_p.index_price:,.2f} index here).
    - Black-Scholes is inverted on that dollar price with the forward set to Deribit's index, zero interest rates,
      and the time left to Deribit's expiry (Friday 08:00 UTC) in years: hours ÷ 8,760 ({_p.tau_d * 8760:.1f} hours
      here). Result: {_p.deribit_iv:.2%}.
    - An observation is all trades in the same expiry and strike whose **nearest minute end** is the same (a trade
      at 21:44:51 belongs to 21:45). Its vol is the **median** of those trades' vols, so an order split into many fills
      counts once ({_p.n_trades:.0f} fills here). Calls and puts at the same strike are pooled: in theory they have
      the same vol.

    **Kalshi's vol at the same strike and minute.**
    - Kalshi's quotes are recorded per minute and stamped at the **end** of the minute; the row at the trade's
      nearest minute end is used, so the two prices are at most 30 seconds apart ({_p.max_offset_s:.0f} seconds here).
    - That minute's ladder is turned into a smile exactly as in `03` (stacked ladder → calls and puts → Black-Scholes
      inversion), using Kalshi's **own forward** F: the average Friday price implied by Kalshi's ladder (put-call
      parity), not the exchange price. F = {_p.forward:,.2f} here, against ${_p.spot:,.2f} on the exchanges.
    - The Deribit strike is usually a Kalshi strike, and then its vol is read directly. Otherwise it is interpolated
      **linearly in log-strike** between the two neighbouring Kalshi strikes that have a vol. Nothing beyond the
      ladder's last strikes is used. Here: on the ladder, {_p.kalshi_iv:.2%}.
    - Kalshi's **ATM vol** σ_ATM is the smile at ln(K / F) = 0, interpolated the same way between the two strikes
      around F: {_p.kalshi_atm:.2%} here.

    **Standard deviations from the money** (the x-axis everywhere):

    $$ \text{{sd}} = \frac{{\ln(K / F)}}{{\sigma_{{ATM}} \sqrt{{\tau}}}} $$

    - K is the strike, F Kalshi's forward, σ_ATM Kalshi's ATM vol and τ Kalshi's time to expiry in years, all at
      that minute.
    - σ_ATM √τ is Kalshi's **typical move** (one standard deviation, in log terms) between now and Friday 5pm ET;
      sd says how many typical moves the strike is from the forward. It puts a strike 1 day out and one 7 days out on
      the same scale, and calm and volatile weeks too.
    - The same Kalshi-based sd is used for both markets in a pair, so their x-value is identical.
    - Here: ln({_p.strike:,.0f} / {_p.forward:,.2f}) = {_p.log_moneyness:+.4f}; typical move
      {_p.kalshi_atm:.4f} × √({_p.kalshi_hours:.1f} / 8,760) = {_move:.4f}; **sd = {_p.sd:+.2f}**.

    **Pooling.**
    - **Gap** = 100 × (Kalshi vol − Deribit vol), in vol points: +1 means Kalshi's vol is one percentage point higher.
      "Stretched" uses Deribit's vol × the 13-hour factor; "as traded" or "no stretch" uses it unchanged.
    - **At the money** = sd between −0.25 and +0.25.
    - **SD buckets** are half a standard deviation wide, labelled by their centre (−3, −2.5, …, +3): bucket −0.5 holds
      sd from −0.75 up to and including −0.25. This pair (sd {_p.sd:+.2f}) is in bucket −0.5.
    - **Time-to-expiry buckets** use Kalshi's hours left: 1–2 days (24–48 h), 2–3 days (48–72 h), 3–5 days
      (72–120 h), 5–7 days (120–170 h). **Far from expiry** (section 5) = at least
      {pairs.FAR_FROM_EXPIRY_HOURS} hours (4 days) left. This pair: {_p.kalshi_hours:.1f} hours, so it is in
      "{_p.tte_bucket}" and {"also" if _p.kalshi_hours >= pairs.FAR_FROM_EXPIRY_HOURS else "just short of"}
      "far from expiry".
    - Every number reported per bucket is the **median** gap, not the mean, so a few extreme pairs (often deep in the
      wings, where one cent moves the vol a lot) do not drive it. Each observation counts once, whatever its size.
    - "**Kalshi higher**" = share of pairs with a positive gap.
    - **95% intervals** resample whole days: draw as many days as there are, with replacement; pool their pairs; take
      the median gap; repeat 500 times; report the 2.5th and 97.5th percentiles of those medians. Days, not pairs,
      because pairs on the same day share the market's mood, and treating them as independent would make the
      intervals too narrow.
    - Charts drop buckets with fewer than 100 pairs; the grid in section 4 leaves cells with fewer than 50 empty.
    - **Daily series**: per day, the median Kalshi vol and the median (stretched) Deribit vol over the pairs at the
      money; the correlation is between those two daily series.
    - **Smile shape** (section 4): each vol divided by Kalshi's ATM vol at that minute before taking medians, so a
      1.10 means 10% above the ATM vol, whatever the week's overall level.
    - **Surfaces** (section 4): rows are days to Kalshi's expiry (pairs within 12 hours either side of 1, 2, …, 7
      days), columns the half-SD buckets. Kalshi's and Deribit's (stretched) surfaces: per cell, the median of
      vol ÷ Kalshi's ATM vol, times the median Kalshi ATM vol of that row. The gap surface: per cell, the median of
      the pairs' gaps. Cells with fewer than 40 pairs are left empty.

    **The 13-hour factor.**
    - Hour-of-week profile: from Bitcoin's 1-minute prices (Coinbase and Bitstamp), the average squared 1-minute log
      return in each of the 168 hours of the week, × 60 to make it per hour.
    - V(t, T) = the sum of that profile from t to T, partial hours counted pro rata: the movement history expects
      between t and T.
    - Factor = √( [V(t, T_Kalshi) ÷ hours to T_Kalshi] ÷ [V(t, T_Deribit) ÷ hours to T_Deribit] ), computed for each
      pair's own minute and applied to Deribit's vol at every strike ({_p.horizon_ratio:.4f} here).
    - The sensitivity table in section 5 recomputes the factor after multiplying the profile for Friday 08:00–23:00 UTC
      (the hours that contain the extra 13) by 1.5, and also with no factor at all (equivalent to every hour being
      alike).
    """)
    return


@app.cell(hide_code=True)
def _(ROOT, daily_chart, far_gap_chart, gap_chart, gap_surface_chart, levels_chart, mo, shape_chart, surface_3d,
      timeline_chart):
    # Save the figures when run as a script (python 05_kalshi_vs_deribit.py); opening the notebook writes nothing
    if not mo.running_in_notebook():
        _out = ROOT / "outputs" / "figures"
        _out.mkdir(parents=True, exist_ok=True)
        for _name, _chart in [("13h_hour_profile", timeline_chart), ("vol_gap_across_smile", gap_chart),
                              ("smile_shape", shape_chart), ("daily_atm_vol", daily_chart),
                              ("iv_surfaces", levels_chart), ("gap_surface", gap_surface_chart),
                              ("far_from_expiry_gap", far_gap_chart)]:
            _chart.save(_out / f"05_{_name}.png", scale_factor=2)
        surface_3d.write_html(ROOT / "outputs" / "05_iv_surfaces_3d.html", include_plotlyjs="cdn")
    return


if __name__ == "__main__":
    app.run()
