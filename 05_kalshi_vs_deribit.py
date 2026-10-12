import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="05 Kalshi vs Deribit")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    from lib import compare, deribit, pairs

    return alt, compare, deribit, mo, np, pairs, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 05 · Kalshi vs Deribit

    Handout Q2b: **are Kalshi's prices consistent with option prices?** Kalshi sells "$1 if BTC ends above K";
    Deribit sells calls and puts. Two comparisons:

    1. **Implied vols, trade by trade (sections 1–5, the main result).** Kalshi has a full ladder of quotes every
       minute; Deribit only has trades, at scattered strikes and times. So every Deribit trade is the anchor: its
       vol from Deribit's mark price, against Kalshi's vol **at the same strike, in the same minute** (at most 30
       seconds apart). No Deribit smile is built and nothing is interpolated on the Deribit side.
    2. **Prices in cents (sections 6–7).** A tight Deribit call spread pays like a Kalshi contract (handout Q1c).
       Pricing it at Kalshi's strikes needs several Deribit strikes at once, so this comparison still pools Deribit
       trades within 15 minutes of a Kalshi snapshot every 15 minutes.

    **The expiry mismatch** applies to both: Deribit's Friday option expires at 08:00 UTC, Kalshi's at 5pm ET, about
    13 hours later. Deribit's vol is stretched to Kalshi's horizon with the hour-of-week variance profile (section 3).
    Results are shown **with and without** this adjustment, because it is a model.

    The code is in `lib/pairs.py` (vols), `lib/compare.py` (prices) and `lib/deribit.py`, tested in `tests/`.
    Needs `02_clean_data.py` and `03_iv_surface.py` to have run. Writes nothing. Takes about two minutes.
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
    # Vols: every Deribit trade paired with Kalshi in the same minute (lib/pairs.py, about 90 seconds).
    # Prices: Kalshi snapshots matched to Deribit trades within 15 minutes (lib/compare.py). Both shared with 06.
    ROOT = mo.notebook_dir()
    DERIVED = ROOT / "data" / "derived"
    _r = pairs.build_pairs(ROOT)
    paired, funnel, hour_profile, deribit_trades = _r["pairs"], _r["funnel"], _r["hour_profile"], _r["trades"]
    _c = compare.build_comparison(ROOT)
    matched, smiles, snaps, trades = _c["matched"], _c["smiles"], _c["snaps"], _c["trades"]
    return DERIVED, ROOT, deribit_trades, funnel, hour_profile, matched, paired, smiles, snaps, trades


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
def _(H, SLOT1, SLOT2, W, alt, compare, deribit, hour_profile, matched, mo, np, pd, smiles):
    # Walk through the 13-hour adjustment on one snapshot (Monday June 29, 20:30 UTC)
    _ts = pd.Timestamp("2026-06-29 20:30", tz="UTC")
    _snap = int(matched[matched.ts == _ts].snap.iloc[0])
    _s = smiles[_snap]
    _h_k, _h_d = float(_s["kalshi_hours"]), _s["tau_d"] * 8_760
    _kexp, _dexp = _ts + pd.Timedelta(hours=_h_k), _ts + pd.Timedelta(hours=_h_d)

    # hour by hour, how busy each hour usually is (1 = an average hour of the week)
    _avg = hour_profile.mean()
    _hours = pd.date_range(_ts.floor("h"), _kexp, freq="h", inclusive="left")
    _blocks = pd.DataFrame({"hour": _hours,
                            "busy": hour_profile.iloc[_hours.dayofweek * 24 + _hours.hour].to_numpy() / _avg})
    _blocks["period"] = np.where(_blocks.hour < _dexp.floor("h"), "both markets (to Deribit's expiry)",
                                 "Kalshi only: the 13 extra hours")
    timeline_chart = (
        alt.Chart(_blocks).mark_bar(width=4)
        .encode(x=alt.X("hour:T", title=None, axis=alt.Axis(format="%a %H:%M", labelAngle=0, tickCount=8)),
                y=alt.Y("busy:Q", title="how much BTC usually moves (× average hour)"),
                color=alt.Color("period:N", title=None, scale=alt.Scale(
                    domain=["both markets (to Deribit's expiry)", "Kalshi only: the 13 extra hours"], range=[SLOT1, SLOT2])),
                tooltip=[alt.Tooltip("hour:T", format="%a %H:%M UTC"), alt.Tooltip("busy:Q", format=".2f"), "period:N"])
        .properties(title="Every hour from Monday 20:30 UTC to Kalshi's expiry, by how much BTC usually moves then",
                    width=W, height=H)
    )

    # the numbers
    _units_d = deribit.expected_variance(_ts, _dexp, hour_profile) / _avg
    _units_k = deribit.expected_variance(_ts, _kexp, hour_profile) / _avg
    _ratio = deribit.horizon_vol_ratio(_ts, _kexp, _dexp, hour_profile)
    _sig_d = float(compare.interp_vol(0.0, _s["smile"].log_moneyness, _s["smile"].vol))   # Deribit ATM vol
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

    _k = matched[(matched.ts == _ts) & matched.strike.isin([60_000.0, 60_500.0, 61_000.0, 61_500.0, 62_000.0])]
    prices = pd.DataFrame({
        "strike": _k.strike.map("{:,.0f}".format),
        "Kalshi bid / ask": [f"{b * 100:.0f} / {a * 100:.0f}¢" for b, a in zip(_k.bid, _k.ask)],
        "Deribit vol as traded": (100 * _k.deribit_vol).round(1).astype(str) + "%",
        "Deribit vol stretched": (100 * _k.deribit_vol_adj).round(1).astype(str) + "%",
        "Deribit price, to its own expiry": (100 * _k.deribit_digital_raw).round(1).astype(str) + "¢",
        "Deribit price, stretched to Kalshi's": (100 * _k.deribit_digital_adj).round(1).astype(str) + "¢",
    })
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
        move a lot (orange). Below, each bar is one hour from the example snapshot to Kalshi's expiry, and its height
        is how much BTC moved in that hour of the week on average over our year of minute data (`lib/deribit.py`,
        `hourly_variance`). Weekend and late-night hours are quiet; US daytime hours are busy.
        """),
        timeline_chart,
        mo.md(f"""
        **Step by step for this snapshot** (Monday June 29, 20:30 UTC, ATM):

        1. Count each period in "average hours" of movement, weighting every hour by its bar above. Deribit's
           {_h_d:.1f} clock hours are worth **{_units_d:.1f}** average hours. Kalshi's {_h_k:.1f} are worth
           **{_units_k:.1f}**: the 13 extra hours count as **{_units_k - _units_d:.1f}**, because they are busy ones.
        2. Deribit's ATM vol of {_sig_d:.1%} over {_h_d:.1f} hours means a typical move of **±{_move_d:.2%}** by
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

        **The same factor ({_ratio:.3f}) is applied to Deribit's whole smile** for this snapshot, and the stretched
        vols then price Kalshi's contracts over Kalshi's own time to expiry:
        """),
        mo.ui.table(prices, selection=None, show_download=False, pagination=False),
        mo.md(f"""
        Across all matched snapshots the factor ranges from {matched.horizon_ratio.min():.3f} (a week out, when 13 hours
        are a small share) to {matched.horizon_ratio.max():.3f} (a day out, when they are a large share).

        **What this assumes:** Deribit's view of how volatile the week is carries over to the 13 extra hours, in the
        same proportion as history (busy hours stay 1.6× as busy). If traders expect an unusually calm or wild
        Friday, this is off. That is why every result below is shown both ways. A more market-based alternative is
        to interpolate between Deribit's Friday 08:00 and Saturday 08:00 expiries, which bracket Kalshi's.
        """),
    ])
    return


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
    return


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
    mo.vstack([
        mo.md(f"""
        **Day by day** ({len(_d)} days with pairs at the money), the two markets' vols rise and fall together
        (correlation **{_corr:.2f}**), with Kalshi above on {(_d.Kalshi > _d["Deribit (stretched)"]).mean():.0%} of days.
        """),
        daily_chart,
    ])
    return


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
        ## 5 · What changed from the snapshot method

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
def _(mo):
    mo.md("""
    ## 6 · Comparing prices

    Vols are the cleanest comparison, but the handout asks about prices too: what does Deribit charge for Kalshi's
    contract? That needs a Deribit **call spread** at Kalshi's strike, so Deribit prices at two strikes at the same
    time. Deribit rarely trades two strikes in the same minute, so this section still uses the snapshot method: each
    Kalshi snapshot (every 15 minutes) with the Deribit trades within 15 minutes, turned into a Deribit smile, which
    prices the call spread at every Kalshi strike.
    """)
    return


@app.cell(hide_code=True)
def _(DERIVED, compare, mo, np, pd, snaps, trades):
    # Worked example: Monday June 29, 2026, around 20:26 UTC, both contracts expiring Friday July 3
    _snap = snaps[(snaps.ts == pd.Timestamp("2026-06-29 20:30", tz="UTC"))
                  & (snaps.expiry.dt.date == pd.Timestamp("2026-07-03").date())].iloc[0]
    _t = trades[(trades.expiry == _snap.deribit_expiry) & ((trades.ts - _snap.ts).abs() <= pd.Timedelta(minutes=15))]
    _c = _t[_t.cp == "C"].assign(mark_usd=lambda d: d.mark_price_btc * d.index_price).groupby("strike")
    _c = _c.agg(mark_usd=("mark_usd", "median"), index=("index_price", "median"), vol=("iv_mark", "median"))
    _lo, _hi = _c.loc[61_000.0], _c.loc[61_500.0]
    _spread = _lo.mark_usd - _hi.mark_usd
    _k = pd.read_parquet(DERIVED / "iv_strikes.parquet", columns=["snap", "strike", "mid", "iv"],
                         filters=[("snap", "==", int(_snap.snap))]).set_index("strike")
    _k1, _k2 = _k.loc[61_000.0], _k.loc[61_500.0]
    example = pd.DataFrame([
        ("Price of the 61,000 contract / call", f"{_k1.mid * 100:.0f}¢", f"${_lo.mark_usd:,.2f}"),
        ("Price of the 61,500 contract / call", f"{_k2.mid * 100:.0f}¢", f"${_hi.mark_usd:,.2f}"),
        ("P(BTC above about 61,250)", f"{(_k1.mid + _k2.mid) / 2 * 100:.1f}¢ (average of the two)",
         f"(${_lo.mark_usd:,.2f} − ${_hi.mark_usd:,.2f}) ÷ $500 = {_spread / 500 * 100:.1f}¢"),
        ("Implied vol at 61,000", f"{_k1.iv:.1%} (method 2)", f"{_lo.vol:.1%} (mark)"),
    ], columns=["", "Kalshi", "Deribit"])
    _payoff = pd.DataFrame({"BTC on Friday": ["$60,000", "$61,250", "$62,000", "$70,000"],
                            "61,000 call pays": ["$0", "$250", "$1,000", "$9,000"],
                            "61,500 call (sold) costs": ["$0", "$0", "−$500", "−$8,500"],
                            "spread pays": ["$0", "$250", "$500", "$500"],
                            "per $1 (÷ 500)": ["0", "0.50", "1", "1"]})
    mo.vstack([
        mo.md(f"""
        ### One example by hand

        Monday June 29, 2026, around {_snap.ts:%H:%M} UTC, BTC near ${_snap.spot:,.0f}. Both contracts expire
        Friday July 3. Kalshi's "above 61,000" costs a few cents on the dollar; Deribit's 61,000 call costs hundreds
        of dollars, because it pays BTC − 61,000 rather than $1. To compare them, **buy the 61,000 call and sell the
        61,500 call**. That costs ${_spread:,.2f} and pays like a yes/no bet on BTC ending above about 61,250:
        """),
        mo.ui.table(_payoff, selection=None, show_download=False, pagination=False),
        mo.md("Divided by $500 it is a $1 bet, priced in cents like Kalshi's contracts:"),
        mo.ui.table(example, selection=None, show_download=False, pagination=False),
        mo.md(f"""
        So on prices, Deribit says about {_spread / 500 * 100:.0f}¢ where Kalshi says about
        {(_k1.mid + _k2.mid) / 2 * 100:.0f}¢; on vols, Kalshi is {100 * (_k1.iv - _lo.vol):+.1f} points above
        Deribit. The rest of this section does this at every Kalshi strike, every 15 minutes, every afternoon. (Here the Deribit marks are from up to 15 minutes before
        Kalshi's snapshot, and Deribit's option ends 13 hours earlier, so this single example is only illustrative.)
        """),
    ])
    return


@app.cell(hide_code=True)
def _(matched, mo, pd):
    _have = matched.deribit_vol.notna()
    _ba = matched.deribit_digital_adj.notna() & matched.bid.notna() & matched.ask.notna()
    coverage = pd.DataFrame([
        ("Matched Kalshi snapshots", f"{matched.ts.nunique():,} on {matched.ts.dt.date.nunique()} days, "
                                     f"{matched.event_ticker.nunique()} weeklies"),
        ("Deribit strikes per smile", f"median {matched.drop_duplicates('ts').n_deribit_strikes.median():.0f}"),
        ("Kalshi strikes in those snapshots", f"{len(matched):,}"),
        ("… inside Deribit's traded range (vol comparison)", f"{int(_have.sum()):,} ({_have.mean():.0%})"),
        ("… with a Deribit call spread and a Kalshi bid and ask (price comparison)", f"{int(_ba.sum()):,}"),
        ("Expiry adjustment (Deribit vol × ratio)",
         f"median {matched.drop_duplicates('ts').horizon_ratio.median():.3f}, "
         f"range {matched.horizon_ratio.min():.3f}–{matched.horizon_ratio.max():.3f}"),
    ], columns=["", "value"])
    mo.vstack([
        mo.md("""
        ### What could be matched

        Kalshi strikes outside the range Deribit traded in that 15-minute window are left out, so the far strikes
        are thinner here than in `03`. The expiry adjustment raises Deribit's vol by a few percent of itself (a
        ratio of 1.035 turns 45% into 46.6%), more near expiry.
        """),
        mo.ui.table(coverage, selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(H, MUTED, SLOT1, SLOT2, W, alt, matched, mo, pd):
    _x = matched.dropna(subset=["deribit_digital_adj", "deribit_digital_raw", "bid", "ask"]).copy()
    _x["gap_adj"] = 100 * (_x.mid - _x.deribit_digital_adj)
    _x["gap_raw"] = 100 * (_x.mid - _x.deribit_digital_raw)
    _by_m = (_x.groupby("moneyness_bucket", observed=True)
             .agg(strikes=("mid", "size"), kalshi_mid_c=("mid", lambda s: 100 * s.median()),
                  gap_not_stretched_c=("gap_raw", "median"), gap_stretched_c=("gap_adj", "median"),
                  kalshi_spread_c=("ask", lambda s: 100 * (s - _x.loc[s.index, "bid"]).median()))
             .round(1).reset_index())
    _long = _by_m.melt(id_vars=["moneyness_bucket"], value_vars=["gap_not_stretched_c", "gap_stretched_c"],
                       var_name="version", value_name="gap")
    _long["version"] = _long.version.map({"gap_not_stretched_c": "Deribit at its own expiry (08:00 UTC)",
                                         "gap_stretched_c": "Deribit stretched to Kalshi's expiry"})
    _long["moneyness_bucket"] = _long.moneyness_bucket.astype(str)
    _order = list(_by_m.moneyness_bucket.astype(str))
    price_gap_chart = (
        alt.Chart(_long).mark_bar()
        .encode(x=alt.X("moneyness_bucket:N", sort=_order, title="Kalshi strike relative to spot"),
                xOffset=alt.XOffset("version:N"),
                y=alt.Y("gap:Q", title="Kalshi mid − Deribit (cents, median)"),
                color=alt.Color("version:N", title=None, scale=alt.Scale(
                    domain=["Deribit at its own expiry (08:00 UTC)", "Deribit stretched to Kalshi's expiry"],
                    range=[SLOT2, SLOT1])),
                tooltip=["moneyness_bucket:N", "version:N", alt.Tooltip("gap:Q", format="+.1f")])
        .properties(title="Price gap across strikes, with and without the 13-hour stretch", width=W, height=H)
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED).encode(y="y:Q")
    )
    _half = 50 * (_x.ask - _x.bid)
    _summary = pd.DataFrame([
        (v, f"{_x[c].median():+.1f}¢", f"{_x[c].abs().median():.1f}¢", f"{(_x[c].abs() <= _half).mean():.0%}")
        for v, c in [("Deribit at its own expiry (not stretched)", "gap_raw"),
                     ("Deribit stretched to Kalshi's expiry", "gap_adj")]
    ], columns=["Deribit price used", "median gap", "typical size of gap", "within half of Kalshi's spread"])
    mo.vstack([
        mo.md(f"""
        ### Kalshi's prices vs Deribit's call spreads

        For every matched Kalshi strike K: Kalshi's mid price against the cost of Deribit's K − 250 / K + 250 call
        spread per $1, priced on Deribit's smile. Both are "the price of $1 if BTC ends above K", in cents. Two
        versions of Deribit's price:

        - **at its own expiry** (orange): Deribit's answer for BTC at **08:00 UTC** Friday, no stretch;
        - **stretched** (blue): Deribit's vols stretched to Kalshi's **5pm ET** expiry (section 2).
        """),
        mo.ui.table(_summary, selection=None, show_download=False, pagination=False),
        price_gap_chart,
        mo.md(f"""
        - **Without the stretch, the gap has a clear pattern**: Kalshi is higher for strikes above spot and lower for
          strikes below it. That is exactly what 13 extra hours produce: more time gives BTC more chance to climb
          above a high strike, and more chance to fall below a low one. Kalshi is answering a question with more
          time in it.
        - **With the stretch, most of that pattern goes away.** The typical gap is
          **{_x.gap_adj.abs().median():.1f}¢**, about the size of Kalshi's typical bid-ask spread
          ({2 * _half.median():.0f}¢).
        - What remains: Kalshi is a little **cheaper** for strikes at and just below spot and a little **dearer**
          above. Kalshi puts slightly more probability on BTC ending a bit lower, and on large rises, than Deribit.
        """),
        mo.ui.table(_by_m.rename(columns={"moneyness_bucket": "strike vs spot"}), selection=None, show_download=False,
                    pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(matched, mo, pd):
    _x = matched.dropna(subset=["deribit_digital_direct", "deribit_digital_raw", "deribit_digital_adj", "bid", "ask"])
    _versions = [("Direct: Deribit's own dollar prices, real strikes, no model", "deribit_digital_direct"),
                 ("Black-Scholes, Deribit's expiry (not stretched)", "deribit_digital_raw"),
                 ("Black-Scholes, stretched to Kalshi's expiry", "deribit_digital_adj")]
    _rows = []
    for _name, _c in _versions:
        _g = 100 * (_x.mid - _x[_c])
        _rows.append((_name, f"{_g.median():+.1f}¢", f"{_g.abs().median():.1f}¢",
                      f"{((_x[_c] < _x.bid) | (_x[_c] > _x.ask)).mean():.0%}"))
    three_way = pd.DataFrame(_rows, columns=["Deribit price used", "Kalshi − Deribit (median)", "typical size of gap",
                                              "Deribit outside Kalshi's bid-ask"])
    _by_m = (_x.groupby("moneyness_bucket", observed=True)
             .apply(lambda g: pd.Series({"cases": len(g),
                                         "direct": 100 * (g.mid - g.deribit_digital_direct).median(),
                                         "Black-Scholes, not stretched": 100 * (g.mid - g.deribit_digital_raw).median(),
                                         "Black-Scholes, stretched": 100 * (g.mid - g.deribit_digital_adj).median()}))
             .round(1).reset_index().rename(columns={"moneyness_bucket": "strike vs spot"}))
    _bs_vs_direct = 100 * (_x.deribit_digital_direct - _x.deribit_digital_raw).abs().median()
    mo.vstack([
        mo.md(f"""
        ### Without any model: Deribit's own prices

        Does Black-Scholes drive these results? To check, take Deribit's prices exactly as they are: for a Kalshi
        strike K, the dollar mark prices of Deribit's **K − 500** and **K + 500** calls (real strikes, the trade nearest
        in time to the Kalshi snapshot), and divide their difference by $1,000. No vols, no interpolation, no stretch.
        This only works where Deribit has calls at both strikes within the 15 minutes: **{len(_x):,} cases** in
        {_x.ts.nunique():,} snapshots. All three versions below are on those same cases.
        """),
        mo.ui.table(three_way, selection=None, show_download=False, pagination=False),
        mo.md(f"""
        - **The direct prices and the unstretched Black-Scholes prices tell the same story** (they differ by a typical
          {_bs_vs_direct:.1f}¢, mostly because the direct marks are from a few minutes before or after the snapshot,
          while BTC moves). So Black-Scholes is not creating the result; it only lets us line things up.
        - **The direct comparison cannot be stretched**: it is Deribit's answer for 08:00 UTC, and no Deribit price
          exists for 5pm ET. Its gap carries the 13-hour pattern (Kalshi higher above spot, lower below).
        - **Stretching is what closes most of the gap.** That is why the main results use it, and why it is the
          assumption that most deserves a check (for example with Deribit's Saturday expiry).
        """),
        mo.ui.table(_by_m, selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(matched, mo):
    _days = sorted(matched.ts.dt.date.unique())
    _opts = {f"{d:%a %Y-%m-%d}": d for d in _days}
    example_day = mo.ui.dropdown(options=_opts, value="Mon 2026-06-29", label="Day")
    return (example_day,)


@app.cell(hide_code=True)
def _(H, MUTED, SLOT1, SLOT2, W, alt, compare, example_day, matched, mo, np, pd, smiles):
    _day = matched[matched.ts.dt.date == example_day.value]
    _ts = _day.ts.iloc[((_day.ts.dt.hour * 60 + _day.ts.dt.minute) - (20 * 60 + 30)).abs().argmin()]
    _snap = int(_day[_day.ts == _ts].snap.iloc[0])
    _s = smiles[_snap]
    _k = _day[(_day.ts == _ts) & _day.mid.between(0.02, 0.98)]
    _grid = np.arange(_k.strike.min(), _k.strike.max() + 1, 100.0)
    _curve = pd.DataFrame({"strike": _grid,
                           "price": compare.digital_from_smile(_grid, _s["F"], _s["tau_k"], _s["smile"].log_moneyness,
                                                              _s["smile"].vol * _s["ratio"])}).dropna()
    _x = alt.X("strike:Q", title="strike", scale=alt.Scale(zero=False), axis=alt.Axis(format=",.0f"))
    ladder_chart = (
        alt.Chart(_k).mark_rule(color=MUTED, strokeWidth=2).encode(x=_x, y=alt.Y("bid:Q"), y2="ask:Q")
        + alt.Chart(_k).mark_point(filled=True, size=40, color=SLOT1)
        .encode(x=_x, y=alt.Y("mid:Q", title="price of $1 if BTC ends above the strike", axis=alt.Axis(format=".0%")),
                tooltip=[alt.Tooltip("strike:Q", format=","), alt.Tooltip("bid:Q", format=".2f"),
                         alt.Tooltip("ask:Q", format=".2f"), alt.Tooltip("deribit_digital_adj:Q", format=".3f", title="Deribit")])
        + alt.Chart(_curve).mark_line(color=SLOT2, strokeWidth=2).encode(x=_x, y="price:Q")
    ).properties(title=f"{example_day.value:%a %b %d}, {_ts:%H:%M} UTC: Kalshi ladder (blue, gray = bid to ask) "
                       f"vs Deribit call spreads (orange)", width=W, height=H + 60)
    _in = _k.dropna(subset=["deribit_digital_adj", "bid", "ask"])
    _inside = ((_in.deribit_digital_adj >= _in.bid) & (_in.deribit_digital_adj <= _in.ask)).mean()
    mo.vstack([
        mo.md(f"""
        **One snapshot up close.** Each blue dot is a Kalshi contract with its bid-ask range in gray; the orange line
        is Deribit's price for the same contract. Where the orange line runs through the gray bars, the two markets
        agree within Kalshi's spread. On this snapshot that holds for {_inside:.0%} of strikes. Pick another day:
        """),
        example_day,
        ladder_chart,
    ])
    return


@app.cell(hide_code=True)
def _(compare, matched, mo, np, pd):
    _y = matched.dropna(subset=["deribit_digital_adj", "bid", "ask"]).copy()
    # Sell Kalshi at the bid and buy Deribit's spread, or buy Kalshi at the ask and sell Deribit's spread.
    # Edge per $1 contract after Kalshi's taker fee, before Deribit's costs.
    _sell = _y.bid - compare.kalshi_fee(_y.bid) - _y.deribit_digital_adj
    _buy = _y.deribit_digital_adj - _y.ask - compare.kalshi_fee(_y.ask)
    _y["edge_c"] = 100 * np.maximum(_sell, _buy)
    _outside = ((_y.deribit_digital_adj < _y.bid) | (_y.deribit_digital_adj > _y.ask)).mean()
    _raw = _y.dropna(subset=["deribit_digital_raw"])
    _outside_raw = ((_raw.deribit_digital_raw < _raw.bid) | (_raw.deribit_digital_raw > _raw.ask)).mean()
    _edge_raw = 100 * np.maximum(_raw.bid - compare.kalshi_fee(_raw.bid) - _raw.deribit_digital_raw,
                                 _raw.deribit_digital_raw - _raw.ask - compare.kalshi_fee(_raw.ask))
    edges = pd.DataFrame([(c, f"{(_y.edge_c > c).mean():.1%}", round(_y.edge_c[_y.edge_c > c].median() - c, 1),
                           f"{(_edge_raw > c).mean():.1%}") for c in (0, 2, 5, 10)],
                         columns=["assumed Deribit cost (¢ per $1)", "share still profitable (stretched)",
                                  "median edge left (¢, stretched)", "share still profitable (not stretched)"])
    _by_t = (_y.groupby("tte_bucket", observed=True).edge_c.apply(lambda e: (e > 5).mean())
             .map("{:.1%}".format).rename("share profitable at 5¢ Deribit cost").reset_index()
             .rename(columns={"tte_bucket": "Kalshi time to expiry"}))
    mo.vstack([
        mo.md(f"""
        ## 7 · Is there money on the table?

        A gap only matters if you could trade on it. If Deribit's price is **below Kalshi's bid**, you could sell
        on Kalshi and buy the call spread on Deribit; if it is **above Kalshi's ask**, the reverse. Each trade pays
        Kalshi's fee (assumed {compare.KALSHI_FEE_RATE:.0%} × P × (1 − P) per contract, rounded up to the cent;
        check the current schedule) and Deribit's costs.

        - Deribit's price lies **outside Kalshi's bid-ask** for {_outside:.0%} of quotes ({_outside_raw:.0%} without the
          expiry adjustment).
        - After Kalshi's fee, an edge remains for {(_y.edge_c > 0).mean():.0%} of quotes, and where it does it is
          typically {_y.edge_c[_y.edge_c > 0].median():.1f}¢.
        - **Deribit's costs are what decide it, and we do not observe them directly** (we have trades and marks, not
          quotes). Replicating a $1 contract needs two Deribit options for 1/500 of a bitcoin each, and every leg pays
          a fee and half the bid-ask spread. For example, $15 of costs per leg per bitcoin is 2 × 15 / 500 = 6¢ per $1
          contract. The table shows what survives under different assumptions, with Deribit stretched to Kalshi's
          expiry and, for comparison, not stretched. Without the stretch many more quotes look profitable, but those
          "edges" are mostly the 13 extra hours, not mispricing.
        """),
        mo.ui.table(edges, selection=None, show_download=False, pagination=False),
        mo.md("""
        What survives is concentrated in the last days, where the expiry adjustment (a model, not a trade) does the
        most work and the hedge is weakest: the Deribit leg expires 13 hours before the Kalshi leg, so the position
        is exposed to Friday's US session. That is risk, not arbitrage.
        """),
        mo.ui.table(_by_t, selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(compare, matched, mo, np, pairs, paired):
    _a = pairs.atm(paired)
    _t = pairs.gap_table(_a.assign(all=1), "all").iloc[0]
    _by = pairs.gap_table(paired, "sd_bucket").assign(sd=lambda d: d.sd_bucket.astype(float)).set_index("sd")
    _tt = _a.groupby("tte_bucket", observed=True).gap_adj.median()
    _week, _last, _dn = _tt.get("5–7 days", float("nan")), _tt.get("1–2 days", float("nan")), _by.loc[-2.0].median_gap
    _p = matched.dropna(subset=["deribit_digital_adj", "bid", "ask"])
    _edge = 100 * np.maximum(_p.bid - compare.kalshi_fee(_p.bid) - _p.deribit_digital_adj,
                             _p.deribit_digital_adj - _p.ask - compare.kalshi_fee(_p.ask))
    mo.md(f"""
    ## 8 · Main takeaways

    **Q2b, are Kalshi's prices consistent with option prices? Broadly yes, with a richer centre and flatter wings.**

    1. **Vol at the money: Kalshi about {_t.median_gap:.1f} points above Deribit** (95% interval {_t.ci_low:+.1f} to
       {_t.ci_high:+.1f}), from {len(paired):,} pairs taken at most 30 seconds apart, after aligning expiries. Kalshi is
       higher in {_t.kalshi_higher:.0%} of pairs, and the two move together day by day.
    2. **Deribit's smile is steeper.** Two standard deviations below the money Kalshi is
       {_dn:+.1f} points: Deribit traders pay more for crash protection (and, less clearly,
       for large rises).
    3. **The expiry mismatch matters, and the adjustment is not the last word.** Without it the gap at the money is
       {_a.gap_raw.median():+.1f} points. With it, the gap still grows toward expiry (from {_week:+.1f} a week out to
       {_last:+.1f} with 1–2 days left). A market-based alternative, interpolating between
       Deribit's Friday and Saturday 08:00 expiries, would test it.
    4. **Prices agree to a cent or two**: Kalshi's price and Deribit's call spread differ by a typical
       {(100 * (_p.mid - _p.deribit_digital_adj)).abs().median():.1f}¢, about Kalshi's own bid-ask spread.
    5. **There is little free money.** With an assumed 5¢ of Deribit costs per $1 contract, {(_edge > 5).mean():.0%}
       of Kalshi quotes still show an edge, mostly near expiry, where the Deribit leg ends 13 hours early (risk, not
       arbitrage).
    6. **Why gaps can persist at all:** different users (US retail on Kalshi; professional and offshore on Deribit,
       which does not serve US residents), so little capital moves between them; Kalshi's full collateral and fees;
       Kalshi's 1¢ tick and $500 strikes.

    **Caveats.** The pairs only exist where Deribit trades, so the far wings rest on few pairs. Deribit marks are
    Deribit's model values, not executable prices, and Deribit's trading costs are assumed. The 13-hour adjustment uses
    BTC's past hour-by-hour movement, not what the market expected that Friday.

    **Next.** Deribit option-chain quotes (Tardis.dev or similar) would give Deribit's full smile every minute, for
    prices as well as vols. Test the expiry adjustment with Deribit's Saturday expiry. Extend to probabilities for Q3
    (Kalshi vs Deribit vs realized outcomes).
    """)
    return


if __name__ == "__main__":
    app.run()
