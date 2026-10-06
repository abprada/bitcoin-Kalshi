import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="05 Kalshi vs Deribit")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    from lib import compare, deribit

    return alt, compare, deribit, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # 05 · Kalshi vs Deribit

    Handout Q2b: **are Kalshi's prices consistent with option prices?** Kalshi sells "$1 if BTC ends above K";
    Deribit sells calls and puts. They pay different things, so we first put them in the same units, in two ways:

    1. **Prices, as probabilities.** A tight Deribit call spread pays like a Kalshi contract (handout Q1c):
       buy the K − 250 call, sell the K + 250 call, divide by $500. Its cost is Deribit's price for Kalshi's
       contract, in cents, directly comparable with Kalshi's bid and ask.
    2. **Implied vols.** Both markets turned into Black-Scholes implied vol (Kalshi through method 2 in `03`).
       A vol gap means the same thing at every strike and every horizon, so it is the easier summary.

    **Three alignment problems** and how they are handled:

    - **Time.** Deribit data are trades, not continuous quotes. Each Kalshi snapshot (every 15 minutes, 19:30–22:29
      UTC) is paired with the Deribit trades in the matching expiry within **15 minutes** of it. Their *mark* prices
      (Deribit's fair value, see `04`) are turned into a smile: vol against moneyness ln(strike / index). Working in
      moneyness means a small BTC move between the trade and the snapshot does not matter.
    - **Strikes.** Deribit strikes are round thousands, Kalshi's are $500 apart. Deribit's smile is interpolated to
      each Kalshi strike, and only **inside** the range of strikes Deribit traded (no extrapolation).
    - **Expiry.** Deribit's Friday option expires at 08:00 UTC, Kalshi's at 5pm ET, about 13 hours later. Deribit's
      vol is stretched to Kalshi's horizon with the hour-of-week variance profile from `04` (the extra Friday hours are
      busier than average); section 2 walks through it. Results are shown **with and without** this adjustment,
      because it is a model.

    The code is in `lib/compare.py` and `lib/deribit.py`, tested in `tests/test_compare.py` and `tests/test_deribit.py`.
    Needs `03_iv_surface.py` to have run. Writes nothing.
    """)
    return


@app.cell
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


@app.cell
def _(compare, mo):
    # All the matching lives in lib/compare.py (build_comparison), shared with 06_results_summary.py
    ROOT = mo.notebook_dir()
    DERIVED = ROOT / "data" / "derived"
    _c = compare.build_comparison(ROOT)
    matched, smiles, snaps, trades, hour_profile = _c["matched"], _c["smiles"], _c["snaps"], _c["trades"], _c["hour_profile"]
    return DERIVED, hour_profile, matched, smiles, snaps, trades


@app.cell
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
        ## 1 · One example by hand

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
        Deribit. The rest of this notebook does this at every Kalshi strike, every 15 minutes, every afternoon, with
        the three alignment fixes from the introduction. (Here the Deribit marks are from up to 15 minutes before
        Kalshi's snapshot, and Deribit's option ends 13 hours earlier, so this single example is only illustrative.)
        """),
    ])
    return


@app.cell
def _(H, SLOT1, SLOT2, W, alt, compare, deribit, hour_profile, matched, mo, np, pd, smiles):
    # Walk through the 13-hour adjustment on the worked example's snapshot (Monday June 29, 20:30 UTC)
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
        ## 2 · How the 13 extra hours are handled

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
           This is the "stretched" Deribit vol compared with Kalshi (`horizon_vol_ratio`).
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


@app.cell
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
        ## 3 · What could be matched

        Kalshi strikes outside the range Deribit traded in that 15-minute window are left out, so the far strikes
        are thinner here than in `03`. The expiry adjustment raises Deribit's vol by a few percent of itself (a
        ratio of 1.035 turns 45% into 46.6%), more near expiry.
        """),
        mo.ui.table(coverage, selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell
def _(H, MUTED, SLOT1, SLOT2, W, alt, matched, mo, pd):
    _x = matched.dropna(subset=["vol_gap_raw"])
    _by_m = (_x.groupby("moneyness_bucket", observed=True)
             .agg(strikes=("iv", "size"), kalshi_vol=("iv", "median"), deribit_vol=("deribit_vol", "median"),
                  gap_raw=("vol_gap_raw", "median"), gap_adjusted=("vol_gap_adj", "median"))
             .reset_index())
    _long = _by_m.melt(id_vars=["moneyness_bucket"], value_vars=["gap_raw", "gap_adjusted"], var_name="version",
                       value_name="gap")
    _long["version"] = _long.version.map({"gap_raw": "Deribit as traded",
                                         "gap_adjusted": "Deribit stretched to Kalshi's expiry"})
    _order = list(_by_m.moneyness_bucket.astype(str))
    _long["moneyness_bucket"] = _long.moneyness_bucket.astype(str)
    vol_gap_chart = (
        alt.Chart(_long).mark_line(point=True)
        .encode(x=alt.X("moneyness_bucket:N", sort=_order, title="Kalshi strike relative to spot"),
                y=alt.Y("gap:Q", title="Kalshi − Deribit (vol points, median)"),
                color=alt.Color("version:N", title=None, scale=alt.Scale(
                    domain=["Deribit as traded", "Deribit stretched to Kalshi's expiry"], range=[SLOT2, SLOT1])),
                tooltip=["moneyness_bucket:N", "version:N", alt.Tooltip("gap:Q", format="+.1f")])
        .properties(title="Vol gap across strikes", width=W, height=H)
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    )
    _atm = _x[_x.log_moneyness.abs() < 0.005]
    _by_t = (_atm.groupby("tte_bucket", observed=True)
             .agg(strikes=("iv", "size"), gap_raw=("vol_gap_raw", "median"), gap_adjusted=("vol_gap_adj", "median"))
             .round(1).reset_index().rename(columns={"tte_bucket": "Kalshi time to expiry"}))
    _bm = _by_m.assign(kalshi_vol=lambda d: (100 * d.kalshi_vol).round(1), deribit_vol=lambda d: (100 * d.deribit_vol).round(1),
                       gap_raw=lambda d: d.gap_raw.round(1), gap_adjusted=lambda d: d.gap_adjusted.round(1))
    _bm = _bm.rename(columns={"moneyness_bucket": "strike vs spot", "kalshi_vol": "Kalshi vol %", "deribit_vol": "Deribit vol %"})
    _near = _by_m[_by_m.moneyness_bucket == "within 0.5%"].iloc[0]
    _far_dn = _by_m[_by_m.moneyness_bucket == "5–15% below"].iloc[0]
    mo.vstack([
        mo.md(f"""
        ## 4 · Comparing vols

        For every matched Kalshi strike: Kalshi's vol minus Deribit's vol at the same moneyness. Positive means
        Kalshi prices more movement.

        - **At the money**, Kalshi is **{_near.gap_raw:+.1f}** points above Deribit as traded and
          **{_near.gap_adjusted:+.1f}** after stretching Deribit to Kalshi's expiry. (`04` found about 4 points with a
          rougher method: trade prices instead of marks, one median per day.) Kalshi is above Deribit at the money
          on {(_atm.vol_gap_adj > 0).mean():.0%} of matched strikes after the adjustment.
        - **Far below spot** the sign flips: Kalshi is **{_far_dn.gap_adjusted:+.1f}** points (adjusted) relative to
          Deribit. Deribit charges more for crash protection than Kalshi does, so Deribit's smile is steeper on
          the downside.
        - Above spot, the two are close after the adjustment.
        """),
        vol_gap_chart,
        mo.ui.table(_bm, selection=None, show_download=False, pagination=False),
        mo.md("**At the money, by time to expiry.** The adjustment matters most near expiry, where 13 hours are a larger share of what is left:"),
        mo.ui.table(_by_t, selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell
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
        ## 5 · Comparing prices

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


@app.cell
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


@app.cell
def _(matched, mo):
    _days = sorted(matched.ts.dt.date.unique())
    _opts = {f"{d:%a %Y-%m-%d}": d for d in _days}
    example_day = mo.ui.dropdown(options=_opts, value="Mon 2026-06-29", label="Day")
    return (example_day,)


@app.cell
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


@app.cell
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
        ## 6 · Is there money on the table?

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


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Main takeaways

    **Q2b, are Kalshi's prices consistent with option prices? Broadly yes.**

    1. **The two markets agree closely.** Put in the same units, Kalshi's prices and Deribit's call-spread prices
       typically differ by about 1.5¢, the size of Kalshi's own bid-ask spread. Their vols move together day to day.
    2. **The expiry mismatch matters and has to be handled.** Kalshi settles 13 hours after Deribit, during
       Friday's busy US session. Ignoring this makes Kalshi look about 3 vol points too high at the money; handling
       it leaves about **1 point**, with Kalshi higher on roughly two days in three.
    3. **The main real difference is the shape.** Far below the current price, Deribit's vol is about 3 points
       above Kalshi's: Deribit traders pay more for crash protection than Kalshi traders do.
    4. **There is little free money.** Gaps beyond Kalshi's bid-ask are mostly smaller than the cost of trading
       both venues. What survives plausible costs is concentrated in the last days, where the hedge expires 13
       hours early, which is risk rather than arbitrage.
    5. **Why gaps can persist at all:** different users (US retail on Kalshi; professional and offshore on Deribit,
       which does not serve US residents), so little capital moves between them; Kalshi's full collateral and fees;
       Kalshi's 1¢ tick and $500 strikes.

    **Caveats.** The 13-hour adjustment uses BTC's past hour-by-hour movement, not what the market expected that
    Friday. Deribit marks are Deribit's model values, not executable prices, and Deribit's trading costs are
    assumed. The 15-minute matching window and Kalshi's fee formula are assumptions worth varying.

    **Next.** Check robustness (window length, the `exact_0800UTC` window, trade prices instead of marks); get
    Deribit order-book quotes if possible, to measure its real costs; extend to probabilities for Q3 (Kalshi vs
    Deribit vs realized outcomes).
    """)
    return


if __name__ == "__main__":
    app.run()
