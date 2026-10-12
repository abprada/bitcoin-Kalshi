import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="06 Results summary")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    from lib import pairs

    return alt, mo, np, pairs, pd


@app.cell(hide_code=True)
def _(alt):
    # Chart styling: reference palette (slot 1 blue = Kalshi / stretched, slot 2 orange = Deribit / not stretched).
    INK, INK2, MUTED, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
    SLOT1, SLOT2 = "#2a78d6", "#eb6834"
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

    return H, MUTED, SLOT1, SLOT2, W


@app.cell(hide_code=True)
def _(mo, np, pairs, pd):
    ROOT = mo.notebook_dir()
    # every Deribit trade paired with Kalshi's vol at the same strike and minute (lib/pairs.py, about 90 seconds)
    _r = pairs.build_pairs(ROOT)
    paired, hour_profile = _r["pairs"], _r["hour_profile"]
    kalshi_weeks = pd.read_parquet(ROOT / "data" / "derived" / "iv_snapshots.parquet", columns=["event_ticker"]).event_ticker.nunique()

    # headline numbers, all computed here
    _atm = pairs.atm(paired)
    _far = pairs.far_from_expiry(paired)
    _atm_far = pairs.atm(_far)
    _head = pairs.gap_table(_atm.assign(all=1), "all").iloc[0]
    _far_raw = pairs.gap_table(_atm_far.assign(all=1), "all", col="gap_raw").iloc[0]
    _far_adj = pairs.gap_table(_atm_far.assign(all=1), "all").iloc[0]
    _by_sd = pairs.gap_table(paired, "sd_bucket").assign(sd=lambda d: d.sd_bucket.astype(float)).set_index("sd")
    _far_sd = pairs.gap_table(_far, "sd_bucket", col="gap_raw").assign(sd=lambda d: d.sd_bucket.astype(float)).set_index("sd")
    _by_tte = _atm.groupby("tte_bucket", observed=True).gap_adj.median()
    daily = (_atm.groupby("day")[["kalshi_iv", "deribit_iv_adj"]].median().reset_index()
             .rename(columns={"kalshi_iv": "kalshi_atm", "deribit_iv_adj": "deribit_atm_adj"})
             .assign(day=lambda d: pd.to_datetime(d.day)))
    stats = dict(
        pairs=len(paired), pair_days=paired.day.nunique(), pair_weeklies=paired.event_ticker.nunique(),
        deribit_trades=len(_r["trades"]),
        atm_gap_raw=_atm.gap_raw.median(), atm_gap_adj=_head.median_gap, atm_ci_low=_head.ci_low,
        atm_ci_high=_head.ci_high, atm_kalshi_higher=_head.kalshi_higher,
        far_gap_adj=_by_sd.loc[-2.0, "median_gap"], far_up_adj=_by_sd.loc[2.0, "median_gap"],
        week_gap_adj=_by_tte.get("5–7 days", np.nan), last_gap_adj=_by_tte.get("1–2 days", np.nan),
        far_pairs=len(_far), far_days=_far.day.nunique(), far_share=13 / _far.kalshi_hours.median(),
        far_raw=_far_raw.median_gap, far_raw_lo=_far_raw.ci_low, far_raw_hi=_far_raw.ci_high,
        far_adj=_far_adj.median_gap, far_adj_lo=_far_adj.ci_low, far_adj_hi=_far_adj.ci_high,
        far_dn_raw=_far_sd.loc[-2.0, "median_gap"],
        far_ratio_lo=_far.horizon_ratio.min(), far_ratio_hi=_far.horizon_ratio.max(),
        corr=daily[["kalshi_atm", "deribit_atm_adj"]].corr().iloc[0, 1],
        ratio_lo=paired.horizon_ratio.min(), ratio_hi=paired.horizon_ratio.max(),
    )
    return daily, hour_profile, kalshi_weeks, paired, stats


@app.cell(hide_code=True)
def _(mo, stats):
    mo.md(f"""
    # Kalshi vs Deribit: do Bitcoin prediction-market prices agree with options?

    *15.451 Proseminar · results summary · data October 2025 – September 2026*

    **Question.** Kalshi sells weekly contracts that pay **$1 if Bitcoin ends above a price K** at 5pm ET on Friday.
    Deribit, the largest Bitcoin options exchange, sells ordinary calls and puts. Are the two markets' prices
    consistent (Q2b)? We compare them in **implied volatility**, the expected size of Bitcoin's moves that each
    market's prices imply.

    **Answer: broadly yes, with Kalshi slightly richer at the current price and flatter in the wings.**

    1. **At the current price, Kalshi's implied vol is about two points above Deribit's.** Comparing every Deribit
       trade with Kalshi at the same strike and minute ({stats['pairs']:,} pairs), Kalshi is
       **{stats['atm_gap_adj']:+.1f} points** above (95% interval {stats['atm_ci_low']:+.1f} to {stats['atm_ci_high']:+.1f};
       higher {stats['atm_kalshi_higher']:.0%} of the time). The two move together day to day (correlation
       **{stats['corr']:.2f}**).
    2. **Far from expiry, where the timing adjustment barely matters, the conclusion holds.** With 4–7 days left,
       Kalshi is **{stats['far_adj']:+.1f} points** above with the adjustment and **{stats['far_raw']:+.1f}** without
       it: higher either way.
    3. **The clear difference is crash protection.** Two standard deviations below the current price, Deribit's vol
       is about **{-stats['far_gap_adj']:.0f} points higher** than Kalshi's: Deribit traders pay more for downside
       insurance, so Deribit's smile is steeper.
    4. **Getting the timing right is essential.** Deribit's options expire 13 hours before Kalshi's. Ignoring that
       makes Kalshi look {stats['atm_gap_raw']:.1f} vol points higher instead of {stats['atm_gap_adj']:.1f}. Close to
       expiry the result depends on how those hours are handled ({stats['week_gap_adj']:+.1f} a week out,
       {stats['last_gap_adj']:+.1f} with 1–2 days left), which is why point 2 matters.
    """)
    return


@app.cell(hide_code=True)
def _(kalshi_weeks, mo, pd, stats):
    _data = pd.DataFrame([
        ("Kalshi", "Weekly BTC contracts (KXBTCD): '$1 if BTC > K at 5pm ET Friday'",
         f"{kalshi_weeks} weeks, Oct 2025 – Sep 2026, ~54 strikes $500 apart, 1-minute bid/ask"),
        ("Deribit", "BTC options (calls and puts), all expiring 08:00 UTC",
         f"{stats['deribit_trades']:,} trades in Friday expiries, 19:30–22:29 UTC, Jan – Sep 2026, with Deribit's mark price"),
        ("Bitcoin price", "Coinbase and Bitstamp, average", "every minute; within 0.015% of Kalshi's settlement index"),
        ("Pairs", "every Deribit trade with Kalshi's vol at the same strike, at most 30 seconds apart",
         f"{stats['pairs']:,} pairs on {stats['pair_days']} days ({stats['pair_weeklies']} weeks)"),
    ], columns=["Source", "What", "Coverage"])
    mo.vstack([mo.md("## Data"), mo.ui.table(_data, selection=None, show_download=False, pagination=False)])
    return


@app.cell(hide_code=True)
def _(mo, stats):
    mo.md(rf"""
    ## Method

    **Step 1. Turn Kalshi's prices into implied volatility** (notebook `03`). A Kalshi contract pays $1 above K, so its
    price D(K) is the market's probability. Breeden and Litzenberger (1978): D(K) = −∂C/∂K, so **adding up Kalshi's
    prices across strikes gives call and put prices**, C(K) = ∫ above K of D. Beyond the last strike, the tails take a
    Black-Scholes shape bounded by Kalshi's own outer quotes. Put-call parity gives Kalshi's forward, and solving
    Black-Scholes for each option gives one implied vol per strike, at any minute. (Solving N(d₂) = price for
    each contract directly is too noisy near the current price and biased low by the smile, so it is not used.)

    **Step 2. Line Kalshi up with Deribit.** Kalshi quotes every strike every minute; our Deribit data are individual
    trades, 19:30–22:29 UTC, at whatever strikes people trade. Three mismatches, each handled:

    | Mismatch | Fix |
    |---|---|
    | **Time** | Every Deribit trade is paired with Kalshi's ladder at the nearest minute, **at most 30 seconds apart**. Deribit's vol comes from its mark price (Deribit's fair value at that instant); fills at one strike in one minute count once |
    | **Strikes** | Kalshi's vol is read at the Deribit trade's own strike: on Kalshi's ladder 3 times in 4, otherwise between the two neighbouring Kalshi strikes. Nothing is interpolated on the Deribit side |
    | **Expiry:** Deribit ends 08:00 UTC, Kalshi 13 hours later | Deribit's vol is **stretched** over the extra hours. Variance adds up hour by hour, and those 13 hours (Friday's US session) historically move 1.6× an average hour, so Deribit's vol is scaled up by ×{stats['ratio_lo']:.2f} (a week out) to ×{stats['ratio_hi']:.2f} (a day out). Results are shown with and without this step |

    **Step 3. Compare.** For each pair, the **gap** = Kalshi's vol − Deribit's vol, in vol points (+1 = one percentage
    point higher on Kalshi). Pairs are pooled two ways:

    - **Distance from the current price, in standard deviations**:
      $\text{{sd}} = \ln(K / F) \,/\, (\sigma_{{ATM}} \sqrt{{\tau}})$, with K the strike, F Kalshi's forward (the
      average Friday price implied by Kalshi's ladder), σ_ATM Kalshi's at-the-money vol and τ Kalshi's time to expiry
      in years, all at that minute. σ_ATM √τ is Kalshi's typical move to Friday 5pm, so sd counts typical moves; it
      puts 1-day and 7-day contracts, and calm and volatile weeks, on one scale. The same Kalshi-based sd is used for
      both markets in a pair. **At the money** = sd between −0.25 and +0.25; the charts use half-SD buckets.
    - **Time to expiry**, in Kalshi's hours left (1–2, 2–3, 3–5, 5–7 days).

    Each bucket reports the **median** gap (robust to a few extreme pairs in the wings), with a **95% interval** from
    resampling whole days 500 times, because pairs on the same day are not independent. As a check that needs almost
    no expiry adjustment, the comparison is repeated **far from expiry** (at least 4 days left), where the 13 hours are
    about {stats['far_share']:.0%} of the horizon. Full definitions, with a worked example, are in the appendix of
    `05_kalshi_vs_deribit.py`.
    """)
    return


@app.cell(hide_code=True)
def _(H, SLOT1, SLOT2, W, alt, daily, mo, stats):
    _long = daily.melt(id_vars=["day"], value_vars=["kalshi_atm", "deribit_atm_adj"], var_name="market", value_name="vol")
    _long["market"] = _long.market.map({"kalshi_atm": "Kalshi", "deribit_atm_adj": "Deribit (stretched)"})
    fig1 = (
        alt.Chart(_long).mark_line(strokeWidth=1.6)
        .encode(x=alt.X("day:T", title=None, axis=alt.Axis(format="%b %Y")),
                y=alt.Y("vol:Q", title="implied vol at the current price", axis=alt.Axis(format="%"),
                        scale=alt.Scale(zero=False)),
                color=alt.Color("market:N", title=None, scale=alt.Scale(domain=["Kalshi", "Deribit (stretched)"],
                                                                        range=[SLOT1, SLOT2])),
                tooltip=[alt.Tooltip("day:T", format="%Y-%m-%d"), "market:N", alt.Tooltip("vol:Q", format=".1%")])
        .properties(title="Figure 1 · Implied vol at the current price, daily median of the pairs, Kalshi vs Deribit",
                    width=W, height=H)
    )
    mo.vstack([
        mo.md(f"""
        ## Results

        **Figure 1: the two markets price the same amount of movement.** Day by day, Kalshi's and Deribit's implied
        vols at the current price rise and fall together (correlation **{stats['corr']:.2f}** over
        {len(daily)} days), including the spikes in February and June 2026. Kalshi sits slightly above, on
        {(daily.kalshi_atm > daily.deribit_atm_adj).mean():.0%} of days.
        """),
        fig1,
    ])
    return


@app.cell(hide_code=True)
def _(H, MUTED, SLOT1, SLOT2, W, alt, mo, pairs, paired, pd, stats):
    _t = pd.concat([pairs.gap_table(paired, "sd_bucket", col=c).assign(version=v)
                    for c, v in [("gap_raw", "Deribit at its own expiry"), ("gap_adj", "Deribit stretched to Kalshi's expiry")]])
    _t["sd"] = _t.sd_bucket.astype(float)
    _t = _t[_t.pairs >= 100]
    _color = alt.Color("version:N", title=None, scale=alt.Scale(
        domain=["Deribit at its own expiry", "Deribit stretched to Kalshi's expiry"], range=[SLOT2, SLOT1]))
    _x = alt.X("sd:Q", title="strike, in standard deviations from the current price (negative = below)")
    fig2 = (
        alt.Chart(_t).mark_area(opacity=0.15).encode(x=_x, y="ci_low:Q", y2="ci_high:Q", color=_color)
        + alt.Chart(_t).mark_line(point=True).encode(
            x=_x, y=alt.Y("median_gap:Q", title="Kalshi − Deribit (vol points)"), color=_color,
            tooltip=["version:N", alt.Tooltip("sd:Q", title="SD"), alt.Tooltip("median_gap:Q", format="+.1f"), "pairs:Q"])
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    ).properties(title="Figure 2 · Implied vol gap across the smile, all pairs (median; bands: 95% intervals)",
                 width=W, height=H)
    mo.vstack([
        mo.md(f"""
        **Figure 2: Kalshi is richer near the current price; Deribit charges more for large moves.** Each pair compares
        the two vols at the same strike and minute. At the current price, Kalshi is {stats['atm_gap_raw']:+.1f} points
        above Deribit as traded and **{stats['atm_gap_adj']:+.1f}** after stretching Deribit to Kalshi's expiry. Two
        standard deviations below, the gap turns clearly negative (**{stats['far_gap_adj']:+.1f}** points); two above,
        it is {stats['far_up_adj']:+.1f}. Deribit's smile is steeper, above all on the downside.
        """),
        fig2,
    ])
    return


@app.cell(hide_code=True)
def _(H, MUTED, SLOT1, SLOT2, W, alt, hour_profile, mo, pairs, paired, pd, stats):
    _far = pairs.far_from_expiry(paired)
    _versions = [("gap_raw", "no stretch: Deribit as traded"), ("gap_adj", "stretched")]
    _t = pd.concat([pairs.gap_table(_far, "sd_bucket", col=c).assign(version=v) for c, v in _versions])
    _t["sd"] = _t.sd_bucket.astype(float)
    _t = _t[_t.pairs >= 100]
    _color = alt.Color("version:N", title=None, scale=alt.Scale(domain=[v for _, v in _versions], range=[SLOT2, SLOT1]))
    _x = alt.X("sd:Q", title="strike, in standard deviations from the current price (negative = below)")
    fig3 = (
        alt.Chart(_t).mark_area(opacity=0.15).encode(x=_x, y="ci_low:Q", y2="ci_high:Q", color=_color)
        + alt.Chart(_t).mark_line(point=True).encode(
            x=_x, y=alt.Y("median_gap:Q", title="Kalshi − Deribit (vol points)"), color=_color,
            tooltip=["version:N", alt.Tooltip("sd:Q", title="SD"), alt.Tooltip("median_gap:Q", format="+.1f"), "pairs:Q"])
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    ).properties(title="Figure 3 · The same, far from expiry only (4–7 days left)", width=W, height=H)

    # the ATM gap by time to expiry under three assumptions about Kalshi's extra Friday hours
    _busy = hour_profile.copy()
    _busy.iloc[[4 * 24 + h for h in range(8, 23)]] *= 1.5          # Friday 08:00-23:00 UTC, 1.5x its usual variance
    _a = pairs.atm(paired)
    _s = (pd.DataFrame({"time to expiry": _a.tte_bucket, "no stretch": _a.gap_raw, "historical stretch": _a.gap_adj,
                        "extra hours 1.5× busier": pairs.gap_with_profile(_a, _busy)})
          .groupby("time to expiry", observed=True).median())
    _spread = _s.max(axis=1) - _s.min(axis=1)
    _sens = _s.round(1).assign(**{"range": _spread.round(1)}).reset_index().astype({"time to expiry": str})
    mo.vstack([
        mo.md(f"""
        **Figure 3: far from expiry, the result hardly depends on the timing adjustment.** With 4–7 days left
        ({stats['far_pairs']:,} pairs on {stats['far_days']} days), the 13 extra hours are about
        {stats['far_share']:.0%} of Kalshi's horizon and the stretch is only ×{stats['far_ratio_lo']:.3f} to
        ×{stats['far_ratio_hi']:.3f}. At the current price Kalshi is **{stats['far_raw']:+.1f}** points above Deribit
        with no stretch (95% interval {stats['far_raw_lo']:+.1f} to {stats['far_raw_hi']:+.1f}) and
        **{stats['far_adj']:+.1f}** with it ({stats['far_adj_lo']:+.1f} to {stats['far_adj_hi']:+.1f}). No stretch treats
        Kalshi's extra Friday hours as no busier than the rest of the week, which history contradicts, so the true gap
        most likely lies between the two. Deribit's steeper downside holds even with no stretch at all
        ({stats['far_dn_raw']:+.1f} points two standard deviations below).
        """),
        fig3,
        mo.md(f"""
        **Why far from expiry is the cleanest comparison.** The gap at the current price, under three assumptions about
        how busy Kalshi's extra Friday hours are. A week out the three readings span {_spread.get('5–7 days', float('nan')):.1f}
        points; one to two days out, {_spread.get('1–2 days', float('nan')):.1f}.
        """),
        mo.ui.table(_sens, selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo, stats):
    mo.md(f"""
    ## Takeaways

    - **Kalshi's weekly Bitcoin contracts are priced consistently with the options market.** A prediction market
      and an options exchange, with different users and contract designs, imply nearly the same distribution for
      Friday's price: their implied vols move together (correlation {stats['corr']:.2f}).
    - **The level of risk nearly agrees; its shape differs.** Kalshi's vol is about {stats['atm_gap_adj']:.0f} points
      higher at the current price ({stats['far_adj']:+.1f} to {stats['far_raw']:+.1f} far from expiry, whichever way the
      13 hours are handled), and Deribit's is several points higher far below it: options traders pay more for crash
      protection, so Deribit's smile is steeper.
    - **Why gaps can persist:** the users differ (Deribit does not serve US residents), so little capital moves between
      the two; Kalshi requires full collateral and charges fees, and its prices move in 1¢ steps on $500 strikes.
    - **Method points:** the two markets must be compared at the same moment (every Deribit trade against Kalshi
      within 30 seconds), and their expiries aligned; a 13-hour mismatch that includes the US Friday session is
      enough to add {stats['atm_gap_raw'] - stats['atm_gap_adj']:.0f} vol points. Far from expiry the mismatch matters
      least, which makes that comparison the most robust.

    ## Caveats and next steps

    - The expiry stretch uses Bitcoin's **historical** hour-by-hour movement, not what traders expected that Friday,
      and close to expiry the gap still grows after it. Next: interpolate between Deribit's Friday 08:00 and Saturday
      08:00 expiries, which bracket Kalshi's.
    - Our Deribit data are trades only, 3 hours a day: the pairs give the average smile over the year, not a full
      Deribit smile at any moment, and the far wings rest on few pairs. Deribit's marks are fair values, not
      executable prices. Next: Deribit option-chain quotes.
    - Q3 (probabilities): test whether Kalshi's prices come true as often as they say, against Deribit's.

    *Details: `03_iv_surface.py` (Kalshi implied vol), `04_deribit_exploration.py` (Deribit data),
    `05_kalshi_vs_deribit.py` (full comparison). Code in `lib/` (the pairs in `lib/pairs.py`), tested in `tests/`.*
    """)
    return


if __name__ == "__main__":
    app.run()
