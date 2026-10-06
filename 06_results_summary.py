import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="06 Results summary")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    from lib import compare

    return alt, compare, mo, np, pd


@app.cell
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


@app.cell
def _(compare, mo, np, pd):
    ROOT = mo.notebook_dir()
    _c = compare.build_comparison(ROOT)
    matched, smiles, trades = _c["matched"], _c["smiles"], _c["trades"]
    kalshi_snaps = pd.read_parquet(ROOT / "data" / "derived" / "iv_snapshots.parquet",
                                   columns=["event_ticker", "clean", "atm_iv"])

    # headline numbers, all computed here
    _atm = matched[matched.log_moneyness.abs() < 0.005].dropna(subset=["vol_gap_adj"])
    _far = matched[matched.moneyness_bucket == "5–15% below"].dropna(subset=["vol_gap_adj"])
    _p = matched.dropna(subset=["deribit_digital_adj", "deribit_digital_raw", "bid", "ask"])
    _edge = 100 * np.maximum(_p.bid - compare.kalshi_fee(_p.bid) - _p.deribit_digital_adj,
                             _p.deribit_digital_adj - _p.ask - compare.kalshi_fee(_p.ask))
    per_snap = pd.DataFrame([dict(ts=v["ts"], kalshi_atm=v["kalshi_atm"],
                                  deribit_atm=float(compare.interp_vol(0.0, v["smile"].log_moneyness, v["smile"].vol)),
                                  ratio=v["ratio"]) for v in smiles.values()])
    per_snap["deribit_atm_adj"] = per_snap.deribit_atm * per_snap.ratio
    daily = (per_snap.assign(day=per_snap.ts.dt.tz_convert(None).dt.normalize())
             .groupby("day")[["kalshi_atm", "deribit_atm", "deribit_atm_adj"]].median().reset_index())
    stats = dict(
        snapshots=matched.ts.nunique(), days=matched.ts.dt.date.nunique(), weeklies=matched.event_ticker.nunique(),
        kalshi_strikes=len(matched), deribit_trades=len(trades),
        atm_gap_raw=_atm.vol_gap_raw.median(), atm_gap_adj=_atm.vol_gap_adj.median(),
        atm_kalshi_higher=(_atm.vol_gap_adj > 0).mean(), far_gap_adj=_far.vol_gap_adj.median(),
        price_gap_raw=(100 * (_p.mid - _p.deribit_digital_raw)).abs().median(),
        price_gap_adj=(100 * (_p.mid - _p.deribit_digital_adj)).abs().median(),
        kalshi_spread=100 * (_p.ask - _p.bid).median(),
        edge_0=(_edge > 0).mean(), edge_0_median=_edge[_edge > 0].median(), edge_5=(_edge > 5).mean(),
        corr=daily[["kalshi_atm", "deribit_atm_adj"]].corr().iloc[0, 1],
        ratio_lo=matched.horizon_ratio.min(), ratio_hi=matched.horizon_ratio.max(),
    )
    return daily, kalshi_snaps, matched, smiles, stats


@app.cell
def _(mo, stats):
    mo.md(f"""
    # Kalshi vs Deribit: do Bitcoin prediction-market prices agree with options?

    *15.451 Proseminar · results summary · data October 2025 – September 2026*

    **Question.** Kalshi sells weekly contracts that pay **$1 if Bitcoin ends above a price K** at 5pm ET on Friday.
    Deribit, the largest Bitcoin options exchange, sells ordinary calls and puts. In theory a Kalshi contract equals a
    tight call spread (handout Q1c). Are the two markets' prices consistent (Q2b)?

    **Answer: broadly yes.**

    1. **Prices agree to about a cent or two.** Put in the same units, Kalshi's price and Deribit's price for the
       same contract differ by a typical **{stats['price_gap_adj']:.1f}¢**, about Kalshi's own bid-ask spread
       ({stats['kalshi_spread']:.0f}¢).
    2. **Volatility at the current price differs by about a point.** Kalshi's implied vol is
       **{stats['atm_gap_adj']:+.1f} points** above Deribit's (higher {stats['atm_kalshi_higher']:.0%} of the time),
       and the two move together day to day (correlation **{stats['corr']:.2f}**).
    3. **The clear difference is crash protection.** Far below the current price, Deribit's vol is about
       **{-stats['far_gap_adj']:.0f} points higher** than Kalshi's: Deribit traders pay more for downside insurance.
    4. **There is little free money.** Gaps beyond Kalshi's bid-ask are mostly smaller than the cost of trading on
       both exchanges: with an assumed 5¢ of Deribit costs per $1 contract, only **{stats['edge_5']:.0%}** of quotes
       still show an edge.
    5. **Getting the timing right is essential.** Deribit's options expire 13 hours before Kalshi's. Ignoring that
       makes Kalshi look {stats['atm_gap_raw']:.0f} vol points too high instead of {stats['atm_gap_adj']:.0f}.
    """)
    return


@app.cell
def _(mo, pd, stats, kalshi_snaps):
    _data = pd.DataFrame([
        ("Kalshi", "Weekly BTC contracts (KXBTCD): '$1 if BTC > K at 5pm ET Friday'",
         f"{kalshi_snaps.event_ticker.nunique()} weeks, Oct 2025 – Sep 2026, ~54 strikes $500 apart, 1-minute bid/ask"),
        ("Deribit", "BTC options (calls and puts), all expiring 08:00 UTC",
         f"{stats['deribit_trades']:,} trades in Friday expiries, 19:30–22:29 UTC, Jan – Sep 2026, with Deribit's mark price"),
        ("Bitcoin price", "Coinbase and Bitstamp, average", "every minute; within 0.015% of Kalshi's settlement index"),
        ("Matched sample", "Kalshi snapshots with Deribit trades within 15 minutes",
         f"{stats['snapshots']:,} snapshots on {stats['days']} days ({stats['weeklies']} weeks), "
         f"{stats['kalshi_strikes']:,} Kalshi strike-prices"),
    ], columns=["Source", "What", "Coverage"])
    mo.vstack([mo.md("## Data"), mo.ui.table(_data, selection=None, show_download=False, pagination=False)])
    return


@app.cell
def _(mo, stats):
    mo.md(f"""
    ## Method

    **Step 1. Turn Kalshi's prices into implied volatility** (notebook `03`). A Kalshi contract pays $1 above K, so its
    price D(K) is the market's probability. Breeden and Litzenberger (1978): D(K) = −∂C/∂K, so **adding up Kalshi's
    prices across strikes gives call and put prices**, C(K) = ∫ above K of D. Beyond the last strike, the tails take a
    Black-Scholes shape bounded by Kalshi's own outer quotes. Put-call parity gives Kalshi's forward, and solving
    Black-Scholes for each option gives one implied vol per strike, every 15 minutes. (Solving N(d₂) = price for
    each contract directly is too noisy near the current price and biased low by the smile, so it is not used.)

    **Step 2. Line Kalshi up with Deribit.** Three mismatches, each handled:

    | Mismatch | Fix |
    |---|---|
    | **Time:** Deribit data are individual trades | Each Kalshi snapshot (every 15 min, 19:30–22:29 UTC) uses Deribit's mark prices from trades within 15 minutes, turned into a smile: vol by moneyness (strike ÷ BTC price) |
    | **Strikes:** Deribit uses round thousands, Kalshi $500 steps | Deribit's smile is interpolated to Kalshi's strikes, never beyond the strikes Deribit traded |
    | **Expiry:** Deribit ends 08:00 UTC, Kalshi 13 hours later | Deribit's vol is **stretched** over the extra hours. Variance adds up hour by hour, and those 13 hours (Friday's US session) historically move 1.6× an average hour, so Deribit's vol is scaled up by ×{stats['ratio_lo']:.2f} (a week out) to ×{stats['ratio_hi']:.2f} (a day out). Results are shown with and without this step |

    **Step 3. Compare two ways.**

    - **Implied vols:** Kalshi's vol against Deribit's at the same strike.
    - **Prices:** Deribit's price for Kalshi's contract is a call spread, buy the K − 250 call and sell the K + 250
      call, divided by $500, priced from Deribit's smile. It is compared with Kalshi's bid, ask and mid, in cents.
      As a check without any model, the same is done from Deribit's observed prices at real strikes K ± 500.
    """)
    return


@app.cell
def _(H, MUTED, SLOT1, SLOT2, W, alt, compare, matched, mo, np, pd, smiles):
    _ts = pd.Timestamp("2026-06-29 20:30", tz="UTC")
    _k = matched[(matched.ts == _ts) & matched.mid.between(0.02, 0.98)]
    _s = smiles[int(_k.snap.iloc[0])]
    _grid = np.arange(_k.strike.min(), _k.strike.max() + 1, 100.0)
    _curves = pd.concat([
        pd.DataFrame({"strike": _grid, "version": "Deribit, stretched to Kalshi's expiry",
                      "price": compare.digital_from_smile(_grid, _s["F"], _s["tau_k"], _s["smile"].log_moneyness,
                                                         _s["smile"].vol * _s["ratio"])}),
        pd.DataFrame({"strike": _grid, "version": "Deribit at its own expiry (08:00 UTC)",
                      "price": compare.digital_from_smile(_grid, _s["F"], _s["tau_d"], _s["smile"].log_moneyness,
                                                         _s["smile"].vol)}),
    ]).dropna()
    _x = alt.X("strike:Q", title="strike K", scale=alt.Scale(zero=False), axis=alt.Axis(format="$,.0f"))
    _y = alt.Y("price:Q", title="price of $1 if BTC ends above K", axis=alt.Axis(format=".0%"))
    fig1 = (
        alt.Chart(_curves).mark_line(strokeWidth=2)
        .encode(x=_x, y=_y, color=alt.Color("version:N", title=None, legend=alt.Legend(labelLimit=0), scale=alt.Scale(
                    domain=["Deribit, stretched to Kalshi's expiry", "Deribit at its own expiry (08:00 UTC)"],
                    range=[SLOT1, SLOT2])),
                strokeDash=alt.StrokeDash("version:N", title=None, scale=alt.Scale(
                    domain=["Deribit, stretched to Kalshi's expiry", "Deribit at its own expiry (08:00 UTC)"],
                    range=[[1, 0], [5, 4]])))
        + alt.Chart(_k).mark_rule(color=MUTED, strokeWidth=3).encode(x=_x, y=alt.Y("bid:Q"), y2="ask:Q")
        + alt.Chart(_k).mark_point(filled=True, size=36, color="#0b0b0b")
        .encode(x=_x, y=alt.Y("mid:Q"), tooltip=[alt.Tooltip("strike:Q", format="$,.0f"),
                                                  alt.Tooltip("bid:Q", format=".2f"), alt.Tooltip("ask:Q", format=".2f"),
                                                  alt.Tooltip("deribit_digital_adj:Q", format=".3f", title="Deribit")])
    ).properties(title="Figure 1 · One snapshot: Kalshi's prices (black, gray = bid to ask) vs Deribit's price "
                       "for the same contracts", width=W, height=H + 40)
    _gap = (100 * (_k.mid - _k.deribit_digital_adj)).abs().dropna()
    mo.vstack([
        mo.md(f"""
        ## Results

        **Figure 1** shows the whole comparison on one moment: Monday June 29, 2026, 20:30 UTC, BTC near
        ${_s['F']:,.0f}, contracts expiring Friday July 3. Each black dot is a Kalshi contract with its bid-ask range;
        the lines are Deribit's price for the same contract. The two markets follow the same curve closely: Deribit's
        stretched price is a median **{_gap.median():.1f}¢** from Kalshi's mid here (at most {_gap.max():.1f}¢). The
        dashed line (Deribit at its own, earlier expiry) is steeper: with 13 hours less, outcomes are less spread out.
        """),
        fig1,
    ])
    return


@app.cell
def _(H, SLOT1, SLOT2, W, alt, daily, mo, stats):
    _long = daily.melt(id_vars=["day"], value_vars=["kalshi_atm", "deribit_atm_adj"], var_name="market", value_name="vol")
    _long["market"] = _long.market.map({"kalshi_atm": "Kalshi", "deribit_atm_adj": "Deribit (stretched)"})
    fig2 = (
        alt.Chart(_long).mark_line(strokeWidth=1.6)
        .encode(x=alt.X("day:T", title=None, axis=alt.Axis(format="%b %Y")),
                y=alt.Y("vol:Q", title="implied vol at the current price", axis=alt.Axis(format="%"),
                        scale=alt.Scale(zero=False)),
                color=alt.Color("market:N", title=None, scale=alt.Scale(domain=["Kalshi", "Deribit (stretched)"],
                                                                        range=[SLOT1, SLOT2])),
                tooltip=[alt.Tooltip("day:T", format="%Y-%m-%d"), "market:N", alt.Tooltip("vol:Q", format=".1%")])
        .properties(title="Figure 2 · Implied vol at the current price, daily median, Kalshi vs Deribit",
                    width=W, height=H)
    )
    mo.vstack([
        mo.md(f"""
        **Figure 2: the two markets price the same amount of movement.** Day by day, Kalshi's and Deribit's implied
        vols at the current price rise and fall together (correlation **{stats['corr']:.2f}** over
        {len(daily)} days), including the spikes in February and June 2026. Kalshi sits slightly above.
        """),
        fig2,
    ])
    return


@app.cell
def _(H, MUTED, SLOT1, SLOT2, W, alt, compare, matched, mo, pd, stats):
    _x = matched.dropna(subset=["vol_gap_raw"])
    _by = (_x.groupby("moneyness_bucket", observed=True)[["vol_gap_raw", "vol_gap_adj"]].median().reset_index())
    _long = _by.melt(id_vars=["moneyness_bucket"], var_name="version", value_name="gap")
    _long["version"] = _long.version.map({"vol_gap_raw": "Deribit at its own expiry",
                                         "vol_gap_adj": "Deribit stretched to Kalshi's expiry"})
    _long["moneyness_bucket"] = _long.moneyness_bucket.astype(str)
    fig3 = (
        alt.Chart(_long).mark_line(point=True)
        .encode(x=alt.X("moneyness_bucket:N", sort=compare.MONEYNESS_LABELS, title="strike relative to the current price"),
                y=alt.Y("gap:Q", title="Kalshi − Deribit (vol points)"),
                color=alt.Color("version:N", title=None, scale=alt.Scale(
                    domain=["Deribit at its own expiry", "Deribit stretched to Kalshi's expiry"], range=[SLOT2, SLOT1])),
                tooltip=["moneyness_bucket:N", "version:N", alt.Tooltip("gap:Q", format="+.1f")])
        .properties(title="Figure 3 · Implied vol gap across strikes (median)", width=W, height=H)
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    )
    mo.vstack([
        mo.md(f"""
        **Figure 3: vols match near the current price; Deribit charges more for crash protection.** At the current
        price, Kalshi is {stats['atm_gap_raw']:+.1f} points above Deribit as traded and **{stats['atm_gap_adj']:+.1f}**
        after stretching Deribit to Kalshi's expiry. Far below the current price the gap turns negative
        (**{stats['far_gap_adj']:+.1f}** points): Deribit's smile is steeper on the downside.
        """),
        fig3,
    ])
    return


@app.cell
def _(H, MUTED, SLOT1, SLOT2, W, alt, compare, matched, mo, pd, stats):
    _x = matched.dropna(subset=["deribit_digital_adj", "deribit_digital_raw", "bid", "ask"])
    _by = (_x.assign(raw=100 * (_x.mid - _x.deribit_digital_raw), adj=100 * (_x.mid - _x.deribit_digital_adj))
           .groupby("moneyness_bucket", observed=True)[["raw", "adj"]].median().reset_index())
    _long = _by.melt(id_vars=["moneyness_bucket"], var_name="version", value_name="gap")
    _long["version"] = _long.version.map({"raw": "Deribit at its own expiry",
                                         "adj": "Deribit stretched to Kalshi's expiry"})
    _long["moneyness_bucket"] = _long.moneyness_bucket.astype(str)
    fig4 = (
        alt.Chart(_long).mark_bar()
        .encode(x=alt.X("moneyness_bucket:N", sort=compare.MONEYNESS_LABELS, title="strike relative to the current price"),
                xOffset=alt.XOffset("version:N"),
                y=alt.Y("gap:Q", title="Kalshi − Deribit (cents per $1)"),
                color=alt.Color("version:N", title=None, scale=alt.Scale(
                    domain=["Deribit at its own expiry", "Deribit stretched to Kalshi's expiry"], range=[SLOT2, SLOT1])),
                tooltip=["moneyness_bucket:N", "version:N", alt.Tooltip("gap:Q", format="+.1f")])
        .properties(title="Figure 4 · Price gap across strikes (median, cents)", width=W, height=H)
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED).encode(y="y:Q")
    )
    _d = matched.dropna(subset=["deribit_digital_direct", "deribit_digital_raw", "deribit_digital_adj", "bid", "ask"])
    _robust = pd.DataFrame([
        (name, f"{(100 * (_d.mid - _d[c])).abs().median():.1f}¢")
        for name, c in [("Deribit's observed prices, real strikes, no model", "deribit_digital_direct"),
                        ("Black-Scholes from Deribit's smile, not stretched", "deribit_digital_raw"),
                        ("Black-Scholes from Deribit's smile, stretched", "deribit_digital_adj")]
    ], columns=["Deribit price used", "typical gap to Kalshi"])
    mo.vstack([
        mo.md(f"""
        **Figure 4: prices agree to a cent or two once expiries are aligned.** Without the stretch (orange), Kalshi
        looks cheap below the current price and dear above it, which is exactly what 13 extra hours of possible movement
        produce. With the stretch (blue) the typical gap falls from {stats['price_gap_raw']:.1f}¢ to
        **{stats['price_gap_adj']:.1f}¢**, about Kalshi's own bid-ask spread ({stats['kalshi_spread']:.0f}¢).
        """),
        fig4,
        mo.md(f"""
        **Check without any model.** Where Deribit traded calls at both K − 500 and K + 500 ({len(_d):,} cases), its
        observed prices give the same picture as the Black-Scholes version before stretching, so the model is not
        driving the result; the expiry stretch is what closes the gap:
        """),
        mo.ui.table(_robust, selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell
def _(compare, matched, mo, np, pd, stats):
    _p = matched.dropna(subset=["deribit_digital_adj", "bid", "ask"])
    _edge = 100 * np.maximum(_p.bid - compare.kalshi_fee(_p.bid) - _p.deribit_digital_adj,
                             _p.deribit_digital_adj - _p.ask - compare.kalshi_fee(_p.ask))
    _t = pd.DataFrame([(f"{c}¢", f"{(_edge > c).mean():.1%}") for c in (0, 2, 5, 10)],
                      columns=["assumed Deribit cost per $1 contract", "Kalshi quotes still profitable"])
    mo.vstack([
        mo.md(f"""
        **Is there money on the table? Little.** Selling on Kalshi and buying Deribit's call spread (or the reverse)
        pays Kalshi's fee and Deribit's costs. After Kalshi's fee, an edge remains on {stats['edge_0']:.0%} of quotes,
        typically {stats['edge_0_median']:.1f}¢. Deribit's costs are not in our data (we have trades, not quotes), but
        copying a $1 contract takes two Deribit options for 1/500 of a bitcoin each, so a few cents per $1 is
        plausible:
        """),
        mo.ui.table(_t, selection=None, show_download=False, pagination=False),
        mo.md("What survives is concentrated in the last two days, where the Deribit leg expires 13 hours before "
              "the Kalshi leg: that is risk, not arbitrage."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Kalshi's weekly Bitcoin contracts are priced consistently with the options market.** A prediction market
      and an options exchange, with different users and contract designs, imply nearly the same distribution for
      Friday's price.
    - **The level of risk agrees; its shape differs.** Kalshi's vol is about a point higher at the current price,
      and Deribit's is a few points higher far below it: options traders pay more for crash protection.
    - **The remaining gaps are within trading costs**, which can persist because the users differ (Deribit does not
      serve US residents), Kalshi requires full collateral and charges fees, and its prices move in 1¢ steps on $500
      strikes.
    - **Method point:** comparing the two requires aligning expiries; a 13-hour mismatch that includes the US
      Friday session is enough to create a spurious 3-point vol gap.

    ## Caveats and next steps

    - The expiry stretch uses Bitcoin's **historical** hour-by-hour movement, not what traders expected that Friday.
      Next: interpolate between Deribit's Friday 08:00 and Saturday 08:00 expiries, which bracket Kalshi's.
    - Deribit marks are Deribit's fair values, not executable prices; its trading costs are assumed. Next: Deribit
      order-book quotes.
    - Kalshi's fee formula (7% × P × (1 − P), rounded up) is to be confirmed against the current schedule.
    - Q3 (probabilities): test whether Kalshi's prices come true as often as they say, against Deribit's.

    *Details: `03_iv_surface.py` (Kalshi implied vol), `04_deribit_exploration.py` (Deribit data),
    `05_kalshi_vs_deribit.py` (full comparison). Code in `lib/`, 127 tests in `tests/`.*
    """)
    return


if __name__ == "__main__":
    app.run()
