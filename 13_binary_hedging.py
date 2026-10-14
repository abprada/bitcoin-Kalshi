import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="13 Hedging a Kalshi binary")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    from lib import binary_hedge as bh

    return alt, bh, mo, np, pd


@app.cell(hide_code=True)
def _(alt):
    # Chart styling as in 05/06. Long Kalshi = blue, short call spread = orange, net = aqua.
    INK, INK2, MUTED, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
    SLOT1, SLOT2, SLOT3, SLOT4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    W, H = 300, 170


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
                "header": {"labelColor": INK2, "labelFontSize": 12},
                "line": {"color": SLOT1, "strokeWidth": 2},
                "point": {"color": SLOT1, "size": 64, "filled": True},
            }
        }

    LEGS = {"long Kalshi binary": SLOT1, "short call spread": SLOT2, "net position": SLOT3}
    return H, LEGS, MUTED, SLOT1, SLOT2, SLOT3, SLOT4, W


@app.cell(hide_code=True)
def _(mo, pd):
    ROOT = mo.notebook_dir()
    D = ROOT / "data" / "derived" / "strategies"
    FIG = ROOT / "outputs" / "figures"
    inputs = pd.read_parquet(D / "hedge_day_inputs.parquet")
    mids = pd.read_parquet(D / "hedge_day_kalshi_mids.parquet")
    steps = pd.read_parquet(D / "hedge_day_steps.parquet")
    summ = pd.read_parquet(D / "hedge_day_summary.parquet")
    cross = pd.read_parquet(D / "hedge_cross_day.parquet")
    SETTLE = float(summ.settle.iloc[0])
    return FIG, SETTLE, cross, inputs, mids, steps, summ


@app.cell(hide_code=True)
def _(SETTLE, bh, inputs, mids, mo, np, pd, summ):
    _r = inputs.iloc[0]
    _sm = bh.Smile(_r.spot, _r.tau, _r.atm, _r.b, _r.c, _r.m_lo, _r.m_hi)
    _rows = []
    for _k in sorted(float(c) for c in mids.columns):
        _m = mids[str(_k)].iloc[0] if str(_k) in mids.columns else mids[_k].iloc[0]
        if np.isfinite(_m):
            _rows.append({"strike": _k, "Kalshi mid": _m, "option-implied binary": bh.binary_value(_r.spot, _k, _r.tau, _sm),
                          "option-implied $500 call spread": bh.spread_value(_r.spot, _k, 250, _r.tau, _sm)})
    prices0 = pd.DataFrame(_rows)
    prices0["Kalshi cheaper by (¢)"] = 100 * (prices0["option-implied binary"] - prices0["Kalshi mid"])
    _main = summ[(summ.K == 123000.0) & (summ.scheme == "call spread $500 wide, static")].iloc[0]
    case = dict(K=123000.0, price=float(prices0.set_index("strike").loc[123000.0, "Kalshi mid"]),
                contracts=_main.contracts, spreads=_main.contracts, edge=_main.edge_at_entry, spot=_r.spot,
                vol=_r.atm)
    mo.md(f"""
    # 13 · Hedging a long Kalshi binary against a short call spread

    *Prompt: suppose Kalshi were cheap versus options. Go long Kalshi and short the call spread. How do you hedge
    dynamically as expiry approaches; what happens to the gamma and vega of the two sides; what does the P&L look
    like on 9 October with $1 million of Kalshi, hedged every half hour, with no frictions; is there a better way?*

    **Answer in four lines.**

    1. **The call spread *is* the hedge.** A Kalshi binary "BTC above K" pays like a tight call spread around K, so
       the short spread replicates it statically; delta-hedging what is left is a small correction, and on most
       days the static pair locks in the price difference with no trading at all.
    2. **Delta-hedging the binary on its own does not work near expiry.** Its gamma grows like 1 / (time left)
       around the strike: with $1M of at-the-money binaries and 30 minutes left, the delta hedge is thousands of
       BTC. Across all 222 Kalshi dailies of 2026, half-hourly delta hedging leaves an error of about ±$500k on a
       $1M position, worse than many outcomes it is meant to protect.
    3. **What is left after the spread is pin risk**: if BTC settles inside the spread's ±$250, the binary pays 0
       or 1 while the spread pays in between. That is where all the large errors come from.
    4. **The better hedge is static and short-lived:** keep the spread, do not chase the residual delta, and unwind
       both legs about an hour before expiry. Across the 2026 dailies that cuts the spread of outcomes from about
       ±$240k to ±$70k on a $1M position, while keeping the locked edge.

    **9 October 2025.** Kalshi's daily KXBTCD-25OCT0917 settles at 5pm ET (21:00 UTC). At 00:00 UTC, with BTC at
    ${case['spot']:,.0f}, Kalshi's $123,000 strike traded at {100 * case['price']:.1f}¢, about 1.4¢ below the
    price implied by Deribit's options at Kalshi's expiry (table). $1,000,000 buys
    **{case['contracts']:,.0f} contracts**; the matching hedge is to sell {case['contracts']:,.0f} units of the
    $122,750 / $123,250 call spread (each unit pays up to $1, i.e. {case['contracts'] / 500:,.0f} one-BTC call
    spreads). BTC then fell to about $120,000 and settled at **${SETTLE:,.2f}**: the binary expired worthless, so
    unhedged the trade loses the whole $1M.
    """)
    return case, prices0


@app.cell(hide_code=True)
def _(mo, prices0):
    mo.vstack([mo.md("**9 October 2025, 00:00 UTC: Kalshi vs options, strikes with a usable Kalshi quote**"),
               mo.ui.table(prices0.round(3), selection=None, show_download=False, pagination=False)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method

    - **Market data.** Kalshi: 1-minute quotes for the daily (`00_fetch_kalshi_event.py`), mid when the spread is at
      most 20¢ (02's rule). BTC: Coinbase/Bitstamp average every minute. Options: Deribit's mark smile from its
      trades every 30 minutes, with its ATM vol interpolated to Kalshi's 5pm ET expiry (as in 09) and the smile's
      shape from Deribit's nearest expiry.
    - **Frictionless.** The call spread is priced on that smile at Kalshi's own expiry (no such Deribit option
      exists: Deribit expires at 08:00 UTC), BTC and every option trade at model value, no fees, any size.
    - **Greeks** by finite differences, holding each strike's vol fixed. Gamma is shown as the P&L from a 1% move
      due to curvature alone (½ Γ (1% × S)²); vega per vol point.
    - **Hedging** with BTC itself, every 30 minutes unless stated. P&L = Kalshi leg (at Kalshi's mid, settling on its
      result) + call spread leg + hedge.
    """)
    return


@app.cell(hide_code=True)
def _(FIG, H, LEGS, W, alt, mo, pd, steps):
    # Greeks of each leg along the actual path, every 30 minutes
    _cases = {123000.0: "Kalshi cheap, $123,000 strike (from 00:00 UTC)",
              121000.0: "pin case, $121,000 strike (from 12:00 UTC)"}
    _s = steps[(steps.scheme == "call spread $500 wide + delta, 30 min") & steps.bin_delta.notna()].copy()
    _rows = []
    for _, _r in _s.iterrows():
        for _metric, _a, _b in [("delta (BTC)", "bin_delta", "sp_delta"),
                                ("gamma: P&L from a 1% move ($)", "bin_gamma_1pct", "sp_gamma_1pct"),
                                ("vega ($ per vol point)", "bin_vega", "sp_vega")]:
            for _leg, _v in [("long Kalshi binary", _r[_a]), ("short call spread", _r[_b]), ("net position", _r[_a] + _r[_b])]:
                _rows.append({"t": _r.t, "case": _cases[_r.K], "metric": _metric, "leg": _leg, "value": _v})
    _g = pd.DataFrame(_rows)
    fig_greeks = (
        alt.Chart(_g).mark_line(strokeWidth=1.8).encode(
            x=alt.X("t:T", title="UTC", axis=alt.Axis(format="%H:%M"), scale=alt.Scale(type="utc")),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(format="~s")),
            color=alt.Color("leg:N", title=None, scale=alt.Scale(domain=list(LEGS), range=list(LEGS.values()))),
            tooltip=[alt.Tooltip("t:T", format="%H:%M"), "leg:N", alt.Tooltip("value:Q", format=",.0f")])
        .properties(width=W, height=H)
        .facet(row=alt.Row("metric:N", title=None, sort=["delta (BTC)", "gamma: P&L from a 1% move ($)",
                                                         "vega ($ per vol point)"]),
               column=alt.Column("case:N", title=None, sort=list(_cases.values())))
        .resolve_scale(y="independent", x="independent")
        .properties(title="Figure 1 · Greeks of each side through 9 October 2025 (UTC), every 30 minutes")
    )
    fig_greeks.save(FIG / "13_fig1_greeks_through_the_day.png", scale_factor=2)
    mo.vstack([
        mo.md("""
        ## 1 · What happens to gamma and vega on each side

        Figure 1 follows the delta, gamma and vega of the long Kalshi binary (blue), the short call spread
        (orange) and the net position (aqua), for the trade above (left) and, as a stress case, for the
        $121,000 strike bought at noon, which settled only $176 in the money (right).

        - **The two sides mirror each other.** The binary's Greeks are the call spread's with a tighter width, so the
          short spread cancels almost all of them: the aqua line stays near zero while blue and orange swing.
        - **Gamma changes sign at the strike and grows as expiry approaches.** Long a binary you are long gamma below
          the strike and short gamma above it. On the left, BTC moves away from $123,000 and both sides' gamma fades.
          On the right, BTC sits near $121,000 into the close and gamma explodes on both sides; the net stays smaller
          but not small, because the binary's gamma is concentrated in a narrower band than the spread's.
        - **Vega also flips sign at the strike and fades to zero at expiry**: a binary's value depends on the
          probability of finishing above K, which more vol pushes toward 50% from either side.
        """),
        fig_greeks,
    ])
    return


@app.cell(hide_code=True)
def _(FIG, H, LEGS, W, alt, bh, mo, np, pd):
    # Theory: gamma of $1M of binaries vs the matching call spread, across spot, at several times left
    _K, _vol, _S0 = 121000.0, 0.42, 121000.0
    _N = 1e6 / 0.5
    _rows = []
    for _lab, _hours in [("12 hours left", 12), ("3 hours left", 3), ("1 hour left", 1), ("15 minutes left", 0.25)]:
        _tau = _hours / 8760
        _sm = bh.Smile(_S0, _tau, _vol, 0.0, 0.0, -9, 9)
        for _S in np.linspace(_K * 0.985, _K * 1.015, 241):
            _gb = bh.greeks(lambda S, t, s: bh.binary_value(S, _K, t, s), _S, _tau, _sm)["gamma"]
            _gs = bh.greeks(lambda S, t, s: bh.spread_value(S, _K, 250, t, s), _S, _tau, _sm)["gamma"]
            _scale = 0.5 * (0.01 * _S) ** 2 * _N
            for _leg, _v in [("long Kalshi binary", _gb * _scale), ("short call spread", -_gs * _scale),
                             ("net position", (_gb - _gs) * _scale)]:
                _rows.append({"time left": _lab, "BTC": _S, "leg": _leg, "gamma": _v})
    _t = pd.DataFrame(_rows)
    fig_gamma = (
        alt.Chart(_t).mark_line(strokeWidth=1.8).encode(
            x=alt.X("BTC:Q", title="BTC price ($), strike $121,000", scale=alt.Scale(zero=False),
                    axis=alt.Axis(format="~s")),
            y=alt.Y("gamma:Q", title="P&L from a 1% move ($)", axis=alt.Axis(format="~s")),
            color=alt.Color("leg:N", title=None, scale=alt.Scale(domain=list(LEGS), range=list(LEGS.values()))),
            tooltip=["time left:N", "leg:N", alt.Tooltip("BTC:Q", format=",.0f"), alt.Tooltip("gamma:Q", format=",.0f")])
        .properties(width=W, height=H)
        .facet(column=alt.Column("time left:N", title=None,
                                 sort=["12 hours left", "3 hours left", "1 hour left", "15 minutes left"]), spacing=10)
        .resolve_scale(y="independent")
        .properties(title="Figure 2 · Gamma of $1M of Kalshi binaries and of the matching $500-wide call spread (vol 42%)")
    )
    fig_gamma.save(FIG / "13_fig2_gamma_explosion.png", scale_factor=2)
    _tau = 0.5 / 8760
    _sm = bh.Smile(_S0, _tau, _vol, 0.0, 0.0, -9, 9)
    _hedge = _N * bh.greeks(lambda S, t, s: bh.binary_value(S, _K, t, s), _S0, _tau, _sm)["delta"]
    mo.vstack([
        mo.md(f"""
        **Why delta-hedging the binary alone breaks down** (Figure 2, scales differ panel by panel). The binary's
        gamma is a pair of spikes either side of the strike whose height grows like 1 / (time left) and whose width
        shrinks like √(time left). With 30 minutes left and BTC at the strike, the delta of $1M of at-the-money
        binaries is about **{_hedge:,.0f} BTC (${_hedge * _S0 / 1e6:,.0f} million)**: a $500 move flips the hedge
        by most of that. The call spread's gamma is bounded by its $500 width; it tracks the binary until the
        expected move is about as small as the width, and then the two separate. That is the window in the last
        hour or so where the hedge gets hard.
        """),
        fig_gamma,
    ])
    return


@app.cell(hide_code=True)
def _(FIG, H, SLOT1, SLOT2, SLOT3, SLOT4, W, alt, mo, pd, steps, summ):
    _show = {"delta, 30 min": SLOT2, "call spread $500 wide, static": SLOT1,
             "call spread $500 wide + delta, 30 min": SLOT3, "call spread $500 wide, unwind 1h before": SLOT4}
    _p = steps[(steps.K == 123000.0) & steps.scheme.isin(list(_show))][["t", "scheme", "pnl_total"]]
    fig_path = (
        alt.Chart(_p).mark_line(strokeWidth=1.8, point=alt.OverlayMarkDef(size=18)).encode(
            x=alt.X("t:T", title="9 October 2025 (UTC)", axis=alt.Axis(format="%H:%M"), scale=alt.Scale(type="utc")),
            y=alt.Y("pnl_total:Q", title="cumulative P&L ($)", axis=alt.Axis(format="~s")),
            color=alt.Color("scheme:N", title=None, scale=alt.Scale(domain=list(_show), range=list(_show.values())),
                            legend=alt.Legend(columns=2, labelLimit=400)),
            tooltip=[alt.Tooltip("t:T", format="%H:%M"), "scheme:N", alt.Tooltip("pnl_total:Q", format=",.0f")])
        .properties(width=2 * W + 40, height=H + 60,
                    title="Figure 3 · $1M long Kalshi ($123,000 strike), P&L every half hour by hedge")
    )
    fig_path.save(FIG / "13_fig3_pnl_path_oct9.png", scale_factor=2)
    _t = summ.assign(case=summ.K.map({123000.0: "$123,000 (Kalshi cheap)", 121000.0: "$121,000 (pin)"}))[
        ["case", "scheme", "edge_at_entry", "pnl_kalshi", "pnl_spread", "pnl_hedge", "total"]].rename(columns={
            "edge_at_entry": "edge at entry $", "pnl_kalshi": "Kalshi leg $", "pnl_spread": "call spread leg $",
            "pnl_hedge": "BTC hedge $", "total": "total $"})
    mo.vstack([
        mo.md("""
        ## 2 · The P&L on 9 October, hedged every half hour

        Figure 3: the $123,000 trade, marked every half hour (Kalshi at its own mid, the spread at model value).
        Unhedged it loses the full $1,000,000 (not shown). Delta-hedging the binary alone every 30 minutes ends near
        −$27,000 after falling as low as −$120,000 during the day, and needs up to 430 BTC of hedge. With the short
        call spread it ends at about +$24,000, the 1.4¢ edge times 1.75 million contracts. Marked every half hour,
        the pair stays between −$38,000 and +$33,000 whether or not the residual is delta-hedged (the swings are
        mostly Kalshi's own quote moving against the option model, not hedge error): +$24,000 static, +$27,000
        with half-hourly delta hedges, +$22,000 if both legs are unwound an hour before expiry.

        The table gives every scheme for both strikes. On the pin strike ($121,000, settled $176 above), the
        half-hourly delta hedge of the binary alone reached **1,202 BTC (about $145 million) at 20:30 UTC** for a
        $1M position, and every scheme's result is dominated by where BTC landed inside the spread's band.
        """),
        fig_path,
        mo.ui.table(_t.round(0), selection=None, show_download=True, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(FIG, H, SLOT1, SLOT2, W, alt, cross, mo, np, pd):
    _g = cross.groupby("scheme")
    tbl = pd.DataFrame({
        "days": _g.size(), "edge locked at entry $ (avg)": _g.edge.mean(), "total P&L $ (avg)": _g.total.mean(),
        "hedging error sd $": _g.hedge_error.std(), "median |error| $": _g.hedge_error.apply(lambda x: x.abs().median()),
        "5th pct $": _g.hedge_error.quantile(0.05), "95th pct $": _g.hedge_error.quantile(0.95),
        "worst day $": _g.hedge_error.min(), "share of days |error| < $10k": _g.hedge_error.apply(lambda x: (x.abs() < 1e4).mean()),
    }).sort_values("hedging error sd $").reset_index()
    _long = tbl.melt(id_vars=["scheme"], value_vars=["5th pct $", "95th pct $"], value_name="v")
    _rng = alt.Chart(tbl).mark_rule(strokeWidth=3, color=SLOT1).encode(
        y=alt.Y("scheme:N", sort=list(tbl.scheme), title=None, axis=alt.Axis(labelLimit=400)), x=alt.X("5th pct $:Q", title="hedging error per day, $ (5th–95th percentile, sd as dot)", axis=alt.Axis(format="~s")),
        x2="95th pct $:Q")
    _sd = alt.Chart(tbl).mark_point(filled=True, size=70, color=SLOT2).encode(
        y=alt.Y("scheme:N", sort=list(tbl.scheme)), x="hedging error sd $:Q",
        tooltip=["scheme:N", alt.Tooltip("hedging error sd $:Q", format=",.0f"), alt.Tooltip("worst day $:Q", format=",.0f")])
    fig_cross = (_rng + _sd + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#898781").encode(x="x:Q")).properties(
        width=2 * W, height=22 * len(tbl),
        title="Figure 4 · Every 2026 Kalshi daily, $1M long at the money: how well does each hedge work?")
    fig_cross.save(FIG / "13_fig4_hedges_across_days.png", scale_factor=2)
    # where the errors come from: distance of the settlement from the strike
    _c = cross[cross.scheme.isin(["call spread $500 wide, static", "delta, 30 min"])].copy()
    _c["|settlement − strike| (in expected moves)"] = _c.dist_settle_sd.abs()
    fig_pin = alt.Chart(_c).mark_point(filled=True, size=26, opacity=0.6).encode(
        x=alt.X("|settlement − strike| (in expected moves):Q", scale=alt.Scale(type="sqrt")),
        y=alt.Y("hedge_error:Q", title="hedging error ($)", axis=alt.Axis(format="~s")),
        color=alt.Color("scheme:N", title=None, scale=alt.Scale(domain=["call spread $500 wide, static", "delta, 30 min"],
                                                                range=[SLOT1, SLOT2])),
        tooltip=["day:T", "scheme:N", alt.Tooltip("hedge_error:Q", format=",.0f"), "K:Q", "settle:Q"]
    ).properties(width=2 * W, height=H + 40,
                 title="Figure 5 · The errors are pin risk: they happen when BTC settles close to the strike")
    fig_pin.save(FIG / "13_fig5_pin_risk.png", scale_factor=2)
    mo.vstack([
        mo.md(f"""
        ## 3 · Is one day representative? Every 2026 Kalshi daily

        The same experiment on each of the {cross.event_ticker.nunique()} Kalshi dailies of 2026: buy $1M of the strike
        nearest BTC at the first hour of the contract (about 24 hours before expiry), at Kalshi's mid, and hedge by
        each scheme, frictionless. The **hedging error** is the final P&L minus the edge locked at entry (the
        model value of what was sold minus Kalshi's price): it is what the hedge is responsible for.

        - **Best: the static spread, unwound an hour before expiry** (at Kalshi's actual mid then): hedging error
          about ±$72k, against ±$237k if held to expiry. The last hour is where all the pin risk lives.
        - **The static centred spread held to expiry** is next: its error is exactly zero whenever BTC settles
          outside the strike ± $250 (most days), and comes entirely from settlements inside that band.
        - **Delta-hedging the residual adds nothing** (±$229k to ±$244k whether plain, capped at the position's
          size, or stopped 3 hours early), and the plain version makes the worst day worse (−$1.7M): it re-hedges a
          residual whose gamma explodes in the last hour.
        - **Delta-hedging the binary alone** has an error of about ±$500k (30 minutes) or ±$300k (5 minutes) on a $1M
          position, with single days worse than −$4M.
        - **One-sided spreads** (K to K + $500, or K − $500 to K) make the expiry outcome bounded on one side, but
          the protection is paid for up front (the locked edge is about −$160k or +$205k instead of +$25k), so they
          change the shape of the risk, not its size.

        Figure 5 shows why: the large errors, for every scheme, are the days BTC settles within about 0.1 expected
        moves of the strike.
        """),
        fig_cross,
        mo.ui.table(tbl.round(0), selection=None, show_download=True, pagination=False),
        fig_pin,
    ])
    return (tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Answers

    - **How would you dynamically hedge it?** Mostly you would not: the short call spread is a static hedge of
      the binary. What is left (binary minus spread) has small Greeks until the last hour, when BTC near the strike
      makes both sides' gamma explode in slightly different places. Delta-hedge that residual early in the day if
      at all, and stop once the expected move to expiry is smaller than the spread's width; or cap the hedge at the
      position's maximum payoff in BTC.
    - **What happens to gamma and vega?** For the binary both change sign at the strike (long gamma and long vega
      below, short above) and gamma grows like 1 / (time left) near the strike while vega fades to zero; the short
      spread carries the same shapes with the opposite sign and bounded size, so the net is small away from the
      strike and concentrated in a band of about the spread's width in the last hours.
    - **9 October, $1M, half-hourly hedge, no frictions.** Kalshi cheap by 1.4¢ at $123,000: the spread hedge
      with half-hourly delta returns about +$27,000 (the edge was $24,000; +$24,000 without any delta hedge)
      and is never more than about $38,000 under water; delta-hedging the binary alone ends near −$27,000 after falling to −$120,000; no hedge loses
      $1,000,000.
    - **A better way given the gamma explosion:** hedge statically with a spread no wider than Kalshi's $500
      strike spacing, do not chase the residual delta, and unwind both legs before the final hour; across 2026 that
      cuts the hedging error by more than two thirds. Wider spreads and finer delta hedging both do worse.

    **Limits.** The matching call spread does not exist on Deribit (its options expire at 08:00 UTC, Kalshi's at
    21:00 UTC); it is priced on Deribit's smile at Kalshi's expiry, as the frictionless exercise allows. Real
    trading would add Deribit's fee (0.03% of BTC per option, on two options per spread) and Kalshi's.
    """)
    return


if __name__ == "__main__":
    app.run()
