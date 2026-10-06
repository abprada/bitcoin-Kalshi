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
      busier than average). Results are shown **with and without** this adjustment, because it is a model.

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
def _(compare, deribit, mo, np, pd):
    ROOT = mo.notebook_dir()
    DATA, CLEAN, DERIVED = ROOT / "data", ROOT / "data" / "clean", ROOT / "data" / "derived"
    WINDOW_MIN = 15          # Deribit trades within this many minutes of a Kalshi snapshot form its smile
    MIN_DERIBIT_STRIKES = 3  # and they must cover at least 3 strikes, on both sides of the money

    # Deribit: afternoon trades in Friday expiries (the only ones whose date matches a Kalshi weekly)
    trades = deribit.load_trades(DATA / "deribit_btc_options.csv.gz")
    trades = trades[(trades.window == "us_afternoon") & (trades.expiry.dt.dayofweek == 4)
                    & trades.iv_mark.notna() & (trades.tau_years > 0)]
    by_expiry = {e: g for e, g in trades.groupby("expiry")}

    # Kalshi: clean snapshots in the same afternoon window, before the Deribit option with the same date expires
    snaps = pd.read_parquet(DERIVED / "iv_snapshots.parquet")
    snaps["expiry"] = snaps.ts + pd.to_timedelta(snaps.hours_to_expiry, unit="h")
    _minute = snaps.ts.dt.hour * 60 + snaps.ts.dt.minute
    snaps = snaps[snaps.clean & _minute.between(19 * 60 + 30, 22 * 60 + 29)].copy()
    snaps["deribit_expiry"] = (pd.to_datetime(snaps.expiry.dt.date.astype(str)).dt.tz_localize("UTC")
                               + pd.Timedelta(hours=8))
    snaps = snaps[snaps.ts < snaps.deribit_expiry]

    _spot = pd.read_parquet(CLEAN / "btc_spot_1m.parquet", columns=["ts", "spot"]).set_index("ts").spot
    hour_profile = deribit.hourly_variance(_spot)
    _k = pd.read_parquet(DERIVED / "iv_strikes.parquet",
                         columns=["snap", "strike", "mid", "D_used", "iv", "spot", "tau_years"],
                         filters=[("snap", "in", snaps.snap.tolist())])
    _k_by_snap = {s: g for s, g in _k.groupby("snap")}

    _rows, smiles = [], {}
    for _s in snaps.itertuples():
        _g = by_expiry.get(_s.deribit_expiry)
        if _g is None or _s.snap not in _k_by_snap:
            continue
        _sm = compare.smile_from_trades(_g[(_g.ts - _s.ts).abs() <= pd.Timedelta(minutes=WINDOW_MIN)])
        if len(_sm) < MIN_DERIBIT_STRIKES or not (_sm.log_moneyness.min() < 0 < _sm.log_moneyness.max()):
            continue
        _kk = _k_by_snap[_s.snap]
        _F, _tau_k = float(_kk.spot.iloc[0]), float(_kk.tau_years.iloc[0])
        _tau_d = (_s.deribit_expiry - _s.ts).total_seconds() / (365 * 86_400)
        _ratio = deribit.horizon_vol_ratio(_s.ts, _s.expiry, _s.deribit_expiry, hour_profile)
        _K = _kk.strike.to_numpy()
        _vol = compare.interp_vol(np.log(_K / _F), _sm.log_moneyness, _sm.vol)
        smiles[_s.snap] = dict(smile=_sm, F=_F, tau_k=_tau_k, tau_d=_tau_d, ratio=_ratio, ts=_s.ts,
                               event_ticker=_s.event_ticker, kalshi_hours=_s.hours_to_expiry)
        _rows.append(_kk.assign(
            ts=_s.ts, event_ticker=_s.event_ticker, kalshi_hours=_s.hours_to_expiry,
            deribit_hours=_tau_d * 8_760, horizon_ratio=_ratio, n_deribit_strikes=len(_sm),
            deribit_vol=_vol, deribit_vol_adj=_vol * _ratio,
            deribit_digital_adj=compare.digital_from_smile(_K, _F, _tau_k, _sm.log_moneyness, _sm.vol * _ratio),
            deribit_digital_raw=compare.digital_from_smile(_K, _F, _tau_d, _sm.log_moneyness, _sm.vol),
        ))
    matched = pd.concat(_rows, ignore_index=True)

    # Kalshi's bid and ask at each matched snapshot, from the clean quotes
    _q = []
    for _ev, _g in matched.groupby("event_ticker"):
        _qq = pd.read_parquet(CLEAN / "kalshi_weekly_quotes" / f"{_ev}.parquet", columns=["strike", "ts", "bid", "ask"])
        _q.append(_qq[_qq.ts.isin(_g.ts.unique())].assign(event_ticker=_ev))
    matched = matched.merge(pd.concat(_q, ignore_index=True), on=["event_ticker", "ts", "strike"], how="left")
    matched["log_moneyness"] = np.log(matched.strike / matched.spot)
    matched["moneyness_bucket"] = pd.cut(matched.log_moneyness, [-0.15, -0.05, -0.02, -0.005, 0.005, 0.02, 0.05, 0.15],
                                         labels=["5–15% below", "2–5% below", "0.5–2% below", "within 0.5%",
                                                 "0.5–2% above", "2–5% above", "5–15% above"])
    matched["tte_bucket"] = pd.cut(matched.kalshi_hours, [24, 48, 72, 120, 170],
                                   labels=["1–2 days", "2–3 days", "3–5 days", "5–7 days"])
    matched["vol_gap_raw"] = 100 * (matched.iv - matched.deribit_vol)
    matched["vol_gap_adj"] = 100 * (matched.iv - matched.deribit_vol_adj)
    matched["price_gap_c"] = 100 * (matched.mid - matched.deribit_digital_adj)
    return DERIVED, matched, smiles, snaps, trades


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
        ## 2 · What could be matched

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
        ## 3 · Comparing vols

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
def _(H, MUTED, W, alt, matched, mo, pd):
    _x = matched.dropna(subset=["price_gap_c"])
    _by_m = (_x.groupby("moneyness_bucket", observed=True)
             .agg(strikes=("mid", "size"), kalshi_mid_c=("mid", lambda s: 100 * s.median()),
                  deribit_c=("deribit_digital_adj", lambda s: 100 * s.median()),
                  gap_c=("price_gap_c", "median"), mean_gap_c=("price_gap_c", "mean"),
                  kalshi_spread_c=("ask", lambda s: 100 * (s - _x.loc[s.index, "bid"]).median()))
             .round(1).reset_index())
    _order = list(_by_m.moneyness_bucket.astype(str))
    price_gap_chart = (
        alt.Chart(_by_m.assign(moneyness_bucket=_by_m.moneyness_bucket.astype(str))).mark_bar(width=28)
        .encode(x=alt.X("moneyness_bucket:N", sort=_order, title="Kalshi strike relative to spot"),
                y=alt.Y("gap_c:Q", title="Kalshi mid − Deribit spread (cents, median)"),
                tooltip=["moneyness_bucket:N", alt.Tooltip("gap_c:Q", format="+.1f"), "strikes:Q"])
        .properties(title="Price gap across strikes (Deribit stretched to Kalshi's expiry)", width=W, height=H)
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED).encode(y="y:Q")
    )
    _abs = _x.price_gap_c.abs()
    _half = 50 * (_x.ask - _x.bid)
    mo.vstack([
        mo.md(f"""
        ## 4 · Comparing prices

        For every matched Kalshi strike K: Kalshi's mid price against the cost of Deribit's K − 250 / K + 250 call
        spread per $1, priced on Deribit's smile at Kalshi's horizon. Both are "the price of $1 if BTC ends above
        K", in cents.

        - The gaps are **small**: the typical difference is **{_abs.median():.1f}¢**, about the size of Kalshi's
          typical bid-ask spread ({2 * _half.median():.0f}¢). Deribit's price is within half a spread of Kalshi's mid
          for {(_abs <= _half).mean():.0%} of quotes.
        - Kalshi is a little **cheaper** than Deribit for strikes at and just below spot, and a little **dearer**
          above spot. Kalshi puts slightly more probability on BTC ending a bit lower, and slightly more on large
          rises, than Deribit does.
        """),
        price_gap_chart,
        mo.ui.table(_by_m.rename(columns={"moneyness_bucket": "strike vs spot"}), selection=None, show_download=False,
                    pagination=False),
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
    edges = pd.DataFrame([(c, (_y.edge_c > c).mean(), _y.edge_c[_y.edge_c > c].median()) for c in (0, 2, 5, 10)],
                         columns=["assumed Deribit cost (¢ per $1)", "share of Kalshi quotes still profitable",
                                  "median edge left (¢)"])
    edges["share of Kalshi quotes still profitable"] = edges["share of Kalshi quotes still profitable"].map("{:.1%}".format)
    edges["median edge left (¢)"] = (edges["median edge left (¢)"] - edges["assumed Deribit cost (¢ per $1)"]).round(1)
    _by_t = (_y.groupby("tte_bucket", observed=True).edge_c.apply(lambda e: (e > 5).mean())
             .map("{:.1%}".format).rename("share profitable at 5¢ Deribit cost").reset_index()
             .rename(columns={"tte_bucket": "Kalshi time to expiry"}))
    mo.vstack([
        mo.md(f"""
        ## 5 · Is there money on the table?

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
          contract. The table shows what survives under different assumptions.
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
    ## 6 · What this says so far

    **Q2b, are Kalshi's prices consistent with option prices?** Broadly yes. Once the two are put in the same
    units and the 13-hour expiry difference is accounted for:

    - prices typically differ by a cent or two, about the size of Kalshi's own bid-ask spread;
    - ATM vols differ by about a point, with Kalshi higher on roughly two days in three;
    - the clearest difference is the **shape**: Deribit charges more for crash protection (steeper downside skew);
    - the gaps that remain are mostly smaller than plausible trading costs, so they are not free money.

    **Possible explanations for what remains** (for the write-up): different users (US retail on Kalshi,
    professional and offshore on Deribit; Deribit does not serve US residents), so little capital can move between
    them; Kalshi's full collateral and fees; the 1¢ tick and $500 strikes on Kalshi; BRTI vs Deribit's index.

    **Caveats.** The expiry adjustment assumes both markets expect each hour to be as volatile as it has been
    historically. Deribit marks are Deribit's model values, not executable prices. The 15-minute matching window
    and the fee formula are assumptions worth varying.

    **Next.** Check robustness (window length, the `exact_0800UTC` window, trade prices instead of marks); get
    Deribit order-book quotes if possible, to measure its real costs; extend to probabilities for Q3 (Kalshi vs
    Deribit vs realized outcomes).
    """)
    return


if __name__ == "__main__":
    app.run()
