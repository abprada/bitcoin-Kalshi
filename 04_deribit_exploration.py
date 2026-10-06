import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="04 Deribit exploration")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    from lib import deribit

    return alt, deribit, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # 04 · Deribit exploration

    First look at the options market we compare Kalshi against (handout Q2b). Deribit is the largest BTC
    options exchange. The file `data/deribit_btc_options.csv.gz` (from a collaborator) holds individual
    **trades**, not continuous quotes: each row is one fill with its price, size, direction, Deribit's
    implied vol and Deribit's mark price (its own fair-value estimate) at that moment.

    Two things to keep in mind throughout:

    - **Deribit options are inverse.** They are quoted and settled in BTC, but price in BTC × index price is an
      ordinary USD option price, so the usual Black-Scholes formula applies (`lib/deribit.py`).
    - **Expiries do not line up.** Every Deribit option expires at **08:00 UTC**; Kalshi weeklies expire at
      **5pm ET** (21:00 or 22:00 UTC). A Friday Deribit expiry ends about 13 hours before the Kalshi weekly
      with the same date.

    This notebook reads `data/derived/iv_snapshots.parquet` and `iv_strikes.parquet`, so run
    `03_iv_surface.py` first. It writes nothing.
    """)
    return


@app.cell
def _(alt):
    # Chart styling: reference palette (slot 1 blue = Kalshi, slot 2 orange = Deribit), recessive gray chrome.
    INK, INK2, MUTED, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
    SLOT1, SLOT2 = "#2a78d6", "#eb6834"
    WINDOW_SCALE = alt.Scale(domain=["us_afternoon", "exact_0800UTC"], range=[SLOT1, SLOT2])
    SOURCE_SCALE = alt.Scale(domain=["Kalshi", "Deribit"], range=[SLOT1, SLOT2])
    W, H = 640, 200


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
                "bar": {"color": SLOT1, "cornerRadiusEnd": 2},
                "line": {"color": SLOT1, "strokeWidth": 2},
                "point": {"color": SLOT1, "size": 64, "filled": True},
            }
        }

    return H, MUTED, SLOT1, SLOT2, SOURCE_SCALE, W, WINDOW_SCALE


@app.cell
def _(deribit, mo, pd):
    ROOT = mo.notebook_dir()
    DERIVED = ROOT / "data" / "derived"
    trades = deribit.load_trades(ROOT / "data" / "deribit_btc_options.csv.gz")
    kalshi_snaps = pd.read_parquet(DERIVED / "iv_snapshots.parquet",
                                   columns=["snap", "event_ticker", "ts", "atm_iv", "hours_to_expiry", "clean"])
    kalshi_snaps["expiry"] = kalshi_snaps.ts + pd.to_timedelta(kalshi_snaps.hours_to_expiry, unit="h")
    return DERIVED, kalshi_snaps, trades


@app.cell
def _(mo, pd, trades):
    _e = trades.groupby("expiry").agg(first=("ts", "min"), n=("ts", "size"))
    _e["weekday"] = _e.index.day_name()
    _e["lead_days"] = (_e.index - _e["first"]).dt.total_seconds() / 86_400
    schedule = (_e.groupby("weekday", sort=False)
                .agg(expiries=("n", "size"), median_trades_per_expiry=("n", "median"),
                     earliest_trade_days_before_expiry=("lead_days", "max"))
                .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                .round(1).reset_index())
    _fri = trades[(trades.window == "us_afternoon")].expiry.dt.dayofweek.eq(4).mean()
    mo.vstack([
        mo.md(f"""
        ## 1 · How Deribit options work

        **The contract.** A Deribit BTC option is an ordinary **European** call or put: it can only be exercised at
        expiry, and it pays the difference between the settlement price and the strike. A call struck at $90,000
        pays $1,000 if BTC settles at $91,000. Unlike Kalshi's yes/no contracts, the payoff grows with the move.
        One contract covers 1 BTC and the smallest trade is 0.1 BTC. Settlement is in cash, against a 30-minute
        average of Deribit's own BTC index ending at expiry.

        **Paid in BTC.** Prices are quoted in BTC (`price_btc`) and the payoff is paid in BTC: (S − K) / S bitcoin for
        a call, worth exactly S − K dollars. So **price in BTC × index price** (`price_usd`) is an ordinary dollar
        option price, and the usual Black-Scholes formula applies.

        **Expiries.** Every expiry is at **08:00 UTC**. Deribit lists **daily** expiries (every day), **weekly**
        expiries (every Friday), plus monthly and quarterly ones (the last Friday of the month or quarter). Many
        expiries are live at once and new ones are listed as old ones expire. This is unlike Kalshi, where each
        weekly opens on Friday and lives exactly one week. A Friday Deribit expiry is listed weeks in advance.

        **What this file contains.** The collaborator kept only trades in two windows each day (`us_afternoon`
        around Kalshi's 5pm ET expiry, and `exact_0800UTC`, the hour before Deribit's expiry), and only expiries a
        few days ahead: daily expiries up to 3.5 days before they expire, Friday expiries up to 8.5 days before.
        So for any day we see trades in about five expiries at once; in the afternoon window
        {_fri:.0%} of trades are in a Friday expiry, the only ones whose date matches a Kalshi weekly.

        **Mark price.** `mark_price_btc` is Deribit's own estimate of an option's fair value at that moment, computed
        from its implied-vol surface and the index. It is not a trade. Deribit uses it to value open positions and
        to decide margin calls and liquidations. Because it moves smoothly, it is a steadier reference than any
        single trade.
        """),
        mo.ui.table(schedule, selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell
def _(mo, np, pd, trades):
    _t = trades
    _span = _t.groupby("window").ts.agg(lambda x: f"{x.dt.strftime('%H:%M').min()}–{x.dt.strftime('%H:%M').max()} UTC")
    _ours = trades.dropna(subset=["iv_deribit", "iv_mark"])
    inventory = pd.DataFrame([
        ("Trades", f"{len(_t):,}"),
        ("Days", f"{_t.date.nunique()} ({_t.date.min()} to {_t.date.max()})"),
        ("Window `us_afternoon`", f"{_span['us_afternoon']}, around Kalshi's 5pm ET expiry"),
        ("Window `exact_0800UTC`", f"{_span['exact_0800UTC']}, the hour before Deribit's 08:00 UTC expiry"),
        ("Expiries", f"{_t.expiry.nunique()} distinct, all at 08:00 UTC, up to {_t.tau_years.max() * 365:.1f} days ahead"),
        ("Calls / puts", f"{(_t.cp == 'C').sum():,} / {(_t.cp == 'P').sum():,}"),
        ("Median trade size", f"{_t.amount.median():.1f} BTC (mean {_t.amount.mean():.1f}, max {_t.amount.max():,.0f})"),
        ("`iv` placeholders (0 or 999)", f"{int((~_t.iv_valid).sum()):,} ({(~_t.iv_valid).mean():.1%})"),
        ("Exact duplicate rows", f"{int(_t.exact_dup.sum()):,} ({_t.exact_dup.mean():.1%}); no trade id, so kept"),
        ("`price_usd` = `price_btc` × `index_price`",
         f"max relative error {((_t.price_btc * _t.index_price - _t.price_usd).abs() / _t.price_usd).max():.0e}"),
        ("Deribit `iv` vs our inversion of its mark price",
         f"median gap {100 * np.median(np.abs(_ours.iv_deribit - _ours.iv_mark)):.2f} vol points"),
    ], columns=["", "value"])
    mo.vstack([
        mo.md("## 2 · Inventory"),
        mo.ui.table(inventory, selection=None, show_download=False, pagination=False),
        mo.md("""
        The data is internally consistent. The `iv` column needs its placeholders removed (`iv_deribit` is NaN
        there). Deribit's vols match a plain Black-Scholes inversion with the index as the forward, so we can
        compute vols for any Deribit price ourselves (`iv_mark`).
        """),
    ])
    return


@app.cell
def _(H, W, WINDOW_SCALE, alt, mo, pd, trades):
    _daily = trades.groupby(["date", "window"]).size().rename("trades").reset_index()
    _daily["date"] = pd.to_datetime(_daily.date)
    per_day_chart = (
        alt.Chart(_daily).mark_line(strokeWidth=1.6)
        .encode(x=alt.X("date:T", title=None), y=alt.Y("trades:Q", title="trades per day"),
                color=alt.Color("window:N", scale=WINDOW_SCALE, title=None),
                tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), "window:N", alt.Tooltip("trades:Q", format=",")])
        .properties(title="Trades per day, by window", width=W, height=H)
    )
    _tte = (trades.assign(days=(trades.tau_years * 365 // 0.25) * 0.25)
            .groupby(["window", "days"]).size().rename("trades").reset_index())
    tte_chart = (
        alt.Chart(_tte).mark_bar(opacity=0.85, width=4)
        .encode(x=alt.X("days:Q", title="days to expiry at the trade (0.25-day bins)"),
                y=alt.Y("trades:Q", title="trades", stack=None),
                color=alt.Color("window:N", scale=WINDOW_SCALE, title=None),
                tooltip=["window:N", alt.Tooltip("days:Q", title="days from"), alt.Tooltip("trades:Q", format=",")])
        .properties(title="Time to expiry at the trade", width=W, height=H)
    )
    mo.vstack([
        mo.md(f"""
        ## 3 · When Deribit trades

        Every day has both windows. The median day has {int(trades.groupby('date').size().median()):,} trades,
        with spikes on busy days. {(trades.tau_years * 365 < 4).mean():.0%} of trades are in options with
        **less than four days** left. In the `exact_0800UTC` window,
        {(trades[trades.window == 'exact_0800UTC'].tau_years * 8760 < 1).mean():.0%} of trades are in the option
        expiring within the hour; the rest are in later expiries.
        """),
        per_day_chart,
        tte_chart,
    ])
    return


@app.cell
def _(H, W, alt, mo, trades):
    _x = trades[trades.log_moneyness.abs() < 0.2]
    _x = ((_x.strike / _x.index_price).round(2).rename("moneyness").value_counts().rename("trades")
          .reset_index().sort_values("moneyness"))
    strike_chart = (
        alt.Chart(_x).mark_bar(width=5)
        .encode(x=alt.X("moneyness:Q", title="moneyness (strike / index, rounded to 0.01)", scale=alt.Scale(zero=False)),
                y=alt.Y("trades:Q", title="trades"),
                tooltip=[alt.Tooltip("moneyness:Q", format=".2f"), alt.Tooltip("trades:Q", format=",")])
        .properties(title="Where trades happen: strike relative to the index", width=W, height=H)
    )
    _near = (trades.log_moneyness.abs() < 0.01).mean()
    _within5 = (trades.log_moneyness.abs() < 0.05).mean()
    mo.vstack([
        mo.md(f"""
        ## 4 · Strikes traded

        Deribit strikes are round thousands (Kalshi's are $500 apart at …,249.99). {_near:.0%} of trades are
        within 1% of the index and {_within5:.0%} within 5%, so Deribit is most informative near the money,
        the same region where Kalshi's vols are most reliable.
        """),
        strike_chart,
    ])
    return


@app.cell
def _(H, W, alt, mo, trades):
    _x = trades.dropna(subset=["iv_deribit", "iv_mark"])
    _x = _x.assign(gap=100 * (_x.iv_deribit - _x.iv_mark))
    _h = ((_x.gap.clip(-10, 10) // 0.25 * 0.25).rename("gap").value_counts().rename("trades")
          .reset_index().sort_values("gap"))
    gap_chart = (
        alt.Chart(_h).mark_bar(width=4)
        .encode(x=alt.X("gap:Q", title="trade vol − mark vol (vol points, 0.25 bins; clipped at ±10)"),
                y=alt.Y("trades:Q", title="trades"),
                tooltip=[alt.Tooltip("gap:Q", title="from"), alt.Tooltip("trades:Q", format=",")])
        .properties(title="How far trades print from Deribit's mark", width=W, height=H)
    )
    mo.vstack([
        mo.md(f"""
        ## 5 · Trade prices vs mark prices

        **The question:** how much can we trust one trade as a measure of Deribit's vol?

        Every trade has a price, and we can turn that price into a vol. The mark price (section 1) can be turned
        into a vol too. If trades happened exactly at fair value, the two would match. They don't quite, because
        a buyer usually pays a little above fair value (they cross the bid-ask spread) and a seller receives a little
        below it. The chart shows, for every trade, its vol minus the mark vol at that same moment:

        - The typical trade is **{_x.gap.abs().median():.1f} vol points** away from the mark. Only
          {(_x.gap.abs() > 5).mean():.1%} are more than 5 points away.
        - Buyers pay a bit above the mark (median {_x[_x.direction == 'buy'].gap.median():+.1f} points) and sellers
          receive a bit below it (median {_x[_x.direction == 'sell'].gap.median():+.1f}), which is the spread at work.

        **What it means for us:** a single trade's vol is off by about a point, so the comparisons below use the
        **median over many trades**. A typical error of one point is small next to the gap with Kalshi in section 6.
        """),
        gap_chart,
    ])
    return


@app.cell
def _(DERIVED, deribit, kalshi_snaps, pd, trades):
    # Daily ATM vol in the us_afternoon window, Deribit vs Kalshi, matched on the expiry date
    # (Deribit Friday 08:00 UTC vs the Kalshi weekly expiring 5pm ET the same Friday).
    _d = trades[trades.window == "us_afternoon"].assign(expiry_date=lambda x: x.expiry.dt.date,
                                                       day=lambda x: x.ts.dt.date)
    _deribit = deribit.atm_vol(_d, by=["day", "expiry_date"])
    _minute = kalshi_snaps.ts.dt.hour * 60 + kalshi_snaps.ts.dt.minute
    _k = kalshi_snaps[kalshi_snaps.clean & _minute.between(19 * 60 + 30, 22 * 60 + 29)]
    _kalshi = (_k.assign(day=_k.ts.dt.date, expiry_date=_k.expiry.dt.date)
               .groupby(["day", "expiry_date"])
               .agg(kalshi_atm=("atm_iv", "median"), kalshi_hours=("hours_to_expiry", "median"),
                    kalshi_expiry=("expiry", "first"))
               .reset_index())
    atm_daily = (_deribit.rename(columns={"atm_iv": "deribit_atm", "hours_to_expiry": "deribit_hours"})
                 .merge(_kalshi, on=["day", "expiry_date"]))
    atm_daily["gap_pts"] = 100 * (atm_daily.kalshi_atm - atm_daily.deribit_atm)

    # How much of the gap the 13 extra hours alone would produce, if each hour of the week carried its usual
    # (realized) variance: Kalshi's period includes Friday 08:00 UTC to 5pm ET, Deribit's stops at 08:00 UTC.
    _spot = pd.read_parquet(DERIVED.parent / "clean" / "btc_spot_1m.parquet", columns=["ts", "spot"]).set_index("ts").spot
    hour_profile = deribit.hourly_variance(_spot)
    _start = atm_daily.kalshi_expiry - pd.to_timedelta(atm_daily.kalshi_hours, unit="h")
    _deribit_expiry = pd.to_datetime(atm_daily.expiry_date.astype(str)).dt.tz_localize("UTC") + pd.Timedelta(hours=8)
    atm_daily["hours_ratio"] = [deribit.horizon_vol_ratio(s, k, d, hour_profile)
                                for s, k, d in zip(_start, atm_daily.kalshi_expiry, _deribit_expiry)]
    atm_daily["from_extra_hours_pts"] = 100 * atm_daily.deribit_atm * (atm_daily.hours_ratio - 1)
    atm_daily["day"] = pd.to_datetime(atm_daily.day)
    return atm_daily, hour_profile


@app.cell
def _(H, MUTED, SLOT1, SLOT2, SOURCE_SCALE, W, alt, atm_daily, hour_profile, mo, pd):
    _long = atm_daily.melt(id_vars=["day"], value_vars=["kalshi_atm", "deribit_atm"], var_name="source", value_name="vol")
    _long["source"] = _long.source.map({"kalshi_atm": "Kalshi", "deribit_atm": "Deribit"})
    atm_chart = (
        alt.Chart(_long).mark_line(strokeWidth=1.6, point=alt.OverlayMarkDef(size=14))
        .encode(x=alt.X("day:T", title=None), y=alt.Y("vol:Q", title="ATM implied vol", axis=alt.Axis(format="%")),
                color=alt.Color("source:N", scale=SOURCE_SCALE, title=None),
                tooltip=[alt.Tooltip("day:T", format="%Y-%m-%d"), "source:N", alt.Tooltip("vol:Q", format=".1%")])
        .properties(title="ATM implied vol, Kalshi vs Deribit (US afternoon, same expiry date)", width=W, height=H)
    )
    gap_chart_daily = (
        alt.Chart(atm_daily[["day", "gap_pts", "kalshi_hours", "deribit_hours"]]).mark_circle(size=40, opacity=0.7)
        .encode(x=alt.X("kalshi_hours:Q", title="Kalshi hours to expiry", scale=alt.Scale(reverse=True)),
                y=alt.Y("gap_pts:Q", title="Kalshi − Deribit (vol points)"),
                tooltip=[alt.Tooltip("day:T", format="%Y-%m-%d"), alt.Tooltip("gap_pts:Q", format="+.1f"),
                         alt.Tooltip("kalshi_hours:Q", format=".0f"), alt.Tooltip("deribit_hours:Q", format=".0f")])
        .properties(title="Daily gap by time to expiry", width=W, height=H)
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    )

    _days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    _p = pd.DataFrame({"hour_of_week": range(168), "rel_variance": hour_profile.to_numpy() / hour_profile.mean()})
    _p["label"] = [f"{_days[h // 24]} {h % 24:02d}:00 UTC" for h in _p.hour_of_week]
    _p["period"] = ["Kalshi only: Fri 08:00 UTC to 5pm ET" if 4 * 24 + 8 <= h < 4 * 24 + 21 else "rest of the week"
                    for h in _p.hour_of_week]
    profile_chart = (
        alt.Chart(_p).mark_bar(width=2.5)
        .encode(x=alt.X("hour_of_week:Q", title="hour of the week (UTC)",
                        axis=alt.Axis(values=list(range(0, 168, 24)), labelExpr=f"{_days}[datum.value / 24]")),
                y=alt.Y("rel_variance:Q", title="variance vs weekly average"),
                color=alt.Color("period:N", title=None, scale=alt.Scale(
                    domain=["rest of the week", "Kalshi only: Fri 08:00 UTC to 5pm ET"], range=[SLOT1, SLOT2])),
                tooltip=["label:N", alt.Tooltip("rel_variance:Q", format=".2f", title="× average")])
        .properties(title="How much BTC moves in each hour of the week (realized, Oct 2025 – Sep 2026)", width=W, height=H)
    )
    _fri = _p[_p.period != "rest of the week"].rel_variance.mean()
    _wkend = _p[_p.hour_of_week >= 5 * 24].rel_variance.mean()

    _b = atm_daily.assign(bucket=pd.cut(atm_daily.kalshi_hours, [24, 48, 72, 120, 170],
                                        labels=["1–2 days", "2–3 days", "3–5 days", "5–7 days"]))
    gap_by_tte = (_b.groupby("bucket", observed=True)
                  .agg(days=("gap_pts", "size"), kalshi_hours=("kalshi_hours", "median"),
                       deribit_hours=("deribit_hours", "median"), gap=("gap_pts", "mean"),
                       from_extra_hours=("from_extra_hours_pts", "mean"))
                  .assign(unexplained=lambda d: d.gap - d.from_extra_hours)
                  .round(1).reset_index().rename(columns={"bucket": "Kalshi time to expiry"}))
    _corr = atm_daily[["kalshi_atm", "deribit_atm"]].corr().iloc[0, 1]
    _gap, _hrs = atm_daily.gap_pts.mean(), atm_daily.from_extra_hours_pts.mean()
    _ex = atm_daily.iloc[(atm_daily.kalshi_hours - 96).abs().argmin()]
    mo.vstack([
        mo.md(f"""
        ## 6 · First comparison: ATM vol

        **What is compared.** One number per day for each market, taken at the same time of day:

        - **When:** the `us_afternoon` window, 19:30–22:29 UTC (about 3:30–6:30pm in New York), on the same day
          for both markets.
        - **Which options:** the Deribit Friday expiry and the Kalshi weekly with the **same expiry date**.
        - **At the money:** only Deribit trades whose strike is within 1% of the index at the moment of the trade.
          With BTC at $90,000 that means strikes from $89,100 to $90,900, which in practice is the one or two
          round-thousand strikes nearest the price. Deribit's number is the median vol of those trades.
        - **Kalshi's number** is the median ATM vol of its clean snapshots (one every 15 minutes) in the same window.

        **How time to expiry lines up.** It doesn't, exactly. Both are measured from the same afternoon, but
        Deribit's option ends at 08:00 UTC Friday and Kalshi's at 5pm ET Friday. Example: on
        {_ex.day:%A %b %d}, the Kalshi weekly had {_ex.kalshi_hours:.0f} hours left and the Deribit option had
        {_ex.deribit_hours:.0f} hours left. The charts and table below sort days by **Kalshi's** hours to expiry; Deribit's
        are always about 13 hours fewer (both are shown in the table).

        **Result over {len(atm_daily)} matched days.** The two move together closely (correlation
        **{_corr:.2f}**), but Kalshi is higher on {(atm_daily.gap_pts > 0).mean():.0%} of days, by
        **{_gap:+.1f} vol points** on average (median {atm_daily.gap_pts.median():+.1f}), and more so near expiry.
        """),
        atm_chart,
        gap_chart_daily,
        mo.md(f"""
        **Part of the gap: Kalshi's 13 extra hours are busy ones.** Implied vol is quoted per year, as an average
        over the option's remaining life. Kalshi's remaining life includes Friday from 08:00 UTC to 5pm ET, which is
        the US trading day, and BTC moves more then: those hours carry **{_fri:.1f}×** the average hour's variance
        (orange below), while weekend hours carry {_wkend:.1f}×. Kalshi's average therefore includes more busy
        hours than Deribit's, even if both markets agreed hour by hour.

        That also explains why the gap grows toward expiry: with a week left, 13 extra hours are a small share of
        the remaining time; with a day left, they are a large share.

        To size this, `from_extra_hours` is the gap we would see if both markets priced every hour at its usual
        realized variance (`lib/deribit.py`, `horizon_vol_ratio`). It explains **{_hrs:+.1f}** of the
        **{_gap:+.1f}** points on average, about {_hrs / _gap:.0%}.
        """),
        profile_chart,
        mo.ui.table(gap_by_tte, selection=None, show_download=False, pagination=False),
        mo.md("""
        **The rest of the gap**, about two points, is still open. Candidates to test in `05`:

        - **Demand on Kalshi.** Kalshi is a US-regulated exchange used mostly by retail traders. If they mostly buy contracts (betting on moves)
          and few professionals sell, prices sit above fair value. Kalshi's ATM vol is also above BTC's realized vol
          (notebook 03), which fits.
        - **Limits to arbitrage.** Correcting the gap means selling on Kalshi and buying on Deribit. That needs
          accounts on both (Deribit does not serve US residents), capital fully posted on Kalshi, and paying both
          venues' fees and spreads. A gap smaller than those costs can persist.
        - **Measurement.** Kalshi's ATM vol comes from a ladder of $500 strikes and 1¢ prices (method 2), Deribit's
          from trades. The tail assumption moves Kalshi's ATM vol by about a point (see the method note).
        - **Settlement index.** BRTI vs Deribit's index differ by a few basis points; too small to matter here.
        """),
    ])
    return


@app.cell
def _(atm_daily, mo):
    _opts = {f"{r.day:%Y-%m-%d} (expiry {r.expiry_date}, {r.n_trades} ATM trades)": i
             for i, r in atm_daily.sort_values("day").iterrows()}
    _default = max(_opts, key=lambda k: atm_daily.loc[_opts[k], "n_trades"])
    day_select = mo.ui.dropdown(options=_opts, value=_default, label="Day")
    return (day_select,)


@app.cell
def _(DERIVED, H, MUTED, SLOT1, SLOT2, W, alt, atm_daily, day_select, kalshi_snaps, mo, pd, trades):
    _r = atm_daily.loc[day_select.value]
    _day, _exp = _r.day.date(), _r.expiry_date
    _d = trades[(trades.window == "us_afternoon") & (trades.ts.dt.date == _day) & (trades.expiry.dt.date == _exp)
                & trades.iv_deribit.notna() & (trades.log_moneyness.abs() < 0.12)]
    _d = _d.assign(moneyness=_d.strike / _d.index_price, vol=_d.iv_deribit)
    _marks = (_d.groupby("strike").agg(moneyness=("moneyness", "median"), vol=("iv_mark", "median")).reset_index())
    # the clean Kalshi snapshot for the same expiry closest to the median Deribit trade time
    _cand = kalshi_snaps[kalshi_snaps.clean & (kalshi_snaps.expiry.dt.date == _exp)]
    _snap = _cand.iloc[(_cand.ts - _d.ts.median()).abs().argsort()[:1]]
    _k = pd.read_parquet(DERIVED / "iv_strikes.parquet", columns=["snap", "moneyness", "iv", "D_used"],
                         filters=[("snap", "==", int(_snap.snap.iloc[0]))])
    _k = _k[_k.D_used.between(0.05, 0.95)]
    _x = alt.X("moneyness:Q", title="moneyness (strike / spot)", scale=alt.Scale(zero=False))
    _y = alt.Y("vol:Q", title="implied vol", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False))
    smile_chart = (
        alt.Chart(_d).mark_circle(size=22, opacity=0.35, color=SLOT2)
        .encode(x=_x, y=_y, tooltip=[alt.Tooltip("strike:Q", format=","), "cp:N", alt.Tooltip("vol:Q", format=".1%"),
                                     alt.Tooltip("amount:Q", title="size (BTC)"), "direction:N"])
        + alt.Chart(_marks).mark_line(color=MUTED, strokeDash=[4, 3], strokeWidth=1.5).encode(x=_x, y=_y)
        + alt.Chart(_k.rename(columns={"iv": "vol"})).mark_line(color=SLOT1, point=True)
        .encode(x=_x, y=_y, tooltip=[alt.Tooltip("moneyness:Q", format=".3f"), alt.Tooltip("vol:Q", format=".1%")])
    ).properties(title=f"Smiles on {_day}: Kalshi weekly vs Deribit, expiry {_exp}", width=W, height=H + 60)
    mo.vstack([
        mo.md(f"""
        ## 7 · One day's smiles

        Blue: the Kalshi smile from the clean snapshot at {_snap.ts.iloc[0]:%H:%M} UTC
        ({_snap.hours_to_expiry.iloc[0]:.0f} hours to Kalshi's expiry). Orange dots: every Deribit trade in that
        expiry during the afternoon window ({len(_d)} trades, sized equally). Dashed gray: Deribit's mark vol per
        strike. Pick another day to compare.
        """),
        day_select,
        smile_chart,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · Next steps

    1. **Match the horizons** (`05`): put Kalshi and Deribit on the same time to expiry, for example by
       interpolating Deribit's total variance between its expiries, or by comparing Deribit with Kalshi
       snapshots 13 hours earlier relative to each one's expiry.
    2. **Price-level test** (handout Q2b): build Kalshi-equivalent binaries from Deribit call spreads,
       [C(K₁) − C(K₂)] / (K₂ − K₁), at the Kalshi strikes, and compare with Kalshi's bid and ask after fees.
    3. **Explain the gap**: Friday-session variance, fees and spreads, BRTI vs Deribit's index, who trades where.
    """)
    return


if __name__ == "__main__":
    app.run()
