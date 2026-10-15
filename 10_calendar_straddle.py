import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="10 Calendar straddle backtest")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    from lib import calendar_straddle as cs
    from lib import cleaning

    return alt, cleaning, cs, mo, np, pd


@app.cell(hide_code=True)
def _(alt):
    # Chart styling as in 05/06 (slot 1 blue, slot 2 orange, slot 3 aqua).
    INK, INK2, MUTED, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
    SLOT1, SLOT2, SLOT3 = "#2a78d6", "#eb6834", "#1baf7a"
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

    return H, MUTED, SLOT1, SLOT2, SLOT3, W


@app.cell(hide_code=True)
def _(cleaning, cs, mo, np, pd):
    ROOT = mo.notebook_dir()
    D = ROOT / "data" / "derived" / "strategies"
    FIG = ROOT / "outputs" / "figures"
    daily_all = pd.read_parquet(D / "calendar_daily.parquet")
    daily_all["gross"] = daily_all.options + daily_all.hedge + daily_all.funding
    variants = pd.read_parquet(D / "calendar_variants.parquet").set_index("variant").description
    trades = pd.read_parquet(D / "calendar_trades.parquet")
    term = pd.read_parquet(D / "deribit_term_structure.parquet")
    panel = pd.read_parquet(D / "vol_panel.parquet")
    half_spreads = pd.read_parquet(D / "deribit_half_spreads.parquet")
    _spot = cleaning.clean_spot(pd.read_parquet(ROOT / "data" / "raw" / "btc_spot" / "btcusd_1m.parquet")).spot
    _r = np.log(_spot).diff()
    day_rv = np.sqrt((_r**2).groupby(_r.index.normalize()).sum() * 365)
    START, END = "2026-01-15", "2026-10-01"
    # the three stress windows named in the plan (early Feb, early June, late Aug 2026)
    JUMPS = {"early Feb (2–8)": pd.date_range("2026-02-02", "2026-02-08", tz="UTC"),
             "early Jun (1–7)": pd.date_range("2026-06-01", "2026-06-07", tz="UTC"),
             "late Aug (20–31)": pd.date_range("2026-08-20", "2026-08-31", tz="UTC")}


    def summarize(period: str) -> pd.DataFrame:
        rows = []
        for v, g in daily_all[daily_all.period == period].groupby("variant", sort=False):
            g = g.set_index("t")
            n, gr = cs.summary(g.pnl, JUMPS), cs.summary(g.gross)
            rows.append({"variant": v, "what": variants.get(v, ""), "gross $/day": gr["mean"],
                         "gross Sharpe": gr["sharpe"], "option costs $/day": g.opt_cost.mean(),
                         "hedge costs $/day": g.perp_cost.mean(), "net $/day": n["mean"], "net Sharpe": n["sharpe"],
                         "worst day $": n["worst"], "worst day": n["worst_day"], "max drawdown $": n["max_dd"],
                         "share of days up": n["hit"], **{k: n[f"jump_{k}"] for k in JUMPS}})
        return pd.DataFrame(rows)


    table_2026 = summarize("2026")
    table_q4 = summarize("2025Q4")
    table_pre = summarize("pre2026")
    base = table_2026.set_index("variant").loc["base"]
    best = table_2026.sort_values("net $/day", ascending=False).iloc[0]
    best_gross = table_2026.sort_values("gross $/day", ascending=False).iloc[0]
    return (D, END, FIG, JUMPS, START, base, best, best_gross, daily_all, day_rv, half_spreads, panel,
            table_2026, table_pre, table_q4, term, trades)


@app.cell(hide_code=True)
def _(base, best, best_gross, daily_all, mo, table_2026):
    _n_pos = int((table_2026["net $/day"] > 0).sum())
    _days = int(((daily_all.period == "2026") & (daily_all.variant == "base")).sum())
    mo.md(f"""
    # 10 · Deribit calendar straddle: long the week, short the day

    *A rules-based version of the trade, on Deribit data, 15 January – 30 September 2026 ({_days} days). Built by
    `08_strategy_data.py` (`lib/calendar_straddle.py`); this notebook only reads its results.*

    **The trade.** Every day at 08:00 UTC, buy one at-the-money straddle on the Deribit expiry about 7 days out and
    sell about 2.6 at-the-money straddles on the next daily expiry (24 hours out), so that the two vegas match.
    Delta-hedge the book every hour with the BTC perpetual. Buy the short leg back 4 hours before it expires, sell
    the long leg after a day, and repeat. The bet: very short-dated options are rich (Kalshi's dailies price their
    last hours well above the vol BTC then realizes), so selling them and hedging the vega with a longer expiry
    should earn that premium.

    **Answer: no. It loses money before costs, and costs make it much worse.**

    1. **Deribit's daily options carry no short-dated premium.** Their implied vol *falls* into expiry (about 37% with
       1–4 days left, 33–35% in the last 12 hours, weekday expiries) and stays below the vol BTC then realizes.
       Kalshi's rises gently. This is the
       "first check" of the plan, and it fails: Deribit's dailies expire at 08:00 UTC after the quiet night, and
       they are priced for it.
    2. **Gross of all costs the base case makes {base['gross $/day']:+,.0f} $ a day** (Sharpe {base['gross Sharpe']:.1f}),
       **net {base['net $/day']:+,.0f} $ a day** (Sharpe {base['net Sharpe']:.1f}). Option fees and spreads cost
       {base['option costs $/day']:,.0f} $ a day and hourly hedging {base['hedge costs $/day']:,.0f} $: Deribit's
       0.03%-of-BTC fee is about 4% of a daily ATM option's price, paid on 2.6 straddles each way every day.
    3. **{_n_pos} of the {len(table_2026)} variations are profitable after costs.** The best gross version
       (`{best_gross['variant']}`: {best_gross['what']}) makes {best_gross['gross $/day']:+,.0f} $ a day before
       costs, from the quiet overnight hours, and {best_gross['net $/day']:+,.0f} $ after. The least bad after costs
       is `{best['variant']}` at {best['net $/day']:+,.0f} $ a day.
    4. **The jump days are where short gamma bleeds**: the early-February crash week costs the base case
       {base['early Feb (2–8)']:+,.0f} $ and late August {base['late Aug (20–31)']:+,.0f} $; the single worst
       day ({base['worst day']}) is an intraday −3% / +3% whipsaw that an hourly hedge cannot follow.

    P&L is in dollars per long straddle (1 BTC call + 1 BTC put on the long leg).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data and method

    | | |
    |---|---|
    | **Deribit options** | Every BTC option trade from Deribit's public history API (`00_fetch_deribit.py`, or `00_fetch_deribit_all.py` cut to the same expiries), expiries up to 15 days: 2.7 million trades in October 2025 – September 2026, about 13× the collaborator's file, which kept only 07:00–08:00 and 19:30–22:30 UTC; 5.6 million from 15 November 2024, the start of our Kalshi data, for the longer check. All 200,000 of the collaborator's trades in these expiries are in it, with identical marks and vols |
    | **Option values** | Each trade carries Deribit's mark price and mark vol. Per expiry and hour, the trades within ±30 minutes give Deribit's smile (a quadratic in log-moneyness per root year, at least 3 strikes around the forward); its forward comes from inverting the mark price at the mark vol. Any option is then valued by Black-Scholes at its fitted vol (`lib/deribit_marks.py`). An hour without trades in an expiry reuses its last fit (at most 3 hours old) with the current index |
    | **Hedge** | BTC-PERPETUAL 1-minute prices, hourly funding (Deribit API). Black-Scholes deltas at the fitted vols |
    | **Settlement** | Deribit's delivery price (its 30-minute index average before 08:00 UTC); Oct–Nov 2025 rebuilt from the index on trades |

    **Costs, as in the plan.** Each option trade pays the taker fee, 0.03% of the index per option capped at 12.5% of
    its price, plus half the bid-ask spread. We have no Deribit quotes, but a taker buy fills at the ask and a taker
    sell at the bid, so the signed gap between the trade price and Deribit's mark is the half-spread actually paid;
    its median by time to expiry and moneyness is applied (table below; about $5–8 per at-the-money option, 0.1–0.4
    vol points). The perpetual pays 0.05% of notional per hedge trade (taker), or 0 for a maker hedge. Settlement
    at expiry is free.
    """)
    return


@app.cell(hide_code=True)
def _(half_spreads, mo):
    _t = half_spreads[half_spreads.n >= 100].rename(columns={"tb": "days to expiry", "sb": "|sd| from forward", "hs_usd": "half-spread $",
                                      "hs_frac": "as share of price", "hs_vol": "in vol points", "n": "trades"})
    mo.accordion({"Measured Deribit half-spreads (trade price vs mark)": mo.ui.table(
        _t.round(3), selection=None, show_download=False, pagination=False)})
    return


@app.cell(hide_code=True)
def _(END, FIG, H, SLOT1, SLOT2, SLOT3, START, W, alt, mo, np, panel, pd, term):
    # The plan's first check: Deribit's ATM term structure vs Kalshi's and vs the vol BTC realized afterwards
    _edges = [0, 4, 8, 12, 18, 24, 36, 48, 72, 96, 120, 168]
    # Tuesday-Friday expiries only: Saturday and Sunday dailies span quiet weekend nights (about 23-29% vol) and
    # Kalshi's dailies run Monday to Friday, so pooling them would drag Deribit's short end down
    _g = term[(term.t >= START) & (term.t < END) & (term.tau_years * 8760 <= 168)
              & term.expiry.dt.dayofweek.isin([1, 2, 3, 4])].assign(h=lambda d: pd.cut(d.tau_years * 8760, _edges))
    _d = _g.groupby("h", observed=True).agg(implied=("atm_iv", "median"), realized=("rv_to_expiry", "median"),
                                            implied_rms=("atm_iv", lambda x: np.sqrt((x**2).mean())),
                                            realized_rms=("rv_to_expiry", lambda x: np.sqrt((x**2).mean())),
                                            n=("atm_iv", "size"))
    # Kalshi: dailies up to 24 hours left, weeklies beyond (each where it is the contract trading)
    _kp = panel[((panel.kind == "daily") & (panel.hours_left <= 24)) | ((panel.kind == "weekly") & (panel.hours_left > 24))]
    _k = _kp.assign(h=pd.cut(_kp.hours_left, _edges)).groupby("h", observed=True).atm_iv.median()
    _rows = []
    for _h, _r in _d.iterrows():
        _mid = (_h.left + _h.right) / 2
        _rows += [{"hours": _mid, "series": "Deribit implied", "vol": _r.implied},
                  {"hours": _mid, "series": "BTC realized to Deribit expiry", "vol": _r.realized}]
        if _h in _k.index:
            _rows.append({"hours": _mid, "series": "Kalshi implied", "vol": _k[_h]})
    _long = pd.DataFrame(_rows)
    _dom = ["Kalshi implied", "Deribit implied", "BTC realized to Deribit expiry"]
    fig_term = (
        alt.Chart(_long).mark_line(point=True)
        .encode(x=alt.X("hours:Q", title="hours to expiry", scale=alt.Scale(reverse=True)),
                y=alt.Y("vol:Q", title="annualized vol (median)", axis=alt.Axis(format="%"),
                        scale=alt.Scale(zero=False)),
                color=alt.Color("series:N", title=None, scale=alt.Scale(domain=_dom, range=[SLOT1, SLOT2, SLOT3])),
                tooltip=["series:N", alt.Tooltip("hours:Q", format=".0f"), alt.Tooltip("vol:Q", format=".1%")])
        .properties(title="Figure 1 · ATM implied vol by time to expiry: Kalshi edges up near expiry, Deribit falls",
                    width=W, height=H)
    )
    fig_term.save(FIG / "10_fig1_term_structure.png", scale_factor=2)
    _ratio = (_d.implied_rms / _d.realized_rms)
    _k6 = panel[(panel.kind == "daily") & (panel.hours_left <= 6)].dropna(subset=["rv_to_expiry"])
    term_stats = dict(d_week=_d.implied.iloc[-3:].median(), d_last12=_d.implied.iloc[:3].median(),
                      ratio_lo=_ratio.min(), ratio_hi=_ratio.max(), k_last=_k.iloc[0], k_week=_k.iloc[-1],
                      k6_implied=_k6.atm_iv.mean(), k6_realized=np.sqrt((_k6.rv_to_expiry**2).mean()))
    mo.vstack([
        mo.md(f"""
        ## 1 · First check: is there a short-dated premium on Deribit?

        The trade needs the 1-day straddle to be expensive relative to the 7-day one *and* to what BTC then does.
        Figure 1 pools Deribit's fitted ATM vol for every Tuesday–Friday expiry and hour (Jan 15 – Sep 30, 2026) by
        hours left, with the vol BTC actually realized from that hour to that expiry (1-minute returns), and Kalshi's
        ATM vol from its hourly snapshots (dailies up to 24 hours left, weeklies beyond; medians per bucket). Every
        Kalshi snapshot with an ATM vol is used: 05's strikes-per-SD rule keeps only high-vol moments near expiry and
        made `03`'s term structure spike (88% in the last hours); unfiltered, the rise is gentle.

        - **Kalshi** rises gently into expiry: {term_stats['k_week']:.0%} a week out, {term_stats['k_last']:.0%} in the
          last 4 hours. In their last 6 hours Kalshi's dailies price {term_stats['k6_implied']:.0%} against
          {term_stats['k6_realized']:.0%} realized: Kalshi's last hours are rich.
        - **Deribit** does the opposite: about {term_stats['d_week']:.0%} with 3–7 days left and
          {term_stats['d_last12']:.0%} in the last 12 hours. Deribit's dailies expire at 08:00 UTC, so their last
          12 hours are 20:00–08:00 UTC, the quietest of the day (05's hour profile), and they are priced for it.
          (Weekend expiries, left out here, are lower still: about 29% Saturday and 23% Sunday.)
        - **Deribit is not rich against realized vol at any horizon.** In the median hour its last-12-hour vol is a
          point or two above what BTC then realizes (the small overnight premium of section 4), but the big days
          dominate variance: in root-mean-square terms, which is what a short-gamma book pays, Deribit's implied vol
          is {term_stats['ratio_lo']:.2f}–{term_stats['ratio_hi']:.2f} times the vol realized to the same expiry.
          Over Jan–Sep 2026 BTC realized about 44% a year (44% at 1-minute or 5-minute sampling, 43% hourly), above
          most of Deribit's at-the-money vols.

        So the premium the trade is built to harvest is a Kalshi feature, not a Deribit one.
        """),
        fig_term,
    ])
    return


@app.cell(hide_code=True)
def _(END, FIG, H, SLOT1, SLOT2, SLOT3, START, W, alt, base, daily_all, mo, pd):
    _show = {"base": "base, before costs", "base_net": "base, after costs", "best_practical": "short to expiry + maker hedge, after costs"}
    _b = daily_all[(daily_all.period == "2026") & (daily_all.variant == "base")].set_index("t")
    _p = daily_all[(daily_all.period == "2026") & (daily_all.variant == "best_practical")].set_index("t")
    _cum = pd.concat([
        pd.DataFrame({"day": _b.index, "series": _show["base"], "cum": _b.gross.cumsum().to_numpy()}),
        pd.DataFrame({"day": _b.index, "series": _show["base_net"], "cum": _b.pnl.cumsum().to_numpy()}),
        pd.DataFrame({"day": _p.index, "series": _show["best_practical"], "cum": _p.pnl.cumsum().to_numpy()}),
    ])
    fig_cum = (
        alt.Chart(_cum).mark_line(strokeWidth=1.8)
        .encode(x=alt.X("day:T", title=None, axis=alt.Axis(format="%b %Y")),
                y=alt.Y("cum:Q", title="cumulative P&L ($ per long straddle)", axis=alt.Axis(format=",.0f")),
                color=alt.Color("series:N", title=None, scale=alt.Scale(domain=list(_show.values()),
                                                                        range=[SLOT1, SLOT2, SLOT3])),
                tooltip=[alt.Tooltip("day:T", format="%Y-%m-%d"), "series:N", alt.Tooltip("cum:Q", format=",.0f")])
        .properties(title="Figure 2 · Cumulative P&L of the calendar straddle, 2026", width=W, height=H)
    )
    fig_cum.save(FIG / "10_fig2_cumulative_pnl.png", scale_factor=2)
    _comp = _b[["options", "hedge", "funding", "opt_cost", "perp_cost", "pnl"]].mean()
    mo.vstack([
        mo.md(f"""
        ## 2 · The base case

        Long ~7-day ATM straddle, short ~2.6 one-day ATM straddles (vega-neutral at entry), entered 08:00 UTC, hedged
        hourly with the perpetual (taker fee), short bought back 4 hours before expiry, long sold after 24 hours.

        | per day, average | $ |
        |---|---|
        | options, marked to Deribit's smile | {_comp.options:+,.0f} |
        | perpetual hedge | {_comp.hedge:+,.0f} |
        | funding on the hedge | {_comp.funding:+,.0f} |
        | **gross** | **{_comp.options + _comp.hedge + _comp.funding:+,.0f}** |
        | option fees and half-spreads | {-_comp.opt_cost:+,.0f} |
        | perpetual fees | {-_comp.perp_cost:+,.0f} |
        | **net** | **{_comp.pnl:+,.0f}** |

        Sharpe (daily, ×√365): {base['gross Sharpe']:.1f} gross, {base['net Sharpe']:.1f} net. Worst day
        {base['worst day $']:+,.0f} $ ({base['worst day']}); maximum drawdown {base['max drawdown $']:+,.0f} $ over
        the {len(_b)} days. The options and the hedge offset each other as they should (the book is short gamma: the
        options lose when BTC moves and the hedge earns some of it back); what is left is close to zero before
        costs, and the costs are about ten times larger.
        """),
        fig_cum,
    ])
    return


@app.cell(hide_code=True)
def _(JUMPS, base, daily_all, day_rv, mo, pd):
    _b = daily_all[(daily_all.period == "2026") & (daily_all.variant == "base")].set_index("t")
    _g = daily_all[(daily_all.period == "2026") & (daily_all.variant == "gross")].set_index("t")
    _top = day_rv["2026-01-15":"2026-09-30"].sort_values(ascending=False).head(10)
    _t = pd.DataFrame({"day": _top.index.date, "BTC realized vol that day": _top.map("{:.0%}".format).to_numpy(),
                       "P&L before costs $": _g.pnl.reindex(_top.index).round(0).to_numpy(),
                       "P&L after costs $": _b.pnl.reindex(_top.index).round(0).to_numpy()})
    _w = pd.DataFrame([{"window": k, "days": len(v), "before costs $": _g.pnl.reindex(v).sum(),
                        "after costs $": _b.pnl.reindex(v).sum()} for k, v in JUMPS.items()]).round(0)
    mo.vstack([
        mo.md(f"""
        ## 3 · Jump days

        The book is short gamma, so the days BTC moves most are its worst. The three windows in the plan, and the ten
        highest-realized-vol days of 2026, for the base case. The single worst day, {base['worst day']}, is not
        among them: BTC fell 3% between 16:00 and 17:00 UTC and was back by 20:00, a round trip that an hourly hedge
        misses and a daily close-to-close vol does not show.
        """),
        mo.ui.table(_w, selection=None, show_download=False, pagination=False),
        mo.ui.table(_t, selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(daily_all, mo, pd):
    # P&L by weekday of entry (08:00 UTC): the short leg expires the next morning, so Friday and Saturday entries
    # sell Saturday's and Sunday's quiet weekend dailies
    _rows = []
    for _v in ["base", "best_practical", "overnight_expiry"]:
        _x = daily_all[(daily_all.period == "2026") & (daily_all.variant == _v)].copy()
        _x["weekday"] = _x.t.dt.day_name()
        _g = _x.groupby("weekday")[["gross", "pnl"]].mean().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        _rows.append(pd.DataFrame({"variant": _v, "weekday": _g.index, "before costs $/day": _g.gross.round(0).to_numpy(),
                                   "after costs $/day": _g.pnl.round(0).to_numpy()}))
    _t = pd.concat(_rows, ignore_index=True)
    _wk = daily_all[(daily_all.period == "2026") & (daily_all.variant == "base")]
    _mon_thu = _wk[_wk.t.dt.dayofweek <= 3].gross.mean()
    mo.vstack([
        mo.md(f"""
        **By weekday.** Deribit's Saturday and Sunday dailies cover quiet weekend nights. Restricting the base case
        to Monday–Thursday entries (short leg expiring Tuesday–Friday), it makes {_mon_thu:+,.0f} $ a day before
        costs: the weekday version is no better, so the verdict does not come from the weekend.
        """),
        mo.ui.table(_t, selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(daily_all, mo, table_2026, table_pre, table_q4):
    _cols = ["variant", "what", "gross $/day", "gross Sharpe", "option costs $/day", "hedge costs $/day", "net $/day",
             "net Sharpe", "worst day $", "max drawdown $", "share of days up", "early Feb (2–8)", "early Jun (1–7)",
             "late Aug (20–31)"]
    _fmt = lambda d: d[_cols].round(2)
    _pre = table_pre.set_index("variant").stack()      # (variant, column) -> value
    _pre_days = int(((daily_all.period == "pre2026") & (daily_all.variant == "base")).sum())
    mo.vstack([
        mo.md("""
        ## 4 · Variations: what changes the answer, and what does not

        Every rule of the base case moved one at a time (click a column to sort). Gross = before any cost.

        - **Holding the short to expiry** removes the buy-back (no fee, no spread at settlement) and keeps the last 4
          quiet hours of decay: gross goes from about −70 to about 0 $ a day.
        - **Selling only the overnight hours** (enter 20:00 UTC, sell the 12-hour daily, hold to the 08:00 expiry)
          is the one clearly positive gross: about +80–100 $ a day, Sharpe about 1–1.4 before costs, also positive
          in Oct–Dec 2025. Deribit's 08:00 dailies are slightly rich *for the night*. But the fees on a 12-hour
          option are about 5% of its price each way and take all of it, and one bad night (early June: about
          −13,000 $ in a week) wipes out months.
        - **Hedging more often** costs more than it saves at taker fees (15-minute hedging: −970 $ a day); **hedging
          once a day** is cheaper but turns the jump weeks into −27,000 $ (early February).
        - **The long leg** (3, 7 or 14 days) and **the ratio** (vega-neutral, 1:1, long only) change the size of the
          loss, not its sign. The long straddle alone is about flat before costs: Deribit's weekly vol was about fair.
        """),
        mo.ui.table(_fmt(table_2026), selection=None, show_download=True, pagination=False),
        mo.md("""
        **2 October 2025 – 14 January 2026, as an out-of-sample check** (before the period in the plan). Same picture: the base case loses before costs, the overnight short is positive before costs and
        negative after.
        """),
        mo.ui.table(_fmt(table_q4), selection=None, show_download=True, pagination=False),
        mo.md(f"""
        **16 November 2024 – 14 January 2026, the longest check the data allow** ({_pre_days} days, every Deribit
        trade from the full-day fetch; it contains the Q4 2025 check above). The same picture again: the base case
        is {_pre['base', 'gross $/day']:+.0f} $ a day before costs and {_pre['base', 'net $/day']:+,.0f} after;
        the overnight short is the one positive gross, {_pre['overnight_expiry', 'gross $/day']:+.0f} $ a day
        (Sharpe {_pre['overnight_expiry', 'gross Sharpe']:.1f}), and {_pre['overnight_expiry', 'net $/day']:+.0f}
        after costs; the long straddle alone is {_pre['long_only', 'gross $/day']:+.0f} $ a day before costs.
        """),
        mo.ui.table(_fmt(table_pre), selection=None, show_download=True, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Conclusion and limits

    **Verdict: do not run it.** On Deribit, the short-dated premium the trade relies on is not there: daily options
    expire after the quiet overnight hours and are priced at or below the vol BTC then realizes. The richness seen in
    Kalshi's last hours (`03`, Figure 1 here) belongs to Kalshi's 5pm ET expiry and its US-afternoon hours, which
    Deribit has no option for. What is left, a small premium on the overnight hours, is about the size of one
    option fee.

    **What would have to change for it to work:** fees well below 0.03% of BTC per option (a market maker's rebate
    schedule, or block trades with a negotiated fee), or a short leg that covers the busy US hours (a Deribit expiry at
    Kalshi's 21:00 UTC does not exist).

    **Limits.**
    - Values come from Deribit's mark smile fitted to trades, not from quotes. The mark is Deribit's own fair value
      and the fit is checked to recover synthetic smiles (`tests/test_strategies.py`), but the exact price we could
      have traded at is not observed; half-spreads come from trades vs marks, so they reflect the times people chose
      to trade.
    - Inverse (BTC-margined) contracts are valued in dollars, as elsewhere in the project; the BTC-collateral effect
      and margin are not modelled (the plan's section 3), nor are jump-diffusion models or liquidity in stressed
      markets beyond what the trades show.
    - One sample: January–September 2026, with three stress episodes. Oct 2025 – mid-January 2026, and the longer
      Nov 2024 – mid-January 2026, are used as checks.
    """)
    return


if __name__ == "__main__":
    app.run()
