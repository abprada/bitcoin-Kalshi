import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="11 Kalshi vs Deribit dislocations")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    from lib import dislocation as dl
    from lib import vol_spread as vs

    return alt, dl, mo, np, pd, vs


@app.cell(hide_code=True)
def _(alt):
    # Chart styling as in 05/06 (slot 1 blue = Kalshi, slot 2 orange = Deribit, slot 3 aqua = outcome).
    INK, INK2, MUTED, CONTEXT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
    SLOT1, SLOT2, SLOT3 = "#2a78d6", "#eb6834", "#1baf7a"
    W, H = 300, 220


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
def _(dl, mo, np, pd):
    ROOT = mo.notebook_dir()
    D = ROOT / "data" / "derived" / "strategies"
    FIG = ROOT / "outputs" / "figures"
    p = pd.read_parquet(D / "disl_panel.parquet")
    p["K"], p["Db"], p["RV"] = 100 * p.atm_iv, 100 * p.deribit_sig, 100 * p.rv_to_expiry
    p["quarter"] = "Q" + p.ts.dt.quarter.astype(str) + " 2026"
    fs = pd.read_parquet(D / "forecast_strikes.parquet")
    fs = fs[fs.p_deribit.notna()].copy()
    fs["bk"], fs["lk"] = dl.scores(fs.p_kalshi, fs.y)
    fs["bd"], fs["ld"] = dl.scores(fs.p_deribit, fs.y)
    fs["b_diff"], fs["l_diff"] = fs.bd - fs.bk, fs.ld - fs.lk
    tr = pd.read_parquet(D / "disl_trades.parquet")
    _v = tr.vega_k
    tr["k_taker"], tr["d_taker"] = (tr.k_gross - tr.k_cost) / _v, (tr.d_gross - tr.d_cost) / _v
    # maker: Kalshi strip filled at mid with a quarter of the taker fee, no Deribit half-spread (fee kept), free hedge
    tr["k_maker"], tr["d_maker"] = (tr.k_gross - 0.25 * tr.k_fee) / _v, (tr.d_gross - tr.d_fee) / _v
    tr["abs_disl"] = tr.signal.abs()
    tr["quarter"] = "Q" + tr.ts.dt.quarter.astype(str)
    return D, FIG, fs, p, tr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 11 · Kalshi vs Deribit: dislocations, mean reversion, and who predicts BTC better

    *January – September 2026, hourly. Built by `08_strategy_data.py` (`lib/dislocation.py`, `lib/vol_spread.py`);
    this notebook only reads the results.*

    **A dislocation** = Kalshi's ATM vol minus Deribit's at the same expiry, *minus the gap's own normal*: its median
    over the previous 30 days for the same kind of contract (daily or weekly) and the same time to expiry. Kalshi's
    premium has a shape (bigger near expiry) and a trend (smaller every quarter), so only a departure from that
    normal is a candidate mispricing. All signals use only information available at the time.

    | Question | Answer | Tradable? |
    |---|---|---|
    | When the two diverge, who adjusts? | **Depends on the direction.** Kalshi unusually *cheap* (Deribit spiked, Kalshi did not follow): Deribit comes back down, about two thirds of the gap closes within a day. Kalshi unusually *rich*: the gap barely closes (85–99% left after a day) | see below |
    | Who was right? | **Kalshi, mostly, for the weeklies.** Kalshi cheap → Deribit overpriced BTC's later moves by 4–15 vol points; Kalshi rich → Deribit underpriced them by about 6. For the 24-hour dailies, extreme Kalshi richness was Kalshi being wrong | — |
    | Is Kalshi's dislocation a signal for Deribit? | **Yes, and Deribit's own spike is not.** After Deribit's biggest vol jumps, BTC keeps moving (Deribit still too low) unless Kalshi stayed calm, in which case Deribit was too high | partly |
    | Mean-reversion trades | **Two survive, both thin.** (A) Sell Deribit's straddle when Kalshi is ≥ 6 points cheaper than normal, hold to expiry: +6 vol points per trade after taker costs, but 12 weeks. (B) Sell Kalshi's daily when ≥ 3 points richer than normal, hold to expiry: +6 points per daily *only as a maker*, fading from +14 in Q1 to +3 in Q3. Short holds (1–24 hours) never pay | marginal |
    | Better predictor of BTC's price distribution | **Deribit, barely** (Brier 0.1013 vs 0.1018). Kalshi overprices big up-moves (a longshot bias) | no |
    | Better predictor of BTC's direction | **Neither.** Both are coin flips; an apparent contrarian Kalshi signal disappears with one observation per contract | no |
    | Better predictor of BTC's volatility | **Deribit on its own, Kalshi as a complement**: Deribit explains more of realized vol (R² 0.64 vs 0.56 for dailies), and the dislocation adds to it | — |
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, p, pd):
    # Corrections raised in review, on this project's hourly data
    _s = pd.read_parquet(mo.notebook_dir() / "data" / "derived" / "kalshi_hourly_snapshots.parquet")
    _s = _s[_s.atm_iv.notna()].assign(h=lambda d: pd.cut(d.tau_years * 8760, [0, 3, 6, 12, 24, 170]))
    _f = (_s.groupby(["kind", "h"], observed=True)
          .apply(lambda g: pd.Series({"snapshots": len(g), "share passing 05's grid rule": g.usable.mean(),
                                      "ATM vol, passing only": g.atm_iv[g.usable].median(),
                                      "ATM vol, all snapshots": g.atm_iv.median()}))
          .reset_index())
    _f["h"] = _f.h.astype(str)
    _q = (p.groupby(["quarter", "kind"]).gap_sig.median().unstack().round(1).reset_index()
          .rename(columns={"daily": "gap, dailies", "weekly": "gap, weeklies"}))
    _d6 = p[(p.kind == "daily") & (p.hours_left <= 6)].groupby("quarter").apply(
        lambda g: pd.Series({"Kalshi daily implied, last 6h": g.K.mean(), "realized": np.sqrt((g.RV**2).mean())}))
    _q = _q.merge(_d6.round(1).reset_index(), on="quarter", how="left")
    mo.vstack([
        mo.md("""
        ## 0 · Corrections carried into this analysis

        **1. The near-expiry vol spike was mostly a filter.** 05's rule (at least 2 strikes per standard deviation)
        passes near expiry only when vol is high. This panel now uses every snapshot with an ATM vol and keeps the
        rule as a precision flag. On the hourly data (below), weekly snapshots with 0–3 hours left show 96% median
        ATM vol if filtered, 58% if not. 09 and 10 now use the unfiltered panel; 03's figures (other authors) are
        unchanged here and need the same fix.
        """),
        mo.ui.table(_f.round(3), selection=None, show_download=False, pagination=False),
        mo.md("""
        **2. Kalshi's premium shrinks through the year.** Median Kalshi − Deribit gap and Kalshi's last-6-hours
        daily vol against realized, by quarter: the daily gap falls from about 7 points in Q1 to about 2.5. Every
        result below is shown by quarter where it matters.
        """),
        mo.ui.table(_q, selection=None, show_download=False, pagination=False),
        mo.md("""
        **3. Deribit's short end has a weekend effect.** Deribit dailies expiring Saturday and Sunday trade at
        about 29% and 23% vol against 35–37% on weekdays (BTC is quieter then too). 10 now shows Deribit's term
        structure for Tuesday–Friday expiries and the calendar P&L by weekday.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(FIG, H, SLOT1, SLOT2, W, alt, dl, mo, np, p, pd):
    # Who adjusts after a large dislocation? Changes relative to quiet hours (|dislocation| < 1), so the normal
    # drift of vols toward expiry is netted out. Weeklies (dailies end within the day).
    _w = dl.forward_values(p[p.kind == "weekly"], ["K", "Db"], [1, 2, 4, 8, 12, 24])
    _w["state"] = np.select([_w.disl <= -4, _w.disl >= 4, _w.disl.abs() < 1],
                            ["Kalshi ≥ 4 points cheaper than normal", "Kalshi ≥ 4 points richer than normal", "quiet"],
                            "other")
    _rows = []
    for _h in [0, 1, 2, 4, 8, 12, 24]:
        for _st in ["Kalshi ≥ 4 points cheaper than normal", "Kalshi ≥ 4 points richer than normal"]:
            for _mkt, _c in [("Kalshi", "K"), ("Deribit", "Db")]:
                if _h == 0:
                    _val = 0.0
                else:
                    _val = (_w.loc[_w.state == _st, f"{_c}_f{_h}"].mean()
                            - _w.loc[_w.state == "quiet", f"{_c}_f{_h}"].mean())
                _rows.append({"hours": _h, "state": _st, "market": _mkt, "change": _val})
    _r = pd.DataFrame(_rows)
    fig_adj = (
        alt.Chart(_r).mark_line(point=True).encode(
            x=alt.X("hours:Q", title="hours after the dislocation"),
            y=alt.Y("change:Q", title="vol change vs quiet hours (points)"),
            color=alt.Color("market:N", title=None, scale=alt.Scale(domain=["Kalshi", "Deribit"], range=[SLOT1, SLOT2])),
            tooltip=["state:N", "market:N", "hours:Q", alt.Tooltip("change:Q", format="+.2f")])
        .properties(width=W, height=H)
        .facet(column=alt.Column("state:N", title=None))
        .properties(title="Figure 1 · Weeklies: who moves after a large dislocation")
    )
    fig_adj.save(FIG / "11_fig1_who_adjusts.png", scale_factor=2)
    # who was right: implied minus realized to Kalshi's expiry, by dislocation bucket
    _b = pd.cut(p.disl, [-np.inf, -6, -3, -1, 1, 3, 6, np.inf])
    _right = (p.dropna(subset=["RV"]).assign(bin=_b)
              .groupby(["kind", "bin"], observed=True)
              .apply(lambda g: pd.Series({"hours": len(g), "days": g.day.nunique(), "Kalshi − realized": (g.K - g.RV).mean(),
                                          "Deribit − realized": (g.Db - g.RV).mean()}))
              .reset_index())
    _right["bin"] = _right.bin.astype(str)
    mo.vstack([
        mo.md("""
        ## 1 · After a large dislocation, who adjusts, and who was right?

        Figure 1 follows Kalshi's and Deribit's ATM vol after weekly hours where the gap is at least 4 points away
        from its normal, net of what both do in quiet hours.

        - **Kalshi unusually cheap** (left): Deribit falls about 2.5 points within 4 hours and 4 within a day,
          while Kalshi barely moves. These are Deribit spikes that Kalshi did not follow, and Deribit gives them back.
        - **Kalshi unusually rich** (right): the two drift toward each other by only 1–2 points each, so most of the
          gap is still there a day later.

        Against the vol BTC then realized (table), Deribit is too high when Kalshi is cheap and too low when Kalshi
        is rich: on the weeklies, Kalshi's side of the dislocation was usually the right one. On the 24-hour
        dailies, extreme Kalshi richness (more than 6 points) was Kalshi overpricing by about 15 points.
        """),
        fig_adj,
        mo.ui.table(_right.round(2), selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(dl, mo, p, pd, vs):
    # Is it Kalshi's signal, or just "sell Deribit after it spikes"? Gross edge of selling Deribit vol to expiry:
    # Deribit's implied minus the vol BTC then realized.
    _w = p[p.kind == "weekly"].dropna(subset=["RV"]).copy()
    _m = vs.changes(_w, ["Db"], 12)
    _w["dD12"] = _m.d_Db.to_numpy()
    _w["edge"] = _w.Db - _w.RV
    _q90 = _w.dD12.quantile(0.9)
    _cases = [("all weekly hours", _w),
              ("Deribit's vol rose most over 12h (top 10%)", _w[_w.dD12 >= _q90]),
              ("… and Kalshi followed (dislocation > −3)", _w[(_w.dD12 >= _q90) & (_w.disl > -3)]),
              ("… and Kalshi stayed cheap (dislocation ≤ −3)", _w[(_w.dD12 >= _q90) & (_w.disl <= -3)]),
              ("Kalshi ≥ 3 points cheaper than normal", _w[_w.disl <= -3]),
              ("Kalshi ≥ 6 points cheaper than normal", _w[_w.disl <= -6]),
              ("Kalshi ≥ 3 points richer than normal", _w[_w.disl >= 3])]
    _rows = []
    for _lab, _x in _cases:
        _mm, _lo, _hi = dl.boot_mean_diff(_x, "edge")
        _rows.append({"weekly hours where…": _lab, "hours": len(_x), "weeks": _x.event_ticker.nunique(),
                      "Deribit − realized (vol pts)": _mm, "95% low": _lo, "95% high": _hi})
    _reg = vs.ols_clustered(_w.edge, _w[["disl", "dD12"]], _w.event_ticker, ["disl", "dD12"])
    mo.vstack([
        mo.md(f"""
        ## 2 · Kalshi's dislocation is a signal for Deribit; Deribit's own spike is not

        If Deribit simply overshot when vol jumps, selling Deribit after its biggest rises would pay. It does the
        opposite: after Deribit's largest 12-hour vol increases, BTC goes on to move *more* than Deribit priced (vol
        clusters). What separates the cases is Kalshi. When Kalshi followed Deribit up, Deribit was still too low;
        when Kalshi stayed calm, Deribit was about right or too high. In one regression (clustered by week), each
        point of dislocation moves Deribit's later error by {_reg.coef['disl']:+.2f} points (t = {_reg.t['disl']:.1f}),
        and Deribit's own 12-hour move adds nothing (t = {_reg.t['dD12']:.1f}).
        """),
        mo.ui.table(pd.DataFrame(_rows).round(1), selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo, pd, tr, vs):
    _cands = [
        ("A · sell Deribit straddle when Kalshi is cheap vs normal (weeklies)", "weekly", -1, "d"),
        ("B · sell Kalshi daily strip when Kalshi is rich vs normal", "daily", 1, "k"),
        ("C · sell Kalshi weekly strip when Kalshi is rich vs normal", "weekly", 1, "k"),
        ("D · buy Deribit straddle when Kalshi is rich vs normal (weeklies)", "weekly", 1, "d"),
    ]
    _rows = []
    for _name, _kind, _sgn, _leg in _cands:
        for _hold in ["4", "24", "expiry"]:
            for _thr in [3, 6, 10]:
                _x = tr[(tr.kind == _kind) & ((tr.signal > 0) == (_sgn > 0)) & (tr.abs_disl >= _thr) & (tr.hold == _hold)]
                if _x.event_ticker.nunique() < 5:
                    continue
                _first = _x.sort_values("ts").groupby("event_ticker").head(1)
                _sa = vs.trade_summary(_x.assign(g=1), "g", cols=(f"{_leg}_taker", f"{_leg}_maker")).iloc[0]
                _sf = vs.trade_summary(_first.assign(g=1), "g", cols=(f"{_leg}_taker", f"{_leg}_maker")).iloc[0]
                _byq = _x.groupby("quarter")[f"{_leg}_maker"].mean()
                _rows.append({"trade": _name, "held": _hold, "|dislocation| ≥": _thr, "contracts": _x.event_ticker.nunique(),
                              "taker, all signals": f"{_sa[f'{_leg}_taker']:+.1f} [{_sa[f'{_leg}_taker_lo']:+.1f}, {_sa[f'{_leg}_taker_hi']:+.1f}]",
                              "maker, all signals": f"{_sa[f'{_leg}_maker']:+.1f} [{_sa[f'{_leg}_maker_lo']:+.1f}, {_sa[f'{_leg}_maker_hi']:+.1f}]",
                              "taker, first signal per contract": f"{_sf[f'{_leg}_taker']:+.1f} [{_sf[f'{_leg}_taker_lo']:+.1f}, {_sf[f'{_leg}_taker_hi']:+.1f}]",
                              "maker, first signal per contract": f"{_sf[f'{_leg}_maker']:+.1f} [{_sf[f'{_leg}_maker_lo']:+.1f}, {_sf[f'{_leg}_maker_hi']:+.1f}]",
                              "maker by quarter": "  ".join(f"{q}: {v:+.1f}" for q, v in _byq.items())})
    cand = pd.DataFrame(_rows)
    mo.vstack([
        mo.md("""
        ## 3 · Mean-reversion trades on dislocations

        Every hour with |dislocation| ≥ 2 points was traded (`vol_spread.backtest`, as in 09): one leg or both,
        delta-hedged hourly with the perpetual, closed after 1, 4 or 24 hours or held to expiry. P&L in vol points
        of the Kalshi straddle's vega, with 95% intervals resampling days. Two cost cases: **taker** (cross every
        spread, full fees, as in 09) and **maker** (an upper bound: Kalshi's strip filled at mid with a quarter of
        the fee, no Deribit half-spread but the full Deribit fee, free perpetual hedging). Because one dislocation
        triggers many hourly signals, each trade is also shown with **only the first signal per contract**.

        - **A works at large dislocations, held to expiry**: +6.3 points per trade after taker costs on the first
          signal per week (interval +0.7 to +11.5), +10.9 as maker. But it rests on 12 weeks, nearly all in Q1–Q2;
          at a 3-point threshold the first-signal version is not positive.
        - **B works only as a maker**: +5.6 points per daily on the first signal at a 3-point threshold (106
          dailies, interval +1.5 to +9.4; at higher thresholds the intervals reach zero), negative as a taker, and
          the quarterly numbers fall from about +14 in Q1 to about +3 in Q3: it is the shrinking premium of
          section 0.
        - **C** (sell Kalshi's weekly) is borderline as a maker and negative as a taker; **D** (buy Deribit when Kalshi
          is rich) and every 1- to 24-hour hold lose on the first-signal test. Both legs together never pay.
        """),
        mo.ui.table(cand, selection=None, show_download=True, pagination=False),
    ])
    return (cand,)


@app.cell(hide_code=True)
def _(FIG, H, SLOT1, SLOT2, SLOT3, W, alt, dl, fs, mo, np, p, pd, vs):
    # Which market predicts BTC better? (1) its price distribution: Kalshi's binary mids vs Deribit's smile turned
    # into the same binaries, scored on how each contract settled
    _rows = []
    for (_k, _hb), _g in fs.assign(hb=pd.cut(fs.hours_left, [0, 12, 24, 96, 170]).astype(str)).groupby(["kind", "hb"]):
        _m, _lo, _hi = dl.boot_mean_diff(_g, "b_diff")
        _rows.append({"contract": _k, "hours left": _hb, "forecasts": len(_g), "Brier Kalshi": _g.bk.mean(),
                      "Brier Deribit": _g.bd.mean(), "Deribit − Kalshi": _m, "95% low": _lo, "95% high": _hi})
    _m, _lo, _hi = dl.boot_mean_diff(fs, "b_diff")
    _rows.append({"contract": "all", "hours left": "all", "forecasts": len(fs), "Brier Kalshi": fs.bk.mean(),
                  "Brier Deribit": fs.bd.mean(), "Deribit − Kalshi": _m, "95% low": _lo, "95% high": _hi})
    brier = pd.DataFrame(_rows)
    # calibration by distance from today's price (in Deribit standard deviations)
    _sd = np.log(fs.strike / fs.spot) / (fs.deribit_sig * np.sqrt(fs.tau_years))
    _cal = (fs.assign(b=pd.cut(_sd, [-9, -1.5, -0.5, 0.5, 1.5, 9],
                               labels=["1.5+ SD below", "0.5–1.5 below", "within 0.5", "0.5–1.5 above", "1.5+ SD above"]))
            .groupby("b", observed=True)
            .agg(contracts=("y", "size"), Kalshi=("p_kalshi", "mean"), Deribit=("p_deribit", "mean"), happened=("y", "mean"))
            .reset_index())
    _err = _cal.assign(Kalshi=100 * (_cal.Kalshi - _cal.happened), Deribit=100 * (_cal.Deribit - _cal.happened))
    _long = _err.melt(id_vars=["b"], value_vars=["Kalshi", "Deribit"], var_name="market", value_name="err")
    _long["b"] = _long.b.astype(str)
    fig_cal = (
        alt.Chart(_long).mark_bar(width=18, cornerRadiusEnd=4).encode(
            x=alt.X("b:N", title="strike vs today's price (Deribit standard deviations)", sort=list(_cal.b.astype(str))),
            xOffset=alt.XOffset("market:N", sort=["Kalshi", "Deribit"]),
            y=alt.Y("err:Q", title="forecast − how often it happened (points)"),
            color=alt.Color("market:N", title=None, scale=alt.Scale(domain=["Kalshi", "Deribit"], range=[SLOT1, SLOT2])),
            tooltip=["b:N", "market:N", alt.Tooltip("err:Q", format="+.2f")])
        + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#898781").encode(y="y:Q")
    ).properties(width=2 * W, height=H,
                 title="Figure 2 · Probability that BTC ends above the strike: forecast minus outcome")
    fig_cal.save(FIG / "11_fig2_calibration.png", scale_factor=2)
    # (2) direction: one observation per contract, so overlapping hours do not inflate t-stats
    _dir = []
    for _kind, _target in [("daily", 20), ("weekly", 120), ("weekly", 48)]:
        _z = p[(p.kind == _kind)].dropna(subset=["k_p_up", "d_p_up", "settle"]).copy()
        _z["dist"] = (_z.hours_left - _target).abs()
        _one = _z.loc[_z.groupby("event_ticker").dist.idxmin()]
        _one = _one[_one.dist < 3]
        _up = (_one.settle > _one.spot).astype(float)
        _r = vs.ols_clustered(100 * _one.ret_expiry, _one[["k_p_up", "d_p_up"]], _one.event_ticker, ["k", "d"])
        _dir.append({"contract": _kind, "hours left": _target, "contracts": len(_one), "BTC ended higher": _up.mean(),
                     "Brier, Kalshi P(up)": float(dl.scores(_one.k_p_up, _up)[0].mean()),
                     "Brier, Deribit P(up)": float(dl.scores(_one.d_p_up, _up)[0].mean()),
                     "t, Kalshi P(up) → return": _r.t["k"], "t, Deribit P(up) → return": _r.t["d"]})
    # (3) volatility
    _vol = []
    for _kind in ["weekly", "daily"]:
        _z = p[p.kind == _kind].dropna(subset=["RV"])
        _rk = vs.ols_clustered(_z.RV, _z[["K"]], _z.event_ticker, ["K"])
        _rd = vs.ols_clustered(_z.RV, _z[["Db"]], _z.event_ticker, ["Db"])
        _rx = vs.ols_clustered(_z.RV - _z.Db, _z[["disl"]], _z.event_ticker, ["disl"])
        _vol.append({"contract": _kind, "hours": len(_z), "R² Kalshi": _rk.r2.iloc[0], "R² Deribit": _rd.r2.iloc[0],
                     "RMSE Kalshi": np.sqrt(((_z.K - _z.RV) ** 2).mean()), "RMSE Deribit": np.sqrt(((_z.Db - _z.RV) ** 2).mean()),
                     "dislocation → Deribit's error (per point)": _rx.coef["disl"], "t": _rx.t["disl"]})
    mo.vstack([
        mo.md("""
        ## 4 · Which market predicts BTC better?

        **Price distribution.** Every Kalshi ladder strike at every hour (Kalshi's monotone mid) against Deribit's
        smile turned into the same "$1 if BTC ends above K" at Kalshi's expiry (Black-Scholes digital with the skew
        term), both scored on how the contract settled. Lower Brier = better. Deribit is better by about 0.5%,
        statistically clear on the full sample but economically tiny. Figure 2 shows each market's average
        probability minus how often BTC actually finished above the strike. The difference is in the upside tail:
        Kalshi prices a finish 1.5+ standard deviations above today at about 4%, Deribit at under 3%, and it
        happened about 2.5% of the time. That is a longshot bias on Kalshi's upside (handout question 1). Both
        overstated the upside in 2026, a year in which BTC fell.
        """),
        mo.ui.table(brier.round(4), selection=None, show_download=False, pagination=False),
        fig_cal,
        mo.ui.table(_cal.round(3), selection=None, show_download=False, pagination=False),
        mo.md("""
        **Direction.** Each market's probability that BTC ends above today's price, against what happened, one
        observation per contract. Both score no better than a coin flip (Brier 0.25), and neither predicts the
        return. With every hour counted, Kalshi's probability looked contrarian (t ≈ −3); with one observation per
        contract it is insignificant and changes sign across quarters, so it was overlap, not signal.
        """),
        mo.ui.table(pd.DataFrame(_dir).round(3), selection=None, show_download=False, pagination=False),
        mo.md("""
        **Volatility.** Deribit's ATM vol explains more of the vol BTC then realized than Kalshi's (especially for
        the dailies), but the dislocation adds information: when Kalshi is richer than normal, BTC moves more
        than Deribit priced (section 2).
        """),
        mo.ui.table(pd.DataFrame(_vol).round(3), selection=None, show_download=False, pagination=False),
    ])
    return (brier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Bottom line and limits

    - **Kalshi's vol is not just noisier Deribit.** Its departures from its usual premium carry information about
      Deribit's mispricing: Deribit spikes that Kalshi ignores tend to be overdone, and Kalshi richness that
      Deribit lags tends to be right (weeklies).
    - **Trading it is hard.** The only taker-profitable version is selling Deribit vol on large Kalshi-cheap
      dislocations, held to expiry, on 12 weeks: a lead for a longer sample, not a strategy yet. Selling Kalshi's
      rich dailies needs maker execution and its edge has faded with Kalshi's growth.
    - **For forecasting BTC,** Deribit is marginally better on the distribution, neither says anything about
      direction, and Kalshi's upside wing is overpriced.

    **Limits.** Kalshi is hourly (the collaborator's snapshots); the maker case assumes fills at mid with no
    adverse selection, so it is an upper bound; Kalshi's maker fee is assumed at a quarter of the taker fee; the
    30-day normal needs a month of history, so January is partly excluded; one year of data, with the large
    dislocations concentrated in a few stress weeks (February, June).
    """)
    return


if __name__ == "__main__":
    app.run()
