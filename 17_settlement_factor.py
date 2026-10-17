import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="17 Settlement factor")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt

    from lib import deribit, pairs, settlement, slides

    return alt, deribit, mo, np, pairs, pd, settlement, slides


@app.cell
def _(mo):
    mo.md(r"""
    # 17 · Deribit on Kalshi's terms: the settlement factor

    Deribit's Friday option and Kalshi's weekly settle on different rules, so their implied vols are not directly
    comparable:

    | | expiry | settlement price |
    |---|---|---|
    | Deribit | Friday 08:00 UTC | average of Deribit's BTC index over the last 30 minutes |
    | Kalshi | Friday 5pm ET: 21:00 UTC in summer, 22:00 in winter | average of CF Benchmarks' BTC index over the last 60 seconds |

    Each market quotes Black-Scholes vols to its own expiry. The **factor** R = Kalshi vol ÷ Deribit vol at the same
    strike is the ratio the two vols would have if both markets priced the same BTC; Deribit's vol × R is Deribit on
    Kalshi's terms. Two effects enter:

    1. **Busy hours.** Kalshi's extra 13 hours (14 in winter), Friday 08:00–21:00 UTC, are among BTC's most volatile,
       so Kalshi's vol per year is higher. This is the stretch used so far (`deribit.horizon_vol_ratios`, in 05 and 16).
    2. **Averaging.** An average over a window keeps only a third of the window's variance. Deribit's 30-minute
       average removes 20 minutes of variance, which matters on its last day; Kalshi's 60 seconds barely matter.

    **Model** (`lib/settlement.py`): BTC is a geometric Brownian motion whose variance follows the hour-of-week
    profile from 1-minute BTC returns, with zero rates. Each settlement is the arithmetic average of the path sampled
    every second over its window. Options on it are priced by conditional Monte Carlo (only the window is simulated;
    the rest is Black-Scholes) and inverted with each venue's own maturity, out-of-the-money calls and puts.

    **Model-dependent.** A fixed variance schedule, no jumps or stochastic vol. Under it each settlement is close to
    lognormal, so the factor hardly depends on the strike (checked below against the Monte Carlo error). A factor that
    genuinely varies by strike would need a stochastic-vol model, or Deribit's next (Saturday) expiry as a second
    anchor at each strike.
    """)
    return


@app.cell
def _(alt, slides):
    # MIT colors (lib/slides.py), as on slide 6: Kalshi red, Deribit gray
    def mit_style(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
        return (chart.configure(font=slides.FONT, background="white")
                .configure_view(stroke=None)
                .configure_axis(labelColor=slides.MUTED, titleColor=slides.INK, labelFontSize=12, titleFontSize=13,
                                titleFontWeight="normal", grid=False, domainColor=slides.SILVER, tickColor=slides.SILVER)
                .configure_legend(labelColor=slides.MUTED, titleColor=slides.INK, titleFontWeight="normal")
                .configure_title(anchor="start", color=slides.INK, fontSize=18, fontWeight="normal",
                                 subtitleColor=slides.MUTED, subtitleFontSize=13, subtitlePadding=6, offset=14))

    sd_label = lambda x: "ATM" if x == 0 else f"{x:+g}".replace("-", "−")
    return mit_style, sd_label


@app.cell
def _(deribit, mo, np, pd, settlement):
    ROOT = mo.notebook_dir()
    DERIVED = ROOT / "data" / "derived"
    # The same hour-of-week profile as 16's stretch when its cache is here, else from the 1-minute BTC prices
    _cached = DERIVED / "pairs_hour_profile.parquet"
    if _cached.exists():
        profile = pd.read_parquet(_cached).iloc[:, 0]
    else:
        profile = deribit.hourly_variance(pd.read_parquet(ROOT / "data" / "clean" / "btc_spot_1m.parquet",
                                                          columns=["ts", "spot"]).set_index("ts").spot)
    SUMMER = pd.Timestamp("2026-07-10 21:00", tz="UTC")      # a Friday, 5pm EDT: Kalshi 13 hours after Deribit
    N_PATHS = 50_000
    # Every hour from the start of Deribit's window to 170 hours before Kalshi's expiry, at slide 6's strikes
    factor = settlement.factor_grid(profile, SUMMER, settlement.expiry_hours(SUMMER), np.round(np.arange(-3, 3.01, 0.5), 2),
                                    n_paths=N_PATHS)
    return DERIVED, N_PATHS, ROOT, SUMMER, factor, profile


@app.cell
def _(deribit, factor, mo, np, pd, profile, settlement, SUMMER):
    # How much the factor moves across strikes, against its Monte Carlo error
    _r = factor[factor.reliable]
    _dev = (_r.factor - _r.groupby("kalshi_hours").factor.transform("median")).abs()
    strike_dev, max_se = float(_dev.max()), float(_r.factor_se.max())
    _beyond = float((_dev > 2 * _r.factor_se).mean())

    # At the money, by time to expiry: the new factor vs the busy-hours-only stretch used so far, and vs the
    # constant-vol version that only has the averaging
    _h = np.array([15, 18, 24, 48, 72, 96, 120, 144, 168])
    _dh = _h - 13
    _t0 = SUMMER - pd.to_timedelta(_h, unit="h")
    _end = lambda t: pd.DatetimeIndex([t] * len(_h))
    _atm = factor[factor.sd == 0].set_index("kalshi_hours")
    atm_table = pd.DataFrame({
        "Kalshi hours left": _h, "Deribit hours left": _dh,
        "busy hours only (used so far)": deribit.horizon_vol_ratios(_t0, _end(SUMMER), _end(SUMMER.normalize() + pd.Timedelta(hours=8)), profile),
        "averaging only (constant vol)": settlement.atm_approx(_h - 1 / 60, 1 / 60, _h, _dh - 0.5, 0.5, _dh),
        "busy hours + averaging (new)": _atm.factor.reindex(_h).to_numpy(),
        "Monte Carlo error": _atm.factor_se.reindex(_h).to_numpy(),
    }).round(4)
    mo.vstack([
        mo.md(f"""
        **The factor is flat across strikes.** In every hour, each reliable strike is within **{strike_dev:.2%}** of
        that hour's median factor; {_beyond:.0%} of them differ by more than twice their Monte Carlo standard error
        (the largest is {max_se:.2%}). So the factor varies with time to expiry only.

        **At the money**, summer (Kalshi 13 hours after Deribit). The busy hours are most of the effect; averaging adds
        about 0.7% a day out from Deribit's expiry and more in its last hours. The constant-vol model (the
        spec's) has only the averaging.
        """),
        mo.ui.table(atm_table, selection=None, show_download=False, pagination=False),
    ])
    return max_se, strike_dev


@app.cell
def _(alt, factor, mit_style, np, sd_label, slides, strike_dev):
    # Figure 1: the factor by time to Kalshi's expiry (one row per hour) and strike
    _hours = np.sort(factor.kalshi_hours.unique())
    _next = dict(zip(_hours, np.r_[_hours[1:], _hours[-1] + 1]))
    _f = factor[factor.reliable].assign(
        # cells overlap their neighbours by a sliver so no hairlines show between them
        d_lo=lambda d: d.kalshi_hours / 24, d_hi=lambda d: d.kalshi_hours.map(_next) / 24 + 0.01,
        sd_lo=lambda d: d.sd - 0.25, sd_hi=lambda d: d.sd + 0.26, sd_label=lambda d: d.sd.map(sd_label),
        moneyness=lambda d: d.strike - 1, extra=lambda d: d.factor - 1)
    # Log scale on factor - 1: +2% a week out and +37% on Deribit's last hours both stay readable
    _scale = alt.Scale(type="log", domain=[float(_f.extra.min()), 0.1, float(_f.extra.max())],
                       range=["#f7e1e5", slides.LIGHT_RED, slides.MIT_RED], interpolate="rgb")
    factor_chart = mit_style(alt.Chart(_f).mark_rect().encode(
        x=alt.X("sd_lo:Q", title="strike: standard deviations from the money", scale=alt.Scale(domain=[-3.25, 3.25], nice=False),
                axis=alt.Axis(values=list(range(-3, 4)), labelExpr="datum.value == 0 ? 'ATM' : "
                              "(datum.value > 0 ? '+' + datum.value : '\u2212' + abs(datum.value))")),
        x2="sd_hi:Q",
        y=alt.Y("d_lo:Q", title="days to Kalshi's expiry", scale=alt.Scale(domain=[0.5, 7.2], nice=False),
                axis=alt.Axis(values=list(range(1, 8)), format="d")),
        y2="d_hi:Q",
        color=alt.Color("extra:Q", title="factor", scale=_scale,
                        legend=alt.Legend(type="gradient", gradientLength=320, values=[0.02, 0.05, 0.1, 0.2, 0.3],
                                          labelExpr="format(datum.value + 1, '.2f')")),
        tooltip=[alt.Tooltip("kalshi_hours:Q", title="Kalshi hours left", format=".1f"),
                 alt.Tooltip("deribit_hours:Q", title="Deribit hours left", format=".1f"),
                 alt.Tooltip("sd_label:N", title="strike (SD)"),
                 alt.Tooltip("moneyness:Q", title="strike vs forward", format="+.2%"),
                 alt.Tooltip("factor:Q", format=".4f"), alt.Tooltip("factor_se:Q", title="Monte Carlo error", format=".4f"),
                 alt.Tooltip("kalshi_iv:Q", title="Kalshi vol (model)", format=".1%"),
                 alt.Tooltip("deribit_iv:Q", title="Deribit vol (model)", format=".1%")],
    ).properties(width=640, height=440, title=alt.TitleParams(
        "Deribit → Kalshi vol factor, by time to expiry and strike",
        subtitle=["Kalshi vol ÷ Deribit vol at the same strike if both priced the same BTC: busy hours + settlement averaging",
                  "(Deribit 30 minutes, Kalshi 60 seconds). Summer: Kalshi expires 13 hours after Deribit.",
                  f"Flat across strikes: at most {strike_dev:.2%} from each hour's median, within Monte Carlo error. "
                  "Blank: option too cheap or noisy to invert."])))
    factor_chart
    return (factor_chart,)


@app.cell
def _(N_PATHS, SUMMER, mo, np, pd, profile, settlement):
    # Convergence: more paths, and coarser sampling of the averaging windows
    _h, _sd = [15, 18, 24, 37, 72, 168], [-2.0, 0.0, 2.0]
    _runs = {f"{N_PATHS:,} paths, every second": dict(n_paths=N_PATHS),
             "200,000 paths, every second": dict(n_paths=200_000),
             f"{N_PATHS:,} paths, every 10 seconds": dict(n_paths=N_PATHS, step_s=10)}
    _reliable = lambda g: g.set_index(["kalshi_hours", "sd"]).pipe(lambda g: g.factor.where(g.reliable))
    _t = pd.concat({k: _reliable(settlement.factor_grid(profile, SUMMER, _h, _sd, **v)) for k, v in _runs.items()}, axis=1)
    _gap = np.abs(_t.sub(_t.iloc[:, 0], axis=0)).max()
    mo.vstack([
        mo.md(f"""
        **Convergence.** The factor at three strikes and six times to expiry, from three runs (empty: not reliable). The largest difference from
        the first run: {_gap.iloc[1]:.5f} with four times the paths, {_gap.iloc[2]:.5f} sampling the windows every 10
        seconds instead of every second.
        """),
        mo.ui.table(_t.round(5).reset_index(), selection=None, show_download=False, pagination=False),
    ])
    return


@app.cell
def _(DERIVED, N_PATHS, alt, mit_style, mo, pairs, pd, profile, sd_label, settlement, slides):
    # Figure 2: Kalshi − Deribit with Deribit on Kalshi's terms, at slide 6's 15-minute snapshots and strikes
    _path = DERIVED / "snapshot15_strikes.parquet"
    spread_chart = None
    if not _path.exists():
        _view = mo.callout(mo.md(
            "**Spread figure skipped:** it needs the saved 15-minute comparison, `data/derived/snapshot15_strikes.parquet`, "
            "and `pairs_hour_profile.parquet` so the busy-hours profile matches slide 6. Building them here needs every "
            "Deribit trade (`00_fetch_deribit_all.py`)."), kind="warn")
    else:
        _s = pd.read_parquet(_path)
        # Each strike-snapshot's own factor: its Kalshi expiry (21:00 UTC in summer, 22:00 in winter), hours left, strike
        _s["factor"] = settlement.factor_for(_s, profile, n_paths=N_PATHS)
        _n_all = len(_s)
        _s = _s[_s.factor.notna()]
        _new = _s.assign(deribit_iv_adj=_s.deribit_iv * _s.factor,
                         gap_adj=lambda d: 100 * (d.kalshi_iv - d.deribit_iv_adj))
        _cells = pairs.surface(_new).merge(
            pairs.surface(_s)[["hours_lo", "sd", "gap"]].rename(columns={"gap": "gap_before"}), on=["hours_lo", "sd"])
        _cells = _cells[_cells.sd.abs() <= 3].assign(
            days=lambda d: ((d.hours_lo + d.hours_hi) / 48).round().astype(int), sd_label=lambda d: d.sd.map(sd_label),
            change=lambda d: d.gap - d.gap_before)
        _gmax = float(_cells.gap.abs().quantile(0.95))
        _order = [sd_label(x) for x in sorted(_cells.sd.unique())]
        _b = alt.Chart(_cells).encode(
            x=alt.X("sd_label:O", sort=_order, title="strike: standard deviations from the money", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("days:O", title="days to Kalshi's expiry", sort="descending"),
            tooltip=[alt.Tooltip("days:O", title="days left"), alt.Tooltip("sd_label:N", title="strike (SD)"),
                     alt.Tooltip("kalshi:Q", format=".1%", title="Kalshi"),
                     alt.Tooltip("deribit:Q", format=".1%", title="Deribit on Kalshi's terms"),
                     alt.Tooltip("gap:Q", format="+.2f", title="spread (points)"),
                     alt.Tooltip("gap_before:Q", format="+.2f", title="spread, busy hours only"),
                     alt.Tooltip("change:Q", format="+.2f", title="change (points)"),
                     alt.Tooltip("pairs:Q", format=",", title="strike-snapshots")])
        spread_chart = mit_style((
            _b.mark_rect(stroke="white", strokeWidth=1.5).encode(color=alt.Color(
                "gap:Q", title="points", scale=alt.Scale(domain=[-_gmax, 0, _gmax], range=[slides.DARK_GRAY, "#f7f7f7", slides.MIT_RED],
                                                         interpolate="rgb", clamp=True)))
            + _b.mark_text(fontSize=11).encode(text=alt.Text("gap:Q", format="+.1f"), color=alt.condition(
                f"abs(datum.gap) > {_gmax * 0.6:.2f}", alt.value("white"), alt.value(slides.INK)))
        ).properties(width=700, height=300, title=alt.TitleParams(
            "Spread: Kalshi − Deribit on Kalshi's terms, vol points",
            subtitle=["Deribit's vol × the settlement factor (busy hours + averaging), at slide 6's 15-minute snapshots and strikes.",
                      "Red: Kalshi higher; gray: Deribit higher. Median of the strike-snapshots' own spreads per cell, ±3 SD."])))
        _view = mo.vstack([
            mo.md(f"""
            **{len(_s):,} of {_n_all:,}** strike-snapshots have a reliable factor (the rest start inside Deribit's
            settlement window or sit where the model option is too cheap to invert). Against the busy-hours-only
            stretch, the spread moves by a median **{_cells.change.median():+.2f}** points per cell, most on the last
            day ({_cells[_cells.days == 1].change.median():+.2f}).
            """),
            spread_chart,
        ])
    _view
    return (spread_chart,)


@app.cell
def _(ROOT, factor_chart, mo, spread_chart):
    # Save the figures when run as a script (python 17_settlement_factor.py); opening the notebook writes nothing
    if not mo.running_in_notebook():
        _out = ROOT / "outputs" / "presentation"
        _out.mkdir(parents=True, exist_ok=True)
        for _name, _chart in [("17a_settlement_factor", factor_chart), ("17b_spread_on_kalshi_terms", spread_chart)]:
            if _chart is not None:
                _chart.save(_out / f"{_name}.png", scale_factor=2)
                _chart.save(_out / f"{_name}.html")
    return


if __name__ == "__main__":
    app.run()
