"""Summarize raw weekly Kalshi BTC volume and duration-weighted spreads.

Run with .venv/bin/python liquidity_summary.py. Outputs CSVs and an HTML chart.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw/kalshi_btc_weekly"
OUT = ROOT / "outputs/liquidity"
BINS = [0, 1 / 12, 1, 6, 24, 72, 171]
LABELS = ["0–5 min", "5–60 min", "1–6 h", "6–24 h", "1–3 d", "3–7 d"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(RAW / "events.parquet").set_index("event_ticker")
    rows, daily = [], []
    for path in sorted((RAW / "candles").glob("*.parquet")):
        c = pd.read_parquet(path).sort_values(["market_ticker", "ts"])
        event = c.event_ticker.iloc[0]
        expiry = events.loc[event, "close_time"]
        tau = (expiry - c.ts).dt.total_seconds().to_numpy() / 3600
        nxt = c.groupby("market_ticker").ts.shift(-1).fillna(expiry)
        end_tau = (expiry - nxt).dt.total_seconds().to_numpy() / 3600
        bid, ask = c.bid_close.to_numpy(), c.ask_close.to_numpy()
        valid = (bid > 0) & (ask < 1) & (ask > bid)
        mid = (bid + ask) / 2
        central = valid & (mid >= .1) & (mid <= .9)
        spread = 100 * (ask - bid)
        # Candle volume belongs to (ts - 1 minute, ts]; use its midpoint
        # so the final five minutes include exactly five one-minute bars.
        bucket = pd.cut(tau + .5 / 60, BINS, labels=False, include_lowest=True)
        life = (expiry - events.loc[event, "open_time"]).total_seconds() / 3600
        for i, label in enumerate(LABELS):
            # Split carried quotes exactly at maturity-bin boundaries.
            weight = np.maximum(0, np.minimum(tau, BINS[i + 1]) - np.maximum(end_tau, BINS[i]))
            exposure = max(0, min(life, BINS[i + 1]) - BINS[i])
            rows.append(dict(event_ticker=event, maturity=label,
                             volume=c.volume.to_numpy()[bucket == i].sum(), event_hours=exposure,
                             quote_hours=weight.sum(), two_sided_hours=weight[valid].sum(),
                             spread_cent_hours=(weight[valid] * spread[valid]).sum(),
                             central_hours=weight[central].sum(),
                             central_spread_cent_hours=(weight[central] * spread[central]).sum()))
        daily.append(c.groupby(c.ts.dt.floor("D")).volume.sum())
    per_event = pd.DataFrame(rows)
    agg = per_event.groupby("maturity", sort=False).sum(numeric_only=True).reindex(LABELS)
    agg["mean_volume_per_event"] = agg.volume / len(events)
    agg["contracts_per_event_hour"] = agg.volume / agg.event_hours
    agg["volume_share"] = agg.volume / agg.volume.sum()
    agg["mean_spread_cents"] = agg.spread_cent_hours / agg.two_sided_hours
    agg["central_mean_spread_cents"] = agg.central_spread_cent_hours / agg.central_hours
    agg["two_sided_share"] = agg.two_sided_hours / agg.quote_hours
    day = pd.concat(daily).groupby(level=0).sum().rename("volume").to_frame()
    week = events[["close_time", "volume", "n_strikes"]].sort_values("close_time")
    week["candle_volume"] = per_event.groupby("event_ticker").volume.sum()
    week["candle_minus_reported_volume"] = week.candle_volume - week.volume
    for name, df in [("daily_volume", day), ("weekly_volume", week),
                     ("maturity_summary", agg), ("event_maturity", per_event)]:
        df.to_csv(OUT / f"{name}.csv", index=name != "event_maturity")
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "Daily trading volume (UTC)", "Volume by weekly expiry",
        "Average volume rate by time to expiry", "Duration-weighted YES bid–ask spreads"])
    fig.add_trace(go.Scatter(x=day.index, y=day.volume, name="Daily volume"), row=1, col=1)
    fig.add_trace(go.Bar(x=week.close_time, y=week.volume, name="Weekly volume"), row=1, col=2)
    fig.add_trace(go.Bar(x=agg.index, y=agg.contracts_per_event_hour, name="Contracts / event hour"), row=2, col=1)
    for col, name in [("mean_spread_cents", "All two-sided quotes"),
                      ("central_mean_spread_cents", "Midpoint 10–90¢")]:
        fig.add_trace(go.Scatter(x=agg.index, y=agg[col], mode="lines+markers", name=name), row=2, col=2)
    fig.update_yaxes(title_text="Contracts", row=1, col=1)
    fig.update_yaxes(title_text="Contracts", row=1, col=2)
    fig.update_yaxes(title_text="Contracts / hour", row=2, col=1)
    fig.update_yaxes(title_text="Cents per $1 contract", row=2, col=2)
    fig.update_layout(template="plotly_white", height=800, title="Kalshi weekly BTC liquidity", legend=dict(orientation="h", y=-.15))
    fig.write_html(OUT / "liquidity.html", include_plotlyjs=True)
    print(week.describe().to_string())
    print(agg[["mean_volume_per_event", "contracts_per_event_hour", "volume_share", "mean_spread_cents", "central_mean_spread_cents", "two_sided_share"]].round(3).to_string())
    print(f"Events: {len(events)}; total candle volume: {agg.volume.sum():,.2f}; "
          f"reported: {week.volume.sum():,.2f}; daily span: {day.index.min()} to {day.index.max()}")


if __name__ == "__main__":
    main()
