"""Two dislocation strategies as portfolios, with execution variants and an out-of-sample test (12_strategies_ab.py).

A · Weeklies, Kalshi unusually CHEAP vs Deribit (dislocation <= -threshold): Deribit spiked and Kalshi did not
    follow. SELL Deribit's straddle at the strike nearest Kalshi's forward, delta-hedge hourly with the perpetual,
    hold to Deribit's expiry before Kalshi's (Friday 08:00 UTC), or close after `hold` hours.
B · Dailies, Kalshi unusually RICH vs Deribit (dislocation >= threshold): SELL Kalshi's straddle (the strip of
    binaries, `vol_spread.strip_at`), delta-hedge hourly, hold to Kalshi's settlement or close after `hold` hours.

The trades themselves come from `vol_spread.backtest` (every hour the signal is on); this module turns them into a
book: one position per contract (the first hour the dislocation crosses the threshold), each sized to $1,000 of
vega per vol point, P&L booked on the day the position closes.

Execution variants (costs per trade, from the backtest's components):
- taker:  cross the spread on every option/binary, full fees, perpetual hedge at the taker fee (0.05%);
- maker:  options/binaries filled at mid (Kalshi pays a quarter of its taker fee, an assumption; Deribit keeps its
          0.03% fee, which makers also pay), perpetual hedge at 0;
- maker, taker hedge: as maker, but the hedge pays the perpetual taker fee;
- half spread: in between, fills halfway between mid and the touch, full fees, taker hedge;
- gross:  no costs at all.
"""

import numpy as np
import pandas as pd

VEGA_USD = 1_000.0          # every position is sized to $1,000 per vol point of vega
KALSHI_MAKER_FEE_SHARE = 0.25

STRATEGIES = {
    "A": dict(name="A · sell Deribit vol when Kalshi is unusually cheap (weeklies)", kind="weekly", sign=-1, leg="d"),
    "B": dict(name="B · sell Kalshi vol when Kalshi is unusually rich (dailies)", kind="daily", sign=1, leg="k"),
}
EXECUTION = ["gross", "maker", "maker, taker hedge", "half spread", "taker"]


def costs(tr: pd.DataFrame, leg: str, mode: str) -> pd.Series:
    """USD cost per trade of one leg under an execution mode (per 1 BTC Kalshi strip, as in the backtest)."""
    if mode == "gross":
        return pd.Series(0.0, index=tr.index)
    if leg == "d":
        return {"taker": tr.d_spread + tr.d_fee + tr.d_perp, "maker": tr.d_fee,
                "maker, taker hedge": tr.d_fee + tr.d_perp,
                "half spread": 0.5 * tr.d_spread + tr.d_fee + tr.d_perp}[mode]
    return {"taker": tr.k_spread + tr.k_fee + tr.k_perp, "maker": KALSHI_MAKER_FEE_SHARE * tr.k_fee,
            "maker, taker hedge": KALSHI_MAKER_FEE_SHARE * tr.k_fee + tr.k_perp,
            "half spread": 0.5 * tr.k_spread + tr.k_fee + tr.k_perp}[mode]


def book(tr: pd.DataFrame, strategy: str, threshold: float, hold: str, mode: str) -> pd.DataFrame:
    """One position per contract: the first hour |dislocation| >= threshold on the strategy's side.

    `tr`: `vol_spread.backtest` rows (signal = dislocation) with `hold` as a string. Returns one row per position
    with `pnl_vol` (vol points of vega) and `pnl` (USD at VEGA_USD per vol point), `close` (when it is booked).
    """
    s = STRATEGIES[strategy]
    x = tr[(tr.kind == s["kind"]) & (np.sign(tr.signal) == s["sign"]) & (tr.signal.abs() >= threshold)
           & (tr.hold == hold)]
    x = x.sort_values("ts").groupby("event_ticker").head(1).copy()
    gross = x[f"{s['leg']}_gross"]
    x["pnl_vol"] = (gross - costs(x, s["leg"], mode)) / x.vega_k
    x["pnl"] = x.pnl_vol * VEGA_USD
    if hold == "expiry":
        x["close"] = (x.deribit_expiry if s["leg"] == "d" else x.kalshi_expiry).dt.normalize()
    else:
        x["close"] = (x.ts + pd.Timedelta(hours=int(hold))).dt.normalize()
    return x.sort_values("close")


def metrics(b: pd.DataFrame, start, end) -> dict:
    """Trades, hit rate, average and total P&L, Sharpe (daily P&L over every calendar day in [start, end), x sqrt
    365), maximum drawdown, worst trade, and a t-statistic of the average trade."""
    days = pd.date_range(start, end, freq="D", tz="UTC", inclusive="left")
    daily = b.groupby("close").pnl.sum().reindex(days, fill_value=0.0)
    cum = daily.cumsum()
    n = len(b)
    sd = b.pnl_vol.std() if n > 1 else np.nan
    return dict(trades=n, hit=(b.pnl > 0).mean() if n else np.nan, avg_vol=b.pnl_vol.mean() if n else np.nan,
                t=b.pnl_vol.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan,
                total=b.pnl.sum(), sharpe=np.sqrt(365) * daily.mean() / daily.std() if daily.std() > 0 else np.nan,
                max_dd=(cum - cum.cummax()).min(), worst=b.pnl.min() if n else np.nan,
                best=b.pnl.max() if n else np.nan)


def out_of_sample(tr: pd.DataFrame, strategy: str, hold: str, mode: str, window: int, split="2026-05-01",
                  start="2026-01-15", end="2026-10-01", thresholds=(3, 4, 5, 6, 8, 10), min_trades: int = 4) -> dict:
    """Pick the threshold with the best Sharpe on [start, split) (at least `min_trades` trades), then report the
    same rule on [split, end) alongside every threshold's out-of-sample result."""
    t = tr[tr.window == window]
    rows = []
    for thr in thresholds:
        b = book(t, strategy, thr, hold, mode)
        ins, oos = b[b.close < split], b[b.close >= split]
        rows.append({"threshold": thr, **{f"in_{k}": v for k, v in metrics(ins, start, split).items()},
                     **{f"out_{k}": v for k, v in metrics(oos, split, end).items()}})
    r = pd.DataFrame(rows)
    ok = r[r.in_trades >= min_trades]
    chosen = ok.loc[ok.in_sharpe.idxmax(), "threshold"] if len(ok) else np.nan
    return {"table": r, "chosen": chosen}
