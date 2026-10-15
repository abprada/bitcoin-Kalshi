"""Kalshi vs Deribit implied vols, paired trade by trade (used by 05_kalshi_vs_deribit.py and 06_results_summary.py).

Kalshi has a full ladder of quotes every minute; Deribit only has trades, at scattered strikes and times. So each
Deribit trade is the anchor, and Kalshi is read at the same moment:

1. Deribit: the trade's mark price (Deribit's fair value at that instant) inverted for a vol (`deribit.mark_iv`).
   Fills at the same strike whose nearest minute end is the same are one observation (usually pieces of one order).
2. Kalshi: the ladder at that minute end (at most 30 seconds from the trade), turned into a smile exactly as in 03
   (`iv.ladder_quotes`, `iv.snapshot_iv`), read at the Deribit strike: the strike's own vol when it is on the ladder,
   otherwise linear in log-strike between its neighbours, never beyond the ladder.
3. Expiry: Deribit's Friday option expires 08:00 UTC, Kalshi's weekly 5pm ET the same day. Deribit's vol is
   stretched to Kalshi's expiry with the hour-of-week variance profile (`deribit.horizon_vol_ratios`).
4. Gap = Kalshi vol - Deribit vol, in vol points, with and without step 3. Pooled by distance from the money in
   standard deviations, ln(K / Kalshi forward) / (Kalshi ATM vol x sqrt(time left)), and by time to expiry.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lib import compare, deribit, iv

SD_EDGES = np.round(np.arange(-3.25, 3.26, 0.5), 2)   # half-SD buckets; the middle one, |sd| <= 0.25, is "at the money"
SD_CENTRES = np.round((SD_EDGES[:-1] + SD_EDGES[1:]) / 2, 2)
TTE_EDGES = [24, 48, 72, 120, 170]                    # Kalshi hours to expiry
TTE_LABELS = ["1–2 days", "2–3 days", "3–5 days", "5–7 days"]
FAR_FROM_EXPIRY_HOURS = 96   # 4+ days left: the 13 extra hours are about a tenth of the horizon, the stretch about x1.03
KALSHI_COLS = ["event_ticker", "strike", "ts", "tau_years", "bid", "ask", "mid", "spread", "quote_age_min", "spot",
               "exchange_open", "usable"]


def deribit_observations(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per expiry, strike and Kalshi minute: the trades whose nearest minute end is that minute.

    `trades` as from `deribit.load_trades`. Returns the median mark vol (`deribit_iv`), index and time to expiry,
    the number of trades and calls, the amount traded, and the largest distance of a trade from the minute
    (`max_offset_s`, at most 30 seconds).
    """
    t = trades.assign(minute=trades.ts.dt.round("min"), is_call=(trades.cp == "C").astype(int))
    o = (t.groupby(["expiry", "strike", "minute"])
         .agg(deribit_iv=("iv_mark", "median"), index_price=("index_price", "median"), tau_d=("tau_years", "median"),
              n_trades=("iv_mark", "size"), n_calls=("is_call", "sum"), amount=("amount", "sum"),
              first_trade=("ts", "min"), last_trade=("ts", "max"))
         .reset_index())
    o["max_offset_s"] = np.maximum((o.minute - o.first_trade).dt.total_seconds().abs(),
                                   (o.last_trade - o.minute).dt.total_seconds().abs())
    return o


def iv_at_strikes(K, ladder_K, ladder_iv) -> np.ndarray:
    """Kalshi's vol at strikes `K` from one smile: linear in log-strike between the ladder strikes that have a vol
    (so exactly the ladder's vol on a ladder strike); NaN outside them."""
    ladder_K, ladder_iv = np.asarray(ladder_K, dtype=float), np.asarray(ladder_iv, dtype=float)
    ok = np.isfinite(ladder_iv)
    if ok.sum() < 2:
        return np.full(np.shape(K), np.nan)
    o = np.argsort(ladder_K[ok])
    return compare.interp_vol(np.log(np.asarray(K, dtype=float)), np.log(ladder_K[ok][o]), ladder_iv[ok][o])


def kalshi_smiles(quotes: pd.DataFrame, minutes) -> dict:
    """Kalshi's smile at each of `minutes` for one weekly, built as in 03.

    `quotes` are one weekly's clean quotes (02's columns). Returns {minute: (snapshot dict, strikes DataFrame)} for
    the minutes whose ladder passes `iv.ladder_ok`; the snapshot dict is `iv.snapshot_iv`'s plus `spot` and `tau_years`.
    """
    ladder, outer = iv.ladder_quotes(quotes[quotes.ts.isin(minutes)])
    outer_at = {m: g for m, g in outer.groupby("ts")}
    none = outer.iloc[:0]
    out = {}
    for m, g in ladder.groupby("ts"):
        g = g.sort_values("strike")
        spot, tau = float(g.spot.iloc[0]), float(g.tau_years.iloc[0])
        if not iv.ladder_ok(len(g), g.strike.iloc[0], g.strike.iloc[-1], spot):
            continue
        o = outer_at.get(m, none)
        strikes, snap = iv.snapshot_iv(g.strike, g.mid, 1 / g.spread, spot, tau,
                                       K_out=o.strike, bid_out=o.bid, ask_out=o.ask)
        out[m] = ({**snap, "spot": spot, "tau_years": tau}, strikes)
    return out


def build_pairs(root: Path) -> dict:
    """Pair every Deribit trade with Kalshi's vol at the same strike and minute.

    Deribit: trades in Friday expiries (the ones whose date matches a Kalshi weekly), with a mark vol, before expiry;
    every hour from the full file (`deribit.trades_path`), 19:30-22:29 UTC only from the collaborator's CSV. Kalshi: the weekly expiring 5pm ET that Friday, its ladder at the trade's minute, used if
    it has an ATM vol and a fine enough strike grid (`compare.usable_kalshi_snapshots`). Needs data/clean from 02.

    Returns a dict: `pairs` (one row per observation with both vols), `funnel` (how many observations survive
    each step), `trades` (the Deribit trades considered), `hour_profile`.
    """
    clean = root / "data" / "clean"
    path = deribit.trades_path(root)
    trades = deribit.load_trades(path, expiry_weekday=4)
    trades = trades[trades.iv_mark.notna() & (trades.ts < trades.expiry)
                    & ((trades.window == "us_afternoon") | deribit.is_full(path))]
    obs = deribit_observations(trades)
    spot = pd.read_parquet(clean / "btc_spot_1m.parquet", columns=["ts", "spot"]).set_index("ts").spot
    hour_profile = deribit.hourly_variance(spot)

    parts, n_live, n_ladder = [], 0, 0
    for f in sorted((clean / "kalshi_weekly_quotes").glob("*.parquet")):
        q = pd.read_parquet(f, columns=KALSHI_COLS)
        kalshi_expiry = (q.ts + pd.to_timedelta(q.tau_years.astype(float) * deribit.YEAR_SECONDS, unit="s")).median().round("min")
        deribit_expiry = kalshi_expiry.normalize() + pd.Timedelta(hours=8)
        o = obs[(obs.expiry == deribit_expiry) & obs.minute.isin(set(q.ts))]
        if o.empty:
            continue
        n_live += len(o)
        smiles = kalshi_smiles(q, o.minute.unique())
        for m, g in o.groupby("minute"):
            if m not in smiles:
                continue
            snap, strikes = smiles[m]
            if not compare.usable_kalshi_snapshots(pd.DataFrame([snap])).iloc[0]:
                continue
            n_ladder += len(g)
            K = g.strike.to_numpy()
            on_ladder = strikes.set_index("strike")
            parts.append(g.assign(
                event_ticker=q.event_ticker.iloc[0], kalshi_expiry=kalshi_expiry, deribit_expiry=deribit_expiry,
                spot=snap["spot"], forward=snap["forward"], kalshi_atm=snap["atm_iv"], tau_k=snap["tau_years"],
                kalshi_iv=iv_at_strikes(K, strikes.strike, strikes.iv),
                kalshi_mid=on_ladder.mid.reindex(K).to_numpy(), on_ladder=np.isin(K, strikes.strike)))
    p = pd.concat(parts, ignore_index=True)
    n_inside = int(p.kalshi_iv.notna().sum())
    p = p[p.kalshi_iv.notna()].reset_index(drop=True)

    p["horizon_ratio"] = deribit.horizon_vol_ratios(p.minute, p.kalshi_expiry, p.deribit_expiry, hour_profile)
    p["deribit_iv_adj"] = p.deribit_iv * p.horizon_ratio
    p["gap_raw"] = 100 * (p.kalshi_iv - p.deribit_iv)
    p["gap_adj"] = 100 * (p.kalshi_iv - p.deribit_iv_adj)
    p["kalshi_hours"] = p.tau_k * 8760
    p["log_moneyness"] = np.log(p.strike / p.forward)
    p["sd"] = p.log_moneyness / (p.kalshi_atm * np.sqrt(p.tau_k))
    p["sd_bucket"] = pd.cut(p.sd, SD_EDGES, labels=SD_CENTRES)
    p["tte_bucket"] = pd.cut(p.kalshi_hours, TTE_EDGES, labels=TTE_LABELS)
    p["day"] = p.minute.dt.date

    funnel = pd.DataFrame([
        (f"Deribit trades: {'all day' if deribit.is_full(path) else 'afternoon window'}, Friday expiries, with a mark vol",
         len(trades)),
        ("… as observations (fills at one strike and minute merged)", len(obs)),
        ("… while Kalshi's weekly for that Friday is trading", n_live),
        ("… with a usable Kalshi ladder that minute", n_ladder),
        ("… at a strike inside Kalshi's ladder: pairs", n_inside),
    ], columns=["step", "count"])
    return dict(pairs=p, funnel=funnel, trades=trades, hour_profile=hour_profile)


def gap_table(pairs: pd.DataFrame, by, col: str = "gap_adj", n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Per group: number of pairs and days, median gap, share where Kalshi is higher, and a 95% interval for the
    median that resamples whole days (pairs on the same day share the market's mood, so they are not independent)."""
    by = [by] if isinstance(by, str) else list(by)
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in pairs.dropna(subset=[col]).groupby(by, observed=True):
        per_day = [d[col].to_numpy() for _, d in g.groupby("day")]
        boot = [np.median(np.concatenate([per_day[i] for i in rng.integers(0, len(per_day), len(per_day))]))
                for _ in range(n_boot)]
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "pairs": len(g), "days": len(per_day), "median_gap": g[col].median(),
                     "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                     "kalshi_higher": (g[col] > 0).mean()})
    return pd.DataFrame(rows)


def atm(pairs: pd.DataFrame) -> pd.DataFrame:
    """Pairs at the money: within a quarter of a standard deviation of Kalshi's forward."""
    return pairs[pairs.sd.abs() <= 0.25]


def far_from_expiry(pairs: pd.DataFrame, hours: float = FAR_FROM_EXPIRY_HOURS) -> pd.DataFrame:
    """Pairs with at least `hours` left on Kalshi's contract, where the expiry mismatch matters least."""
    return pairs[pairs.kalshi_hours >= hours]


# Kalshi hours to expiry, one row per day: Deribit trades only in the US afternoon, so every pair is within about two
# hours of a whole number of days before Kalshi's Friday 5pm expiry; rows are centred on 1, 2, ..., 7 days
SURFACE_HOURS = list(range(12, 181, 24))


def surface(pairs: pd.DataFrame, hour_edges=SURFACE_HOURS, sd_edges=SD_EDGES, min_pairs: int = 40) -> pd.DataFrame:
    """Kalshi and Deribit (stretched) implied-vol surfaces on the same pairs, plus their gap, by time to expiry and
    distance from the money in standard deviations.

    As for 03's surface, vols are pooled as shapes: each pair's vols divided by Kalshi's ATM vol at that minute, the
    median per cell, times the median Kalshi ATM vol of that time-to-expiry row; this keeps calm and volatile weeks
    from distorting the edges. The gap is the median of the pairs' own gaps (vol points), not the difference of the
    two surfaces. Cells with fewer than `min_pairs` pairs are dropped. One row per cell: `hours_lo`, `hours_hi`,
    `sd` (bucket centre), `kalshi`, `deribit`, `gap`, `pairs`, `days`.
    """
    d = pairs.assign(h=pd.cut(pairs.kalshi_hours, hour_edges), m=pd.cut(pairs.sd, sd_edges),
                     k_ratio=pairs.kalshi_iv / pairs.kalshi_atm, d_ratio=pairs.deribit_iv_adj / pairs.kalshi_atm
                     ).dropna(subset=["h", "m"])
    row_atm = d.groupby("h", observed=True).kalshi_atm.median()
    g = (d.groupby(["h", "m"], observed=True)
         .agg(k_ratio=("k_ratio", "median"), d_ratio=("d_ratio", "median"), gap=("gap_adj", "median"),
              pairs=("gap_adj", "size"), days=("day", "nunique"))
         .reset_index())
    g = g[g.pairs >= min_pairs]
    atm_row = g.h.map(row_atm).astype(float)
    return pd.DataFrame({"hours_lo": [i.left for i in g.h], "hours_hi": [i.right for i in g.h],
                         "sd": [round(i.mid, 2) for i in g.m], "kalshi": g.k_ratio * atm_row,
                         "deribit": g.d_ratio * atm_row, "gap": g.gap, "pairs": g.pairs, "days": g.days}).reset_index(drop=True)


def gap_with_profile(pairs: pd.DataFrame, profile: pd.Series) -> pd.Series:
    """Kalshi - Deribit (vol points) with Deribit stretched by another hour-of-week variance profile, to test how
    much the result depends on the historical one."""
    ratio = deribit.horizon_vol_ratios(pairs.minute, pairs.kalshi_expiry, pairs.deribit_expiry, profile)
    return 100 * (pairs.kalshi_iv - pairs.deribit_iv * ratio)
