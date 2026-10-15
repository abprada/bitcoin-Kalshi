"""Putting Kalshi and Deribit in the same units (used by 05_kalshi_vs_deribit.py).

Deribit's vols are read as a smile in log-moneyness ln(K / index), from the mark prices of trades near one moment.
Interpolating that smile to a Kalshi strike gives a Deribit vol there (the vol comparison). Pricing a tight call
spread on the same smile gives a Deribit price for Kalshi's binary (the price comparison, handout Q1c/Q2b):
D(K) = -dC/dK ≈ [C(K - h) - C(K + h)] / 2h. The spread uses each leg's own smile vol, so the skew is included.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lib import deribit, iv

SPREAD_HALF_WIDTH = 250.0   # dollars: the call spread runs from K - 250 to K + 250
WINDOW_MIN = 15             # Deribit trades within this many minutes of a Kalshi snapshot form its smile
MIN_DERIBIT_STRIKES = 3     # and they must cover at least 3 strikes, on both sides of the money
AFTERNOON_UTC = (19 * 60 + 30, 22 * 60 + 29)   # the us_afternoon window, in minutes after midnight UTC
MONEYNESS_EDGES = [-0.15, -0.05, -0.02, -0.005, 0.005, 0.02, 0.05, 0.15]
MONEYNESS_LABELS = ["5–15% below", "2–5% below", "0.5–2% below", "within 0.5%", "0.5–2% above", "2–5% above",
                    "5–15% above"]
KALSHI_FEE_RATE = 0.07      # Kalshi's general taker fee per order = round up(rate x C x P x (1 - P)) for C contracts
                            # at price P (Kalshi fee schedule, checked Oct 2026; makers pay a quarter of it)
MIN_STRIKES_PER_SD = 2      # Kalshi snapshot grid check, as in 03's "clean"


def usable_kalshi_snapshots(snaps: pd.DataFrame) -> pd.Series:
    """Kalshi snapshots used in the comparison: an ATM vol exists and the strike grid resolves the distribution.

    03's "clean" flag also requires Kalshi's implied forward within +/-25 bps of exchange spot. That check is left
    out here on purpose: the vols are inverted against Kalshi's own forward, so a basis does not distort them, large
    bases neither lag nor lead BTC, and dropping them would remove exactly the cases where Kalshi disagrees with
    spot. Bad individual quotes are still screened strike by strike in 02/03 (spread <= 20c, quote age <= 15 min).
    """
    return snaps.atm_iv.notna() & (snaps.strikes_per_sd >= MIN_STRIKES_PER_SD)


def smile_from_trades(trades: pd.DataFrame, vol_col: str = "iv_mark") -> pd.DataFrame:
    """One point per strike: median log-moneyness and median vol over the given trades, sorted by moneyness."""
    x = trades.dropna(subset=[vol_col])
    return (x.groupby("strike").agg(log_moneyness=("log_moneyness", "median"), vol=(vol_col, "median"),
                                    n_trades=(vol_col, "size"))
            .reset_index().sort_values("log_moneyness", ignore_index=True))


def interp_vol(k, smile_k, smile_vol) -> np.ndarray:
    """Linear interpolation of the smile at log-moneyness `k`; NaN outside the traded range (no extrapolation)."""
    k = np.asarray(k, dtype=float)
    out = np.interp(k, smile_k, smile_vol)
    return np.where((k >= np.min(smile_k)) & (k <= np.max(smile_k)), out, np.nan)


def digital_from_smile(K, F: float, tau: float, smile_k, smile_vol, h: float = SPREAD_HALF_WIDTH) -> np.ndarray:
    """Price of "$1 if S_T > K" from a call spread K - h / K + h priced on the smile; NaN outside the smile."""
    K = np.asarray(K, dtype=float)
    lo, hi = K - h, K + h
    v_lo, v_hi = interp_vol(np.log(lo / F), smile_k, smile_vol), interp_vol(np.log(hi / F), smile_k, smile_vol)
    return (iv.bs_call(F, lo, v_lo, tau) - iv.bs_call(F, hi, v_hi, tau)) / (2 * h)


def nearest_marks(trades: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    """Per call strike, the trade closest in time to `ts`: its mark in dollars, index and time. No model involved."""
    c = trades[trades.cp == "C"]
    if c.empty:
        return pd.DataFrame(columns=["mark_usd", "index_price", "ts"])
    c = c.loc[(c.ts - ts).abs().groupby(c.strike).idxmin()]
    return (c.assign(mark_usd=c.mark_price_btc * c.index_price)
            .set_index("strike")[["mark_usd", "index_price", "ts"]])


def direct_digital(K, marks: pd.DataFrame, half_width: float = 500.0) -> np.ndarray:
    """Model-free Deribit price of "$1 if S_T > K" at Deribit's own expiry: [C(K - w) - C(K + w)] / 2w from the
    observed call marks, only where both strikes have a mark (NaN otherwise)."""
    K = np.asarray(K, dtype=float)
    lo = marks.mark_usd.reindex(K - half_width).to_numpy(dtype=float)
    hi = marks.mark_usd.reindex(K + half_width).to_numpy(dtype=float)
    return (lo - hi) / (2 * half_width)


def kalshi_fee(price, rate: float = KALSHI_FEE_RATE, contracts: int = 1) -> np.ndarray:
    """Kalshi taker fee per contract in dollars, for an order of `contracts` at `price`: the order's fee
    rate x C x P x (1 - P) is rounded up to the cent, so the rounding matters less the bigger the order."""
    p = np.asarray(price, dtype=float)
    return np.ceil(np.round(100 * rate * contracts * p * (1 - p), 9)) / 100 / contracts


def build_comparison(root: Path, window_min: int = WINDOW_MIN, min_deribit_strikes: int = MIN_DERIBIT_STRIKES) -> dict:
    """Match every usable Kalshi snapshot in the US afternoon to Deribit and put both in the same units.

    For each Kalshi snapshot (19:30-22:29 UTC) and the Deribit Friday expiry with the same date: Deribit's smile from
    the marks of trades within `window_min` minutes, interpolated to Kalshi's strikes (vols), stretched to Kalshi's
    expiry with the hour-of-week variance profile, and priced as K -/+ 250 call spreads (Kalshi's binary). Also the
    unstretched and the model-free (observed K -/+ 500 marks) Deribit prices. Needs data/derived from 03.

    Returns a dict: `matched` (one row per Kalshi strike and snapshot), `smiles` (per snapshot: Deribit smile, spot,
    times to expiry, stretch ratio), `snaps` (the Kalshi snapshots considered), `trades` (the Deribit trades used),
    `hour_profile`.
    """
    clean, derived = root / "data" / "clean", root / "data" / "derived"

    # Deribit: trades in Friday expiries (the only ones whose date matches a Kalshi weekly); from the full file every
    # trade near a snapshot counts, from the collaborator's CSV only its afternoon window
    path = deribit.trades_path(root)
    trades = deribit.load_trades(path, expiry_weekday=4)
    trades = trades[trades.iv_mark.notna() & (trades.tau_years > 0)
                    & ((trades.window == "us_afternoon") | deribit.is_full(path))]
    by_expiry = {e: g for e, g in trades.groupby("expiry")}

    # Kalshi: usable snapshots in the same window, before the Deribit option with the same date expires
    snaps = pd.read_parquet(derived / "iv_snapshots.parquet")
    snaps["expiry"] = snaps.ts + pd.to_timedelta(snaps.hours_to_expiry, unit="h")
    minute = snaps.ts.dt.hour * 60 + snaps.ts.dt.minute
    snaps = snaps[usable_kalshi_snapshots(snaps) & minute.between(*AFTERNOON_UTC)].copy()
    snaps["deribit_expiry"] = (pd.to_datetime(snaps.expiry.dt.date.astype(str)).dt.tz_localize("UTC")
                               + pd.Timedelta(hours=8))
    snaps = snaps[snaps.ts < snaps.deribit_expiry]

    spot = pd.read_parquet(clean / "btc_spot_1m.parquet", columns=["ts", "spot"]).set_index("ts").spot
    hour_profile = deribit.hourly_variance(spot)
    strikes = pd.read_parquet(derived / "iv_strikes.parquet",
                              columns=["snap", "strike", "mid", "D_used", "iv", "spot", "tau_years"],
                              filters=[("snap", "in", snaps.snap.tolist())])
    by_snap = {s: g for s, g in strikes.groupby("snap")}

    rows, smiles = [], {}
    for s in snaps.itertuples():
        g = by_expiry.get(s.deribit_expiry)
        if g is None or s.snap not in by_snap:
            continue
        w = g[(g.ts - s.ts).abs() <= pd.Timedelta(minutes=window_min)]
        sm = smile_from_trades(w)
        if len(sm) < min_deribit_strikes or not (sm.log_moneyness.min() < 0 < sm.log_moneyness.max()):
            continue
        kk = by_snap[s.snap]
        F, tau_k = float(kk.spot.iloc[0]), float(kk.tau_years.iloc[0])
        tau_d = (s.deribit_expiry - s.ts).total_seconds() / deribit.YEAR_SECONDS
        ratio = deribit.horizon_vol_ratio(s.ts, s.expiry, s.deribit_expiry, hour_profile)
        K = kk.strike.to_numpy()
        vol = interp_vol(np.log(K / F), sm.log_moneyness, sm.vol)
        smiles[s.snap] = dict(smile=sm, F=F, tau_k=tau_k, tau_d=tau_d, ratio=ratio, ts=s.ts,
                              event_ticker=s.event_ticker, kalshi_hours=s.hours_to_expiry, kalshi_atm=s.atm_iv)
        rows.append(kk.assign(
            ts=s.ts, event_ticker=s.event_ticker, kalshi_hours=s.hours_to_expiry,
            deribit_hours=tau_d * 8_760, horizon_ratio=ratio, n_deribit_strikes=len(sm),
            deribit_vol=vol, deribit_vol_adj=vol * ratio,
            deribit_digital_adj=digital_from_smile(K, F, tau_k, sm.log_moneyness, sm.vol * ratio),
            deribit_digital_raw=digital_from_smile(K, F, tau_d, sm.log_moneyness, sm.vol),
            # no model: observed call marks at K - 500 and K + 500, nearest in time to the snapshot
            deribit_digital_direct=direct_digital(K, nearest_marks(w, s.ts)),
        ))
    matched = pd.concat(rows, ignore_index=True)

    # Kalshi's bid and ask at each matched snapshot, from the clean quotes
    quotes = []
    for ev, g in matched.groupby("event_ticker"):
        q = pd.read_parquet(clean / "kalshi_weekly_quotes" / f"{ev}.parquet", columns=["strike", "ts", "bid", "ask"])
        quotes.append(q[q.ts.isin(g.ts.unique())].assign(event_ticker=ev))
    matched = matched.merge(pd.concat(quotes, ignore_index=True), on=["event_ticker", "ts", "strike"], how="left")
    matched["log_moneyness"] = np.log(matched.strike / matched.spot)
    matched["moneyness_bucket"] = pd.cut(matched.log_moneyness, MONEYNESS_EDGES, labels=MONEYNESS_LABELS)
    matched["tte_bucket"] = pd.cut(matched.kalshi_hours, [24, 48, 72, 120, 170],
                                   labels=["1–2 days", "2–3 days", "3–5 days", "5–7 days"])
    matched["vol_gap_raw"] = 100 * (matched.iv - matched.deribit_vol)
    matched["vol_gap_adj"] = 100 * (matched.iv - matched.deribit_vol_adj)
    matched["price_gap_c"] = 100 * (matched.mid - matched.deribit_digital_adj)
    return dict(matched=matched, smiles=smiles, snaps=snaps, trades=trades, hour_profile=hour_profile)


SNAPSHOT_HALF_WINDOW = "30min"   # Deribit trades within +/- this of a Kalshi snapshot form Deribit's smile there


def snapshot_comparison(root: Path, hour_profile: pd.Series, half_window: str = SNAPSHOT_HALF_WINDOW) -> dict:
    """Kalshi vs Deribit at every clean 15-minute Kalshi snapshot (03), each snapshot counting once whether or not
    Deribit traded at that minute (`pairs` instead anchors on Deribit trades, so busy moments count more).

    Deribit at a snapshot: its fitted smile for the Friday 08:00 expiry with the Kalshi weekly's date, from the trades
    within +/- `half_window` (`deribit_marks.smile_grid` on a 15-minute grid: a quadratic in log-moneyness per root
    year, at least 3 strikes around the forward). It is read at each Kalshi strike only inside the strikes it was
    fitted to (no extrapolation), and its constant is Deribit's ATM vol. Both are stretched to Kalshi's expiry with
    `hour_profile` (`deribit.horizon_vol_ratios`), as in `pairs`.

    Returns `strikes` (one row per Kalshi strike and snapshot with a Deribit vol, in `pairs`' column names: kalshi_iv,
    kalshi_atm, deribit_iv, deribit_iv_adj, gap_raw, gap_adj, sd, kalshi_hours, day, ...), `snaps` (one row per clean
    Kalshi snapshot: ATM vols of both markets, `has_deribit`) and `coverage` (counts).
    """
    from lib import deribit_marks as dm   # here, not at the top: deribit_marks imports deribit, which this module also uses
    from lib import pairs

    k = pd.read_parquet(Path(root) / "data" / "derived" / "iv_strikes.parquet",
                        columns=["snap", "event_ticker", "ts", "strike", "iv", "clean", "forward", "atm_iv", "tau_years",
                                 "hours_to_expiry", "sd_from_forward", "spot"])
    k = k[k.clean].drop(columns="clean")
    k["kalshi_expiry"] = (k.ts + pd.to_timedelta(k.hours_to_expiry, unit="h")).dt.round("h")
    k["deribit_expiry"] = k.kalshi_expiry.dt.normalize() + pd.Timedelta(hours=8)
    snaps = k.drop_duplicates("snap")[["snap", "event_ticker", "ts", "atm_iv", "tau_years", "hours_to_expiry",
                                       "kalshi_expiry", "deribit_expiry"]].reset_index(drop=True)

    # Deribit: trades in those Friday expiries, from a day before the first snapshot, on a 15-minute smile grid
    t = dm.read_full_trades(root, start=snaps.ts.min() - pd.Timedelta(days=1), end=snaps.ts.max(), max_tte_days=8)
    t = dm.add_forward(t[t.expiry.isin(set(snaps.deribit_expiry))])   # a Kalshi weekly lives a week: 8 days is enough
    grid = dm.smile_grid(t, freq="15min", half_window=half_window, centred=True)
    grid = grid.rename(columns={"expiry": "deribit_expiry", "t": "ts", "atm_iv": "deribit_atm", "tau_years": "tau_d"})

    snaps = snaps.merge(grid[["deribit_expiry", "ts", "deribit_atm", "n_trades"]], on=["deribit_expiry", "ts"], how="left")
    # How far the nearest Deribit trade in that expiry is from each snapshot, either side (seconds)
    us = "datetime64[us, UTC]"                       # merge_asof needs one resolution on both sides
    near = pd.merge_asof(snaps[["snap", "ts", "deribit_expiry"]].astype({"ts": us, "deribit_expiry": us}).sort_values("ts"),
                         t[["ts", "expiry"]].rename(columns={"ts": "trade_ts", "expiry": "deribit_expiry"})
                         .astype({"trade_ts": us, "deribit_expiry": us}).sort_values("trade_ts"),
                         left_on="ts", right_on="trade_ts", by="deribit_expiry", direction="nearest")
    snaps = snaps.merge(near.assign(nearest_trade_s=(near.trade_ts - near.ts).dt.total_seconds().abs())[
        ["snap", "nearest_trade_s"]], on="snap", how="left")
    snaps["deribit_live"] = snaps.ts < snaps.deribit_expiry
    snaps["has_deribit"] = snaps.deribit_atm.notna()
    snaps["horizon_ratio"] = np.nan
    live = snaps.deribit_live
    snaps.loc[live, "horizon_ratio"] = deribit.horizon_vol_ratios(
        snaps.ts[live], snaps.kalshi_expiry[live], snaps.deribit_expiry[live], hour_profile)
    snaps["deribit_atm_adj"] = snaps.deribit_atm * snaps.horizon_ratio
    snaps["gap_atm_raw"] = 100 * (snaps.atm_iv - snaps.deribit_atm)
    snaps["gap_atm_adj"] = 100 * (snaps.atm_iv - snaps.deribit_atm_adj)
    snaps["day"] = snaps.ts.dt.date

    # Strike level: Deribit's fitted vol at each Kalshi strike, only inside the strikes Deribit's smile was fitted to
    s = k.merge(grid, on=["deribit_expiry", "ts"]).merge(snaps[["snap", "horizon_ratio"]], on="snap")
    fwd_d = s["index"] * (1 + s.basis)
    m = np.log(s.strike / fwd_d) / np.sqrt(s.tau_d)
    inside = (m >= s.m_lo) & (m <= s.m_hi)
    s["deribit_iv"] = np.where(inside, s.deribit_atm + s.b * m + s.c * m**2, np.nan)
    s = s[s.deribit_iv.notna() & s.iv.notna()].copy()
    s = s.rename(columns={"iv": "kalshi_iv", "atm_iv": "kalshi_atm", "tau_years": "tau_k", "sd_from_forward": "sd",
                          "hours_to_expiry": "kalshi_hours", "ts": "minute"})
    s["deribit_iv_adj"] = s.deribit_iv * s.horizon_ratio
    s["gap_raw"] = 100 * (s.kalshi_iv - s.deribit_iv)
    s["gap_adj"] = 100 * (s.kalshi_iv - s.deribit_iv_adj)
    s["sd_bucket"] = pd.cut(s.sd, pairs.SD_EDGES, labels=pairs.SD_CENTRES)
    s["tte_bucket"] = pd.cut(s.kalshi_hours, pairs.TTE_EDGES, labels=pairs.TTE_LABELS)
    s["day"] = s.minute.dt.date
    s = s.drop(columns=["b", "c", "m_lo", "m_hi", "basis", "index", "n_strikes", "n_trades"]).reset_index(drop=True)

    coverage = pd.DataFrame([
        ("Clean Kalshi 15-minute snapshots", len(snaps)),
        ("… while Deribit's Friday expiry is still live (before Friday 08:00 UTC)", int(live.sum())),
        (f"… with a Deribit smile (3+ strikes traded within ±{half_window})", int(snaps.has_deribit.sum())),
        ("Kalshi strike vols compared (inside Deribit's fitted strike range)", len(s)),
    ], columns=["step", "count"])
    return dict(strikes=s, snaps=snaps, coverage=coverage)


def cached_snapshot_comparison(root: Path, hour_profile: pd.Series) -> dict:
    """`snapshot_comparison`, saved to data/derived/snapshot15*.parquet on first use (delete them to rebuild)."""
    d = Path(root) / "data" / "derived"
    files = {k: d / f"snapshot15_{k}.parquet" for k in ("strikes", "snaps", "coverage")}
    if all(f.exists() for f in files.values()):
        return {k: pd.read_parquet(f) for k, f in files.items()}
    r = snapshot_comparison(root, hour_profile)
    for k, f in files.items():
        r[k].to_parquet(f)
    return r
