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
KALSHI_FEE_RATE = 0.07      # Kalshi taker fee per contract = rate x P x (1 - P), rounded up to the cent (check the
                            # current fee schedule before relying on it)
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


def kalshi_fee(price, rate: float = KALSHI_FEE_RATE) -> np.ndarray:
    """Kalshi taker fee per contract in dollars: rate x P x (1 - P), rounded up to the cent."""
    p = np.asarray(price, dtype=float)
    return np.ceil(np.round(100 * rate * p * (1 - p), 9)) / 100


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
    data, clean, derived = root / "data", root / "data" / "clean", root / "data" / "derived"

    # Deribit: afternoon trades in Friday expiries (the only ones whose date matches a Kalshi weekly)
    trades = deribit.load_trades(data / "deribit_btc_options.csv.gz")
    trades = trades[(trades.window == "us_afternoon") & (trades.expiry.dt.dayofweek == 4)
                    & trades.iv_mark.notna() & (trades.tau_years > 0)]
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
