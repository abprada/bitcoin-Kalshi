"""Deribit option values at any time from full-day trades (00_fetch_deribit.py; used by 07 and 08).

Every Deribit trade carries Deribit's mark price and mark vol for its instrument at that instant, plus the index. So
trades in one expiry over a short window trace Deribit's own mark smile, even though no quotes are recorded:

1. Forward. Deribit's mark vol is quoted against its forward for that expiry, not the index. Given the mark price
   and the mark vol, Black-Scholes pins the forward each trade was marked against (`implied_forward`); its median
   over a window is that expiry's forward, kept as a basis to the index.
2. Smile. In a window, each strike's median mark vol is fitted as a quadratic in m = ln(K / F) / sqrt(tau)
   (log-moneyness per root year, so no vol enters the coordinate), using strikes within 2.5 ATM standard
   deviations of the forward and requiring at least three strikes with the forward between the lowest and highest.
   The constant is the at-the-money vol. Fitting from sums makes it one groupby for the whole year (`smile_grid`).
3. Value. An option at time t is priced with Black-Scholes at the fitted vol for its strike, the expiry's forward
   (index x (1 + basis)) and the time left (`value`).
4. Cost. Our data have no bid or ask. A taker buy fills at the ask and a taker sell at the bid, so the signed gap
   between the traded price and the mark, (price - mark) for buys and (mark - price) for sells, measures the
   half-spread actually paid (`effective_half_spread`). It is summarised in vol points, by time to expiry and by
   distance from the money.

Deribit options are inverse (paid in BTC); valued in USD (price in BTC x index) they are ordinary options, as in
lib/deribit.py. Rates are zero.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from lib import iv

YEAR_SECONDS = 365 * 86_400
TAKER_FEE = 0.0003          # options: 0.03% of the underlying per option ...
FEE_CAP = 0.125             # ... capped at 12.5% of the option's price
PERP_TAKER_FEE = 0.0005     # BTC-PERPETUAL taker fee, share of notional
SMILE_MAX_SD = 2.5          # strikes used in a smile fit: within this many ATM standard deviations
MIN_SMILE_STRIKES = 3


def load_options(root: Path, start=None, end=None) -> pd.DataFrame:
    """Full-day Deribit BTC option trades with time to expiry, decimal mark vol, USD prices and implied forward.

    Drops Deribit's placeholder vols (0, 999), expired rows and rows whose forward cannot be solved. Added columns:
    `tau_years`, `iv_d` (Deribit mark vol, decimal), `mark_usd`, `price_usd`, `fwd` (forward implied by the mark),
    `basis` = fwd / index - 1, `k` = ln(K / fwd), `m` = k / sqrt(tau), `vega` (USD per vol point, i.e. per 0.01).
    """
    files = sorted((Path(root) / "data" / "raw" / "deribit" / "options").glob("*.parquet"))
    d = pd.concat(map(pd.read_parquet, files), ignore_index=True)
    if start is not None:
        d = d[d.ts >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        d = d[d.ts < pd.Timestamp(end, tz="UTC")]
    d["tau_years"] = (d.expiry - d.ts).dt.total_seconds() / YEAR_SECONDS
    return add_forward(d)


def add_forward(d: pd.DataFrame) -> pd.DataFrame:
    """`load_options`' columns for any trades frame with ts, expiry, tau_years, strike, cp, iv (percent),
    mark_price_btc, price_btc and index_price (e.g. `deribit.load_trades` on the collaborator's CSV)."""
    d = d[(d.iv > 0) & (d.iv < 500) & (d.mark_price_btc > 0) & (d.tau_years > 0)].copy()
    d["iv_d"] = d.iv / 100
    d["mark_usd"] = d.mark_price_btc * d.index_price
    d["price_usd"] = d.price_btc * d.index_price
    d["fwd"] = implied_forward(d.mark_usd.to_numpy(), d.strike.to_numpy(), d.iv_d.to_numpy(), d.tau_years.to_numpy(),
                               (d.cp == "C").to_numpy(), d.index_price.to_numpy())
    d = d[np.isfinite(d.fwd)]
    d["basis"] = d.fwd / d.index_price - 1
    d["k"] = np.log(d.strike / d.fwd)
    d["m"] = d.k / np.sqrt(d.tau_years)
    d["vega"] = bs_vega(d.fwd.to_numpy(), d.strike.to_numpy(), d.iv_d.to_numpy(), d.tau_years.to_numpy()) / 100
    return d.reset_index(drop=True)


def bs_vega(F, K, sigma, tau):
    """dPrice / dsigma (USD per 1.00 of vol)."""
    F, K, sigma = (np.asarray(x, dtype=float) for x in (F, K, sigma))
    sd = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return F * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * np.sqrt(tau)


def bs_delta(F, K, sigma, tau, is_call):
    F, K, sigma = (np.asarray(x, dtype=float) for x in (F, K, sigma))
    sd = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return np.where(is_call, ndtr(d1), ndtr(d1) - 1)


def bs_price(F, K, sigma, tau, is_call):
    return np.where(is_call, iv.bs_call(F, K, sigma, tau), iv.bs_put(F, K, sigma, tau))


def implied_forward(price, K, sigma, tau, is_call, index, width: float = 0.05, iters: int = 60) -> np.ndarray:
    """Forward F with Black-Scholes(F, K, sigma, tau) = price, searched within index x (1 +/- width).

    Calls rise and puts fall with F, so bisection works for both. NaN when the price is outside the range the
    search interval allows (e.g. a deep option whose price barely depends on F).
    """
    price, K, sigma, tau, index = (np.asarray(x, dtype=float) for x in (price, K, sigma, tau, index))
    is_call = np.asarray(is_call)
    lo, hi = index * (1 - width), index * (1 + width)
    f = lambda F: bs_price(F, K, sigma, tau, is_call) - price
    flo, fhi = f(lo), f(hi)
    valid = np.isfinite(price) & (flo * fhi <= 0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        same = np.sign(fm) == np.sign(flo)
        lo, flo, hi = np.where(same, mid, lo), np.where(same, fm, flo), np.where(same, hi, mid)
    return np.where(valid, 0.5 * (lo + hi), np.nan)


def window_strikes(d: pd.DataFrame, freq: str = "1h", half_window: str = "30min",
                   centred: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The trades `smile_grid` fits, summarised: per (expiry, t, strike) the median `m` and mark vol `iv` and the
    trade count `n`; per (expiry, t) the median `basis` and `index`, `n_trades` and the median vol `ref`.
    Windows as in `smile_grid`; grid times at or after the expiry are dropped."""
    step, hw = pd.Timedelta(freq), pd.Timedelta(half_window)
    n_back = int(np.ceil(2 * hw / step))
    if centred:
        first = (d.ts - hw).dt.ceil(freq)
    else:
        first = d.ts.dt.ceil(freq)
    parts = []
    for j in range(n_back + 1):
        t = first + j * step
        keep = (t - d.ts <= hw) & (d.ts - t <= hw) if centred else (t - d.ts <= 2 * hw) & (t >= d.ts)
        parts.append(pd.DataFrame({"expiry": d.expiry[keep], "t": t[keep], "strike": d.strike[keep],
                                   "m": d.m[keep], "iv": d.iv_d[keep], "basis": d.basis[keep],
                                   "index": d.index_price[keep]}))
    x = pd.concat(parts, ignore_index=True)
    x = x[x.t < x.expiry]
    per_k = x.groupby(["expiry", "t", "strike"], sort=False).agg(m=("m", "median"), iv=("iv", "median"),
                                                                  n=("iv", "size")).reset_index()
    g = x.groupby(["expiry", "t"], sort=False).agg(basis=("basis", "median"), index=("index", "median"),
                                                     n_trades=("iv", "size"), ref=("iv", "median"))
    return per_k, g


def smile_grid(d: pd.DataFrame, freq: str = "1h", half_window: str = "30min", centred: bool = True) -> pd.DataFrame:
    """Deribit's smile for every expiry at every grid time, from the trades in a window around it.

    `centred`: trades within +/- `half_window` of the grid time (for valuing positions); otherwise the trades in
    the `2 x half_window` before it (for signals that must not see the future). Per strike the median mark vol and
    m (`window_strikes`); then a quadratic iv = a + b m + c m^2 by least squares over strikes within SMILE_MAX_SD ATM standard
    deviations (at least MIN_SMILE_STRIKES strikes, with the forward inside their range). Returns one row per (expiry, t): `atm_iv` (= a), `b`, `c`, `basis` (median),
    `index` (median), `n_strikes`, `n_trades`, `m_lo`, `m_hi` (range of strikes used), `tau_years` at t.
    """
    per_k, g = window_strikes(d, freq, half_window, centred)
    per_k = per_k.join(g.ref, on=["expiry", "t"])
    per_k = per_k[np.abs(per_k.m) <= SMILE_MAX_SD * per_k.ref]
    m, y = per_k.m.to_numpy(), per_k.iv.to_numpy()
    sums = pd.DataFrame({"expiry": per_k.expiry, "t": per_k.t, "s0": 1.0, "s1": m, "s2": m**2, "s3": m**3, "s4": m**4,
                         "y0": y, "y1": y * m, "y2": y * m**2, "m_lo": m, "m_hi": m})
    s = sums.groupby(["expiry", "t"], sort=False).agg(
        **{c: (c, "sum") for c in ["s0", "s1", "s2", "s3", "s4", "y0", "y1", "y2"]},
        m_lo=("m_lo", "min"), m_hi=("m_hi", "max"))
    s = s[(s.s0 >= MIN_SMILE_STRIKES) & (s.m_lo < 0) & (s.m_hi > 0)]
    A = np.stack([np.stack([s.s0, s.s1, s.s2], -1), np.stack([s.s1, s.s2, s.s3], -1),
                  np.stack([s.s2, s.s3, s.s4], -1)], 1)
    rhs = np.stack([s.y0, s.y1, s.y2], -1)
    ok = np.abs(np.linalg.det(A)) > 1e-12 * np.maximum(1, np.abs(A).max(axis=(1, 2)) ** 3)
    coef = np.full((len(s), 3), np.nan)
    coef[ok] = np.linalg.solve(A[ok], rhs[ok][..., None])[..., 0]
    out = pd.DataFrame({"atm_iv": coef[:, 0], "b": coef[:, 1], "c": coef[:, 2], "n_strikes": s.s0.astype(int),
                        "m_lo": s.m_lo, "m_hi": s.m_hi}, index=s.index).join(g[["basis", "index", "n_trades"]])
    out = out.reset_index()
    out["tau_years"] = (out.expiry - out.t).dt.total_seconds() / YEAR_SECONDS
    out = out[(out.atm_iv > 0.05) & (out.atm_iv < 5)]
    return out.sort_values(["expiry", "t"]).reset_index(drop=True)


def smile_iv(row, K, F):
    """Fitted vol at strike(s) K for one `smile_grid` row, with forward F; flat beyond the fitted strike range."""
    m = np.clip(np.log(np.asarray(K, dtype=float) / F) / np.sqrt(row["tau_years"]), row["m_lo"], row["m_hi"])
    return row["atm_iv"] + row["b"] * m + row["c"] * m**2


def value(row, K, is_call, index: float, tau: float):
    """USD value and delta of options on one expiry at strikes K, from a `smile_grid` row (vol and basis) and the
    index and time left at the valuation instant. Returns (price, delta, vol, vega per vol point)."""
    F = index * (1 + row["basis"])
    sig = smile_iv(row, K, F)
    p = bs_price(F, K, sig, tau, is_call)
    return p, bs_delta(F, K, sig, tau, is_call), sig, bs_vega(F, K, sig, tau) / 100


def effective_half_spread(d: pd.DataFrame) -> pd.DataFrame:
    """Per trade: the half-spread the taker paid, in USD and in vol points (USD / vega), and a size flag.

    Positive when the taker paid up (bought above the mark or sold below it).
    """
    sign = np.where(d.direction == "buy", 1.0, -1.0)
    hs = sign * (d.price_usd - d.mark_usd)
    return d.assign(hs_usd=hs, hs_vol=hs / d.vega, hs_frac=hs / d.mark_usd)


def option_fee(index, price_usd):
    """Deribit taker fee per option, USD: 0.03% of the index, capped at 12.5% of the option's price."""
    return np.minimum(TAKER_FEE * np.asarray(index, dtype=float), FEE_CAP * np.asarray(price_usd, dtype=float))


def delivery_prices(root: Path, trades: pd.DataFrame) -> pd.Series:
    """Deribit's 08:00 UTC delivery price by date: data/deribit_delivery_prices.csv (Deribit's own, from Dec 2025),
    filled before that with the average of the index printed on trades between 07:30 and 08:00 UTC (Deribit
    settles on a 30-minute average of its index)."""
    x = trades.set_index("ts").index_price
    tw = x[(x.index.hour == 7) & (x.index.minute >= 30)]
    twap = tw.groupby(tw.index.date).mean()
    f = pd.read_csv(Path(root) / "data" / "deribit_delivery_prices.csv")
    official = pd.Series(f.delivery_price.to_numpy(dtype=float), index=pd.to_datetime(f.date).dt.date)
    return official.combine_first(twap).sort_index()
