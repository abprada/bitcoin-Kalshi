"""Black-Scholes implied volatility from Kalshi binary ("digital") prices (used by 03_iv_surface.py).

A Kalshi strike pays $1 if BTC settles above K, so its price is the risk-neutral probability
D(K) = P(S_T > K), and D(K) = -dC/dK for vanilla calls C. Integrating the ladder of binary prices
across Kalshi's strikes gives call and put prices, which invert for ordinary (vanilla) implied vols.
Beyond the outermost usable strikes the tails are anchored to Kalshi too (see `snapshot_iv`): only
Kalshi prices enter, never BTC volatility. Rates are zero: Kalshi pays interest on positions, so
discounting cancels.
"""

import numpy as np
import pandas as pd
from scipy.special import ndtr

SIGMA_LO, SIGMA_HI = 0.01, 5.0   # implied vol search range
SIGMA_PRIOR = 0.5                # reference vol for picking the branch of the binary inversion
_ITERS = 100                     # bisection steps: brackets shrink far below float precision


def bs_call(F, K, sigma, tau):
    F, K, sigma = (np.asarray(x, dtype=float) for x in (F, K, sigma))
    sd = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return F * ndtr(d1) - K * ndtr(d1 - sd)


def bs_put(F, K, sigma, tau):
    return bs_call(F, K, sigma, tau) - (np.asarray(F, dtype=float) - np.asarray(K, dtype=float))


def implied_vol(price, F, K, tau, is_call) -> np.ndarray:
    """Vanilla Black-Scholes implied vol by bisection; NaN outside the attainable price range."""
    price, K, is_call = np.asarray(price, dtype=float), np.asarray(K, dtype=float), np.asarray(is_call)
    model = lambda s: np.where(is_call, bs_call(F, K, s, tau), bs_put(F, K, s, tau))
    a, b = np.full(price.shape, SIGMA_LO), np.full(price.shape, SIGMA_HI)
    intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    valid = np.isfinite(price) & (price > intrinsic) & (price >= model(a)) & (price <= model(b))
    for _ in range(_ITERS):
        m = 0.5 * (a + b)
        up = model(m) < price          # price increases with sigma
        a, b = np.where(up, m, a), np.where(up, b, m)
    return np.where(valid, 0.5 * (a + b), np.nan)


def digital_iv(D, F, K, tau, sigma_ref=None) -> np.ndarray:
    """Direct inversion of D = N(d2) for sigma; NaN when no root exists.

    With spot at or above the strike the price falls monotonically in sigma (one root). With spot
    below, it rises to a maximum at sigma* = sqrt(2 |ln(S/K)| / tau) and then falls, so there are two
    roots, and prices above the maximum have none. The root returned is the one on the same side of
    sigma* as `sigma_ref` (far from the money sigma* is large and that is the lower root; very near
    the money sigma* is tiny and the realistic vol is the upper root). Without `sigma_ref`, the lower
    root.
    """
    D, K = np.asarray(D, dtype=float), np.asarray(K, dtype=float)
    m = np.log(F / K)
    g = lambda s: ndtr(m / (s * np.sqrt(tau)) - 0.5 * s * np.sqrt(tau)) - D
    s_star = np.sqrt(2 * np.abs(m) / tau)
    upper = (m < 0) & (sigma_ref is not None) & (s_star < (np.inf if sigma_ref is None else sigma_ref))
    a = np.where(upper, np.maximum(s_star, SIGMA_LO), SIGMA_LO)
    b = np.where((m < 0) & ~upper, np.minimum(SIGMA_HI, s_star), SIGMA_HI)
    ga, gb = g(a), g(b)
    valid = (D > 0) & (D < 1) & (b > a) & (ga * gb <= 0)
    for _ in range(_ITERS):
        mid = 0.5 * (a + b)
        gm = g(mid)
        same = np.sign(gm) == np.sign(ga)
        a, ga, b = np.where(same, mid, a), np.where(same, gm, ga), np.where(same, b, mid)
    return np.where(valid, 0.5 * (a + b), np.nan)


def isotonic_decreasing(y, w=None) -> np.ndarray:
    """Weighted least-squares non-increasing fit (pool-adjacent-violators)."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    vals, wts, cnts = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnts.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            wt = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wt
            wts[-2], cnts[-2] = wt, cnts[-2] + cnts[-1]
            vals.pop(); wts.pop(); cnts.pop()
    return np.repeat(vals, cnts)


def integrate_ladder(K, D, tail_up=0.0, tail_dn=0.0) -> dict:
    """Call and put prices at every strike from a monotone ladder of binary prices.

    D is linear between strikes. `tail_up` is the value of the call at the top strike (the integral of D
    above it) and `tail_dn` the value of the put at the bottom strike (the integral of 1 - D below it);
    with the defaults nothing lies beyond the outermost strikes. C - P + K is the same number at every
    strike: the mean of the distribution, i.e. the forward implied by put-call parity (`forward`).
    """
    K, D = np.asarray(K, dtype=float), np.asarray(D, dtype=float)
    dK = np.diff(K)
    seg_up = 0.5 * (D[:-1] + D[1:]) * dK
    seg_dn = 0.5 * ((1 - D[:-1]) + (1 - D[1:])) * dK
    call = np.r_[np.cumsum(seg_up[::-1])[::-1], 0.0] + tail_up
    put = np.r_[0.0, np.cumsum(seg_dn)] + tail_dn
    return {"call": call, "put": put, "forward": float(np.mean(call - put + K))}


def _outward_bounds(dist, lower, upper, edge) -> tuple[float, float]:
    """(lower, upper) bounds on the integral, from the edge strike to the last listed strike, of a function
    that only falls moving away from the ladder. `dist`: distances of the outer strikes from the edge
    (increasing); `lower`/`upper`: pointwise bounds there (NaN = none); `edge`: its value at the edge."""
    ub = np.minimum.accumulate(np.r_[edge, np.where(np.isnan(upper), 1.0, upper)])
    lb = np.maximum.accumulate(np.where(np.isnan(lower), 0.0, lower)[::-1])[::-1]
    lb = np.minimum(lb, ub[1:])
    dK = np.diff(np.r_[0.0, dist])
    return float((lb * dK).sum()), float((ub[:-1] * dK).sum())


def tail_bounds(K, D, K_out, bid_out, ask_out) -> dict:
    """Model-free bounds on the tail values from Kalshi's other listed strikes beyond the ladder.

    D only falls as the strike rises, so above the ladder D(x) is at most the lowest ask at or below x
    (and the top of the ladder) and at least the highest bid at or above x; below the ladder the same
    holds for 1 - D with 1 - bid and 1 - ask. Integrating these step bounds gives (lower, upper) for the
    call tail above the top strike (`up`) and the put tail below the bottom strike (`dn`), up to the
    last strike Kalshi lists on each side (`K_up`, `K_dn`). Beyond those, Kalshi's quotes say nothing.
    """
    K, D = np.asarray(K, dtype=float), np.asarray(D, dtype=float)
    K_out, bid_out, ask_out = (np.asarray(x, dtype=float) for x in (K_out, bid_out, ask_out))
    up = np.argsort(K_out)
    up = up[K_out[up] > K[-1]]
    dn = np.argsort(-K_out)
    dn = dn[K_out[dn] < K[0]]
    return {
        "up": _outward_bounds(K_out[up] - K[-1], bid_out[up], ask_out[up], D[-1]),
        "dn": _outward_bounds(K[0] - K_out[dn], 1 - ask_out[dn], 1 - bid_out[dn], 1 - D[0]),
        "K_up": float(K_out[up][-1]) if len(up) else float(K[-1]),
        "K_dn": float(K_out[dn][-1]) if len(dn) else float(K[0]),
    }


def snapshot_iv(K, mid, weight, F, tau, K_out=(), bid_out=(), ask_out=()) -> tuple[pd.DataFrame, dict]:
    """One IV per strike for one snapshot of a weekly's ladder.

    Monotone ladder -> tails -> integrate to calls and puts -> forward implied by put-call parity ->
    invert out-of-the-money calls (K >= forward) and puts (K < forward) against that forward.

    Tails: beyond each outermost strike, Black-Scholes at that edge strike's own binary-implied vol,
    clipped to the bounds that Kalshi's other listed strikes (`K_out`, `bid_out`, `ask_out`: quotes
    too wide or too cheap for the ladder) put on it up to the last strike Kalshi lists; beyond that,
    the Black-Scholes tail alone (negligible). Only Kalshi prices enter. `F` (exchange spot) is only
    the starting point and the reference for the reported basis: Kalshi's ladder can be centred a
    few bps away from exchange spot, and forcing spot would distort cheap wing options. Also returns
    the direct binary IV per strike and snapshot-level diagnostics.
    """
    spot = float(F)
    o = np.argsort(K)
    K, mid, weight = (np.asarray(x, dtype=float)[o] for x in (K, mid, weight))
    D = isotonic_decreasing(mid, weight)
    bounds = tail_bounds(K, D, K_out, bid_out, ask_out)
    fwd = spot
    for _ in range(2):  # the tails depend on the forward and the forward on the tails; converges at once
        dig_used = digital_iv(D, fwd, K, tau, sigma_ref=SIGMA_PRIOR)
        finite = np.flatnonzero(np.isfinite(dig_used))
        sig_lo, sig_hi = (dig_used[finite[0]], dig_used[finite[-1]]) if len(finite) else (np.nan, np.nan)
        # model tail between the edge and the last listed strike, kept within Kalshi's bounds, plus beyond
        raw_up = float(bs_call(fwd, K[-1], sig_hi, tau) - bs_call(fwd, bounds["K_up"], sig_hi, tau))
        raw_dn = float(bs_put(fwd, K[0], sig_lo, tau) - bs_put(fwd, bounds["K_dn"], sig_lo, tau))
        cut_up, cut_dn = np.clip(raw_up, *bounds["up"]), np.clip(raw_dn, *bounds["dn"])
        tail_up = cut_up + float(bs_call(fwd, bounds["K_up"], sig_hi, tau))
        tail_dn = cut_dn + float(bs_put(fwd, bounds["K_dn"], sig_lo, tau))
        ladder = integrate_ladder(K, D, tail_up, tail_dn)
        fwd = ladder["forward"]
    F = fwd
    is_call = K >= F
    price = np.where(is_call, ladder["call"], ladder["put"])
    vol = implied_vol(price, F, K, tau, is_call)

    k, ok = np.log(K / F), np.isfinite(vol)
    atm = float(np.interp(0.0, k[ok], vol[ok])) if ok.sum() >= 2 and k[ok][0] <= 0 <= k[ok][-1] else np.nan
    strikes = pd.DataFrame({
        "strike": K, "mid": mid, "D_used": D,
        "digital_iv": digital_iv(mid, F, K, tau, sigma_ref=atm if np.isfinite(atm) else SIGMA_PRIOR),
        "option": np.where(is_call, "call", "put"), "price": price, "iv": vol,
    })
    snap = {
        "n_strikes": len(K),
        "monotone_fixes": int((np.abs(D - mid) > 1e-12).sum()),
        "forward": F,
        "forward_basis_bps": (F / spot - 1) * 1e4,
        "prob_outside": float((1 - D[0]) + D[-1]),
        "sigma_low": sig_lo, "sigma_high": sig_hi,
        "tails_clipped": int(abs(cut_up - raw_up) > 1e-6) + int(abs(cut_dn - raw_dn) > 1e-6),
        "atm_iv": atm,
        "strikes_per_sd": F * atm * np.sqrt(tau) / np.median(np.diff(K)) if len(K) > 1 else np.nan,
    }
    return strikes, snap


def winsorize_by_group(x, groups, lower=0.01, upper=0.99) -> pd.Series:
    """Clip x to its own [lower, upper] percentiles within each group; NaNs stay NaN and are ignored."""
    x = pd.Series(x, dtype=float)
    g = pd.Series(groups, index=x.index)
    by = x.groupby(g, observed=True)
    return x.clip(by.transform(lambda s: s.quantile(lower)), by.transform(lambda s: s.quantile(upper)))


def pooled_surface(df: pd.DataFrame, h_edges, m_edges, min_obs: int = 200, min_snaps: int = 20) -> pd.DataFrame:
    """Shape-normalized surface: median(iv / snapshot ATM vol) per cell x median ATM vol of that time bin.

    The horizontal coordinate sd_from_spot = (K / spot - 1) / (snapshot ATM IV * sqrt(tau)).
    Pooling raw vols mixes volatility regimes: near expiry, far strikes only have usable quotes on very
    volatile days, so raw medians spike at the edges. Pooling each snapshot's smile relative to its own
    ATM vol and rescaling by the typical ATM vol of the time bin removes that selection effect. Cells
    with fewer than `min_obs` strike-snapshots or `min_snaps` distinct snapshots are left empty.
    `df` needs snap, hours_to_expiry, sd_from_spot, atm_iv and iv_surface. Returns time bins x ATM standard-deviation bins.
    """
    d = df[df.iv_surface.notna()].assign(
        h=lambda x: pd.cut(x.hours_to_expiry, h_edges, include_lowest=True),
        m=lambda x: pd.cut(x.sd_from_spot, m_edges, include_lowest=True),
        ratio=lambda x: x.iv_surface / x.atm_iv,
    ).dropna(subset=["h", "m"])
    atm = d.drop_duplicates("snap").groupby("h", observed=False).atm_iv.median()
    g = d.groupby(["h", "m"], observed=False).agg(ratio=("ratio", "median"), n=("ratio", "size"), snaps=("snap", "nunique"))
    vals = g.ratio * g.index.get_level_values("h").map(atm).to_numpy(dtype=float)
    return vals.where((g.n >= min_obs) & (g.snaps >= min_snaps)).unstack("m")
