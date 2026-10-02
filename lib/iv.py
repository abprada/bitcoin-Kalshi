"""Black-Scholes implied volatility from Kalshi binary ("digital") prices (used by 03_iv_surface.py).

A Kalshi strike pays $1 if BTC settles above K, so its price is the risk-neutral probability
D(K) = P(S_T > K), and D(K) = -dC/dK for vanilla calls C. Integrating the ladder of binary prices
across strikes gives call prices above spot and put prices below spot, which invert for ordinary
(vanilla) implied vols. Rates are zero and the forward is spot: Kalshi pays interest on
collateral, and over at most a week discounting is below 0.1 cents.
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
    valid = np.isfinite(price) & (price >= model(a)) & (price <= model(b))
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


def integrate_ladder(K, D, F, tau, sigma_low, sigma_high) -> dict:
    """Call and put prices at every strike from a monotone ladder of binary prices.

    D is linear between strikes. Beyond the outermost strikes the distribution follows Black-Scholes
    (forward F) at that edge strike's vol, so the tails are closed-form: the call above the top strike
    and the put below the bottom strike. C - P + K is then the same number at every strike: the mean
    of the ladder's distribution, i.e. the forward implied by put-call parity (`forward`).
    """
    K, D = np.asarray(K, dtype=float), np.asarray(D, dtype=float)
    dK = np.diff(K)
    seg_up = 0.5 * (D[:-1] + D[1:]) * dK
    seg_dn = 0.5 * ((1 - D[:-1]) + (1 - D[1:])) * dK
    call = np.r_[np.cumsum(seg_up[::-1])[::-1], 0.0] + bs_call(F, K[-1], sigma_high, tau)
    put = np.r_[0.0, np.cumsum(seg_dn)] + bs_put(F, K[0], sigma_low, tau)
    return {"call": call, "put": put, "forward": float(np.mean(call - put + K))}


def snapshot_iv(K, mid, weight, F, tau) -> tuple[pd.DataFrame, dict]:
    """One IV per strike for one snapshot of a weekly's ladder.

    Monotone ladder -> edge vols for the tails -> integrate to calls and puts -> forward implied by
    put-call parity -> invert out-of-the-money calls (K >= forward) and puts (K < forward) against
    that forward. `spot` is only the starting point: Kalshi's ladder can be centred a few bps away
    from exchange spot, and forcing spot would distort cheap wing options. Also returns the direct
    binary IV per strike and snapshot-level diagnostics.
    """
    spot = float(F)
    o = np.argsort(K)
    K, mid, weight = (np.asarray(x, dtype=float)[o] for x in (K, mid, weight))
    D = isotonic_decreasing(mid, weight)
    fwd = spot
    for _ in range(2):  # the tails depend on the forward and the forward on the tails; converges at once
        dig_used = digital_iv(D, fwd, K, tau, sigma_ref=SIGMA_PRIOR)
        finite = np.flatnonzero(np.isfinite(dig_used))
        sig_lo, sig_hi = (dig_used[finite[0]], dig_used[finite[-1]]) if len(finite) else (np.nan, np.nan)
        ladder = integrate_ladder(K, D, fwd, tau, sig_lo, sig_hi)
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
        "sigma_low": sig_lo, "sigma_high": sig_hi,
        "atm_iv": atm,
        "strikes_per_sd": F * atm * np.sqrt(tau) / np.median(np.diff(K)) if len(K) > 1 else np.nan,
    }
    return strikes, snap
