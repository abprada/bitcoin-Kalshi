"""Putting Kalshi and Deribit in the same units (used by 05_kalshi_vs_deribit.py).

Deribit's vols are read as a smile in log-moneyness ln(K / index), from the mark prices of trades near one moment.
Interpolating that smile to a Kalshi strike gives a Deribit vol there (the vol comparison). Pricing a tight call
spread on the same smile gives a Deribit price for Kalshi's binary (the price comparison, handout Q1c/Q2b):
D(K) = -dC/dK ≈ [C(K - h) - C(K + h)] / 2h. The spread uses each leg's own smile vol, so the skew is included.
"""

import numpy as np
import pandas as pd

from lib import iv

SPREAD_HALF_WIDTH = 250.0   # dollars: the call spread runs from K - 250 to K + 250
KALSHI_FEE_RATE = 0.07      # Kalshi taker fee per contract = rate x P x (1 - P), rounded up to the cent (check the
                            # current fee schedule before relying on it)


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


def kalshi_fee(price, rate: float = KALSHI_FEE_RATE) -> np.ndarray:
    """Kalshi taker fee per contract in dollars: rate x P x (1 - P), rounded up to the cent."""
    p = np.asarray(price, dtype=float)
    return np.ceil(np.round(100 * rate * p * (1 - p), 9)) / 100
