# Kalshi BTC weekly: how we build the implied-vol surface

**Data.** 51 weekly Kalshi BTC contracts (Oct 2025 – Sep 2026), 1-minute quotes, about 50 strikes each. Each strike pays $1 if BTC settles above K on Friday at 5pm ET, so its price is the market's probability of that.

**Goal.** Turn those probabilities into a Black-Scholes implied vol per strike, then a surface over strike and time to expiry.

## Method 1: invert each strike directly
- The Black-Scholes price of a binary is N(d₂); solve it for σ from the mid.
- Simple, but it breaks near the money. The price barely moves with σ there (1¢ of noise ≈ 17 vol points), there's often no solution at all (29% of strikes within 0.5% of the money), and it ignores the skew.
- We keep it only as a comparison.

## Method 2: integrate, then invert (what we use)
- A binary is the slope of the call price across strikes, so adding up the binary prices across strikes gives call and put prices.
- Steps:
  1. Clean the ladder so prices fall as the strike rises.
  2. Add a tail beyond each outermost strike: Black-Scholes at that edge strike's own Kalshi vol, kept within the bounds set by Kalshi's other listed strikes (bids from below, asks from above). No BTC vol enters.
  3. Integrate to get calls above the forward and puts below it.
  4. Take the forward from put-call parity.
  5. Invert each option with standard Black-Scholes.
- Result: one ordinary implied vol per strike. It's stable at the money because quote noise averages out across strikes, and it's directly comparable with option-market vols (Deribit, IBIT).

## What we see
32k smiles (one every 15 minutes), 82% passing quality checks:
- ATM vol is about 43%;
- vol rises toward expiry, from about 40% a week out to about 50% in the final hours;
- smiles have expensive wings, with puts richer than calls.

## Caveats
- Within about 6 hours of expiry the $500 strike spacing is too coarse to trust, and the first weeks (Oct 2025) are thin.
- About 5–6% of the probability lies beyond the outermost usable strikes. Its location is a modelling choice (Black-Scholes shape), constrained by Kalshi's own outer quotes. Dropping the tails instead would bias vols low by about 1.5–2 points at the money and 5–7 in the wings.

## Where
`03_iv_surface.py` (method in `lib/iv.py`), with figures in `outputs/`. The rotatable 3D surface is `outputs/iv_surface_3d.html`.
