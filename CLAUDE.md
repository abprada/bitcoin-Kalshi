# Prediction Markets as Macro Data (Kalshi / Polymarket)

## Research questions
1. **Calibration:** Are Kalshi prices on macro events rational? Compare Kalshi vs. WIRP (and SPF, option-implied) probabilities for rate hike/cut states over t ∈ [-60, -1] days, scored by Brier score / cross-entropy. Hypothesis: Kalshi overprices the tails (favorite–longshot bias).
2. **Price discovery:** How fast do Kalshi prices incorporate CPI, NFP, and FOMC announcements? Event study of Kalshi jumps around releases vs. high-frequency fed funds futures moves; Kalshi vs. Polymarket lead–lag.
3. **Forecasting:** Use Kalshi data to predict macro quantities (unemployment, GDP growth, CPI), possibly with ML models (e.g. economic transformers).

## Conventions
- Use **marimo** notebooks (`.py`), not Jupyter (`.ipynb`).
