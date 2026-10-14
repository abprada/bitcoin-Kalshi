"""Shared code for the numbered pipeline steps.

fetch_utils  HTTP client, time windows and parquet writing (00_fetch_*.py)
cleaning     cleaning rules for quotes and spot (02_clean_data.py)
iv           Black-Scholes implied vol from binary prices (03_iv_surface.py)
deribit      Deribit option trades: loading, flags, ATM vols, hour-of-week variance (04, 05)
compare      Kalshi vs Deribit in the same units: smile interpolation, call-spread binaries, fees (05)
pairs        Kalshi vs Deribit vols paired trade by trade (05, 06)
deribit_marks     Deribit's hourly mark smile, forward, option values and half-spreads from full-day trades (08-10)
kalshi_hourly     Kalshi smiles from the collaborator's hourly snapshots, and the binary-strip straddle (08, 09)
vol_spread        Kalshi vs Deribit hourly panel, lead-lag, and the long/short vol-spread backtest (08, 09)
calendar_straddle Deribit calendar straddle backtest: long ~7d / short ~1d, delta-hedged (08, 10)
dislocation       Kalshi vs Deribit dislocations and market-vs-market BTC forecasts (08, 11)
strategy_ab       the two dislocation strategies as portfolios, execution variants, out of sample (08, 12)
binary_hedge      hedging a long Kalshi binary with a short call spread, frictionless (08, 13)
"""
