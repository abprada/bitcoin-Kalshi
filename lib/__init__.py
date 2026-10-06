"""Shared code for the numbered pipeline steps.

fetch_utils  HTTP client, time windows and parquet writing (00_fetch_*.py)
cleaning     cleaning rules for quotes and spot (02_clean_data.py)
iv           Black-Scholes implied vol from binary prices (03_iv_surface.py)
deribit      Deribit option trades: loading, flags, ATM vols, hour-of-week variance (04, 05)
compare      Kalshi vs Deribit in the same units: smile interpolation, call-spread binaries, fees (05)
"""
