"""Utility constants and notes for the intraday order-book time grid."""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# TIME GRID FOR INTRADAY STOCHASTIC SIMULATION
#
# Trading session: 9h00 → 17h00  =  8 h  =  28 800 s per day
#
# Default time step: dt = 50 ms = 0.05 s
#   → resolves individual order-book events (typical inter-arrival ~10–200 ms
#     for liquid names) without entering the HFT/latency regime (<1 ms).
#
# Memory at dt = 50 ms:
#   1 session  →  N =    576 000  steps  ≈  4.4 MB
#   1 month    →  N = 12 096 000  steps  ≈ 92.5 MB   (21 trading days)
#
# Implementation: np.linspace(0, T, N+1)  rather than np.arange(0, T, dt)
#   np.arange with a float step accumulates rounding errors over millions of
#   steps; linspace divides T exactly and guarantees t[-1] == T.
# ─────────────────────────────────────────────────────────────────────────────


