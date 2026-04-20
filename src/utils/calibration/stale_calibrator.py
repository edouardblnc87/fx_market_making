"""
Stale calibrator — estimates the optimal stale_s threshold from Phase 1.

If fills at level 1 follow a Poisson process with rate λ_1 (fills/second),
time-to-fill is exponential with mean 1/λ_1. An order resting beyond the
75th percentile of that distribution has only a 25% chance of ever filling
before the next price move — a good threshold for aggressive repricing.

    P(T > t) = exp(−λ_1 · t) = 0.25  →  t_75 = log(4) / λ_1

λ_1 is estimated as (total fills at level 1) / (total simulation time).
Level 1 is used because it is the most liquid and gives the tightest
(most conservative) estimate — deep levels have lower λ so their t_75
would be larger, meaning level-1 t_75 is a lower bound on stale_s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_DEFAULT = {"stale_s": 3.0}


class StaleCalibrator:
    """Estimate stale_s from the empirical fill rate at level 1."""

    def __init__(self, fill_df: pd.DataFrame, step_log: pd.DataFrame) -> None:
        """Store MM fills and step log for stale threshold estimation."""
        self._mm       = fill_df[~fill_df["is_hedge"]].copy()
        self._step_log = step_log

    def fit(self) -> dict:
        """Return {"stale_s"}."""
        mm = self._mm
        if len(mm) < 20 or self._step_log.empty:
            return dict(_DEFAULT)

        total_t = float(self._step_log["t"].iloc[-1])
        if total_t <= 0:
            return dict(_DEFAULT)

        n_fills_1 = int((mm["level"] == 1).sum())
        if n_fills_1 < 10:
            return dict(_DEFAULT)

        lambda_1 = n_fills_1 / total_t          # fills per second at level 1
        stale_s  = float(np.log(4.0) / lambda_1)  # 75th percentile of exp TTF
        stale_s  = float(np.clip(stale_s, 0.5, 60.0))

        return {"stale_s": stale_s}
