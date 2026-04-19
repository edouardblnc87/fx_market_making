"""
Ladder calibrator — fits beta and Q_base from Phase 1 fill volumes.

Model: quote size at level i = Q_base · exp(−beta · i)

The ladder shape should match where client volume actually lands. Since every
reprice posts all levels, differences in fill volume across levels are driven
by client arrival-rate decay — the same exponential structure. Taking logs:

    log(V_i) = log(Q_base) − beta · i

OLS on (i, log(V_i)) gives beta = −slope, Q_base = exp(intercept).

Note: beta ≈ k · tick_bps in a consistent model (arrival decay matches size
decay). If they diverge it signals asymmetric client flow or persistent skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_DEFAULTS = {"beta": 0.3, "Q_base": 100_000.0}


class LadderCalibrator:
    """Calibrate ladder shape parameters from Phase 1 MM fill volumes."""

    def __init__(self, fill_df: pd.DataFrame) -> None:
        self._mm = fill_df[~fill_df["is_hedge"]].copy()

    def fit(self) -> dict:
        """Return {"beta", "Q_base"}."""
        mm = self._mm
        if len(mm) < 20:
            return dict(_DEFAULTS)

        vol_by_level = mm.groupby("level")["size"].sum()
        if len(vol_by_level) < 3:
            return dict(_DEFAULTS)

        levels   = vol_by_level.index.values.astype(float)
        log_vol  = np.log(vol_by_level.values.clip(min=1.0).astype(float))

        X = np.column_stack([np.ones(len(levels)), levels])
        coeffs, _, _, _ = np.linalg.lstsq(X, log_vol, rcond=None)

        Q_base = float(np.clip(np.exp(coeffs[0]), 1_000.0, 1_000_000.0))
        beta   = float(np.clip(-coeffs[1], 0.01, 2.0))

        return {"beta": beta, "Q_base": Q_base}
