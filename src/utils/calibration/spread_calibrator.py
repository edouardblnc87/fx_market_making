"""
Spread calibrator — OLS regression on realized effective spreads.

Estimates alpha_spread (inventory spread weight) and alpha_imbalance
(order-flow imbalance tilt) from Phase 1 fills.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_DEFAULTS = {"alpha_spread": 0.5, "alpha_imbalance": 0.0002}


class SpreadCalibrator:
    """Calibrate spread components from Phase 1 data."""

    def __init__(
        self,
        fill_df: pd.DataFrame,
        step_log: pd.DataFrame,
        capital_K: float,
    ) -> None:
        """Prepare MM fills and step log for OLS regression on effective spreads."""
        self._capital_K = capital_K

        mm = fill_df[~fill_df["is_hedge"]].copy()
        mm = mm.sort_values("step").reset_index(drop=True)
        self._mm = mm
        self._step_log = step_log

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> dict:
        """Return {"alpha_spread", "alpha_imbalance"}."""
        mm = self._mm
        if len(mm) < 20:
            return dict(_DEFAULTS)

        # Effective spread for each fill
        effective_spread = (mm["price"] - mm["fair_mid"]).abs().values

        # Inventory ratio squared
        inv_ratio = mm["inventory_after"].values / self._capital_K
        inv_ratio_sq = inv_ratio ** 2

        # Rolling order-flow imbalance from fill directions (window=50)
        is_buy = (mm["direction"] == "buy").astype(float).values
        window = min(50, len(mm))
        imbalance = np.empty(len(mm))
        for i in range(len(mm)):
            start = max(0, i - window + 1)
            chunk = is_buy[start : i + 1]
            imbalance[i] = 2.0 * chunk.mean() - 1.0  # in [-1, 1]

        # OLS: effective_spread = beta0 + beta1 * inv_ratio^2 + beta2 * imbalance
        X = np.column_stack([
            np.ones(len(mm)),
            inv_ratio_sq,
            imbalance,
        ])
        y = effective_spread

        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        beta0, beta1, beta2 = coeffs

        # Map back to config parameters
        if beta0 > 0:
            alpha_spread = beta1 / beta0
        else:
            alpha_spread = _DEFAULTS["alpha_spread"]

        fair_mid_mean = float(mm["fair_mid"].mean())
        if fair_mid_mean > 0:
            alpha_imbalance = abs(beta2) / fair_mid_mean
        else:
            alpha_imbalance = _DEFAULTS["alpha_imbalance"]

        # Clamp to reasonable range
        alpha_spread = float(np.clip(alpha_spread, 0.0, 5.0))
        alpha_imbalance = float(np.clip(alpha_imbalance, 0.0, 0.01))

        return {
            "alpha_spread": alpha_spread,
            "alpha_imbalance": alpha_imbalance,
        }
