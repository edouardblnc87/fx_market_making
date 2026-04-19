"""
K calibrator — fits the order-arrival intensity decay k from Phase 1 fills.

Model: λ(δ) = A · exp(−k · δ)   where δ is quoted distance from fair mid in bps.

Since every full reprice posts all 10 levels simultaneously, n_posted[i] is
roughly constant across levels. Fill counts therefore serve as a proportional
proxy for fill rate. Taking logs:

    log(n_fills[i]) = log(A·C) − k · δ_i

OLS on (δ_i, log(n_fills[i])) across levels gives k = −slope.
Run separately for buy and sell; report the average as cfg.k.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_DEFAULTS = {"k": 0.3, "k_buy": 0.3, "k_sell": 0.3}


class KCalibrator:
    """Calibrate order-arrival intensity decay k from Phase 1 MM fills."""

    def __init__(self, fill_df: pd.DataFrame) -> None:
        self._mm = fill_df[~fill_df["is_hedge"]].copy()

    def fit(self) -> dict:
        """Return {"k", "k_buy", "k_sell"}."""
        mm = self._mm
        if len(mm) < 20:
            return dict(_DEFAULTS)

        results: dict = {}
        for direction in ("buy", "sell"):
            df = mm[mm["direction"] == direction]
            k = self._fit_direction(df)
            results[f"k_{direction}"] = k

        k_avg = (results["k_buy"] + results["k_sell"]) / 2.0
        results["k"] = float(k_avg)
        return results

    # ------------------------------------------------------------------

    def _fit_direction(self, df: pd.DataFrame) -> float:
        if len(df) < 10:
            return _DEFAULTS["k"]

        grp = df.groupby("level")
        n_fills = grp.size()
        mean_dist_bps = grp.apply(
            lambda g: ((g["price"] - g["fair_mid"]).abs() * 10_000.0).mean()
        )

        # Keep only levels with enough fills for a stable estimate
        mask = n_fills >= 3
        if mask.sum() < 3:
            return _DEFAULTS["k"]

        y = np.log(n_fills[mask].values.astype(float))
        x = mean_dist_bps[mask].values
        X = np.column_stack([np.ones(len(x)), x])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        k = float(-coeffs[1])
        return float(np.clip(k, 0.01, 5.0))
