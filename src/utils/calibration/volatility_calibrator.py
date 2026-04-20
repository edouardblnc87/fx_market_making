"""
Volatility calibrator — EWMA span selection + GARCH(1,1) MLE.

Finds the optimal vol_window for the Quoter's EWMA estimator and optionally
fits a GARCH(1,1) model, comparing via AIC/BIC.

References: Bollerslev (1986); RiskMetrics (1996)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..stock_simulation.config import TRADING_SECONDS_PER_YEAR


class VolatilityCalibrator:
    """Calibrate volatility model from Phase 1 market B mid-prices."""

    def __init__(self, mid_prices: np.ndarray, dt: float) -> None:
        """Store mid-prices and time step; pre-compute log-returns for all fit methods."""
        self._prices = mid_prices
        self._dt = dt
        self._dt_frac = dt / TRADING_SECONDS_PER_YEAR
        self._log_rets = np.diff(np.log(mid_prices))

    # ------------------------------------------------------------------
    # EWMA
    # ------------------------------------------------------------------

    def fit_ewma(self) -> dict:
        """
        Grid-search over EWMA spans, pick the one that minimises
        out-of-sample RMSE of variance prediction.

        Returns {"vol_window_s": float, "rmse": float}  (seconds, not steps)
        Result is cached after first call.
        """
        if hasattr(self, "_ewma_cache"):
            return self._ewma_cache
        r = self._log_rets
        n = len(r)
        split = int(n * 0.7)
        if split < 500:
            return {"vol_window_s": 6000 * self._dt, "rmse": float("nan")}

        r_train = r[:split]
        r_test = r[split:]
        r2_test = r_test ** 2

        candidates = [w for w in [500, 1000, 2000, 4000, 6000, 10000] if w < split]
        if not candidates:
            return {"vol_window_s": 6000 * self._dt, "rmse": float("nan")}

        best_w, best_rmse = 6000, float("inf")

        for w in candidates:
            ewma_var = pd.Series(r_train ** 2).ewm(span=w).mean().values
            last_var = ewma_var[-1]
            alpha_ewm = 2.0 / (w + 1)
            forecast = np.empty(len(r_test))
            v = last_var
            for i in range(len(r_test)):
                forecast[i] = v
                v = alpha_ewm * r2_test[i] + (1 - alpha_ewm) * v

            rmse = float(np.sqrt(np.mean((forecast - r2_test) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_w = w

        self._ewma_cache = {"vol_window_s": best_w * self._dt, "rmse": best_rmse}
        return self._ewma_cache

    # ------------------------------------------------------------------
    # GARCH(1,1)
    # ------------------------------------------------------------------

    def fit_garch(self) -> dict | None:
        """
        MLE for GARCH(1,1): h[t+1] = omega + alpha * r[t]^2 + beta * h[t].

        Returns {"alpha", "beta", "omega", "persistence", "aic"} or None on failure.
        """
        r = self._log_rets
        n = len(r)
        if n < 2000:
            return None

        # Subsample to 50K max — GARCH has a pure-Python loop and 51M
        # rows is overkill; estimates converge well before that.
        _MAX_GARCH = 50_000
        if n > _MAX_GARCH:
            step = n // _MAX_GARCH
            r = r[::step][:_MAX_GARCH]
            n = len(r)

        r2 = r ** 2
        var_r = float(r2.mean())

        def neg_log_lik(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e15
            h = np.empty(n)
            h[0] = var_r
            for t in range(1, n):
                h[t] = omega + alpha * r2[t - 1] + beta * h[t - 1]
                if h[t] < 1e-20:
                    h[t] = 1e-20
            # Gaussian log-likelihood (drop constant)
            ll = -0.5 * np.sum(np.log(h) + r2 / h)
            return -ll

        a0, b0 = 0.05, 0.94
        o0 = var_r * (1.0 - a0 - b0)

        res = minimize(
            neg_log_lik,
            x0=[o0, a0, b0],
            method="L-BFGS-B",
            bounds=[(1e-12, None), (1e-6, 0.5), (0.5, 0.9999)],
        )

        if not res.success:
            return None

        omega, alpha, beta = res.x
        persistence = alpha + beta
        aic = 2 * 3 + 2 * res.fun        # 2*k + 2*NLL
        bic = 3 * np.log(n) + 2 * res.fun

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "omega": float(omega),
            "persistence": float(persistence),
            "aic": float(aic),
            "bic": float(bic),
        }

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def compare(self) -> dict:
        """
        Compare EWMA vs GARCH via AIC/BIC.

        Returns dict with both results and recommendation.
        """
        ewma = self.fit_ewma()
        garch = self.fit_garch()

        # Compute EWMA log-likelihood for AIC comparison
        r = self._log_rets
        r2 = r ** 2
        w = round(ewma["vol_window_s"] / self._dt)   # seconds → steps for ewm span
        ewma_var = pd.Series(r2).ewm(span=w).mean().values
        ewma_var = np.maximum(ewma_var, 1e-20)
        ll_ewma = -0.5 * np.sum(np.log(ewma_var) + r2 / ewma_var)
        n = len(r)
        ewma_aic = 2 * 1 - 2 * ll_ewma    # 1 parameter (span)
        ewma_bic = 1 * np.log(n) - 2 * ll_ewma

        recommendation = "ewma"
        if garch is not None and garch["aic"] < ewma_aic:
            recommendation = "garch"

        return {
            "ewma": {**ewma, "aic": float(ewma_aic), "bic": float(ewma_bic)},
            "garch": garch,
            "recommendation": recommendation,
        }
