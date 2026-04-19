"""
Gamma optimizer — grid search on expected net utility.

Finds optimal (gamma, omega) that maximises:
    utility = spread_P&L - inventory_risk_cost

using calibrated arrival/size/vol parameters and the Avellaneda-Stoikov
spread formula replicated from the Quoter.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..stock_simulation.config import TRADING_SECONDS_PER_YEAR


class GammaOptimizer:
    """Optimise risk-aversion and horizon from Phase 1 data."""

    def __init__(
        self,
        fill_df: pd.DataFrame,
        step_log: pd.DataFrame,
        mid_prices: np.ndarray,
        dt: float,
        arrival_params: dict,
        size_params: dict,
        vol_params: dict,
    ) -> None:
        self._dt = dt
        self._mid_prices = mid_prices
        self._arrival = arrival_params
        self._size = size_params
        self._vol = vol_params

        # Realised annualised vol from price data
        log_rets = np.diff(np.log(mid_prices))
        dt_frac = dt / TRADING_SECONDS_PER_YEAR
        self._sigma = float(np.std(log_rets) / np.sqrt(dt_frac))

        # Expected size from truncated Pareto
        self._E_size = self._expected_size()

        # Fair mid mean (for converting bps → price)
        self._fair_mid = float(np.mean(mid_prices))

    def optimize(self) -> dict:
        """
        Return {"gamma", "omega", "expected_sharpe"}.
        """
        gamma_grid = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        omega_grid = [1.0 / (24 * 3600), 1.0 / (8 * 3600), 1.0 / (4 * 3600)]

        best_util = -np.inf
        best_gamma = 0.1
        best_omega = 1.0 / (8 * 3600)

        for omega in omega_grid:
            for gamma in gamma_grid:
                u = self._utility(gamma, omega)
                if u > best_util:
                    best_util = u
                    best_gamma = gamma
                    best_omega = omega

        # Refine gamma around best grid point via bounded minimisation
        idx = gamma_grid.index(best_gamma) if best_gamma in gamma_grid else 3
        lo = gamma_grid[max(0, idx - 1)] * 0.8
        hi = gamma_grid[min(len(gamma_grid) - 1, idx + 1)] * 1.2

        try:
            res = minimize_scalar(
                lambda g: -self._utility(g, best_omega),
                bounds=(lo, hi),
                method="bounded",
            )
            if res.success and self._utility(res.x, best_omega) > best_util:
                best_gamma = float(res.x)
                best_util = self._utility(best_gamma, best_omega)
        except Exception:
            pass

        # Expected Sharpe (daily)
        T_day = 24 * 3600
        pnl_day = best_util * T_day

        A_buy = self._arrival["A_buy"]
        k_buy = self._arrival["k_buy"]
        A_sell = self._arrival["A_sell"]
        k_sell = self._arrival["k_sell"]
        inv_horizon_y = 1.0 / (best_omega * TRADING_SECONDS_PER_YEAR)
        k_eff = (k_buy + k_sell) / 2.0
        spread_bps = (
            best_gamma * (self._sigma * 100.0) ** 2 * inv_horizon_y
            + (2.0 / best_gamma) * np.log(1.0 + best_gamma / k_eff)
        )
        half_bps = spread_bps / 2.0
        lam_total = (
            A_buy * np.exp(-k_buy * half_bps)
            + A_sell * np.exp(-k_sell * half_bps)
        )

        pnl_std = (
            self._sigma
            * np.sqrt(lam_total)
            * self._E_size
            * np.sqrt(T_day / TRADING_SECONDS_PER_YEAR)
        )
        sharpe = float(pnl_day / pnl_std) if pnl_std > 0 else 0.0

        return {
            "gamma": float(best_gamma),
            "omega": float(best_omega),
            "expected_sharpe": sharpe,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _utility(self, gamma: float, omega: float) -> float:
        """
        Expected net utility per second for a given (gamma, omega).

        Replicates the A-S spread formula from quoter.py:264-268.
        """
        sigma = self._sigma
        A_buy = self._arrival["A_buy"]
        k_buy = self._arrival["k_buy"]
        A_sell = self._arrival["A_sell"]
        k_sell = self._arrival["k_sell"]
        E_size = self._E_size

        inv_horizon_y = 1.0 / (omega * TRADING_SECONDS_PER_YEAR)
        k_eff = (k_buy + k_sell) / 2.0

        # A-S optimal spread in bps (same formula as Quoter.compute_quotes)
        spread_AS_bps = (
            gamma * (sigma * 100.0) ** 2 * inv_horizon_y
            + (2.0 / gamma) * np.log(1.0 + gamma / k_eff)
        )
        half_spread_bps = max(spread_AS_bps / 2.0, 0.01)  # floor at 0.01 bp

        # Convert to price units
        half_spread_price = half_spread_bps / 10_000.0 * self._fair_mid

        # Expected fill rate at this half-spread
        lam_buy = A_buy * np.exp(-k_buy * half_spread_bps)
        lam_sell = A_sell * np.exp(-k_sell * half_spread_bps)
        lam_total = lam_buy + lam_sell

        # Spread P&L per second
        spread_pnl = lam_total * half_spread_price * E_size

        # Inventory variance cost per second
        # Inventory does a random walk with step E_size at rate lam_total
        inv_cost = (
            gamma * sigma ** 2 * inv_horizon_y
            * lam_total * E_size ** 2
            / TRADING_SECONDS_PER_YEAR
        )

        return spread_pnl - inv_cost

    def _expected_size(self) -> float:
        """E[X] for truncated Pareto."""
        alpha = self._size["alpha"]
        x_min = self._size["size_min"]
        x_max = self._size["size_max"]

        if alpha <= 1.0 or x_min >= x_max:
            return (x_min + x_max) / 2.0

        ratio = x_min / x_max
        E = (
            alpha * x_min ** alpha
            / (alpha - 1.0)
            * (1.0 - ratio ** (alpha - 1.0))
            / (1.0 - ratio ** alpha)
        )
        return float(E)
