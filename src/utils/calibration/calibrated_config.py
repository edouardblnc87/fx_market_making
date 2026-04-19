"""
Calibrated config builder — orchestrates all calibrators and produces
a QuoterConfig ready for Phase 2.

Only the quoting strategy is re-tuned.  Client order flow is exogenous
and must stay identical between Phase 1 and Phase 2 so that the P&L
comparison isolates the effect of better quotes.

Usage:
    builder = CalibratedConfigBuilder(phase1_controller)
    quoter_cfg = builder.build()
    print(builder.summary())
"""

from __future__ import annotations

from ..market_maker.quoter import QuoterConfig
from ..client_flow import config as flow_defaults

from .volatility_calibrator import VolatilityCalibrator
from .spread_calibrator import SpreadCalibrator
from .gamma_optimizer import GammaOptimizer


# Default arrival/size params (from client_flow/config.py).
# Used as fixed inputs to GammaOptimizer — we observe these from
# the market but do not optimise them.
_ARRIVAL_DEFAULTS = {
    "A_buy": flow_defaults.A_BUY,
    "A_sell": flow_defaults.A_SELL,
    "k_buy": flow_defaults.K_BUY,
    "k_sell": flow_defaults.K_SELL,
}
_SIZE_DEFAULTS = {
    "alpha": flow_defaults.ALPHA,
    "size_min": flow_defaults.SIZE_MIN,
    "size_max": flow_defaults.SIZE_MAX,
}


class CalibratedConfigBuilder:
    """
    Extract Phase 1 data from a Controller, run the quoter-side
    calibrators, and build a calibrated QuoterConfig.

    What is calibrated (our strategy):
      - gamma, omega      — risk aversion & horizon  (GammaOptimizer)
      - vol_window         — EWMA span               (VolatilityCalibrator)
      - alpha_spread       — inventory spread weight  (SpreadCalibrator)
      - alpha_imbalance    — OFI tilt                 (SpreadCalibrator)

    What is NOT calibrated (exogenous market):
      - A_buy/A_sell, k_buy/k_sell  — client arrival rates
      - alpha, size_min, size_max   — client order sizes
    """

    def __init__(self, controller) -> None:
        self._ctrl = controller
        self.diagnostics: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> QuoterConfig:
        """Run calibration pipeline. Return a calibrated QuoterConfig."""
        ctrl = self._ctrl
        trade_history = ctrl.trade_history
        step_log = ctrl.step_log
        mid_prices = ctrl.market_B.noised_mid_price
        dt = ctrl.market_B.stock.time_step
        capital_K = ctrl.quoter.capital_K

        # --- Quoter-side calibrators ---

        vol_cal = VolatilityCalibrator(mid_prices, dt)
        vol_params = vol_cal.fit_ewma()

        spread_cal = SpreadCalibrator(trade_history, step_log, capital_K)
        spread_params = spread_cal.fit()

        # GammaOptimizer uses default arrival/size as fixed market assumptions
        gamma_opt = GammaOptimizer(
            trade_history, step_log, mid_prices, dt,
            _ARRIVAL_DEFAULTS, _SIZE_DEFAULTS, vol_params,
        )
        gamma_params = gamma_opt.optimize()

        # --- Store diagnostics ---

        self.diagnostics = {
            "volatility_ewma": vol_params,
            "volatility_comparison": vol_cal.compare(),
            "spread": spread_params,
            "gamma": gamma_params,
        }

        # --- Build calibrated QuoterConfig ---

        quoter_cfg = QuoterConfig(
            gamma=gamma_params["gamma"],
            omega=gamma_params["omega"],
            vol_window=vol_params["vol_window"],
            alpha_spread=spread_params["alpha_spread"],
            alpha_imbalance=spread_params["alpha_imbalance"],
        )

        self._quoter_cfg = quoter_cfg
        return quoter_cfg

    def summary(self) -> str:
        """Formatted table: heuristic defaults vs calibrated QuoterConfig."""
        if not self.diagnostics:
            return "No calibration run yet — call build() first."

        d = self.diagnostics
        lines = [
            "=" * 68,
            "  CALIBRATION SUMMARY — Phase 1 → Phase 2  (QuoterConfig only)",
            "=" * 68,
            "",
            f"  {'Parameter':<25} {'Default':>12} {'Calibrated':>12}",
            f"  {'-'*25} {'-'*12} {'-'*12}",
        ]

        rows = [
            ("gamma",           "0.1",    d["gamma"]["gamma"]),
            ("omega",           f"{1/(8*3600):.2e}", d["gamma"]["omega"]),
            ("vol_window",      "6000",   d["volatility_ewma"]["vol_window"]),
            ("alpha_spread",    "0.5",    d["spread"]["alpha_spread"]),
            ("alpha_imbalance", "0.0002", d["spread"]["alpha_imbalance"]),
        ]

        for name, default, calibrated in rows:
            lines.append(f"  {name:<25} {default:>12} {calibrated:>12.6g}")

        lines.append("")
        lines.append(f"  Expected daily Sharpe: {d['gamma']['expected_sharpe']:.3f}")

        # Vol comparison
        vc = d.get("volatility_comparison", {})
        if vc:
            lines.append(f"  Vol model recommendation: {vc.get('recommendation', '?')}")

        lines.append("=" * 68)
        return "\n".join(lines)
