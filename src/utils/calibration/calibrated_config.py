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

from concurrent.futures import ThreadPoolExecutor

from ..market_maker.quoter import QuoterConfig
from ..client_flow import config as flow_defaults

from .volatility_calibrator import VolatilityCalibrator
from .spread_calibrator import SpreadCalibrator
from .gamma_optimizer import GammaOptimizer
from .k_calibrator import KCalibrator
from .stale_calibrator import StaleCalibrator


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
      - vol_window_s       — EWMA span               (VolatilityCalibrator)
      - alpha_spread       — inventory spread weight  (SpreadCalibrator)
      - alpha_imbalance    — OFI tilt                 (SpreadCalibrator)
      - k                  — arrival intensity decay  (KCalibrator)
      - beta, Q_base       — ladder shape             (from target_sweep design input)
      - stale_s            — staleness threshold      (StaleCalibrator)

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

    def build(self, target_sweep: float = 0.05, beta: float = 1.0) -> QuoterConfig:
        """
        Run calibration pipeline. Return a calibrated QuoterConfig.

        Parameters
        ----------
        target_sweep : fraction of client orders that should sweep past level 1
                       and reach level 2.  Controls Q_base via the Pareto CDF:
                       Q_base = x_min × target_sweep^(-1/alpha).
                       Default 0.05 → ~5% sweep-through → Q_base ≈ 7 370 EUR.
        beta         : ladder size decay per level.  Manual input — not calibrated
                       from data (fill-volume data cannot reliably recover beta).
                       Default 1.0 → level 2 ≈ 37% of level 1.
        """
        ctrl = self._ctrl
        trade_history = ctrl.trade_history
        step_log = ctrl.step_log
        mid_prices = ctrl.market_B.noised_mid_price
        dt = ctrl.market_B.stock.time_step
        capital_K = ctrl.quoter.capital_K

        # --- Quoter-side calibrators (parallelised where dependencies allow) ---

        vol_cal    = VolatilityCalibrator(mid_prices, dt)
        spread_cal = SpreadCalibrator(trade_history, step_log, capital_K)
        k_cal      = KCalibrator(trade_history)
        stale_cal  = StaleCalibrator(trade_history, step_log)

        # Round 1: all fill-based + vol calibrators are mutually independent.
        with ThreadPoolExecutor(max_workers=4) as pool:
            f_vol    = pool.submit(vol_cal.fit_ewma)
            f_spread = pool.submit(spread_cal.fit)
            f_k      = pool.submit(k_cal.fit)
            f_stale  = pool.submit(stale_cal.fit)
            vol_params    = f_vol.result()
            spread_params = f_spread.result()
            k_params      = f_k.result()
            stale_params  = f_stale.result()

        # Q_base from target sweep rate — Pareto CDF inversion.
        # P(X > Q_base) = target_sweep  →  Q_base = x_min × target_sweep^(-1/alpha)
        alpha_pareto = _SIZE_DEFAULTS["alpha"]    # 1.5
        x_min        = _SIZE_DEFAULTS["size_min"] # 1 000 EUR
        Q_base       = x_min * target_sweep ** (-1.0 / alpha_pareto)
        ladder_params = {
            "Q_base":        Q_base,
            "beta":          beta,
            "target_sweep":  target_sweep,
        }

        # Round 2: gamma optimisation + vol comparison both need vol_params.
        gamma_opt = GammaOptimizer(
            trade_history, step_log, mid_prices, dt,
            _ARRIVAL_DEFAULTS, _SIZE_DEFAULTS, vol_params,
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_gamma = pool.submit(gamma_opt.optimize)
            f_comp  = pool.submit(vol_cal.compare)
            gamma_params   = f_gamma.result()
            vol_comparison = f_comp.result()

        # --- Store diagnostics ---

        self.diagnostics = {
            "volatility_ewma": vol_params,
            "volatility_comparison": vol_comparison,
            "spread": spread_params,
            "gamma": gamma_params,
            "k": k_params,
            "ladder": ladder_params,
            "stale": stale_params,
        }

        # --- Build calibrated QuoterConfig ---

        quoter_cfg = QuoterConfig(
            gamma=gamma_params["gamma"],
            omega=gamma_params["omega"],
            vol_window_s=vol_params["vol_window_s"],
            alpha_spread=spread_params["alpha_spread"],
            alpha_imbalance=spread_params["alpha_imbalance"],
            k=k_params["k"],
            beta=beta,
            Q_base=Q_base,
            stale_s=stale_params["stale_s"],
        )

        self._dt = dt
        self._quoter_cfg = quoter_cfg
        return quoter_cfg

    def summary(self) -> str:
        """Formatted table: heuristic defaults vs calibrated QuoterConfig."""
        if not self.diagnostics:
            return "No calibration run yet — call build() first."

        d   = self.diagnostics
        cfg = QuoterConfig()   # default instance — single source of truth

        lines = [
            "=" * 68,
            "  CALIBRATION SUMMARY — Phase 1 → Phase 2  (QuoterConfig only)",
            "=" * 68,
            "",
            f"  {'Parameter':<25} {'Default':>12} {'Calibrated':>12}",
            f"  {'-'*25} {'-'*12} {'-'*12}",
        ]

        rows = [
            ("gamma",           cfg.gamma,            d["gamma"]["gamma"]),
            ("omega",           cfg.omega,             d["gamma"]["omega"]),
            ("vol_window_s",    cfg.vol_window_s,      d["volatility_ewma"]["vol_window"] * self._dt),
            ("alpha_spread",    cfg.alpha_spread,      d["spread"]["alpha_spread"]),
            ("alpha_imbalance", cfg.alpha_imbalance,   d["spread"]["alpha_imbalance"]),
            ("k",               cfg.k,                 d["k"]["k"]),
            ("k_buy",           0.3,                   d["k"]["k_buy"]),
            ("k_sell",          0.3,                   d["k"]["k_sell"]),
            ("beta",            cfg.beta,              d["ladder"]["beta"]),
            ("Q_base",          cfg.Q_base,            d["ladder"]["Q_base"]),
            ("stale_s",         cfg.stale_s,           d["stale"]["stale_s"]),
            ("target_sweep",    "n/a",       d["ladder"]["target_sweep"]),
        ]

        for name, default, calibrated in rows:
            try:
                lines.append(f"  {name:<25} {default:>12.6g} {calibrated:>12.6g}")
            except:
                lines.append(f"  {name:<25} {default:>12} {calibrated:>12}")



        lines.append("")
        lines.append(f"  Expected daily Sharpe: {d['gamma']['expected_sharpe']:.3f}")

        # Vol comparison
        vc = d.get("volatility_comparison", {})
        if vc:
            lines.append(f"  Vol model recommendation: {vc.get('recommendation', '?')}")

        lines.append("=" * 68)
        return "\n".join(lines)
