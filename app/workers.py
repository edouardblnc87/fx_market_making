"""
workers.py — QThread workers for each simulation phase.

Each worker runs its task in a background thread and emits:
  finished(*results)  on success
  error(str)          on exception

Figure capture: patch plt.show so every call appends the current figure
to a list instead of opening a window. This captures all figures that
report() produces without modifying any source files.
"""
from __future__ import annotations

import sys
import pathlib
import time
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; figures are embedded in Qt
import matplotlib.pyplot as plt

from PySide6.QtCore import QThread, Signal

# Make src/ importable regardless of working directory
_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _capture_figures(fn) -> list:
    """Call fn() while intercepting plt.show; return list of Figure objects."""
    figs = []
    orig = plt.show

    def _capture(*a, **kw):
        fig = plt.gcf()
        if fig not in figs:
            figs.append(fig)

    plt.show = _capture
    try:
        fn()
    finally:
        plt.show = orig
    return figs


# ── Stock + Market build ───────────────────────────────────────────────────────

class StockWorker(QThread):
    """Simulate stock price path and build markets B & C."""
    finished = Signal(object, object, object)  # stock, market_B, market_C
    error    = Signal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.p = params

    def run(self):
        try:
            from utils.stock_simulation.stock import Stock
            from utils.report.fast_config import build_markets_B_C

            p = self.p
            import numpy as np
            np.random.seed(p["seed"])

            stock = Stock(drift=0.0, vol=p["vol"], origin=p["origin"])

            model = p["model"]
            if model == "GBM":
                stock.simulate_gbm(n_days=p["n_days"], dt_seconds=p["dt"])
            elif model == "Heston":
                stock.simulate_heston(
                    n_days=p["n_days"], dt_seconds=p["dt"],
                    kappa=p["kappa"], theta=p["theta"],
                    xi=p["xi"], rho=p["rho"],
                )
            else:  # GARCH
                stock.simulate_garch(
                    n_days=p["n_days"], dt_seconds=p["dt"],
                    alpha=p["alpha"], beta=p["beta"],
                    lam=p["lam"], sigma_J=p["sigma_j"],
                )

            market_B, market_C = build_markets_B_C(stock)
            self.finished.emit(stock, market_B, market_C)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Phase 1 ───────────────────────────────────────────────────────────────────

class Phase1Worker(QThread):
    finished = Signal(object, list, float)  # ctrl, figs, elapsed_s
    error    = Signal(str)

    def __init__(self, market_B, market_C, n_steps: int,
                 capital: float, client_seed: int, parent=None):
        super().__init__(parent)
        self._market_B   = market_B
        self._market_C   = market_C
        self._n_steps    = n_steps
        self._capital    = capital
        self._client_seed = client_seed

    def run(self):
        try:
            from utils.order_book.order_book_impl import Order_book
            from utils.market_maker.quoter import Quoter, QuoterConfig
            from utils.report.controller import Controller
            from utils.client_flow.flow_generator import ClientFlowGenerator

            book = Order_book()
            mm   = Quoter(self._market_B, self._market_C,
                          config=QuoterConfig(), capital_K=self._capital)
            book.register_quoter_listener(mm.on_fill)
            gen  = ClientFlowGenerator(seed=self._client_seed)
            ctrl = Controller(
                self._market_B, self._market_C, book, mm,
                lambda step, t, mid, bid, ask, dt:
                    gen.generate_step(mid_price=mid, best_bid=bid, best_ask=ask, dt=dt),
            )

            t0 = time.time()
            ctrl.simulate(self._n_steps)
            elapsed = time.time() - t0

            figs = _capture_figures(ctrl.report)
            self.finished.emit(ctrl, figs, elapsed)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Calibration ───────────────────────────────────────────────────────────────

class CalibrationWorker(QThread):
    finished = Signal(object, str)  # QuoterConfig, summary_text
    error    = Signal(str)

    def __init__(self, ctrl_p1, parent=None):
        super().__init__(parent)
        self._ctrl = ctrl_p1

    def run(self):
        try:
            from utils.calibration.calibrated_config import CalibratedConfigBuilder
            builder = CalibratedConfigBuilder(self._ctrl)
            cfg     = builder.build()
            summary = builder.summary()
            self.finished.emit(cfg, summary)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Phase 2 ───────────────────────────────────────────────────────────────────

class Phase2Worker(QThread):
    finished = Signal(object, list, float)  # ctrl, figs, elapsed_s
    error    = Signal(str)

    def __init__(self, market_B, market_C, quoter_cfg,
                 n_steps: int, capital: float, client_seed: int, parent=None):
        super().__init__(parent)
        self._market_B   = market_B
        self._market_C   = market_C
        self._cfg        = quoter_cfg
        self._n_steps    = n_steps
        self._capital    = capital
        self._client_seed = client_seed

    def run(self):
        try:
            from utils.order_book.order_book_impl import Order_book
            from utils.market_maker.quoter import Quoter
            from utils.report.controller import Controller
            from utils.client_flow.flow_generator import ClientFlowGenerator

            book = Order_book()
            mm   = Quoter(self._market_B, self._market_C,
                          config=self._cfg, capital_K=self._capital)
            book.register_quoter_listener(mm.on_fill)
            gen  = ClientFlowGenerator(seed=self._client_seed)
            ctrl = Controller(
                self._market_B, self._market_C, book, mm,
                lambda step, t, mid, bid, ask, dt:
                    gen.generate_step(mid_price=mid, best_bid=bid, best_ask=ask, dt=dt),
            )

            t0 = time.time()
            ctrl.simulate(self._n_steps)
            elapsed = time.time() - t0

            figs = _capture_figures(ctrl.report)
            self.finished.emit(ctrl, figs, elapsed)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Phase 3 (HFT) ─────────────────────────────────────────────────────────────

class Phase3Worker(QThread):
    finished = Signal(object, list, float)  # ctrl, figs, elapsed_s
    error    = Signal(str)

    def __init__(self, market_B, market_C, quoter_cfg,
                 n_steps: int, capital: float, client_seed: int,
                 n_days: float, hft_depth: float, parent=None):
        super().__init__(parent)
        self._market_B    = market_B
        self._market_C    = market_C
        self._cfg         = quoter_cfg
        self._n_steps     = n_steps
        self._capital     = capital
        self._client_seed = client_seed
        self._n_days      = n_days
        self._hft_depth   = hft_depth

    def run(self):
        try:
            from utils.order_book.order_book_impl import Order_book
            from utils.market_maker.quoter import Quoter
            from utils.report.controller import Controller
            from utils.client_flow.flow_generator import ClientFlowGenerator
            from utils.hft.hft_config import HFTConfig
            from utils.hft.scenarios import make_realistic_schedule

            book = Order_book()
            mm   = Quoter(self._market_B, self._market_C,
                          config=self._cfg, capital_K=self._capital)
            book.register_quoter_listener(mm.on_fill)
            gen  = ClientFlowGenerator(seed=self._client_seed)
            ctrl = Controller(
                self._market_B, self._market_C, book, mm,
                lambda step, t, mid, bid, ask, dt:
                    gen.generate_step(mid_price=mid, best_bid=bid, best_ask=ask, dt=dt),
                hft=True,
                hft_config=HFTConfig(max_depth_eur=self._hft_depth),
                hft_schedule=make_realistic_schedule(self._n_days),
            )

            t0 = time.time()
            ctrl.simulate(self._n_steps)
            elapsed = time.time() - t0

            figs = _capture_figures(ctrl.report)
            self.finished.emit(ctrl, figs, elapsed)
        except Exception as exc:
            self.error.emit(str(exc))
