"""
workers.py — QThread workers for each simulation phase.

Each worker emits:
  progress(str)         live status text shown in the app
  finished(*results)    on success
  error(str)            on exception
"""
from __future__ import annotations

import sys
import pathlib
import time
import threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PySide6.QtCore import QThread, Signal

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))



def _make_progress_tqdm(emit_fn):
    """
    Return a tqdm subclass that calls emit_fn(pct, msg) on every update.
    Patches utils.report.controller.tqdm so ctrl.simulate() reports progress
    without touching any src/ file.
    """
    import tqdm as _tqdm_mod
    orig = _tqdm_mod.tqdm

    class _ProgressTqdm(orig):
        def update(self, n=1):
            super().update(n)
            if self.total and self.total > 0:
                pct = min(int(self.n / self.total * 100), 100)
                elapsed = self.format_dict.get("elapsed", 0) or 0
                rate    = self.format_dict.get("rate") or 0
                emit_fn(pct, f"{pct}%  ({elapsed:.0f}s elapsed)")

    return _ProgressTqdm, orig


def _patch_tqdm(progress_tqdm):
    """Replace tqdm in the controller module namespace."""
    import utils.report.controller as _cm
    orig = _cm.tqdm
    _cm.tqdm = progress_tqdm
    return orig


def _unpatch_tqdm(orig):
    import utils.report.controller as _cm
    _cm.tqdm = orig


# ── Stock + Market build ──────────────────────────────────────────────────────

class StockWorker(QThread):
    progress = Signal(int, str)            # pct, message
    finished = Signal(object, object, object)
    error    = Signal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.p = params

    def run(self):
        try:
            from utils.stock_simulation.stock import Stock
            from utils.report.fast_config import build_markets_B_C
            import numpy as np

            p = self.p
            np.random.seed(p["seed"])
            stock = Stock(drift=0.0, vol=p["vol"], origin=p["origin"])

            # Ticker thread: simulate_garch/gbm/heston has no progress hook,
            # so we show elapsed seconds with an indeterminate bar (pct=-1).
            _stop  = threading.Event()
            _t0    = time.time()
            def _tick():
                while not _stop.is_set():
                    self.progress.emit(-1, f"Simulating stock… ({time.time()-_t0:.0f}s)")
                    _stop.wait(0.5)

            ticker = threading.Thread(target=_tick, daemon=True)
            ticker.start()
            try:
                model = p["model"]
                if model == "GBM":
                    stock.simulate_gbm(n_days=p["n_days"], dt_seconds=p["dt"])
                elif model == "Heston":
                    stock.simulate_heston(
                        n_days=p["n_days"], dt_seconds=p["dt"],
                        kappa=p["kappa"], theta=p["theta"],
                        xi=p["xi"], rho=p["rho"],
                    )
                else:
                    stock.simulate_garch(
                        n_days=p["n_days"], dt_seconds=p["dt"],
                        alpha=p["alpha"], beta=p["beta"],
                        lam=p["lam"], sigma_J=p["sigma_j"],
                    )
            finally:
                _stop.set()
                ticker.join(timeout=1.0)

            self.progress.emit(-1, "Building markets B & C (spreads + depth)…")
            market_B, market_C = build_markets_B_C(stock)

            self.progress.emit(100, f"Done — {stock.n_steps:,} steps")
            self.finished.emit(stock, market_B, market_C)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Shared simulate helper ────────────────────────────────────────────────────

def _simulate_with_progress(worker, ctrl, n_steps: int, label: str):
    """Run ctrl.simulate() with tqdm patched to emit progress signals."""
    def _emit(pct, msg):
        worker.progress.emit(pct, f"{label}  {msg}")

    progress_tqdm, orig = _make_progress_tqdm(_emit)
    orig_ctrl = _patch_tqdm(progress_tqdm)
    try:
        t0 = time.time()
        ctrl.simulate(n_steps)
        elapsed = time.time() - t0
    finally:
        _unpatch_tqdm(orig_ctrl)
    return elapsed


# ── Build Markets (standalone rebuild from existing stock) ────────────────────

class BuildMarketsWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(object, object)   # market_B, market_C
    error    = Signal(str)

    def __init__(self, stock, parent=None):
        super().__init__(parent)
        self._stock = stock

    def run(self):
        try:
            from utils.report.fast_config import build_markets_B_C
            self.progress.emit(10, "Building spread & depth arrays…")
            market_B, market_C = build_markets_B_C(self._stock)
            self.progress.emit(100, "Markets ready")
            self.finished.emit(market_B, market_C)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Phase 1 ──────────────────────────────────────────────────────────────────

class Phase1Worker(QThread):
    progress = Signal(int, str)
    finished = Signal(object, float)  # ctrl, elapsed_s
    error    = Signal(str)

    def __init__(self, market_B, market_C, n_steps: int,
                 capital: float, client_seed: int, parent=None):
        super().__init__(parent)
        self._market_B    = market_B
        self._market_C    = market_C
        self._n_steps     = n_steps
        self._capital     = capital
        self._client_seed = client_seed

    def run(self):
        try:
            from utils.order_book.order_book_impl import Order_book
            from utils.market_maker.quoter import Quoter, QuoterConfig
            from utils.report.controller import Controller
            from utils.client_flow.flow_generator import ClientFlowGenerator

            self.progress.emit(2, "Building order book…")
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

            self.progress.emit(5, "Simulating…  0%")
            elapsed = _simulate_with_progress(self, ctrl, self._n_steps, "Phase 1")
            self.progress.emit(100, f"Done  ({elapsed:.0f}s)")
            self.finished.emit(ctrl, elapsed)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Calibration ───────────────────────────────────────────────────────────────

class CalibrationWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(object, str)
    error    = Signal(str)

    def __init__(self, ctrl_p1, parent=None):
        super().__init__(parent)
        self._ctrl = ctrl_p1

    def run(self):
        try:
            from utils.calibration.calibrated_config import CalibratedConfigBuilder

            self.progress.emit(10, "Fitting volatility EWMA…")
            builder = CalibratedConfigBuilder(self._ctrl)

            # CalibratedConfigBuilder.build() runs multiple calibrators in parallel.
            # Emit milestones around the two ThreadPoolExecutor rounds.
            self.progress.emit(20, "Running calibrators (vol, spread, k, stale)…")
            cfg = builder.build()
            self.progress.emit(90, "Building summary…")
            summary = builder.summary()
            self.progress.emit(100, "Calibration done")
            self.finished.emit(cfg, summary)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Shared stock builder for Phase 2 / 3 ────────────────────────────────────

def _build_fresh_markets(worker, stock_params: dict):
    """
    Build a new Stock + markets from stock_params dict.
    Emits progress on worker. Returns (market_B, market_C, n_steps).
    """
    import numpy as np
    from utils.stock_simulation.stock import Stock
    from utils.report.fast_config import build_markets_B_C

    p = stock_params
    worker.progress.emit(-1, f"Simulating fresh stock (seed={p['seed']})…")
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
    else:
        stock.simulate_garch(
            n_days=p["n_days"], dt_seconds=p["dt"],
            alpha=p["alpha"], beta=p["beta"],
            lam=p["lam"], sigma_J=p["sigma_j"],
        )
    worker.progress.emit(-1, "Building markets B & C…")
    market_B, market_C = build_markets_B_C(stock)
    return market_B, market_C, stock.n_steps


# ── Phase 2 ──────────────────────────────────────────────────────────────────

class Phase2Worker(QThread):
    progress = Signal(int, str)
    finished = Signal(object, float)  # ctrl, elapsed_s
    error    = Signal(str)

    def __init__(self, market_B, market_C, quoter_cfg,
                 n_steps: int, capital: float, client_seed: int,
                 stock_params: dict | None = None, parent=None):
        super().__init__(parent)
        self._market_B    = market_B
        self._market_C    = market_C
        self._cfg         = quoter_cfg
        self._n_steps     = n_steps
        self._capital     = capital
        self._client_seed = client_seed
        self._stock_params = stock_params

    def run(self):
        try:
            from utils.order_book.order_book_impl import Order_book
            from utils.market_maker.quoter import Quoter
            from utils.report.controller import Controller
            from utils.client_flow.flow_generator import ClientFlowGenerator

            market_B, market_C, n_steps = self._market_B, self._market_C, self._n_steps
            if self._stock_params:
                market_B, market_C, n_steps = _build_fresh_markets(self, self._stock_params)

            self.progress.emit(2, "Building order book…")
            book = Order_book()
            mm   = Quoter(market_B, market_C, config=self._cfg, capital_K=self._capital)
            book.register_quoter_listener(mm.on_fill)
            gen  = ClientFlowGenerator(seed=self._client_seed)
            ctrl = Controller(
                market_B, market_C, book, mm,
                lambda step, t, mid, bid, ask, dt:
                    gen.generate_step(mid_price=mid, best_bid=bid, best_ask=ask, dt=dt),
            )

            self.progress.emit(5, "Simulating…  0%")
            elapsed = _simulate_with_progress(self, ctrl, n_steps, "Phase 2")

            self.progress.emit(100, f"Done  ({elapsed:.0f}s)")
            self.finished.emit(ctrl, elapsed)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Phase 3 (HFT) ────────────────────────────────────────────────────────────

class Phase3Worker(QThread):
    progress = Signal(int, str)
    finished = Signal(object, float)  # ctrl, elapsed_s
    error    = Signal(str)

    def __init__(self, market_B, market_C, quoter_cfg,
                 n_steps: int, capital: float, client_seed: int,
                 n_days: float, hft_depth: float,
                 stock_params: dict | None = None, parent=None):
        super().__init__(parent)
        self._market_B    = market_B
        self._market_C    = market_C
        self._cfg         = quoter_cfg
        self._n_steps     = n_steps
        self._capital     = capital
        self._client_seed = client_seed
        self._n_days      = n_days
        self._hft_depth   = hft_depth
        self._stock_params = stock_params

    def run(self):
        try:
            from utils.order_book.order_book_impl import Order_book
            from utils.market_maker.quoter import Quoter
            from utils.report.controller import Controller
            from utils.client_flow.flow_generator import ClientFlowGenerator
            from utils.hft.hft_config import HFTConfig
            from utils.hft.scenarios import make_realistic_schedule

            market_B, market_C, n_steps = self._market_B, self._market_C, self._n_steps
            if self._stock_params:
                market_B, market_C, n_steps = _build_fresh_markets(self, self._stock_params)

            self.progress.emit(2, "Building order book + HFT agent…")
            book = Order_book()
            mm   = Quoter(market_B, market_C, config=self._cfg, capital_K=self._capital)
            book.register_quoter_listener(mm.on_fill)
            gen  = ClientFlowGenerator(seed=self._client_seed)
            ctrl = Controller(
                market_B, market_C, book, mm,
                lambda step, t, mid, bid, ask, dt:
                    gen.generate_step(mid_price=mid, best_bid=bid, best_ask=ask, dt=dt),
                hft=True,
                hft_config=HFTConfig(max_depth_eur=self._hft_depth),
                hft_schedule=make_realistic_schedule(self._n_days),
            )

            self.progress.emit(5, "Simulating…  0%")
            elapsed = _simulate_with_progress(self, ctrl, n_steps, "Phase 3 (HFT)")

            self.progress.emit(100, f"Done  ({elapsed:.0f}s)")
            self.finished.emit(ctrl, elapsed)
        except Exception as exc:
            self.error.emit(str(exc))
