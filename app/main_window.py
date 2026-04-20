"""
main_window.py — MainWindow for the FX Market Making desktop app.
"""
from __future__ import annotations

import sys
import pathlib
import time

# Make src/ importable
_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QStatusBar, QDialog, QTextEdit, QMessageBox,
    QScrollArea, QSizePolicy,
)
from PySide6.QtCore import Qt

from .workers import StockWorker, Phase1Worker, CalibrationWorker, Phase2Worker, Phase3Worker
from .result_window import ResultWindow


def _spin(val: float, lo: float, hi: float, decimals: int = 4,
          step: float = 0.01) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(val)
    w.setMinimumWidth(90)
    return w


def _ispin(val: int, lo: int, hi: int) -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    w.setMinimumWidth(90)
    return w


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FX Market Making — Simulation Dashboard")

        # ── Simulation state ──────────────────────────────────────────────
        self.stock    = None
        self.market_B = None
        self.market_C = None
        self.ctrl_p1  = None
        self.cal_cfg  = None
        self.ctrl_p2  = None
        self.ctrl_p3  = None

        self._active_worker = None
        self._result_windows: list[ResultWindow] = []

        self._build_ui()
        self._refresh_buttons()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        outer.addWidget(self._stock_group())
        outer.addWidget(self._phases_group())
        outer.addStretch()

        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status = sb

    def _stock_group(self) -> QGroupBox:
        grp = QGroupBox("Stock Simulation")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        # Row 1: model selector + seed + n_days
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Model:"))
        self._model = QComboBox()
        self._model.addItems(["GARCH", "GBM", "Heston"])
        self._model.setMinimumWidth(110)
        self._model.currentTextChanged.connect(self._on_model_change)
        r1.addWidget(self._model)
        r1.addSpacing(12)
        r1.addWidget(QLabel("Seed:"))
        self._seed = _ispin(32, 0, 99999)
        r1.addWidget(self._seed)
        r1.addSpacing(12)
        r1.addWidget(QLabel("N Days:"))
        self._n_days = _ispin(30, 1, 365)
        r1.addWidget(self._n_days)
        r1.addStretch()
        lay.addLayout(r1)

        # Row 2: vol + origin + dt
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Vol:"))
        self._vol = _spin(0.07, 0.001, 5.0, 4, 0.01)
        r2.addWidget(self._vol)
        r2.addSpacing(12)
        r2.addWidget(QLabel("Origin:"))
        self._origin = _spin(1.10, 0.01, 100.0, 4, 0.01)
        r2.addWidget(self._origin)
        r2.addSpacing(12)
        r2.addWidget(QLabel("DT (s):"))
        self._dt = _spin(0.1, 0.001, 10.0, 3, 0.05)
        r2.addWidget(self._dt)
        r2.addStretch()
        lay.addLayout(r2)

        # GARCH params (default shown)
        self._garch_widget = QWidget()
        rg = QHBoxLayout(self._garch_widget)
        rg.setContentsMargins(0, 0, 0, 0)
        rg.addWidget(QLabel("α:"))
        self._g_alpha = _spin(0.05, 0.0, 1.0, 4, 0.01)
        rg.addWidget(self._g_alpha)
        rg.addWidget(QLabel("β:"))
        self._g_beta = _spin(0.94, 0.0, 1.0, 4, 0.01)
        rg.addWidget(self._g_beta)
        rg.addWidget(QLabel("λ (jumps/yr):"))
        self._g_lam = _spin(100.0, 0.0, 10000.0, 1, 10.0)
        rg.addWidget(self._g_lam)
        rg.addWidget(QLabel("σJ:"))
        self._g_sigj = _spin(0.005, 0.0, 1.0, 4, 0.001)
        rg.addWidget(self._g_sigj)
        rg.addStretch()
        lay.addWidget(self._garch_widget)

        # Heston params (hidden initially)
        self._heston_widget = QWidget()
        rh = QHBoxLayout(self._heston_widget)
        rh.setContentsMargins(0, 0, 0, 0)
        rh.addWidget(QLabel("κ:"))
        self._h_kappa = _spin(2.0, 0.0, 100.0, 2, 0.1)
        rh.addWidget(self._h_kappa)
        rh.addWidget(QLabel("θ:"))
        self._h_theta = _spin(0.0049, 0.0001, 1.0, 5, 0.001)
        rh.addWidget(self._h_theta)
        rh.addWidget(QLabel("ξ:"))
        self._h_xi = _spin(0.3, 0.0, 5.0, 3, 0.05)
        rh.addWidget(self._h_xi)
        rh.addWidget(QLabel("ρ:"))
        self._h_rho = _spin(-0.1, -1.0, 1.0, 2, 0.05)
        rh.addWidget(self._h_rho)
        rh.addStretch()
        self._heston_widget.setVisible(False)
        lay.addWidget(self._heston_widget)

        # Capital
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Capital K (EUR/USD):"))
        self._capital = QDoubleSpinBox()
        self._capital.setRange(10_000, 100_000_000)
        self._capital.setDecimals(0)
        self._capital.setSingleStep(100_000)
        self._capital.setValue(1_000_000)
        self._capital.setMinimumWidth(130)
        r3.addWidget(self._capital)
        r3.addStretch()
        lay.addLayout(r3)

        # Buttons + status
        rb = QHBoxLayout()
        self._btn_simulate = QPushButton("Simulate Stock")
        self._btn_simulate.clicked.connect(self._run_stock)
        rb.addWidget(self._btn_simulate)

        self._btn_build = QPushButton("Build Markets")
        self._btn_build.clicked.connect(self._run_build_markets)
        rb.addWidget(self._btn_build)

        self._lbl_status_stock = QLabel("")
        self._lbl_status_stock.setObjectName("lbl_status_stock")
        rb.addSpacing(12)
        rb.addWidget(self._lbl_status_stock)
        rb.addStretch()
        lay.addLayout(rb)

        return grp

    def _phases_group(self) -> QGroupBox:
        grp = QGroupBox("Simulation Phases")
        lay = QVBoxLayout(grp)
        lay.setSpacing(8)

        # Phase 1
        r1 = QHBoxLayout()
        self._btn_p1 = QPushButton("Run Phase 1")
        self._btn_p1.setObjectName("btn_phase1")
        self._btn_p1.clicked.connect(self._run_phase1)
        r1.addWidget(self._btn_p1)
        self._lbl_p1 = QLabel("")
        self._lbl_p1.setObjectName("lbl_status_p1")
        r1.addSpacing(12)
        r1.addWidget(self._lbl_p1)
        r1.addStretch()
        lay.addLayout(r1)

        # Calibration
        r2 = QHBoxLayout()
        self._btn_cal = QPushButton("Run Calibration")
        self._btn_cal.setObjectName("btn_calibrate")
        self._btn_cal.clicked.connect(self._run_calibration)
        r2.addWidget(self._btn_cal)
        self._lbl_cal = QLabel("(requires Phase 1)")
        self._lbl_cal.setObjectName("lbl_status_cal")
        r2.addSpacing(12)
        r2.addWidget(self._lbl_cal)
        r2.addStretch()
        lay.addLayout(r2)

        # Phase 2
        r3 = QHBoxLayout()
        self._btn_p2 = QPushButton("Run Phase 2")
        self._btn_p2.setObjectName("btn_phase2")
        self._btn_p2.clicked.connect(self._run_phase2)
        r3.addWidget(self._btn_p2)
        self._lbl_p2 = QLabel("(uses calibrated config if available)")
        self._lbl_p2.setObjectName("lbl_status_p2")
        r3.addSpacing(12)
        r3.addWidget(self._lbl_p2)
        r3.addStretch()
        lay.addLayout(r3)

        # Phase 3
        r4 = QHBoxLayout()
        self._btn_p3 = QPushButton("Run Phase 3 (HFT)")
        self._btn_p3.setObjectName("btn_phase3")
        self._btn_p3.clicked.connect(self._run_phase3)
        r4.addWidget(self._btn_p3)
        r4.addWidget(QLabel("Quote size / side (EUR):"))
        self._hft_depth = _spin(2000.0, 100.0, 1_000_000.0, 0, 500.0)
        r4.addWidget(self._hft_depth)
        self._lbl_p3 = QLabel("")
        self._lbl_p3.setObjectName("lbl_status_p3")
        r4.addSpacing(12)
        r4.addWidget(self._lbl_p3)
        r4.addStretch()
        lay.addLayout(r4)

        return grp

    # ── Model selector ────────────────────────────────────────────────────

    def _on_model_change(self, model: str):
        self._garch_widget.setVisible(model == "GARCH")
        self._heston_widget.setVisible(model == "Heston")

    # ── Button state ──────────────────────────────────────────────────────

    def _refresh_buttons(self):
        busy = self._active_worker is not None
        has_stock   = self.stock is not None
        has_markets = self.market_B is not None
        has_p1      = self.ctrl_p1 is not None
        has_cal     = self.cal_cfg is not None

        self._btn_simulate.setEnabled(not busy)
        self._btn_build.setEnabled(not busy and has_stock)
        self._btn_p1.setEnabled(not busy and has_markets)
        self._btn_cal.setEnabled(not busy and has_p1)
        self._btn_p2.setEnabled(not busy and has_markets)
        self._btn_p3.setEnabled(not busy and has_markets)

    def _set_busy(self, msg: str):
        self._status.showMessage(f"⏳  {msg}")
        self._refresh_buttons()

    def _set_idle(self, msg: str = "Ready"):
        w = self._active_worker
        self._active_worker = None
        if w is not None:
            w.deleteLater()
        self._status.showMessage(msg)
        self._refresh_buttons()

    # ── Worker plumbing ───────────────────────────────────────────────────

    def _start_worker(self, worker, busy_msg: str):
        self._active_worker = worker
        self._set_busy(busy_msg)
        worker.start()

    def _on_error(self, msg: str):
        self._set_idle("Error — see dialog")
        QMessageBox.critical(self, "Simulation Error", msg)

    # ── Stock simulation ──────────────────────────────────────────────────

    def _run_stock(self):
        model = self._model.currentText()
        params = {
            "model":   model,
            "seed":    self._seed.value(),
            "n_days":  self._n_days.value(),
            "vol":     self._vol.value(),
            "origin":  self._origin.value(),
            "dt":      self._dt.value(),
            # GARCH
            "alpha":   self._g_alpha.value(),
            "beta":    self._g_beta.value(),
            "lam":     self._g_lam.value(),
            "sigma_j": self._g_sigj.value(),
            # Heston
            "kappa":   self._h_kappa.value(),
            "theta":   self._h_theta.value(),
            "xi":      self._h_xi.value(),
            "rho":     self._h_rho.value(),
        }
        w = StockWorker(params, self)
        w.finished.connect(self._on_stock_done)
        w.error.connect(self._on_error)
        self._start_worker(w, f"Simulating stock ({model})…")

    def _on_stock_done(self, stock, market_B, market_C):
        self.stock    = stock
        self.market_B = market_B
        self.market_C = market_C
        n = stock.n_steps
        self._lbl_status_stock.setText(
            f"✓ {self._model.currentText()}  {n:,} steps"
        )
        self._lbl_status_stock.setObjectName("lbl_status_stock")
        self._set_idle(f"Stock simulated — {n:,} steps")

    def _run_build_markets(self):
        if self.stock is None:
            QMessageBox.warning(self, "No Stock", "Simulate a stock first.")
            return
        # rebuild markets from existing stock (fast — no re-simulation)
        self._set_busy("Building markets B & C…")
        try:
            from utils.report.fast_config import build_markets_B_C
            self.market_B, self.market_C = build_markets_B_C(self.stock)
            self._lbl_status_stock.setText("✓ Markets B & C built")
            self._set_idle("Markets built")
        except Exception as exc:
            self._set_idle("Error")
            QMessageBox.critical(self, "Build Error", str(exc))

    # ── Phase 1 ───────────────────────────────────────────────────────────

    def _run_phase1(self):
        n_steps = self.stock.n_steps if self.stock else int(
            self._n_days.value() * 86_400 / self._dt.value())
        w = Phase1Worker(
            self.market_B, self.market_C,
            n_steps=n_steps,
            capital=self._capital.value(),
            client_seed=self._seed.value(),
            parent=self,
        )
        w.finished.connect(self._on_p1_done)
        w.error.connect(self._on_error)
        self._start_worker(w, "Running Phase 1…")

    def _on_p1_done(self, ctrl, figs, elapsed):
        self.ctrl_p1 = ctrl
        self._lbl_p1.setText(f"✓ done in {elapsed:.0f}s")
        self._lbl_cal.setText("(requires Phase 1 ✓)")
        self._set_idle(f"Phase 1 complete ({elapsed:.0f} s)")
        self._open_result(figs, "Phase 1 — Results")

    # ── Calibration ───────────────────────────────────────────────────────

    def _run_calibration(self):
        w = CalibrationWorker(self.ctrl_p1, parent=self)
        w.finished.connect(self._on_cal_done)
        w.error.connect(self._on_error)
        self._start_worker(w, "Running calibration…")

    def _on_cal_done(self, cfg, summary):
        self.cal_cfg = cfg
        self._lbl_cal.setText("✓ calibrated")
        self._lbl_p2.setText("(calibrated config ready)")
        self._set_idle("Calibration complete")
        self._show_cal_summary(summary)

    def _show_cal_summary(self, text: str):
        dlg = QDialog(self)
        dlg.setWindowTitle("Calibration Summary")
        dlg.resize(600, 440)
        lay = QVBoxLayout(dlg)
        te  = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(text)
        te.setFont(__import__("PySide6.QtGui", fromlist=["QFont"]).QFont("Courier New", 11))
        lay.addWidget(te)
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn, alignment=Qt.AlignRight)
        dlg.exec()

    # ── Phase 2 ───────────────────────────────────────────────────────────

    def _run_phase2(self):
        from utils.market_maker.quoter import QuoterConfig  # noqa: PLC0415
        cfg = self.cal_cfg if self.cal_cfg is not None else QuoterConfig()
        n_steps = self.stock.n_steps if self.stock else int(
            self._n_days.value() * 86_400 / self._dt.value())
        w = Phase2Worker(
            self.market_B, self.market_C, cfg,
            n_steps=n_steps,
            capital=self._capital.value(),
            client_seed=self._seed.value(),
            parent=self,
        )
        w.finished.connect(self._on_p2_done)
        w.error.connect(self._on_error)
        self._start_worker(w, "Running Phase 2…")

    def _on_p2_done(self, ctrl, figs, elapsed):
        self.ctrl_p2 = ctrl
        label = "calibrated" if self.cal_cfg else "default config"
        self._lbl_p2.setText(f"✓ done in {elapsed:.0f}s ({label})")
        self._set_idle(f"Phase 2 complete ({elapsed:.0f} s)")
        self._open_result(figs, "Phase 2 — Results")

    # ── Phase 3 ───────────────────────────────────────────────────────────

    def _run_phase3(self):
        from utils.market_maker.quoter import QuoterConfig  # noqa: PLC0415
        cfg = self.cal_cfg if self.cal_cfg is not None else QuoterConfig()
        n_days  = float(self._n_days.value())
        n_steps = self.stock.n_steps if self.stock else int(
            n_days * 86_400 / self._dt.value())
        w = Phase3Worker(
            self.market_B, self.market_C, cfg,
            n_steps=n_steps,
            capital=self._capital.value(),
            client_seed=self._seed.value(),
            n_days=n_days,
            hft_depth=self._hft_depth.value(),
            parent=self,
        )
        w.finished.connect(self._on_p3_done)
        w.error.connect(self._on_error)
        self._start_worker(w, "Running Phase 3 (HFT)…")

    def _on_p3_done(self, ctrl, figs, elapsed):
        self.ctrl_p3 = ctrl
        self._lbl_p3.setText(f"✓ done in {elapsed:.0f}s")
        self._set_idle(f"Phase 3 complete ({elapsed:.0f} s)")
        self._open_result(figs, "Phase 3 — HFT Results")

    # ── Result window ─────────────────────────────────────────────────────

    def _open_result(self, figs: list, title: str):
        win = ResultWindow(figs, title, parent=None)
        self._result_windows.append(win)
        win.show_and_raise()
